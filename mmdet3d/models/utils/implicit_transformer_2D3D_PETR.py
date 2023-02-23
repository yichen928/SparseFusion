import copy
import numpy as np
import torch
from mmcv.cnn import ConvModule, build_conv_layer, kaiming_init
from mmcv.runner import force_fp32
from torch import nn
import torch.nn.functional as F

from mmdet3d.core import (circle_nms, draw_heatmap_gaussian, gaussian_radius,
                          xywhr2xyxyr, limit_period, PseudoSampler)
from mmdet3d.core import Box3DMode, LiDARInstance3DBoxes
from mmdet3d.models.builder import HEADS, build_loss
from mmdet3d.models.fusion_layers import apply_3d_transformation

from mmdet3d.models.utils import TransformerDecoderLayer, MultiheadAttention, PositionEmbeddingLearned
from mmdet3d.models.utils import clip_sigmoid, inverse_sigmoid


def denormalize_pos(normal_pos, x_max, y_max, sigmoid=True):
    max_xy = torch.Tensor([x_max, y_max]).to(normal_pos.device).view(1, 1, 2)
    if sigmoid:
        pos = normal_pos.sigmoid() * max_xy
    else:
        pos = normal_pos * max_xy
    return pos

def normalize_pos(pos, x_max, y_max):
    max_xy = torch.Tensor([x_max, y_max]).to(pos.device).view(1, 1, 2)
    normal_pos = pos / max_xy
    return inverse_sigmoid(normal_pos)

class ImageTransformer2D_3D_PETR(nn.Module):
    def __init__(self, num_views, hidden_channel, num_heads, num_decoder_layers, prediction_heads, img_smca,
                 out_size_factor_img, ffn_channel, dropout, activation, test_cfg, query_pos, key_pos, supervision2d):
        super(ImageTransformer2D_3D_PETR, self).__init__()
        self.hidden_channel = hidden_channel
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.prediction_heads = prediction_heads
        self.img_smca = img_smca
        self.num_views = num_views
        self.out_size_factor_img = out_size_factor_img
        self.test_cfg = test_cfg
        self.img_x_size = test_cfg['img_scale'][0] / out_size_factor_img
        self.img_y_size = test_cfg['img_scale'][1] / out_size_factor_img
        self.grid_x_size = test_cfg['grid_size'][0] // test_cfg['out_size_factor']
        self.grid_y_size = test_cfg['grid_size'][1] // test_cfg['out_size_factor']
        self.supervision2d = supervision2d

        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    hidden_channel, num_heads, ffn_channel, dropout, activation,
                    self_posembed=query_pos,
                    cross_posembed=key_pos,
                )
            )

    def forward(self, img_query_feat, normal_img_query_pos, img_query_view, img_feat_flatten, normal_img_feat_pos, img_metas):
        batch_size = img_query_feat.shape[0]
        num_proposals = img_query_feat.shape[2]
        ret_dicts = []
        res_layer = self.prediction_heads(img_query_feat)
        res_layer['bbox'][:, :2] = normal_img_query_pos.permute(0, 2, 1)
        denormal_img_feat_pos = denormalize_pos(normal_img_feat_pos, self.img_x_size, self.img_y_size)  # [1, h*w, 2]

        for i in range(self.num_decoder_layers):
            img_query_pos_3d = torch.zeros([batch_size, num_proposals, 96]).to(img_query_feat.device)
            centers = denormalize_pos(normal_img_query_pos, self.img_x_size, self.img_y_size)  # [bs, num_proposals, 2]
            dims = denormalize_pos(res_layer['bbox'][:, 2:4].permute(0, 2, 1), self.img_x_size, self.img_y_size)  # [bs, num_proposals, 2]
            img_prev_query_feat = img_query_feat.clone()  # [BS, C, num_proposals]
            img_query_feat = torch.zeros_like(img_query_feat)  # create new container for img query feature
            for sample_idx in range(batch_size):
                for view_idx in range(self.num_views):
                    on_the_image = img_query_view[sample_idx] == view_idx  # [num_on_the_image, ]
                    if torch.sum(on_the_image) <= 1:
                        continue
                    centers_view = centers[sample_idx, on_the_image]  # [num_on_the_image, 2]
                    img_query_feat_view = img_prev_query_feat[sample_idx, :, on_the_image]  # [C, num_on_the_image]
                    if self.img_smca:
                        corners_view = dims[sample_idx, on_the_image]  # [num_on_the_image, 2]
                        radius = torch.ceil(corners_view.norm(dim=-1, p=2) / 2).int()  # [num_on_the_image, ]
                        sigma = (radius * 2 + 1) / 6.0
                        distance = (centers_view[:, None, :] - (denormal_img_feat_pos - 0.5)).norm(dim=-1) ** 2  # [num_on_the_image, H*W]
                        gaussian_mask = (-distance / (2 * sigma[:, None] ** 2)).exp()  # [num_on_the_image, H*W]
                        gaussian_mask[gaussian_mask < torch.finfo(torch.float32).eps] = 0
                        attn_mask = gaussian_mask

                        img_query_feat_view = self.decoder[i](
                            img_query_feat_view[None], img_feat_flatten[sample_idx:sample_idx + 1, view_idx],
                            normal_img_query_pos[sample_idx:sample_idx + 1, on_the_image], normal_img_feat_pos,
                            attn_mask=attn_mask.log()
                        )
                    else:
                        img_query_feat_view = self.decoder[i](
                            img_query_feat_view[None], img_feat_flatten[sample_idx:sample_idx + 1, view_idx],
                            normal_img_query_pos[sample_idx:sample_idx + 1, on_the_image], normal_img_feat_pos
                        )
                    img_query_feat[sample_idx, :, on_the_image] = img_query_feat_view.clone()

            res_layer = self.prediction_heads(img_query_feat)
            res_layer['bbox'][:, :2] = res_layer['bbox'][:, :2] + normal_img_query_pos.permute(0, 2, 1)

            if self.supervision2d:
                normal_img_query_pos = res_layer['bbox'][:, :2].detach().clone().permute(0, 2, 1)

            res_layer['bbox'] = res_layer['bbox'].sigmoid()
            ret_dicts.append(res_layer)

            for sample_idx in range(batch_size):
                for view_idx in range(self.num_views):
                    img_scale_factor = (
                        normal_img_query_pos.new_tensor(img_metas[sample_idx]['scale_factor'][:2]
                                                if 'scale_factor' in img_metas[sample_idx].keys() else [1.0, 1.0])
                    )
                    on_the_image = img_query_view[sample_idx] == view_idx  # [num_on_the_image, ]
                    denormal_img_query_pos_view = denormalize_pos(
                        normal_img_query_pos[sample_idx:sample_idx + 1, on_the_image], self.img_x_size, self.img_y_size
                    )  # [1, on_the_image, 2]
                    lidar2img_rt = img_query_pos_3d.new_tensor(img_metas[sample_idx]['lidar2img'])[view_idx]
                    img_query_pos_view_3d = self.img_pos_3d(denormal_img_query_pos_view[0].detach().clone(), lidar2img_rt, img_scale_factor) # [on_the_image, 96]
                    img_query_pos_3d[sample_idx, on_the_image] = img_query_pos_view_3d

        return img_query_feat, normal_img_query_pos, img_query_pos_3d, ret_dicts

    def img_pos_3d(self, img_pos, lidar2img, img_scale_factor, D=32):
        # img_pos: [W*H, 2]
        img_pos = img_pos[:, None].repeat(1, D, 1) * self.out_size_factor_img # [W*H, D, 2]
        img_pos = img_pos / img_scale_factor
        index = torch.arange(start=0, end=D, step=1, device=img_pos.device).float()  # [D, ]
        index_1 = index + 1
        bin_size = (self.test_cfg['pc_range'][3] - 1) / (D * (1 + D))
        coords_d = 1 + bin_size * index * index_1  # [D, ]
        depth = coords_d.view(1, D, 1).repeat(img_pos.shape[0], 1, 1)  # [W*H, D, 1]
        coords = torch.cat([img_pos, depth], dim=2) # [W*H, D, 3]
        coords = torch.cat([coords, torch.ones_like(coords[..., :1])], dim=2)  # [W*H, D, 4]

        eps = 1e-5
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)

        img2lidars = torch.inverse(lidar2img)
        coords3d = torch.matmul(img2lidars, coords.unsqueeze(-1)).squeeze(-1)[..., :3]  # [W*H, D, 3]

        # coords3d[..., 0:1] = (coords3d[..., 0:1] - self.test_cfg['pc_range'][0]) / (self.test_cfg['pc_range'][3] - self.test_cfg['pc_range'][0])
        # coords3d[..., 1:2] = (coords3d[..., 1:2] - self.test_cfg['pc_range'][1]) / (self.test_cfg['pc_range'][4] - self.test_cfg['pc_range'][1])
        # coords3d[..., 2:3] = (coords3d[..., 2:3] - self.test_cfg['pc_range'][2]) / (self.test_cfg['pc_range'][5] - self.test_cfg['pc_range'][2])

        # coords3d = inverse_sigmoid(coords3d)

        coords3d = coords3d.contiguous().view(coords3d.size(0), D*3)
        return coords3d


class PointTransformer2D_3D_PETR(nn.Module):
    def __init__(self, hidden_channel, num_heads, num_decoder_layers, prediction_heads, pts_smca, ffn_channel, dropout, activation, test_cfg, query_pos, key_pos):
        super(PointTransformer2D_3D_PETR, self).__init__()
        self.hidden_channel = hidden_channel
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.prediction_heads = prediction_heads
        self.pts_smca = pts_smca
        self.test_cfg = test_cfg
        self.init_height = nn.Sequential(
            nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channel, 1, kernel_size=1),
        )

        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    hidden_channel, num_heads, ffn_channel, dropout, activation,
                    self_posembed=query_pos,
                    cross_posembed=key_pos,
                )
            )

    def forward(self, pts_query_feat, pts_query_pos_2d, lidar_feat_flatten, bev_pos):
        ret_dicts = []
        init_height = self.init_height(pts_query_feat).transpose(2, 1)  # [BS, num_proposals, 1]
        pts_query_pos = torch.cat([pts_query_pos_2d, init_height], dim=2)

        res_layer = self.prediction_heads(pts_query_feat)
        res_layer['center'] = pts_query_pos_2d.permute(0, 2, 1)  # [BS, 2, num_proposals]
        res_layer['height'] = init_height.transpose(2, 1)

        for i in range(self.num_decoder_layers):
            # Transformer Decoder Layer
            # :param query: B C Pq    :param query_pos: B Pq 3/6
            if self.pts_smca:
                # centers = pts_query_pos.detach().clone()  # [BS, num_proposals, 2]
                centers = res_layer['center'].detach().clone().permute(0, 2, 1)  # [BS, num_proposals, 2]
                dims = res_layer['dim'].detach().clone().permute(0, 2, 1)  # [BS, num_proposals, 3]
                corners = dims[..., :2].exp() / self.test_cfg['out_size_factor'] / self.test_cfg['voxel_size'][0]  # [BS, num_proposals, 2]
                radius = torch.ceil(corners.norm(dim=-1, p=2) / 2).int()  # [BS, num_proposals]
                #  radius of the minimum circumscribed circle of the wireframe
                sigma = (radius * 2 + 1) / 6.0  # [BS, num_proposals]
                distance = (centers[:, :, None, :] - (bev_pos - 0.5)).norm(dim=-1) ** 2  # [BS, num_proposals, H*W]
                gaussian_mask = (-distance / (2 * sigma[:, :, None] ** 2)).exp()  # [BS, num_proposals, H*W]
                gaussian_mask[gaussian_mask < torch.finfo(torch.float32).eps] = 0
                attn_mask = gaussian_mask

                pts_query_feat = self.decoder[i](
                    pts_query_feat, lidar_feat_flatten, pts_query_pos, bev_pos, attn_mask=attn_mask.log())
            else:
                pts_query_feat = self.decoder[i](pts_query_feat, lidar_feat_flatten, pts_query_pos, bev_pos)

            # Prediction
            res_layer = self.prediction_heads(pts_query_feat)

            res_layer['center'] = res_layer['center'] + pts_query_pos[..., :2].permute(0, 2, 1)
            res_layer['height'] = res_layer['height'] + pts_query_pos[..., 2:].permute(0, 2, 1)
            ret_dicts.append(res_layer)
            # for next level positional embedding
            pts_query_pos_2d = res_layer['center'].detach().clone().permute(0, 2, 1)
            pts_query_pos_height = res_layer['height'].detach().clone().permute(0, 2, 1)
            pts_query_pos = torch.cat([pts_query_pos_2d, pts_query_pos_height], dim=2)
        return pts_query_feat, pts_query_pos, ret_dicts


class FusionTransformer2D_3D_PETR(nn.Module):
    def __init__(self, hidden_channel, num_heads, num_decoder_layers, prediction_heads, ffn_channel, dropout, activation, test_cfg,
                 query_pos, key_pos, pts_projection=nn.Identity(), img_projection=nn.Identity()):
        super(FusionTransformer2D_3D_PETR, self).__init__()
        self.hidden_channel = hidden_channel
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.prediction_heads = prediction_heads
        self.test_cfg = test_cfg

        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    hidden_channel, num_heads, ffn_channel, dropout, activation,
                    self_posembed=query_pos, cross_posembed=key_pos,
                )
            )

        self.pts_projection = pts_projection
        self.img_projection = img_projection

    def forward(self, pts_query_feat, pts_query_pos, img_query_feat, img_query_pos):
        ret_dicts = []
        pts_query_feat = self.pts_projection(pts_query_feat)
        img_query_feat = self.img_projection(img_query_feat)
        raw_pts_query_feat = pts_query_feat.detach().clone()
        for i in range(self.num_decoder_layers):
            # Transformer Decoder Layer
            # :param query: B C Pq    :param query_pos: B Pq 3/6
            pts_query_feat = self.decoder[i](pts_query_feat, img_query_feat, pts_query_pos, img_query_pos)

            # Prediction
            pts_query_feat = torch.cat([raw_pts_query_feat, pts_query_feat], dim=1)
            res_layer = self.prediction_heads(pts_query_feat)

            pts_query_pos_2d = pts_query_pos[..., :2]
            pts_query_pos_height = pts_query_pos[..., 2:]

            res_layer['center'] = res_layer['center'] + pts_query_pos_2d.permute(0, 2, 1)
            res_layer['height'] = res_layer['height'] + pts_query_pos_height.permute(0, 2, 1)

            ret_dicts.append(res_layer)

            pts_query_pos_2d = res_layer['center'].detach().clone().permute(0, 2, 1)
            pts_query_pos_height = res_layer['height'].detach().clone().permute(0, 2, 1)
            pts_query_pos = torch.cat([pts_query_pos_2d, pts_query_pos_height], dim=2)

        return pts_query_feat, pts_query_pos, ret_dicts
