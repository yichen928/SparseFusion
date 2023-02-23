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

from mmdet3d.models.utils import TransformerDecoderLayer, MultiheadAttention, FFN
from mmdet3d.models.utils import clip_sigmoid, inverse_sigmoid


def normalize_grid(pos):
    max_x = torch.max(pos[...,0]) + 0.5
    max_y = torch.max(pos[...,1]) + 0.5
    max_value = torch.Tensor([max_x, max_y]).to(pos.device)
    pos = pos / max_value.view(1, 1, 2)
    return inverse_sigmoid(pos)


class FusionTransformer3D(nn.Module):
    def __init__(self, hidden_channel, num_heads, num_decoder_layers, prediction_heads, ffn_channel, dropout, activation, query_pos, key_pos):
        super(FusionTransformer3D, self).__init__()
        self.hidden_channel = hidden_channel
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.prediction_heads = prediction_heads

        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    hidden_channel, num_heads, ffn_channel, dropout, activation,
                    self_posembed=query_pos,
                    cross_posembed=key_pos,
                    cross_only=True
                )
            )

    def forward(self, pts_query_feat, pts_query_pos, img_query_feat, img_query_pos):
        all_query_feat = torch.cat([pts_query_feat, img_query_feat], dim=2)
        all_query_pos = torch.cat([pts_query_pos, img_query_pos], dim=1)
        ret_dicts = []
        for i in range(self.num_decoder_layers):
            # Transformer Decoder Layer
            # :param query: B C Pq    :param query_pos: B Pq 3/6
            all_query_feat = self.decoder[i](all_query_feat, all_query_feat, all_query_pos, all_query_pos)

            # Prediction
            res_layer = self.prediction_heads(all_query_feat)
            res_layer['center'] = res_layer['center'] + all_query_pos.permute(0, 2, 1)
            ret_dicts.append(res_layer)
            # for next level positional embedding
            all_query_pos = res_layer['center'].detach().clone().permute(0, 2, 1)

        return all_query_feat, all_query_pos, ret_dicts


class PointTransformer3D(nn.Module):
    def __init__(self, hidden_channel, num_heads, num_decoder_layers, prediction_heads, pts_smca, ffn_channel, dropout, activation, test_cfg, query_pos, key_pos):
        super(PointTransformer3D, self).__init__()
        self.hidden_channel = hidden_channel
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.prediction_heads = prediction_heads
        self.pts_smca = pts_smca
        self.test_cfg = test_cfg

        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    hidden_channel, num_heads, ffn_channel, dropout, activation,
                    self_posembed=query_pos,
                    cross_posembed=key_pos,
                )
            )

    def forward(self, pts_query_feat, pts_query_pos, lidar_feat_flatten, bev_pos):
        ret_dicts = []
        res_layer = self.prediction_heads(pts_query_feat)
        res_layer['center'] = res_layer['center'] + pts_query_pos.permute(0, 2, 1)  # [BS, 2, num_proposals]
        pts_query_pos = res_layer['center'].permute(0, 2, 1)  # [BS, num_proposals, 2]
        normal_bev_pos = normalize_grid(bev_pos)
        for i in range(self.num_decoder_layers):
            # Transformer Decoder Layer
            # :param query: B C Pq    :param query_pos: B Pq 3/6
            if self.pts_smca:
                # centers = pts_query_pos.detach().clone()  # [BS, num_proposals, 2]
                centers = res_layer['center'].detach().clone().permute(0, 2, 1)  # [BS, num_proposals, 2]
                centers = centers.sigmoid() * (self.test_cfg['pc_range'][3] - self.test_cfg['pc_range'][0]) / self.test_cfg['out_size_factor'] / self.test_cfg['voxel_size'][0]
                dims = res_layer['dim'].detach().clone().permute(0, 2, 1)
                corners = dims[..., :2].exp() / self.test_cfg['out_size_factor'] / self.test_cfg['voxel_size'][0]  # [BS, num_proposals, 2]
                radius = torch.ceil(corners.norm(dim=-1, p=2) / 2).int()  # [BS, num_proposals]
                #  radius of the minimum circumscribed circle of the wireframe
                sigma = (radius * 2 + 1) / 6.0  # [BS, num_proposals]
                distance = (centers[:, :, None, :] - (bev_pos - 0.5)).norm(dim=-1) ** 2  # [BS, num_proposals, H*W]
                gaussian_mask = (-distance / (2 * sigma[:, :, None] ** 2)).exp()  # [BS, num_proposals, H*W]
                gaussian_mask[gaussian_mask < torch.finfo(torch.float32).eps] = 0
                attn_mask = gaussian_mask

                pts_query_feat = self.decoder[i](pts_query_feat, lidar_feat_flatten, pts_query_pos, normal_bev_pos,
                                                 attn_mask=attn_mask.log())
            else:
                pts_query_feat = self.decoder[i](pts_query_feat, lidar_feat_flatten, pts_query_pos, normal_bev_pos)

            # Prediction
            res_layer = self.prediction_heads(pts_query_feat)
            res_layer['center'] = res_layer['center'] + pts_query_pos.permute(0, 2, 1)
            ret_dicts.append(res_layer)
            # for next level positional embedding
            pts_query_pos = res_layer['center'].detach().clone().permute(0, 2, 1)

        return pts_query_feat, pts_query_pos, ret_dicts


class ImageTransformer3D(nn.Module):
    def __init__(self, num_views, hidden_channel, num_heads, num_decoder_layers, prediction_heads, img_smca,
                 out_size_factor_img, bbox_coder, ffn_channel, dropout, activation, test_cfg, query_pos, key_pos):
        super(ImageTransformer3D, self).__init__()
        self.hidden_channel = hidden_channel
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.prediction_heads = prediction_heads
        self.img_smca = img_smca
        self.num_views = num_views
        self.bbox_coder = bbox_coder
        self.out_size_factor_img = out_size_factor_img
        self.test_cfg = test_cfg

        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    hidden_channel, num_heads, ffn_channel, dropout, activation,
                    self_posembed=query_pos,
                    cross_posembed=key_pos,
                )
            )

    def forward(self, img_query_feat, img_query_pos, img_feat_flatten, img_feat_pos, img_metas):
        num_proposals = img_query_feat.shape[2]
        batch_size = img_query_feat.shape[0]
        # normal_img_feat_pos = normalize_grid(img_feat_pos)
        ret_dicts = []
        res_layer = self.prediction_heads(img_query_feat)
        res_layer['center'] = res_layer['center'] + img_query_pos.permute(0, 2, 1)
        img_query_pos = res_layer['center'].permute(0, 2, 1)  # [BS, num_proposals, 2]
        # res_layer['center'] = img_query_pos.permute(0, 2, 1)
        for i in range(self.num_decoder_layers):
            img_prev_query_feat = img_query_feat.detach().clone()  # [BS, C, num_proposals]
            img_query_feat = torch.zeros_like(img_query_feat)  # create new container for img query feature

            img_query_pos_realmetric = img_query_pos.permute(0, 2, 1).sigmoid() * (self.test_cfg['pc_range'][3] -
                                    self.test_cfg['pc_range'][0]) + self.test_cfg['pc_range'][0]  # [BS, 2, num_proposals]
            height = res_layer['height'].sigmoid() * (self.test_cfg['pc_range'][5] -
                                    self.test_cfg['pc_range'][2]) + self.test_cfg['pc_range'][2]
            img_query_pos_3d = torch.cat([img_query_pos_realmetric, height], dim=1).detach().clone()  # [BS, 3, num_proposals]

            if 'vel' in res_layer:
                vel = copy.deepcopy(res_layer['vel'].detach())
            else:
                vel = None
            center = copy.deepcopy(res_layer['center'].detach())
            center[:, 0] = center[:, 0].sigmoid() * (self.test_cfg['pc_range'][3] - self.test_cfg['pc_range'][0]) + self.test_cfg['pc_range'][0]
            center[:, 1] = center[:, 1].sigmoid() * (self.test_cfg['pc_range'][4] - self.test_cfg['pc_range'][1]) + self.test_cfg['pc_range'][1]
            height = copy.deepcopy(res_layer['height'].detach())
            height = height.sigmoid() * (self.test_cfg['pc_range'][5] - self.test_cfg['pc_range'][2]) + self.test_cfg['pc_range'][2]

            pred_boxes = self.bbox_coder.decode(
                copy.deepcopy(res_layer['heatmap'].detach()),
                copy.deepcopy(res_layer['rot'].detach()),
                copy.deepcopy(res_layer['dim'].detach()),
                center,
                height,
                # copy.deepcopy(res_layer['center'].detach()),
                # copy.deepcopy(res_layer['height'].detach()),
                vel,
            )

            on_the_image_mask = torch.ones([batch_size, num_proposals]).to(img_query_pos_3d.device) * -1

            for sample_idx in range(batch_size):
                lidar2img_rt = img_query_pos_3d.new_tensor(img_metas[sample_idx]['lidar2img'])
                img_scale_factor = (
                    img_query_pos_3d.new_tensor(img_metas[sample_idx]['scale_factor'][:2]
                                                if 'scale_factor' in img_metas[sample_idx].keys() else [1.0, 1.0])
                )
                img_flip = img_metas[sample_idx]['flip'] if 'flip' in img_metas[sample_idx].keys() else False
                img_crop_offset = (
                    img_query_pos_3d.new_tensor(img_metas[sample_idx]['img_crop_offset'])
                    if 'img_crop_offset' in img_metas[sample_idx].keys() else 0
                )
                img_shape = img_metas[sample_idx]['img_shape'][:2]
                img_pad_shape = img_metas[sample_idx]['input_shape'][:2]
                boxes = LiDARInstance3DBoxes(pred_boxes[sample_idx]['bboxes'][:, :7], box_dim=7)
                img_query_pos_3d_with_corners = torch.cat(
                    [img_query_pos_3d[sample_idx], boxes.corners.permute(2, 0, 1).view(3, -1)], dim=-1
                )  # [3, num_proposals] + [3, num_proposals*8]
                # transform point clouds back to original coordinate system by reverting the data augmentation
                if batch_size == 1:  # skip during inference to save time
                    points = img_query_pos_3d_with_corners.T
                else:
                    points = apply_3d_transformation(
                        img_query_pos_3d_with_corners.T, 'LIDAR', img_metas[sample_idx], reverse=True
                    ).detach()
                num_points = points.shape[0]

                for view_idx in range(self.num_views):
                    pts_4d = torch.cat([points, points.new_ones(size=(num_points, 1))], dim=-1)
                    pts_2d = pts_4d @ lidar2img_rt[view_idx].t()

                    pts_2d[:, 2] = torch.clamp(pts_2d[:, 2], min=1e-5)
                    pts_2d[:, 0] /= pts_2d[:, 2]
                    pts_2d[:, 1] /= pts_2d[:, 2]

                    # img transformation: scale -> crop -> flip
                    # the image is resized by img_scale_factor
                    img_coors = pts_2d[:, 0:2] * img_scale_factor  # Nx2
                    img_coors -= img_crop_offset

                    # grid sample, the valid grid range should be in [-1,1]
                    coor_x, coor_y = torch.split(img_coors, 1, dim=1)  # each is Nx1

                    if img_flip:
                        # by default we take it as horizontal flip
                        # use img_shape before padding for flip
                        orig_h, orig_w = img_shape
                        coor_x = orig_w - coor_x

                    coor_x, coor_corner_x = coor_x[0:num_proposals, :], coor_x[num_proposals:, :]
                    coor_y, coor_corner_y = coor_y[0:num_proposals, :], coor_y[num_proposals:, :]
                    coor_corner_x = coor_corner_x.reshape(num_proposals, 8, 1)
                    coor_corner_y = coor_corner_y.reshape(num_proposals, 8, 1)
                    coor_corner_xy = torch.cat([coor_corner_x, coor_corner_y], dim=-1)  # [num_proposals, 8, 2]

                    h, w = img_pad_shape
                    on_the_image = (coor_x > 0) * (coor_x < w) * (coor_y > 0) * (coor_y < h)
                    on_the_image = on_the_image.squeeze()  # [num_proposals, ]
                    # skip the following computation if no object query fall on current image
                    if on_the_image.sum() <= 1:
                        continue
                    on_the_image_mask[sample_idx, on_the_image] = view_idx

                    # add spatial constraint
                    center_ys = coor_y[on_the_image] / self.out_size_factor_img
                    center_xs = coor_x[on_the_image] / self.out_size_factor_img
                    # normal_center_ys = inverse_sigmoid(coor_y[on_the_image] / h)
                    # normal_center_xs = inverse_sigmoid(coor_x[on_the_image] / w)

                    img_query_feat_view = img_prev_query_feat[sample_idx, :, on_the_image]  # [C, N_image]
                    img_query_pos_view = img_query_pos[sample_idx, on_the_image]  # [N_image, 2]
                    # img_query_pos_view = torch.cat([normal_center_xs, normal_center_ys], dim=-1)  # [N_image, 2]

                    img_feat_pos_view = img_feat_pos * self.out_size_factor_img  # [1, W*H, 2]
                    if img_flip:
                        # by default we take it as horizontal flip
                        # use img_shape before padding for flip
                        orig_h, orig_w = img_shape
                        img_feat_pos_view[...,0] = orig_w - img_feat_pos_view[...,0]  # [1, W*H, 2]

                    img_feat_pos_view = img_feat_pos_view / img_scale_factor  # Nx2
                    img_feat_pos_view = img_feat_pos_view + img_crop_offset  # [1, W*H, 2]

                    img_feat_pos_view_3d = self.img_pos_3d(img_feat_pos_view[0], lidar2img_rt[view_idx])

                    if self.img_smca:
                        centers = torch.cat([center_xs, center_ys], dim=-1).int()
                        # [num_proposals, 2] # center on the feature map
                        corners = (coor_corner_xy[on_the_image].max(1).values - coor_corner_xy[on_the_image].min(
                            1).values) / self.out_size_factor_img
                        radius = torch.ceil(corners.norm(dim=-1, p=2) / 2).int()  # radius of the minimum circumscribed circle of the wireframe
                        sigma = (radius * 2 + 1) / 6.0
                        distance = (centers[:, None, :] - (img_feat_pos - 0.5)).norm(dim=-1) ** 2
                        gaussian_mask = (-distance / (2 * sigma[:, None] ** 2)).exp()
                        gaussian_mask[gaussian_mask < torch.finfo(torch.float32).eps] = 0
                        attn_mask = gaussian_mask
                        img_query_feat_view = self.decoder[i](
                            img_query_feat_view[None], img_feat_flatten[sample_idx:sample_idx + 1,view_idx],
                            img_query_pos_view[None], img_feat_pos_view_3d[None], attn_mask=attn_mask.log()
                        )
                    else:
                        img_query_feat_view = self.decoder[i](
                            img_query_feat_view[None], img_feat_flatten[sample_idx:sample_idx+1, view_idx],
                            img_query_pos_view[None], img_feat_pos_view_3d[None]
                        )

                    img_query_feat[sample_idx, :, on_the_image] = img_query_feat_view.clone()

            self.on_the_image_mask = (on_the_image_mask != -1)
            prev_res_layer = res_layer
            res_layer = self.prediction_heads(img_query_feat)
            res_layer['center'] = res_layer['center'] + img_query_pos.permute(0, 2, 1)

            for key, value in res_layer.items():
                pred_dim = value.shape[1]
                res_layer[key][~self.on_the_image_mask.unsqueeze(1).repeat(1, pred_dim, 1)] = \
                    prev_res_layer[key][~self.on_the_image_mask.unsqueeze(1).repeat(1, pred_dim, 1)]

            img_query_pos = res_layer['center'].detach().clone().permute(0, 2, 1)
            ret_dicts.append(res_layer)

        return img_query_feat, img_query_pos, ret_dicts, self.on_the_image_mask

    def img_pos_3d(self, img_pos, lidar2img, D=32):
        # img_pos: [W*H, 2]
        img_pos = img_pos[:, None].repeat(1, D, 1)  # [W*H, D, 2]
        depth = torch.arange(D).to(img_pos.device).view(1, D, 1).repeat(img_pos.shape[0], 1, 1)  # [W*H, D, 1]
        coords = torch.cat([img_pos, depth], dim=2)  # [W*H, D, 3]
        coords = torch.cat([coords, torch.ones_like(coords[..., :1])], dim=2)  # [W*H, D, 4]

        eps = 1e-5
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)

        img2lidars = torch.inverse(lidar2img)
        coords3d = torch.matmul(img2lidars, coords.unsqueeze(-1)).squeeze(-1)[..., :3]  # [W*H, D, 3]

        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.test_cfg['pc_range'][0]) / (self.test_cfg['pc_range'][3] - self.test_cfg['pc_range'][0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.test_cfg['pc_range'][1]) / (self.test_cfg['pc_range'][4] - self.test_cfg['pc_range'][1])
        coords3d[..., 2:3] = (coords3d[..., 2:3] - self.test_cfg['pc_range'][2]) / (self.test_cfg['pc_range'][5] - self.test_cfg['pc_range'][2])

        coords3d = inverse_sigmoid(coords3d)
        coords3d = coords3d.view(coords3d.size(0), D*3)
        return coords3d