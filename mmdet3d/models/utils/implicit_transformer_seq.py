import copy
import numpy as np
import torch
import time
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
from mmdet3d.models.utils.deformable_decoder import DeformableTransformerDecoderLayer
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

class ImageTransformer_Seq_MS(nn.Module):
    def __init__(self, num_views, hidden_channel, num_heads, num_decoder_layers, prediction_heads, out_size_factor_img,
                 ffn_channel, dropout, activation, test_cfg, query_pos, key_pos):
        super(ImageTransformer_Seq_MS, self).__init__()
        self.hidden_channel = hidden_channel
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.prediction_heads = prediction_heads
        self.num_views = num_views
        self.out_size_factor_img = out_size_factor_img
        self.test_cfg = test_cfg

        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                DeformableTransformerDecoderLayer(
                    hidden_channel, num_heads, dim_feedforward=ffn_channel, dropout=dropout, activation=activation,
                    self_posembed=query_pos[i], cross_posembed=key_pos[i],
                )
            )

    def forward(self, img_query_feat, img_query_pos, img_query_view_mask, img_feats, normal_img_feats_pos_stack,
                img_metas):
        # img_query_feat [bs, C, num_proposals]
        # img_query_pos [bs, num_proposals, 3]
        # img_query_view_mask [bs, num_proposals, num_views]
        # normal_img_feats_pos_stack [1, h*w (sum), 2]

        num_img_proposals = img_query_feat.shape[-1]
        level_num = len(img_feats)
        batch_size = img_query_feat.shape[0]
        img_feats_flatten = []
        level_start_index = [0]
        spatial_shapes = []
        for lvl in range(level_num):
            img_feat = img_feats[lvl]
            h, w = img_feat.shape[-2], img_feat.shape[-1]
            img_feat_flatten = img_feat.view(batch_size, self.num_views, self.hidden_channel, h * w)  # [bs, num_view, C, h*w]
            img_feats_flatten.append(img_feat_flatten)
            level_start_index.append(level_start_index[-1] + h * w)
            spatial_shapes.append([h, w])
        level_start_index = level_start_index[:-1]
        level_start_index = torch.LongTensor(level_start_index).to(img_query_feat.device)
        spatial_shapes = torch.LongTensor(spatial_shapes).to(img_query_feat.device)

        img_feats_stack = torch.cat(img_feats_flatten, dim=3)  # [bs, num_view, C, h*w (sum)]

        img_feats_pos_stack = torch.ones([batch_size, self.num_views, normal_img_feats_pos_stack.shape[1], 4]).to(img_feats_stack.device)  # [bs, num_views, h*w (sum), 4]
        cam_ts = []
        lidar2img_rts = []
        for sample_idx in range(batch_size):
            img_pad_shape = img_metas[sample_idx]['input_shape'][:2]
            h, w = img_pad_shape
            img_feats_pos_stack[sample_idx, :, :, :2] = denormalize_pos(normal_img_feats_pos_stack, w, h)

            lidar2cam_r = torch.Tensor(img_metas[sample_idx]['lidar2cam_r']).to(img_feats_pos_stack.device)  # [num_views, 3, 3]
            lidar2cam_t = torch.Tensor(img_metas[sample_idx]['lidar2cam_t']).to(img_feats_pos_stack.device)  # [num_views, 3]
            lidar2cam_t = lidar2cam_t.unsqueeze(-1)  # [num_views, 3, 1]
            cam_t = -torch.matmul(lidar2cam_r.permute(0, 2, 1), lidar2cam_t).squeeze(-1)  # [num_views, 3]
            cam_ts.append(cam_t)

            lidar2img_rt = torch.Tensor(img_metas[sample_idx]['lidar2img_rt']).to(img_feats_pos_stack.device)  # [num_views, h*w (sum), 4, 4]
            lidar2img_rts.append(lidar2img_rt)

            img_feats_pos_stack[sample_idx] = torch.matmul(torch.inverse(lidar2img_rt[:, None]), img_feats_pos_stack[sample_idx].unsqueeze(-1)).squeeze(-1) # [bs, num_views, h*w (sum), 4]

        img_feats_pos_stack = img_feats_pos_stack[..., :3]  # [bs, num_views, h*w (sum), 3]
        # img_feats_pos_stack = F.normalize(img_feats_pos_stack, dim=-1)  # [bs, num_views, h*w (sum), 3]
        lidar2img_rts = torch.stack(lidar2img_rts, dim=0)
        cam_ts = torch.stack(cam_ts, dim=0)  # [bs, num_views, 3]
        cam_ts = cam_ts.unsqueeze(2)  # [bs, num_views, 1, 3]
        cam_ts = cam_ts.repeat(1, 1, img_feats_pos_stack.shape[2], 1)  # [bs, num_views, h*w (sum), 3]
        img_feats_pos_stack = torch.cat([img_feats_pos_stack, cam_ts], dim=-1)

        ret_dicts = []
        for i in range(self.num_decoder_layers):
            img_prev_query_feat = img_query_feat.clone()  # [BS, C, num_proposals]
            img_query_feat = torch.zeros([batch_size, self.hidden_channel, num_img_proposals, self.num_views]).to(img_prev_query_feat.device)
            img_query_count = torch.zeros([batch_size, 1, num_img_proposals, self.num_views]).to(img_prev_query_feat.device)

            for sample_idx in range(batch_size):
                for view_idx in range(self.num_views):
                    on_the_image = img_query_view_mask[sample_idx, :, view_idx] > 0 # [num_on_the_image, ]

                    if torch.sum(on_the_image) <= 1:
                        continue

                    query_pos_view = img_query_pos[sample_idx, on_the_image]  # [num_on_the_image, 3]
                    query_pos_view_pad = torch.cat([query_pos_view, torch.ones_like(query_pos_view[..., :1])], dim=-1)  # [num_on_the_image, 4]

                    query_pos_img_view_pad = query_pos_view_pad @ lidar2img_rts[sample_idx, view_idx].T  # [num_on_the_image, 4]
                    query_pos_img_view = query_pos_img_view_pad[..., :2]  # [num_on_the_image, 2]
                    query_pos_img_view = query_pos_img_view / query_pos_img_view_pad[..., 2:3]
                    reference_points = torch.zeros_like(query_pos_img_view)
                    # if 'valid_shape' in img_metas[sample_idx]:
                    #     valid_w, valid_h = img_metas[sample_idx]['valid_shape'][view_idx]
                    # else:
                    #     valid_h, valid_w = img_metas[sample_idx]['input_shape']
                    img_h, img_w = img_metas[sample_idx]['input_shape']

                    reference_points[..., 0] = query_pos_img_view[..., 0] / img_w
                    reference_points[..., 1] = query_pos_img_view[..., 1] / img_h

                    reference_points = reference_points[:, None].repeat(1, level_num, 1)

                    img_query_feat_view = img_prev_query_feat[sample_idx, :, on_the_image]  # [C, num_on_the_image]

                    img_query_feat_view = self.decoder[i](
                        img_query_feat_view[None], img_feats_stack[sample_idx:sample_idx + 1, view_idx],
                        query_pos_view[None], img_feats_pos_stack[sample_idx:sample_idx+1, view_idx],
                        reference_points=reference_points[None],
                        level_start_index=level_start_index, spatial_shapes=spatial_shapes
                    )
                    img_query_feat[sample_idx, :, on_the_image, view_idx] = img_query_feat_view.clone()
                    img_query_count[sample_idx, :, on_the_image, view_idx] += 1

            img_query_feat = torch.sum(img_query_feat, dim=-1)  # [BS, C, num_proposals]
            img_query_count = torch.sum(img_query_count, dim=-1)  # [BS, 1, num_proposals]
            img_query_feat = img_query_feat / (img_query_count + 1e-4)
            res_layer = self.prediction_heads(img_query_feat)

            res_layer['center_view'] = res_layer['center_view'] + img_query_pos.permute(0, 2, 1)
            img_query_pos = res_layer['center_view'].detach().clone().permute(0, 2, 1)

            res_layer['height_view'] = res_layer['center_view'][:, 2:3]

            res_layer['center_view'] = res_layer['center_view'][:, :2]
            res_layer['center_view'][:, 0] = (res_layer['center_view'][:, 0] - self.test_cfg['pc_range'][0]) / (
                    self.test_cfg['pc_range'][3] - self.test_cfg['pc_range'][0])
            res_layer['center_view'][:, 1] = (res_layer['center_view'][:, 1] - self.test_cfg['pc_range'][1]) / (
                    self.test_cfg['pc_range'][4] - self.test_cfg['pc_range'][1])

            res_layer['center_view'][:, 0] = res_layer['center_view'][:, 0] * (
                        self.test_cfg['grid_size'][0] // self.test_cfg['out_size_factor'])
            res_layer['center_view'][:, 1] = res_layer['center_view'][:, 1] * (
                        self.test_cfg['grid_size'][1] // self.test_cfg['out_size_factor'])
            ret_dicts.append(res_layer)

            img_query_pos_bev = res_layer['center_view'].detach().clone().permute(0, 2, 1)

        return img_query_feat, img_query_pos, img_query_pos_bev, ret_dicts


class SEBlock(nn.Module):
    def __init__(self, hidden_channel, cwa_avg=True):
        super(SEBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channel, hidden_channel),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, hidden_channel),
            nn.Sigmoid()
        )
        self.cwa_avg = cwa_avg

    def forward(self, feat):
        if self.cwa_avg:
            feat_avg = torch.mean(feat, dim=-1)
            mul = self.mlp(feat_avg).unsqueeze(-1)
        else:
            mul = self.mlp(feat.permute(0, 2, 1)).permute(0, 2, 1)
        feat = feat * mul
        return feat


class FusionTransformer_Seq(nn.Module):
    def __init__(self, hidden_channel, num_heads, num_decoder_layers, prediction_heads, ffn_channel, dropout, activation, test_cfg,
                 query_pos, key_pos, fuse_projection, cwa_avg):
        super(FusionTransformer_Seq, self).__init__()
        self.hidden_channel = hidden_channel
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.prediction_heads = prediction_heads
        self.test_cfg = test_cfg
        self.grid_x_size = test_cfg['grid_size'][0] // test_cfg['out_size_factor']
        self.grid_y_size = test_cfg['grid_size'][1] // test_cfg['out_size_factor']
        self.fuse_projection = fuse_projection
        self.cwa_avg = cwa_avg

        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    hidden_channel, num_heads, ffn_channel, dropout, activation,
                    self_posembed=query_pos[i], cross_posembed=key_pos[i], cross_only=True
                )
            )

        self.se = SEBlock(hidden_channel*2, cwa_avg=cwa_avg)

    def forward(self, pts_query_feat, img_query_feat, pts_query_pos):
        ret_dicts = []
        fuse_query_feat = torch.cat([pts_query_feat, img_query_feat], dim=1)
        fuse_query_feat = self.se(fuse_query_feat)
        fuse_query_feat = self.fuse_projection(fuse_query_feat)

        for i in range(self.num_decoder_layers):
            # Transformer Decoder Layer
            # :param query: B C Pq    :param query_pos: B Pq 3/6
            fuse_query_feat = self.decoder[i](fuse_query_feat, fuse_query_feat, pts_query_pos, pts_query_pos)

            # Prediction
            res_layer = self.prediction_heads(fuse_query_feat)

            res_layer['center'] = res_layer['center'] + pts_query_pos.permute(0, 2, 1)

            ret_dicts.append(res_layer)

            pts_query_pos = res_layer['center'].detach().clone().permute(0, 2, 1)

        return fuse_query_feat, pts_query_pos, ret_dicts

class ImageTransformer_Seq_DETR_MS(nn.Module):
    def __init__(self, num_views, hidden_channel, num_heads, num_decoder_layers, prediction_heads, out_size_factor_img,
                 ffn_channel, dropout, activation, test_cfg, query_pos, key_pos):
        super(ImageTransformer_Seq_DETR_MS, self).__init__()
        self.hidden_channel = hidden_channel
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.prediction_heads = prediction_heads
        self.num_views = num_views
        self.out_size_factor_img = out_size_factor_img
        self.test_cfg = test_cfg

        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    hidden_channel, num_heads, dim_feedforward=ffn_channel, dropout=dropout, activation=activation,
                    self_posembed=query_pos[i], cross_posembed=key_pos[i],
                )
            )

    def forward(self, img_query_feat, img_query_pos, img_query_view_mask, img_feats, normal_img_feats_pos_stack,
                img_metas):
        # img_query_feat [bs, C, num_proposals]
        # img_query_pos [bs, num_proposals, 3]
        # img_query_view_mask [bs, num_proposals, num_views]
        # normal_img_feats_pos_stack [1, h*w (sum), 2]

        num_img_proposals = img_query_feat.shape[-1]
        level_num = len(img_feats)
        batch_size = img_query_feat.shape[0]
        img_feats_flatten = []
        for lvl in range(level_num):
            img_feat = img_feats[lvl]
            h, w = img_feat.shape[-2], img_feat.shape[-1]
            img_feat_flatten = img_feat.view(batch_size, self.num_views, self.hidden_channel, h * w)  # [bs, num_view, C, h*w]
            img_feats_flatten.append(img_feat_flatten)

        img_feats_stack = torch.cat(img_feats_flatten, dim=3)  # [bs, num_view, C, h*w (sum)]
        # img_feats_stack = img_feats_stack.permute(0, 2, 1, 3).contiguous()
        # img_feats_stack = img_feats_stack.view(batch_size, self.hidden_channel, -1)

        img_feats_pos_stack = torch.ones([batch_size, self.num_views, normal_img_feats_pos_stack.shape[1], 4]).to(img_feats_stack.device)  # [bs, num_views, h*w (sum), 4]
        cam_ts = []
        for sample_idx in range(batch_size):
            img_pad_shape = img_metas[sample_idx]['input_shape'][:2]
            h, w = img_pad_shape
            img_feats_pos_stack[sample_idx, :, :, :2] = denormalize_pos(normal_img_feats_pos_stack, w, h)

            lidar2cam_r = torch.Tensor(img_metas[sample_idx]['lidar2cam_r']).to(img_feats_pos_stack.device)  # [num_views, 3, 3]
            lidar2cam_t = torch.Tensor(img_metas[sample_idx]['lidar2cam_t']).to(img_feats_pos_stack.device)  # [num_views, 3]
            lidar2cam_t = lidar2cam_t.unsqueeze(-1)  # [num_views, 3, 1]
            cam_t = -torch.matmul(lidar2cam_r.permute(0, 2, 1), lidar2cam_t).squeeze(-1)  # [num_views, 3]
            cam_ts.append(cam_t)

            lidar2img_rt = torch.Tensor(img_metas[sample_idx]['lidar2img_rt']).to(img_feats_pos_stack.device)  # [num_views, h*w (sum), 4, 4]

            img_feats_pos_stack[sample_idx] = torch.matmul(torch.inverse(lidar2img_rt[:, None]), img_feats_pos_stack[sample_idx].unsqueeze(-1)).squeeze(-1) # [bs, num_views, h*w (sum), 4]

        img_feats_pos_stack = img_feats_pos_stack[..., :3]  # [bs, num_views, h*w (sum), 3]
        # img_feats_pos_stack = F.normalize(img_feats_pos_stack, dim=-1)  # [bs, num_views, h*w (sum), 3]

        cam_ts = torch.stack(cam_ts, dim=0)  # [bs, num_views, 3]
        cam_ts = cam_ts.unsqueeze(2)  # [bs, num_views, 1, 3]
        cam_ts = cam_ts.repeat(1, 1, img_feats_pos_stack.shape[2], 1)  # [bs, num_views, h*w (sum), 3]
        img_feats_pos_stack = torch.cat([img_feats_pos_stack, cam_ts], dim=-1)
        # img_feats_pos_stack = img_feats_pos_stack.view(batch_size, -1, 6)

        ret_dicts = []
        for i in range(self.num_decoder_layers):
            img_prev_query_feat = img_query_feat.clone()  # [BS, C, num_proposals]
            img_query_feat = torch.zeros([batch_size, self.hidden_channel, num_img_proposals, self.num_views]).to(img_prev_query_feat.device)
            img_query_count = torch.zeros([batch_size, 1, num_img_proposals, self.num_views]).to(img_prev_query_feat.device)

            for sample_idx in range(batch_size):
                for view_idx in range(self.num_views):
                    on_the_image = img_query_view_mask[sample_idx, :, view_idx] > 0 # [num_on_the_image, ]

                    if torch.sum(on_the_image) <= 1:
                        continue

                    query_pos_view = img_query_pos[sample_idx, on_the_image]  # [num_on_the_image, 3]
                    img_query_feat_view = img_prev_query_feat[sample_idx, :, on_the_image]  # [C, num_on_the_image]

                    img_query_feat_view = self.decoder[i](
                        img_query_feat_view[None], img_feats_stack[sample_idx:sample_idx + 1, view_idx],
                        query_pos_view[None], img_feats_pos_stack[sample_idx:sample_idx+1, view_idx],
                    )
                    img_query_feat[sample_idx, :, on_the_image, view_idx] = img_query_feat_view.clone()
                    img_query_count[sample_idx, :, on_the_image, view_idx] += 1

            img_query_feat = torch.sum(img_query_feat, dim=-1)  # [BS, C, num_proposals]
            img_query_count = torch.sum(img_query_count, dim=-1)  # [BS, 1, num_proposals]
            img_query_feat = img_query_feat / (img_query_count + 1e-4)
            # img_query_feat = self.decoder[i](img_query_feat, img_feats_stack, img_query_pos, img_feats_pos_stack)
            res_layer = self.prediction_heads(img_query_feat)

            res_layer['center_view'] = res_layer['center_view'] + img_query_pos.permute(0, 2, 1)
            img_query_pos = res_layer['center_view'].detach().clone().permute(0, 2, 1)

            res_layer['height_view'] = res_layer['center_view'][:, 2:3]

            res_layer['center_view'] = res_layer['center_view'][:, :2]
            res_layer['center_view'][:, 0] = (res_layer['center_view'][:, 0] - self.test_cfg['pc_range'][0]) / (
                    self.test_cfg['pc_range'][3] - self.test_cfg['pc_range'][0])
            res_layer['center_view'][:, 1] = (res_layer['center_view'][:, 1] - self.test_cfg['pc_range'][1]) / (
                    self.test_cfg['pc_range'][4] - self.test_cfg['pc_range'][1])

            res_layer['center_view'][:, 0] = res_layer['center_view'][:, 0] * (
                        self.test_cfg['grid_size'][0] // self.test_cfg['out_size_factor'])
            res_layer['center_view'][:, 1] = res_layer['center_view'][:, 1] * (
                        self.test_cfg['grid_size'][1] // self.test_cfg['out_size_factor'])
            ret_dicts.append(res_layer)

            img_query_pos_bev = res_layer['center_view'].detach().clone().permute(0, 2, 1)

        return img_query_feat, img_query_pos, img_query_pos_bev, ret_dicts


class ImageTransformer_Seq_PETR_MS(nn.Module):
    def __init__(self, num_views, hidden_channel, num_heads, num_decoder_layers, prediction_heads, out_size_factor_img,
                 ffn_channel, dropout, activation, test_cfg, query_pos, key_pos, depth_num=32):
        super(ImageTransformer_Seq_PETR_MS, self).__init__()
        self.hidden_channel = hidden_channel
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.prediction_heads = prediction_heads
        self.num_views = num_views
        self.out_size_factor_img = out_size_factor_img
        self.test_cfg = test_cfg
        self.depth_num = depth_num

        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    hidden_channel, num_heads, dim_feedforward=ffn_channel, dropout=dropout, activation=activation,
                    self_posembed=query_pos[i], cross_posembed=key_pos[i],
                )
            )

    def forward(self, img_query_feat, img_query_pos, img_query_view_mask, img_feats, normal_img_feats_pos_stack,
                img_metas):
        # img_query_feat [bs, C, num_proposals]
        # img_query_pos [bs, num_proposals, 3]
        # img_query_view_mask [bs, num_proposals, num_views]
        # normal_img_feats_pos_stack [1, h*w (sum), 2]

        num_img_proposals = img_query_feat.shape[-1]
        level_num = len(img_feats)
        batch_size = img_query_feat.shape[0]
        img_feats_flatten = []
        for lvl in range(level_num):
            img_feat = img_feats[lvl]
            h, w = img_feat.shape[-2], img_feat.shape[-1]
            img_feat_flatten = img_feat.view(batch_size, self.num_views, self.hidden_channel, h * w)  # [bs, num_view, C, h*w]
            img_feats_flatten.append(img_feat_flatten)

        img_feats_stack = torch.cat(img_feats_flatten, dim=3)  # [bs, num_view, C, h*w (sum)]
        # img_feats_stack = img_feats_stack.permute(0, 2, 1, 3).contiguous()
        # img_feats_stack = img_feats_stack.view(batch_size, self.hidden_channel, -1)

        img_feats_pos_stack = torch.ones([batch_size, self.num_views, normal_img_feats_pos_stack.shape[1], self.depth_num, 4]).to(img_feats_stack.device)  # [bs, num_views, h*w (sum), D, 4]
        cam_ts = []
        index = torch.arange(start=0, end=self.depth_num, step=1, device=img_query_feat.device).float()  # [D, ]
        index_1 = index + 1
        bin_size = (self.test_cfg['pc_range'][3] - 1) / (self.depth_num * (1 + self.depth_num))
        coords_d = 1 + bin_size * index * index_1  # [D, ]
        coords_d = torch.maximum(coords_d, torch.ones_like(coords_d) * 1e-5)
        coords_d = coords_d.reshape(1, 1, self.depth_num, 1)  # [1, 1, D, 1]
        for sample_idx in range(batch_size):
            img_pad_shape = img_metas[sample_idx]['input_shape'][:2]
            h, w = img_pad_shape

            img_feats_pos_stack_view = denormalize_pos(normal_img_feats_pos_stack, w, h)  # [1, h*w (sum), 2]
            img_feats_pos_stack_view = img_feats_pos_stack_view[:, :, None].repeat(1, 1, self.depth_num, 1)  # [1, h*w (sum), D, 2]
            img_feats_pos_stack_view = torch.cat([img_feats_pos_stack_view, torch.ones_like(img_feats_pos_stack_view[..., :1])], dim=-1)  # [1, h*w (sum), D, 3]
            img_feats_pos_stack_view = img_feats_pos_stack_view * coords_d

            img_feats_pos_stack[sample_idx, :, :, :, :3] = img_feats_pos_stack_view.clone()

            lidar2img_rt = torch.Tensor(img_metas[sample_idx]['lidar2img_rt']).to(img_feats_pos_stack.device)  # [num_views, h*w (sum), 4, 4]

            img_feats_pos_stack[sample_idx] = torch.matmul(img_feats_pos_stack[sample_idx], torch.inverse(lidar2img_rt[:, None]).permute(0, 1, 3, 2)) # [bs, num_views, h*w (sum), D, 4]

        img_feats_pos_stack = img_feats_pos_stack[..., :3]  # [bs, num_views, h*w (sum), D, 3]
        img_feats_pos_stack = img_feats_pos_stack.reshape(batch_size, self.num_views, img_feats_pos_stack.shape[2], self.depth_num * 3)

        ret_dicts = []
        for i in range(self.num_decoder_layers):
            img_prev_query_feat = img_query_feat.clone()  # [BS, C, num_proposals]
            img_query_feat = torch.zeros([batch_size, self.hidden_channel, num_img_proposals, self.num_views]).to(img_prev_query_feat.device)
            img_query_count = torch.zeros([batch_size, 1, num_img_proposals, self.num_views]).to(img_prev_query_feat.device)

            for sample_idx in range(batch_size):
                for view_idx in range(self.num_views):
                    on_the_image = img_query_view_mask[sample_idx, :, view_idx] > 0 # [num_on_the_image, ]

                    if torch.sum(on_the_image) <= 1:
                        continue

                    query_pos_view = img_query_pos[sample_idx, on_the_image]  # [num_on_the_image, 3]
                    img_query_feat_view = img_prev_query_feat[sample_idx, :, on_the_image]  # [C, num_on_the_image]

                    img_query_feat_view = self.decoder[i](
                        img_query_feat_view[None], img_feats_stack[sample_idx:sample_idx + 1, view_idx],
                        query_pos_view[None], img_feats_pos_stack[sample_idx:sample_idx+1, view_idx],
                    )
                    img_query_feat[sample_idx, :, on_the_image, view_idx] = img_query_feat_view.clone()
                    img_query_count[sample_idx, :, on_the_image, view_idx] += 1

            img_query_feat = torch.sum(img_query_feat, dim=-1)  # [BS, C, num_proposals]
            img_query_count = torch.sum(img_query_count, dim=-1)  # [BS, 1, num_proposals]
            img_query_feat = img_query_feat / (img_query_count + 1e-4)
            # img_query_feat = self.decoder[i](img_query_feat, img_feats_stack, img_query_pos, img_feats_pos_stack)
            res_layer = self.prediction_heads(img_query_feat)

            res_layer['center_view'] = res_layer['center_view'] + img_query_pos.permute(0, 2, 1)
            img_query_pos = res_layer['center_view'].detach().clone().permute(0, 2, 1)

            res_layer['height_view'] = res_layer['center_view'][:, 2:3]

            res_layer['center_view'] = res_layer['center_view'][:, :2]
            res_layer['center_view'][:, 0] = (res_layer['center_view'][:, 0] - self.test_cfg['pc_range'][0]) / (
                    self.test_cfg['pc_range'][3] - self.test_cfg['pc_range'][0])
            res_layer['center_view'][:, 1] = (res_layer['center_view'][:, 1] - self.test_cfg['pc_range'][1]) / (
                    self.test_cfg['pc_range'][4] - self.test_cfg['pc_range'][1])

            res_layer['center_view'][:, 0] = res_layer['center_view'][:, 0] * (
                        self.test_cfg['grid_size'][0] // self.test_cfg['out_size_factor'])
            res_layer['center_view'][:, 1] = res_layer['center_view'][:, 1] * (
                        self.test_cfg['grid_size'][1] // self.test_cfg['out_size_factor'])
            ret_dicts.append(res_layer)

            img_query_pos_bev = res_layer['center_view'].detach().clone().permute(0, 2, 1)

        return img_query_feat, img_query_pos, img_query_pos_bev, ret_dicts