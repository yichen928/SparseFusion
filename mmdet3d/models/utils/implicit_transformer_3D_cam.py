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

from mmdet3d.models.utils import TransformerDecoderLayer, MultiheadAttention, PositionEmbeddingLearned, PositionEmbeddingLearnedwoNorm
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


class CameraSEAdd(nn.Module):
    def __init__(self, cam_dim, hidden_channel):
        super(CameraSEAdd, self).__init__()
        self.bn = nn.BatchNorm1d(cam_dim)

        self.hidden_channel = hidden_channel
        self.mlp_depth = nn.Sequential(
            nn.Conv1d(cam_dim, hidden_channel, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1),
        )

        self.mlp_proj = nn.Sequential(
            nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1),
        )

    def forward(self, feat, cam_info):
        cam_info_bn = self.bn(cam_info)
        cam_feat = self.mlp_depth(cam_info_bn)

        pred = feat * self.mlp_proj(cam_feat + feat).sigmoid()
        return pred


class CameraSE(nn.Module):
    def __init__(self, cam_dim, hidden_channel):
        super(CameraSE, self).__init__()
        self.bn = nn.BatchNorm1d(cam_dim)

        self.hidden_channel = hidden_channel
        self.mlp_depth = nn.Sequential(
            nn.Conv1d(cam_dim, hidden_channel, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1),
        )

    def forward(self, feat, cam_info):
        cam_info_bn = self.bn(cam_info)
        pred = feat * self.mlp_depth(cam_info_bn).sigmoid()
        return pred


class CameraAdd(nn.Module):
    def __init__(self, cam_dim, hidden_channel):
        super(CameraAdd, self).__init__()
        self.bn = nn.BatchNorm1d(cam_dim)

        self.hidden_channel = hidden_channel
        self.mlp_depth = nn.Sequential(
            nn.Linear(cam_dim, hidden_channel),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, hidden_channel),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, hidden_channel),
            nn.LayerNorm(hidden_channel)
        )

    def forward(self, feat, cam_info):
        cam_info_bn = self.bn(cam_info)
        pred = feat + self.mlp_depth(cam_info_bn.permute(0, 2, 1)).permute(0, 2, 1)
        return pred


class CameraCat(nn.Module):
    def __init__(self, cam_dim, hidden_channel):
        super(CameraCat, self).__init__()
        self.bn = nn.BatchNorm1d(cam_dim)

        self.hidden_channel = hidden_channel
        self.mlp_camera = nn.Sequential(
            nn.Conv1d(cam_dim, hidden_channel, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1),
        )

        self.mlp_proj = nn.Sequential(
            nn.Linear(hidden_channel*2, hidden_channel),
            nn.LayerNorm(hidden_channel),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, hidden_channel),
            nn.LayerNorm(hidden_channel),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, hidden_channel),
        )

    def forward(self, feat, cam_info):
        cam_info = self.bn(cam_info)
        cam_feat = self.mlp_camera(cam_info)
        feat = torch.cat([feat, cam_feat], dim=1)
        feat = feat.transpose(2, 1)
        pred = self.mlp_proj(feat)
        pred = pred.transpose(2, 1)

        return pred

class ImageTransformer_Cam_3D_MS(nn.Module):
    def __init__(self, num_views, hidden_channel, num_heads, num_decoder_layers, prediction_heads, out_size_factor_img,
                 ffn_channel, dropout, activation, test_cfg, query_pos, key_pos, use_camera=None, bbox_pos=False,
                 allocentric=False, virtual_depth=False):
        super(ImageTransformer_Cam_3D_MS, self).__init__()
        self.hidden_channel = hidden_channel
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.prediction_heads = prediction_heads
        self.num_views = num_views
        self.out_size_factor_img = out_size_factor_img
        self.test_cfg = test_cfg
        self.use_camera = use_camera
        self.bbox_pos = bbox_pos
        self.allocentric = allocentric
        self.virtual_depth = virtual_depth

        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                DeformableTransformerDecoderLayer(
                    hidden_channel, num_heads, dim_feedforward=ffn_channel, dropout=dropout, activation=activation,
                    self_posembed=query_pos[i], cross_posembed=key_pos[i],
                )
            )

        if virtual_depth:
            camera_dim = 18
        else:
            camera_dim = 16

        if use_camera == 'se':
            self.camera_net = CameraSE(camera_dim, hidden_channel)
        elif self.use_camera == 'se_add':
            self.camera_net = CameraSEAdd(camera_dim, hidden_channel)
        elif self.use_camera == 'se_sep':
            self.camera_net = CameraSE(4, hidden_channel)
        elif self.use_camera == 'add':
            self.camera_net = CameraAdd(camera_dim, hidden_channel)
        elif self.use_camera == 'disentangle':
            self.camera_net_int = CameraSE(4, hidden_channel)
            self.camera_net = CameraSE(camera_dim, hidden_channel)
        else:
            self.camera_net = CameraCat(camera_dim, hidden_channel)

    def forward(self, img_query_feat, normal_img_query_pos, img_query_view, img_feats, normal_img_feats_pos_stack, lidar2cam_rt, cam_intrinsic, img_metas, input_padding_mask):
        num_img_proposals = img_query_feat.shape[-1]
        level_num = len(img_feats)
        batch_size = img_query_feat.shape[0]
        img_feats_flatten = []
        level_start_index = [0]
        spatial_shapes = []
        for lvl in range(level_num):
            img_feat = img_feats[lvl]
            h, w = img_feat.shape[-2], img_feat.shape[-1]
            img_feat_flatten = img_feat.view(batch_size, self.num_views, self.hidden_channel, h*w)  # [bs, num_view, C, h*w]
            img_feats_flatten.append(img_feat_flatten)
            level_start_index.append(level_start_index[-1] + h*w)
            spatial_shapes.append([h, w])
        level_start_index = level_start_index[:-1]
        level_start_index = torch.LongTensor(level_start_index).to(img_query_feat.device)
        spatial_shapes = torch.LongTensor(spatial_shapes).to(img_query_feat.device)

        img_feats_stack = torch.cat(img_feats_flatten, dim=3)  # [bs, num_view, C, h*w (sum)]
        reference_points = normal_img_query_pos.sigmoid()  # [bs, num_img_proposal, 2]
        reference_points = reference_points[:, :, None].repeat(1, 1, level_num, 1)

        if self.virtual_depth:
            camera_info = torch.zeros([batch_size, 18, num_img_proposals]).to(img_query_feat.device)
            camera_info[:, 16:] = 1
        else:
            camera_info = torch.zeros([batch_size, 16, num_img_proposals]).to(img_query_feat.device)

        camera_info[:, :9] = lidar2cam_rt[:, :, :3, :3].permute(0, 2, 3, 1).reshape(batch_size, 9, num_img_proposals)
        camera_info[:, 9:12] = lidar2cam_rt[:, :, :3, 3].permute(0, 2, 1)
        camera_info[:, 12] = cam_intrinsic[:, :, 0, 0]
        camera_info[:, 13] = cam_intrinsic[:, :, 1, 1]
        camera_info[:, 14:16] = cam_intrinsic[:, :, :2, 2].permute(0, 2, 1)

        ret_dicts = []

        for i in range(self.num_decoder_layers):
            img_prev_query_feat = img_query_feat.clone()  # [BS, C, num_proposals]
            img_query_feat = torch.zeros_like(img_query_feat)  # create new container for img query feature

            for sample_idx in range(batch_size):
                if self.virtual_depth and 'pcd_scale_factor' in img_metas[sample_idx]:
                    camera_info[sample_idx, 16] = img_metas[sample_idx]['pcd_scale_factor']
                bincount = torch.bincount(img_query_view[sample_idx], minlength=self.num_views)
                view_mask = bincount > 1
                max_len = torch.max(bincount)
                sample_query_feats = torch.zeros([self.num_views, self.hidden_channel, max_len]).type_as(camera_info)
                samples_normal_query_pos = torch.zeros([self.num_views, max_len, 2]).type_as(camera_info)
                sample_reference_points = torch.zeros([self.num_views, max_len, level_num, 2]).type_as(camera_info)
                sample_padding_mask = torch.zeros([self.num_views, max_len], dtype=torch.bool, device=camera_info.device)
                for view_idx in range(self.num_views):
                    on_the_image = img_query_view[sample_idx] == view_idx  # [num_on_the_image, ]
                    if self.virtual_depth and 'img_scale_ratios' in img_metas[sample_idx]:
                        camera_info[sample_idx, 17, on_the_image] = img_metas[sample_idx]['img_scale_ratios'][view_idx]

                    view_count = bincount[view_idx]
                    if torch.sum(on_the_image) <= 1:
                        continue

                    sample_query_feats[view_idx, :, :view_count] = img_prev_query_feat[sample_idx, :, on_the_image]
                    samples_normal_query_pos[view_idx, :view_count] = normal_img_query_pos[sample_idx, on_the_image]
                    sample_reference_points[view_idx, :view_count] = reference_points[sample_idx, on_the_image]
                    sample_padding_mask[view_idx, view_count:] = True

                if input_padding_mask is None:
                    sample_query_feats[view_mask] = self.decoder[i](
                        sample_query_feats[view_mask], img_feats_stack[sample_idx, view_mask], samples_normal_query_pos[view_mask],
                        normal_img_feats_pos_stack.repeat(view_mask.sum(), 1, 1), reference_points=sample_reference_points[view_mask],
                        level_start_index=level_start_index, spatial_shapes=spatial_shapes,
                        query_padding_mask=sample_padding_mask[view_mask]
                    )
                else:
                    sample_query_feats[view_mask] = self.decoder[i](
                        sample_query_feats[view_mask], img_feats_stack[sample_idx, view_mask], samples_normal_query_pos[view_mask],
                        normal_img_feats_pos_stack.repeat(view_mask.sum(), 1, 1), reference_points=sample_reference_points[view_mask],
                        level_start_index=level_start_index, spatial_shapes=spatial_shapes,
                        query_padding_mask=sample_padding_mask[view_mask], input_padding_mask=input_padding_mask[sample_idx,view_mask]
                    )

                for view_idx in range(self.num_views):
                    on_the_image = img_query_view[sample_idx] == view_idx  # [num_on_the_image, ]
                    if torch.sum(on_the_image) <= 1:
                        continue
                    view_count = bincount[view_idx]
                    img_query_feat[sample_idx, :, on_the_image] = sample_query_feats[view_idx, :, :view_count]

            # for sample_idx in range(batch_size):
            #     for view_idx in range(self.num_views):
            #         on_the_image = img_query_view[sample_idx] == view_idx  # [num_on_the_image, ]
            #         if torch.sum(on_the_image) <= 1:
            #             continue
            #         img_query_feat_view = img_prev_query_feat[sample_idx, :, on_the_image]  # [C, num_on_the_image]
            #
            #         img_query_feat_view = self.decoder[i](
            #             img_query_feat_view[None], img_feats_stack[sample_idx:sample_idx + 1, view_idx],
            #             normal_img_query_pos[sample_idx:sample_idx + 1, on_the_image], normal_img_feats_pos_stack,
            #             reference_points=reference_points[sample_idx:sample_idx+1, on_the_image],
            #             level_start_index=level_start_index, spatial_shapes=spatial_shapes
            #         )
            #         img_query_feat[sample_idx, :, on_the_image] = img_query_feat_view.clone()

            if self.use_camera is not None and self.use_camera == 'se_sep':
                img_query_feat = self.camera_net(img_query_feat, camera_info[:, 12:16].clone())

            if self.use_camera is not None and self.use_camera == 'disentangle':
                img_query_feat_wint = self.camera_net_int(img_query_feat, camera_info[:, 12:16].clone())
                res_layer = self.prediction_heads(img_query_feat_wint)
            else:
                res_layer = self.prediction_heads(img_query_feat)
            if self.virtual_depth and 'img_scale_ratios' in img_metas[0]:
                for sample_idx in range(batch_size):
                    for view_idx in range(self.num_views):
                        ratio = img_metas[sample_idx]['img_scale_ratios'][view_idx]
                        view_mask = img_query_view[sample_idx] == view_idx
                        res_layer['depth_2d'][sample_idx, :, view_mask] = res_layer['depth_2d'][sample_idx, :, view_mask] * ratio

            if 'center_img' in res_layer:
                res_layer['center_img'] = res_layer['center_img'] + normal_img_query_pos.permute(0, 2, 1)
                res_layer['center_img'] = res_layer['center_img'].sigmoid()
                res_layer['dim_img'] = res_layer['dim_img'].sigmoid()

            res_layer['center_2d'] = res_layer['center_2d'] + normal_img_query_pos.permute(0, 2, 1)
            normal_img_query_pos = res_layer['center_2d'].detach().clone().permute(0, 2, 1)

            res_layer['center_2d'] = res_layer['center_2d'].sigmoid()

            if batch_size > 1 or i == self.num_decoder_layers-1: # only when training
                center_2d = res_layer['center_2d'].clone().permute(0, 2, 1)  # [bs, num_proposals, 2]
                depth = res_layer['depth_2d'].clone().permute(0, 2, 1)[..., :1]  # [bs, num_proposals, 1]
                h, w = img_metas[0]['input_shape'][:2]
                center_pos = denormalize_pos(center_2d, w, h, sigmoid=False)  # [bs, num_proposals, 2]
                center_pos = center_pos * depth
                camera_coords = torch.cat([center_pos, depth], dim=2)  # [bs, num_proposals, 3]
                loc_cam_3d = torch.matmul(torch.inverse(cam_intrinsic[:, :, :3, :3]), camera_coords.unsqueeze(-1)).squeeze(-1)  # [bs, num_proposals, 3]

                res_layer['loc_cam_3d'] = loc_cam_3d.permute(0, 2, 1)

            ret_dicts.append(res_layer)

        if self.use_camera is not None and self.use_camera != 'se_sep':
            img_query_feat = self.camera_net(img_query_feat, camera_info.clone())

        loc_cam_3d = copy.deepcopy(ret_dicts[-1]['loc_cam_3d'].detach()).permute(0, 2, 1)[..., None]

        if self.virtual_depth and 'pcd_scale_factor' in img_metas[0]:
            for sample_idx in range(batch_size):
                loc_cam_3d[sample_idx] = loc_cam_3d[sample_idx] * img_metas[sample_idx]['pcd_scale_factor']

        lidar2cam_r = camera_info[:, :9, :].permute(0, 2, 1)
        lidar2cam_r = lidar2cam_r.reshape(batch_size, num_img_proposals, 3, 3)

        lidar2cam_t = camera_info[:, 9:12, :].permute(0, 2, 1)[..., None] 
        bev_coords = torch.matmul(torch.inverse(lidar2cam_r), loc_cam_3d - lidar2cam_t)
        bev_coords = bev_coords.squeeze(-1)

        bev_coords[..., 0:1] = (bev_coords[..., 0:1] - self.test_cfg['pc_range'][0]) / (
                    self.test_cfg['pc_range'][3] - self.test_cfg['pc_range'][0])
        bev_coords[..., 1:2] = (bev_coords[..., 1:2] - self.test_cfg['pc_range'][1]) / (
                    self.test_cfg['pc_range'][4] - self.test_cfg['pc_range'][1])

        bev_coords[..., 0:1] = bev_coords[..., 0:1] * (self.test_cfg['grid_size'][0] // self.test_cfg['out_size_factor'])
        bev_coords[..., 1:2] = bev_coords[..., 1:2] * (self.test_cfg['grid_size'][1] // self.test_cfg['out_size_factor'])

        # if self.bbox_pos:
        #     dims, rots, vels = self.transform_bbox(ret_dicts[-1], camera_info)
        #     bev_coords = torch.cat([bev_coords, rots, vels, dims], dim=2)
        dims, rots, vels = self.transform_bbox(ret_dicts[-1], camera_info, w, img_metas)
        bev_coords = torch.cat([bev_coords, rots, vels, dims], dim=2)

        return img_query_feat, normal_img_query_pos, bev_coords, camera_info, ret_dicts

    def transform_bbox(self, ret_dict, camera_info, width, img_metas):
        bs = camera_info.shape[0]
        num_proposal = camera_info.shape[2]

        lidar2cam_rs = camera_info[:, :9]
        lidar2cam_rs = lidar2cam_rs.reshape(bs, 3, 3, num_proposal)
        lidar2cam_rs = lidar2cam_rs.permute(0, 3, 1, 2)  # [bs, num_proposals, 3, 3]
        cam2lidar_rs = torch.inverse(lidar2cam_rs)

        cam_dims = ret_dict['dim_2d'].detach().clone()  # [bs, 3, num_proposals]
        cam_rots = ret_dict['rot_2d'].detach().clone()  # [bs, 2, num_proposals]
        cam_vels = ret_dict['vel_2d'].detach().clone()  # [bs, 2, num_proposals]

        if self.virtual_depth and 'pcd_scale_factor' in img_metas[0]:
            for sample_id in range(bs):
                cam_dims[sample_id] = cam_dims[sample_id] + np.log(img_metas[sample_id]['pcd_scale_factor'])
                cam_vels[sample_id] = cam_vels[sample_id] * img_metas[sample_id]['pcd_scale_factor']

        dims = cam_dims[:, [2, 0, 1]]
        dims = dims.permute(0, 2, 1)

        if self.allocentric:
            cam_centers = ret_dict['center_2d'].detach().clone()
            cam_angle = torch.atan2(cam_rots[:, 0], cam_rots[:, 1])
            cam_angle = cam_angle + torch.atan2(cam_centers[:, 0] * width - camera_info[:, 14], camera_info[:, 12])
            sin_rots = -torch.sin(cam_angle[:, None])
            cos_rots = torch.cos(cam_angle[:, None])

        else:
            sin_rots = -cam_rots[:, 0:1]
            cos_rots = cam_rots[:, 1:2]
        rot_dirs = torch.cat([cos_rots, torch.zeros_like(sin_rots), sin_rots], dim=1)  # [bs, 3, num_proposals]
        rot_dirs = rot_dirs.permute(0, 2, 1).unsqueeze(-1)  # [bs, num_proposals, 3, 1]
        rot_dirs = torch.matmul(cam2lidar_rs, rot_dirs)  # [bs, num_proposals, 3, 1]
        lidar_rots = -rot_dirs[:, :, [0, 1], 0]  # [bs, num_proposals, 2]

        cam_vels_x = cam_vels[:, 0:1, :]
        cam_vels_z = cam_vels[:, 1:2, :]
        vels = torch.cat([cam_vels_x, torch.zeros_like(cam_vels_x), cam_vels_z], dim=1)  # [bs, 3, num_proposals]
        vels = vels.permute(0, 2, 1).unsqueeze(-1)  # [bs, num_proposals, 3, 1]
        vels = torch.matmul(cam2lidar_rs, vels)  # [bs, num_proposals, 3, 1]
        lidar_vels = vels[:, :, [0, 1], 0]

        return dims, lidar_rots, lidar_vels
    #
    # def camera2lidar(self, camera_coords, lidar2img, img_meta, batch_size):
    #     # img_pos: [W*H, 2]
    #
    #     coords = torch.cat([camera_coords, torch.ones_like(camera_coords[..., :1])], dim=1)  # [N, 4]
    #
    #     img2lidars = torch.inverse(lidar2img)
    #     coords3d = torch.matmul(img2lidars, coords.unsqueeze(-1)).squeeze(-1)[..., :3]  # [N, 3]
    #
    #     if batch_size > 1:
    #         coords3d = apply_3d_transformation(coords3d, 'LIDAR', img_meta, reverse=False).detach()
    #
    #     coords3d[..., 0:1] = (coords3d[..., 0:1] - self.test_cfg['pc_range'][0]) / (
    #                 self.test_cfg['pc_range'][3] - self.test_cfg['pc_range'][0])
    #     coords3d[..., 1:2] = (coords3d[..., 1:2] - self.test_cfg['pc_range'][1]) / (
    #                 self.test_cfg['pc_range'][4] - self.test_cfg['pc_range'][1])
    #
    #     coords3d[..., 0:1] = coords3d[..., 0:1] * (self.test_cfg['grid_size'][0] // self.test_cfg['out_size_factor'])
    #     coords3d[..., 1:2] = coords3d[..., 1:2] * (self.test_cfg['grid_size'][1] // self.test_cfg['out_size_factor'])
    #
    #     coords3d = coords3d[..., :3]  # [N, 2]
    #     coords3d = coords3d.contiguous().view(coords3d.size(0), 3)
    #
    #     return coords3d


class ViewTransformer(nn.Module):
    def __init__(self, hidden_channel, num_heads, prediction_heads, ffn_channel, dropout, activation, test_cfg,
                 query_pos, key_pos, view_projection, bbox_pos, use_camera, pos_early, virtual_depth):
        super(ViewTransformer, self).__init__()
        self.hidden_channel = hidden_channel
        self.num_heads = num_heads
        self.prediction_heads = prediction_heads
        self.test_cfg = test_cfg
        self.grid_x_size = test_cfg['grid_size'][0] // test_cfg['out_size_factor']
        self.grid_y_size = test_cfg['grid_size'][1] // test_cfg['out_size_factor']
        self.view_projection = view_projection
        self.bbox_pos = bbox_pos
        self.use_camera = use_camera
        self.pos_early = pos_early
        self.virtual_depth = virtual_depth

        if pos_early:
            self.decoder = TransformerDecoderLayer(
                hidden_channel, num_heads, ffn_channel, activation=activation, dropout=dropout,
                self_posembed=None, cross_posembed=None,
                cross_only=True
            )
            self.query_pos = query_pos
        else:
            self.decoder = TransformerDecoderLayer(
                hidden_channel, num_heads, ffn_channel, activation=activation, dropout=dropout,
                self_posembed=query_pos, cross_posembed=key_pos,
                cross_only=True
            )

        if self.use_camera == 'se_sep':
            self.camera_net = CameraSE(12, self.hidden_channel)

        # self.bbox_encoder = nn.Sequential(
        #     nn.Linear(7, hidden_channel),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_channel, hidden_channel),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_channel, hidden_channel),
        # )

    def forward(self, img_query_feat, img_query_pos_bev, normal_img_query_pos, img_ret_dicts, camera_info):
        bs = img_query_feat.shape[0]
        num_proposals = img_query_feat.shape[-1]

        camera_info = camera_info.permute(0, 2, 1)  # [bs, num_proposal, 16]
        center_3d = img_ret_dicts[-1]['loc_cam_3d'].detach().clone().permute(0, 2, 1)  # [bs, num_proposal, 3]
        center_3d = center_3d[:, -num_proposals:]

        img_query_feat = self.view_projection(img_query_feat)

        camera_R = camera_info[:, :, :9].reshape(bs, num_proposals, 3, 3)
        camera_t = camera_info[:, :, 9:12].reshape(bs, num_proposals, 3, 1)

        camera_t = -torch.matmul(camera_R.permute(0, 1, 3, 2), camera_t).squeeze(-1)

        camera_t[..., 0:1] = (camera_t[..., 0:1] - self.test_cfg['pc_range'][0]) / (
                        self.test_cfg['pc_range'][3] - self.test_cfg['pc_range'][0])
        camera_t[..., 1:2] = (camera_t[..., 1:2] - self.test_cfg['pc_range'][1]) / (
                        self.test_cfg['pc_range'][4] - self.test_cfg['pc_range'][1])
        camera_t[..., 0:1] = camera_t[..., 0:1] * (self.test_cfg['grid_size'][0] // self.test_cfg['out_size_factor'])
        camera_t[..., 1:2] = camera_t[..., 1:2] * (self.test_cfg['grid_size'][0] // self.test_cfg['out_size_factor'])

        if self.bbox_pos:
            # bbox_pos = img_query_pos_bev[..., 2:]
            # bbox_feat = self.bbox_encoder(bbox_pos).permute(0, 2, 1)
            #
            # img_query_feat = img_query_feat + bbox_feat
            # img_query_pos = torch.cat([img_query_pos_bev[..., :3], center_3d, camera_t], dim=2)

            if self.virtual_depth:
                img_query_pos = copy.deepcopy(img_query_pos_bev)
            else:
                img_query_pos = copy.deepcopy(img_query_pos_bev[..., :7])
            img_query_pos[..., :2] = inverse_sigmoid((img_query_pos[..., :2] + 12) / 204)
            img_query_pos[..., 2] = inverse_sigmoid((img_query_pos[..., 2] + 10) / 20)
            img_query_pos[..., 3:5] = inverse_sigmoid((img_query_pos[..., 3:5] + 1) / 2)

            img_query_pos = torch.cat([img_query_pos, normal_img_query_pos], dim=2)

        else:
            img_query_pos = torch.cat([camera_t, center_3d, img_query_pos_bev[..., :3]], dim=2)

            # normal_img_query_pos_bev = copy.deepcopy(img_query_pos_bev)
            # normal_img_query_pos_bev[..., :2] = (normal_img_query_pos_bev[..., :2] + 12) / 204
            # normal_img_query_pos_bev[..., 2] = (normal_img_query_pos_bev[..., 2] + 10) / 20
            #
            # img_query_pos = inverse_sigmoid(normal_img_query_pos_bev)

        if self.use_camera == 'se_sep':
            img_query_feat = self.camera_net(img_query_feat, camera_info.permute(0, 2, 1)[:, :12])

        if self.pos_early:
            img_query_feat = img_query_feat + self.query_pos(img_query_pos)
        img_query_feat = self.decoder(img_query_feat, img_query_feat, img_query_pos, img_query_pos)

        # Prediction
        res_layer = self.prediction_heads(img_query_feat)

        res_layer['center_mono'] = img_query_pos_bev[..., 0:2].permute(0, 2, 1)
        res_layer['height_mono'] = img_query_pos_bev[..., 2:3].permute(0, 2, 1)
        res_layer['rot_mono'] = img_query_pos_bev[..., 3:5].permute(0, 2, 1)
        res_layer['vel_mono'] = img_query_pos_bev[..., 5:7].permute(0, 2, 1)
        res_layer['dim_mono'] = img_query_pos_bev[..., 7:10].permute(0, 2, 1)

        res_layer['center_view'] = res_layer['center_view'] + img_query_pos_bev[..., 0:2].permute(0, 2, 1)

        img_query_pos_bev = res_layer['center_view'].detach().clone().permute(0, 2, 1)

        return img_query_feat, img_query_pos_bev, [res_layer]

class Gate(nn.Module):
    def __init__(self, hidden_channel):
        super(Gate, self).__init__()

        self.hidden_channel = hidden_channel
        self.mlp = nn.Sequential(
            nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channel, hidden_channel*2, kernel_size=1),
        )

    def forward(self, img_feat, box_feat):
        z = self.mlp(img_feat).sigmoid()
        z1 = z[:, :self.hidden_channel]
        z2 = z[:, self.hidden_channel:]

        img_feat = img_feat * z1 + box_feat * z2

        return img_feat


class ViewTransformerPoint(nn.Module):
    def __init__(self, hidden_channel, num_heads, prediction_heads, ffn_channel, dropout, activation, test_cfg,
                 query_pos, key_pos, view_projection, bbox_pos, bev_projection, use_gate=True):
        super(ViewTransformerPoint, self).__init__()
        self.hidden_channel = hidden_channel
        self.num_heads = num_heads
        self.prediction_heads = prediction_heads
        self.test_cfg = test_cfg
        self.grid_x_size = test_cfg['grid_size'][0] // test_cfg['out_size_factor']
        self.grid_y_size = test_cfg['grid_size'][1] // test_cfg['out_size_factor']
        self.view_projection = view_projection
        self.bev_projection = bev_projection
        self.bbox_pos = bbox_pos

        self.decoder = TransformerDecoderLayer(
            hidden_channel, num_heads, ffn_channel, activation=activation, dropout=dropout,
            self_posembed=query_pos, cross_posembed=key_pos,
        )

        self.use_gate = use_gate
        if use_gate:
            self.gate = Gate(hidden_channel)

        if bbox_pos:
            self.bbox_encoder = nn.Sequential(
                nn.Linear(7, hidden_channel),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channel, hidden_channel),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channel, hidden_channel),
            )

    def forward(self, img_query_feat, img_query_pos_bev, lidar_feat_flatten, bev_pos, img_ret_dicts, camera_info):
        bs = img_query_feat.shape[0]
        num_proposals = img_query_feat.shape[-1]

        camera_info = camera_info.permute(0, 2, 1)  # [bs, num_proposal, 16]
        center_3d = img_ret_dicts[-1]['loc_cam_3d'].detach().clone().permute(0, 2, 1)  # [bs, num_proposal, 3]
        center_3d = center_3d[:, -num_proposals:]

        # center_3d[..., 0:1] = (center_3d[..., 0:1] - self.test_cfg['pc_range'][0]) / (
        #                 self.test_cfg['pc_range'][3] - self.test_cfg['pc_range'][0])
        # center_3d[..., 1:2] = (center_3d[..., 1:2] - self.test_cfg['pc_range'][1]) / (
        #                 self.test_cfg['pc_range'][4] - self.test_cfg['pc_range'][1])
        # center_3d[..., 0:1] = center_3d[..., 0:1] * (self.test_cfg['grid_size'][0] // self.test_cfg['out_size_factor'])
        # center_3d[..., 1:2] = center_3d[..., 1:2] * (self.test_cfg['grid_size'][0] // self.test_cfg['out_size_factor'])

        img_query_feat = self.view_projection(img_query_feat)
        if self.bev_projection is not None:
            lidar_feat_flatten = lidar_feat_flatten.view(bs, self.hidden_channel, self.grid_y_size, self.grid_x_size)
            lidar_feat_flatten = self.bev_projection(lidar_feat_flatten)
            lidar_feat_flatten = lidar_feat_flatten.view(bs, self.hidden_channel, -1)

        camera_R = camera_info[:, :, :9].reshape(bs, num_proposals, 3, 3)
        camera_t = camera_info[:, :, 9:12].reshape(bs, num_proposals, 3, 1)

        camera_t = -torch.matmul(camera_R.permute(0, 1, 3, 2), camera_t).squeeze(-1)

        camera_t[..., 0:1] = (camera_t[..., 0:1] - self.test_cfg['pc_range'][0]) / (
                        self.test_cfg['pc_range'][3] - self.test_cfg['pc_range'][0])
        camera_t[..., 1:2] = (camera_t[..., 1:2] - self.test_cfg['pc_range'][1]) / (
                        self.test_cfg['pc_range'][4] - self.test_cfg['pc_range'][1])
        camera_t[..., 0:1] = camera_t[..., 0:1] * (self.test_cfg['grid_size'][0] // self.test_cfg['out_size_factor'])
        camera_t[..., 1:2] = camera_t[..., 1:2] * (self.test_cfg['grid_size'][0] // self.test_cfg['out_size_factor'])

        if self.bbox_pos:
            # bbox_pos = img_query_pos_bev[..., 2:].permute(0, 2, 1)
            # bbox_feat = self.bbox_encoder(bbox_pos)
            bbox_pos = img_query_pos_bev[..., 2:]
            bbox_feat = self.bbox_encoder(bbox_pos).permute(0, 2, 1)

            if self.use_gate:
                img_query_feat = self.gate(img_query_feat, bbox_feat)
            else:
                img_query_feat = img_query_feat + bbox_feat
            # img_query_pos = torch.cat([img_query_pos_bev[..., :3], center_3d, camera_t], dim=2)
            img_query_pos = img_query_pos_bev[..., :2]

        else:
            img_query_pos = torch.cat([camera_t, center_3d, img_query_pos_bev], dim=2)
            # img_query_pos = self.bn(img_query_pos.permute(0, 2, 1)).permute(0, 2, 1)

        img_query_feat = self.decoder(img_query_feat, lidar_feat_flatten, img_query_pos, bev_pos)

        # Prediction
        res_layer = self.prediction_heads(img_query_feat)

        res_layer['center_view'] = res_layer['center_view'] + img_query_pos_bev[..., 0:2].permute(0, 2, 1)
        img_query_pos_bev = res_layer['center_view'].detach().clone().permute(0, 2, 1)

        return img_query_feat, img_query_pos_bev, [res_layer]


class ViewTransformerFFN(nn.Module):
    def __init__(self, hidden_channel, num_heads, prediction_heads, ffn_channel, dropout, activation, test_cfg,
                 query_pos, view_projection):
        super(ViewTransformerFFN, self).__init__()
        self.hidden_channel = hidden_channel
        self.num_heads = num_heads
        self.prediction_heads = prediction_heads
        self.test_cfg = test_cfg
        self.grid_x_size = test_cfg['grid_size'][0] // test_cfg['out_size_factor']
        self.grid_y_size = test_cfg['grid_size'][1] // test_cfg['out_size_factor']
        self.view_projection = view_projection
        self.query_pos = query_pos

        self.decoder = nn.Sequential(
            nn.Linear(hidden_channel, hidden_channel),
            nn.LayerNorm(hidden_channel),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, hidden_channel)
        )
        self.layernorm = nn.LayerNorm(hidden_channel)

        self.dropout = nn.Dropout(dropout)

    def forward(self, img_query_feat, img_query_pos_bev, img_ret_dicts, camera_info):
        bs = img_query_feat.shape[0]
        num_proposals = img_query_feat.shape[-1]

        img_query_pos = copy.deepcopy(img_query_pos_bev[..., :3])
        img_query_pos[..., 2] = (img_query_pos[..., 2] + 5) * 10

        img_query_feat_raw = img_query_feat + self.query_pos(img_query_pos)
        img_query_feat_raw = self.view_projection(img_query_feat_raw)
        img_query_feat_raw = img_query_feat_raw.permute(0, 2, 1)
        img_query_feat = self.decoder(img_query_feat_raw)
        # img_query_feat = img_query_feat + self.dropout(img_query_feat_raw)
        img_query_feat = self.layernorm(img_query_feat).permute(0, 2, 1)

        # Prediction
        res_layer = self.prediction_heads(img_query_feat)

        # res_layer['center_view'] = res_layer['center_view'] + img_query_pos_bev[..., 0:2].permute(0, 2, 1)
        # img_query_pos_bev = res_layer['center_view'].detach().clone().permute(0, 2, 1)

        res_layer['center_view'] = img_query_pos_bev[..., 0:2].permute(0, 2, 1)
        # res_layer['heatmap_view'] = img_ret_dicts[-1]['cls'].detach().clone()
        # res_layer['dim_view'] = img_ret_dicts[-1]['dim_2d'][:, [2, 0, 1]].detach().clone()

        return img_query_feat, img_query_pos_bev, [res_layer]


class ViewAdder(nn.Module):
    def __init__(self, hidden_channel, prediction_heads, ffn_channel, dropout, test_cfg, view_projection):
        super(ViewAdder, self).__init__()
        self.hidden_channel = hidden_channel
        self.test_cfg = test_cfg
        self.grid_x_size = test_cfg['grid_size'][0] // test_cfg['out_size_factor']
        self.grid_y_size = test_cfg['grid_size'][1] // test_cfg['out_size_factor']
        self.view_projection = view_projection
        self.prediction_heads = prediction_heads

        self.decoder = nn.Sequential(
            nn.Linear(hidden_channel, ffn_channel),
            nn.ReLU(inplace=True),
            nn.Linear(ffn_channel, hidden_channel)
        )
        self.layernorm = nn.LayerNorm(hidden_channel)

        self.pos_embd = nn.Sequential(
            nn.Linear(7, hidden_channel),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, hidden_channel)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, img_query_feat, img_query_pos_bev):

        img_query_pos = copy.deepcopy(img_query_pos_bev)
        img_query_pos[..., :2] = inverse_sigmoid((img_query_pos[..., :2] + 12) / 204)
        img_query_pos[..., 2] = inverse_sigmoid((img_query_pos[..., 2] + 10) / 20)
        img_query_pos[..., 3:5] = inverse_sigmoid((img_query_pos[..., 3:5] + 1) / 2)

        img_query_feat = self.view_projection(img_query_feat)
        img_query_feat_raw = img_query_feat + self.pos_embd(img_query_pos).permute(0, 2, 1)

        img_query_feat_raw = img_query_feat_raw.permute(0, 2, 1)
        img_query_feat = self.decoder(img_query_feat_raw)
        img_query_feat = img_query_feat + self.dropout(img_query_feat_raw)
        img_query_feat = self.layernorm(img_query_feat).permute(0, 2, 1)

        # Prediction
        res_layer = self.prediction_heads(img_query_feat)

        # res_layer['center_view'] = res_layer['center_view'] + img_query_pos_bev[..., 0:2].permute(0, 2, 1)
        # img_query_pos_bev = res_layer['center_view'].detach().clone().permute(0, 2, 1)

        res_layer['center_view'] = img_query_pos_bev[..., 0:2].permute(0, 2, 1)
        # res_layer['heatmap_view'] = img_ret_dicts[-1]['cls'].detach().clone()
        # res_layer['dim_view'] = img_ret_dicts[-1]['dim_2d'][:, [2, 0, 1]].detach().clone()


        return img_query_feat, [res_layer]