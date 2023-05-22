import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from mmdet3d.models.fusion_layers import apply_3d_transformation

from mmdet3d.models.utils import TransformerDecoderLayer, inverse_sigmoid
from mmdet3d.models.utils.deformable_decoder import DeformableTransformerDecoderLayer
from mmdet3d.models.utils.network_modules import LayerNorm, denormalize_pos, normalize_pos


class PointTransformer2D_3D(nn.Module):
    def __init__(self, hidden_channel, num_heads, num_decoder_layers, prediction_heads, ffn_channel, dropout, activation, test_cfg, query_pos, key_pos):
        super(PointTransformer2D_3D, self).__init__()
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
                    self_posembed=query_pos[i],
                    cross_posembed=key_pos[i],
                )
            )

    def forward(self, pts_query_feat, pts_query_pos, lidar_feat_flatten, bev_pos):
        ret_dicts = []
        res_layer = self.prediction_heads(pts_query_feat)
        res_layer['center'] = pts_query_pos.permute(0, 2, 1)  # [BS, 2, num_proposals]

        for i in range(self.num_decoder_layers):
            # Transformer Decoder Layer
            # :param query: B C Pq    :param query_pos: B Pq 3/6
            pts_query_feat = self.decoder[i](pts_query_feat, lidar_feat_flatten, pts_query_pos, bev_pos)

            # Prediction
            res_layer = self.prediction_heads(pts_query_feat)
            res_layer['center'] = res_layer['center'] + pts_query_pos.permute(0, 2, 1)

            ret_dicts.append(res_layer)
            # for next level positional embedding
            pts_query_pos = res_layer['center'].detach().clone().permute(0, 2, 1)

        return pts_query_feat, pts_query_pos, ret_dicts


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


class ImageTransformer_Cam_3D_MS(nn.Module):
    def __init__(self, num_views, hidden_channel, num_heads, num_decoder_layers, prediction_heads, out_size_factor_img,
                 ffn_channel, dropout, activation, test_cfg, query_pos, key_pos):
        super(ImageTransformer_Cam_3D_MS, self).__init__()
        self.hidden_channel = hidden_channel
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.prediction_heads = prediction_heads
        self.num_views = num_views
        self.out_size_factor_img = out_size_factor_img
        self.test_cfg = test_cfg
        # self.use_camera = use_camera

        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                DeformableTransformerDecoderLayer(
                    hidden_channel, num_heads, dim_feedforward=ffn_channel, dropout=dropout, activation=activation,
                    self_posembed=query_pos[i], cross_posembed=key_pos[i],
                )
            )

        camera_dim = 16

        # if use_camera == 'se':
        #     self.camera_net = CameraSE(camera_dim, hidden_channel)

    def forward(self, img_query_feat, normal_img_query_pos, img_query_view, img_feats, normal_img_feats_pos_stack, lidar2cam_rt, cam_intrinsic, img_metas, input_padding_mask=None):
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
                bincount = torch.bincount(img_query_view[sample_idx], minlength=self.num_views)
                view_mask = bincount > 1
                max_len = torch.max(bincount)
                sample_query_feats = torch.zeros([self.num_views, self.hidden_channel, max_len]).type_as(camera_info)
                samples_normal_query_pos = torch.zeros([self.num_views, max_len, 2]).type_as(camera_info)
                sample_reference_points = torch.zeros([self.num_views, max_len, level_num, 2]).type_as(camera_info)
                sample_padding_mask = torch.zeros([self.num_views, max_len], dtype=torch.bool, device=camera_info.device)
                for view_idx in range(self.num_views):
                    on_the_image = img_query_view[sample_idx] == view_idx  # [num_on_the_image, ]
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

            res_layer = self.prediction_heads(img_query_feat)

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

        # img_query_feat = self.camera_net(img_query_feat, camera_info.clone())

        loc_cam_3d = copy.deepcopy(ret_dicts[-1]['loc_cam_3d'].detach()).permute(0, 2, 1)[..., None]

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

        dims = cam_dims[:, [2, 0, 1]]
        dims = dims.permute(0, 2, 1)

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


class ViewTransformer(nn.Module):
    def __init__(self, hidden_channel, num_heads, prediction_heads, ffn_channel, dropout, activation, test_cfg,
                 query_pos, key_pos, view_projection, use_camera):
        super(ViewTransformer, self).__init__()
        self.hidden_channel = hidden_channel
        self.num_heads = num_heads
        self.prediction_heads = prediction_heads
        self.test_cfg = test_cfg
        self.grid_x_size = test_cfg['grid_size'][0] // test_cfg['out_size_factor']
        self.grid_y_size = test_cfg['grid_size'][1] // test_cfg['out_size_factor']
        self.view_projection = view_projection
        self.use_camera = use_camera

        if use_camera is not None:
            assert use_camera == "se"
            self.camera_net = CameraSE(16, hidden_channel)

        self.decoder = TransformerDecoderLayer(
            hidden_channel, num_heads, ffn_channel, activation=activation, dropout=dropout,
            self_posembed=query_pos, cross_posembed=key_pos,
            cross_only=True
        )


    def forward(self, img_query_feat, img_query_pos_bev, normal_img_query_pos, img_ret_dicts, camera_info):
        bs = img_query_feat.shape[0]
        num_proposals = img_query_feat.shape[-1]

        center_3d = img_ret_dicts[-1]['loc_cam_3d'].detach().clone().permute(0, 2, 1)  # [bs, num_proposal, 3]
        center_3d = center_3d[:, -num_proposals:]

        if self.use_camera is not None:
            img_query_feat = self.camera_net(img_query_feat, camera_info)

        camera_info = camera_info.permute(0, 2, 1)  # [bs, num_proposal, 16]

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

        img_query_pos = copy.deepcopy(img_query_pos_bev[..., :7])
        img_query_pos[..., :2] = inverse_sigmoid((img_query_pos[..., :2] + 12) / 204)
        img_query_pos[..., 2] = inverse_sigmoid((img_query_pos[..., 2] + 10) / 20)
        img_query_pos[..., 3:5] = inverse_sigmoid((img_query_pos[..., 3:5] + 1) / 2)

        img_query_pos = torch.cat([img_query_pos, normal_img_query_pos], dim=2)

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


class FusionTransformer2D_3D_Self(nn.Module):
    def __init__(self, hidden_channel, num_heads, num_decoder_layers, prediction_heads, ffn_channel, dropout, activation, test_cfg,
                 query_pos, key_pos, pts_projection, img_projection, num_proposals):
        super(FusionTransformer2D_3D_Self, self).__init__()
        self.hidden_channel = hidden_channel
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.prediction_heads = prediction_heads
        self.test_cfg = test_cfg
        self.grid_x_size = test_cfg['grid_size'][0] // test_cfg['out_size_factor']
        self.grid_y_size = test_cfg['grid_size'][1] // test_cfg['out_size_factor']
        self.pts_projection = pts_projection
        self.img_projection = img_projection
        self.num_proposals = num_proposals

        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    hidden_channel, num_heads, ffn_channel, dropout, activation,
                    self_posembed=query_pos[i], cross_posembed=key_pos[i], cross_only=True
                )
            )

    def forward(self, pts_query_feat, pts_query_pos, img_query_feat, img_query_pos, need_weights=False):
        ret_dicts = []
        pts_query_feat = self.pts_projection(pts_query_feat)
        img_query_feat = self.img_projection(img_query_feat)

        all_query_feat = torch.cat([pts_query_feat, img_query_feat], dim=2)
        all_query_pos = torch.cat([pts_query_pos, img_query_pos], dim=1)

        for i in range(self.num_decoder_layers):
            # Transformer Decoder Layer
            # :param query: B C Pq    :param query_pos: B Pq 3/6
            all_query_feat_raw = all_query_feat.clone()

            if need_weights:
                all_query_feat, attn_weights = self.decoder[i](all_query_feat, all_query_feat, all_query_pos, all_query_pos, need_weights=True)
            else:
                all_query_feat = self.decoder[i](all_query_feat, all_query_feat, all_query_pos, all_query_pos)

            all_query_feat_pred = all_query_feat

            # Prediction
            res_layer = self.prediction_heads(all_query_feat_pred)
            res_layer['center'] = res_layer['center'] + all_query_pos.permute(0, 2, 1)

            ret_dicts.append(res_layer)

            all_query_pos = res_layer['center'].detach().clone().permute(0, 2, 1)

        # return all_query_feat, all_query_pos, ret_dicts
        if need_weights:
            return all_query_feat, all_query_pos, ret_dicts, attn_weights
        else:
            return all_query_feat, all_query_pos, ret_dicts


class ImageTransformer2D_3D_MS(nn.Module):
    def __init__(self, num_views, hidden_channel, num_heads, num_decoder_layers, prediction_heads, out_size_factor_img,
                 ffn_channel, dropout, activation, test_cfg, query_pos, key_pos, supervision2d):
        super(ImageTransformer2D_3D_MS, self).__init__()
        self.hidden_channel = hidden_channel
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.prediction_heads = prediction_heads
        self.num_views = num_views
        self.out_size_factor_img = out_size_factor_img
        self.test_cfg = test_cfg
        self.supervision2d = supervision2d

        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                DeformableTransformerDecoderLayer(
                    hidden_channel, num_heads, dim_feedforward=ffn_channel, dropout=dropout, activation=activation,
                    self_posembed=query_pos[i], cross_posembed=key_pos[i],
                )
            )

    def forward(self, img_query_feat, normal_img_query_pos, img_query_view, img_feats, normal_img_feats_pos_stack, img_metas):
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
        ret_dicts = []

        for i in range(self.num_decoder_layers):
            img_prev_query_feat = img_query_feat.clone()  # [BS, C, num_proposals]
            img_query_feat = torch.zeros_like(img_query_feat)  # create new container for img query feature
            for sample_idx in range(batch_size):
                for view_idx in range(self.num_views):
                    on_the_image = img_query_view[sample_idx] == view_idx  # [num_on_the_image, ]
                    if torch.sum(on_the_image) <= 1:
                        continue
                    img_query_feat_view = img_prev_query_feat[sample_idx, :, on_the_image]  # [C, num_on_the_image]

                    img_query_feat_view = self.decoder[i](
                        img_query_feat_view[None], img_feats_stack[sample_idx:sample_idx + 1, view_idx],
                        normal_img_query_pos[sample_idx:sample_idx + 1, on_the_image], normal_img_feats_pos_stack,
                        reference_points=reference_points[sample_idx:sample_idx+1, on_the_image],
                        level_start_index=level_start_index, spatial_shapes=spatial_shapes
                    )
                    img_query_feat[sample_idx, :, on_the_image] = img_query_feat_view.clone()

            res_layer = self.prediction_heads(img_query_feat)
            if 'center_offset' in res_layer:
                assert 'center_2d' not in res_layer and 'offset' not in res_layer
                res_layer['center_2d'] = res_layer['center_offset'][:, :2]
                res_layer['offset'] = res_layer['center_offset'][:, 2:]

            res_layer['center_2d'] = res_layer['center_2d'] + normal_img_query_pos.permute(0, 2, 1)

            if self.supervision2d:
                normal_img_query_pos = res_layer['center_2d'].detach().clone().permute(0, 2, 1)

            res_layer['center_2d'] = res_layer['center_2d'].sigmoid()
            res_layer['offset'] = res_layer['offset'].sigmoid()

            bbox_width = res_layer['offset'][:, 0] + res_layer['offset'][:, 2]
            bbox_height = res_layer['offset'][:, 1] + res_layer['offset'][:, 3]

            bbox_cx = (res_layer['center_2d'][:, 0] - res_layer['offset'][:, 0] + res_layer['center_2d'][:, 0] + res_layer['offset'][:, 2]) / 2
            bbox_cy = (res_layer['center_2d'][:, 1] - res_layer['offset'][:, 1] + res_layer['center_2d'][:, 1] + res_layer['offset'][:, 3]) / 2

            res_layer['bbox_2d'] = torch.stack([bbox_cx, bbox_cy, bbox_width, bbox_height], dim=1).detach().clone()

            ret_dicts.append(res_layer)

        return img_query_feat, normal_img_query_pos, ret_dicts

    def camera2lidar(self, camera_coords, lidar2img, img_meta, batch_size):
        # img_pos: [W*H, 2]

        coords = torch.cat([camera_coords, torch.ones_like(camera_coords[..., :1])], dim=1)  # [N, 4]

        img2lidars = torch.inverse(lidar2img)
        coords3d = torch.matmul(img2lidars, coords.unsqueeze(-1)).squeeze(-1)[..., :3]  # [N, 3]

        if batch_size > 1:
            coords3d = apply_3d_transformation(coords3d, 'LIDAR', img_meta, reverse=False).detach()
        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.test_cfg['pc_range'][0]) / (
                    self.test_cfg['pc_range'][3] - self.test_cfg['pc_range'][0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.test_cfg['pc_range'][1]) / (
                    self.test_cfg['pc_range'][4] - self.test_cfg['pc_range'][1])

        coords3d[..., 0:1] = coords3d[..., 0:1] * (self.test_cfg['grid_size'][0] // self.test_cfg['out_size_factor'])
        coords3d[..., 1:2] = coords3d[..., 1:2] * (self.test_cfg['grid_size'][1] // self.test_cfg['out_size_factor'])

        if not self.pos_3d:
            coords3d = coords3d[..., :2]  # [N, 3]

        if self.pos_3d:
            coords3d = coords3d.contiguous().view(coords3d.size(0), 3)
        else:
            coords3d = coords3d.contiguous().view(coords3d.size(0), 2)

        return coords3d