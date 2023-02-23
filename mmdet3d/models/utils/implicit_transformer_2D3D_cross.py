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
from mmdet3d.models.utils.network_modules import LayerNorm


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

class ImageTransformer2D_3D_Cross(nn.Module):
    def __init__(self, num_views, hidden_channel, num_heads, num_decoder_layers, prediction_heads, img_smca,
                 out_size_factor_img, ffn_channel, dropout, activation, test_cfg, query_pos, key_pos, supervision2d):
        super(ImageTransformer2D_3D_Cross, self).__init__()
        self.hidden_channel = hidden_channel
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.prediction_heads = prediction_heads
        self.img_smca = img_smca
        self.num_views = num_views
        self.out_size_factor_img = out_size_factor_img
        self.test_cfg = test_cfg

        self.grid_x_size = test_cfg['grid_size'][0] // test_cfg['out_size_factor']
        self.grid_y_size = test_cfg['grid_size'][1] // test_cfg['out_size_factor']
        self.supervision2d = supervision2d

        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    hidden_channel, num_heads, ffn_channel, dropout, activation,
                    self_posembed=query_pos, cross_posembed=key_pos,
                )
            )

    def forward(self, img_query_feat, normal_img_query_pos, img_query_view, img_feat_flatten, normal_img_feat_pos, img_metas):
        batch_size = img_query_feat.shape[0]
        ret_dicts = []
        res_layer = self.prediction_heads(img_query_feat)
        res_layer['bbox'][:, :2] = normal_img_query_pos.permute(0, 2, 1)

        img_scale = img_metas[0]['pad_shape'][0]
        img_y_size, img_x_size = img_scale
        img_x_size = img_x_size / self.out_size_factor_img
        img_y_size = img_y_size / self.out_size_factor_img
        denormal_img_feat_pos = denormalize_pos(normal_img_feat_pos, img_x_size, img_y_size)  # [1, h*w, 2]

        for i in range(self.num_decoder_layers):
            centers = denormalize_pos(normal_img_query_pos, img_x_size, img_y_size)  # [bs, num_proposals, 2]
            dims = denormalize_pos(res_layer['bbox'][:, 2:4].permute(0, 2, 1), img_x_size, img_y_size)  # [bs, num_proposals, 2]
            img_prev_query_feat = img_query_feat.clone()  # [BS, C, num_proposals]
            img_query_feat = torch.zeros_like(img_query_feat)  # create new container for img query feature
            for sample_idx in range(batch_size):
                for view_idx in range(self.num_views):
                    on_the_image = img_query_view[sample_idx] == view_idx  # [num_on_the_image, ]
                    if torch.sum(on_the_image) <= 1:
                        continue
                    img_query_feat_view = img_prev_query_feat[sample_idx, :, on_the_image]  # [C, num_on_the_image]
                    if self.img_smca:
                        centers_view = centers[sample_idx, on_the_image]  # [num_on_the_image, 2]
                        dims_view = dims[sample_idx, on_the_image]  # [num_on_the_image, 2]
                        radius = torch.ceil(dims_view.norm(dim=-1, p=2) / 2).int()  # [num_on_the_image, ]
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

        return img_query_feat, normal_img_query_pos, ret_dicts


class CameraNetAdd(nn.Module):
    def __init__(self, cam_dim, hidden_channel, depth_stop_grad=False):
        super(CameraNetAdd, self).__init__()
        self.depth_stop_grad = depth_stop_grad
        self.bn = nn.BatchNorm1d(cam_dim)

        if not self.depth_stop_grad:
            self.reduce_conv_context = nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1)
            self.mlp_context = nn.Sequential(
                nn.Conv1d(cam_dim, hidden_channel, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1),
            )

        self.reduce_conv_depth = nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1)
        self.mlp_depth = nn.Sequential(
            nn.Conv1d(cam_dim, hidden_channel, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1),
        )

    def forward(self, feat, cam_info):
        cam_info = self.bn(cam_info)
        if self.depth_stop_grad:
            context = feat.clone()
        else:
            feat_context = self.reduce_conv_context(feat)
            context = feat_context + self.mlp_context(cam_info)
        feat_pred = self.reduce_conv_depth(feat)
        pred = feat_pred + self.mlp_depth(cam_info)
        return context, pred

class CameraNetBEVDepth(nn.Module):
    def __init__(self, cam_dim, hidden_channel, depth_stop_grad=False):
        super(CameraNetBEVDepth, self).__init__()
        self.bn = nn.BatchNorm1d(cam_dim)
        self.depth_stop_grad = depth_stop_grad

        if not depth_stop_grad:
            self.reduce_conv_context = nn.Sequential(
                nn.Linear(hidden_channel, hidden_channel),
                nn.LayerNorm(hidden_channel),
                nn.ReLU(inplace=True),
            )
            self.mlp_context = nn.Sequential(
                nn.Conv1d(cam_dim, hidden_channel, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1),
            )

        self.reduce_conv_depth = nn.Sequential(
            nn.Linear(hidden_channel, hidden_channel),
            nn.LayerNorm(hidden_channel),
            nn.ReLU(inplace=True),
        )
        self.mlp_depth = nn.Sequential(
            nn.Conv1d(cam_dim, hidden_channel, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1),
        )

    def forward(self, feat, cam_info):
        cam_info = self.bn(cam_info)
        if self.depth_stop_grad:
            context = feat.clone()
        else:
            feat_context = self.reduce_conv_context(feat.transpose(2, 1)).transpose(2, 1)
            context = feat_context * self.mlp_context(cam_info).sigmoid()
        feat_pred = self.reduce_conv_depth(feat.transpose(2, 1)).transpose(2, 1)
        pred = feat_pred * self.mlp_depth(cam_info).sigmoid()
        return context, pred


class CameraNet(nn.Module):
    def __init__(self, cam_dim, hidden_channel, depth_stop_grad=False, extra_camera=False):
        super(CameraNet, self).__init__()
        self.bn = nn.BatchNorm1d(cam_dim)

        # self.reduce_conv_context = nn.Sequential(
        #     nn.Linear(hidden_channel, hidden_channel),
        #     nn.LayerNorm(hidden_channel)
        # )
        self.hidden_channel = hidden_channel
        self.extra_camera = extra_camera
        if not depth_stop_grad:
            if extra_camera:
                self.mlp_context = nn.Sequential(
                    nn.Conv1d(cam_dim, hidden_channel*2, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(hidden_channel*2, hidden_channel*2, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(hidden_channel*2, hidden_channel*2, kernel_size=1),
                )
            else:
                self.mlp_context = nn.Sequential(
                    nn.Conv1d(cam_dim, hidden_channel, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1),
                )

        # self.reduce_conv_depth = nn.Sequential(
        #     nn.Linear(hidden_channel, hidden_channel),
        #     nn.LayerNorm(hidden_channel)
        # )

        self.mlp_depth = nn.Sequential(
            nn.Conv1d(cam_dim, hidden_channel, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1),
        )
        self.depth_stop_grad = depth_stop_grad

    def forward(self, feat, cam_info):
        cam_info = self.bn(cam_info)
        # feat_context = self.reduce_conv_context(feat.transpose(2, 1)).transpose(2, 1)
        extra_mul = None
        if self.depth_stop_grad:
            context = feat.clone()
        else:
            if self.extra_camera:
                mul = self.mlp_context(cam_info).sigmoid()
                context = feat * mul[:, :self.hidden_channel]
                extra_mul = mul[:, self.hidden_channel:]
            else:
                context = feat * self.mlp_context(cam_info).sigmoid()
        # feat_pred = self.reduce_conv_depth(feat.transpose(2, 1)).transpose(2, 1)
        pred = feat * self.mlp_depth(cam_info).sigmoid()
        return context, pred, extra_mul


class DepthTransformer2D_3D(nn.Module):
    def __init__(self, hidden_channel, num_heads, prediction_heads, ffn_channel, dropout, activation, test_cfg,
                 query_pos, key_pos, pts_projection, img_projection, dbound=None, use_camera=None, cross_smca=False,
                 extra_camera=False, pos_3d=False, depth_stop_grad=False, depth_key_3d=False):
        super(DepthTransformer2D_3D, self).__init__()
        self.hidden_channel = hidden_channel
        self.num_heads = num_heads
        self.prediction_heads = prediction_heads
        self.test_cfg = test_cfg
        self.grid_x_size = test_cfg['grid_size'][0] // test_cfg['out_size_factor']
        self.grid_y_size = test_cfg['grid_size'][1] // test_cfg['out_size_factor']
        self.use_camera = use_camera
        self.cross_smca = cross_smca
        self.pos_3d = pos_3d
        self.depth_key_3d = depth_key_3d

        self.depth_decoder = TransformerDecoderLayer(
            hidden_channel, num_heads, ffn_channel, dropout, activation,
            self_posembed=query_pos, cross_posembed=key_pos
            # cross_only=False
        )
 
        self.pts_projection = pts_projection
        self.img_projection = img_projection
        self.dbound = dbound
        self.depth_stop_grad = depth_stop_grad

        if self.use_camera is not None:
            if self.use_camera == 'se':
                self.camera_net = CameraNet(16, hidden_channel, depth_stop_grad=depth_stop_grad, extra_camera=extra_camera)
            elif self.use_camera == 'add':
                self.camera_net = CameraNetAdd(16, hidden_channel, depth_stop_grad=depth_stop_grad)
            elif self.use_camera == 'bevdepth':
                self.camera_net = CameraNetBEVDepth(16, hidden_channel, depth_stop_grad=depth_stop_grad)
            else:
                raise NotImplementedError

    def forward(self, pts_query_feat, pts_query_pos, pts_query_height, img_query_feat, normal_img_query_pos, img_query_view,
                normal_img_query_bboxes, img_metas, img_feat_flatten, normal_img_feat_pos):

        pts_query_feat = self.pts_projection(pts_query_feat)
        img_query_feat = self.img_projection(img_query_feat)
        num_img_proposals = img_query_feat.shape[-1]
        batch_size = pts_query_feat.shape[0]

        query_pos_realmetric = pts_query_pos.permute(0, 2, 1) * self.test_cfg['out_size_factor'] * \
                               self.test_cfg['voxel_size'][0] + self.test_cfg['pc_range'][0]
        query_pos_3d = torch.cat([query_pos_realmetric, pts_query_height], dim=1).detach().clone()  # [BS, 3, num_proposals]
        new_img_query_feat = torch.zeros_like(img_query_feat)  # create new container for img query feature
        update_mask = torch.zeros([batch_size, num_img_proposals]).to(query_pos_3d.device)
        camera_info = torch.zeros([batch_size, 16, num_img_proposals]).to(query_pos_3d.device)

        normal_img_query_size = normal_img_query_bboxes[..., 2:] / 2 # [BS, num_proposals, 2]
        # normal_img_query_radius = normal_img_query_size.norm(dim=-1, p=2) / 2  # [BS, num_proposals]

        for sample_idx in range(batch_size):
            sample_img_query_feat = img_query_feat[sample_idx]  # [C, num_proposals]
            sample_normal_img_query_pos = normal_img_query_pos[sample_idx]  # [num_proposals, 2]
            sample_img_query_view = img_query_view[sample_idx]  # [num_proposals]
            sample_query_pos_3d = query_pos_3d[sample_idx]  # [3, num_proposals]
            sample_normal_img_query_size = normal_img_query_size[sample_idx]  # [num_proposals, 2]

            lidar2img_rt = sample_query_pos_3d.new_tensor(img_metas[sample_idx]['lidar2img'])
            img_scale_factor = (
                sample_query_pos_3d.new_tensor(img_metas[sample_idx]['scale_factor'][:2]
                                        if 'scale_factor' in img_metas[sample_idx].keys() else [1.0, 1.0])
            )

            img_pad_shape = img_metas[sample_idx]['input_shape'][:2]
            # transform point clouds back to original coordinate system by reverting the data augmentation
            if batch_size == 1:  # skip during inference to save time
                points = sample_query_pos_3d.T
            else:
                points = apply_3d_transformation(sample_query_pos_3d.T, 'LIDAR', img_metas[sample_idx], reverse=True).detach()  # [num_proposals, 3]
            num_points = points.shape[0]
            num_views = len(lidar2img_rt)
            for view_idx in range(num_views):
                pts_4d = torch.cat([points, points.new_ones(size=(num_points, 1))], dim=-1)  # [N, 4]
                pts_2d = pts_4d @ lidar2img_rt[view_idx].t()

                pts_2d[:, 2] = torch.clamp(pts_2d[:, 2], min=1e-5)
                pts_2d[:, 0] /= pts_2d[:, 2]
                pts_2d[:, 1] /= pts_2d[:, 2]

                # img transformation: scale -> crop -> flip
                # the image is resized by img_scale_factor
                img_coors = pts_2d[:, 0:2] * img_scale_factor  # Nx2

                # grid sample, the valid grid range should be in [-1,1]
                coor_x, coor_y = torch.split(img_coors, 1, dim=1)  # each is Nx1

                h, w = img_pad_shape
                valid_shape = img_metas[sample_idx]['valid_shape'][view_idx] if 'valid_shape' in img_metas[sample_idx].keys() else [w, h]
                red_w = int(valid_shape[0])
                red_h = int(valid_shape[1])

                on_the_image = (coor_x > 0) * (coor_x < red_w) * (coor_y > 0) * (coor_y < red_h)  # [N, 1]
                on_the_image = on_the_image.squeeze()  # [N, ]
                # skip the following computation if no object query fall on current image
                if on_the_image.sum() <= 1:
                    continue

                # add spatial constraint
                center_ys = coor_y[on_the_image]
                center_xs = coor_x[on_the_image]

                pts_query_feat_view = pts_query_feat[sample_idx, :, on_the_image]  # [C, N]
                pts_query_pos_view = torch.cat([center_xs, center_ys], dim=-1)  # [N, 2]
                normal_pts_query_pos_view = normalize_pos(pts_query_pos_view[None], w, h)[0]  # [N, 2]
                pts_query_pos_depth_view = pts_2d[on_the_image, 2:3]

                img_query_view_mask = sample_img_query_view == view_idx

                lidar2cam_r = pts_4d.new_tensor(img_metas[sample_idx]['lidar2cam_r'][view_idx])
                lidar2cam_t = pts_4d.new_tensor(img_metas[sample_idx]['lidar2cam_t'][view_idx])
                cam_intrinsic = pts_4d.new_tensor(img_metas[sample_idx]['cam_intrinsic'][view_idx])

                camera_info[sample_idx, :9, img_query_view_mask] = lidar2cam_r.reshape(9, 1)
                camera_info[sample_idx, 9:12, img_query_view_mask] = lidar2cam_t.reshape(3, 1)
                camera_info[sample_idx, 12, img_query_view_mask] = cam_intrinsic[0, 0]
                camera_info[sample_idx, 13, img_query_view_mask] = cam_intrinsic[1, 1]
                camera_info[sample_idx, 14:16, img_query_view_mask] = cam_intrinsic[:2, 2].reshape(2, 1)

                if img_query_view_mask.sum() <= 1:
                    continue
                update_mask[sample_idx, img_query_view_mask] = 1
                img_query_feat_view = sample_img_query_feat[:, img_query_view_mask]  # [C, M]
                normal_img_query_pos_view = sample_normal_img_query_pos[img_query_view_mask]  # [M, 2]

                if self.cross_smca:
                    img_query_size_view = sample_normal_img_query_size[img_query_view_mask]  # [M, 2]
                    distance = normal_img_query_pos_view[:, None, :].sigmoid() - normal_pts_query_pos_view[None, :, :].sigmoid()  # [M, N, 2]
                    sigma = (img_query_size_view * 2 + 1e-4) / 6  # [M, 2]
                    gaussian_mask = (-(distance[..., 0] ** 2) / (2 * sigma[:, None, 0] ** 2) + (-distance[..., 1] ** 2) / (2 * sigma[:, None, 1] ** 2)).exp()
                    # gaussian_mask = (-distance / (2 * sigma[:, None] ** 2)).exp()  # [M, N]
                    gaussian_mask[gaussian_mask < torch.finfo(torch.float32).eps] = 0
                    attn_mask = gaussian_mask

                    attn_mask_log = attn_mask.log()
                    inf_mask = torch.max(attn_mask_log, dim=1)[0].isinf()  # [M, ]
                    attn_mask_log[inf_mask] = 0
                    if self.depth_key_3d:
                        normal_pts_query_pos_view = torch.cat([normal_pts_query_pos_view, pts_query_pos_depth_view.log()], dim=-1)
                    new_img_query_feat_view = self.depth_decoder(
                        img_query_feat_view[None], pts_query_feat_view[None], normal_img_query_pos_view[None], normal_pts_query_pos_view[None],
                        attn_mask=attn_mask_log
                    )
                else:
                    if self.depth_key_3d:
                        normal_pts_query_pos_view = torch.cat([normal_pts_query_pos_view, pts_query_pos_depth_view.log()], dim=-1)
                    new_img_query_feat_view = self.depth_decoder(
                        img_query_feat_view[None], pts_query_feat_view[None], normal_img_query_pos_view[None], normal_pts_query_pos_view[None])

                new_img_query_feat[sample_idx, :, img_query_view_mask] = new_img_query_feat_view.clone()

        hidden_dim = new_img_query_feat.shape[1]
        new_img_query_feat[update_mask.unsqueeze(1).repeat(1, hidden_dim, 1)==0] = img_query_feat[update_mask.unsqueeze(1).repeat(1, hidden_dim, 1)==0].detach().clone()

        if self.use_camera:
            new_img_query_feat, img_query_feat_for_pred, extra_mul = self.camera_net(new_img_query_feat, camera_info)
        else:
            img_query_feat_for_pred = new_img_query_feat
            extra_mul = None
        depth_prediction = self.prediction_heads(img_query_feat_for_pred)
        depth_prediction['proj_center'] = depth_prediction['proj_center'] + normal_img_query_pos.permute(0, 2, 1)  # [BS, 2, num_proposals]

        if self.dbound is None:
            depth = depth_prediction['depth'].transpose(2, 1)  # (BS, num_proposal, 1)
            depth = torch.exp(depth)
        else:
            depth = depth_prediction['depth']  # (BS, num_depth_class, num_proposal)
            depth = torch.max(depth, dim=1, keepdim=True)[1].transpose(2, 1).float()  # [BS, num_proposal, 1]
            depth = depth * self.dbound[2] + self.dbound[0]  # [BS, num_proposal, 1]

        center = depth_prediction['proj_center'].detach().clone().transpose(2, 1)  # [BS, num_proposal, 2]
        depth_prediction['proj_center'] = depth_prediction['proj_center'].sigmoid()

        if self.pos_3d:
            bev_coords = torch.zeros([batch_size, num_img_proposals, 3]).to(center.device)
        else:
            bev_coords = torch.zeros([batch_size, num_img_proposals, 2]).to(center.device)

        for sample_idx in range(batch_size):
            img_pad_shape = img_metas[sample_idx]['input_shape'][:2]
            h, w = img_pad_shape
            img_scale_factor = (
                center.new_tensor(img_metas[sample_idx]['scale_factor'][:2]
                                        if 'scale_factor' in img_metas[sample_idx].keys() else [1.0, 1.0])
            )
            w = w / img_scale_factor[0]
            h = h / img_scale_factor[1]

            center_sample = denormalize_pos(center[sample_idx:sample_idx+1], w, h)[0]  # [num_proposal, 2]
            depth_sample = depth[sample_idx]   # [num_proposal, 1]

            center_sample = center_sample * depth_sample
            camera_coords = torch.cat([center_sample, depth_sample], dim=1)  #  [num_proposal, 3]
            lidar2img_rt = query_pos_3d.new_tensor(img_metas[sample_idx]['lidar2img'])
            num_views = len(lidar2img_rt)
            view_sample = img_query_view[sample_idx]   # [num_proposal, ]

            for view_idx in range(num_views):
                lidar2img = lidar2img_rt[view_idx]
                view_mask = view_sample == view_idx
                camera_coords_view = camera_coords[view_mask]
                bev_coords_view = self.camera2lidar(camera_coords_view, lidar2img, img_metas[sample_idx], batch_size)
                bev_coords[sample_idx, view_mask] = bev_coords_view.clone()

        if self.depth_stop_grad:
            new_img_query_feat = new_img_query_feat.detach()

        return new_img_query_feat, bev_coords, [depth_prediction], extra_mul

    def camera2lidar(self, camera_coords, lidar2img, img_meta, batch_size):
        # img_pos: [W*H, 2]

        coords = torch.cat([camera_coords, torch.ones_like(camera_coords[..., :1])], dim=1)  # [N, 4]

        img2lidars = torch.inverse(lidar2img)
        coords3d = torch.matmul(img2lidars, coords.unsqueeze(-1)).squeeze(-1)[..., :3]  # [N, 3]

        if batch_size > 1:
            coords3d = apply_3d_transformation(coords3d, 'LIDAR', img_meta, reverse=False).detach()
        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.test_cfg['pc_range'][0]) / (self.test_cfg['pc_range'][3] - self.test_cfg['pc_range'][0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.test_cfg['pc_range'][1]) / (self.test_cfg['pc_range'][4] - self.test_cfg['pc_range'][1])

        coords3d[..., 0:1] = coords3d[..., 0:1] * (self.test_cfg['grid_size'][0] // self.test_cfg['out_size_factor'])
        coords3d[..., 1:2] = coords3d[..., 1:2] * (self.test_cfg['grid_size'][1] // self.test_cfg['out_size_factor'])

        if not self.pos_3d:
            coords3d = coords3d[..., :2]  # [N, 3]

        if self.pos_3d:
            coords3d = coords3d.contiguous().view(coords3d.size(0), 3)
        else:
            coords3d = coords3d.contiguous().view(coords3d.size(0), 2)

        return coords3d

class SimpleGate(nn.Module):
    def __init__(self, hidden_channel):
        super(SimpleGate, self).__init__()
        self.gate1 = nn.Sequential(
            nn.Conv1d(hidden_channel*2, hidden_channel, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1),
            nn.Sigmoid()
        )

        self.gate2 = nn.Sequential(
            nn.Conv1d(hidden_channel*2, hidden_channel, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, raw_feat, feat):
        cat_feat = torch.cat([raw_feat, feat], dim=1)
        weight1 = self.gate1(cat_feat)
        weight2 = self.gate2(cat_feat)

        new_feat = raw_feat * weight1 + feat * weight2
        return new_feat


class GRUGate(nn.Module):
    def __init__(self, hidden_channel):
        super(GRUGate, self).__init__()
        self.reset_gate = nn.Sequential(
            nn.Conv1d(hidden_channel*2, hidden_channel, kernel_size=1),
            nn.Sigmoid()
        )

        self.update_gate = nn.Sequential(
            nn.Conv1d(hidden_channel*2, hidden_channel, kernel_size=1),
            nn.Sigmoid()
        )

        self.proj = nn.Sequential(
            nn.Linear(hidden_channel*2, hidden_channel, kernel_size=1),
            nn.LayerNorm(hidden_channel)
        )

    def forward(self, raw_feat, feat):
        cat_feat = torch.cat([raw_feat, feat], dim=1)
        reset_weight = self.reset_gate(cat_feat)
        update_weight = self.update_gate(cat_feat)

        raw_feat_2 = raw_feat * reset_weight
        cat_feat_2 = torch.cat([raw_feat_2, feat], dim=1).permute(0, 2, 1)
        proj_feat = self.proj(cat_feat_2).permute(0, 2, 1)

        new_feat = proj_feat * update_weight + raw_feat * (1 - update_weight)
        return new_feat


class FusionTransformer2D_3D_Cross(nn.Module):
    def __init__(self, hidden_channel, num_heads, num_decoder_layers, prediction_heads, ffn_channel, dropout, activation, test_cfg,
                 query_pos, key_pos, pts_projection, img_projection, fusion_gate, fuse_cat):
        super(FusionTransformer2D_3D_Cross, self).__init__()
        self.hidden_channel = hidden_channel
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.prediction_heads = prediction_heads
        self.test_cfg = test_cfg
        self.grid_x_size = test_cfg['grid_size'][0] // test_cfg['out_size_factor']
        self.grid_y_size = test_cfg['grid_size'][1] // test_cfg['out_size_factor']
        self.pts_projection = pts_projection
        self.img_projection = img_projection
        self.fusion_gate = fusion_gate
        self.fuse_cat = fuse_cat

        self.decoder = nn.ModuleList()
        if self.fusion_gate:
            self.gates = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    hidden_channel, num_heads, ffn_channel, dropout, activation,
                    self_posembed=query_pos[i], cross_posembed=key_pos[i],
                    cross_only=False
                )
            )
            if self.fusion_gate:
                self.gates.append(SimpleGate(hidden_channel))

    def forward(self, pts_query_feat, pts_query_pos, img_query_feat, img_query_pos):
        ret_dicts = []
        pts_query_feat = self.pts_projection(pts_query_feat)
        img_query_feat = self.img_projection(img_query_feat)
        for i in range(self.num_decoder_layers):
            raw_pts_query_feat = pts_query_feat.clone()

            # Transformer Decoder Layer
            # :param query: B C Pq    :param query_pos: B Pq 3/6
            pts_query_feat = self.decoder[i](pts_query_feat, img_query_feat, pts_query_pos, img_query_pos)

            # pts_query_feat = torch.cat([raw_pts_query_feat, pts_query_feat], dim=1)
            # Prediction

            if self.fusion_gate:
                pts_query_feat = self.gates[i](raw_pts_query_feat, pts_query_feat)

            if self.fuse_cat:
                pts_query_feat_pred = torch.cat([pts_query_feat, raw_pts_query_feat], dim=1)
            else:
                pts_query_feat_pred = pts_query_feat

            res_layer = self.prediction_heads(pts_query_feat_pred)

            res_layer['center'] = res_layer['center'] + pts_query_pos.permute(0, 2, 1)

            ret_dicts.append(res_layer)

            pts_query_pos = res_layer['center'].detach().clone().permute(0, 2, 1)

        return pts_query_feat, pts_query_pos, ret_dicts


class ImageTransformer2D_3D_Cross_Proj(nn.Module):
    def __init__(self, num_views, hidden_channel, num_heads, num_decoder_layers, prediction_heads, out_size_factor_img,
                 ffn_channel, dropout, activation, test_cfg, query_pos, key_pos, supervision2d):
        super(ImageTransformer2D_3D_Cross_Proj, self).__init__()
        self.hidden_channel = hidden_channel
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.prediction_heads = prediction_heads
        self.num_views = num_views
        self.out_size_factor_img = out_size_factor_img
        self.test_cfg = test_cfg

        self.grid_x_size = test_cfg['grid_size'][0] // test_cfg['out_size_factor']
        self.grid_y_size = test_cfg['grid_size'][1] // test_cfg['out_size_factor']
        self.supervision2d = supervision2d

        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    hidden_channel, num_heads, ffn_channel, dropout, activation,
                    self_posembed=query_pos[i], cross_posembed=key_pos[i],
                )
            )

    def forward(self, img_query_feat, normal_img_query_pos, img_query_view, img_feat_flatten, normal_img_feat_pos, img_metas):
        batch_size = img_query_feat.shape[0]
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
                        img_query_feat_view[None], img_feat_flatten[sample_idx:sample_idx + 1, view_idx],
                        normal_img_query_pos[sample_idx:sample_idx + 1, on_the_image], normal_img_feat_pos
                    )
                    img_query_feat[sample_idx, :, on_the_image] = img_query_feat_view.clone()

            res_layer = self.prediction_heads(img_query_feat)
            if 'center_offset' in res_layer:
                assert 'center_2d' not in res_layer and 'offset' not in res_layer
                res_layer['center_2d'] = res_layer['center_offset'][:, :2]
                res_layer['offset'] = res_layer['center_offset'][:, 2:]

            res_layer['center_2d'] = res_layer['center_2d'] + normal_img_query_pos.permute(0, 2, 1)
            # res_layer['bbox'][:, :2] = res_layer['bbox'][:, :2] + normal_img_query_pos.permute(0, 2, 1)

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


class FusionCWA_early(nn.Module):
    def __init__(self, hidden_channel):
        super(FusionCWA_early, self).__init__()

        self.mlp = nn.Sequential(
            nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1)
        )

        self.proj = nn.Sequential(
            nn.Linear(hidden_channel, hidden_channel),
            nn.LayerNorm(hidden_channel)
        )

    def forward(self, fusion_feat):
        mul = self.mlp(fusion_feat).sigmoid()
        fusion_feat = fusion_feat * mul

        fusion_feat = self.proj(fusion_feat.transpose(1, 2)).transpose(1, 2)
        return fusion_feat


class FusionCWA(nn.Module):
    def __init__(self, hidden_channel):
        super(FusionCWA, self).__init__()

        self.mlp = nn.Sequential(
            nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1)
        )

        self.pos_mlp = nn.Sequential(
            nn.Conv1d(2, hidden_channel, kernel_size=1),
            nn.BatchNorm1d(hidden_channel),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1)
        )

    def forward(self, raw_feat, fusion_feat, pos_mod1, pos_mod2):
        dist_center = torch.norm(pos_mod1-90, dim=-1, p=2) # [bs, N1]
        pos_mod1_exp = pos_mod1.unsqueeze(dim=1)  # [bs, 1, N1, 2]
        pos_mod2_exp = pos_mod2.unsqueeze(dim=2)  # [bs, N2, 1, 2]
        dist = torch.norm(pos_mod1_exp-pos_mod2_exp, dim=-1, p=2)  # [bs, N2, N1]
        dist = torch.min(dist, dim=1)[0]  # [bs, N1]
        dist = torch.stack([dist, dist_center], dim=1)  # [bs, 2, N1]
        dist_embed = self.pos_mlp(dist)
        mul = self.mlp(raw_feat+dist_embed).sigmoid()

        return fusion_feat * mul


class FusionTransformer2D_3D_Self(nn.Module):
    def __init__(self, hidden_channel, num_heads, num_decoder_layers, prediction_heads, ffn_channel, dropout, activation, test_cfg,
                 query_pos, key_pos, pts_projection, img_projection, fusion_cwa, num_proposals, lidar_query_only, fuse_cat):
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
        self.fusion_cwa = fusion_cwa
        self.num_proposals = num_proposals
        self.lidar_query_only = lidar_query_only
        self.fuse_cat = fuse_cat
        if self.fusion_cwa:
            # self.lidar_cwa = FusionCWA(hidden_channel)
            # self.camera_cwa = FusionCWA(hidden_channel)
            self.fusion_cwa = FusionCWA_early(hidden_channel)

        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    hidden_channel, num_heads, ffn_channel, dropout, activation,
                    self_posembed=query_pos[i], cross_posembed=key_pos[i], cross_only=True
                )
            )

    def forward(self, pts_query_feat, pts_query_pos, img_query_feat, img_query_pos):
        ret_dicts = []
        pts_query_feat = self.pts_projection(pts_query_feat)
        img_query_feat = self.img_projection(img_query_feat)

        all_query_feat = torch.cat([pts_query_feat, img_query_feat], dim=2)
        all_query_pos = torch.cat([pts_query_pos, img_query_pos], dim=1)

        if self.fusion_cwa:
            all_query_feat = self.fusion_cwa(all_query_feat)

        for i in range(self.num_decoder_layers):
            # Transformer Decoder Layer
            # :param query: B C Pq    :param query_pos: B Pq 3/6
            all_query_feat_raw = all_query_feat.clone()

            all_query_feat = self.decoder[i](all_query_feat, all_query_feat, all_query_pos, all_query_pos)

            # if self.fusion_cwa:
            #     all_query_feat_lidar = self.lidar_cwa(all_query_feat_raw[..., :self.num_proposals], all_query_feat[..., :self.num_proposals], all_query_pos[:, :self.num_proposals], all_query_pos[:, self.num_proposals:])
            #     all_query_feat_camera = self.camera_cwa(all_query_feat_raw[..., self.num_proposals:], all_query_feat[..., self.num_proposals:], all_query_pos[:, self.num_proposals:], all_query_pos[:, :self.num_proposals])
            #     all_query_feat = torch.cat([all_query_feat_lidar, all_query_feat_camera], dim=2)

            if self.fuse_cat:
                all_query_feat_pred = torch.cat([all_query_feat, all_query_feat_raw], dim=1)
            else:
                all_query_feat_pred = all_query_feat

            # Prediction
            if self.lidar_query_only:
                res_layer = self.prediction_heads(all_query_feat_pred[..., :self.num_proposals])
                res_layer['center'] = res_layer['center'] + all_query_pos.permute(0, 2, 1)[..., :self.num_proposals]
            else:
                res_layer = self.prediction_heads(all_query_feat_pred)
                res_layer['center'] = res_layer['center'] + all_query_pos.permute(0, 2, 1)

            ret_dicts.append(res_layer)

            all_query_pos = res_layer['center'].detach().clone().permute(0, 2, 1)

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
            # res_layer['bbox'][:, :2] = res_layer['bbox'][:, :2] + normal_img_query_pos.permute(0, 2, 1)

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


class DepthEstimation(nn.Module):
    def __init__(self, hidden_channel, prediction_heads, test_cfg, dbound=None, use_camera=None, extra_camera=False, pos_3d=False):
        super(DepthEstimation, self).__init__()
        self.hidden_channel = hidden_channel
        self.prediction_heads = prediction_heads
        self.test_cfg = test_cfg
        self.use_camera = use_camera
        self.pos_3d = pos_3d

        self.dbound = dbound

        if self.use_camera is not None:
            if self.use_camera == 'se':
                self.camera_net = CameraNet(16, hidden_channel, depth_stop_grad=False,
                                            extra_camera=extra_camera)
            elif self.use_camera == 'add':
                self.camera_net = CameraNetAdd(16, hidden_channel, depth_stop_grad=False)
            elif self.use_camera == 'bevdepth':
                self.camera_net = CameraNetBEVDepth(16, hidden_channel, depth_stop_grad=False)
            else:
                raise NotImplementedError

    def forward(self, img_query_feat, normal_img_query_pos, img_query_view, img_metas):

        num_img_proposals = img_query_feat.shape[-1]
        batch_size = img_query_feat.shape[0]

        camera_info = torch.zeros([batch_size, 16, num_img_proposals]).to(img_query_feat.device)

        for sample_idx in range(batch_size):
            sample_img_query_view = img_query_view[sample_idx]  # [num_proposals]

            num_views = len(img_metas[sample_idx]['lidar2img'])
            for view_idx in range(num_views):

                img_query_view_mask = sample_img_query_view == view_idx

                lidar2cam_r = camera_info.new_tensor(img_metas[sample_idx]['lidar2cam_r'][view_idx])
                lidar2cam_t = camera_info.new_tensor(img_metas[sample_idx]['lidar2cam_t'][view_idx])
                cam_intrinsic = camera_info.new_tensor(img_metas[sample_idx]['cam_intrinsic'][view_idx])

                camera_info[sample_idx, :9, img_query_view_mask] = lidar2cam_r.reshape(9, 1)
                camera_info[sample_idx, 9:12, img_query_view_mask] = lidar2cam_t.reshape(3, 1)
                camera_info[sample_idx, 12, img_query_view_mask] = cam_intrinsic[0, 0]
                camera_info[sample_idx, 13, img_query_view_mask] = cam_intrinsic[1, 1]
                camera_info[sample_idx, 14:16, img_query_view_mask] = cam_intrinsic[:2, 2].reshape(2, 1)

        depth_prediction = self.prediction_heads(img_query_feat)

        if self.use_camera:
            new_img_query_feat, _, extra_mul = self.camera_net(img_query_feat, camera_info)
        else:
            new_img_query_feat = img_query_feat
            extra_mul = None

        if self.dbound is None:
            depth = depth_prediction['depth'].transpose(2, 1)  # (BS, num_proposal, 1)
            depth = torch.exp(depth)
        else:
            depth = depth_prediction['depth']  # (BS, num_depth_class, num_proposal)
            depth = torch.max(depth, dim=1, keepdim=True)[1].transpose(2, 1).float()  # [BS, num_proposal, 1]
            depth = depth * self.dbound[2] + self.dbound[0]  # [BS, num_proposal, 1]

        center = normal_img_query_pos.detach().clone()  # [BS, num_proposal, 2]

        if self.pos_3d:
            bev_coords = torch.zeros([batch_size, num_img_proposals, 3]).to(center.device)
        else:
            bev_coords = torch.zeros([batch_size, num_img_proposals, 2]).to(center.device)

        for sample_idx in range(batch_size):
            img_pad_shape = img_metas[sample_idx]['input_shape'][:2]
            h, w = img_pad_shape
            img_scale_factor = (
                center.new_tensor(img_metas[sample_idx]['scale_factor'][:2]
                                  if 'scale_factor' in img_metas[sample_idx].keys() else [1.0, 1.0])
            )
            w = w / img_scale_factor[0]
            h = h / img_scale_factor[1]

            center_sample = denormalize_pos(center[sample_idx:sample_idx + 1], w, h)[0]  # [num_proposal, 2]
            depth_sample = depth[sample_idx]  # [num_proposal, 1]

            center_sample = center_sample * depth_sample
            camera_coords = torch.cat([center_sample, depth_sample], dim=1)  # [num_proposal, 3]
            lidar2img_rt = camera_info.new_tensor(img_metas[sample_idx]['lidar2img'])
            num_views = len(lidar2img_rt)
            view_sample = img_query_view[sample_idx]  # [num_proposal, ]

            for view_idx in range(num_views):
                lidar2img = lidar2img_rt[view_idx]
                view_mask = view_sample == view_idx
                camera_coords_view = camera_coords[view_mask]
                bev_coords_view = self.camera2lidar(camera_coords_view, lidar2img, img_metas[sample_idx], batch_size)
                bev_coords[sample_idx, view_mask] = bev_coords_view.clone()

        return new_img_query_feat, bev_coords, [depth_prediction], extra_mul

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


class FusionTransformer2D_3D_DoubleCross(nn.Module):
    def __init__(self, hidden_channel, num_heads, num_decoder_layers, prediction_heads, ffn_channel, dropout, activation, test_cfg,
                 query_pos, key_pos, pts_projection, img_projection):
        super(FusionTransformer2D_3D_DoubleCross, self).__init__()
        self.hidden_channel = hidden_channel
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.prediction_heads = prediction_heads
        self.test_cfg = test_cfg
        self.grid_x_size = test_cfg['grid_size'][0] // test_cfg['out_size_factor']
        self.grid_y_size = test_cfg['grid_size'][1] // test_cfg['out_size_factor']
        self.pts_projection = pts_projection
        self.img_projection = img_projection

        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    hidden_channel, num_heads, ffn_channel, dropout, activation,
                    self_posembed=query_pos[i*2], cross_posembed=key_pos[i*2],
                    cross_only=False
                )
            )
            self.decoder.append(
                TransformerDecoderLayer(
                    hidden_channel, num_heads, ffn_channel, dropout, activation,
                    self_posembed=query_pos[i*2+1], cross_posembed=key_pos[i*2+1],
                    cross_only=False
                )
            )

    def forward(self, pts_query_feat, pts_query_pos, img_query_feat, img_query_pos):
        ret_dicts = []
        pts_query_feat = self.pts_projection(pts_query_feat)
        img_query_feat = self.img_projection(img_query_feat)
        for i in range(self.num_decoder_layers):

            # Transformer Decoder Layer
            # :param query: B C Pq    :param query_pos: B Pq 3/6
            img_query_feat = self.decoder[i*2](img_query_feat, pts_query_feat, img_query_pos, pts_query_pos)
            res_layer_img = self.prediction_heads(img_query_feat)
            res_layer_img['center'] = res_layer_img['center'] + img_query_pos.permute(0, 2, 1)
            img_query_pos = res_layer_img['center'].detach().clone().permute(0, 2, 1)

            pts_query_feat = self.decoder[i*2+1](pts_query_feat, img_query_feat, pts_query_pos, img_query_pos)
            res_layer_pts = self.prediction_heads(pts_query_feat)
            res_layer_pts['center'] = res_layer_pts['center'] + pts_query_pos.permute(0, 2, 1)
            pts_query_pos = res_layer_pts['center'].detach().clone().permute(0, 2, 1)

            # pts_query_feat = torch.cat([raw_pts_query_feat, pts_query_feat], dim=1)
            # Prediction

            all_query_pos = torch.cat([pts_query_pos, img_query_pos], dim=1)
            # all_query_feature = torch.cat([pts_query_feat, img_query_feat], dim=2)
            # res_layer = self.prediction_heads(all_query_feature)

            res_layer = {}
            for key in res_layer_pts:
                res_layer[key] = torch.cat([res_layer_pts[key], res_layer_img[key]], dim=2)
            # res_layer['center'] = res_layer['center'] + all_query_pos.permute(0, 2, 1)

            ret_dicts.append(res_layer)

            # pts_query_pos = res_layer['center'].detach().clone().permute(0, 2, 1)

        return pts_query_feat, pts_query_pos, ret_dicts


class FusionTransformer2D_3D_SepPos_Self(nn.Module):
    def __init__(self, hidden_channel, num_heads, num_decoder_layers, prediction_heads, ffn_channel, dropout, activation, test_cfg,
                 pts_pos, img_pos, pts_projection, img_projection, num_proposals):
        super(FusionTransformer2D_3D_SepPos_Self, self).__init__()
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

        self.pts_pos = nn.ModuleList(pts_pos)
        self.img_pos = nn.ModuleList(img_pos)

        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    hidden_channel, num_heads, ffn_channel, dropout, activation,
                    self_posembed=nn.Identity(), cross_posembed=nn.Identity(), cross_only=True
                )
            )

    def forward(self, pts_query_feat, pts_query_pos, img_query_feat, img_query_pos):
        ret_dicts = []
        pts_query_feat = self.pts_projection(pts_query_feat)
        img_query_feat = self.img_projection(img_query_feat)

        all_query_feat = torch.cat([pts_query_feat, img_query_feat], dim=2)

        for i in range(self.num_decoder_layers):
            # Transformer Decoder Layer
            # :param query: B C Pq    :param query_pos: B Pq 3/6

            pts_query_pos_embed = self.pts_pos[i](pts_query_pos)
            img_query_pos_embed = self.img_pos[i](img_query_pos)

            all_query_pos_embed = torch.cat([pts_query_pos_embed, img_query_pos_embed], dim=2)
            all_query_feat = self.decoder[i](all_query_feat, all_query_feat, all_query_pos_embed, all_query_pos_embed)

            # Prediction
            all_query_pos = torch.cat([pts_query_pos, img_query_pos], dim=1)
            res_layer = self.prediction_heads(all_query_feat)
            res_layer['center'] = res_layer['center'] + all_query_pos.permute(0, 2, 1)

            ret_dicts.append(res_layer)

            all_query_pos = res_layer['center'].detach().clone().permute(0, 2, 1)
            pts_query_pos = all_query_pos[:, :self.num_proposals]
            img_query_pos = all_query_pos[:, self.num_proposals:]

        return all_query_feat, all_query_pos, ret_dicts


class FusionIPOT(nn.Module):
    def __init__(self, hidden_channel, prediction_heads, ffn_channel, dropout, activation, test_cfg,
                 query_pos, key_pos, pts_projection, img_projection):
        super(FusionIPOT, self).__init__()
        self.hidden_channel = hidden_channel
        self.prediction_heads = prediction_heads
        self.test_cfg = test_cfg
        self.grid_x_size = test_cfg['grid_size'][0] // test_cfg['out_size_factor']
        self.grid_y_size = test_cfg['grid_size'][1] // test_cfg['out_size_factor']
        self.pts_projection = pts_projection
        self.img_projection = img_projection

        self.query_pos = nn.ModuleList(query_pos)
        self.key_pos = nn.ModuleList(key_pos)

        self.out_projection = nn.Sequential(
            nn.Linear(hidden_channel*2, hidden_channel),
            nn.LayerNorm(hidden_channel),
        )

        self.ffn = nn.Sequential(
            nn.Linear(hidden_channel, ffn_channel),
            nn.ReLU(inplace=True),
            nn.Linear(ffn_channel, hidden_channel),
        )
        self.dropout = nn.Dropout(p=dropout)
        self.ln = nn.LayerNorm(hidden_channel)

    def forward(self, pts_query_feat, pts_query_pos, img_query_feat, img_query_pos, pts_cls, img_cls):
        ret_dicts = []
        pts_query_feat = self.pts_projection(pts_query_feat)
        img_query_feat = self.img_projection(img_query_feat)

        pts_query_pos_embed = self.query_pos[0](pts_query_pos)
        img_query_pos_embed = self.key_pos[0](img_query_pos)

        pts_query_feat = pts_query_feat + pts_query_pos_embed
        img_query_feat = img_query_feat + img_query_pos_embed

        pts_cls = torch.max(torch.sigmoid(pts_cls), dim=1)[0].detach().clone()  # [bs, num_proposals]
        img_cls = torch.max(torch.sigmoid(img_cls), dim=1)[0].detach().clone()  # [bs, num_proposals]

        pts_query_pos_exp = pts_query_pos.unsqueeze(dim=2)  # [bs, num_proposals, 1, 2]
        img_query_pos_exp = img_query_pos.unsqueeze(dim=1)  # [bs, 1, num_proposals, 2]
        cost_matrix = torch.norm(pts_query_pos_exp - img_query_pos_exp, dim=-1, p=2)  # [bs, num_proposals, num_proposals]

        pts_query_feat = self.ipot(pts_query_feat, img_query_feat, pts_cls, img_cls, cost_matrix)
        pts_query_feat = self.out_projection(pts_query_feat.transpose(1, 2))
        pts_query_feat = self.ffn(pts_query_feat) + self.dropout(pts_query_feat)
        pts_query_feat = self.ln(pts_query_feat).transpose(1, 2)

        res_layer = self.prediction_heads(pts_query_feat)
        res_layer['center'] = res_layer['center'] + pts_query_pos.permute(0, 2, 1)

        ret_dicts.append(res_layer)

        pts_query_pos = res_layer['center'].detach().clone().permute(0, 2, 1)

        return pts_query_feat, pts_query_pos, ret_dicts


    def ipot(self, pts_query_feat, img_query_feat, pts_cls, img_cls, cost_matrix):
        bs = pts_query_feat.shape[0]
        num_proposals = pts_cls.shape[1]
        eps = 10
        b = torch.ones_like(pts_cls) / num_proposals
        G = torch.exp(-cost_matrix/eps)

        res = torch.ones_like(G)
        for t in range(100):
            Q = G * res
            Q = torch.clamp(Q, min=1e-6)
            a = pts_cls / (torch.matmul(Q, b.unsqueeze(-1)).squeeze(-1) + 1e-6)
            b = img_cls / (torch.matmul(Q.transpose(1, 2), a.unsqueeze(-1)).squeeze(-1) + 1e-6)

            res = torch.diag_embed(a) @ Q @ torch.diag_embed(b)

        res = F.normalize(res, dim=2, p=1)

        img_query_feat_att = torch.matmul(res, img_query_feat.transpose(1, 2)).transpose(1, 2)
        pts_query_feat = torch.cat([pts_query_feat, img_query_feat_att], dim=1)

        return pts_query_feat