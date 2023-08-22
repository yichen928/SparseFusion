import copy
import numpy as np
import torch
import functools
import pickle
import os
from mmcv.cnn import ConvModule, build_conv_layer, kaiming_init
from mmcv.runner import force_fp32
from torch import nn
import torch.nn.functional as F
import time

from mmdet3d.core import (circle_nms, draw_heatmap_gaussian, gaussian_radius,
                          xywhr2xyxyr, limit_period, PseudoSampler, BboxOverlaps3D)
from mmdet3d.models.builder import HEADS, build_loss
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu
from mmdet3d.models.utils import clip_sigmoid, inverse_sigmoid
from mmdet3d.models.fusion_layers import apply_3d_transformation
from mmdet.core import build_bbox_coder, multi_apply, build_assigner, build_sampler, AssignResult

from mmdet3d.models.utils import FFN, TransformerDecoderLayer, PositionEmbeddingLearned, PositionEmbeddingLearnedwoNorm,\
    PointTransformer2D_3D, ImageTransformer_Cam_3D_MS, ProjectionLayerNorm, FusionTransformer2D_3D_Self, \
    ViewTransformer, DepthEncoderResNet, LayerNorm, ConvLN, FFNLN, normalize_pos

from mmdet3d.models.utils.ops.modules import MSDeformAttn
from mmdet3d.models.utils.deformable_decoder import DeformableTransformerDecoderLayer


@HEADS.register_module()
class SparseFusionHead2D_Deform(nn.Module):
    def __init__(self,
                 num_views=0,
                 in_channels_img=64,
                 out_size_factor_img=4,
                 num_proposals=128,
                 num_img_proposals=128,
                 in_channels=128 * 3,
                 hidden_channel=128,
                 num_classes=4,
                 # config for Transformer
                 num_pts_decoder_layers=1,
                 num_img_decoder_layers=1,
                 num_fusion_decoder_layers=1,
                 num_heads=8,
                 initialize_by_heatmap=True,
                 semantic_transfer=True,
                 cross_only=True,
                 range_num=5,
                 cross_heatmap_layer=1,
                 img_heatmap_layer=2,
                 img_reg_layer=3,
                 nms_kernel_size=3,
                 img_nms_kernel_size=3,
                 ffn_channel=256,
                 dropout=0.1,
                 bn_momentum=0.1,
                 activation='relu',
                 # config for FFN
                 common_heads=dict(),
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 bias='auto',
                 # loss
                 loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_bbox=dict(type='L1Loss', reduction='mean'),
                 loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_heatmap_2d=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_center_2d=dict(type='L1Loss', reduction='mean'),
                 # others
                 train_cfg=None,
                 test_cfg=None,
                 bbox_coder=None,
                 bbox_2d_coder=None,
                 use_camera='se',
                 level_num=4,
                 img_reg_bn=False,
                 geometric_transfer=True,
                 view_transform=True,
                 depth_input_channel=2,
                 ):
        super(SparseFusionHead2D_Deform, self).__init__()
        self.num_proposals = num_proposals
        self.num_img_proposals = num_img_proposals
        self.num_classes = num_classes
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.bbox_2d_coder = build_bbox_coder(bbox_2d_coder)

        self.bn_momentum = bn_momentum
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.initialize_by_heatmap = initialize_by_heatmap
        self.semantic_transfer = semantic_transfer
        self.cross_only = cross_only
        self.level_num = level_num
        self.in_channels_img = in_channels_img
        self.view_transform = view_transform
        self.range_num = range_num

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_heatmap = build_loss(loss_heatmap)
        self.loss_heatmap_2d = build_loss(loss_heatmap_2d)
        self.loss_center_2d = build_loss(loss_center_2d)

        self.num_img_decoder_layers = num_img_decoder_layers
        self.num_pts_decoder_layers = num_pts_decoder_layers
        self.num_fusion_decoder_layers = num_fusion_decoder_layers
        self.hidden_channel = hidden_channel
        self.sampling = False
        self.out_size_factor_img = out_size_factor_img
        self.geometric_transfer = geometric_transfer

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1

        heads3d = copy.deepcopy(common_heads)
        heads3d.update(dict(heatmap=(self.num_classes, 2)))
        pts_prediction_heads = FFN(hidden_channel, heads3d, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias)

        fusion_heads = dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2), heatmap=(self.num_classes, 2))
        fusion_prediction_heads = FFN(hidden_channel, fusion_heads, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias)

        heads2d = dict(center_2d=(2, img_reg_layer, img_reg_bn), depth_2d=(1, img_reg_layer, img_reg_bn), cls=(self.num_classes, 2),
                    dim_2d=(3, img_reg_layer, img_reg_bn), rot_2d=(2, img_reg_layer, img_reg_bn), vel_2d=(2, img_reg_layer, img_reg_bn)
                )

        # img_prediction_heads = FFN(hidden_channel, heads2d, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias)
        img_prediction_heads = FFNLN(hidden_channel, heads2d, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias)

        pts_query_pos_embed = [PositionEmbeddingLearned(2, hidden_channel) for _ in range(num_pts_decoder_layers)]
        pts_key_pos_embed = [PositionEmbeddingLearned(2, hidden_channel) for _ in range(num_pts_decoder_layers)]
        self.point_transformer = PointTransformer2D_3D(
            hidden_channel=hidden_channel, num_heads=num_heads, num_decoder_layers=num_pts_decoder_layers,
            prediction_heads=pts_prediction_heads, ffn_channel=ffn_channel, dropout=dropout, activation=activation, test_cfg=test_cfg,
            query_pos=pts_query_pos_embed, key_pos=pts_key_pos_embed
        )

        img_query_pos_embed = [PositionEmbeddingLearnedwoNorm(2, hidden_channel) for _ in range(num_img_decoder_layers)]
        img_key_pos_embed = [PositionEmbeddingLearnedwoNorm(2, hidden_channel) for _ in range(num_img_decoder_layers)]

        self.img_transformer = ImageTransformer_Cam_3D_MS(
            hidden_channel=hidden_channel, num_heads=num_heads, num_decoder_layers=num_img_decoder_layers, out_size_factor_img=out_size_factor_img,
            num_views=num_views, prediction_heads=img_prediction_heads, ffn_channel=ffn_channel, dropout=dropout, activation=activation, test_cfg=test_cfg,
            query_pos=img_query_pos_embed, key_pos=img_key_pos_embed
        )

        if view_transform:
            heads_view = dict(center_view=(2, 2), height_view=(1, 2), dim_view=(3, 2), rot_view=(2, 2),
                              vel_view=(2, 2), heatmap_view=(self.num_classes, 2))
            view_prediction_heads = FFN(hidden_channel, heads_view, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias)
            # view_prediction_heads = FFNLN(hidden_channel, heads_view, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias)

            view_query_pos_embed = PositionEmbeddingLearnedwoNorm(9, hidden_channel)
            view_key_pos_embed = PositionEmbeddingLearnedwoNorm(9, hidden_channel)

            view_projection = ProjectionLayerNorm(hidden_channel)

            self.view_transformer = ViewTransformer(
                hidden_channel=hidden_channel, num_heads=num_heads, prediction_heads=view_prediction_heads,
                ffn_channel=ffn_channel, dropout=dropout, activation=activation, test_cfg=test_cfg,
                query_pos=view_query_pos_embed, key_pos=view_key_pos_embed, view_projection=view_projection,
                use_camera=use_camera
            )

        fusion_query_pos_embed = [PositionEmbeddingLearned(2, hidden_channel) for _ in range(self.num_fusion_decoder_layers)]
        fusion_key_pos_embed = [PositionEmbeddingLearned(2, hidden_channel) for _ in range(self.num_fusion_decoder_layers)]

        fuse_pts_projection = ProjectionLayerNorm(hidden_channel)
        fuse_img_projection = ProjectionLayerNorm(hidden_channel)

        self.fusion_transformer = FusionTransformer2D_3D_Self(
            hidden_channel=hidden_channel, num_heads=num_heads, num_decoder_layers=num_fusion_decoder_layers,
            prediction_heads=fusion_prediction_heads, ffn_channel=ffn_channel, dropout=dropout,
            activation=activation, test_cfg=test_cfg, query_pos=fusion_query_pos_embed, key_pos=fusion_query_pos_embed,
            pts_projection=fuse_pts_projection, img_projection=fuse_img_projection,
            num_proposals=num_proposals
        )

        if self.initialize_by_heatmap and self.semantic_transfer:
            self.heatmap_pts_proj = nn.Sequential(
                nn.Linear(hidden_channel, hidden_channel),
                nn.LayerNorm(hidden_channel)
            )
            self.heatmap_img_proj = nn.Sequential(
                nn.Linear(hidden_channel, hidden_channel),
                nn.LayerNorm(hidden_channel)
            )
            self.cross_heatmap_head = self.build_heatmap_LN(hidden_channel, bias, num_classes, layer_num=cross_heatmap_layer)

            colattn_query_pos = PositionEmbeddingLearnedwoNorm(3, hidden_channel)
            colattn_key_pos = PositionEmbeddingLearnedwoNorm(2, hidden_channel)
            self.cross_heatmap_decoder = DeformableTransformerDecoderLayer(
                hidden_channel, num_heads, dim_feedforward=ffn_channel, dropout=dropout, activation=activation,
                self_posembed=colattn_query_pos, cross_posembed=colattn_key_pos, cross_only=False
            )

            self.reduce_conv = ConvLN(
                hidden_channel+1, hidden_channel, kernel_size=3, padding=1
            )

        # a shared convolution
        self.shared_conv = build_conv_layer(
            dict(type='Conv2d'),
            in_channels,
            hidden_channel,
            kernel_size=3,
            padding=1,
            bias=bias,
        )

        # transformer decoder layers for object query with LiDAR feature
        self.num_views = num_views
        if self.geometric_transfer:
            self.shared_conv_img = nn.Identity()
            blocks = [1] * self.level_num
            assert len(blocks) == self.level_num
            self.depth_resnet = DepthEncoderResNet(depth_input_channel, in_channels_img, hidden_channel, depth_layers=blocks)

        else:
            self.shared_conv_img = build_conv_layer(
                dict(type='Conv2d'),
                in_channels_img,  # channel of img feature map
                hidden_channel,
                kernel_size=3,
                padding=1,
                bias=bias,
            )

        # Position Embedding for Cross-Attention, which is re-used during training
        x_size = self.test_cfg['grid_size'][0] // self.test_cfg['out_size_factor']
        y_size = self.test_cfg['grid_size'][1] // self.test_cfg['out_size_factor']
        self.bev_pos = self.create_2D_grid(x_size, y_size)

        if self.initialize_by_heatmap:
            self.heatmap_head = self.build_heatmap(hidden_channel, bias, num_classes)
            self.img_heatmap_head = nn.ModuleList()
            for lvl in range(self.level_num):
                self.img_heatmap_head.append(self.build_heatmap_LN(hidden_channel, bias, num_classes, layer_num=img_heatmap_layer))

            self.class_encoding = nn.Conv1d(num_classes, hidden_channel, 1)
            self.img_class_encoding = nn.Conv1d(num_classes, hidden_channel, 1)
        else:
            # query feature
            self.pts_query_feat = nn.Parameter(torch.randn(1, hidden_channel, self.num_proposals))
            self.pts_query_pos = nn.Parameter(torch.rand([1, self.num_proposals, 2])*torch.Tensor([x_size, y_size]).reshape(1, 1, 2), requires_grad=True)

            self.img_query_feat = nn.Parameter(torch.randn(1, hidden_channel, self.num_img_proposals))
            self.img_query_pos = nn.Parameter(torch.rand([1, self.num_img_proposals, 2]), requires_grad=True)
            self.img_query_pos = inverse_sigmoid(self.img_query_pos)

        self.nms_kernel_size = nms_kernel_size
        self.img_nms_kernel_size = img_nms_kernel_size
        self.img_feat_pos = None
        self.img_feat_collapsed_pos = None

        self.init_weights()
        self._init_assigner_sampler()

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_y, batch_x = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def init_weights(self):
        # initialize transformer
        for m in self.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

        self.init_bn_momentum()

        if self.geometric_transfer:
            level_pos = torch.zeros([self.level_num, self.hidden_channel])
        else:
            level_pos = torch.zeros([self.level_num, self.in_channels_img])
        self.level_pos = nn.Parameter(level_pos, requires_grad=True)
        torch.nn.init.normal_(self.level_pos)

    def _init_assigner_sampler(self):
        """Initialize the target assigner and sampler of the head."""
        if self.train_cfg is None:
            return

        if self.sampling:
            self.bbox_sampler = build_sampler(self.train_cfg.sampler)
        else:
            self.bbox_sampler = PseudoSampler()
        if isinstance(self.train_cfg.assigner, dict):
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
        elif isinstance(self.train_cfg.assigner, list):
            self.bbox_assigner = [
                build_assigner(res) for res in self.train_cfg.assigner
            ]
        if isinstance(self.train_cfg.assigner_2d, dict):
            self.bbox_assigner_2d = build_assigner(self.train_cfg.assigner_2d)
        elif isinstance(self.train_cfg.assigner_2d, list):
            self.bbox_assigner_2d = [
                build_assigner(res) for res in self.train_cfg.assigner_2d
            ]

    def forward_single(self, inputs, img_inputs, img_metas, sparse_depth):
        """
        Args:
            inputs (torch.Tensor): Input feature map with the shape of
                [B, C, 128(H), 128(W)]. (consistent with L748)
            img_inputs (torch.Tensor): Input feature map with the shape of
                [B*num_view, C, image_H, image_W]

            sparse_depth (torch.Tensor): Input normalized depth with the shape of
                [B, num_views, num_scales, depth_C, depth_H, depth_W]

        Returns:
            list[dict]: Output results for tasks.
        """

        batch_size = inputs.shape[0]
        sparse_depth = sparse_depth[:, :, 0, :2]

        if self.geometric_transfer:
            sparse_depth = sparse_depth.view(batch_size*self.num_views, 1, -1, sparse_depth.shape[-2], sparse_depth.shape[-1])
            img_inputs = self.depth_resnet(sparse_depth[:, 0], img_inputs)

        img_feats = []
        for i in range(self.level_num):
            img_inputs_level = img_inputs[i] + self.level_pos[i].reshape(1, self.level_pos[i].shape[0], 1, 1)
            img_feat = self.shared_conv_img(img_inputs_level)
            img_feats.append(img_feat)
        input_padding_mask = self.construct_input_padding_mask(img_feats, img_metas)
        # input_padding_mask = None
        img_feats_pos = []
        normal_img_feats_pos = []
        for lvl in range(self.level_num):
            h, w = img_feats[lvl].shape[-2], img_feats[lvl].shape[-1]
            img_feat_pos = self.create_2D_grid(h, w).to(img_feats[lvl].device)  # (1, h*w, 2)
            img_feats_pos.append(img_feat_pos)
            normal_img_feat_pos = normalize_pos(img_feat_pos, w, h)  # (1, h*w, 2)
            normal_img_feats_pos.append(normal_img_feat_pos)
        normal_img_feats_pos_stack = torch.cat(normal_img_feats_pos, dim=1)  # (1, h*w (sum), 2)
        self.normal_img_feats_pos_stack = normal_img_feats_pos_stack
        normal_img_feats_pos_repeat = normal_img_feats_pos_stack.repeat(batch_size, 1, 1)

        proj_matrix = self.construct_projection_matrix(img_metas, normal_img_feats_pos_stack.device)

        inputs, min_voxel_height, max_voxel_height = inputs[:, :-2], inputs[:, -2], inputs[:, -1]
        lidar_feat = self.shared_conv(inputs)  # [BS, C, H, W]
        #################################
        # image to BEV
        #################################
        lidar_feat_flatten = lidar_feat.view(batch_size, lidar_feat.shape[1], -1)  # [BS, C, H*W]
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)  # [BS, H*W, 2]
        if self.initialize_by_heatmap:
            if self.semantic_transfer:
                img_feat_cross = []
                for level in range(self.level_num):
                    img_feat_cross.append(img_feats[level].clone())
            else:
                img_feat_cross = None
            heatmap, dense_heatmap, pts_top_proposals_class, pts_top_proposals_index = self.generate_heatmap(lidar_feat.clone(), min_voxel_height, max_voxel_height, batch_size, img_metas, proj_matrix['lidar2img_rt'], img_feat_cross, input_padding_mask)
            pts_query_feat = lidar_feat_flatten.gather(
                index=pts_top_proposals_index[:, None, :].expand(-1, lidar_feat_flatten.shape[1], -1), dim=-1
            )  # [BS, C, num_proposals]
            # add category embedding
            one_hot = F.one_hot(pts_top_proposals_class, num_classes=self.num_classes).permute(0, 2, 1)  # [BS, num_classes, num_proposals]
            query_cat_encoding = self.class_encoding(one_hot.float())  # [BS, C, num_proposals]
            self.query_labels = pts_top_proposals_class
            pts_query_feat += query_cat_encoding
            pts_query_pos = bev_pos.gather(
                index=pts_top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]), dim=1
            )  # [BS, num_proposals, 2]
        else:
            pts_query_feat = self.pts_query_feat.repeat(batch_size, 1, 1)  # [BS, C, num_proposals]
            pts_query_pos = self.pts_query_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)  # [BS, num_proposals, 2]

        if self.initialize_by_heatmap:
            img_feats_heatmap = []
            for lvl in range(self.level_num):
                img_feats_heatmap.append(img_feats[lvl].clone())

            img_heatmap, img_dense_heatmap, img_top_proposals_class, img_top_proposals_index, img_top_proposals_view_idx, img_top_proposals_pos_id = \
                self.generate_heatmap_img(img_feats_heatmap, batch_size)
            img_feats_flatten = []
            for lvl in range(self.level_num):
                img_feat = img_feats[lvl]
                h, w = img_feat.shape[-2], img_feat.shape[-1]
                img_feat_flatten = img_feat.reshape(batch_size, self.num_views, self.hidden_channel, h * w)
                img_feat_flatten = img_feat_flatten.permute(0, 2, 1, 3)  # [BS, C, num_view, h*w]
                img_feats_flatten.append(img_feat_flatten)
            img_feat_stack = torch.cat(img_feats_flatten, dim=-1)  # [BS, C, num_view, h*w (sum)]
            img_feat_stack = img_feat_stack.view(batch_size, self.hidden_channel, self.num_views*img_feat_stack.shape[-1])
            normal_img_query_pos = normal_img_feats_pos_repeat.gather(
                index=img_top_proposals_pos_id[:, None, :].permute(0, 2, 1).expand(-1, -1, normal_img_feats_pos_stack.shape[-1]), dim=1
            )  # [BS, num_proposals, 2]
            img_query_feat = img_feat_stack.gather(
                index=img_top_proposals_index[:, None, :].expand(-1, img_feat_stack.shape[1], -1), dim=-1
            )  # [BS, C, num_proposals]
            img_query_view = img_top_proposals_view_idx.clone()  #  [BS, num_proposals]
            one_hot = F.one_hot(img_top_proposals_class, num_classes=self.num_classes).permute(0, 2, 1)  # [BS, num_classes, num_proposals]
            self.img_query_label = img_top_proposals_class
            img_query_cat_encoding = self.img_class_encoding(one_hot.float())  # [BS, C, num_proposals]
            img_query_feat += img_query_cat_encoding
        else:
            img_query_feat = self.img_query_feat.repeat(batch_size, 1, 1)  # [BS, C, num_proposals]
            normal_img_query_pos = self.img_query_pos.repeat(batch_size, 1, 1).to(img_feat.device)  # [BS, num_proposals, 2]
            img_query_pos_view = torch.arange(self.num_img_proposals).reshape(1, -1).repeat(batch_size, 1).to(img_feat.device)
            img_query_view = img_query_pos_view % self.num_views
        view_proj_matrix = self.construction_view_projection_matrix(proj_matrix, img_query_view)

        #################################
        # transformer decoder layer (LiDAR feature as K,V)
        #################################
        ret_dicts = []
        pts_query_feat, pts_query_pos, pts_ret_dicts = self.point_transformer(pts_query_feat, pts_query_pos, lidar_feat_flatten, bev_pos)
        ret_dicts.extend(pts_ret_dicts)

        #################################
        # transformer decoder layer (img feature as K,V)
        #################################

        img_query_feat, normal_img_query_pos, img_query_pos_bev, camera_info, img_ret_dicts = \
            self.img_transformer(img_query_feat, normal_img_query_pos, img_query_view, img_feats, normal_img_feats_pos_stack, view_proj_matrix['lidar2cam_rt'], view_proj_matrix['cam_intrinsic'], img_metas, input_padding_mask)

        #################################
        # view transformation layer
        #################################

        if self.view_transform:
            img_query_feat, img_query_pos_bev, view_ret_dicts = self.view_transformer(img_query_feat, img_query_pos_bev, normal_img_query_pos[..., :2], img_ret_dicts, camera_info)

        img_query_pos_bev = img_query_pos_bev[..., :2]

        #################################
        # fusion layer
        #################################

        all_query_feat, all_query_pos, fusion_ret_dicts = self.fusion_transformer(pts_query_feat, pts_query_pos, img_query_feat, img_query_pos_bev)

        ret_dicts.extend(fusion_ret_dicts)
        if self.initialize_by_heatmap:
            ret_dicts[0]['query_heatmap_score'] = heatmap.gather(index=pts_top_proposals_index[:, None, :].expand(-1, self.num_classes, -1), dim=-1)  # [bs, num_classes, num_proposals]
            ret_dicts[0]['dense_heatmap'] = dense_heatmap
            ret_dicts[0]['img_query_heatmap_score'] = img_heatmap.gather(index=img_top_proposals_index[:, None, :].expand(-1, self.num_classes, -1), dim=-1)  # [bs, num_classes, num_proposals]
            ret_dicts[0]['img_dense_heatmap'] = img_dense_heatmap

        # return all the layer's results for auxiliary superivison
        new_res = {}
        for key in ret_dicts[0].keys():
            if key not in ['dense_heatmap', 'query_heatmap_score', 'img_query_heatmap_score', 'img_dense_heatmap']:
                new_res[key] = torch.cat([ret_dict[key] for ret_dict in ret_dicts], dim=-1)
            else:
                new_res[key] = ret_dicts[0][key]
        for key in img_ret_dicts[0].keys():
            new_res[key] = torch.cat([ret_dict[key] for ret_dict in img_ret_dicts], dim=-1)
        new_res['view'] = img_query_view.repeat(1, self.num_img_decoder_layers)
        if self.view_transform:
            for key in view_ret_dicts[0].keys():
                new_res[key] = torch.cat([ret_dict[key] for ret_dict in view_ret_dicts], dim=-1)

        return [new_res]

    def forward(self, feats, img_feats, img_metas, sparse_depth=None):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            tuple(list[dict]): Output results. first index by level, second index by layer
        """

        if img_feats is None:
            img_feats = [None]
        else:
            img_feats = [img_feats[:self.level_num]]
        if sparse_depth is None:
            sparse_depth = [None]
        else:
            sparse_depth = [sparse_depth[:, :, :self.level_num]]
        res = multi_apply(self.forward_single, feats, img_feats, [img_metas], sparse_depth)
        assert len(res) == 1, "only support one level features."
        return res

    def construct_input_padding_mask(self, img_feats, img_metas):
        batch_size = len(img_metas)
        device = img_feats[0].device
        img_h_lvl = []
        img_w_lvl = []
        for img_feat_lvl in img_feats:
            img_h_lvl.append(img_feat_lvl.shape[-2])
            img_w_lvl.append(img_feat_lvl.shape[-1])
        padding_mask = []
        for sample_idx in range(batch_size):
            sample_mask = []
            for view_idx in range(self.num_views):
                view_mask = []

                if 'valid_shape' in img_metas[sample_idx]:
                    valid_shape = img_metas[sample_idx]['valid_shape'][view_idx] / self.out_size_factor_img
                else:
                    valid_shape = np.array([img_metas[sample_idx]['img_shape'][1], img_metas[sample_idx]['img_shape'][0]]) / self.out_size_factor_img
                for lvl_idx in range(self.level_num):
                    lvl_mask = torch.ones([img_h_lvl[lvl_idx], img_w_lvl[lvl_idx]], dtype=torch.bool, device=device)
                    valid_shape_lvl = valid_shape // (2 ** lvl_idx)
                    valid_w_lvl = int(valid_shape_lvl[0])
                    valid_h_lvl = int(valid_shape_lvl[1])
                    lvl_mask[:valid_h_lvl, :valid_w_lvl] = False
                    view_mask.append(lvl_mask.view(-1))
                view_mask = torch.cat(view_mask)
                sample_mask.append(view_mask)
            sample_mask = torch.stack(sample_mask, dim=0)
            padding_mask.append(sample_mask)
        padding_mask = torch.stack(padding_mask, dim=0)

        return padding_mask

    def construction_view_projection_matrix(self, proj_matrix, img_query_view):
        view_proj_matrix = {}
        batch_size = img_query_view.shape[0]
        batch_ids = torch.arange(batch_size)[:, None].repeat(1, self.num_img_proposals)
        batch_ids = batch_ids.to(img_query_view.device)
        for key in proj_matrix:
            view_proj_matrix[key] = proj_matrix[key][batch_ids, img_query_view]
        return view_proj_matrix

    def construct_projection_matrix(self, img_metas, device):
        batch_size = len(img_metas)
        cam_ints = torch.zeros([batch_size, self.num_views, 4, 4], device=device)
        cam_ints[:, :, 3, 3] = 1
        for sample_id in range(batch_size):
            cam_ints[sample_id, :, :3, :3] = torch.Tensor(img_metas[sample_id]['cam_intrinsic']).to(device)

        lidar2cam_rt = torch.zeros([batch_size, self.num_views, 4, 4], device=device)
        lidar2cam_rt[:, :, 3, 3] = 1
        for sample_id in range(batch_size):
            lidar2cam_rt[sample_id, :, :3, :3] = torch.Tensor(img_metas[sample_id]['lidar2cam_r']).to(device)
            lidar2cam_rt[sample_id, :, :3, 3] = torch.Tensor(img_metas[sample_id]['lidar2cam_t']).to(device)

        lidar2img_rt = torch.matmul(cam_ints, lidar2cam_rt)
        proj_matrix = {"cam_intrinsic": cam_ints, "lidar2cam_rt": lidar2cam_rt, "lidar2img_rt": lidar2img_rt}
        return proj_matrix

    def build_heatmap_LN(self, hidden_channel, bias, num_classes, layer_num=2, kernel_size=3):
        layers = []
        for i in range(layer_num-1):
            layers.append(ConvLN(
                hidden_channel,
                hidden_channel,
                kernel_size=kernel_size,
                padding=(kernel_size-1)//2,
            ))

        layers.append(build_conv_layer(
            dict(type='Conv2d'),
            hidden_channel,
            num_classes,
            kernel_size=kernel_size,
            padding=(kernel_size-1)//2,
            bias=bias,
        ))
        return nn.Sequential(*layers)

    def build_heatmap(self, hidden_channel, bias, num_classes, layer_num=2, kernel_size=3):
        layers = []
        for i in range(layer_num-1):
            layers.append(ConvModule(
                hidden_channel,
                hidden_channel,
                kernel_size=kernel_size,
                padding=(kernel_size-1)//2,
                bias=bias,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
            ))

        layers.append(build_conv_layer(
            dict(type='Conv2d'),
            hidden_channel,
            num_classes,
            kernel_size=kernel_size,
            padding=(kernel_size-1)//2,
            bias=bias,
        ))
        return nn.Sequential(*layers)

    def generate_heatmap_deform(self, lidar_feat, img_feat, voxel_height, img_metas, lidar2img_rt, input_padding_mask=None):
        # img_feat [bs*num_view, C, img_h, img_w]
        # lidar_feat [BS, C, H, W]

        batch_size = lidar_feat.shape[0]
        H, W = lidar_feat.shape[2], lidar_feat.shape[3]
        voxel_height = voxel_height.view(batch_size, H*W)
        valid_height_mask = voxel_height > -50

        level_start_index = [0]
        spatial_shapes = []
        img_feats_flatten = []

        for lvl in range(self.level_num):
            img_h_lvl, img_w_lvl = img_feat[lvl].shape[-2], img_feat[lvl].shape[-1]
            img_feat[lvl] = self.heatmap_img_proj(img_feat[lvl].permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            # img_feat[lvl] = self.heatmap_img_proj(img_feat[lvl])
            img_feat[lvl] = img_feat[lvl].view(batch_size, self.num_views, self.hidden_channel, img_h_lvl, img_w_lvl)
            img_feat_flatten = img_feat[lvl].view(batch_size, self.num_views, self.hidden_channel, img_h_lvl*img_w_lvl)
            img_feats_flatten.append(img_feat_flatten)
            level_start_index.append(level_start_index[-1] + img_h_lvl * img_w_lvl)
            spatial_shapes.append([img_h_lvl, img_w_lvl])

        level_start_index = level_start_index[:-1]
        level_start_index = torch.LongTensor(level_start_index).to(lidar_feat.device)
        spatial_shapes = torch.LongTensor(spatial_shapes).to(lidar_feat.device)
        img_feats_stack = torch.cat(img_feats_flatten, dim=3)  # [bs, num_view, C, h*w (sum)]
        normal_img_feats_pos_stack = self.normal_img_feats_pos_stack  # [1, h*w (sum), 2]

        lidar_feat = self.heatmap_pts_proj(lidar_feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # lidar_feat = self.heatmap_pts_proj(lidar_feat)

        lidar_feat_flatten = lidar_feat.reshape(batch_size, self.hidden_channel, H*W)  # [bs, C, H*W]
        lidar_feat_output = torch.zeros(batch_size, self.hidden_channel, H*W).to(lidar_feat.device)
        lidar_feat_count = torch.zeros(batch_size, 1, H*W).to(lidar_feat.device)

        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)
        query_pos_realmetric = bev_pos.permute(0, 2, 1) * self.test_cfg['out_size_factor'] * \
                               self.test_cfg['voxel_size'][0] + self.test_cfg['pc_range'][0]  # (bs, 2, H*W)

        query_pos_3d = torch.cat([query_pos_realmetric, voxel_height[:, None]], dim=1) # (bs, 3, H*W)
        points_4d = torch.cat([query_pos_3d, torch.ones_like(query_pos_3d[:, :1])], dim=1).permute(0, 2, 1)  # (bs, H*W, 4)
        points_2d = torch.matmul(points_4d[:, None], lidar2img_rt.transpose(-1, -2))  # (bs, num_view, H*W, 4)
        points_2d[..., 2] = torch.clamp(points_2d[..., 2], min=1e-5)
        points_2d[..., :2] = points_2d[..., :2] / points_2d[..., 2:3] / self.out_size_factor_img

        if 'valid_shape' in img_metas[0]:
            valid_shape = []
            for sample_idx in range(batch_size):
                sample_valid_shape = img_metas[sample_idx]['valid_shape'] / self.out_size_factor_img
                valid_shape.append(sample_valid_shape)
            valid_shape = np.array(valid_shape)
            valid_img_w = valid_shape[..., 0]
            valid_img_h = valid_shape[..., 1]
        else:
            valid_img_w = np.full([batch_size, self.num_views], img_feat[0].shape[-1])
            valid_img_h = np.full([batch_size, self.num_views], img_feat[0].shape[-2])

        valid_img_w = torch.from_numpy(valid_img_w).to(points_2d.device)
        valid_img_h = torch.from_numpy(valid_img_h).to(points_2d.device)

        img_h, img_w = img_feat[0].shape[-2], img_feat[0].shape[-1]
        center_xs = points_2d[..., 0]  # (bs, num_view, H*W)
        center_ys = points_2d[..., 1]

        on_the_image = (center_xs >= 0) & (center_xs < valid_img_w[..., None]) & (center_ys >= 0) & \
                       (center_ys < valid_img_h[..., None]) & valid_height_mask[:, None]  # [bs, num_view, H*W]
        depth = points_2d[..., 2]   # [bs, num_view, H*W]
        depth = torch.log(depth)

        for sample_idx in range(batch_size):
            on_the_image_sample = on_the_image[sample_idx]  # [num_view, H*W]
            bincount = torch.sum(on_the_image_sample, dim=1)
            max_len = torch.max(bincount)
            sample_query_feature = torch.zeros([self.num_views, self.hidden_channel, max_len], device=points_2d.device)
            sample_query_pos = torch.zeros([self.num_views, max_len, 3], device=points_2d.device)
            sample_reference_points = torch.zeros([self.num_views, max_len, 2], device=points_2d.device)
            sample_padding_mask = torch.zeros([self.num_views, max_len], device=points_2d.device, dtype=torch.bool)

            for view_idx in range(self.num_views):
                on_the_image_view = on_the_image_sample[view_idx]
                center_xs_view = center_xs[sample_idx, view_idx, on_the_image_view]  # [N, ]
                center_ys_view = center_ys[sample_idx, view_idx, on_the_image_view]  # [N, ]
                reference_points = torch.stack([center_xs_view / img_w, center_ys_view / img_h], dim=-1)  # [N, 2]

                view_count = bincount[view_idx]
                sample_reference_points[view_idx, :view_count] = reference_points
                sample_query_feature[view_idx, :, :view_count] = lidar_feat_flatten[sample_idx, :, on_the_image_view]
                sample_query_pos[view_idx, :view_count, 2] = depth[sample_idx, view_idx, on_the_image_view]
                sample_padding_mask[view_idx, view_count:] = True

            sample_centers_normal = sample_reference_points * 2 - 1
            sample_query_img_feat = []
            for lvl in range(self.level_num):
                img_feat_lvl = img_feat[lvl][sample_idx]
                img_feat_lvl = F.grid_sample(img_feat_lvl, sample_centers_normal[:, None], mode='bilinear', padding_mode="border", align_corners=False)
                img_feat_lvl = img_feat_lvl[:, :, 0]
                sample_query_img_feat.append(img_feat_lvl)
            sample_query_img_feat = torch.stack(sample_query_img_feat, dim=0)
            sample_query_img_feat = torch.max(sample_query_img_feat, dim=0)[0]  # [num_view, C, max_len]

            sample_query_feature = sample_query_feature + sample_query_img_feat
            sample_query_pos[..., :2] = inverse_sigmoid(sample_reference_points)

            sample_reference_points = sample_reference_points[:, :, None].repeat(1, 1, self.level_num, 1)

            if batch_size == 1: # whether it is doing evaluation or training
                if input_padding_mask is None:
                    sample_input_padding_mask = None
                else:
                    sample_input_padding_mask = input_padding_mask[sample_idx:sample_idx+1]
                output = self.cross_heatmap_decoder(
                    sample_query_feature, img_feats_stack[sample_idx],
                    sample_query_pos, normal_img_feats_pos_stack.repeat(self.num_views, 1, 1),
                    reference_points=sample_reference_points, level_start_index=level_start_index, spatial_shapes=spatial_shapes,
                    query_padding_mask=sample_padding_mask, input_padding_mask=sample_input_padding_mask
                )
            else:
                output = []
                for view_idx in range(self.num_views):
                    view_query_feature = sample_query_feature[view_idx, :, torch.logical_not(sample_padding_mask[view_idx])]
                    view_query_pos = sample_query_pos[view_idx, torch.logical_not(sample_padding_mask[view_idx])]
                    view_reference_points = sample_reference_points[view_idx, torch.logical_not(sample_padding_mask[view_idx])]

                    if input_padding_mask is None:
                        view_input_padding_mask = None
                    else:
                        view_input_padding_mask = input_padding_mask[sample_idx, view_idx, None]

                    output_item = self.cross_heatmap_decoder(
                        view_query_feature[None], img_feats_stack[sample_idx, view_idx, None],
                        view_query_pos[None], normal_img_feats_pos_stack,
                        reference_points=view_reference_points[None], level_start_index=level_start_index, spatial_shapes=spatial_shapes,
                        input_padding_mask=view_input_padding_mask
                    )
                    output_item_pad = torch.zeros([output_item.shape[1], sample_padding_mask.shape[1]]).type_as(output_item)
                    output_item_pad[:, torch.logical_not(sample_padding_mask[view_idx])] = output_item[0]
                    output.append(output_item_pad)
                output = torch.stack(output, dim=0)

            for view_idx in range(self.num_views):
                view_count = bincount[view_idx]
                on_the_image_view = on_the_image_sample[view_idx]
                overlap_mask = lidar_feat_count[sample_idx, 0, on_the_image_view] > 0
                output_view = output[view_idx, :, :view_count]
                nonoverlap_mask = torch.logical_not(overlap_mask)
                lidar_feat_output_view = lidar_feat_output[sample_idx, :, on_the_image_view]
                lidar_feat_output_view[:, overlap_mask] = torch.maximum(lidar_feat_output_view[:, overlap_mask], output_view[:, overlap_mask])
                lidar_feat_output_view[:, nonoverlap_mask] = output_view[:, nonoverlap_mask]
                lidar_feat_output[sample_idx, :, on_the_image_view] = lidar_feat_output_view
                lidar_feat_count[sample_idx, :, on_the_image_view] += 1

        lidar_feat_output = lidar_feat_output.reshape(batch_size, lidar_feat_output.shape[1], H, W)
        # lidar_feat_output = self.reduce_conv(lidar_feat_output)
        lidar_feat_count = lidar_feat_count.reshape(batch_size, 1, H, W)
        lidar_feat_flag = torch.where(lidar_feat_count>0, torch.ones_like(lidar_feat_count), torch.zeros_like(lidar_feat_count))
        lidar_feat_output = lidar_feat_output + (1 - lidar_feat_flag) * lidar_feat
        lidar_feat_output = torch.cat([lidar_feat_output, lidar_feat_flag], dim=1)
        lidar_feat_output = self.reduce_conv(lidar_feat_output)

        heatmap_output = self.cross_heatmap_head(lidar_feat_output.contiguous())

        return heatmap_output

    def generate_heatmap(self, lidar_feat, min_voxel_height, max_voxel_height, batch_size, img_metas, lidar2img_rt, img_feat=None, input_padding_mask=None):
        dense_heatmap = self.heatmap_head(lidar_feat)  # [BS, num_class, H, W]
        if img_feat is None:
            heatmap = dense_heatmap.detach().sigmoid()  # [BS, num_class, H, W]
        else:
            voxel_height = (min_voxel_height + max_voxel_height) / 2
            dense_heatmap_cross = self.generate_heatmap_deform(lidar_feat, img_feat, voxel_height, img_metas, lidar2img_rt, input_padding_mask)

            if self.cross_only:
                heatmap = dense_heatmap_cross.detach().sigmoid()
            else:
                heatmap = (dense_heatmap.detach().sigmoid() + dense_heatmap_cross.detach().sigmoid()) / 2
            dense_heatmap = dense_heatmap_cross
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        # equals to nms radius = voxel_size * out_size_factor * kenel_size
        local_max_inner = F.max_pool2d(heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0)
        local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner
        ## for Pedestrian & Traffic_cone in nuScenes
        if self.test_cfg['dataset'] == 'nuScenes':
            local_max[:, 8, ] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0)
            local_max[:, 9, ] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)
        elif self.test_cfg['dataset'] == 'Waymo':  # for Pedestrian & Cyclist in Waymo
            local_max[:, 1, ] = F.max_pool2d(heatmap[:, 1], kernel_size=1, stride=1, padding=0)
            local_max[:, 2, ] = F.max_pool2d(heatmap[:, 2], kernel_size=1, stride=1, padding=0)
        heatmap = heatmap * (heatmap == local_max)  # [BS, num_class, H, W]
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)  # [BS, num_class, H*W]

        # top #num_proposals among all classes
        top_proposals = heatmap.reshape(batch_size, -1).argsort(dim=-1, descending=True)[..., :self.num_proposals]  # [BS, num_proposals]

        top_proposals_class = top_proposals // heatmap.shape[-1]  # [BS, num_proposals]
        top_proposals_index = top_proposals % heatmap.shape[-1]  # [BS, num_proposals]
        return heatmap, dense_heatmap, top_proposals_class, top_proposals_index

    def generate_heatmap_img(self, img_feats, batch_size):

        img_dense_heatmaps = []
        img_heatmaps = []
        for lvl in range(self.level_num):

            # img_dense_heatmap = self.img_heatmap_head(img_feats[lvl])  # [BS*num_view, num_class, h, w]
            img_dense_heatmap = self.img_heatmap_head[lvl](img_feats[lvl])  # [BS*num_view, num_class, h, w]

            img_heatmap = img_dense_heatmap.detach().sigmoid()  # [BS*num_view, num_class, h, w]
            padding = self.img_nms_kernel_size // 2
            local_max = torch.zeros_like(img_heatmap)
            # equals to nms radius = voxel_size * out_size_factor * kenel_size
            local_max_inner = F.max_pool2d(img_heatmap, kernel_size=self.img_nms_kernel_size, stride=1, padding=0)
            local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner
            img_heatmap = img_heatmap * (img_heatmap == local_max)  # [BS*num_view, num_class, h, w]
            img_heatmap = img_heatmap.view(batch_size, self.num_views, img_heatmap.shape[1], -1)  # [BS, num_views, num_class, h*w]
            img_heatmap = img_heatmap.permute(0, 2, 1, 3) # [BS, num_class, num_views, h*w]
            img_heatmaps.append(img_heatmap)

            img_dense_heatmap = img_dense_heatmap.view(batch_size, self.num_views, img_dense_heatmap.shape[1],
                        img_dense_heatmap.shape[2], img_dense_heatmap.shape[3])  # [BS, num_views, num_class, h, w]
            img_dense_heatmap = img_dense_heatmap.permute(0, 2, 1, 3, 4)  # [BS, num_class, num_views, h, w]
            img_dense_heatmap = img_dense_heatmap.view(batch_size, self.num_classes, self.num_views, img_dense_heatmap.shape[-2]*img_dense_heatmap.shape[-1])
            img_dense_heatmaps.append(img_dense_heatmap)

        img_heatmap_stack = torch.cat(img_heatmaps, dim=3)  # [BS, num_class, num_views, h*w (sum)]
        # top #num_proposals among all classes
        top_proposals = img_heatmap_stack.view(batch_size, -1).argsort(dim=-1, descending=True)[..., :self.num_img_proposals]  # [BS, num_proposals]
        top_proposals_class = top_proposals // (img_heatmap_stack.shape[-1]*img_heatmap_stack.shape[-2])  # [BS, num_proposals]

        top_proposals_view_index = top_proposals % (img_heatmap_stack.shape[-1]*img_heatmap_stack.shape[-2]) // img_heatmap_stack.shape[-1]  # [BS, num_proposals]
        top_proposals_pos_index = top_proposals % img_heatmap_stack.shape[-1]  # [BS, num_proposals]
        top_proposals_index = top_proposals % (img_heatmap_stack.shape[-1]*img_heatmap_stack.shape[-2])  # [BS, num_proposals]

        img_heatmap_stack = img_heatmap_stack.contiguous().view(batch_size, img_heatmap_stack.shape[1], -1)
        img_dense_heatmaps_stack = torch.cat(img_dense_heatmaps, dim=-1)

        return img_heatmap_stack, img_dense_heatmaps_stack, top_proposals_class, top_proposals_index, top_proposals_view_index, top_proposals_pos_index

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels, gt_img_centers_view, gt_bboxes_cam_view, gt_visible, gt_bboxes_lidar_view, preds_dict, img_metas):
        """Generate training targets.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.
            preds_dicts (tuple of dict): first index by layer (default 1)
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.

                - torch.Tensor: classification target.  [BS, num_proposals]
                - torch.Tensor: classification weights (mask)  [BS, num_proposals]
                - torch.Tensor: regression target. [BS, num_proposals, 8]
                - torch.Tensor: regression weights. [BS, num_proposals, 8]
        """
        # change preds_dict into list of dict (index by batch_id)
        # preds_dict[0]['center'].shape [bs, 3, num_proposal]
        list_of_pred_dict = []
        for batch_idx in range(len(gt_bboxes_3d)):
            pred_dict = {}
            for key in preds_dict[0].keys():
                pred_dict[key] = preds_dict[0][key][batch_idx:batch_idx + 1]
            list_of_pred_dict.append(pred_dict)

        assert len(gt_bboxes_3d) == len(list_of_pred_dict)

        res_tuple = multi_apply(self.get_targets_single, gt_bboxes_3d, gt_labels_3d, gt_visible, list_of_pred_dict, np.arange(len(gt_labels_3d)))

        labels = torch.cat(res_tuple[0], dim=0)
        label_weights = torch.cat(res_tuple[1], dim=0)
        bbox_targets = torch.cat(res_tuple[2], dim=0)
        bbox_weights = torch.cat(res_tuple[3], dim=0)
        ious = torch.cat(res_tuple[4], dim=0)
        num_pos_layer = np.concatenate(res_tuple[5], axis=0)  # [BS, num_layer]
        # matched_ious = np.mean(res_tuple[6])
        matched_ious = torch.cat(res_tuple[6], dim=0)

        res_tuple_2d = multi_apply(self.get_targets_single_2d, gt_bboxes, gt_labels, gt_img_centers_view, gt_bboxes_cam_view, gt_bboxes_lidar_view, list_of_pred_dict, img_metas, np.arange(len(gt_bboxes)))
        labels_2d = torch.cat(res_tuple_2d[0], dim=0)
        label_weights_2d = torch.cat(res_tuple_2d[1], dim=0)
        bbox_targets_2d = torch.cat(res_tuple_2d[2], dim=0)
        bbox_weights_2d = torch.cat(res_tuple_2d[3], dim=0)
        ious_2d = torch.cat(res_tuple_2d[4], dim=0)
        num_pos_layer_2d = np.concatenate(res_tuple_2d[5], axis=0)  # [BS, num_layer]
        matched_ious_2d = torch.cat(res_tuple_2d[6], dim=0)

        if self.view_transform:
            res_tuple_view = multi_apply(self.get_targets_single_view, gt_bboxes_3d, gt_labels_3d, gt_visible, list_of_pred_dict, np.arange(len(gt_bboxes)))
            labels_view = torch.cat(res_tuple_view[0], dim=0)
            label_weights_view = torch.cat(res_tuple_view[1], dim=0)
            bbox_targets_view = torch.cat(res_tuple_view[2], dim=0)
            bbox_weights_view = torch.cat(res_tuple_view[3], dim=0)
            ious_view = torch.cat(res_tuple_view[4], dim=0)
            num_pos_layer_view = np.concatenate(res_tuple_view[5], axis=0)  # [BS, num_layer]
            matched_ious_view = torch.cat(res_tuple_view[6], dim=0)

        if self.initialize_by_heatmap:
            heatmap = torch.cat(res_tuple[7], dim=0)
            heatmap_2d = torch.cat(res_tuple_2d[7], dim=0)
            if self.view_transform:
                return labels, label_weights, bbox_targets, bbox_weights, ious, num_pos_layer, matched_ious, heatmap, \
                   labels_2d, label_weights_2d, bbox_targets_2d, bbox_weights_2d, ious_2d, num_pos_layer_2d, \
                   matched_ious_2d, heatmap_2d, labels_view, label_weights_view, bbox_targets_view, bbox_weights_view, \
                   ious_view, num_pos_layer_view, matched_ious_view
            else:
                return labels, label_weights, bbox_targets, bbox_weights, ious, num_pos_layer, matched_ious, heatmap, \
                   labels_2d, label_weights_2d, bbox_targets_2d, bbox_weights_2d, ious_2d, num_pos_layer_2d, \
                   matched_ious_2d, heatmap_2d
        else:
            if self.view_transform:
                return labels, label_weights, bbox_targets, bbox_weights, ious, num_pos_layer, matched_ious, \
                    labels_2d, label_weights_2d, bbox_targets_2d, bbox_weights_2d, ious_2d, num_pos_layer_2d, matched_ious_2d, \
                    labels_view, label_weights_view, bbox_targets_view, bbox_weights_view, ious_view, num_pos_layer_view, \
                    matched_ious_view
            else:
                return labels, label_weights, bbox_targets, bbox_weights, ious, num_pos_layer, matched_ious, \
                    labels_2d, label_weights_2d, bbox_targets_2d, bbox_weights_2d, ious_2d, num_pos_layer_2d, matched_ious_2d,

    def get_targets_single_2d(self, gt_bboxes, gt_labels, gt_centers_2d, gt_bboxes_cam_view, gt_bboxes_lidar_view, preds_dict, img_metas, batch_idx):
        num_proposals = preds_dict['cls'].shape[-1]
        loc_cam_3d = copy.deepcopy(preds_dict['loc_cam_3d'].detach())
        dim = copy.deepcopy(preds_dict['dim_2d'].detach())
        rot = copy.deepcopy(preds_dict['rot_2d'].detach())
        if 'vel_2d' in preds_dict.keys():
            vel = copy.deepcopy(preds_dict['vel_2d'].detach())
        else:
            vel = None
        view = copy.deepcopy(preds_dict['view'].detach())[0] # [num_proposals, ]
        score = copy.deepcopy(preds_dict['cls'].detach())

        bboxes_dict = self.bbox_2d_coder.decode(score, rot, dim, loc_cam_3d, vel)
        bboxes_3d_tensor = bboxes_dict[0]['bboxes']

        gt_bboxes_3d_tensor = gt_bboxes_cam_view.tensor.to(score.device)
        gt_bboxes_lidar_view_tensor = gt_bboxes_lidar_view.tensor.to(score.device)

        assert gt_bboxes_lidar_view_tensor.shape[0] == gt_bboxes_3d_tensor.shape[0]

        img_shape = img_metas['pad_shape']
        img_scale =[img_shape[1], img_shape[0], img_shape[1], img_shape[0]]

        img_scale = torch.Tensor(img_scale).to(score.device).unsqueeze(0)
        gt_centers_2d = gt_centers_2d.float()
        normal_gt_centers = gt_centers_2d[..., :2] / img_scale[..., :2]
        normal_gt_bboxes = gt_bboxes.float() / img_scale

        assign_result_list = []
        for idx_layer in range(self.num_img_decoder_layers):
            bboxes_tensor_layer = bboxes_3d_tensor[idx_layer*self.num_img_proposals:(idx_layer+1)*self.num_img_proposals, :]  # [num_proposals, 10]
            score_layer = score[..., idx_layer*self.num_img_proposals:(idx_layer+1)*self.num_img_proposals]  # [1, num_class, num_proposal]
            view_layer = view[idx_layer*self.num_img_proposals:(idx_layer+1)*self.num_img_proposals]  # [num_proposals]

            assign_result = self.bbox_assigner_2d.assign(bboxes_tensor_layer, gt_bboxes_3d_tensor, gt_labels, score_layer, view_layer, self.train_cfg)
            assign_result_list.append(assign_result)

        # combine assign result of each layer
        assign_result_ensemble = AssignResult(
            num_gts=sum([res.num_gts for res in assign_result_list]),
            gt_inds=torch.cat([res.gt_inds for res in assign_result_list]),
            max_overlaps=torch.cat([res.max_overlaps for res in assign_result_list]),
            labels=torch.cat([res.labels for res in assign_result_list]),
        )
        sampling_result = self.bbox_sampler.sample(assign_result_ensemble, bboxes_3d_tensor, gt_bboxes_3d_tensor)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        assert len(pos_inds) + len(neg_inds) == num_proposals

        start = 0
        pos_num_layers = []
        for idx_layer in range(self.num_img_decoder_layers):
            layer_num_proposal = self.num_img_proposals
            layer_mask = torch.logical_and(pos_inds>=start, pos_inds<start+layer_num_proposal)
            pos_inds_layer = pos_inds[layer_mask]
            count = pos_inds_layer.shape[0]
            pos_num_layers.append(count)
            start += layer_num_proposal
        pos_num_layers = np.array(pos_num_layers)
        assert np.sum(pos_num_layers) == pos_inds.shape[0]

        # create target for loss computation
        bbox_targets = torch.zeros([num_proposals, self.bbox_2d_coder.code_size]).to(score.device)
        bbox_weights = torch.zeros([num_proposals, self.bbox_2d_coder.code_size]).to(score.device)
        view_targets = score.new_zeros(num_proposals, dtype=torch.long)
        ious = assign_result_ensemble.max_overlaps
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = score.new_zeros(num_proposals, dtype=torch.long)
        label_weights = score.new_ones(num_proposals, dtype=torch.long)
        center_targets = torch.zeros([num_proposals, 2]).to(score.device)
        center_weights = torch.zeros([num_proposals, 2]).to(score.device)
        depth_labels = score.new_zeros(num_proposals)
        depth_weights = score.new_zeros(num_proposals, dtype=torch.long)

        bbox_lidar_targets = torch.zeros([self.num_img_proposals, self.bbox_coder.code_size]).to(score.device)
        bbox_lidar_weights = torch.zeros([self.num_img_proposals, self.bbox_coder.code_size]).to(score.device)
        labels_lidar = score.new_zeros(self.num_img_proposals, dtype=torch.long)
        label_lidar_weights = score.new_ones(self.num_img_proposals, dtype=torch.long)
        pos_inds_lastlayer = pos_inds[layer_mask] - (self.num_img_decoder_layers - 1) * self.num_img_proposals
        pos_assigned_gt_inds_lastlayer = sampling_result.pos_assigned_gt_inds[layer_mask]

        ious_lidar = torch.zeros_like(bbox_lidar_targets[:, 0]) - 1

        if gt_labels is not None:  # default label is -1
            labels += self.num_classes
            labels_lidar += self.num_classes

        # both pos and neg have classification loss, only pos has regression and iou loss
        if len(pos_inds) > 0:
            # bbox_targets[pos_inds, :] = sampling_result.pos_gt_bboxes

            bbox_weights[pos_inds, :] = 1.0
            pos_gt_bboxes = sampling_result.pos_gt_bboxes

            pos_bbox_targets = self.bbox_2d_coder.encode(pos_gt_bboxes)
            bbox_targets[pos_inds, :pos_bbox_targets.shape[1]] = pos_bbox_targets

            view_targets[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds, 1]

            if gt_labels is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds, 0]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

            center_targets[pos_inds, :] = normal_gt_centers[sampling_result.pos_assigned_gt_inds, :2]
            center_weights[pos_inds] = 1.0

            depth = gt_centers_2d[sampling_result.pos_assigned_gt_inds, 2]
            depth_labels[pos_inds] = depth
            depth_weights[pos_inds] = 1

            view_mask_ignore = view_targets != view
            bbox_weights[view_mask_ignore, :] = 0
            label_weights[view_mask_ignore] = 0

            if len(neg_inds) > 0:
                label_weights[neg_inds] = 1.0

            bbox_targets[:, :2] = center_targets
            bbox_targets[:, 2] = depth_labels

        # # compute dense heatmap targets
        if self.initialize_by_heatmap:
            device = labels.device
            feature_map_size = (img_shape[1] // self.out_size_factor_img, img_shape[0] // self.out_size_factor_img)

            w, h = feature_map_size
            heatmaps = []
            for lvl in range(self.level_num):
                heatmaps.append(score.new_zeros(self.num_classes, self.num_views, h, w))
                h = h // 2
                w = w // 2

            for idx in range(len(gt_bboxes)):
                width = gt_bboxes[idx][2]
                length = gt_bboxes[idx][3]

                max_l = max(length, width)

                width = width / self.out_size_factor_img
                length = length / self.out_size_factor_img
                view_id = gt_labels[idx][1]
                if width > 0 and length > 0:
                    radius = gaussian_radius((length, width), min_overlap=self.train_cfg['gaussian_overlap_2d'])
                    radius = max(self.train_cfg['min_radius'], radius)
                    radius = min(self.train_cfg['max_radius'], radius)

                    x, y = gt_centers_2d[idx][0], gt_centers_2d[idx][1]
                    # x, y = gt_bboxes[idx][0], gt_bboxes[idx][1]

                    coor_x = x / self.out_size_factor_img
                    coor_y = y / self.out_size_factor_img

                    center = torch.tensor([coor_x, coor_y], dtype=torch.float32, device=device)

                    if self.level_num == 4:
                        if max_l < 48:
                            lvl = 0
                        elif max_l < 96:
                            lvl = 1
                            center = center / 2
                            radius = radius / 2
                        elif max_l < 192:
                            lvl = 2
                            center = center / 4
                            radius = radius / 4
                        else:
                            lvl = 3
                            center = center / 8
                            radius = radius / 8
                    elif self.level_num == 3:
                        if max_l < 48:
                            lvl = 0
                        elif max_l < 96:
                            lvl = 1
                            center = center / 2
                            radius = radius / 2
                        else:
                            lvl = 2
                            center = center / 4
                            radius = radius / 4
                    elif self.level_num == 2:
                        if max_l < 96:
                            lvl = 0
                        else:
                            lvl = 1
                            center = center / 2
                            radius = radius / 2
                    else:
                        assert self.level_num == 1
                        lvl = 0

                    center_int = center.to(torch.int32)
                    radius = int(radius)

                    draw_heatmap_gaussian(heatmaps[lvl][gt_labels[idx][0], view_id], center_int, radius)

            for lvl in range(self.level_num):
                heatmaps[lvl] = heatmaps[lvl].view(self.num_classes, self.num_views, heatmaps[lvl].shape[-2]*heatmaps[lvl].shape[-1])
            heatmap = torch.cat(heatmaps, dim=-1)
            matched_ious = torch.ones_like(ious) * -1
            matched_ious[pos_inds] = ious[pos_inds]

            return labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], ious[None], pos_num_layers[None], matched_ious[None], heatmap[None], labels_lidar[None], label_lidar_weights[None], bbox_lidar_targets[None], bbox_lidar_weights[None], ious_lidar[None]
        else:
            matched_ious = torch.ones_like(ious) * -1
            matched_ious[pos_inds] = ious[pos_inds]
            return labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], ious[None], pos_num_layers[None], matched_ious[None], labels_lidar[None], label_lidar_weights[None], bbox_lidar_targets[None], bbox_lidar_weights[None], ious_lidar[None]

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, gt_visible, preds_dict, batch_idx):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.
            gt_bboxes (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes 2d.
            gt_labels (torch.Tensor): Labels of boxes 2d.
            preds_dict (dict): dict of prediction result for a single sample
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.

                - torch.Tensor: classification target.  [1, num_proposals]
                - torch.Tensor: classification weights (mask)  [1, num_proposals]
                - torch.Tensor: regression target. [1, num_proposals, 8]
                - torch.Tensor: regression weights. [1, num_proposals, 8]
                - torch.Tensor: iou target. [1, num_proposals]
                - int: number of positive proposals
        """
        num_proposals = preds_dict['center'].shape[-1]

        # get pred boxes, carefully ! donot change the network outputs
        score = copy.deepcopy(preds_dict['heatmap'].detach())
        center = copy.deepcopy(preds_dict['center'].detach())
        height = copy.deepcopy(preds_dict['height'].detach())
        dim = copy.deepcopy(preds_dict['dim'].detach())
        rot = copy.deepcopy(preds_dict['rot'].detach())
        if 'vel' in preds_dict.keys():
            vel = copy.deepcopy(preds_dict['vel'].detach())
        else:
            vel = None

        boxes_dict = self.bbox_coder.decode(score, rot, dim, center, height, vel)  # decode the prediction to real world metric bbox
        bboxes_tensor = boxes_dict[0]['bboxes']
        gt_bboxes_tensor = gt_bboxes_3d.tensor.to(score.device)

        num_fusion_decoder_layers = self.num_fusion_decoder_layers

        num_layer = self.num_pts_decoder_layers + num_fusion_decoder_layers

        start = 0
        pos_inds = []
        neg_inds = []
        pos_gt_bboxes = []
        pos_gt_labels = []
        ious = []
        for idx_layer in range(num_layer):
            layer_num_proposal = self.get_layer_num_proposal(idx_layer)

            bboxes_tensor_layer = bboxes_tensor[start:start + layer_num_proposal, :]
            score_layer = score[..., start:start + layer_num_proposal]

            gt_bboxes_tensor_layer = gt_bboxes_tensor
            gt_labels_3d_layer = gt_labels_3d

            if self.train_cfg.assigner.type == 'HungarianAssigner3D':
                assign_result = self.bbox_assigner.assign(bboxes_tensor_layer, gt_bboxes_tensor_layer, gt_labels_3d_layer, score_layer, self.train_cfg)
            elif self.train_cfg.assigner.type == 'HeuristicAssigner':
                assign_result = self.bbox_assigner.assign(bboxes_tensor_layer, gt_bboxes_tensor_layer, None, gt_labels_3d_layer, self.query_labels[batch_idx])
            else:
                raise NotImplementedError
            # assign_result_list.append(assign_result)

            sampling_result_layer = self.bbox_sampler.sample(assign_result, bboxes_tensor_layer, gt_bboxes_tensor_layer)
            pos_inds_layer = sampling_result_layer.pos_inds + start
            neg_inds_layer = sampling_result_layer.neg_inds + start

            pos_inds.append(pos_inds_layer)
            neg_inds.append(neg_inds_layer)

            pos_gt_bboxes_layer = sampling_result_layer.pos_gt_bboxes
            pos_gt_labels_layer = gt_labels_3d_layer[sampling_result_layer.pos_assigned_gt_inds]

            pos_gt_bboxes.append(pos_gt_bboxes_layer)
            pos_gt_labels.append(pos_gt_labels_layer)

            ious_layer = assign_result.max_overlaps
            ious.append(ious_layer)
            start += layer_num_proposal


        pos_inds = torch.cat(pos_inds)
        neg_inds = torch.cat(neg_inds)


        pos_gt_bboxes = torch.cat(pos_gt_bboxes, dim=0)
        pos_gt_labels = torch.cat(pos_gt_labels, dim=0)
        assert len(pos_inds) + len(neg_inds) == num_proposals

        start = 0
        pos_num_layers = []
        for idx_layer in range(num_layer):
            layer_num_proposal = self.get_layer_num_proposal(idx_layer)
            count = pos_inds[torch.logical_and(pos_inds>=start, pos_inds<start+layer_num_proposal)].shape[0]
            pos_num_layers.append(count)
            start += layer_num_proposal
        pos_num_layers = np.array(pos_num_layers)
        assert np.sum(pos_num_layers) == pos_inds.shape[0]

        # create target for loss computation
        bbox_targets = torch.zeros([num_proposals, self.bbox_coder.code_size]).to(center.device)
        bbox_weights = torch.zeros([num_proposals, self.bbox_coder.code_size]).to(center.device)
        ious = torch.cat(ious)
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
        label_weights = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)

        if gt_labels_3d is not None:  # default label is -1
            labels += self.num_classes

        # both pos and neg have classification loss, only pos has regression and iou loss
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = pos_gt_labels
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

        # # compute dense heatmap targets
        if self.initialize_by_heatmap:
            device = labels.device
            gt_bboxes_3d = torch.cat([gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]], dim=1).to(device)
            grid_size = torch.tensor(self.train_cfg['grid_size'])
            pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
            voxel_size = torch.tensor(self.train_cfg['voxel_size'])
            feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']  # [x_len, y_len]
            heatmap = gt_bboxes_3d.new_zeros(self.num_classes, feature_map_size[1], feature_map_size[0])
            for idx in range(len(gt_bboxes_3d)):
                width = gt_bboxes_3d[idx][3]
                length = gt_bboxes_3d[idx][4]
                width = width / voxel_size[0] / self.train_cfg['out_size_factor']
                length = length / voxel_size[1] / self.train_cfg['out_size_factor']
                if width > 0 and length > 0:
                    radius = gaussian_radius((length, width), min_overlap=self.train_cfg['gaussian_overlap'])
                    radius = max(self.train_cfg['min_radius'], int(radius))
                    x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                    coor_x = (x - pc_range[0]) / voxel_size[0] / self.train_cfg['out_size_factor']
                    coor_y = (y - pc_range[1]) / voxel_size[1] / self.train_cfg['out_size_factor']

                    center_img = torch.tensor([coor_x, coor_y], dtype=torch.float32, device=device)
                    center_int = center_img.to(torch.int32)
                    draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]], center_int, radius)

            matched_ious = torch.ones_like(ious) * -1
            matched_ious[pos_inds] = ious[pos_inds]
            return labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], ious[None], pos_num_layers[None], matched_ious[None], heatmap[None]
        else:
            matched_ious = torch.ones_like(ious) * -1
            matched_ious[pos_inds] = ious[pos_inds]
            return labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], ious[None], pos_num_layers[None], matched_ious[None]

    def get_targets_single_view(self, gt_bboxes_3d, gt_labels_3d, gt_visible_3d, preds_dict, batch_idx):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.
            gt_bboxes (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes 2d.
            gt_labels (torch.Tensor): Labels of boxes 2d.
            preds_dict (dict): dict of prediction result for a single sample
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.

                - torch.Tensor: classification target.  [1, num_proposals]
                - torch.Tensor: classification weights (mask)  [1, num_proposals]
                - torch.Tensor: regression target. [1, num_proposals, 8]
                - torch.Tensor: regression weights. [1, num_proposals, 8]
                - torch.Tensor: iou target. [1, num_proposals]
                - int: number of positive proposals
        """
        num_proposals = preds_dict['center_view'].shape[-1]

        # get pred boxes, carefully ! donot change the network outputs
        score = copy.deepcopy(preds_dict['heatmap_view'].detach())
        center = copy.deepcopy(preds_dict['center_view'].detach())
        height = copy.deepcopy(preds_dict['height_view'].detach())
        dim = copy.deepcopy(preds_dict['dim_view'].detach())
        rot = copy.deepcopy(preds_dict['rot_view'].detach())
        if 'vel_view' in preds_dict.keys():
            vel = copy.deepcopy(preds_dict['vel_view'].detach())
        else:
            vel = None

        boxes_dict = self.bbox_coder.decode(score, rot, dim, center, height, vel)  # decode the prediction to real world metric bbox
        bboxes_tensor = boxes_dict[0]['bboxes']

        assert gt_visible_3d.shape[0] == gt_bboxes_3d.tensor.shape[0] == gt_labels_3d.shape[0]
        gt_mask = gt_visible_3d == 1
        gt_bboxes_3d = gt_bboxes_3d[gt_mask]
        gt_labels_3d = gt_labels_3d[gt_mask]
        gt_bboxes_tensor = gt_bboxes_3d.tensor.to(score.device)

        num_layer = 1
        assign_result_list = []
        start = 0
        for idx_layer in range(num_layer):
            layer_num_proposal = self.get_layer_num_proposal(idx_layer)

            bboxes_tensor_layer = bboxes_tensor[start:start + layer_num_proposal, :]
            score_layer = score[..., start:start + layer_num_proposal]
            start += layer_num_proposal

            if self.train_cfg.assigner.type == 'HungarianAssigner3D':
                assign_result = self.bbox_assigner.assign(bboxes_tensor_layer, gt_bboxes_tensor, gt_labels_3d, score_layer, self.train_cfg)
            elif self.train_cfg.assigner.type == 'HeuristicAssigner':
                assign_result = self.bbox_assigner.assign(bboxes_tensor_layer, gt_bboxes_tensor, None, gt_labels_3d, self.query_labels[batch_idx])
            else:
                raise NotImplementedError
            assign_result_list.append(assign_result)

        # combine assign result of each layer
        assign_result_ensemble = AssignResult(
            num_gts=sum([res.num_gts for res in assign_result_list]),
            gt_inds=torch.cat([res.gt_inds for res in assign_result_list]),
            max_overlaps=torch.cat([res.max_overlaps for res in assign_result_list]),
            labels=torch.cat([res.labels for res in assign_result_list]),
        )
        sampling_result = self.bbox_sampler.sample(assign_result_ensemble, bboxes_tensor, gt_bboxes_tensor)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        assert len(pos_inds) + len(neg_inds) == num_proposals

        start = 0
        pos_num_layers = []
        for idx_layer in range(num_layer):
            layer_num_proposal = self.get_layer_num_proposal(idx_layer)
            count = pos_inds[torch.logical_and(pos_inds>=start, pos_inds<start+layer_num_proposal)].shape[0]
            pos_num_layers.append(count)
            start += layer_num_proposal
        pos_num_layers = np.array(pos_num_layers)
        assert np.sum(pos_num_layers) == pos_inds.shape[0]

        # create target for loss computation
        bbox_targets = torch.zeros([num_proposals, self.bbox_coder.code_size]).to(center.device)
        bbox_weights = torch.zeros([num_proposals, self.bbox_coder.code_size]).to(center.device)
        ious = assign_result_ensemble.max_overlaps
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
        label_weights = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)

        if gt_labels_3d is not None:  # default label is -1
            labels += self.num_classes

        # both pos and neg have classification loss, only pos has regression and iou loss
        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels_3d[sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        matched_ious = torch.ones_like(ious) * -1
        matched_ious[pos_inds] = ious[pos_inds]

        return labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], ious[None], pos_num_layers[None], matched_ious[None]

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels, gt_pts_centers_view, gt_img_centers_view, gt_bboxes_cam_view, gt_visible_3d, gt_bboxes_lidar_view, img_metas, preds_dicts, **kwargs):
        """Loss function for CenterHead.

        Args:
            **The followings are in the same order of "gt_bboxes_3d" :**
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            gt_visible_3d (list[torch.Tensor]): visibility of LiDAR boxes for camera

            **The followings are in the same order of "gt_bboxes":**
            gt_bboxes (list[torch.Tensor]): Ground truth of projected 2d boxes.
            (one LiDAR box may be projected to zero/one/two camera views, so "gt_bboxes" has different number with "gt_bboxes_3d")
            gt_labels (list[torch.Tensor]): Labels and camera view ids of projected 2d boxes.
            gt_pts_centers_view (list[torch.Tensor]): 3D center of each boxes in the LiDAR coordinate
            gt_img_centers_view (list[torch.Tensor]): 3D center of each boxes in the corresponding camera coordinate
            gt_bboxes_cam_view (list[:obj:`CameraInstance3DBoxes`]): ground truth boxes in the corresponding camera coordinate
            gt_bboxes_lidar_view (list[:obj:`LiDARInstance3DBoxes`]): ground truth boxes in the LiDAR coordinate

            preds_dicts (list[list[dict]]): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        if self.initialize_by_heatmap:
            if self.view_transform:
                labels, label_weights, bbox_targets, bbox_weights, ious, num_pos_layer, matched_ious, heatmap, \
                labels_2d, label_weights_2d, bbox_targets_2d, bbox_weights_2d, ious_2d, num_pos_layer_2d, \
                matched_ious_2d, heatmap_2d, labels_view, label_weights_view, bbox_targets_view, bbox_weights_view, ious_view, \
                num_pos_layer_view, matched_ious_view = self.get_targets(gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels, gt_img_centers_view, gt_bboxes_cam_view, gt_visible_3d, gt_bboxes_lidar_view, preds_dicts[0], img_metas)
            else:
                labels, label_weights, bbox_targets, bbox_weights, ious, num_pos_layer, matched_ious, heatmap, \
                labels_2d, label_weights_2d, bbox_targets_2d, bbox_weights_2d, ious_2d, num_pos_layer_2d, \
                matched_ious_2d, heatmap_2d = self.get_targets(gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels, gt_img_centers_view, gt_bboxes_cam_view, gt_visible_3d, gt_bboxes_lidar_view, preds_dicts[0], img_metas)
        else:
            if self.view_transform:
                labels, label_weights, bbox_targets, bbox_weights, ious, num_pos_layer, matched_ious, \
                labels_2d, label_weights_2d, bbox_targets_2d, bbox_weights_2d, ious_2d, num_pos_layer_2d, \
                matched_ious_2d, labels_view, label_weights_view, bbox_targets_view, bbox_weights_view, ious_view, \
                num_pos_layer_view, matched_ious_view = self.get_targets(gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels, gt_img_centers_view, gt_bboxes_cam_view, gt_visible_3d, preds_dicts[0], img_metas)
            else:
                labels, label_weights, bbox_targets, bbox_weights, ious, num_pos_layer, matched_ious, \
                labels_2d, label_weights_2d, bbox_targets_2d, bbox_weights_2d, ious_2d, num_pos_layer_2d, matched_ious_2d = \
                    self.get_targets(gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels, gt_img_centers_view, gt_bboxes_cam_view, gt_visible_3d, preds_dicts[0], img_metas)        # if hasattr(self, 'on_the_image_mask'):

        preds_dict = preds_dicts[0][0]
        loss_dict = dict()

        if self.initialize_by_heatmap:
            # compute heatmap loss
            loss_heatmap = self.loss_heatmap(clip_sigmoid(preds_dict['dense_heatmap']), heatmap, avg_factor=max(heatmap.eq(1).float().sum().item(), 1))
            if 'valid_shape' in img_metas[0].keys():
                bs = heatmap_2d.shape[0]
                num_view = heatmap_2d.shape[2]
                # heatmap_2d_weight = torch.zeros_like(heatmap_2d)
                heatmaps_2d_weight = []
                img_w, img_h = self.test_cfg['img_scale']
                img_w = img_w // self.out_size_factor_img
                img_h = img_h // self.out_size_factor_img
                for lvl in range(self.level_num):
                    heatmap_2d_weight = torch.zeros(heatmap_2d.shape[0], self.num_classes, self.num_views, img_h, img_w).to(heatmap_2d.device)
                    heatmaps_2d_weight.append(heatmap_2d_weight)
                    img_h = img_h // 2
                    img_w = img_w // 2

                for sample_idx in range(bs):
                    for view_idx in range(num_view):
                        valid_shape = img_metas[sample_idx]['valid_shape'][view_idx] / self.out_size_factor_img
                        red_width = int(valid_shape[0])
                        red_height = int(valid_shape[1])
                        for lvl in range(self.level_num):
                            heatmaps_2d_weight[lvl][sample_idx, :, view_idx, :red_height, :red_width] = 1
                            red_width = red_width // 2
                            red_height = red_height // 2

                for lvl in range(self.level_num):
                    heatmaps_2d_weight[lvl] = heatmaps_2d_weight[lvl].view(heatmaps_2d_weight[lvl].shape[0], self.num_classes, self.num_views, heatmaps_2d_weight[lvl].shape[-2]*heatmaps_2d_weight[lvl].shape[-1])
                heatmap_2d_weight = torch.cat(heatmaps_2d_weight, dim=-1)

                loss_heatmap_2d = self.loss_heatmap_2d(clip_sigmoid(preds_dict['img_dense_heatmap']), heatmap_2d, weight=heatmap_2d_weight, avg_factor=max(heatmap_2d.eq(1).float().sum().item(), 1))
            else:
                loss_heatmap_2d = self.loss_heatmap_2d(clip_sigmoid(preds_dict['img_dense_heatmap']), heatmap_2d, avg_factor=max(heatmap_2d.eq(1).float().sum().item(), 1))

            loss_dict['loss_heatmap'] = loss_heatmap
            loss_dict['loss_heatmap_2d'] = loss_heatmap_2d

        # compute loss for each layer
        start = 0
        num_pos_layer = np.sum(num_pos_layer, axis=0)
        num_pos_layer_2d = np.sum(num_pos_layer_2d, axis=0)
        if self.view_transform:
            num_pos_layer_view = np.sum(num_pos_layer_view, axis=0)

        num_fusion_decoder_layers = self.num_fusion_decoder_layers

        num_layer = self.num_pts_decoder_layers + num_fusion_decoder_layers
        for idx_layer in range(num_layer):
            layer_num_proposals = self.get_layer_num_proposal(idx_layer)
            if idx_layer < self.num_pts_decoder_layers:
                prefix = f'layer_pts_{idx_layer}'
            else:
                prefix = f'layer_fusion_{idx_layer-self.num_pts_decoder_layers}'

            layer_labels = labels[..., start:start + layer_num_proposals].reshape(-1)
            layer_label_weights = label_weights[..., start:start + layer_num_proposals].reshape(-1)
            layer_score = preds_dict['heatmap'][..., start:start + layer_num_proposals]
            layer_cls_score = layer_score.permute(0, 2, 1).reshape(-1, self.num_classes)
            layer_loss_cls = self.loss_cls(layer_cls_score, layer_labels, layer_label_weights, avg_factor=max(num_pos_layer[idx_layer], 1))

            layer_center = preds_dict['center'][..., start:start + layer_num_proposals]
            layer_height = preds_dict['height'][..., start:start + layer_num_proposals]
            layer_rot = preds_dict['rot'][..., start:start + layer_num_proposals]
            layer_dim = preds_dict['dim'][..., start:start + layer_num_proposals]
            preds = torch.cat([layer_center, layer_height, layer_dim, layer_rot], dim=1).permute(0, 2, 1)  # [BS, num_proposals, code_size]
            if 'vel' in preds_dict.keys():
                layer_vel = preds_dict['vel'][..., start:start + layer_num_proposals]
                preds = torch.cat([layer_center, layer_height, layer_dim, layer_rot, layer_vel], dim=1).permute(0, 2, 1)  # [BS, num_proposals, code_size]
            code_weights = self.train_cfg.get('code_weights', None)
            layer_bbox_weights = bbox_weights[:, start:start + layer_num_proposals, :]
            layer_reg_weights = layer_bbox_weights * layer_bbox_weights.new_tensor(code_weights)
            layer_bbox_targets = bbox_targets[:, start:start + layer_num_proposals, :]
            layer_loss_bbox = self.loss_bbox(preds, layer_bbox_targets, layer_reg_weights, avg_factor=max(num_pos_layer[idx_layer], 1))

            layer_match_ious = matched_ious[..., start:start + layer_num_proposals]
            layer_match_ious = torch.sum(layer_match_ious*(layer_match_ious>=0), dim=-1) / torch.sum(layer_match_ious>=0, dim=-1)
            layer_match_ious = torch.mean(layer_match_ious)
            start += layer_num_proposals

            loss_dict[f'{prefix}_loss_cls'] = layer_loss_cls
            loss_dict[f'{prefix}_loss_bbox'] = layer_loss_bbox
            loss_dict[f'{prefix}_matched_ious'] = layer_match_ious

        start = 0
        for idx_layer in range(self.num_img_decoder_layers):
            prefix = f'layer_img_{idx_layer}'
            layer_num_proposals = self.num_img_proposals
            layer_labels_2d = labels_2d[..., start:start + layer_num_proposals].reshape(-1)
            layer_label_weights_2d = label_weights_2d[..., start:start + layer_num_proposals].reshape(-1)
            layer_score_2d = preds_dict['cls'][..., start:start + layer_num_proposals]
            layer_cls_score_2d = layer_score_2d.permute(0, 2, 1).reshape(-1, self.num_classes)
            layer_loss_cls_2d = self.loss_cls(layer_cls_score_2d, layer_labels_2d, layer_label_weights_2d, avg_factor=max(num_pos_layer_2d[idx_layer], 1))
            preds_2d_center = preds_dict['center_2d'][..., start:start + layer_num_proposals]  # [bs, 2, num_proposal]
            preds_2d_depth = preds_dict['depth_2d'][..., start:start + layer_num_proposals]  # [bs, 1, num_proposal]
            preds_2d_dim = preds_dict['dim_2d'][..., start:start + layer_num_proposals]  # [bs, 3, num_proposal]
            preds_2d_rot = preds_dict['rot_2d'][..., start:start + layer_num_proposals]  # [bs, 2, num_proposal]
            preds_2d_vel = preds_dict['vel_2d'][..., start:start + layer_num_proposals]  # [bs, 2, num_proposal]
            preds_2d = torch.cat([preds_2d_center, preds_2d_depth[:, :1], preds_2d_dim, preds_2d_rot, preds_2d_vel], dim=1).permute(0, 2, 1)  # [bs, num_proposal, 10]
            layer_bbox_targets_2d = bbox_targets_2d[:, start:start + layer_num_proposals, :preds_2d.shape[2]]
            layer_reg_weights_2d = bbox_weights_2d[:, start:start + layer_num_proposals, :preds_2d.shape[2]]
            code_weights = self.train_cfg.get('img_code_weights', None)
            layer_reg_weights_2d = layer_reg_weights_2d * layer_reg_weights_2d.new_tensor(code_weights)
            layer_loss_center_2d = self.loss_center_2d(preds_2d[...,:2], layer_bbox_targets_2d[...,:2], layer_reg_weights_2d[...,:2], avg_factor=max(num_pos_layer_2d[idx_layer], 1))

            layer_loss_depth_2d = self.loss_bbox(preds_2d[...,2:3], layer_bbox_targets_2d[...,2:3], layer_reg_weights_2d[...,2:3], avg_factor=max(num_pos_layer_2d[idx_layer], 1))
            layer_loss_dim_2d = self.loss_bbox(preds_2d[...,3:6], layer_bbox_targets_2d[...,3:6], layer_reg_weights_2d[...,3:6], avg_factor=max(num_pos_layer_2d[idx_layer], 1))
            layer_loss_rot_2d = self.loss_bbox(preds_2d[...,6:8], layer_bbox_targets_2d[...,6:8], layer_reg_weights_2d[...,6:8], avg_factor=max(num_pos_layer_2d[idx_layer], 1))
            layer_match_ious_2d = matched_ious_2d[..., start:start + layer_num_proposals]
            layer_match_ious_2d = torch.sum(layer_match_ious_2d*(layer_match_ious_2d>=0), dim=-1) / (torch.sum(layer_match_ious_2d>=0, dim=-1) + 1e-2)
            layer_match_ious_2d = torch.mean(layer_match_ious_2d)
            start += layer_num_proposals
            loss_dict[f'{prefix}_loss_cls_2d'] = layer_loss_cls_2d
            loss_dict[f'{prefix}_loss_center_2d'] = layer_loss_center_2d
            loss_dict[f'{prefix}_loss_depth_2d'] = layer_loss_depth_2d
            loss_dict[f'{prefix}_loss_dim_2d'] = layer_loss_dim_2d
            loss_dict[f'{prefix}_loss_rot_2d'] = layer_loss_rot_2d
            if preds_2d.shape[-1] > 8:
                layer_loss_vel_2d = self.loss_bbox(preds_2d[...,8:10], layer_bbox_targets_2d[...,8:10], layer_reg_weights_2d[...,8:10], avg_factor=max(num_pos_layer_2d[idx_layer], 1))
                loss_dict[f'{prefix}_loss_vel_2d'] = layer_loss_vel_2d
            else:
                layer_loss_vel_2d = 0
            loss_dict[f'{prefix}_matched_ious_2d'] = layer_match_ious_2d
            loss_dict[f'{prefix}_reg_bbox_2d'] = (layer_loss_center_2d+layer_loss_depth_2d+layer_loss_dim_2d+layer_loss_rot_2d+layer_loss_vel_2d).detach()
        if self.view_transform:
            layer_labels_view = labels_view.reshape(-1)
            layer_label_weights_view = label_weights_view.reshape(-1)
            layer_cls_score = preds_dict['heatmap_view'].permute(0, 2, 1).reshape(-1, self.num_classes)
            layer_loss_cls_view = self.loss_cls(
                layer_cls_score, layer_labels_view, layer_label_weights_view, avg_factor=max(num_pos_layer_view[0], 1)
            )
            layer_center_view = preds_dict['center_view']
            layer_height_view = preds_dict['height_view']
            layer_rot_view = preds_dict['rot_view']
            layer_dim_view = preds_dict['dim_view']
            preds_view = torch.cat([layer_center_view, layer_height_view, layer_dim_view, layer_rot_view],
                                   dim=1).permute(0, 2, 1)  # [BS, num_proposals, code_size]
            if 'vel' in preds_dict.keys():
                layer_vel_view = preds_dict['vel_view']
                preds_view = torch.cat([layer_center_view, layer_height_view, layer_dim_view, layer_rot_view, layer_vel_view],
                                  dim=1).permute(0, 2, 1)  # [BS, num_proposals, code_size]
            code_weights = self.train_cfg.get('code_weights', None)
            layer_reg_weights_view = bbox_weights_view * bbox_weights_view.new_tensor(code_weights)
            layer_loss_bbox_view = self.loss_bbox(preds_view, bbox_targets_view, layer_reg_weights_view, avg_factor=max(num_pos_layer_view[0], 1))

            layer_match_ious_view = matched_ious_view
            layer_match_ious_view = torch.sum(layer_match_ious_view * (layer_match_ious_view >= 0), dim=-1) / torch.sum(
                layer_match_ious_view >= 0, dim=-1)
            layer_match_ious_view = torch.mean(layer_match_ious_view)
            loss_dict['view_loss_cls'] = layer_loss_cls_view

            loss_dict['view_loss_bbox'] = layer_loss_bbox_view
            loss_dict['view_matched_ious'] = layer_match_ious_view

        return loss_dict

    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False, for_roi=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.

        Returns:
            list[list[dict]]: Decoded bbox, scores and labels for each layer & each batch
        """
        rets = []
        for id, preds_dict in enumerate(preds_dicts):
            layer_num_proposal = self.num_proposals + self.num_img_proposals
            batch_size = preds_dict[0]['heatmap'].shape[0]

            batch_score_raw = preds_dict[0]['heatmap'][..., -layer_num_proposal:].sigmoid()

            one_hot = F.one_hot(self.query_labels, num_classes=self.num_classes).permute(0, 2, 1)
            query_heatmap_score = preds_dict[0]['query_heatmap_score'] * one_hot
            one_hot_img = F.one_hot(self.img_query_label, num_classes=self.num_classes).permute(0, 2, 1)
            img_query_label_decoder = torch.max(preds_dict[0]['cls'], dim=1)[1]
            one_hot_img_decoder = F.one_hot(img_query_label_decoder, num_classes=self.num_classes).permute(0, 2, 1)
            img_query_heatmap_score = preds_dict[0]['img_query_heatmap_score'] * one_hot_img * one_hot_img_decoder * 0.5
            query_heatmap_score = torch.cat([query_heatmap_score, img_query_heatmap_score], dim=2)


            batch_score = batch_score_raw * query_heatmap_score

            batch_center = preds_dict[0]['center'][..., -layer_num_proposal:]
            batch_height = preds_dict[0]['height'][..., -layer_num_proposal:]
            batch_dim = preds_dict[0]['dim'][..., -layer_num_proposal:]
            batch_rot = preds_dict[0]['rot'][..., -layer_num_proposal:]

            batch_vel = None
            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel'][..., -layer_num_proposal:]

            temp = self.bbox_coder.decode(batch_score, batch_rot, batch_dim, batch_center, batch_height, batch_vel, filter=True)

            if self.test_cfg['dataset'] == 'nuScenes':
                self.tasks = [
                    dict(num_class=1, class_names=['car'], indices=[0], radius=0.35),
                    dict(num_class=1, class_names=['truck'], indices=[1], radius=0.35),
                    dict(num_class=1, class_names=['construction_vehicle'], indices=[2], radius=0.35),
                    dict(num_class=1, class_names=['bus'], indices=[3], radius=0.35),
                    dict(num_class=1, class_names=['trailer'], indices=[4], radius=0.35),
                    dict(num_class=1, class_names=['barrier'], indices=[5], radius=0.175),
                    dict(num_class=1, class_names=['motorcycle'], indices=[6], radius=0.1),
                    dict(num_class=1, class_names=['bicycle'], indices=[7], radius=-1),

                    dict(num_class=1, class_names=['pedestrian'], indices=[8], radius=0.1),
                    dict(num_class=1, class_names=['traffic_cone'], indices=[9], radius=0.1),
                ]

                # self.tasks = [
                #     dict(num_class=8, class_names=[], indices=[0, 1, 2, 3, 4, 5, 6, 7], radius=-1),
                #     dict(num_class=1, class_names=['pedestrian'], indices=[8], radius=0.175),
                #     dict(num_class=1, class_names=['traffic_cone'], indices=[9], radius=0.175),
                # ]
            elif self.test_cfg['dataset'] == 'Waymo':
                self.tasks = [
                    dict(num_class=1, class_names=['Car'], indices=[0], radius=0.7),
                    dict(num_class=1, class_names=['Pedestrian'], indices=[1], radius=0.7),
                    dict(num_class=1, class_names=['Cyclist'], indices=[2], radius=0.7),
                ]

            ret_layer = []
            for i in range(batch_size):
                boxes3d = temp[i]['bboxes']
                scores = temp[i]['scores']
                labels = temp[i]['labels']

                ## adopt circle nms for different categories
                if self.test_cfg['nms_type'] != None:
                    keep_mask = torch.zeros_like(scores)
                    for task in self.tasks:
                        task_mask = torch.zeros_like(scores)
                        for cls_idx in task['indices']:
                            task_mask += labels == cls_idx
                        task_mask = task_mask.bool()
                        if task['radius'] > 0 and task_mask.sum() > 0:
                            if self.test_cfg['nms_type'] == 'circle':
                                boxes_for_nms = torch.cat([boxes3d[task_mask][:, :2], scores[:, None][task_mask]], dim=1)
                                task_keep_indices = torch.tensor(
                                    circle_nms(
                                        boxes_for_nms.detach().cpu().numpy(),
                                        task['radius'],
                                        # 5,
                                        post_max_size=500
                                    )
                                )
                            else:
                                boxes_for_nms = xywhr2xyxyr(img_metas[i]['box_type_3d'](boxes3d[task_mask][:, :7], 7).bev)
                                top_scores = scores[task_mask]

                                task_keep_indices = nms_gpu(
                                    boxes_for_nms,
                                    top_scores,
                                    thresh=task['radius'],
                                    # pre_maxsize=self.test_cfg['pre_maxsize'],
                                    # post_max_size=self.test_cfg['post_maxsize'],
                                )
                        else:
                            task_keep_indices = torch.arange(task_mask.sum())
                        if task_keep_indices.shape[0] != 0:
                            keep_indices = torch.where(task_mask != 0)[0][task_keep_indices]
                            keep_mask[keep_indices] = 1
                    keep_mask = keep_mask.bool()
                    ret = dict(bboxes=boxes3d[keep_mask], scores=scores[keep_mask], labels=labels[keep_mask])
                else:  # no nms
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                ret_layer.append(ret)
            rets.append(ret_layer)
        assert len(rets) == 1
        assert len(rets[0]) == 1

        res = [[
            img_metas[0]['box_type_3d'](rets[0][0]['bboxes'], box_dim=rets[0][0]['bboxes'].shape[-1]),
            rets[0][0]['scores'],
            rets[0][0]['labels'].int()
        ]]
        return res

    def get_layer_num_proposal(self, idx_layer):
        if idx_layer >= self.num_pts_decoder_layers:
            layer_num_proposal = self.num_proposals + self.num_img_proposals
        else:
            layer_num_proposal = self.num_proposals

        return layer_num_proposal