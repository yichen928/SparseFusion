import copy
import numpy as np
import torch
import functools
import pickle
from mmcv.cnn import ConvModule, build_conv_layer, kaiming_init
from mmcv.runner import force_fp32
from torch import nn
import torch.nn.functional as F
import time

from mmdet3d.core import (circle_nms, draw_heatmap_gaussian, gaussian_radius,
                          xywhr2xyxyr, limit_period, PseudoSampler)
from mmdet3d.models.builder import HEADS, build_loss
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu
from mmdet3d.models.utils import clip_sigmoid, inverse_sigmoid
from mmdet3d.models.fusion_layers import apply_3d_transformation
from mmdet.core import build_bbox_coder, multi_apply, build_assigner, build_sampler, AssignResult

from mmdet3d.models.utils import FFN, TransformerDecoderLayer, PositionEmbeddingLearned, PositionEmbeddingLearnedwoNorm, \
    PositionEmbeddingLearnedMultiInput, PointTransformer2D_3D, ImageTransformer2D_3D_Cross, FusionTransformer2D_3D_Cross, \
    DepthTransformer2D_3D, PointProjection, ImageProjection, ProjectionL2Norm, ProjectionLayerNorm, MSCABlock, \
    ImageTransformer2D_3D_Cross_Proj, MSFusion, FusionTransformer2D_3D_Self, PositionEmbeddingLearnedLN, ASPP


def denormalize_pos(normal_pos, x_max, y_max):
    max_xy = torch.Tensor([x_max, y_max]).to(normal_pos.device).view(1, 1, 2)
    pos = normal_pos.sigmoid() * max_xy
    return pos


def normalize_pos(pos, x_max, y_max):
    max_xy = torch.Tensor([x_max, y_max]).to(pos.device).view(1, 1, 2)
    normal_pos = pos / max_xy
    return inverse_sigmoid(normal_pos)

@HEADS.register_module()
class ImplicitHead2D_3D_Transfer(nn.Module):
    def __init__(self,
                 with_pts=False,
                 with_img=False,
                 pts_smca=False,
                 img_smca=False,
                 cross_smca=False,
                 num_views=0,
                 in_channels_img=64,
                 out_size_factor_img=4,
                 num_proposals=128,
                 num_img_proposals=128,
                 in_channels=128 * 3,
                 hidden_channel=128,
                 num_classes=4,
                 # config for Transformer
                 num_pts_decoder_layers=3,
                 num_img_decoder_layers=3,
                 num_fusion_decoder_layers=3,
                 # num_projection_layers=2,
                 num_heads=8,
                 initialize_by_heatmap=False,
                 cross_heatmap=False,
                 cross_only=False,
                 colattn_heatmap=False,
                 colattn_pos=False,
                 cat_point=True,
                 fuse_bev_feature=False,
                 img_heatmap_dcn=False,
                 cross_heatmap_dcn=False,
                 cross_heatmap_layer=2,
                 img_heatmap_layer=3,
                 need_reduce_conv=False,
                 img_reg_layer=2,
                 img_reg_bn=True,
                 learnable_query_pos=True,
                 nms_kernel_size=3,
                 img_nms_kernel_size=5,
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
                 loss_iou=dict(type='VarifocalLoss', use_sigmoid=True, iou_weighted=True, reduction='mean'),
                 loss_bbox=dict(type='L1Loss', reduction='mean'),
                 loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_heatmap_2d=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_cls_2d=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_bbox_2d=dict(type='L1Loss', reduction='mean'),
                 loss_depth=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_center_2d=dict(type='L1Loss', reduction='mean'),
                 loss_depth_reg=dict(type='L1Loss', reduction='mean'),
                 # others
                 train_cfg=None,
                 test_cfg=None,
                 bbox_coder=None,
                 projection='v1',
                 supervision2d=True,
                 supervision3d=True,
                 stop_grad=False,
                 pe_norm='bn',
                 pos_3d=False,
                 depth_key_3d=False,
                 depth_reg=False,
                 use_camera=None,
                 extra_camera=False,
                 fuse_projection=True,
                 extra_class_sup=False,
                 dbound=[2, 65, 0.5],
                 img_skip_connect=False,
                 cross_heatmap_stop_grad=False,
                 img_heatmap_stop_grad=False,
                 depth_stop_grad=False,
                 cross_heatmap_trick='none',
                 fuse_self=False,
                 merge_head_2d=False,
                 ):
        super(ImplicitHead2D_3D_Transfer, self).__init__()

        self.with_img = with_img
        self.with_pts = with_pts
        self.num_proposals = num_proposals
        self.num_img_proposals = num_img_proposals
        self.num_classes = num_classes
        self.bbox_coder = build_bbox_coder(bbox_coder)

        self.bn_momentum = bn_momentum
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.initialize_by_heatmap = initialize_by_heatmap
        self.cross_heatmap = cross_heatmap
        self.cross_only = cross_only
        self.colattn_heatmap = colattn_heatmap
        self.colattn_pos = colattn_pos
        self.fuse_bev_feature = fuse_bev_feature
        self.extra_class_sup = extra_class_sup
        self.depth_stop_grad = depth_stop_grad
        self.depth_key_3d = depth_key_3d
        self.cat_point = cat_point

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)
        self.loss_heatmap = build_loss(loss_heatmap)
        self.loss_heatmap_2d = build_loss(loss_heatmap_2d)
        self.loss_cls_2d = build_loss(loss_cls_2d)
        self.loss_bbox_2d = build_loss(loss_bbox_2d)
        if depth_reg:
            self.loss_depth_reg = build_loss(loss_depth_reg)
        else:
            self.loss_depth = build_loss(loss_depth)
        self.loss_center_2d = build_loss(loss_center_2d)
        if self.initialize_by_heatmap is True:
            assert learnable_query_pos is False, "initialized by heatmap is conflicting with learnable query position"

        self.num_img_decoder_layers = num_img_decoder_layers * with_img
        self.num_pts_decoder_layers = num_pts_decoder_layers * with_pts
        self.num_fusion_decoder_layers = num_fusion_decoder_layers * with_img * with_pts
        # self.num_projection_layers = num_projection_layers
        self.hidden_channel = hidden_channel
        self.sampling = False
        self.out_size_factor_img = out_size_factor_img
        self.supervision2d = supervision2d
        self.supervision3d = supervision3d
        self.stop_grad = stop_grad
        self.pe_norm = pe_norm
        self.depth_reg = depth_reg
        if not depth_reg:
            self.dbound = dbound
            self.num_depth_class = int((dbound[1] - dbound[0]) / dbound[2]) + 1
        else:
            dbound = None
            self.dbound = None
            self.num_depth_class = None
        self.fuse_projection = fuse_projection
        self.img_skip_connect = img_skip_connect
        self.cross_heatmap_stop_grad = cross_heatmap_stop_grad
        self.img_heatmap_stop_grad = img_heatmap_stop_grad
        self.cross_heatmap_trick = cross_heatmap_trick
        self.pos_3d = pos_3d
        self.fuse_self = fuse_self

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1

        heads3d = copy.deepcopy(common_heads)
        heads3d.update(dict(heatmap=(self.num_classes, 2)))
        pts_prediction_heads = FFN(hidden_channel, heads3d, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias)
        if fuse_self:
            fusion_prediction_heads = FFN(hidden_channel, heads3d, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias)
        else:
            fusion_prediction_heads = FFN(hidden_channel*2, heads3d, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias)

        if merge_head_2d:
            heads2d = dict(center_offset=(6, img_reg_layer, img_reg_bn), cls=(self.num_classes, 2))
        else:
            heads2d = dict(center_2d=(2, img_reg_layer, img_reg_bn), offset=(4, img_reg_layer, img_reg_bn), cls=(self.num_classes, 2))

        img_prediction_heads = FFN(hidden_channel, heads2d, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias)

        if self.depth_reg:
            heads_depth = dict(proj_center=(2, img_reg_layer, img_reg_bn), depth=(1, img_reg_layer, img_reg_bn))
        else:
            heads_depth = dict(proj_center=(2, img_reg_layer, img_reg_bn), depth=(self.num_depth_class, img_reg_layer))
        if extra_class_sup:
            heads_depth.update(extra_cls=(self.num_classes, 2))
        depth_prediction_heads = FFN(hidden_channel, heads_depth, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias)

        if with_pts:
            pts_query_pos_embed = [PositionEmbeddingLearned(2, hidden_channel) for _ in range(num_pts_decoder_layers)]
            pts_key_pos_embed = [PositionEmbeddingLearned(2, hidden_channel) for _ in range(num_pts_decoder_layers)]
            self.point_transformer = PointTransformer2D_3D(
                pts_smca=pts_smca, hidden_channel=hidden_channel, num_heads=num_heads, num_decoder_layers=num_pts_decoder_layers,
                prediction_heads=pts_prediction_heads, ffn_channel=ffn_channel, dropout=dropout, activation=activation, test_cfg=test_cfg,
                query_pos=pts_query_pos_embed, key_pos=pts_key_pos_embed, supervision3d=supervision3d
            )
        if with_img:
            assert learnable_query_pos or with_pts
            if self.pe_norm == 'bn':
                img_query_pos_embed = [PositionEmbeddingLearned(2, hidden_channel) for _ in range(num_img_decoder_layers)]
                img_key_pos_embed = [PositionEmbeddingLearned(2, hidden_channel) for _ in range(num_img_decoder_layers)]
            else:
                img_query_pos_embed = [PositionEmbeddingLearnedwoNorm(2, hidden_channel) for _ in range(num_img_decoder_layers)]
                img_key_pos_embed = [PositionEmbeddingLearnedwoNorm(2, hidden_channel) for _ in range(num_img_decoder_layers)]
            self.img_transformer = ImageTransformer2D_3D_Cross_Proj(
                hidden_channel=hidden_channel, num_heads=num_heads, num_decoder_layers=num_img_decoder_layers, out_size_factor_img=out_size_factor_img,
                num_views=num_views, prediction_heads=img_prediction_heads, ffn_channel=ffn_channel, dropout=dropout, activation=activation, test_cfg=test_cfg,
                query_pos=img_query_pos_embed, key_pos=img_key_pos_embed, supervision2d=supervision2d
            )
        if with_pts and with_img:
            if self.pe_norm == 'bn':
                depth_query_pos_embed = PositionEmbeddingLearned(2, hidden_channel)
                if depth_key_3d:
                    depth_key_pos_embed = PositionEmbeddingLearned(3, hidden_channel)
                else:
                    depth_key_pos_embed = PositionEmbeddingLearned(2, hidden_channel)
            else:
                depth_query_pos_embed = PositionEmbeddingLearnedwoNorm(2, hidden_channel)
                if depth_key_3d:
                    depth_key_pos_embed = PositionEmbeddingLearnedwoNorm(3, hidden_channel)
                else:
                    depth_key_pos_embed = PositionEmbeddingLearnedwoNorm(2, hidden_channel)

            if self.pe_norm == 'bn':
                fusion_query_pos_embed = [PositionEmbeddingLearned(2, hidden_channel) for _ in range(self.num_fusion_decoder_layers)]
                if self.pos_3d:
                    fusion_key_pos_embed = [PositionEmbeddingLearned(3, hidden_channel) for _ in range(self.num_fusion_decoder_layers)]
                else:
                    fusion_key_pos_embed = [PositionEmbeddingLearned(2, hidden_channel) for _ in range(self.num_fusion_decoder_layers)]

            else:
                fusion_query_pos_embed = [PositionEmbeddingLearnedwoNorm(2, hidden_channel) for _ in range(self.num_fusion_decoder_layers)]
                if self.pos_3d:
                    fusion_key_pos_embed = [PositionEmbeddingLearnedwoNorm(3, hidden_channel) for _ in range(self.num_fusion_decoder_layers)]
                else:
                    fusion_key_pos_embed = [PositionEmbeddingLearnedwoNorm(2, hidden_channel) for _ in range(self.num_fusion_decoder_layers)]

            if projection == 'v1':
                pts_projection = nn.Sequential(
                    nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1),
                )
                img_projection = nn.Sequential(
                    nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1),
                )
                if self.fuse_projection:
                    fuse_pts_projection = nn.Sequential(
                        nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1),
                        nn.ReLU(inplace=True),
                        nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1),
                    )
                    fuse_img_projection = nn.Sequential(
                        nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1),
                        nn.ReLU(inplace=True),
                        nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1),
                    )
                else:
                    fuse_pts_projection = nn.Identity()
                    fuse_img_projection = nn.Identity()
            elif projection == 'l2norm':
                pts_projection = ProjectionL2Norm(hidden_channel)
                img_projection = ProjectionL2Norm(hidden_channel)
                if self.fuse_projection:
                    fuse_pts_projection = ProjectionL2Norm(hidden_channel)
                    fuse_img_projection = ProjectionL2Norm(hidden_channel)
                else:
                    fuse_pts_projection = nn.Identity()
                    fuse_img_projection = nn.Identity()
            elif projection == 'layernorm':
                pts_projection = ProjectionLayerNorm(hidden_channel)
                img_projection = ProjectionLayerNorm(hidden_channel)
                if self.fuse_projection:
                    fuse_pts_projection = ProjectionLayerNorm(hidden_channel)
                    fuse_img_projection = ProjectionLayerNorm(hidden_channel)
                else:
                    fuse_pts_projection = nn.Identity()
                    fuse_img_projection = nn.Identity()
            elif projection == 'fc':
                pts_projection = ProjectionLayerNorm(hidden_channel, norm=False)
                img_projection = ProjectionLayerNorm(hidden_channel, norm=False)
                if self.fuse_projection:
                    fuse_pts_projection = ProjectionLayerNorm(hidden_channel, norm=False)
                    fuse_img_projection = ProjectionLayerNorm(hidden_channel, norm=False)
                else:
                    fuse_pts_projection = nn.Identity()
                    fuse_img_projection = nn.Identity()
            else:
                raise NotImplementedError

            if img_skip_connect:
                assert projection == 'layernorm'
                fuse_img_projection = ProjectionLayerNorm(hidden_channel, input_channel=hidden_channel*2)

            self.depth_transformer = DepthTransformer2D_3D(
                hidden_channel=hidden_channel, num_heads=num_heads, prediction_heads=depth_prediction_heads,
                ffn_channel=ffn_channel, dropout=dropout,  activation=activation, test_cfg=test_cfg,
                query_pos=depth_query_pos_embed, key_pos=depth_key_pos_embed, pts_projection=pts_projection,
                img_projection=img_projection, dbound=dbound, use_camera=use_camera, cross_smca=cross_smca,
                pos_3d=pos_3d, depth_stop_grad=depth_stop_grad, extra_camera=extra_camera,
                depth_key_3d=depth_key_3d
            )

            if fuse_self:
                self.fusion_transformer = FusionTransformer2D_3D_Self(
                    hidden_channel=hidden_channel, num_heads=num_heads, num_decoder_layers=num_fusion_decoder_layers,
                    prediction_heads=fusion_prediction_heads,  ffn_channel=ffn_channel, dropout=dropout,
                    activation=activation, test_cfg=test_cfg, query_pos=fusion_query_pos_embed, key_pos=fusion_query_pos_embed,
                    pts_projection=fuse_pts_projection, img_projection=fuse_img_projection,
                )
            else:
                self.fusion_transformer = FusionTransformer2D_3D_Cross(
                    hidden_channel=hidden_channel, num_heads=num_heads, num_decoder_layers=num_fusion_decoder_layers,
                    prediction_heads=fusion_prediction_heads,  ffn_channel=ffn_channel, dropout=dropout,
                    activation=activation, test_cfg=test_cfg, query_pos=fusion_query_pos_embed, key_pos=fusion_key_pos_embed,
                    pts_projection=fuse_pts_projection, img_projection=fuse_img_projection,
                )

            if self.initialize_by_heatmap and self.cross_heatmap:
                self.heatmap_pts_proj = ConvModule(
                    hidden_channel, hidden_channel, kernel_size=1, bias=bias,
                    conv_cfg=dict(type='Conv2d'), norm_cfg=dict(type='BN2d'),
                    act_cfg=None
                )
                if cross_heatmap_trick == 'msfusion':
                    self.heatmap_img_proj = MSFusion(hidden_channel, dict(type='BN2d'))
                elif cross_heatmap_trick == 'aspp':
                    self.heatmap_img_proj = ASPP(hidden_channel)
                else:
                    self.heatmap_img_proj = nn.Sequential(
                        ConvModule(
                            hidden_channel, hidden_channel, kernel_size=1, bias=bias,
                            conv_cfg=dict(type='Conv2d'), norm_cfg=dict(type='BN2d'),
                            act_cfg=None
                        ),
                    )
                self.cross_heatmap_head = self.build_heatmap(hidden_channel, bias, num_classes, dcn=cross_heatmap_dcn, layer_num=cross_heatmap_layer)
                if self.colattn_pos:
                    colattn_query_pos = PositionEmbeddingLearnedLN(8, hidden_channel)
                    colattn_key_pos = PositionEmbeddingLearnedLN(8, hidden_channel)
                    self.col_decoder = TransformerDecoderLayer(
                        hidden_channel, num_heads, ffn_channel, dropout, activation,
                        self_posembed=colattn_query_pos, cross_posembed=colattn_key_pos, cross_only=True
                    )
                else:
                    self.col_decoder = TransformerDecoderLayer(
                        hidden_channel, num_heads, ffn_channel, dropout, activation, cross_only=True
                    )
                self.reduce_conv = ConvModule(
                    hidden_channel*2, hidden_channel, kernel_size=1, bias=bias,
                    conv_cfg=dict(type='Conv2d'), norm_cfg=dict(type='BN2d'),
                )

        # a shared convolution
        if self.with_pts:
            self.shared_conv = build_conv_layer(
                dict(type='Conv2d'),
                in_channels,
                hidden_channel,
                kernel_size=3,
                padding=1,
                bias=bias,
            )

        # transformer decoder layers for object query with LiDAR feature
        if self.with_img:
            self.num_views = num_views
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
            self.img_heatmap_head = self.build_heatmap(hidden_channel, bias, num_classes, layer_num=img_heatmap_layer, dcn=img_heatmap_dcn)

            self.class_encoding = nn.Conv1d(num_classes, hidden_channel, 1)
            self.img_class_encoding = nn.Conv1d(num_classes, hidden_channel, 1)
        else:
            # query feature
            self.pts_query_feat = nn.Parameter(torch.randn(1, hidden_channel, self.num_proposals))
            self.pts_query_pos = nn.Parameter(torch.rand([1, self.num_proposals, 2])*torch.Tensor([x_size, y_size]).reshape(1, 1, 2), requires_grad=learnable_query_pos)

            self.img_query_feat = nn.Parameter(torch.randn(1, hidden_channel, self.num_img_proposals))
            self.img_query_pos = nn.Parameter(torch.rand([1, self.num_img_proposals, 2]), requires_grad=learnable_query_pos)
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
        # if self.with_pts:
        #     for m in self.point_transformer.parameters():
        #         if m.dim() > 1:
        #             nn.init.xavier_uniform_(m)
        # if self.with_img:
        #     for m in self.img_transformer.parameters():
        #         if m.dim() > 1:
        #             nn.init.xavier_uniform_(m)
        #     for m in self.shared_conv_img.parameters():
        #         if m.dim() > 1:
        #             nn.init.xavier_uniform_(m)
        # if self.with_pts and self.with_img:
        #     for m in self.fusion_transformer.parameters():
        #         if m.dim() > 1:
        #             nn.init.xavier_uniform_(m)
        #     for m in self.depth_transformer.parameters():
        #         if m.dim() > 1:
        #             nn.init.xavier_uniform_(m)

        for m in self.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        self.init_bn_momentum()

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

    def forward_single(self, inputs, img_inputs, img_metas):
        """
        Args:
            inputs (torch.Tensor): Input feature map with the shape of
                [B, C, 128(H), 128(W)]. (consistent with L748)
            img_inputs (torch.Tensor): Input feature map with the shape of
                [B*num_view, C, image_H, image_W]

        Returns:
            list[dict]: Output results for tasks.
        """

        if self.with_pts:
            batch_size = inputs.shape[0]
        else:
            batch_size = img_inputs.shape[0] // self.num_views

        if self.with_img:
            img_feat = self.shared_conv_img(img_inputs)  # [BS * n_views, C, H, W]
            img_h, img_w, num_channel = img_inputs.shape[-2], img_inputs.shape[-1], img_feat.shape[1]

        if self.with_pts:
            lidar_feat = self.shared_conv(inputs)  # [BS, C, H, W]

            #################################
            # image to BEV
            #################################
            lidar_feat_flatten = lidar_feat.view(batch_size, lidar_feat.shape[1], -1)  # [BS, C, H*W]
            bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)  # [BS, H*W, 2]

            if self.initialize_by_heatmap:
                if self.with_img and self.cross_heatmap:
                    if self.cross_heatmap_stop_grad:
                        img_feat_cross = img_feat.detach().clone()
                    else:
                        img_feat_cross = img_feat.clone()
                else:
                    img_feat_cross = None

                heatmap, dense_heatmap, pts_top_proposals_class, pts_top_proposals_index, fuse_lidar_feat_flatten = self.generate_heatmap(lidar_feat.clone(), batch_size, img_metas, img_feat_cross)

                if self.fuse_bev_feature:
                    assert fuse_lidar_feat_flatten is not None
                    pts_query_feat = fuse_lidar_feat_flatten.gather(
                        index=pts_top_proposals_index[:, None, :].expand(-1, lidar_feat_flatten.shape[1], -1), dim=-1
                    )  # [BS, C, num_proposals]
                else:
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

        if self.with_img:
            raw_img_feat = img_feat.view(batch_size, self.num_views, num_channel, img_h, img_w).permute(0, 2, 3, 1, 4) # [BS, C, H, n_views, W]
            (h, w) = img_inputs.shape[-2], img_inputs.shape[-1]
            img_feat_pos = self.create_2D_grid(h, w).to(raw_img_feat.device)  # (1, h*w, 2)
            normal_img_feat_pos = normalize_pos(img_feat_pos, w, h)
            normal_img_feat_pos_repeat = normal_img_feat_pos.repeat(batch_size, 1, 1)

            if self.initialize_by_heatmap:
                if self.img_heatmap_stop_grad:
                    img_feat_heatmap = img_feat.detach().clone()
                else:
                    img_feat_heatmap = img_feat.clone()

                img_heatmap, img_dense_heatmap, img_top_proposals_class, img_top_proposals_index, img_top_proposals_view_idx, img_top_proposals_pos_id = \
                    self.generate_heatmap_img(img_feat_heatmap, batch_size)
                img_feat_stack = raw_img_feat.permute(0, 1, 3, 2, 4).contiguous().view(batch_size, num_channel, self.num_views*img_h*img_w)
                normal_img_query_pos = normal_img_feat_pos_repeat.gather(
                    index=img_top_proposals_pos_id[:, None, :].permute(0, 2, 1).expand(-1, -1, normal_img_feat_pos.shape[-1]), dim=1
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

        #################################
        # transformer decoder layer (LiDAR feature as K,V)
        #################################

        ret_dicts = []
        if self.with_pts:
            if self.fuse_bev_feature:
                pts_query_feat, pts_query_pos, pts_ret_dicts = self.point_transformer(pts_query_feat, pts_query_pos, fuse_lidar_feat_flatten, bev_pos)
            else:
                pts_query_feat, pts_query_pos, pts_ret_dicts = self.point_transformer(pts_query_feat, pts_query_pos, lidar_feat_flatten, bev_pos)
            ret_dicts.extend(pts_ret_dicts)

        #################################
        # transformer decoder layer (img feature as K,V)
        #################################
        if self.with_img:
            # positional encoding for image fusion
            img_feat = raw_img_feat.permute(0, 3, 1, 2, 4)  # [BS, n_views, C, H, W]
            img_feat_flatten = img_feat.view(batch_size, self.num_views, num_channel, -1)  # [BS, n_views, C, H*W]

            # torch.save(normal_img_query_pos, "vis/img_query_pos_heatmap.pt")
            # torch.save(self.img_query_label, "vis/img_query_label.pt")

            img_query_feat, normal_img_query_pos, img_ret_dicts = self.img_transformer(img_query_feat, normal_img_query_pos, img_query_view, img_feat_flatten, normal_img_feat_pos, img_metas)

            # torch.save(normal_img_query_pos, "vis/img_query_pos.pt")
            # torch.save(img_query_view, "vis/img_query_view_3d.pt")
            # torch.save(img_ret_dicts[-1]["cls"], "vis/img_query_pred.pt")
            # torch.save(img_ret_dicts[-1]["offset"], "vis/img_query_offset.pt")
            # torch.save(img_ret_dicts[-1]["center_2d"], "vis/img_query_center_2d.pt")

        if self.with_img and self.with_pts:
            if self.stop_grad:
                pts_query_feat = pts_query_feat.detach().clone()
                img_query_feat = img_query_feat.detach().clone()
            pts_query_height = pts_ret_dicts[-1]['height'].detach().clone()

            if self.img_skip_connect:
                raw_img_query_feat = img_query_feat.clone()

            normal_img_query_bboxes = img_ret_dicts[-1]['bbox_2d'].detach().clone().permute(0, 2, 1)
            if self.depth_stop_grad:
                img_query_feat, img_query_pos_bev, depth_ret_dicts, extra_mul = self.depth_transformer(pts_query_feat.detach().clone(), pts_query_pos, pts_query_height, img_query_feat.detach().clone(), normal_img_query_pos, img_query_view, normal_img_query_bboxes, img_metas, img_feat_flatten.detach().clone(), normal_img_feat_pos)
            else:
                img_query_feat, img_query_pos_bev, depth_ret_dicts, extra_mul = self.depth_transformer(pts_query_feat, pts_query_pos, pts_query_height, img_query_feat, normal_img_query_pos, img_query_view, normal_img_query_bboxes, img_metas, img_feat_flatten, normal_img_feat_pos)

            # torch.save(depth_ret_dicts[-1]["proj_center"], "vis/img_query_center_depth.pt")

            if self.img_skip_connect:
                if extra_mul is not None:
                    raw_img_query_feat = raw_img_query_feat * extra_mul
                img_query_feat = torch.cat([img_query_feat, raw_img_query_feat], dim=1)
            all_query_feat, all_query_pos, fusion_ret_dicts = self.fusion_transformer(pts_query_feat, pts_query_pos, img_query_feat, img_query_pos_bev)

            # torch.save(pts_query_pos, 'vis/pts_query_pos.pt')
            # torch.save(img_query_pos_bev, 'vis/img_query_pos_bev.pt')

            ret_dicts.extend(fusion_ret_dicts)

        if self.initialize_by_heatmap:
            if self.with_pts:
                ret_dicts[0]['query_heatmap_score'] = heatmap.gather(index=pts_top_proposals_index[:, None, :].expand(-1, self.num_classes, -1), dim=-1)  # [bs, num_classes, num_proposals]
                ret_dicts[0]['dense_heatmap'] = dense_heatmap
            if self.with_img:
                ret_dicts[0]['img_query_heatmap_score'] = img_heatmap.gather(index=img_top_proposals_index[:, None, :].expand(-1, self.num_classes, -1), dim=-1)  # [bs, num_classes, num_proposals]
                ret_dicts[0]['img_dense_heatmap'] = img_dense_heatmap

        # return all the layer's results for auxiliary superivison
        new_res = {}
        for key in ret_dicts[0].keys():
            if key not in ['dense_heatmap', 'query_heatmap_score', 'img_query_heatmap_score', 'img_dense_heatmap']:
                new_res[key] = torch.cat([ret_dict[key] for ret_dict in ret_dicts], dim=-1)
            else:
                new_res[key] = ret_dicts[0][key]
        if self.with_img:
            for key in img_ret_dicts[0].keys():
                new_res[key] = torch.cat([ret_dict[key] for ret_dict in img_ret_dicts], dim=-1)
            new_res['view'] = img_query_view.repeat(1, self.num_img_decoder_layers)

            for key in depth_ret_dicts[0].keys():
                new_res[key] = torch.cat([ret_dict[key] for ret_dict in depth_ret_dicts], dim=-1)

        return [new_res]

    def forward(self, feats, img_feats, img_metas):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            tuple(list[dict]): Output results. first index by level, second index by layer
        """

        if img_feats is None:
            img_feats = [None]
        res = multi_apply(self.forward_single, feats, img_feats, [img_metas])
        assert len(res) == 1, "only support one level features."
        return res

    def build_heatmap(self, hidden_channel, bias, num_classes, layer_num=2, kernel_size=3, dcn=False):
        layers = []
        for i in range(layer_num-1):
            if dcn and i == layer_num-2:
                layers.append(ConvModule(
                    hidden_channel,
                    hidden_channel,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,
                    bias=bias,
                    conv_cfg=dict(type='DCNv2'),
                    norm_cfg=dict(type='BN2d'),
                ))
            else:
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

    def generate_heatmap_colattn(self, lidar_feat, img_feat, img_metas):
        # img_feat [bs*num_view, C, img_h, img_w]
        # lidar_feat [BS, C, H, W]

        img_feat = self.heatmap_img_proj(img_feat)
        lidar_feat = self.heatmap_pts_proj(lidar_feat)

        batch_size = lidar_feat.shape[0]
        H, W = lidar_feat.shape[2], lidar_feat.shape[3]
        img_h, img_w = img_feat.shape[2], img_feat.shape[3]

        lidar_feat_flatten = lidar_feat.reshape(batch_size, self.hidden_channel, H*W)  # [bs, C, H*W]
        img_feat = img_feat.reshape(batch_size, self.num_views, self.hidden_channel, img_h, img_w)  # (bs, num_view, C, img_h, img_w)
        lidar_feat_output = torch.zeros_like(lidar_feat_flatten)  # [bs, C, H*W]

        if self.colattn_pos:
            img_pos = torch.arange(img_h) + 0.5
            img_pos = img_pos.view(1, img_h, 1).to(img_feat.device)  # [1, img_h, 1]
            img_pos = img_pos.repeat(img_w, 1, 1)  # [img_w, img_h, 1]

        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(img_feat.device)
        query_pos_realmetric = bev_pos.permute(0, 2, 1) * self.test_cfg['out_size_factor'] * \
                               self.test_cfg['voxel_size'][0] + self.test_cfg['pc_range'][0]  # (bs, 2, H*W)

        for sample_idx in range(batch_size):
            query_pos_3d = torch.cat(
                [query_pos_realmetric[sample_idx], torch.ones_like(query_pos_realmetric[sample_idx][:1])],
                dim=0).detach().clone()  # (3, N)

            lidar2img_rt = torch.Tensor(img_metas[sample_idx]['lidar2img']).to(bev_pos.device)
            img_scale_factor = (
                lidar2img_rt.new_tensor(
                    img_metas[sample_idx]['scale_factor'][:2] if 'scale_factor' in img_metas[sample_idx].keys() else [
                        1.0, 1.0])
            )
            # transform point clouds back to original coordinate system by reverting the data augmentation
            if batch_size == 1:  # skip during inference to save time
                points = query_pos_3d.T  # [N, 3]
            else:
                points = apply_3d_transformation(query_pos_3d.T, 'LIDAR', img_metas[sample_idx], reverse=True).detach()

            num_points = points.shape[0]

            # dump_data = []
            for view_idx in range(self.num_views):
                # view_dump_data = {}
                valid_shape = img_metas[sample_idx]['valid_shape'][view_idx] / self.out_size_factor_img \
                    if 'valid_shape' in img_metas[sample_idx].keys() else [img_feat.shape[-1], img_feat.shape[-2]]
                red_img_w = int(valid_shape[0])
                red_img_h = int(valid_shape[1])

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

                center_xs = coor_x / self.out_size_factor_img
                center_xs = center_xs.long().squeeze()  # [num_proposal_on_img, ]

                on_the_image = (center_xs >= 0) & (center_xs < red_img_w)
                center_xs = center_xs[on_the_image]
                num_on_the_image = torch.sum(on_the_image)

                center_xs_onehot = F.one_hot(center_xs, red_img_w) # [N, red_img_w]
                center_xs_cumsum = torch.cumsum(center_xs_onehot, dim=0)  # [N, red_img_w]

                max_num = torch.max(center_xs_cumsum)
                center_xs_cumsum = center_xs_cumsum.long()

                pos_ids = torch.arange(num_on_the_image).to(center_xs.device)
                index = center_xs_cumsum[pos_ids, center_xs] - 1  # [N, ]

                current_view_depth = pts_2d[:, 2]
                lidar_feat_view = torch.zeros([red_img_w, self.hidden_channel, max_num]).to(lidar_feat_output.device)  # [red_img_w, C, max_num]

                if self.colattn_pos:
                    lidar_feat_depth = current_view_depth  # [H*W, ]
                    lidar_feat_depth = lidar_feat_depth[on_the_image]  # [N, ]
                    lidar_pos_view = torch.zeros([red_img_w, max_num, 1]).to(center_xs.device)  # [red_img_w, max_num, 1]
                    lidar_pos_view[center_xs, index, 0] = lidar_feat_depth.clone()
                    lidar_pos_view = lidar_pos_view / 35 - 1
                    img_pos_view = (img_pos[:red_img_w, :red_img_h].clone() / red_img_h) * 2 - 1

                    lidar_pos_view_sine = []
                    img_pos_view_sine = []
                    for L in range(4):
                        lidar_pos_view_sine.append(torch.sin((2**(L-1))*np.pi*lidar_pos_view))
                        lidar_pos_view_sine.append(torch.cos((2**(L-1))*np.pi*lidar_pos_view))

                        img_pos_view_sine.append(torch.sin((2**(L-1))*np.pi*img_pos_view))
                        img_pos_view_sine.append(torch.cos((2**(L-1))*np.pi*img_pos_view))

                    lidar_pos_view_sine = torch.cat(lidar_pos_view_sine, dim=-1)
                    img_pos_view_sine = torch.cat(img_pos_view_sine, dim=-1)

                lidar_feat_view[center_xs, :, index] = lidar_feat_flatten[sample_idx, :, on_the_image].transpose(1, 0)  # [N, C]

                img_feat_view = img_feat[sample_idx, view_idx, :, :red_img_h, :red_img_w]  # [C, red_img_h, red_img_w]
                img_feat_view = img_feat_view.permute(2, 0, 1)  # [red_img_w, C, red_img_h]

                if self.colattn_pos:
                    output, weights = self.col_decoder(lidar_feat_view[...,:max_num], img_feat_view, lidar_pos_view_sine, img_pos_view_sine, need_weights=True) # [red_img_w, C, max_num]
                else:
                    output, weights = self.col_decoder(lidar_feat_view[...,:max_num], img_feat_view, None, None, need_weights=True) # [img_w, C, max_num]

                lidar_feat_output[sample_idx, :, on_the_image] = output[center_xs, :, index].clone().transpose(1, 0)

        lidar_feat_output = lidar_feat_output.reshape(batch_size, self.hidden_channel, H, W)
        if self.cat_point:
            lidar_feat_output = torch.cat([lidar_feat_output, lidar_feat], dim=1)
            lidar_feat_output = self.reduce_conv(lidar_feat_output)

        heatmap_output = self.cross_heatmap_head(lidar_feat_output)

        return heatmap_output, lidar_feat_output.reshape(batch_size, self.hidden_channel, H*W)

    def generate_heatmap(self, lidar_feat, batch_size, img_metas, img_feat=None):
        dense_heatmap = self.heatmap_head(lidar_feat)  # [BS, num_class, H, W]
        fuse_lidar_feature_flatten = None
        if img_feat is None:
            heatmap = dense_heatmap.detach().sigmoid()  # [BS, num_class, H, W]
        else:
            dense_heatmap_cross, fuse_lidar_feature_flatten = self.generate_heatmap_colattn(lidar_feat, img_feat, img_metas)
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
        return heatmap, dense_heatmap, top_proposals_class, top_proposals_index, fuse_lidar_feature_flatten

    def generate_heatmap_img(self, img_feat, batch_size):

        img_dense_heatmap = self.img_heatmap_head(img_feat)  # [BS*num_view, num_class, H, W]
        img_heatmap = img_dense_heatmap.detach().sigmoid()  # [BS*num_view, num_class, H, W]
        padding = self.img_nms_kernel_size // 2
        local_max = torch.zeros_like(img_heatmap)
        # equals to nms radius = voxel_size * out_size_factor * kenel_size
        local_max_inner = F.max_pool2d(img_heatmap, kernel_size=self.img_nms_kernel_size, stride=1, padding=0)
        local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner
        img_heatmap = img_heatmap * (img_heatmap == local_max)  # [BS*num_view, num_class, H, W]
        img_heatmap = img_heatmap.view(batch_size, self.num_views, img_heatmap.shape[1], -1)  # [BS, num_views, num_class, H*W]
        img_heatmap = img_heatmap.permute(0, 2, 1, 3) # [BS, num_class, num_views, H*W]

        # top #num_proposals among all classes
        top_proposals = img_heatmap.contiguous().view(batch_size, -1).argsort(dim=-1, descending=True)[..., :self.num_img_proposals]  # [BS, num_proposals]
        top_proposals_class = top_proposals // (img_heatmap.shape[-1]*img_heatmap.shape[-2])  # [BS, num_proposals]

        top_proposals_view_index = top_proposals % (img_heatmap.shape[-1]*img_heatmap.shape[-2]) // img_heatmap.shape[-1]  # [BS, num_proposals]
        top_proposals_pos_index = top_proposals % img_heatmap.shape[-1]  # [BS, num_proposals]
        top_proposals_index = top_proposals % (img_heatmap.shape[-1]*img_heatmap.shape[-2])  # [BS, num_proposals]

        img_dense_heatmap = img_dense_heatmap.view(batch_size, self.num_views, img_dense_heatmap.shape[1], img_dense_heatmap.shape[2], img_dense_heatmap.shape[3]) # [BS, num_views, num_class, H, W]
        img_dense_heatmap = img_dense_heatmap.permute(0, 2, 1, 3, 4) # [BS, num_class, num_views, H*W]
        img_heatmap = img_heatmap.contiguous().view(batch_size, img_heatmap.shape[1], -1)

        return img_heatmap, img_dense_heatmap, top_proposals_class, top_proposals_index, top_proposals_view_index, top_proposals_pos_index

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels, gt_img_centers_2d, preds_dict, img_metas):
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

        res_tuple = multi_apply(self.get_targets_single, gt_bboxes_3d, gt_labels_3d, list_of_pred_dict, np.arange(len(gt_labels_3d)))

        labels = torch.cat(res_tuple[0], dim=0)
        label_weights = torch.cat(res_tuple[1], dim=0)
        bbox_targets = torch.cat(res_tuple[2], dim=0)
        bbox_weights = torch.cat(res_tuple[3], dim=0)
        ious = torch.cat(res_tuple[4], dim=0)
        num_pos_layer = np.concatenate(res_tuple[5], axis=0)  # [BS, num_layer]
        # matched_ious = np.mean(res_tuple[6])
        matched_ious = torch.cat(res_tuple[6], dim=0)

        if self.with_img:
            res_tuple_2d = multi_apply(self.get_targets_single_2d, gt_bboxes, gt_labels, gt_img_centers_2d, list_of_pred_dict, img_metas, np.arange(len(gt_bboxes)))
            labels_2d = torch.cat(res_tuple_2d[0], dim=0)
            label_weights_2d = torch.cat(res_tuple_2d[1], dim=0)
            bbox_targets_2d = torch.cat(res_tuple_2d[2], dim=0)
            bbox_weights_2d = torch.cat(res_tuple_2d[3], dim=0)

            center_targets_2d = torch.cat(res_tuple_2d[4], dim=0)
            center_weights_2d = torch.cat(res_tuple_2d[5], dim=0)
            depth_labels_2d = torch.cat(res_tuple_2d[6], dim=0)
            depth_weights_2d = torch.cat(res_tuple_2d[7], dim=0)

            ious_2d = torch.cat(res_tuple_2d[8], dim=0)
            num_pos_layer_2d = np.concatenate(res_tuple_2d[9], axis=0)  # [BS, num_layer]
            matched_ious_2d = torch.cat(res_tuple_2d[10], dim=0)

        if self.initialize_by_heatmap:
            heatmap = torch.cat(res_tuple[7], dim=0)
            if self.with_img:
                heatmap_2d = torch.cat(res_tuple_2d[11], dim=0)
                return labels, label_weights, bbox_targets, bbox_weights, ious, num_pos_layer, matched_ious, heatmap, \
                       labels_2d, label_weights_2d, bbox_targets_2d, bbox_weights_2d, center_targets_2d, center_weights_2d, \
                       depth_labels_2d, depth_weights_2d, ious_2d, num_pos_layer_2d, matched_ious_2d, heatmap_2d
            else:
                return labels, label_weights, bbox_targets, bbox_weights, ious, num_pos_layer, matched_ious, heatmap
        else:
            if self.with_img:
                return labels, label_weights, bbox_targets, bbox_weights, ious, num_pos_layer, matched_ious, \
                    labels_2d, label_weights_2d, bbox_targets_2d, bbox_weights_2d, center_targets_2d, center_weights_2d, \
                    depth_labels_2d, depth_weights_2d, ious_2d, num_pos_layer_2d, matched_ious_2d,
            else:
                return labels, label_weights, bbox_targets, bbox_weights, ious, num_pos_layer, matched_ious

    def get_targets_single_2d(self, gt_bboxes, gt_labels, gt_centers_2d, preds_dict, img_metas, batch_idx):
        gt_bboxes = gt_bboxes.float()
        num_proposals = preds_dict['cls'].shape[-1]
        bbox = copy.deepcopy(preds_dict['bbox_2d'].detach()).transpose(2, 1)[0]  # [num_proposals, 4]
        cls = copy.deepcopy(preds_dict['cls'].detach()).transpose(2, 1)[0]  # [num_proposals, num_classes]
        view = copy.deepcopy(preds_dict['view'].detach())[0] # [num_proposals, ]
        img_shape = img_metas['pad_shape']
        img_scale =[img_shape[1], img_shape[0], img_shape[1], img_shape[0]]

        img_scale = torch.Tensor(img_scale).to(bbox.device).unsqueeze(0)
        normal_gt_bbox = gt_bboxes / img_scale  # [num_gt, 4]
        gt_centers_2d = gt_centers_2d.float()
        normal_gt_centers = gt_centers_2d[..., :2] / img_scale[..., :2]

        centers = copy.deepcopy(preds_dict['center_2d'].detach()).transpose(2, 1)[0]  # [num_proposals, 2]
        offsets = copy.deepcopy(preds_dict['offset'].detach()).transpose(2, 1)[0]  # [num_proposals, 4]

        gt_pcx = gt_centers_2d[:, 0]
        gt_pcy = gt_centers_2d[:, 1]
        gt_cx = gt_bboxes[:, 0]
        gt_cy = gt_bboxes[:, 1]
        gt_w = gt_bboxes[:, 2]
        gt_h = gt_bboxes[:, 3]

        gt_wl = gt_pcx - (gt_cx - gt_w / 2)
        gt_wr = (gt_cx + gt_w / 2) - gt_pcx
        gt_ht = gt_pcy - (gt_cy - gt_h / 2)
        gt_hb = (gt_cy + gt_h / 2) - gt_pcy

        gt_offsets = torch.stack([gt_wl, gt_ht, gt_wr, gt_hb], dim=1)

        assign_result_list = []
        for idx_layer in range(self.num_img_decoder_layers):
            bboxes_tensor_layer = bbox[idx_layer*self.num_img_proposals:(idx_layer+1)*self.num_img_proposals, :]  # [num_proposals, 4]
            score_layer = cls[idx_layer*self.num_img_proposals:(idx_layer+1)*self.num_img_proposals, :]  # [num_proposals, num_class]
            view_layer = view[idx_layer*self.num_img_proposals:(idx_layer+1)*self.num_img_proposals]  # [num_proposals]
            centers_layer = centers[idx_layer*self.num_img_proposals:(idx_layer+1)*self.num_img_proposals, :]  # [num_proposals, 2]
            offsets_layer = offsets[idx_layer*self.num_img_proposals:(idx_layer+1)*self.num_img_proposals, :]

            assign_result = self.bbox_assigner_2d.assign(bboxes_tensor_layer, score_layer, centers_layer, offsets_layer, view_layer, gt_bboxes, gt_labels, gt_centers_2d[..., :2], gt_offsets, img_w=img_shape[1], img_h=img_shape[0])
            assign_result_list.append(assign_result)

        # combine assign result of each layer
        assign_result_ensemble = AssignResult(
            num_gts=sum([res.num_gts for res in assign_result_list]),
            gt_inds=torch.cat([res.gt_inds for res in assign_result_list]),
            max_overlaps=torch.cat([res.max_overlaps for res in assign_result_list]),
            labels=torch.cat([res.labels for res in assign_result_list]),
        )
        sampling_result = self.bbox_sampler.sample(assign_result_ensemble, bbox, normal_gt_bbox)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        assert len(pos_inds) + len(neg_inds) == num_proposals

        # torch.save(pos_inds, 'vis/pos_inds_%d.pt'%batch_idx)
        # torch.save(preds_dict['depth'], 'vis/pred_depth_%d.pt'%batch_idx)

        start = 0
        pos_num_layers = []
        for idx_layer in range(self.num_img_decoder_layers):
            layer_num_proposal = self.num_img_proposals
            pos_inds_layer = pos_inds[torch.logical_and(pos_inds>=start, pos_inds<start+layer_num_proposal)]
            count = pos_inds_layer.shape[0]
            pos_num_layers.append(count)
            start += layer_num_proposal
        pos_num_layers = np.array(pos_num_layers)
        assert np.sum(pos_num_layers) == pos_inds.shape[0]

        # create target for loss computation
        bbox_targets = torch.zeros([num_proposals, 6]).to(bbox.device)
        bbox_weights = torch.zeros([num_proposals, 6]).to(bbox.device)
        view_targets = bbox.new_zeros(num_proposals, dtype=torch.long)
        ious = assign_result_ensemble.max_overlaps
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bbox.new_zeros(num_proposals, dtype=torch.long)
        label_weights = bbox.new_zeros(num_proposals, dtype=torch.long)
        center_targets = torch.zeros([num_proposals, 2]).to(bbox.device)
        center_weights = torch.zeros([num_proposals, 2]).to(bbox.device)
        if self.depth_reg:
            depth_labels = bbox.new_zeros(num_proposals)
        else:
            depth_labels = bbox.new_zeros(num_proposals, dtype=torch.long)
        depth_weights = bbox.new_zeros(num_proposals, dtype=torch.long)

        if gt_labels is not None:  # default label is -1
            labels += self.num_classes

        # both pos and neg have classification loss, only pos has regression and iou loss
        if len(pos_inds) > 0:
            # bbox_targets[pos_inds, :] = sampling_result.pos_gt_bboxes
            pcx = normal_gt_centers[sampling_result.pos_assigned_gt_inds, 0]
            pcy = normal_gt_centers[sampling_result.pos_assigned_gt_inds, 1]
            cx = sampling_result.pos_gt_bboxes[:, 0]
            cy = sampling_result.pos_gt_bboxes[:, 1]
            w = sampling_result.pos_gt_bboxes[:, 2]
            h = sampling_result.pos_gt_bboxes[:, 3]

            wl = pcx - (cx - w / 2)
            wr = (cx + w/2) - pcx
            ht = pcy - (cy - h / 2)
            hb = (cy + h / 2) - pcy

            bbox_targets[pos_inds, :2] = normal_gt_centers[sampling_result.pos_assigned_gt_inds, :2]
            bbox_targets[pos_inds, 2] = wl
            bbox_targets[pos_inds, 3] = ht
            bbox_targets[pos_inds, 4] = wr
            bbox_targets[pos_inds, 5] = hb

            bbox_weights[pos_inds, :2] = 1.0
            bbox_weights[pos_inds, 2:] = 1.0

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

            if self.depth_reg:
                depth = gt_centers_2d[sampling_result.pos_assigned_gt_inds, 2]
                depth_labels[pos_inds] = torch.log(depth + 1e-5)

            else:
                depth = torch.clamp(gt_centers_2d[sampling_result.pos_assigned_gt_inds, 2], self.dbound[0], self.dbound[1])
                depth_class = (depth - self.dbound[0]) / self.dbound[2]
                depth_labels[pos_inds] = torch.round(depth_class).long()
            depth_weights[pos_inds] = 1

            view_mask_ignore = view_targets != view
            label_weights[view_mask_ignore] = 0
            bbox_weights[view_mask_ignore, :] = 0
            center_weights[view_mask_ignore] = 0
            depth_weights[view_mask_ignore] = 0

            # torch.save(pos_inds, 'vis/img_pos_inds_%d.pt'%batch_idx)
            # torch.save(center_targets, 'vis/center_targets_%d.pt'%batch_idx)
            # torch.save(center_weights, 'vis/center_weights_%d.pt'%batch_idx)
            # torch.save(depth_labels, 'vis/depth_labels_%d.pt'%batch_idx)
            # torch.save(depth_weights, 'vis/depth_weights_%d.pt'%batch_idx)

            center_targets = center_targets[-self.num_img_proposals:]
            center_weights = center_weights[-self.num_img_proposals:]
            depth_labels = depth_labels[-self.num_img_proposals:]
            depth_weights = depth_weights[-self.num_img_proposals:]

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # # compute dense heatmap targets
        if self.initialize_by_heatmap:
            device = labels.device
            feature_map_size = (img_shape[1] // self.out_size_factor_img, img_shape[0] // self.out_size_factor_img)
            heatmap = gt_bboxes.new_zeros(self.num_classes, self.num_views, feature_map_size[1], feature_map_size[0])

            for idx in range(len(gt_bboxes)):
                width = gt_bboxes[idx][2]
                length = gt_bboxes[idx][3]
                width = width / self.out_size_factor_img
                length = length / self.out_size_factor_img
                view_id = gt_labels[idx][1]
                if width > 0 and length > 0:
                    radius = gaussian_radius((length, width), min_overlap=self.train_cfg['gaussian_overlap_2d'])
                    radius = max(self.train_cfg['min_radius'], int(radius))
                    radius = min(self.train_cfg['max_radius'], int(radius))

                    x, y = gt_centers_2d[idx][0], gt_centers_2d[idx][1]
                    # x, y = gt_bboxes[idx][0], gt_bboxes[idx][1]

                    coor_x = x / self.out_size_factor_img
                    coor_y = y / self.out_size_factor_img

                    center = torch.tensor([coor_x, coor_y], dtype=torch.float32, device=device)
                    center_int = center.to(torch.int32)
                    draw_heatmap_gaussian(heatmap[gt_labels[idx][0], view_id], center_int, radius)

            matched_ious = torch.ones_like(ious) * -1
            matched_ious[pos_inds] = ious[pos_inds]

            return labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], center_targets[None], center_weights[None], depth_labels[None], depth_weights[None], ious[None], pos_num_layers[None], matched_ious[None], heatmap[None]
        else:
            matched_ious = torch.ones_like(ious) * -1
            matched_ious[pos_inds] = ious[pos_inds]
            return labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], center_targets[None], center_weights[None], depth_labels[None], depth_weights[None], ious[None], pos_num_layers[None], matched_ious[None]

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, preds_dict, batch_idx):
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
        # each layer should do label assign seperately.
        # if self.auxiliary:
        #     num_layer = self.num_pts_decoder_layers + self.num_fusion_decoder_layers
        # else:
        #     num_layer = 1
        num_layer = self.num_pts_decoder_layers + self.num_fusion_decoder_layers
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

                    center = torch.tensor([coor_x, coor_y], dtype=torch.float32, device=device)
                    center_int = center.to(torch.int32)
                    draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]], center_int, radius)

            matched_ious = torch.ones_like(ious) * -1
            matched_ious[pos_inds] = ious[pos_inds]
            return labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], ious[None], pos_num_layers[None], matched_ious[None], heatmap[None]
        else:
            matched_ious = torch.ones_like(ious) * -1
            matched_ious[pos_inds] = ious[pos_inds]
            return labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], ious[None], pos_num_layers[None], matched_ious[None]

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels, gt_pts_centers_2d, gt_img_centers_2d, img_metas, preds_dicts, **kwargs):
        """Loss function for CenterHead.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (list[list[dict]]): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        if self.with_img:
            if self.initialize_by_heatmap:
                labels, label_weights, bbox_targets, bbox_weights, ious, num_pos_layer, matched_ious, heatmap, \
                labels_2d, label_weights_2d, bbox_targets_2d, bbox_weights_2d, center_targets_2d, center_weights_2d, \
                depth_2d, depth_weight_2d, ious_2d, num_pos_layer_2d, matched_ious_2d, heatmap_2d = self.get_targets(gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels, gt_img_centers_2d, preds_dicts[0], img_metas)
            else:
                labels, label_weights, bbox_targets, bbox_weights, ious, num_pos_layer, matched_ious, \
                labels_2d, label_weights_2d, bbox_targets_2d, bbox_weights_2d, center_targets_2d, center_weights_2d, \
                depth_2d, depth_weight_2d, ious_2d, num_pos_layer_2d, matched_ious_2d = self.get_targets(gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels, gt_img_centers_2d, preds_dicts[0], img_metas)        # if hasattr(self, 'on_the_image_mask'):
        else:
            if self.initialize_by_heatmap:
                labels, label_weights, bbox_targets, bbox_weights, ious, num_pos_layer, matched_ious, heatmap = self.get_targets(gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels, preds_dicts[0], img_metas)
            else:
                labels, label_weights, bbox_targets, bbox_weights, ious, num_pos_layer, matched_ious = self.get_targets(gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels, preds_dicts[0], img_metas)        # if hasattr(self, 'on_the_image_mask'):

        preds_dict = preds_dicts[0][0]
        loss_dict = dict()

        if self.initialize_by_heatmap:
            # compute heatmap loss
            # pos_mask = torch.max(heatmap, dim=1, keepdim=True)[0] > 0
            # pos_mask = pos_mask.repeat(1, self.num_classes, 1, 1)
            # weight = 0.1 * torch.ones_like(heatmap)
            # weight[pos_mask] = 1
            # loss_heatmap = self.loss_heatmap(clip_sigmoid(preds_dict['dense_heatmap']), heatmap, weight=weight, avg_factor=max(heatmap.eq(1).float().sum().item(), 1))
            loss_heatmap = self.loss_heatmap(clip_sigmoid(preds_dict['dense_heatmap']), heatmap, avg_factor=max(heatmap.eq(1).float().sum().item(), 1))
            if 'valid_shape' in img_metas[0].keys():
                bs = heatmap_2d.shape[0]
                num_view = heatmap_2d.shape[2]
                heatmap_2d_weight = torch.zeros_like(heatmap_2d)
                for sample_idx in range(bs):
                    for view_idx in range(num_view):
                        valid_shape = img_metas[sample_idx]['valid_shape'][view_idx] / self.out_size_factor_img
                        red_width = int(valid_shape[0])
                        red_height = int(valid_shape[1])
                        heatmap_2d_weight[sample_idx, :, view_idx, :red_height, :red_width] = 1

                # torch.save(clip_sigmoid(preds_dict['img_dense_heatmap']), "vis/pred_heatmap_2d.pt")
                # torch.save(heatmap_2d, "vis/gt_heatmap_2d.pt")

                # img_pos_mask = torch.max(heatmap_2d, dim=1, keepdim=True)[0] > 0
                # img_pos_mask = img_pos_mask.repeat(1, self.num_classes, 1, 1, 1)
                # heatmap_2d_weight = 0.1 * heatmap_2d_weight
                # heatmap_2d_weight[img_pos_mask] = 1

                # loss_heatmap_2d = self.loss_heatmap_2d(clip_sigmoid(preds_dict['img_dense_heatmap']), heatmap_2d, weight=heatmap_2d_weight, avg_factor=max(heatmap_2d.eq(1).float().sum().item(), 1))
                loss_heatmap_2d = self.loss_heatmap_2d(clip_sigmoid(preds_dict['img_dense_heatmap']), heatmap_2d, weight=heatmap_2d_weight, avg_factor=max(heatmap_2d.eq(1).float().sum().item(), 1))

            else:
                # import pdb
                # pdb.set_trace()
                loss_heatmap_2d = self.loss_heatmap_2d(clip_sigmoid(preds_dict['img_dense_heatmap']), heatmap_2d, avg_factor=max(heatmap_2d.eq(1).float().sum().item(), 1))

            # loss_heatmap_2d = self.loss_heatmap_2d(clip_sigmoid(preds_dict['img_dense_heatmap']), heatmap_2d, avg_factor=max(heatmap_2d.eq(1).float().sum().item(), 1))

            loss_dict['loss_heatmap'] = loss_heatmap
            loss_dict['loss_heatmap_2d'] = loss_heatmap_2d

        # compute loss for each layer
        start = 0
        num_pos_layer = np.sum(num_pos_layer, axis=0)
        num_pos_layer_2d = np.sum(num_pos_layer_2d, axis=0)

        for idx_layer in range(self.num_pts_decoder_layers+self.num_fusion_decoder_layers):
            layer_num_proposals = self.get_layer_num_proposal(idx_layer)
            if idx_layer < self.num_pts_decoder_layers:
                prefix = f'layer_pts_{idx_layer}'
                if not self.supervision3d:
                    start += layer_num_proposals
                    continue
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

        if self.with_img and self.supervision2d:
            start = 0
            for idx_layer in range(self.num_img_decoder_layers):
                prefix = f'layer_img_{idx_layer}'
                layer_num_proposals = self.num_img_proposals
                layer_labels_2d = labels_2d[..., start:start + layer_num_proposals].reshape(-1)
                layer_label_weights_2d = label_weights_2d[..., start:start + layer_num_proposals].reshape(-1)
                layer_score_2d = preds_dict['cls'][..., start:start + layer_num_proposals]
                layer_cls_score_2d = layer_score_2d.permute(0, 2, 1).reshape(-1, self.num_classes)
                layer_loss_cls_2d = self.loss_cls_2d(layer_cls_score_2d, layer_labels_2d, layer_label_weights_2d, avg_factor=max(num_pos_layer_2d[idx_layer], 1))

                preds_2d_center = preds_dict['center_2d'][..., start:start + layer_num_proposals].permute(0, 2, 1)  # [bs, num_proposal, 4]
                preds_2d_offset = preds_dict['offset'][..., start:start + layer_num_proposals].permute(0, 2, 1)  # [bs, num_proposal, 4]
                preds_2d = torch.cat([preds_2d_center, preds_2d_offset], dim=2)  # [bs, num_proposal, 6]
                layer_bbox_targets_2d = bbox_targets_2d[:, start:start + layer_num_proposals, :]
                layer_reg_weights_2d = bbox_weights_2d[:, start:start + layer_num_proposals, :]
                # code_weights = self.train_cfg.get('code_weights', None)
                # layer_reg_weights_2d = layer_bbox_weights_2d * layer_bbox_weights_2d.new_tensor(code_weights)
                layer_loss_bbox_2d = self.loss_bbox_2d(preds_2d, layer_bbox_targets_2d, layer_reg_weights_2d, avg_factor=max(num_pos_layer_2d[idx_layer], 1))

                layer_match_ious_2d = matched_ious_2d[..., start:start + layer_num_proposals]
                layer_match_ious_2d = torch.sum(layer_match_ious_2d*(layer_match_ious_2d>=0), dim=-1) / (torch.sum(layer_match_ious_2d>=0, dim=-1) + 1e-2)
                layer_match_ious_2d = torch.mean(layer_match_ious_2d)
                start += layer_num_proposals

                loss_dict[f'{prefix}_loss_cls_2d'] = layer_loss_cls_2d
                loss_dict[f'{prefix}_loss_bbox_2d'] = layer_loss_bbox_2d
                loss_dict[f'{prefix}_matched_ious_2d'] = layer_match_ious_2d

                if idx_layer == self.num_img_decoder_layers - 1:
                    pred_centers = preds_dict['proj_center'].permute(0, 2, 1)   # [bs, num_proposal, 2]
                    loss_center_2d = self.loss_center_2d(pred_centers, center_targets_2d, center_weights_2d, avg_factor=max(num_pos_layer_2d[idx_layer], 1))
                    loss_dict[f'{prefix}_loss_proj_center'] = loss_center_2d

                    if self.depth_reg:
                        pred_depth = preds_dict['depth'].permute(0, 2, 1)
                        depth_labels = depth_2d.unsqueeze(-1)
                        depth_weights = depth_weight_2d.unsqueeze(-1)

                        loss_depth = self.loss_depth_reg(pred_depth, depth_labels, depth_weights, avg_factor=max(num_pos_layer_2d[idx_layer], 1))
                        loss_dict[f'{prefix}_loss_depth_reg'] = loss_depth
                    else:
                        pred_depth = preds_dict['depth'].permute(0, 2, 1).reshape(-1, self.num_depth_class)
                        depth_labels = depth_2d.reshape(-1)
                        depth_weights = depth_weight_2d.reshape(-1)
                        loss_depth = self.loss_depth(pred_depth, depth_labels, depth_weights, avg_factor=max(num_pos_layer_2d[idx_layer], 1))
                        loss_dict[f'{prefix}_loss_depth_cls'] = loss_depth

                    if self.extra_class_sup:
                        depth_layer_cls_score_2d = preds_dict['extra_cls'].permute(0, 2, 1).reshape(-1, self.num_classes)
                        depth_layer_loss_cls_2d = self.loss_cls_2d(depth_layer_cls_score_2d, layer_labels_2d, layer_label_weights_2d, avg_factor=max(num_pos_layer_2d[idx_layer], 1))
                        loss_dict[f'{prefix}_loss_extra_cls'] = depth_layer_loss_cls_2d

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
            layer_num_proposal = self.get_layer_num_proposal(self.num_pts_decoder_layers+self.num_fusion_decoder_layers-1)

            batch_size = preds_dict[0]['heatmap'].shape[0]

            batch_score = preds_dict[0]['heatmap'][..., -layer_num_proposal:].sigmoid()

            if self.fuse_self:
                one_hot = F.one_hot(self.query_labels, num_classes=self.num_classes).permute(0, 2, 1)
                query_heatmap_score = preds_dict[0]['query_heatmap_score'] * one_hot
                one_hot_img = F.one_hot(self.img_query_label, num_classes=self.num_classes).permute(0, 2, 1)
                img_query_heatmap_score = preds_dict[0]['img_query_heatmap_score'] * one_hot_img
                query_heatmap_score = torch.cat([query_heatmap_score, img_query_heatmap_score], dim=2)
                query_one_hot = torch.cat([one_hot, one_hot_img], dim=2)
            else:
                one_hot = F.one_hot(self.query_labels, num_classes=self.num_classes).permute(0, 2, 1)
                query_heatmap_score = preds_dict[0]['query_heatmap_score'] * one_hot
            # batch_score = batch_score * query_heatmap_score
            batch_label = torch.max(batch_score, dim=1)[1]
            batch_one_hot = F.one_hot(batch_label, num_classes=self.num_classes).permute(0, 2, 1)
            batch_score = batch_score * query_heatmap_score * batch_one_hot

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
                    dict(num_class=8, class_names=[], indices=[0, 1, 2, 3, 4, 5, 6, 7], radius=-1),
                    dict(num_class=1, class_names=['pedestrian'], indices=[8], radius=0.175),
                    dict(num_class=1, class_names=['traffic_cone'], indices=[9], radius=0.175),
                ]
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
                        if task['radius'] > 0:
                            if self.test_cfg['nms_type'] == 'circle':
                                boxes_for_nms = torch.cat([boxes3d[task_mask][:, :2], scores[:, None][task_mask]], dim=1)
                                task_keep_indices = torch.tensor(
                                    circle_nms(
                                        boxes_for_nms.detach().cpu().numpy(),
                                        task['radius'],
                                    )
                                )
                            else:
                                boxes_for_nms = xywhr2xyxyr(img_metas[i]['box_type_3d'](boxes3d[task_mask][:, :7], 7).bev)
                                top_scores = scores[task_mask]
                                task_keep_indices = nms_gpu(
                                    boxes_for_nms,
                                    top_scores,
                                    thresh=task['radius'],
                                    pre_maxsize=self.test_cfg['pre_maxsize'],
                                    post_max_size=self.test_cfg['post_maxsize'],
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
        if self.fuse_self:
            if self.with_img and self.with_pts and idx_layer >= self.num_pts_decoder_layers:
                layer_num_proposal = self.num_proposals + self.num_img_proposals
            else:
                layer_num_proposal = self.num_proposals
        else:
            layer_num_proposal = self.num_proposals
        return layer_num_proposal