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
                          xywhr2xyxyr, limit_period, PseudoSampler, BboxOverlaps3D, LiDARInstance3DBoxes)
from mmdet3d.models.builder import HEADS, build_loss
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu
from mmdet3d.models.utils import clip_sigmoid, inverse_sigmoid
from mmdet3d.models.fusion_layers import apply_3d_transformation
from mmdet.core import build_bbox_coder, multi_apply, build_assigner, build_sampler, AssignResult

from mmdet3d.models.utils import FFN, TransformerDecoderLayer, PositionEmbeddingLearned, PositionEmbeddingLearnedwoNorm,\
    PointTransformer2D_3D, ProjectionLayerNorm, DepthEncoder, DepthEncoderLarge, DepthEncoderSmall, DepthEncoderResNet, \
    ImageTransformer_Seq_MS, FusionTransformer_Seq, FusionTransformer2D_3D_Cross, PositionEmbeddingLearnedLN, \
    ImageTransformer_Seq_DETR_MS, ImageTransformer_Seq_PETR_MS

from mmdet3d.models.utils.ops.modules import MSDeformAttn
from mmdet3d.models.utils.deformable_decoder import DeformableTransformerDecoderLayer


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ConvLN(nn.Module):
    def __init__(self, input_channel, hidden_channel, kernel_size=3,  stride=1, padding=1, require_act=True):
        super().__init__()
        if require_act:
            self.module = nn.Sequential(
                nn.Conv2d(input_channel, hidden_channel, kernel_size=kernel_size, stride=stride, padding=padding),
                LayerNorm(hidden_channel, data_format="channels_first"),
                nn.ReLU()
            )
        else:
            self.module = nn.Sequential(
                nn.Conv2d(input_channel, hidden_channel, kernel_size=kernel_size, stride=stride, padding=padding),
                LayerNorm(hidden_channel, data_format="channels_first"),
            )

    def forward(self, x):
        # [bs, C, H, W]
        x = self.module(x)
        return x

class SE_Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.att(x)


def denormalize_pos(normal_pos, x_max, y_max):
    max_xy = torch.Tensor([x_max, y_max]).to(normal_pos.device).view(1, 1, 2)
    pos = normal_pos.sigmoid() * max_xy
    return pos


def normalize_pos(pos, x_max, y_max):
    max_xy = torch.Tensor([x_max, y_max]).to(pos.device).view(1, 1, 2)
    normal_pos = pos / max_xy
    return inverse_sigmoid(normal_pos)

@HEADS.register_module()
class ImplicitHead2D_Seq_MS_Deform(nn.Module):
    def __init__(self,
                 with_pts=False,
                 with_img=False,
                 pts_smca=False,
                 num_views=0,
                 in_channels_img=64,
                 out_size_factor_img=4,
                 num_proposals=128,
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
                 cross_type='deform_height',
                 range_num=5,
                 cat_point=True,
                 fuse_bev_feature=False,
                 img_heatmap_dcn=False,
                 cross_heatmap_dcn=False,
                 cross_heatmap_layer=2,
                 cross_heatmap_LN=False,
                 learnable_query_pos=True,
                 nms_kernel_size=3,
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
                 # others
                 train_cfg=None,
                 test_cfg=None,
                 bbox_coder=None,
                 supervision3d=True,
                 cross_heatmap_stop_grad=False,
                 level_num=4,
                 encode_depth=False,
                 heatmap_query_level='max',
                 fuse_self=True,
                 cwa_avg=True,
                 img_decoder_type='deformable'
                 ):
        super(ImplicitHead2D_Seq_MS_Deform, self).__init__()

        self.with_img = with_img
        self.with_pts = with_pts
        self.num_proposals = num_proposals
        self.num_classes = num_classes
        self.bbox_coder = build_bbox_coder(bbox_coder)

        self.bn_momentum = bn_momentum
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.initialize_by_heatmap = initialize_by_heatmap
        self.cross_heatmap = cross_heatmap
        self.cross_only = cross_only
        self.fuse_bev_feature = fuse_bev_feature
        self.cat_point = cat_point
        self.level_num = level_num
        self.in_channels_img = in_channels_img
        self.heatmap_query_level = heatmap_query_level
        self.cross_type = cross_type
        assert cross_type in ['deform_height', 'range_height']
        self.range_num = range_num

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)
        self.loss_heatmap = build_loss(loss_heatmap)
        if self.initialize_by_heatmap is True:
            assert learnable_query_pos is False, "initialized by heatmap is conflicting with learnable query position"

        self.num_img_decoder_layers = num_img_decoder_layers * with_img
        self.num_pts_decoder_layers = num_pts_decoder_layers * with_pts
        self.num_fusion_decoder_layers = num_fusion_decoder_layers * with_img * with_pts
        # self.num_projection_layers = num_projection_layers
        self.hidden_channel = hidden_channel
        self.sampling = False
        self.out_size_factor_img = out_size_factor_img
        self.supervision3d = supervision3d
        self.encode_depth = encode_depth
        self.img_decoder_type = img_decoder_type

        self.cross_heatmap_stop_grad = cross_heatmap_stop_grad

        self.cross_heatmap_LN = cross_heatmap_LN
        self.fuse_self = fuse_self
        self.cwa_avg = cwa_avg

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1

        heads3d = copy.deepcopy(common_heads)
        heads3d.update(dict(heatmap=(self.num_classes, 2)))
        pts_prediction_heads = FFN(hidden_channel, heads3d, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias)
        fusion_prediction_heads = FFN(hidden_channel, heads3d, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias)

        heads_view = dict(center_view=(3, 2), dim_view=(3, 2), rot_view=(2, 2),
                          vel_view=(2, 2), heatmap_view=(self.num_classes, 2))
        img_prediction_heads = FFN(hidden_channel, heads_view, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias)

        if with_pts:
            pts_query_pos_embed = [PositionEmbeddingLearned(2, hidden_channel) for _ in range(num_pts_decoder_layers)]
            pts_key_pos_embed = [PositionEmbeddingLearned(2, hidden_channel) for _ in range(num_pts_decoder_layers)]
            self.point_transformer = PointTransformer2D_3D(
                pts_smca=pts_smca, hidden_channel=hidden_channel, num_heads=num_heads, num_decoder_layers=num_pts_decoder_layers,
                prediction_heads=pts_prediction_heads, ffn_channel=ffn_channel, dropout=dropout, activation=activation, test_cfg=test_cfg,
                query_pos=pts_query_pos_embed, key_pos=pts_key_pos_embed, supervision3d=supervision3d
            )
        if with_img:
            if self.img_decoder_type == 'detr':
                img_query_pos_embed = [PositionEmbeddingLearnedwoNorm(3, hidden_channel) for _ in range(num_img_decoder_layers)]
                img_key_pos_embed = [PositionEmbeddingLearnedwoNorm(6, hidden_channel) for _ in range(num_img_decoder_layers)]
                self.img_transformer = ImageTransformer_Seq_DETR_MS(
                    hidden_channel=hidden_channel, num_heads=num_heads, num_decoder_layers=num_img_decoder_layers, out_size_factor_img=out_size_factor_img,
                    num_views=num_views, prediction_heads=img_prediction_heads, ffn_channel=ffn_channel, dropout=dropout, activation=activation, test_cfg=test_cfg,
                    query_pos=img_query_pos_embed, key_pos=img_key_pos_embed
                )
            elif self.img_decoder_type == 'petr':
                img_query_pos_embed = [PositionEmbeddingLearnedLN(3, hidden_channel) for _ in range(num_img_decoder_layers)]
                img_key_pos_embed = [PositionEmbeddingLearnedLN(96, hidden_channel) for _ in range(num_img_decoder_layers)]
                self.img_transformer = ImageTransformer_Seq_PETR_MS(
                    hidden_channel=hidden_channel, num_heads=num_heads, num_decoder_layers=num_img_decoder_layers, out_size_factor_img=out_size_factor_img,
                    num_views=num_views, prediction_heads=img_prediction_heads, ffn_channel=ffn_channel, dropout=dropout, activation=activation, test_cfg=test_cfg,
                    query_pos=img_query_pos_embed, key_pos=img_key_pos_embed
                )
            else:
                img_query_pos_embed = [PositionEmbeddingLearnedwoNorm(3, hidden_channel) for _ in range(num_img_decoder_layers)]
                img_key_pos_embed = [PositionEmbeddingLearnedwoNorm(6, hidden_channel) for _ in range(num_img_decoder_layers)]
                self.img_transformer = ImageTransformer_Seq_MS(
                    hidden_channel=hidden_channel, num_heads=num_heads, num_decoder_layers=num_img_decoder_layers, out_size_factor_img=out_size_factor_img,
                    num_views=num_views, prediction_heads=img_prediction_heads, ffn_channel=ffn_channel, dropout=dropout, activation=activation, test_cfg=test_cfg,
                    query_pos=img_query_pos_embed, key_pos=img_key_pos_embed
                )

        if with_pts and with_img:
            if self.fuse_self:
                fusion_query_pos_embed = [PositionEmbeddingLearned(2, hidden_channel) for _ in range(self.num_fusion_decoder_layers)]
                fusion_key_pos_embed = [PositionEmbeddingLearned(2, hidden_channel) for _ in range(self.num_fusion_decoder_layers)]
                fuse_projection = ProjectionLayerNorm(hidden_channel, input_channel=hidden_channel * 2)
                self.fusion_transformer = FusionTransformer_Seq(
                    hidden_channel=hidden_channel, num_heads=num_heads, num_decoder_layers=num_fusion_decoder_layers,
                    prediction_heads=fusion_prediction_heads,  ffn_channel=ffn_channel, dropout=dropout,
                    activation=activation, test_cfg=test_cfg, query_pos=fusion_query_pos_embed, key_pos=fusion_query_pos_embed,
                    fuse_projection=fuse_projection, cwa_avg=cwa_avg
                )
            else:
                fuse_query_projection = ProjectionLayerNorm(hidden_channel)
                fuse_key_projection = ProjectionLayerNorm(hidden_channel)

                fusion_query_pos_embed = [PositionEmbeddingLearned(2, hidden_channel) for _ in range(self.num_fusion_decoder_layers)]
                fusion_key_pos_embed = [PositionEmbeddingLearned(3, hidden_channel) for _ in range(self.num_fusion_decoder_layers)]

                self.fusion_transformer = FusionTransformer2D_3D_Cross(
                    hidden_channel=hidden_channel, num_heads=num_heads, num_decoder_layers=num_fusion_decoder_layers,
                    prediction_heads=fusion_prediction_heads, ffn_channel=ffn_channel, dropout=dropout,
                    activation=activation, test_cfg=test_cfg, query_pos=fusion_query_pos_embed,
                    key_pos=fusion_key_pos_embed,  pts_projection=fuse_query_projection, img_projection=fuse_key_projection
                )

            if self.initialize_by_heatmap and self.cross_heatmap:
                # self.heatmap_pts_proj = ConvModule(
                #     hidden_channel, hidden_channel, kernel_size=3, bias=bias, padding=1,
                #     conv_cfg=dict(type='Conv2d'), norm_cfg=dict(type='BN2d'),
                #     act_cfg=None
                # )
                self.heatmap_pts_proj = nn.Sequential(
                    nn.Linear(hidden_channel, hidden_channel),
                    nn.LayerNorm(hidden_channel)
                )
                self.heatmap_img_proj = nn.Sequential(
                    nn.Linear(hidden_channel, hidden_channel),
                    nn.LayerNorm(hidden_channel)
                )
                if cross_heatmap_LN:
                    self.cross_heatmap_head = self.build_heatmap_LN(hidden_channel, bias, num_classes, layer_num=cross_heatmap_layer)
                else:
                    self.cross_heatmap_head = self.build_heatmap(hidden_channel, bias, num_classes, dcn=cross_heatmap_dcn, layer_num=cross_heatmap_layer)

                if self.cross_type == 'deform_height':
                    colattn_query_pos = PositionEmbeddingLearnedwoNorm(3, hidden_channel)
                    colattn_key_pos = PositionEmbeddingLearnedwoNorm(2, hidden_channel)

                    self.col_decoder = DeformableTransformerDecoderLayer(
                        hidden_channel, num_heads, dim_feedforward=ffn_channel, dropout=dropout, activation=activation,
                        self_posembed=colattn_query_pos, cross_posembed=colattn_key_pos
                    )
                else:
                    colattn_query_pos = PositionEmbeddingLearnedwoNorm(1, hidden_channel)
                    colattn_key_pos = PositionEmbeddingLearnedwoNorm(3, hidden_channel)

                    self.col_decoder = TransformerDecoderLayer(
                        hidden_channel, num_heads, dim_feedforward=ffn_channel, dropout=dropout, activation=activation,
                        self_posembed=colattn_query_pos, cross_posembed=colattn_key_pos, cross_only=True
                    )

                if self.heatmap_query_level == 'softmax':
                    self.depth_embed = nn.Sequential(
                        nn.Linear(1, hidden_channel),
                        nn.ReLU(inplace=True),
                        nn.Linear(hidden_channel, hidden_channel),
                    )
                    self.level_classify = nn.Sequential(
                        nn.Linear(hidden_channel, self.level_num),
                        nn.Softmax(dim=-1)
                    )

                if heatmap_query_level == 'cat':
                    self.query_reduce = nn.Sequential(
                        nn.Linear(hidden_channel*(self.level_num+1), hidden_channel),
                        nn.LayerNorm(hidden_channel)
                    )

                if self.cat_point:
                    if cross_heatmap_LN:
                        self.reduce_conv = ConvLN(
                            hidden_channel*2+1, hidden_channel, kernel_size=3, padding=1
                        )
                    else:
                        self.reduce_conv = ConvModule(
                            hidden_channel*2+1, hidden_channel, kernel_size=3, bias=bias, padding=1,
                            conv_cfg=dict(type='Conv2d'), norm_cfg=dict(type='BN2d'),
                        )
                    self.se_block = SE_Block(hidden_channel)
                else:
                    if cross_heatmap_LN:
                        self.reduce_conv = ConvLN(
                            hidden_channel+1, hidden_channel, kernel_size=3, padding=1
                        )
                    else:
                        self.reduce_conv = ConvModule(
                            hidden_channel+1, hidden_channel, kernel_size=3, bias=bias, padding=1,
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
            if self.encode_depth:
                self.shared_conv_img = nn.Identity()
                blocks = [1, 1, 1, 1]
                assert len(blocks) == self.level_num
                self.depth_resnet = DepthEncoderResNet(2, in_channels_img, hidden_channel, depth_layers=blocks)
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
            self.class_encoding = nn.Conv1d(num_classes, hidden_channel, 1)
            self.img_class_encoding = nn.Conv1d(num_classes, hidden_channel, 1)
        else:
            # query feature
            self.pts_query_feat = nn.Parameter(torch.randn(1, hidden_channel, self.num_proposals))
            self.pts_query_pos = nn.Parameter(torch.rand([1, self.num_proposals, 2])*torch.Tensor([x_size, y_size]).reshape(1, 1, 2), requires_grad=learnable_query_pos)

        self.nms_kernel_size = nms_kernel_size
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

        if self.encode_depth:
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

    def forward_single(self, inputs, img_inputs, img_metas, sparse_depth):
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

        if self.encode_depth:
            # sparse_depth = sparse_depth.view(batch_size*self.num_views, self.level_num, 2, sparse_depth.shape[-2], sparse_depth.shape[-1])
            sparse_depth = sparse_depth.view(batch_size*self.num_views, 1, 2, sparse_depth.shape[-2], sparse_depth.shape[-1])
            img_inputs = self.depth_resnet(sparse_depth[:, 0], img_inputs)

        img_metas = self.generate_lidar2cam(img_metas)

        img_feats = []
        if self.with_img:
            for i in range(self.level_num):
                img_inputs_level = img_inputs[i] + self.level_pos[i].reshape(1, self.level_pos[i].shape[0], 1, 1)
                img_feat = self.shared_conv_img(img_inputs_level)
                img_feats.append(img_feat)
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

        if self.with_pts:
            inputs, min_voxel_height, max_voxel_height = inputs[:, :-2], inputs[:, -2], inputs[:, -1]
            lidar_feat = self.shared_conv(inputs)  # [BS, C, H, W]

            #################################
            # image to BEV
            #################################
            lidar_feat_flatten = lidar_feat.view(batch_size, lidar_feat.shape[1], -1)  # [BS, C, H*W]
            bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)  # [BS, H*W, 2]

            if self.initialize_by_heatmap:
                if self.with_img and self.cross_heatmap:
                    img_feat_cross = []
                    for level in range(self.level_num):
                        if self.cross_heatmap_stop_grad:
                            img_feat_cross.append(img_feats[level].detach().clone())
                        else:
                            img_feat_cross.append(img_feats[level].clone())
                else:
                    img_feat_cross = None

                heatmap, dense_heatmap, pts_top_proposals_class, pts_top_proposals_index, fuse_lidar_feat_flatten = self.generate_heatmap(lidar_feat.clone(), min_voxel_height, max_voxel_height, batch_size, img_metas, img_feat_cross)

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
                img_query_cat_encoding = self.img_class_encoding(one_hot.float())
                self.query_labels = pts_top_proposals_class
                pts_query_feat += query_cat_encoding

                pts_query_pos = bev_pos.gather(
                    index=pts_top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]), dim=1
                )  # [BS, num_proposals, 2]
            else:
                pts_query_feat = self.pts_query_feat.repeat(batch_size, 1, 1)  # [BS, C, num_proposals]
                pts_query_pos = self.pts_query_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)  # [BS, num_proposals, 2]

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


        img_query_feat, img_query_pos, img_query_view_mask = self.fetch_img_feature(pts_query_pos, pts_ret_dicts, img_feats, img_metas)
        img_query_feat = img_query_cat_encoding + img_query_feat

        #################################
        # transformer decoder layer (img feature as K,V)
        #################################
        if self.with_img:
            # positional encoding for image fusion
            img_query_feat, img_query_pos, img_query_pos_bev, img_ret_dicts = self.img_transformer(img_query_feat, img_query_pos, img_query_view_mask, img_feats, normal_img_feats_pos_stack, img_metas)

            if self.fuse_self:
                all_query_feat, all_query_pos, fusion_ret_dicts = self.fusion_transformer(pts_query_feat, img_query_feat, pts_query_pos)
            else:
                all_query_feat, all_query_pos, fusion_ret_dicts = self.fusion_transformer(pts_query_feat, pts_query_pos, img_query_feat, img_query_pos)

            ret_dicts.extend(fusion_ret_dicts)

        if self.initialize_by_heatmap:
            if self.with_pts:
                ret_dicts[0]['query_heatmap_score'] = heatmap.gather(index=pts_top_proposals_index[:, None, :].expand(-1, self.num_classes, -1), dim=-1)  # [bs, num_classes, num_proposals]
                ret_dicts[0]['dense_heatmap'] = dense_heatmap

        # return all the layer's results for auxiliary superivison
        new_res = {}
        for key in ret_dicts[0].keys():
            if key not in ['dense_heatmap', 'query_heatmap_score']:
                new_res[key] = torch.cat([ret_dict[key] for ret_dict in ret_dicts], dim=-1)
            else:
                new_res[key] = ret_dicts[0][key]
        if self.with_img:
            for key in img_ret_dicts[0].keys():
                new_res[key] = torch.cat([ret_dict[key] for ret_dict in img_ret_dicts], dim=-1)

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

    def fetch_img_feature(self, pts_query_pos, pts_ret_dicts, img_feats, img_metas):
        bs = pts_query_pos.shape[0]

        preds_dict = pts_ret_dicts[-1]
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

        query_view_mask = torch.zeros([bs, self.num_proposals, self.num_views]).to(pts_query_pos.device)
        query_pos = pts_query_pos * self.test_cfg['out_size_factor'] * \
                               self.test_cfg['voxel_size'][0] + self.test_cfg['pc_range'][0]  # [bs, num_proposals, 2]
        query_pos_3d = torch.cat([query_pos, height.permute(0, 2, 1)], dim=2)  # [bs, num_proposals, 3]

        img_query_feat = torch.zeros([bs, self.num_proposals, self.num_views, self.hidden_channel])  # [bs, num_proposals, num_view, C]
        img_query_feat = img_query_feat.to(query_pos_3d.device)

        for sample_idx in range(bs):
            bboxes_tensor = boxes_dict[sample_idx]['bboxes']
            bboxes = LiDARInstance3DBoxes(bboxes_tensor, box_dim=bboxes_tensor.shape[-1], origin=(0.5, 0.5, 0))

            corners = bboxes.corners  # [num_proposals, 8, 3]

            centers = bboxes.gravity_center[:, None]  # [num_proposals, 1, 3]
            corners = torch.cat([centers, corners], dim=1)

            corners_pad = torch.cat([corners, torch.ones_like(corners[...,:1])], dim=-1)  # [num_proposals, 9, 4]
            corners_pad = corners_pad[:, None]  # [num_proposals, 1, 9, 4]

            lidar2img_rt = img_metas[sample_idx]['lidar2img_rt']  # [num_views, 4, 4]
            lidar2img_rt = torch.Tensor(lidar2img_rt).to(corners_pad.device)[None]  # [1, num_views, 4, 4]

            corners_img = torch.matmul(corners_pad, lidar2img_rt.transpose(3, 2))  # [num_proposals, num_views, 9, 4]

            corners_img[..., 2:3] = torch.clamp(corners_img[..., 2:3], min=1e-4)
            corners_img[..., :2] = corners_img[..., :2] / corners_img[..., 2:3]

            centers_img = corners_img[:, :, 0, :3]   # [num_proposals, num_views, 3]
            corners_img = corners_img[:, :, 1:, :3]  # [num_proposals, num_views, 8, 3]

            for view_idx in range(self.num_views):
                if 'valid_shape' in img_metas[sample_idx]:
                    valid_shape = img_metas[sample_idx]['valid_shape'][view_idx]
                    img_valid_w = valid_shape[0]
                    img_valid_h = valid_shape[1]
                else:
                    valid_shape = img_metas[sample_idx]['input_shape']
                    img_valid_h = valid_shape[0]
                    img_valid_w = valid_shape[1]

                view_mask = (centers_img[:, view_idx, 0] > 0) & (centers_img[:, view_idx, 0] < img_valid_w) & \
                            (centers_img[:, view_idx, 1] > 0) & (centers_img[:, view_idx, 1] < img_valid_h)  # [num_proposals, ]

                if view_mask.sum() == 0:
                    continue
                query_view_mask[sample_idx, view_mask, view_idx] = 1

                corners_view = corners_img[view_mask, view_idx, :, :2].clone()  # [num_on_image, 8, 2]
                corners_view[..., 0] = torch.clamp(corners_view[..., 0], min=0, max=img_valid_w)
                corners_view[..., 1] = torch.clamp(corners_view[..., 1], min=0, max=img_valid_h)

                box_max = torch.max(corners_view, dim=1)[0]  # [num_on_image, 2]
                box_min = torch.min(corners_view, dim=1)[0]  # [num_on_image, 2]

                length = torch.max(box_max - box_min, dim=1)[0]  # [num_on_image, ]

                level_id = torch.log2(length / 48 + 1e-5).long() + 1
                level_id = torch.clamp(level_id, min=0, max=self.level_num-1)  # [num_on_image, ]

                centers_view = centers_img[view_mask, view_idx, :2].clone()

                centers_view = centers_view / self.out_size_factor_img / (2 ** level_id[:, None])
                centers_view = centers_view.long()  # [num_on_image, 2]

                img_query_feat_view = torch.zeros([view_mask.sum(), self.hidden_channel]).to(centers_view.device)  # [num_on_image, C]

                for lvl in range(self.level_num):
                    level_mask = level_id == lvl
                    img_query_feat_view[level_mask] = img_feats[lvl][sample_idx*self.num_views+view_idx, :, centers_view[level_mask, 1], centers_view[level_mask, 0]].T  # [C, num_on_image_level]

                img_query_feat[sample_idx, view_mask, view_idx] = img_query_feat_view.clone()

        img_query_feat = torch.sum(img_query_feat, dim=2)  # [bs, num_proposals, C]
        img_query_feat = img_query_feat / (torch.sum(query_view_mask, dim=2, keepdim=True) + 1e-5)
        img_query_feat = img_query_feat.permute(0, 2, 1)

        return img_query_feat, query_pos_3d, query_view_mask

    def generate_lidar2cam(self, img_metas):
        for img_meta in img_metas:
            lidar2cam_r = np.array(img_meta['lidar2cam_r'])
            lidar2cam_t = np.array(img_meta['lidar2cam_t'])
            lidar2cam_rt = np.eye(4)[None].repeat(self.num_views, axis=0)
            lidar2cam_rt[:, :3, :3] = lidar2cam_r
            lidar2cam_rt[:, :3, 3] = lidar2cam_t
            intrinsic = np.array(img_meta['cam_intrinsic'])
            viewpad = np.eye(4)[None].repeat(self.num_views, axis=0)
            viewpad[:, :intrinsic.shape[1], :intrinsic.shape[2]] = intrinsic
            lidar2img_rt = np.matmul(viewpad, lidar2cam_rt)

            img_meta['lidar2img_rt'] = lidar2img_rt
        return img_metas

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

    def generate_heatmap_range(self, lidar_feat, img_feat, min_voxel_height, max_voxel_height, img_metas):
        # img_feat [bs*num_view, C, img_h, img_w]
        # lidar_feat [BS, C, H, W]

        batch_size = lidar_feat.shape[0]
        H, W = lidar_feat.shape[2], lidar_feat.shape[3]
        min_voxel_height = min_voxel_height.view(batch_size, H*W)
        max_voxel_height = max_voxel_height.view(batch_size, H*W)

        valid_height_mask = max_voxel_height > -50

        for lvl in range(self.level_num):
            img_h_lvl, img_w_lvl = img_feat[lvl].shape[-2], img_feat[lvl].shape[-1]
            img_feat[lvl] = self.heatmap_img_proj(img_feat[lvl].permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

            img_feat[lvl] = img_feat[lvl].view(batch_size, self.num_views, self.hidden_channel, img_h_lvl, img_w_lvl)

        lidar_feat = self.heatmap_pts_proj(lidar_feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        lidar_feat_flatten = lidar_feat.reshape(batch_size, self.hidden_channel, H*W)  # [bs, C, H*W]

        lidar_feat_output = torch.zeros(batch_size, self.hidden_channel, H*W).to(lidar_feat.device)
        lidar_feat_count = torch.zeros(batch_size, 1, H*W).to(lidar_feat.device)

        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)
        query_pos_realmetric = bev_pos.permute(0, 2, 1) * self.test_cfg['out_size_factor'] * \
                               self.test_cfg['voxel_size'][0] + self.test_cfg['pc_range'][0]  # (bs, 2, H*W)

        range_values = torch.arange(self.range_num).type_as(query_pos_realmetric).unsqueeze(0)  # (1, num_int)
        for sample_idx in range(batch_size):
            min_query_pos_3d = torch.cat(
                [query_pos_realmetric[sample_idx], min_voxel_height[sample_idx, None]], dim=0
            ).detach().clone()  # (3, N)

            max_query_pos_3d = torch.cat(
                [query_pos_realmetric[sample_idx], max_voxel_height[sample_idx, None]], dim=0
            ).detach().clone()  # (3, N)

            # img_scale = np.array([img_scale_factor[0], img_scale_factor[1], img_scale_factor[0], img_scale_factor[1]])
            lidar2cam_r = np.array(img_metas[sample_idx]['lidar2cam_r'])
            lidar2cam_t = np.array(img_metas[sample_idx]['lidar2cam_t'])
            lidar2cam_rt = np.eye(4)[None].repeat(self.num_views, axis=0)
            lidar2cam_rt[:, :3, :3] = lidar2cam_r
            lidar2cam_rt[:, :3, 3] = lidar2cam_t
            intrinsic = np.array(img_metas[sample_idx]['cam_intrinsic'])
            viewpad = np.eye(4)[None].repeat(self.num_views, axis=0)
            viewpad[:, :intrinsic.shape[1], :intrinsic.shape[2]] = intrinsic
            lidar2img_rt = np.matmul(viewpad, lidar2cam_rt)
            lidar2img_rt = torch.from_numpy(lidar2img_rt).to(bev_pos.device).float()

            min_points = min_query_pos_3d.clone().T  # [N, 3]
            max_points = max_query_pos_3d.clone().T  # [N, 3]
            num_points = min_points.shape[0]

            # dump_data = []
            for view_idx in range(self.num_views):
                # view_dump_data = {}
                valid_shape = img_metas[sample_idx]['valid_shape'][view_idx] / self.out_size_factor_img \
                    if 'valid_shape' in img_metas[sample_idx].keys() else [img_feat[0].shape[-1], img_feat[0].shape[-2]]
                red_img_w = int(valid_shape[0])
                red_img_h = int(valid_shape[1])

                min_pts_4d = torch.cat([min_points, min_points.new_ones(size=(num_points, 1))], dim=-1)  # [N, 4]
                min_pts_2d = min_pts_4d @ lidar2img_rt[view_idx].t()

                min_pts_2d[:, 2] = torch.clamp(min_pts_2d[:, 2], min=1e-5)
                min_pts_2d[:, 0] /= min_pts_2d[:, 2]
                min_pts_2d[:, 1] /= min_pts_2d[:, 2]

                min_img_coors = min_pts_2d[:, 0:2]

                # grid sample, the valid grid range should be in [-1,1]
                min_coor_x, min_coor_y = torch.split(min_img_coors, 1, dim=1)  # each is Nx1

                min_center_xs = min_coor_x / self.out_size_factor_img
                min_center_ys = min_coor_y / self.out_size_factor_img

                min_center_xs = min_center_xs.squeeze()  # [num_proposal_on_img, ]
                min_center_ys = min_center_ys.squeeze()  # [num_proposal_on_img, ]

                max_pts_4d = torch.cat([max_points, max_points.new_ones(size=(num_points, 1))], dim=-1)  # [N, 4]
                max_pts_2d = max_pts_4d @ lidar2img_rt[view_idx].t()

                max_pts_2d[:, 2] = torch.clamp(max_pts_2d[:, 2], min=1e-5)
                max_pts_2d[:, 0] /= max_pts_2d[:, 2]
                max_pts_2d[:, 1] /= max_pts_2d[:, 2]

                max_img_coors = max_pts_2d[:, 0:2]

                # grid sample, the valid grid range should be in [-1,1]
                max_coor_x, max_coor_y = torch.split(max_img_coors, 1, dim=1)  # each is Nx1

                max_center_xs = max_coor_x / self.out_size_factor_img
                max_center_ys = max_coor_y / self.out_size_factor_img

                max_center_xs = max_center_xs.squeeze()  # [num_proposal_on_img, ]
                max_center_ys = max_center_ys.squeeze()  # [num_proposal_on_img, ]

                min_on_the_image = (min_center_xs > 0) & (min_center_xs < red_img_w) & (min_center_ys > 0)
                max_on_the_image = (max_center_xs > 0) & (max_center_xs < red_img_w) & (max_center_ys < red_img_h)
                on_the_image = (min_on_the_image | max_on_the_image) & valid_height_mask[sample_idx]

                min_center_xs = min_center_xs[on_the_image]
                min_center_ys = min_center_ys[on_the_image]

                max_center_xs = max_center_xs[on_the_image]
                max_center_ys = max_center_ys[on_the_image]

                img_feat_lvls = []
                lins_center_lvls = []
                for lvl in range(self.level_num):
                    min_center_xs_lvl_raw = min_center_xs / (2 ** lvl) - 0.5
                    min_center_ys_lvl_raw = min_center_ys / (2 ** lvl) - 0.5

                    max_center_xs_lvl_raw = max_center_xs / (2 ** lvl) - 0.5
                    max_center_ys_lvl_raw = max_center_ys / (2 ** lvl) - 0.5

                    min_center_xs_lvl = torch.clamp(min_center_xs_lvl_raw , min=0, max=red_img_w / (2 ** lvl) - 1)
                    max_center_xs_lvl = torch.clamp(max_center_xs_lvl_raw , min=0, max=red_img_w / (2 ** lvl) - 1)

                    min_center_ys_lvl = torch.clamp(min_center_ys_lvl_raw , min=0, max=red_img_h / (2 ** lvl) - 1)
                    max_center_ys_lvl = torch.clamp(max_center_ys_lvl_raw , min=0, max=red_img_h / (2 ** lvl) - 1)

                    x_interval = (max_center_xs_lvl - min_center_xs_lvl) / (self.range_num - 1)  # (num_proposal_on_img, )
                    y_interval = (max_center_ys_lvl - min_center_ys_lvl) / (self.range_num - 1)  # (num_proposal_on_img, )

                    lins_center_xs_lvl = min_center_xs_lvl.unsqueeze(-1) + x_interval.unsqueeze(-1) * range_values  # (num_proposal_on_img, num_int)
                    lins_center_ys_lvl = min_center_ys_lvl.unsqueeze(-1) + y_interval.unsqueeze(-1) * range_values  # (num_proposal_on_img, num_int)

                    w11 = (torch.ceil(lins_center_xs_lvl) - lins_center_xs_lvl) * (torch.ceil(lins_center_ys_lvl) - lins_center_ys_lvl)
                    w22 = (lins_center_xs_lvl - torch.floor(lins_center_xs_lvl)) * (lins_center_ys_lvl - torch.floor(lins_center_ys_lvl))
                    w12 = (lins_center_xs_lvl - torch.floor(lins_center_xs_lvl)) * (torch.ceil(lins_center_ys_lvl) - lins_center_ys_lvl)
                    w21 = (torch.ceil(lins_center_xs_lvl) - lins_center_xs_lvl) * (lins_center_ys_lvl - torch.floor(lins_center_ys_lvl))

                    xs_1 = torch.floor(lins_center_xs_lvl).long()
                    ys_1 = torch.floor(lins_center_ys_lvl).long()
                    xs_2 = torch.ceil(lins_center_xs_lvl).long()
                    ys_2 = torch.ceil(lins_center_ys_lvl).long()

                    img_feat_lvl_11 = img_feat[lvl][sample_idx, view_idx, :, ys_1, xs_1]  # [C, num_proposal_on_img, num_int]
                    img_feat_lvl_12 = img_feat[lvl][sample_idx, view_idx, :, ys_1, xs_2]  # [C, num_proposal_on_img, num_int]
                    img_feat_lvl_21 = img_feat[lvl][sample_idx, view_idx, :, ys_2, xs_1]  # [C, num_proposal_on_img, num_int]
                    img_feat_lvl_22 = img_feat[lvl][sample_idx, view_idx, :, ys_2, xs_2]  # [C, num_proposal_on_img, num_int]

                    img_feat_lvl = w11 * img_feat_lvl_11 + w12 * img_feat_lvl_12 + w21 * img_feat_lvl_21 + w22 * img_feat_lvl_22
                    img_feat_lvl = img_feat_lvl.permute(1, 0, 2)  # [num_proposal_on_img, C, num_int]
                    img_feat_lvls.append(img_feat_lvl)

                    lins_center_lvl = torch.stack([lins_center_xs_lvl, lins_center_ys_lvl], dim=-1)  # (num_proposal_on_img, num_int, 2)
                    lins_center_lvl = (lins_center_lvl + 0.5) * (2 ** lvl) * self.out_size_factor_img
                    lins_center_lvls.append(lins_center_lvl)

                img_feat_lvls = torch.cat(img_feat_lvls, dim=2)  # [num_proposal_on_img, C, num_int*level_num]
                lins_center_lvls = torch.cat(lins_center_lvls, dim=1)  # [num_proposal_on_img, num_int*level_num, 2]
                depth_lvls = min_pts_2d[on_the_image, 2].unsqueeze(-1).unsqueeze(-1).repeat(1, self.range_num*self.level_num, 1)
                lins_center_4d = torch.cat([lins_center_lvls.clone(), depth_lvls, torch.ones_like(depth_lvls)], dim=-1)  # [num_proposal_on_img, num_int*level_num, 4]
                lins_center_4d[..., :2] = lins_center_4d[..., :2] * depth_lvls
                lins_pts_center_4d = lins_center_4d @ torch.inverse(lidar2img_rt[view_idx].t())  # [num_proposal_on_img, num_int*level_num, 4]
                lins_heights = lins_pts_center_4d[..., 2:3]

                lins_center_lvls[..., 0] = inverse_sigmoid(lins_center_lvls[..., 0] / (self.out_size_factor_img * red_img_w))
                lins_center_lvls[..., 1] = inverse_sigmoid(lins_center_lvls[..., 1] / (self.out_size_factor_img * red_img_h))

                lins_img_pos = torch.cat([lins_center_lvls, lins_heights], dim=-1)  # [num_proposal_on_img, num_int*level_num, 3]

                lidar_feat_view = lidar_feat_flatten[sample_idx, :, on_the_image].transpose(1, 0)  # [C, N]
                lidar_depth_view = torch.log(min_pts_2d[on_the_image, 2:3])  # [N, 1]

                if lidar_feat_view.shape[0] > 0:
                    output = self.col_decoder(
                        lidar_feat_view[:, :, None], img_feat_lvls, lidar_depth_view[:, None], lins_img_pos,
                    )  # [N, C, 1]

                    lidar_feat_output[sample_idx, :, on_the_image] += output[..., 0].clone().transpose(1, 0)
                    lidar_feat_count[sample_idx, :, on_the_image] += 1
                else:
                    print("batch size zero")

        lidar_feat_output = lidar_feat_output / (lidar_feat_count + 1e-5)
        lidar_feat_output = lidar_feat_output.reshape(batch_size, lidar_feat_output.shape[1], H, W)
        if self.cat_point:
            lidar_feat_output = torch.cat([lidar_feat_output, lidar_feat], dim=1)
            lidar_feat_output = self.reduce_conv(lidar_feat_output)
            lidar_feat_output = self.se_block(lidar_feat_output)
        else:
            # lidar_feat_output = lidar_feat_output + lidar_feat
            lidar_feat_count = lidar_feat_count.reshape(batch_size, 1, H, W)
            lidar_feat_flag = torch.where(lidar_feat_count>0, torch.ones_like(lidar_feat_count), torch.zeros_like(lidar_feat_count))
            lidar_feat_output = lidar_feat_output + (1 - lidar_feat_flag) * lidar_feat
            lidar_feat_output = torch.cat([lidar_feat_output, lidar_feat_flag], dim=1)
            lidar_feat_output = self.reduce_conv(lidar_feat_output)

        heatmap_output = self.cross_heatmap_head(lidar_feat_output.contiguous())

        return heatmap_output, lidar_feat_output.reshape(batch_size, self.hidden_channel, H*W)

    def generate_heatmap_deform(self, lidar_feat, img_feat, voxel_height, img_metas):
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
        # lidar_feat_output = lidar_feat_flatten.clone()  # [bs, C, H*W]
        lidar_feat_output = torch.zeros(batch_size, self.hidden_channel, H*W).to(lidar_feat.device)
        lidar_feat_count = torch.zeros(batch_size, 1, H*W).to(lidar_feat.device)

        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)
        query_pos_realmetric = bev_pos.permute(0, 2, 1) * self.test_cfg['out_size_factor'] * \
                               self.test_cfg['voxel_size'][0] + self.test_cfg['pc_range'][0]  # (bs, 2, H*W)

        for sample_idx in range(batch_size):
            query_pos_3d = torch.cat(
                [query_pos_realmetric[sample_idx], voxel_height[sample_idx, None]],
                dim=0).detach().clone()  # (3, N)

            # lidar2img_rt = torch.Tensor(img_metas[sample_idx]['lidar2img']).to(bev_pos.device)
            # img_scale_factor = (
            #     lidar2img_rt.new_tensor(
            #         img_metas[sample_idx]['scale_factor'][:2] if 'scale_factor' in img_metas[sample_idx].keys() else [
            #             1.0, 1.0])
            # )
            # # transform point clouds back to original coordinate system by reverting the data augmentation
            # if batch_size == 1:  # skip during inference to save time
            #     points = query_pos_3d.clone().T  # [N, 3]
            # else:
            #     points = apply_3d_transformation(query_pos_3d.clone().T, 'LIDAR', img_metas[sample_idx], reverse=True).detach()

            lidar2cam_r = np.array(img_metas[sample_idx]['lidar2cam_r'])
            lidar2cam_t = np.array(img_metas[sample_idx]['lidar2cam_t'])
            lidar2cam_rt = np.eye(4)[None].repeat(self.num_views, axis=0)
            lidar2cam_rt[:, :3, :3] = lidar2cam_r
            lidar2cam_rt[:, :3, 3] = lidar2cam_t
            intrinsic = np.array(img_metas[sample_idx]['cam_intrinsic'])
            viewpad = np.eye(4)[None].repeat(self.num_views, axis=0)
            viewpad[:, :intrinsic.shape[1], :intrinsic.shape[2]] = intrinsic
            lidar2img_rt = np.matmul(viewpad, lidar2cam_rt)
            lidar2img_rt = torch.from_numpy(lidar2img_rt).to(bev_pos.device).float()

            img_scale_factor = lidar2img_rt.new_tensor([1.0, 1.0])

            points = query_pos_3d.clone().T

            num_points = points.shape[0]

            # dump_data = []
            for view_idx in range(self.num_views):
                # view_dump_data = {}
                valid_shape = img_metas[sample_idx]['valid_shape'][view_idx] / self.out_size_factor_img \
                    if 'valid_shape' in img_metas[sample_idx].keys() else [img_feat[0].shape[-1], img_feat[0].shape[-2]]
                red_img_w = int(valid_shape[0])
                red_img_h = int(valid_shape[1])

                img_h, img_w = img_feat[0].shape[-2], img_feat[0].shape[-1]

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
                center_ys = coor_y / self.out_size_factor_img

                center_xs = center_xs.squeeze()  # [num_proposal_on_img, ]
                center_ys = center_ys.squeeze()  # [num_proposal_on_img, ]

                on_the_image = (center_xs >= 0) & (center_xs < red_img_w) & (center_ys >= 0) & (center_ys < red_img_h) & valid_height_mask[sample_idx]
                center_xs = center_xs[on_the_image]
                center_ys = center_ys[on_the_image]
                depth = torch.log(pts_2d[on_the_image, 2])

                query_pos_3d_view = query_pos_3d[:, on_the_image].T
                query_pos_3d_view[:, :2] = query_pos_3d_view[:, :2] / 10

                reference_points = torch.stack([center_xs/img_w, center_ys/img_h], dim=-1)

                lidar_feat_view = lidar_feat_flatten[sample_idx, :, on_the_image]  # [N, C]
                lidar_feat_pos_view = inverse_sigmoid(reference_points)

                center_xs_long = center_xs.long()
                center_ys_long = center_ys.long()
                img_feat_view = []

                if self.heatmap_query_level != 'lidar_only':
                    for lvl in range(self.level_num):
                        # center_xs_lvl = center_xs / (2 ** lvl) - 0.5
                        # center_ys_lvl = center_ys / (2 ** lvl) - 0.5
                        # center_xs_lvl = torch.clamp(center_xs_lvl, min=0, max=red_img_w / (2 ** lvl) - 1)
                        # center_ys_lvl = torch.clamp(center_ys_lvl, min=0, max=red_img_h / (2 ** lvl) - 1)
                        #
                        # w11 = (torch.ceil(center_xs_lvl) - center_xs_lvl) * (torch.ceil(center_ys_lvl) - center_ys_lvl)
                        # w22 = (center_xs_lvl - torch.floor(center_xs_lvl)) * (center_ys_lvl - torch.floor(center_ys_lvl))
                        # w12 = (center_xs_lvl - torch.floor(center_xs_lvl)) * (torch.ceil(center_ys_lvl) - center_ys_lvl)
                        # w21 = (torch.ceil(center_xs_lvl) - center_xs_lvl) * (center_ys_lvl - torch.floor(center_ys_lvl))
                        #
                        # xs_1 = torch.floor(center_xs_lvl).long()
                        # ys_1 = torch.floor(center_ys_lvl).long()
                        # xs_2 = torch.ceil(center_xs_lvl).long()
                        # ys_2 = torch.ceil(center_ys_lvl).long()
                        #
                        # img_feat_lvl_11 = img_feat[lvl][sample_idx, view_idx, :, ys_1, xs_1]  # [C, num_proposal_on_img, num_int]
                        # img_feat_lvl_12 = img_feat[lvl][sample_idx, view_idx, :, ys_1, xs_2]  # [C, num_proposal_on_img, num_int]
                        # img_feat_lvl_21 = img_feat[lvl][sample_idx, view_idx, :, ys_2, xs_1]  # [C, num_proposal_on_img, num_int]
                        # img_feat_lvl_22 = img_feat[lvl][sample_idx, view_idx, :, ys_2, xs_2]  # [C, num_proposal_on_img, num_int]
                        #
                        # img_feat_lvl = w11 * img_feat_lvl_11 + w12 * img_feat_lvl_12 + w21 * img_feat_lvl_21 + w22 * img_feat_lvl_22

                        img_feat_lvl = img_feat[lvl][sample_idx, view_idx, :, center_ys_long//(2**lvl), center_xs_long//(2**lvl)]  # [C, N]
                        img_feat_view.append(img_feat_lvl)

                    if self.heatmap_query_level == 'max':
                        img_feat_view = torch.stack(img_feat_view, dim=0)

                        img_feat_view = torch.max(img_feat_view, dim=0)[0]  # [C, N]
                        lidar_feat_view = lidar_feat_view + img_feat_view
                    elif self.heatmap_query_level == 'softmax':
                        img_feat_view = torch.stack(img_feat_view, dim=0)  # [L, C, N]
                        depth_embedding = self.depth_embed(depth[:, None])  # [N, C]
                        level_weight = self.level_classify(depth_embedding + lidar_feat_view.transpose(1, 0))  # [N, L]
                        level_weight = level_weight.unsqueeze(-1)  # [N, L, 1]
                        img_feat_view = img_feat_view.permute(2, 1, 0)  # [N, C, L]
                        img_feat_view = torch.matmul(img_feat_view, level_weight).squeeze(-1)  # [N, C, 1]
                        lidar_feat_view = lidar_feat_view + img_feat_view.transpose(1, 0)

                    else:
                        assert self.heatmap_query_level == 'cat'
                        lidar_feat_view = torch.cat(img_feat_view + [lidar_feat_view], dim=0)
                        lidar_feat_view = self.query_reduce(lidar_feat_view.transpose(1, 0)).transpose(1, 0)

                lidar_feat_pos_view = torch.cat([lidar_feat_pos_view, depth[:, None]], dim=1)
                reference_points = reference_points[:, None].repeat(1, self.level_num, 1)

                output = self.col_decoder(
                    lidar_feat_view[None], img_feats_stack[sample_idx:sample_idx + 1, view_idx],
                    lidar_feat_pos_view[None], normal_img_feats_pos_stack,
                    reference_points=reference_points[None],
                    level_start_index=level_start_index, spatial_shapes=spatial_shapes
                )  # [1, C, N]

                lidar_feat_output[sample_idx, :, on_the_image] = output[0].clone()
                lidar_feat_count[sample_idx, :, on_the_image] += 1

                # lidar_feat_output[sample_idx, :-1, on_the_image] = output[0].clone()
                # lidar_feat_output[sample_idx, -1, on_the_image] = 1

        # lidar_feat_output = lidar_feat_output / (lidar_feat_count + 1e-5)
        lidar_feat_output = lidar_feat_output.reshape(batch_size, lidar_feat_output.shape[1], H, W)
        if self.cat_point:
            lidar_feat_output = torch.cat([lidar_feat_output, lidar_feat], dim=1)
            lidar_feat_output = self.reduce_conv(lidar_feat_output)
            lidar_feat_output = self.se_block(lidar_feat_output)
        else:
            # lidar_feat_output = self.reduce_conv(lidar_feat_output)
            lidar_feat_count = lidar_feat_count.reshape(batch_size, 1, H, W)
            lidar_feat_flag = torch.where(lidar_feat_count>0, torch.ones_like(lidar_feat_count), torch.zeros_like(lidar_feat_count))
            lidar_feat_output = lidar_feat_output + (1 - lidar_feat_flag) * lidar_feat
            lidar_feat_output = torch.cat([lidar_feat_output, lidar_feat_flag], dim=1)
            lidar_feat_output = self.reduce_conv(lidar_feat_output)

        heatmap_output = self.cross_heatmap_head(lidar_feat_output.contiguous())

        return heatmap_output, lidar_feat_output.reshape(batch_size, self.hidden_channel, H*W)

    def generate_heatmap(self, lidar_feat, min_voxel_height, max_voxel_height, batch_size, img_metas, img_feat=None):
        dense_heatmap = self.heatmap_head(lidar_feat)  # [BS, num_class, H, W]
        fuse_lidar_feature_flatten = None
        if img_feat is None:
            heatmap = dense_heatmap.detach().sigmoid()  # [BS, num_class, H, W]
        else:
            if self.cross_type == 'deform_height':
                voxel_height = (min_voxel_height + max_voxel_height) / 2
                dense_heatmap_cross, fuse_lidar_feature_flatten = self.generate_heatmap_deform(lidar_feat, img_feat, voxel_height, img_metas)
            else:
                dense_heatmap_cross, fuse_lidar_feature_flatten = self.generate_heatmap_range(lidar_feat, img_feat, min_voxel_height, max_voxel_height, img_metas)

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

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels, gt_img_centers_2d, gt_bboxes_cam_3d, gt_visible, gt_bboxes_lidar, preds_dict, img_metas):
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

        if self.with_img:
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
            if self.with_img:
                return labels, label_weights, bbox_targets, bbox_weights, ious, num_pos_layer, matched_ious, heatmap, \
                   labels_view, label_weights_view, bbox_targets_view, bbox_weights_view, ious_view, num_pos_layer_view, \
                    matched_ious_view
            else:
                return labels, label_weights, bbox_targets, bbox_weights, ious, num_pos_layer, matched_ious, heatmap
        else:
            if self.with_img:
                return labels, label_weights, bbox_targets, bbox_weights, ious, num_pos_layer, matched_ious, \
                    labels_view, label_weights_view, bbox_targets_view, bbox_weights_view, ious_view, num_pos_layer_view, \
                    matched_ious_view
            else:
                return labels, label_weights, bbox_targets, bbox_weights, ious, num_pos_layer, matched_ious

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
        # each layer should do label assign seperately.
        # if self.auxiliary:
        #     num_layer = self.num_pts_decoder_layers + self.num_fusion_decoder_layers
        # else:
        #     num_layer = 1
        num_layer = self.num_pts_decoder_layers + self.num_fusion_decoder_layers

        assign_result_list = []
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
            assign_result_list.append(assign_result)

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

        # combine assign result of each layer
        # assign_result_ensemble = AssignResult(
        #     num_gts=sum([res.num_gts for res in assign_result_list]),
        #     gt_inds=torch.cat([res.gt_inds for res in assign_result_list]),
        #     max_overlaps=torch.cat([res.max_overlaps for res in assign_result_list]),
        #     labels=torch.cat([res.labels for res in assign_result_list]),
        # )
        # sampling_result = self.bbox_sampler.sample(assign_result_ensemble, bboxes_tensor, gt_bboxes_tensor)
        # pos_inds = sampling_result.pos_inds
        # neg_inds = sampling_result.neg_inds

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

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # torch.save(pos_inds, 'vis/pos_inds_%d.pt'%batch_idx)
        # torch.save(bboxes_tensor, 'vis/bboxes_tensor_%d.pt'%batch_idx)
        # torch.save(pos_gt_bboxes, 'vis/gt_bboxes_tensor_%d.pt'%batch_idx)

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

        # torch.save(pos_inds, 'vis/pos_view_inds_%d.pt'%batch_idx)
        # torch.save(bboxes_tensor, 'vis/bboxes_view_tensor_%d.pt'%batch_idx)
        # torch.save(sampling_result.pos_gt_bboxes, 'vis/gt_bboxes_view_tensor_%d.pt'%batch_idx)

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
    def loss(self, gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels, gt_pts_centers_2d, gt_img_centers_2d, gt_bboxes_cam_3d, gt_visible_3d, gt_bboxes_lidar, img_metas, preds_dicts, **kwargs):
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
                labels_view, label_weights_view, bbox_targets_view, bbox_weights_view, ious_view, \
                num_pos_layer_view, matched_ious_view = self.get_targets(gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels, gt_img_centers_2d, gt_bboxes_cam_3d, gt_visible_3d, gt_bboxes_lidar, preds_dicts[0], img_metas)
            else:
                labels, label_weights, bbox_targets, bbox_weights, ious, num_pos_layer, matched_ious, \
                labels_view, label_weights_view, bbox_targets_view, bbox_weights_view, ious_view, \
                num_pos_layer_view, matched_ious_view = self.get_targets(gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels, gt_img_centers_2d, gt_bboxes_cam_3d, gt_visible_3d, preds_dicts[0], img_metas)
        else:
            if self.initialize_by_heatmap:
                labels, label_weights, bbox_targets, bbox_weights, ious, num_pos_layer, matched_ious, heatmap = self.get_targets(gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels, gt_visible_3d, preds_dicts[0], img_metas)
            else:
                labels, label_weights, bbox_targets, bbox_weights, ious, num_pos_layer, matched_ious = self.get_targets(gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels, gt_visible_3d, preds_dicts[0], img_metas)        # if hasattr(self, 'on_the_image_mask'):

        preds_dict = preds_dicts[0][0]
        loss_dict = dict()

        if self.initialize_by_heatmap:
            # compute heatmap loss
            loss_heatmap = self.loss_heatmap(clip_sigmoid(preds_dict['dense_heatmap']), heatmap, avg_factor=max(heatmap.eq(1).float().sum().item(), 1))
            loss_dict['loss_heatmap'] = loss_heatmap

        # compute loss for each layer
        start = 0
        num_pos_layer = np.sum(num_pos_layer, axis=0)
        num_pos_layer_view = np.sum(num_pos_layer_view, axis=0)

        num_layer = self.num_pts_decoder_layers + self.num_fusion_decoder_layers
        for idx_layer in range(num_layer):
            layer_num_proposals = self.num_proposals
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

        if self.with_img:
            start = 0
            for idx_layer in range(self.num_img_decoder_layers):
                layer_num_proposals = self.num_proposals

                layer_labels_view = labels_view[..., start:start + layer_num_proposals].reshape(-1)
                layer_label_weights_view = label_weights_view[..., start:start + layer_num_proposals].reshape(-1)
                layer_cls_score = preds_dict['heatmap_view'][..., start:start + layer_num_proposals].permute(0, 2, 1).reshape(-1, self.num_classes)

                layer_loss_cls_view = self.loss_cls(
                    layer_cls_score, layer_labels_view, layer_label_weights_view, avg_factor=max(num_pos_layer_view[0], 1)
                )

                layer_center_view = preds_dict['center_view'][..., start:start + layer_num_proposals]
                layer_height_view = preds_dict['height_view'][..., start:start + layer_num_proposals]
                layer_rot_view = preds_dict['rot_view'][..., start:start + layer_num_proposals]
                layer_dim_view = preds_dict['dim_view'][..., start:start + layer_num_proposals]
                layer_preds_view = torch.cat([layer_center_view, layer_height_view, layer_dim_view, layer_rot_view],
                                       dim=1).permute(0, 2, 1)  # [BS, num_proposals, code_size]
                if 'vel' in preds_dict.keys():
                    layer_vel_view = preds_dict['vel_view'][..., start:start + layer_num_proposals]
                    layer_preds_view = torch.cat([layer_center_view, layer_height_view, layer_dim_view, layer_rot_view, layer_vel_view],
                                      dim=1).permute(0, 2, 1)  # [BS, num_proposals, code_size]
                code_weights = self.train_cfg.get('code_weights', None)
                layer_reg_weights_view = bbox_weights_view[..., start: start + layer_num_proposals]
                layer_reg_weights_view = layer_reg_weights_view * layer_reg_weights_view.new_tensor(code_weights)
                layer_bbox_targets_view = bbox_targets_view[:, start:start + layer_num_proposals, :]

                layer_loss_bbox_view = self.loss_bbox(layer_preds_view, layer_bbox_targets_view, layer_reg_weights_view, avg_factor=max(num_pos_layer_view[0], 1))

                layer_match_ious_view = matched_ious_view[..., start:start + layer_num_proposals]
                layer_match_ious_view = torch.sum(layer_match_ious_view * (layer_match_ious_view >= 0), dim=-1) / torch.sum(
                    layer_match_ious_view >= 0, dim=-1)
                layer_match_ious_view = torch.mean(layer_match_ious_view)

                loss_dict['view_loss_cls'] = layer_loss_cls_view
                loss_dict['view_loss_bbox'] = layer_loss_bbox_view
                loss_dict['view_matched_ious'] = layer_match_ious_view

                start += layer_num_proposals

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
            layer_num_proposal = self.num_proposals

            batch_size = preds_dict[0]['heatmap'].shape[0]

            batch_score_raw = preds_dict[0]['heatmap'][..., -layer_num_proposal:].sigmoid()

            one_hot = F.one_hot(self.query_labels, num_classes=self.num_classes).permute(0, 2, 1)
            query_heatmap_score = preds_dict[0]['query_heatmap_score'] * one_hot

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
        return self.num_proposals