import copy
import numpy as np
import torch
from mmcv.cnn import ConvModule, build_conv_layer, kaiming_init
from mmcv.runner import force_fp32
from torch import nn
import torch.nn.functional as F

from mmdet3d.core import (circle_nms, draw_heatmap_gaussian, gaussian_radius,
                          xywhr2xyxyr, limit_period, PseudoSampler)
from mmdet3d.models.builder import HEADS, build_loss
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu
from mmdet3d.models.utils import clip_sigmoid, inverse_sigmoid
from mmdet3d.models.fusion_layers import apply_3d_transformation
from mmdet.core import build_bbox_coder, multi_apply, build_assigner, build_sampler, AssignResult

from mmdet3d.models.utils import FFN, TransformerDecoderLayer, PositionEmbeddingLearned, PositionEmbeddingLearnedwoNorm, PositionEmbeddingLearnedMulti, PointTransformer2D_3D, ImageTransformer2D_3D, FusionTransformer2D_3D, PointProjection, ImageProjection, ProjectionL2Norm, ProjectionLayerNorm


def denormalize_pos(normal_pos, x_max, y_max):
    max_xy = torch.Tensor([x_max, y_max]).to(normal_pos.device).view(1, 1, 2)
    pos = normal_pos.sigmoid() * max_xy
    return pos


def normalize_pos(pos, x_max, y_max):
    max_xy = torch.Tensor([x_max, y_max]).to(pos.device).view(1, 1, 2)
    normal_pos = pos / max_xy
    return inverse_sigmoid(normal_pos)


@HEADS.register_module()
class ImplicitHead2D_3D(nn.Module):
    def __init__(self,
                 with_pts=False,
                 with_img=False,
                 pts_smca=True,
                 img_smca=True,
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
                 loss_cls_2d=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_bbox_2d=dict(type='L1Loss', reduction='mean'),
                 # others
                 train_cfg=None,
                 test_cfg=None,
                 bbox_coder=None,
                 projection='v1',
                 pos_embed='bev',
                 normal_pos=False,
                 supervision2d=True,
                 stop_grad=False,
                 pe_norm='bn',
        ):
        super(ImplicitHead2D_3D, self).__init__()

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

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)
        self.loss_heatmap = build_loss(loss_heatmap)
        self.loss_cls_2d = build_loss(loss_cls_2d)
        self.loss_bbox_2d = build_loss(loss_bbox_2d)
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
        self.stop_grad = stop_grad
        self.pos_embed = pos_embed
        self.normal_pos = normal_pos
        self.pe_norm = pe_norm

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1

        heads3d = copy.deepcopy(common_heads)
        heads3d.update(dict(heatmap=(self.num_classes, 2)))
        pts_prediction_heads = FFN(hidden_channel, heads3d, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias)
        fusion_prediction_heads = FFN(hidden_channel*2, heads3d, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias)

        heads2d = dict(bbox=(4, 2), cls=(self.num_classes, 2))
        img_prediction_heads = FFN(hidden_channel, heads2d, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias)

        if with_pts:
            pts_query_pos_embed = PositionEmbeddingLearned(2, hidden_channel)
            pts_key_pos_embed = PositionEmbeddingLearned(2, hidden_channel)
            self.point_transformer = PointTransformer2D_3D(
                pts_smca=pts_smca, hidden_channel=hidden_channel, num_heads=num_heads, num_decoder_layers=num_pts_decoder_layers,
                prediction_heads=pts_prediction_heads, ffn_channel=ffn_channel, dropout=dropout, activation=activation, test_cfg=test_cfg,
                query_pos=pts_query_pos_embed, key_pos=pts_key_pos_embed
            )
        if with_img:
            assert learnable_query_pos or with_pts
            if self.pe_norm == 'bn':
                img_query_pos_embed = PositionEmbeddingLearned(2, hidden_channel)
                img_key_pos_embed = PositionEmbeddingLearned(2, hidden_channel)
            else:
                img_query_pos_embed = PositionEmbeddingLearnedwoNorm(2, hidden_channel)
                img_key_pos_embed = PositionEmbeddingLearnedwoNorm(2, hidden_channel)
            self.img_transformer = ImageTransformer2D_3D(
                img_smca=img_smca, hidden_channel=hidden_channel, num_heads=num_heads, num_decoder_layers=num_img_decoder_layers, out_size_factor_img=out_size_factor_img,
                num_views=num_views, prediction_heads=img_prediction_heads, ffn_channel=ffn_channel, dropout=dropout, activation=activation, test_cfg=test_cfg,
                query_pos=img_query_pos_embed, key_pos=img_key_pos_embed, supervision2d=supervision2d
            )
        if with_pts and with_img:
            if self.pos_embed == "bev":
                if self.pe_norm == 'bn':
                    fusion_query_pos = PositionEmbeddingLearned(2, hidden_channel)
                    fusion_key_pos = PositionEmbeddingLearned(96, hidden_channel)
                else:
                    fusion_query_pos = PositionEmbeddingLearnedwoNorm(2, hidden_channel)
                    fusion_key_pos = PositionEmbeddingLearnedwoNorm(96, hidden_channel)
            elif self.pos_embed == "3d":
                if self.pe_norm == 'bn':
                    fusion_query_pos = PositionEmbeddingLearned(3, hidden_channel)
                    fusion_key_pos = PositionEmbeddingLearned(96, hidden_channel)
                else:
                    fusion_query_pos = PositionEmbeddingLearnedwoNorm(3, hidden_channel)
                    fusion_key_pos = PositionEmbeddingLearnedwoNorm(96, hidden_channel)
            elif self.pos_embed == "2d":
                if self.pe_norm == 'bn':
                    fusion_query_pos = PositionEmbeddingLearned(2, hidden_channel)
                    fusion_key_pos = PositionEmbeddingLearned(64, hidden_channel)
                else:
                    fusion_query_pos = PositionEmbeddingLearnedwoNorm(2, hidden_channel)
                    fusion_key_pos = PositionEmbeddingLearnedwoNorm(64, hidden_channel)
            else:
                raise NotImplementedError

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
            elif projection == 'l2norm':
                pts_projection = ProjectionL2Norm(hidden_channel)
                img_projection = ProjectionL2Norm(hidden_channel)
            elif projection == 'layernorm':
                pts_projection = ProjectionLayerNorm(hidden_channel)
                img_projection = ProjectionLayerNorm(hidden_channel)
            elif projection == 'fc':
                pts_projection = ProjectionLayerNorm(hidden_channel, norm=False)
                img_projection = ProjectionLayerNorm(hidden_channel, norm=False)
            self.fusion_transformer = FusionTransformer2D_3D(
                hidden_channel=hidden_channel, num_heads=num_heads, num_decoder_layers=num_fusion_decoder_layers,
                prediction_heads=fusion_prediction_heads, ffn_channel=ffn_channel, dropout=dropout, activation=activation,
                test_cfg=test_cfg, query_pos=fusion_query_pos, key_pos=fusion_key_pos,
                pts_projection=pts_projection, img_projection=img_projection, normal_pos=normal_pos
            )

            if self.initialize_by_heatmap and self.cross_heatmap:
                if self.pe_norm == 'bn':
                    heatmap_pts_pos = PositionEmbeddingLearned(2, hidden_channel)
                    heatmap_img_pos = PositionEmbeddingLearned(64, hidden_channel)
                else:
                    heatmap_pts_pos = PositionEmbeddingLearnedwoNorm(2, hidden_channel)
                    heatmap_img_pos = PositionEmbeddingLearnedwoNorm(64, hidden_channel)

                self.cross_heatmap_decoder = TransformerDecoderLayer(
                    hidden_channel, num_heads, ffn_channel, dropout, activation,
                    self_posembed=heatmap_pts_pos, cross_posembed=heatmap_img_pos, cross_only=True
                )
                self.cross_heatmap_head = self.build_heatmap(hidden_channel, bias, num_classes)
                self.fc = nn.Conv2d(hidden_channel, hidden_channel, kernel_size=3, stride=2, padding=1)

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

        self.init_weights()
        self._init_assigner_sampler()

        # Position Embedding for Cross-Attention, which is re-used during training
        x_size = self.test_cfg['grid_size'][0] // self.test_cfg['out_size_factor']
        y_size = self.test_cfg['grid_size'][1] // self.test_cfg['out_size_factor']
        self.bev_pos = self.create_2D_grid(x_size, y_size)

        if self.initialize_by_heatmap:
            self.heatmap_head = self.build_heatmap(hidden_channel, bias, num_classes)
            self.img_heatmap_head = self.build_heatmap(hidden_channel, bias, num_classes, layer_num=3)
            self.class_encoding = nn.Conv1d(num_classes, hidden_channel, 1)
            self.img_class_encoding = nn.Conv1d(num_classes, hidden_channel, 1)
        else:
            # query feature
            self.pts_query_feat = nn.Parameter(torch.randn(1, hidden_channel, self.num_proposals))
            self.pts_query_pos = nn.Parameter(torch.rand([1, self.num_proposals, 2])*torch.Tensor([x_size, y_size]).reshape(1, 1, 2), requires_grad=learnable_query_pos)

            self.img_query_feat = nn.Parameter(torch.randn(1, hidden_channel, self.num_proposals))
            self.img_query_pos = nn.Parameter(torch.rand([1, self.num_proposals, 2]), requires_grad=learnable_query_pos)
            self.img_query_pos = inverse_sigmoid(self.img_query_pos)

        self.nms_kernel_size = nms_kernel_size
        self.img_nms_kernel_size = img_nms_kernel_size
        self.img_feat_pos = None
        self.img_feat_collapsed_pos = None

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
        if self.with_pts:
            for m in self.point_transformer.parameters():
                if m.dim() > 1:
                    nn.init.xavier_uniform_(m)
        if self.with_img:
            for m in self.img_transformer.parameters():
                if m.dim() > 1:
                    nn.init.xavier_uniform_(m)
            for m in self.shared_conv_img.parameters():
                if m.dim() > 1:
                    nn.init.xavier_uniform_(m)
        if self.with_pts and self.with_img:
            for m in self.fusion_transformer.parameters():
                if m.dim() > 1:
                    nn.init.xavier_uniform_(m)
            # for m in self.pts_projection.parameters():
            #     if m.dim() > 1:
            #         nn.init.xavier_uniform_(m)
            # for m in self.img_projection.parameters():
            #     if m.dim() > 1:
            #         nn.init.xavier_uniform_(m)
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
                    img_feat_cross = img_feat
                else:
                    img_feat_cross = None
                heatmap, dense_heatmap, pts_top_proposals_class, pts_top_proposals_index = self.generate_heatmap(lidar_feat, batch_size, img_metas, img_feat_cross)

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
                img_heatmap, img_dense_heatmap, img_top_proposals_class, img_top_proposals_index, img_top_proposals_view_idx, img_top_proposals_pos_id = \
                    self.generate_heatmap_img(img_feat, batch_size)
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
                img_query_pos_view = torch.arange(self.num_proposals).reshape(1, -1).repeat(batch_size, 1).to(img_feat.device)
                img_query_view = img_query_pos_view % self.num_views

        #################################
        # transformer decoder layer (LiDAR feature as K,V)
        #################################
        ret_dicts = []
        if self.with_pts:
            pts_query_feat, pts_query_pos, pts_ret_dicts = self.point_transformer(pts_query_feat, pts_query_pos, lidar_feat_flatten, bev_pos)
            ret_dicts.extend(pts_ret_dicts)

        #################################
        # transformer decoder layer (img feature as K,V)
        #################################
        if self.with_img:
            # positional encoding for image fusion
            img_feat = raw_img_feat.permute(0, 3, 1, 2, 4)  # [BS, n_views, C, H, W]
            img_feat_flatten = img_feat.view(batch_size, self.num_views, num_channel, -1)  # [BS, n_views, C, H*W]

            # img_query_feat, normal_img_query_pos, normal_img_query_pos_3d, img_ret_dicts = self.img_transformer(img_query_feat, normal_img_query_pos, img_query_view, img_feat_flatten, normal_img_feat_pos, img_metas)
            img_query_feat, normal_img_query_pos, img_query_pos_3d, img_ret_dicts = self.img_transformer(img_query_feat, normal_img_query_pos, img_query_view, img_feat_flatten, normal_img_feat_pos, img_metas)

            # torch.save(img_query_pos_3d, "vis/img_query_pos_3d.pt")
            # torch.save(img_query_view, "vis/img_query_view_3d.pt")

        if self.with_img and self.with_pts:
            if self.normal_pos:
                pts_query_pos[..., 0] = pts_query_pos[..., 0] / (self.test_cfg['grid_size'][0] // self.test_cfg['out_size_factor'])
                pts_query_pos[..., 1] = pts_query_pos[..., 1] / (self.test_cfg['grid_size'][1] // self.test_cfg['out_size_factor'])
                pts_query_pos = pts_query_pos * 2 - 1

                img_query_pos_3d = img_query_pos_3d.reshape(batch_size, self.num_proposals, -1, 3)
                img_query_pos_3d[..., 0] = img_query_pos_3d[..., 0] / (self.test_cfg['grid_size'][0] // self.test_cfg['out_size_factor'])
                img_query_pos_3d[..., 1] = img_query_pos_3d[..., 1] / (self.test_cfg['grid_size'][1] // self.test_cfg['out_size_factor'])
                img_query_pos_3d[..., 2] = (img_query_pos_3d[..., 2] - self.test_cfg['pc_range'][2]) / (self.test_cfg['pc_range'][5] - self.test_cfg['pc_range'][2])
                img_query_pos_3d = img_query_pos_3d * 2 - 1
                img_query_pos_3d = img_query_pos_3d.reshape(batch_size, self.num_proposals, -1)

            if self.stop_grad:
                pts_query_feat = pts_query_feat.detach()
                img_query_feat = img_query_feat.detach()
            if self.pos_embed == 'bev':
                all_query_feat, all_query_pos, fusion_ret_dicts = self.fusion_transformer(pts_query_feat, pts_query_pos, img_query_feat, img_query_pos_3d)
            elif self.pos_embed == '3d':
                height = pts_ret_dicts[-1]['height'].detach().clone().transpose(2, 1)  # [BS, num_proposals, 1]
                if self.normal_pos:
                    height = (height - self.test_cfg['pc_range'][2]) / (self.test_cfg['pc_range'][5] - self.test_cfg['pc_range'][2])
                    height = height * 2 - 1
                pts_query_pos_3d = torch.cat([pts_query_pos, height], dim=2)  # [BS, num_proposals, 3]
                all_query_feat, all_query_pos, fusion_ret_dicts = self.fusion_transformer(pts_query_feat, pts_query_pos_3d, img_query_feat, img_query_pos_3d)
            elif self.pos_embed == '2d':
                img_query_pos_3d = img_query_pos_3d.reshape(batch_size, self.num_proposals, -1, 3)
                img_query_pos_3d = img_query_pos_3d[..., :2]
                img_query_pos_3d = img_query_pos_3d.reshape(batch_size, self.num_proposals, -1)

                all_query_feat, all_query_pos, fusion_ret_dicts = self.fusion_transformer(pts_query_feat, pts_query_pos, img_query_feat, img_query_pos_3d)

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

    def build_heatmap(self, hidden_channel, bias, num_classes, layer_num=2):
        layers = []
        for i in range(layer_num-1):
            layers.append(ConvModule(
                hidden_channel,
                hidden_channel,
                kernel_size=3,
                padding=1,
                bias=bias,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
            ))
        layers.append(build_conv_layer(
            dict(type='Conv2d'),
            hidden_channel,
            num_classes,
            kernel_size=3,
            padding=1,
            bias=bias,
        ))
        return nn.Sequential(*layers)

    def generate_heatmap_cross(self, lidar_feat, img_feat, img_metas):
        # img_feat [bs*num_view, C, img_h, img_w]
        # lidar_feat [BS, C, H, W]
        bs = lidar_feat.shape[0]
        H, W = lidar_feat.shape[2], lidar_feat.shape[3]
        lidar_feat = lidar_feat.reshape(bs, self.hidden_channel, H*W)
        img_h = img_feat.shape[2]
        img_w = img_feat.shape[3]
        img_feat = self.fc(img_feat)
        img_feat = torch.max(img_feat, dim=2)[0]  # (bs*num_view, C, img_w/2)
        img_feat = img_feat.reshape(bs, self.num_views, self.hidden_channel, img_w//2)  # (bs, num_view, C, img_w/2)
        img_feat = img_feat.permute(0, 2, 1, 3).reshape(bs, self.hidden_channel, self.num_views*img_w//2)
        img_pos = self.create_2D_grid(1, img_w//2).to(lidar_feat.device)  # (1, img_w/2, 1, 2)
        img_pos = img_pos.squeeze(2)[0]  # (img_w/2, 2)
        img_pos[..., 1] = img_pos[..., 1] * img_h / 2  # (img_w/2, 2)

        batch_img_pos = []
        for sample_idx in range(bs):
            sample_img_pos = []
            img_scale_factor = (
                img_pos.new_tensor(img_metas[sample_idx]['scale_factor'][:2]
                                                if 'scale_factor' in img_metas[sample_idx].keys() else [1.0, 1.0])
            ) / 2
            for view_idx in range(self.num_views):
                lidar2img_rt = img_pos.new_tensor(img_metas[sample_idx]['lidar2img'])[view_idx]
                img_pos_ray = self.img_pos_3d(img_pos, lidar2img_rt, img_scale_factor)  # (img_w/2, 64)
                sample_img_pos.append(img_pos_ray)
            sample_img_pos = torch.stack(sample_img_pos, dim=0)  # (num_view, img_w/2, 64)
            batch_img_pos.append(sample_img_pos)
        batch_img_pos = torch.stack(batch_img_pos, dim=0)  # (bs, num_view, img_w/2, 64)
        batch_img_pos = batch_img_pos.reshape(bs, self.num_views*img_w//2, 64)

        bev_pos = self.bev_pos.repeat(bs, 1, 1).to(lidar_feat.device)
        if self.normal_pos:
            bev_pos[..., 0] = bev_pos[..., 0] / W * 2 - 1
            bev_pos[..., 1] = bev_pos[..., 1] / H * 2 - 1

        lidar_feat_cross = self.cross_heatmap_decoder(lidar_feat, img_feat, bev_pos, batch_img_pos)  # [BS, C, H*W]
        lidar_feat_cross = lidar_feat_cross.reshape(bs, self.hidden_channel, H, W)
        return self.cross_heatmap_head(lidar_feat_cross)

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
        coords3d = torch.matmul(img2lidars, coords.unsqueeze(-1)).squeeze(-1)[..., :2]  # [W*H, D, 3]

        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.test_cfg['pc_range'][0]) / (self.test_cfg['pc_range'][3] - self.test_cfg['pc_range'][0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.test_cfg['pc_range'][1]) / (self.test_cfg['pc_range'][4] - self.test_cfg['pc_range'][1])
        if self.normal_pos:
            coords3d[..., :2] = coords3d[..., :2] * 2 - 1
        else:
            coords3d[..., 0:1] = coords3d[..., 0:1] * (self.test_cfg['grid_size'][0] // self.test_cfg['out_size_factor'])
            coords3d[..., 1:2] = coords3d[..., 1:2] * (self.test_cfg['grid_size'][1] // self.test_cfg['out_size_factor'])

        coords3d = coords3d.contiguous().view(coords3d.size(0), D*2)
        return coords3d

    def generate_heatmap(self, lidar_feat, batch_size, img_metas, img_feat=None):
        dense_heatmap = self.heatmap_head(lidar_feat)  # [BS, num_class, H, W]
        if img_feat is None:
            heatmap = dense_heatmap.detach().sigmoid()  # [BS, num_class, H, W]
        else:
            dense_heatmap_cross = self.generate_heatmap_cross(lidar_feat, img_feat, img_metas)
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
        top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[..., :self.num_proposals]  # [BS, num_proposals]

        top_proposals_class = top_proposals // heatmap.shape[-1]  # [BS, num_proposals]
        top_proposals_index = top_proposals % heatmap.shape[-1]  # [BS, num_proposals]
        return heatmap, dense_heatmap, top_proposals_class, top_proposals_index

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
        top_proposals = img_heatmap.contiguous().view(batch_size, -1).argsort(dim=-1, descending=True)[..., :self.num_proposals]  # [BS, num_proposals]
        top_proposals_class = top_proposals // (img_heatmap.shape[-1]*img_heatmap.shape[-2])  # [BS, num_proposals]

        top_proposals_view_index = top_proposals % (img_heatmap.shape[-1]*img_heatmap.shape[-2]) // img_heatmap.shape[-1]  # [BS, num_proposals]
        top_proposals_pos_index = top_proposals % img_heatmap.shape[-1]  # [BS, num_proposals]
        top_proposals_index = top_proposals % (img_heatmap.shape[-1]*img_heatmap.shape[-2])  # [BS, num_proposals]

        img_dense_heatmap = img_dense_heatmap.view(batch_size, self.num_views, img_dense_heatmap.shape[1], img_dense_heatmap.shape[2], img_dense_heatmap.shape[3]) # [BS, num_views, num_class, H, W]
        img_dense_heatmap = img_dense_heatmap.permute(0, 2, 1, 3, 4) # [BS, num_class, num_views, H*W]
        img_heatmap = img_heatmap.contiguous().view(batch_size, img_heatmap.shape[1], -1)

        return img_heatmap, img_dense_heatmap, top_proposals_class, top_proposals_index, top_proposals_view_index, top_proposals_pos_index

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels, preds_dict):
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
            res_tuple_2d = multi_apply(self.get_targets_single_2d, gt_bboxes, gt_labels, list_of_pred_dict, np.arange(len(gt_bboxes)))
            labels_2d = torch.cat(res_tuple_2d[0], dim=0)
            label_weights_2d = torch.cat(res_tuple_2d[1], dim=0)
            bbox_targets_2d = torch.cat(res_tuple_2d[2], dim=0)
            bbox_weights_2d = torch.cat(res_tuple_2d[3], dim=0)
            ious_2d = torch.cat(res_tuple_2d[4], dim=0)
            # num_pos_2d = np.sum(res_tuple_2d[5])
            num_pos_layer_2d = np.concatenate(res_tuple_2d[5], axis=0)  # [BS, num_layer]
            matched_ious_2d = torch.cat(res_tuple_2d[6], dim=0)

        if self.initialize_by_heatmap:
            heatmap = torch.cat(res_tuple[7], dim=0)
            if self.with_img:
                heatmap_2d = torch.cat(res_tuple_2d[7], dim=0)
                return labels, label_weights, bbox_targets, bbox_weights, ious, num_pos_layer, matched_ious, heatmap, \
                       labels_2d, label_weights_2d, bbox_targets_2d, bbox_weights_2d, ious_2d, num_pos_layer_2d, matched_ious_2d, heatmap_2d
            else:
                return labels, label_weights, bbox_targets, bbox_weights, ious, num_pos_layer_2d, matched_ious, heatmap
        else:
            if self.with_img:
                return labels, label_weights, bbox_targets, bbox_weights, ious, num_pos_layer, matched_ious, \
                labels_2d, label_weights_2d, bbox_targets_2d, bbox_weights_2d, ious_2d, num_pos_layer_2d, matched_ious_2d,
            else:
                return labels, label_weights, bbox_targets, bbox_weights, ious, num_pos_layer_2d, matched_ious

    def get_targets_single_2d(self, gt_bboxes, gt_labels, preds_dict, batch_idx):
        gt_bboxes = gt_bboxes.float()
        num_proposals = preds_dict['cls'].shape[-1]
        bbox = copy.deepcopy(preds_dict['bbox'].detach()).transpose(2, 1)[0]  # [num_proposals, 4]
        cls = copy.deepcopy(preds_dict['cls'].detach()).transpose(2, 1)[0]  # [num_proposals, num_classes]
        view = copy.deepcopy(preds_dict['view'].detach())[0] # [num_proposals, ]
        img_scale = [self.test_cfg['img_scale'][0], self.test_cfg['img_scale'][1], self.test_cfg['img_scale'][0], self.test_cfg['img_scale'][1]]
        img_scale = torch.Tensor(img_scale).to(bbox.device).unsqueeze(0)
        normal_gt_bbox = gt_bboxes / img_scale  # [num_gt, 4]

        assign_result_list = []
        for idx_layer in range(self.num_img_decoder_layers):
            bboxes_tensor_layer = bbox[idx_layer*self.num_proposals:(idx_layer+1)*self.num_proposals, :]  # [num_proposals, 4]
            score_layer = cls[idx_layer*self.num_proposals:(idx_layer+1)*self.num_proposals, :]  # [num_proposals, num_class]
            view_layer = view[idx_layer*self.num_proposals:(idx_layer+1)*self.num_proposals]  # [num_proposals]

            assign_result = self.bbox_assigner_2d.assign(bboxes_tensor_layer, score_layer, view_layer, gt_bboxes, gt_labels, img_w=self.test_cfg['img_scale'][0], img_h=self.test_cfg['img_scale'][1])
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

        start = 0
        pos_num_layers = []
        for idx_layer in range(self.num_img_decoder_layers):
            layer_num_proposal = self.num_proposals
            count = pos_inds[torch.logical_and(pos_inds>=start, pos_inds<start+layer_num_proposal)].shape[0]
            pos_num_layers.append(count)
            start += layer_num_proposal
        pos_num_layers = np.array(pos_num_layers)
        assert np.sum(pos_num_layers) == pos_inds.shape[0]

        # create target for loss computation
        bbox_targets = torch.zeros([num_proposals, 4]).to(bbox.device)
        bbox_weights = torch.zeros([num_proposals, 4]).to(bbox.device)
        view_targets = bbox.new_zeros(num_proposals, dtype=torch.long)
        ious = assign_result_ensemble.max_overlaps
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bbox.new_zeros(num_proposals, dtype=torch.long)
        label_weights = bbox.new_zeros(num_proposals, dtype=torch.long)

        if gt_labels is not None:  # default label is -1
            labels += self.num_classes

        # both pos and neg have classification loss, only pos has regression and iou loss
        if len(pos_inds) > 0:
            bbox_targets[pos_inds, :] = sampling_result.pos_gt_bboxes
            bbox_weights[pos_inds, :] = 1.0

            view_targets[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds, 1]

            if gt_labels is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds, 0]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

            view_mask_ignore = view_targets != view
            label_weights[view_mask_ignore] = 0
            bbox_weights[view_mask_ignore, :] = 0

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # # compute dense heatmap targets
        if self.initialize_by_heatmap:
            device = labels.device
            img_size = self.test_cfg['img_scale']
            feature_map_size = (img_size[0] // self.out_size_factor_img, img_size[1] // self.out_size_factor_img)
            heatmap = gt_bboxes.new_zeros(self.num_classes, self.num_views, feature_map_size[1], feature_map_size[0])

            for idx in range(len(gt_bboxes)):
                width = gt_bboxes[idx][2]
                length = gt_bboxes[idx][3]
                width = width / self.out_size_factor_img
                length = length / self.out_size_factor_img
                view_id = gt_labels[idx][1]
                if width > 0 and length > 0:
                    radius = gaussian_radius((length, width), min_overlap=self.train_cfg['gaussian_overlap'])
                    radius = max(self.train_cfg['min_radius'], int(radius))
                    radius = min(self.train_cfg['max_radius'], int(radius))

                    x, y = gt_bboxes[idx][0], gt_bboxes[idx][1]

                    coor_x = x / self.out_size_factor_img
                    coor_y = y / self.out_size_factor_img

                    center = torch.tensor([coor_x, coor_y], dtype=torch.float32, device=device)
                    center_int = center.to(torch.int32)
                    draw_heatmap_gaussian(heatmap[gt_labels[idx][0], view_id], center_int, radius)

            matched_ious = torch.ones_like(ious) * -1
            matched_ious[pos_inds] = ious[pos_inds]

            return labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], ious[None], pos_num_layers[None], matched_ious[None], heatmap[None]
        else:
            matched_ious = torch.ones_like(ious) * -1
            matched_ious[pos_inds] = ious[pos_inds]
            return labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], ious[None], pos_num_layers[None], matched_ious[None]

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
    def loss(self, gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels, preds_dicts, **kwargs):
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
                labels_2d, label_weights_2d, bbox_targets_2d, bbox_weights_2d, ious_2d, num_pos_layer_2d, matched_ious_2d, heatmap_2d = self.get_targets(gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels, preds_dicts[0])
            else:
                labels, label_weights, bbox_targets, bbox_weights, ious, num_pos_layer, matched_ious, \
                labels_2d, label_weights_2d, bbox_targets_2d, bbox_weights_2d, ious_2d, num_pos_layer_2d, matched_ious_2d = self.get_targets(gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels, preds_dicts[0])        # if hasattr(self, 'on_the_image_mask'):
        else:
            if self.initialize_by_heatmap:
                labels, label_weights, bbox_targets, bbox_weights, ious, num_pos_layer, matched_ious, heatmap = self.get_targets(gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels, preds_dicts[0])
            else:
                labels, label_weights, bbox_targets, bbox_weights, ious, num_pos_layer, matched_ious = self.get_targets(gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels, preds_dicts[0])        # if hasattr(self, 'on_the_image_mask'):

        preds_dict = preds_dicts[0][0]
        loss_dict = dict()

        if self.initialize_by_heatmap:
            # compute heatmap loss
            loss_heatmap = self.loss_heatmap(clip_sigmoid(preds_dict['dense_heatmap']), heatmap, avg_factor=max(heatmap.eq(1).float().sum().item(), 1))
            loss_heatmap_2d = self.loss_heatmap(clip_sigmoid(preds_dict['img_dense_heatmap']), heatmap_2d, avg_factor=max(heatmap_2d.eq(1).float().sum().item(), 1))

            loss_dict['loss_heatmap'] = loss_heatmap
            loss_dict['loss_heatmap_2d'] = loss_heatmap_2d

        # compute loss for each layer
        start = 0
        num_pos_layer = np.sum(num_pos_layer, axis=0)
        num_pos_layer_2d = np.sum(num_pos_layer_2d, axis=0)

        for idx_layer in range(self.num_pts_decoder_layers+self.num_fusion_decoder_layers):
            if idx_layer < self.num_pts_decoder_layers:
                prefix = f'layer_pts_{idx_layer}'
            else:
                prefix = f'layer_fusion_{idx_layer-self.num_pts_decoder_layers}'
            layer_num_proposals = self.get_layer_num_proposal(idx_layer)

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
            loss_dict[f'{prefix}_loss_matched_ious'] = layer_match_ious

        if self.with_img and self.supervision2d:
            start = 0
            for idx_layer in range(self.num_img_decoder_layers):
                prefix = f'layer_img_{idx_layer}'
                layer_num_proposals = self.num_proposals
                layer_labels_2d = labels_2d[..., start:start + layer_num_proposals].reshape(-1)
                layer_label_weights_2d = label_weights_2d[..., start:start + layer_num_proposals].reshape(-1)
                layer_score_2d = preds_dict['cls'][..., start:start + layer_num_proposals]
                layer_cls_score_2d = layer_score_2d.permute(0, 2, 1).reshape(-1, self.num_classes)
                layer_loss_cls_2d = self.loss_cls_2d(layer_cls_score_2d, layer_labels_2d, layer_label_weights_2d, avg_factor=max(num_pos_layer_2d[idx_layer], 1))

                preds_2d = preds_dict['bbox'][..., start:start + layer_num_proposals].permute(0, 2, 1)  # [bs, num_proposal, 4]
                layer_bbox_targets_2d = bbox_targets_2d[:, start:start + layer_num_proposals, :]
                layer_reg_weights_2d = bbox_weights_2d[:, start:start + layer_num_proposals, :]
                # code_weights = self.train_cfg.get('code_weights', None)
                # layer_reg_weights_2d = layer_bbox_weights_2d * layer_bbox_weights_2d.new_tensor(code_weights)
                layer_loss_bbox_2d = self.loss_bbox_2d(preds_2d, layer_bbox_targets_2d, layer_reg_weights_2d, avg_factor=max(num_pos_layer_2d[idx_layer], 1))

                layer_match_ious_2d = matched_ious_2d[..., start:start + layer_num_proposals]
                layer_match_ious_2d = torch.sum(layer_match_ious_2d*(layer_match_ious_2d>=0), dim=-1) / torch.sum(layer_match_ious_2d>=0, dim=-1)
                layer_match_ious_2d = torch.mean(layer_match_ious_2d)
                start += layer_num_proposals

                loss_dict[f'{prefix}_loss_cls_2d'] = layer_loss_cls_2d
                loss_dict[f'{prefix}_loss_bbox_2d'] = layer_loss_bbox_2d
                loss_dict[f'{prefix}_loss_matched_ious_2d'] = layer_match_ious_2d

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

            # one_hot = F.one_hot(self.query_labels, num_classes=self.num_classes).permute(0, 2, 1)
            # if self.with_img and self.with_pts:
            #     query_heatmap_score = preds_dict[0]['query_heatmap_score'] * one_hot
            #     query_heatmap_score = torch.cat([query_heatmap_score, query_heatmap_score], dim=2)
            # else:
            #     query_heatmap_score = preds_dict[0]['query_heatmap_score'] * one_hot
            one_hot = F.one_hot(self.query_labels, num_classes=self.num_classes).permute(0, 2, 1)
            query_heatmap_score = preds_dict[0]['query_heatmap_score'] * one_hot
            batch_score = batch_score * query_heatmap_score

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
        if self.with_img and self.with_pts and idx_layer >= self.num_pts_decoder_layers:
            layer_num_proposal = self.num_proposals * 2
        else:
            layer_num_proposal = self.num_proposals
        layer_num_proposal = self.num_proposals
        return layer_num_proposal