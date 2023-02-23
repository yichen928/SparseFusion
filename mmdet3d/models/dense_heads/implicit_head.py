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
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu
from mmdet.core import build_bbox_coder, multi_apply, build_assigner, build_sampler, AssignResult

from mmdet3d.models.utils import PositionEmbeddingLearned, TransformerDecoderLayer, MultiheadAttention, FFN, inverse_sigmoid

@HEADS.register_module()
class ImplictFusionHead(nn.Module):
    def __init__(self,
                 with_pts=False,
                 with_img=False,
                 share_prediction_heads=True,
                 pts_prediction=False,
                 img_prediction=False,
                 pts_smca=True,
                 img_smca=True,
                 num_views=0,
                 in_channels_img=64,
                 out_size_factor_img=4,
                 num_proposals=128,
                 auxiliary=True,
                 in_channels=128 * 3,
                 hidden_channel=128,
                 num_classes=4,
                 # config for Transformer
                 num_pts_decoder_layers=3,
                 num_img_decoder_layers=3,
                 num_fusion_decoder_layers=3,
                 num_heads=8,
                 learnable_query_pos=True,
                 nms_kernel_size=1,
                 ffn_channel=256,
                 dropout=0.1,
                 bn_momentum=0.1,
                 activation='relu',
                 # config for FFN
                 common_heads=dict(),
                 num_heatmap_convs=2,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 bias='auto',
                 # loss
                 loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_iou=dict(type='VarifocalLoss', use_sigmoid=True, iou_weighted=True, reduction='mean'),
                 loss_bbox=dict(type='L1Loss', reduction='mean'),
                 # others
                 train_cfg=None,
                 test_cfg=None,
                 bbox_coder=None,
                 ):
        super(ImplictFusionHead, self).__init__()

        self.with_img = with_img
        self.with_pts = with_pts
        self.share_prediction_heads = share_prediction_heads
        self.pts_prediction = pts_prediction or (not with_img) and with_pts
        self.img_prediction = img_prediction or (not with_pts) and with_img
        self.pts_smca = pts_smca
        self.img_smca = img_smca
        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.auxiliary = auxiliary
        self.in_channels = in_channels
        self.num_heads = num_heads
        if not self.with_pts:
            num_pts_decoder_layers = 0
        if not self.with_img:
            num_img_decoder_layers = 0
        if not (self.with_img and self.with_pts):
            num_fusion_decoder_layers = 0
        self.num_pts_decoder_layers = num_pts_decoder_layers
        self.num_img_decoder_layers = num_img_decoder_layers
        self.num_fusion_decoder_layers = num_fusion_decoder_layers

        self.bn_momentum = bn_momentum
        self.learnable_query_pos = learnable_query_pos
        self.nms_kernel_size = nms_kernel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.sampling = False

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
        self.decoder = nn.ModuleList()
        self.prediction_heads = nn.ModuleList()

        pts_prediction_heads = None
        if self.with_pts:
            for i in range(self.num_pts_decoder_layers):
                self.decoder.append(
                    TransformerDecoderLayer(
                        hidden_channel, num_heads, ffn_channel, dropout, activation,
                        self_posembed=PositionEmbeddingLearned(2, hidden_channel),
                        cross_posembed=PositionEmbeddingLearned(2, hidden_channel),
                    )
                )

            if self.pts_prediction and not self.share_prediction_heads:
                # Prediction Head
                heads = copy.deepcopy(common_heads)
                heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs)))
                pts_prediction_heads = FFN(hidden_channel, heads, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias)

        img_prediction_heads = None
        if self.with_img:
            self.num_views = num_views
            self.out_size_factor_img = out_size_factor_img
            self.shared_conv_img = build_conv_layer(
                dict(type='Conv2d'),
                in_channels_img,  # channel of img feature map
                hidden_channel,
                kernel_size=3,
                padding=1,
                bias=bias,
            )
            for i in range(self.num_img_decoder_layers):
                self.decoder.append(
                    TransformerDecoderLayer(
                        hidden_channel, num_heads, ffn_channel, dropout, activation,
                        self_posembed=PositionEmbeddingLearned(2, hidden_channel),
                        cross_posembed=PositionEmbeddingLearned(2, hidden_channel),
                    )
                )

            if self.img_prediction and not self.share_prediction_heads:
                # Prediction Head
                heads = copy.deepcopy(common_heads)
                heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs)))
                img_prediction_heads = FFN(hidden_channel, heads, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias)

        fusion_prediction_heads = None
        if self.with_img and self.with_pts:
            for i in range(self.num_fusion_decoder_layers):
                position_embed = PositionEmbeddingLearned(2, hidden_channel)
                self.decoder.append(
                    TransformerDecoderLayer(
                        hidden_channel, num_heads, ffn_channel, dropout, activation,
                        self_posembed=position_embed,
                        cross_posembed=position_embed,
                        cross_only=True
                    )
                )
            if not self.share_prediction_heads:
                heads = copy.deepcopy(common_heads)
                heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs)))
                fusion_prediction_heads = FFN(hidden_channel, heads, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias)

        if not self.share_prediction_heads:
            self.prediction_heads = nn.ModuleList([pts_prediction_heads, img_prediction_heads, fusion_prediction_heads])
        else:
            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs)))
            prediction_heads = FFN(hidden_channel, heads, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias)
            self.prediction_heads = nn.ModuleList([prediction_heads, prediction_heads, prediction_heads])

        self.num_decoder_layers_with_pred = self.pts_prediction * self.num_pts_decoder_layers + self.img_prediction * self.num_img_decoder_layers + self.with_img * self.with_pts * self.num_fusion_decoder_layers
        assert len(self.decoder) == self.num_pts_decoder_layers + self.num_img_decoder_layers + self.num_fusion_decoder_layers
        self.init_weights()
        self._init_assigner_sampler()

        # Position Embedding for Cross-Attention, which is re-used during training
        x_size = self.test_cfg['grid_size'][0] // self.test_cfg['out_size_factor']
        y_size = self.test_cfg['grid_size'][1] // self.test_cfg['out_size_factor']
        self.bev_pos = self.create_2D_grid(x_size, y_size)

        # query feature
        self.pts_query_feat = nn.Parameter(torch.randn(1, hidden_channel, self.num_proposals))
        self.pts_query_pos = nn.Parameter(torch.rand([1, self.num_proposals, 2])*torch.Tensor([x_size, y_size]).reshape(1, 1, 2), requires_grad=learnable_query_pos)

        self.img_query_feat = nn.Parameter(torch.randn(1, hidden_channel, self.num_proposals))
        self.img_query_pos = nn.Parameter(torch.rand([1, self.num_proposals, 2])*torch.Tensor([x_size, y_size]).reshape(1, 1, 2), requires_grad=learnable_query_pos)

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
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        # initialize transformer
        for m in self.prediction_heads.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        # already initialized when queries are created
        # if hasattr(self, 'query'):
        #     nn.init.xavier_normal_(self.query)
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

        if self.with_pts:
            lidar_feat = self.shared_conv(inputs)

            #################################
            # image to BEV
            #################################
            lidar_feat_flatten = lidar_feat.view(batch_size, lidar_feat.shape[1], -1)  # [BS, C, H*W]
            bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)  # [BS, H*W, 2]

            pts_query_feat = self.pts_query_feat.repeat(batch_size, 1, 1)  # [BS, C, num_proposals]
            pts_query_pos = self.pts_query_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)  # [BS, num_proposals, 2]

        if self.with_img:
            img_feat = self.shared_conv_img(img_inputs)  # [BS * n_views, C, H, W]

            img_h, img_w, num_channel = img_inputs.shape[-2], img_inputs.shape[-1], img_feat.shape[1]
            raw_img_feat = img_feat.view(batch_size, self.num_views, num_channel, img_h, img_w).permute(0, 2, 3, 1, 4) # [BS, C, H, n_views, W]

            img_query_feat = self.img_query_feat.repeat(batch_size, 1, 1)  # [BS, C, num_proposals]
            img_query_pos = self.img_query_pos.repeat(batch_size, 1, 1).to(img_feat.device)  # [BS, num_proposals, 2]

        #################################
        # transformer decoder layer (LiDAR feature as K,V)
        #################################
        ret_dicts = []
        if self.with_pts:
            res_layer = self.prediction_heads[0](pts_query_feat)
            res_layer['center'] = res_layer['center'] + pts_query_pos.permute(0, 2, 1)  # [BS, 2, num_proposals]
            pts_query_pos = res_layer['center'].permute(0, 2, 1)  # [BS, num_proposals, 2]
            for i in range(self.num_pts_decoder_layers):
                # Transformer Decoder Layer
                # :param query: B C Pq    :param query_pos: B Pq 3/6
                if self.pts_smca:
                    centers = res_layer['center'].detach().clone().permute(0, 2, 1)  # [BS, num_proposals, 2]
                    dims = res_layer['dim'].detach().clone().permute(0, 2, 1)
                    corners = dims[..., :2].exp() / self.test_cfg['out_size_factor'] / self.test_cfg['voxel_size'][0]  # [BS, num_proposals, 2]
                    radius = torch.ceil(corners.norm(dim=-1, p=2) / 2).int() # [BS, num_proposals]  # radius of the minimum circumscribed circle of the wireframe
                    sigma = (radius * 2 + 1) / 6.0  # [BS, num_proposals]
                    bev_pos = self.bev_pos.to(centers.device)  # [1, H*W, 2]
                    distance = (centers[:, :, None, :] - (bev_pos - 0.5)).norm(dim=-1) ** 2  # [BS, num_proposals, H*W]
                    gaussian_mask = (-distance / (2 * sigma[:, :, None] ** 2)).exp()  # [BS, num_proposals, H*W]
                    gaussian_mask[gaussian_mask < torch.finfo(torch.float32).eps] = 0
                    attn_mask = gaussian_mask

                    pts_query_feat = self.decoder[i](pts_query_feat, lidar_feat_flatten, pts_query_pos, bev_pos, attn_mask=attn_mask.log())
                else:
                    pts_query_feat = self.decoder[i](pts_query_feat, lidar_feat_flatten, pts_query_pos, bev_pos)

                # Prediction
                res_layer = self.prediction_heads[0](pts_query_feat)
                res_layer['center'] = res_layer['center'] + pts_query_pos.permute(0, 2, 1)
                if self.pts_prediction:
                    ret_dicts.append(res_layer)
                # for next level positional embedding
                pts_query_pos = res_layer['center'].detach().clone().permute(0, 2, 1)

        #################################
        # transformer decoder layer (img feature as K,V)
        #################################
        if self.with_img:
            # positional encoding for image fusion
            img_feat = raw_img_feat.permute(0, 3, 1, 2, 4) # [BS, n_views, C, H, W]
            img_feat_flatten = img_feat.view(batch_size, self.num_views, num_channel, -1)  # [BS, n_views, C, H*W]
            if self.img_feat_pos is None:
                (h, w) = img_inputs.shape[-2], img_inputs.shape[-1]
                img_feat_pos = self.img_feat_pos = self.create_2D_grid(h, w).to(img_feat_flatten.device)  # (1, h*w, 2)
            else:
                img_feat_pos = self.img_feat_pos

            res_layer = self.prediction_heads[1](img_query_feat)
            res_layer['center'] = res_layer['center'] + img_query_pos.permute(0, 2, 1)
            img_query_pos = res_layer['center'].permute(0, 2, 1)  # [BS, num_proposals, 2]
            for i in range(self.num_img_decoder_layers):
                img_prev_query_feat = img_query_feat.detach().clone()  # [BS, C, num_proposals]
                img_query_feat = torch.zeros_like(img_query_feat)  # create new container for img query feature
                img_query_pos_realmetric = img_query_pos.permute(0, 2, 1) * self.test_cfg['out_size_factor'] * \
                                       self.test_cfg['voxel_size'][0] + self.test_cfg['pc_range'][0]  # [BS, 2, num_proposals]
                img_query_pos_3d = torch.cat([img_query_pos_realmetric, res_layer['height']], dim=1).detach().clone()  # [BS, 3, num_proposals]
                if 'vel' in res_layer:
                    vel = copy.deepcopy(res_layer['vel'].detach())
                else:
                    vel = None
                pred_boxes = self.bbox_coder.decode(
                    copy.deepcopy(res_layer['heatmap'].detach()),
                    copy.deepcopy(res_layer['rot'].detach()),
                    copy.deepcopy(res_layer['dim'].detach()),
                    copy.deepcopy(res_layer['center'].detach()),
                    copy.deepcopy(res_layer['height'].detach()),
                    vel,
                )

                on_the_image_mask = torch.ones([batch_size, self.num_proposals]).to(img_query_pos_3d.device) * -1

                for sample_idx in range(batch_size):
                    lidar2img_rt = img_query_pos_3d.new_tensor(img_metas[sample_idx]['lidar2img'])
                    img_scale_factor = (
                        img_query_pos_3d.new_tensor(img_metas[sample_idx]['scale_factor'][:2]
                                                if 'scale_factor' in img_metas[sample_idx].keys() else [1.0, 1.0])
                    )
                    img_flip = img_metas[sample_idx]['flip'] if 'flip' in img_metas[sample_idx].keys() else False
                    img_crop_offset = (
                        img_query_pos_3d.new_tensor(img_metas[sample_idx]['img_crop_offset'])
                        if 'img_crop_offset' in img_metas[sample_idx].keys() else 0)
                    img_shape = img_metas[sample_idx]['img_shape'][:2]
                    img_pad_shape = img_metas[sample_idx]['input_shape'][:2]
                    boxes = LiDARInstance3DBoxes(pred_boxes[sample_idx]['bboxes'][:, :7], box_dim=7)
                    img_query_pos_3d_with_corners = torch.cat([img_query_pos_3d[sample_idx], boxes.corners.permute(2, 0, 1).view(3, -1)],dim=-1)  # [3, num_proposals] + [3, num_proposals*8]
                    # transform point clouds back to original coordinate system by reverting the data augmentation
                    if batch_size == 1:  # skip during inference to save time
                        points = img_query_pos_3d_with_corners.T
                    else:
                        points = apply_3d_transformation(img_query_pos_3d_with_corners.T, 'LIDAR', img_metas[sample_idx], reverse=True).detach()
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

                        coor_x, coor_corner_x = coor_x[0:self.num_proposals, :], coor_x[self.num_proposals:, :]
                        coor_y, coor_corner_y = coor_y[0:self.num_proposals, :], coor_y[self.num_proposals:, :]
                        coor_corner_x = coor_corner_x.reshape(self.num_proposals, 8, 1)
                        coor_corner_y = coor_corner_y.reshape(self.num_proposals, 8, 1)
                        coor_corner_xy = torch.cat([coor_corner_x, coor_corner_y], dim=-1)  # [num_proposals, 8, 2]

                        h, w = img_pad_shape
                        on_the_image = (coor_x > 0) * (coor_x < w) * (coor_y > 0) * (coor_y < h)
                        on_the_image = on_the_image.squeeze() # [num_proposals, ]
                        # skip the following computation if no object query fall on current image
                        if on_the_image.sum() <= 1:
                            continue
                        on_the_image_mask[sample_idx, on_the_image] = view_idx

                        # add spatial constraint
                        center_ys = (coor_y[on_the_image] / self.out_size_factor_img)
                        center_xs = (coor_x[on_the_image] / self.out_size_factor_img)

                        img_query_feat_view = img_prev_query_feat[sample_idx, :, on_the_image]  # [C, N_image]
                        img_query_pos_view = torch.cat([center_xs, center_ys], dim=-1)   # [N_image, 2]

                        if self.img_smca:
                            centers = torch.cat([center_xs, center_ys], dim=-1).int()  # [num_proposals, 2] # center on the feature map
                            corners = (coor_corner_xy[on_the_image].max(1).values - coor_corner_xy[on_the_image].min(1).values) / self.out_size_factor_img
                            radius = torch.ceil(corners.norm(dim=-1, p=2) / 2).int()  # radius of the minimum circumscribed circle of the wireframe
                            sigma = (radius * 2 + 1) / 6.0
                            distance = (centers[:, None, :] - (img_feat_pos - 0.5)).norm(dim=-1) ** 2
                            gaussian_mask = (-distance / (2 * sigma[:, None] ** 2)).exp()
                            gaussian_mask[gaussian_mask < torch.finfo(torch.float32).eps] = 0
                            attn_mask = gaussian_mask
                            img_query_feat_view = self.decoder[i+self.num_pts_decoder_layers](img_query_feat_view[None], img_feat_flatten[sample_idx:sample_idx + 1, view_idx], img_query_pos_view[None], img_feat_pos, attn_mask=attn_mask.log())
                        else:
                            img_query_feat_view = self.decoder[i+self.num_pts_decoder_layers](img_query_feat_view[None], img_feat_flatten[sample_idx:sample_idx + 1, view_idx], img_query_pos_view[None], img_feat_pos)

                        img_query_feat[sample_idx, :, on_the_image] = img_query_feat_view.clone()

                self.on_the_image_mask = (on_the_image_mask != -1)
                prev_res_layer = res_layer
                res_layer = self.prediction_heads[1](img_query_feat)
                res_layer['center'] = res_layer['center'] + img_query_pos.permute(0, 2, 1)
                img_query_pos = res_layer['center'].detach().clone().permute(0, 2, 1)

                for key, value in res_layer.items():
                    pred_dim = value.shape[1]
                    res_layer[key][~self.on_the_image_mask.unsqueeze(1).repeat(1, pred_dim, 1)] = prev_res_layer[key][~self.on_the_image_mask.unsqueeze(1).repeat(1, pred_dim, 1)]
                if self.img_prediction:
                    ret_dicts.append(res_layer)

        if self.with_img and self.with_pts:
            all_query_feat = torch.cat([pts_query_feat, img_query_feat], dim=2)
            all_query_pos = torch.cat([pts_query_pos, img_query_pos], dim=1)

            for i in range(self.num_fusion_decoder_layers):
                # Transformer Decoder Layer
                # :param query: B C Pq    :param query_pos: B Pq 3/6
                all_query_feat = self.decoder[i+self.num_pts_decoder_layers+self.num_img_decoder_layers](all_query_feat, all_query_feat, all_query_pos, all_query_pos)

                # Prediction
                res_layer = self.prediction_heads[2](all_query_feat)
                res_layer['center'] = res_layer['center'] + all_query_pos.permute(0, 2, 1)
                ret_dicts.append(res_layer)
                # for next level positional embedding
                all_query_pos = res_layer['center'].detach().clone().permute(0, 2, 1)

        if self.auxiliary is False:
            # only return the results of last decoder layer
            return [ret_dicts[-1]]

        # return all the layer's results for auxiliary superivison
        new_res = {}
        for key in ret_dicts[0].keys():
            new_res[key] = torch.cat([ret_dict[key] for ret_dict in ret_dicts], dim=-1)

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

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, preds_dict):
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
        num_pos = np.sum(res_tuple[5])
        matched_ious = np.mean(res_tuple[6])

        return labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, preds_dict, batch_idx):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.
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
        if self.auxiliary:
            num_layer = self.num_decoder_layers_with_pred
        else:
            num_layer = 1

        assign_result_list = []
        start = 0
        for idx_layer in range(num_layer):
            if self.with_img and self.with_pts and ((not self.auxiliary) or idx_layer >= num_layer - self.num_fusion_decoder_layers):
                layer_num_proposal = self.num_proposals * 2
            else:
                layer_num_proposal = self.num_proposals

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
        mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
        return labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], ious[None], int(pos_inds.shape[0]), float(mean_iou)


    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """Loss function for CenterHead.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (list[list[dict]]): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """

        labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious = self.get_targets(gt_bboxes_3d, gt_labels_3d, preds_dicts[0])
        # if hasattr(self, 'on_the_image_mask'):
        #     label_weights = label_weights * self.on_the_image_mask
        #     bbox_weights = bbox_weights * self.on_the_image_mask[:, :, None]
        #     num_pos = bbox_weights.max(-1).values.sum()
        preds_dict = preds_dicts[0][0]
        loss_dict = dict()

        # compute loss for each layer
        start = 0
        for idx_layer in range(self.num_decoder_layers_with_pred if self.auxiliary else 1):
            if (not self.auxiliary) and idx_layer == 0:
                prefix = 'layer_-1'
                if self.with_pts and self.with_img:
                    layer_num_proposals = self.num_proposals * 2
                else:
                    layer_num_proposals = self.num_proposals
            elif idx_layer < self.num_pts_decoder_layers:
                layer_num_proposals = self.num_proposals
                prefix = f'layer_pts_{idx_layer}'
            elif idx_layer < self.num_pts_decoder_layers + self.num_img_decoder_layers:
                layer_num_proposals = self.num_proposals
                prefix = f'layer_img_{idx_layer-self.num_pts_decoder_layers}'
            else:
                layer_num_proposals = self.num_proposals * 2
                prefix = f'layer_fusion_{idx_layer-self.num_pts_decoder_layers-self.num_img_decoder_layers}'

            layer_labels = labels[..., start:start + layer_num_proposals].reshape(-1)
            layer_label_weights = label_weights[..., start:start + layer_num_proposals].reshape(-1)
            layer_score = preds_dict['heatmap'][..., start:start + layer_num_proposals]
            layer_cls_score = layer_score.permute(0, 2, 1).reshape(-1, self.num_classes)
            layer_loss_cls = self.loss_cls(layer_cls_score, layer_labels, layer_label_weights, avg_factor=max(num_pos, 1))

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
            layer_loss_bbox = self.loss_bbox(preds, layer_bbox_targets, layer_reg_weights, avg_factor=max(num_pos, 1))

            start += layer_num_proposals

            # layer_iou = preds_dict['iou'][..., idx_layer*self.num_proposals:(idx_layer+1)*self.num_proposals].squeeze(1)
            # layer_iou_target = ious[..., idx_layer*self.num_proposals:(idx_layer+1)*self.num_proposals]
            # layer_loss_iou = self.loss_iou(layer_iou, layer_iou_target, layer_bbox_weights.max(-1).values, avg_factor=max(num_pos, 1))

            loss_dict[f'{prefix}_loss_cls'] = layer_loss_cls
            loss_dict[f'{prefix}_loss_bbox'] = layer_loss_bbox
            # loss_dict[f'{prefix}_loss_iou'] = layer_loss_iou

        loss_dict[f'matched_ious'] = layer_loss_cls.new_tensor(matched_ious)

        return loss_dict

    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False, for_roi=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.

        Returns:
            list[list[dict]]: Decoded bbox, scores and labels for each layer & each batch
        """
        rets = []
        for layer_id, preds_dict in enumerate(preds_dicts):
            if self.with_img and self.with_pts and ((not self.auxiliary) or layer_id >= self.num_decoder_layers_with_pred - self.num_fusion_decoder_layers):
                layer_num_proposal = self.num_proposals * 2
            else:
                layer_num_proposal = self.num_proposals
            batch_size = preds_dict[0]['heatmap'].shape[0]

            batch_score = preds_dict[0]['heatmap'][..., -layer_num_proposal:].sigmoid()

            # one_hot = F.one_hot(self.query_labels, num_classes=self.num_classes).permute(0, 2, 1)
            # batch_score = batch_score * preds_dict[0]['query_heatmap_score'] * one_hot

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