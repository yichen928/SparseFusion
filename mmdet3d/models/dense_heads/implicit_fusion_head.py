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

from mmdet3d.models.utils import PositionEmbeddingLearned, TransformerDecoderLayer, MultiheadAttention, FFN, PointTransformer, ImageTransformer, FusionTransformer

@HEADS.register_module()
class ImplicitHead(nn.Module):
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
                 initialize_by_heatmap=False,
                 learnable_query_pos=True,
                 nms_kernel_size=3,
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
                 loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean'),
                 # others
                 train_cfg=None,
                 test_cfg=None,
                 bbox_coder=None,
                 ):
        super(ImplicitHead, self).__init__()

        self.with_img = with_img
        self.with_pts = with_pts
        self.num_proposals = num_proposals
        self.num_classes = num_classes
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.auxiliary = auxiliary

        self.bn_momentum = bn_momentum
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.initialize_by_heatmap = initialize_by_heatmap

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)
        self.loss_heatmap = build_loss(loss_heatmap)
        if self.initialize_by_heatmap is True:
            assert learnable_query_pos is False, "initialized by heatmap is conflicting with learnable query position"

        self.num_img_decoder_layers = num_img_decoder_layers * with_img
        self.num_pts_decoder_layers = num_pts_decoder_layers * with_pts
        self.num_fusion_decoder_layers = num_fusion_decoder_layers * with_img * with_pts
        self.sampling = False
        self.out_size_factor_img = out_size_factor_img

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1

        if share_prediction_heads:
            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs)))
            prediction_heads = FFN(hidden_channel, heads, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias)
            pts_prediction_heads = prediction_heads
            img_prediction_heads = prediction_heads
            fusion_prediction_heads = prediction_heads
        else:
            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs)))
            pts_prediction_heads = FFN(hidden_channel, heads, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias)
            img_prediction_heads = FFN(hidden_channel, heads, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias)
            fusion_prediction_heads = FFN(hidden_channel, heads, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias)

        self.pts_prediction = pts_prediction or (not with_img) and with_pts
        self.img_prediction = img_prediction or (not with_pts) and with_img

        query_pos_embed = PositionEmbeddingLearned(2, hidden_channel)
        if with_pts:
            pts_key_pos_embed = PositionEmbeddingLearned(2, hidden_channel)
            self.point_transformer = PointTransformer(
                pts_smca=pts_smca, hidden_channel=hidden_channel, num_heads=num_heads, num_decoder_layers=num_pts_decoder_layers,
                prediction_heads=pts_prediction_heads, ffn_channel=ffn_channel, dropout=dropout, activation=activation, test_cfg=test_cfg,
                query_pos=query_pos_embed, key_pos=pts_key_pos_embed
            )
        if with_img:
            assert learnable_query_pos or with_pts
            img_query_pos_embed = PositionEmbeddingLearned(2, hidden_channel)
            img_key_pos_embed = PositionEmbeddingLearned(2, hidden_channel)
            self.img_transformer = ImageTransformer(
                img_smca=img_smca, hidden_channel=hidden_channel, num_heads=num_heads, num_decoder_layers=num_img_decoder_layers, out_size_factor_img=out_size_factor_img,
                num_views=num_views, prediction_heads=img_prediction_heads, bbox_coder=self.bbox_coder, ffn_channel=ffn_channel, dropout=dropout, activation=activation, test_cfg=test_cfg,
                query_pos=img_query_pos_embed, key_pos=img_key_pos_embed
            )
        if with_pts and with_img:
            self.fusion_transformer = FusionTransformer(
                hidden_channel=hidden_channel, num_heads=num_heads, num_decoder_layers=num_fusion_decoder_layers, test_cfg=test_cfg,
                prediction_heads=fusion_prediction_heads, ffn_channel=ffn_channel, dropout=dropout, activation=activation,
                query_pos=query_pos_embed, key_pos=query_pos_embed
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

        self.num_decoder_layers_with_pred = self.pts_prediction * num_pts_decoder_layers + self.img_prediction * num_img_decoder_layers + self.with_img * self.with_pts * num_fusion_decoder_layers
        self.init_weights()
        self._init_assigner_sampler()

        # Position Embedding for Cross-Attention, which is re-used during training
        x_size = self.test_cfg['grid_size'][0] // self.test_cfg['out_size_factor']
        y_size = self.test_cfg['grid_size'][1] // self.test_cfg['out_size_factor']
        self.bev_pos = self.create_2D_grid(x_size, y_size)

        if self.initialize_by_heatmap:
            layers = []
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
            self.heatmap_head = nn.Sequential(*layers)
            self.class_encoding = nn.Conv1d(num_classes, hidden_channel, 1)
        else:
            # query feature
            self.pts_query_feat = nn.Parameter(torch.randn(1, hidden_channel, self.num_proposals))
            self.pts_query_pos = nn.Parameter(torch.rand([1, self.num_proposals, 2])*torch.Tensor([x_size, y_size]).reshape(1, 1, 2), requires_grad=learnable_query_pos)

            self.img_query_feat = nn.Parameter(torch.randn(1, hidden_channel, self.num_proposals))
            self.img_query_pos = nn.Parameter(torch.rand([1, self.num_proposals, 2])*torch.Tensor([x_size, y_size]).reshape(1, 1, 2), requires_grad=learnable_query_pos)

        self.nms_kernel_size = nms_kernel_size
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
        if self.with_pts and self.with_img:
            for m in self.fusion_transformer.parameters():
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

            if self.initialize_by_heatmap:
                dense_heatmap = self.heatmap_head(lidar_feat)  # [BS, C, H, W]
                heatmap = dense_heatmap.detach().sigmoid()  # [BS, num_class, H, W]
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
                pts_query_feat = lidar_feat_flatten.gather(
                    index=top_proposals_index[:, None, :].expand(-1, lidar_feat_flatten.shape[1], -1), dim=-1)  # [BS, C, num_proposals]
                self.query_labels = top_proposals_class  # [BS, num_proposals]

                # add category embedding
                one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(0, 2, 1)  # [BS, num_classes, num_proposals]
                query_cat_encoding = self.class_encoding(one_hot.float())  # [BS, C, num_proposals]
                pts_query_feat += query_cat_encoding

                pts_query_pos = bev_pos.gather(
                    index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]), dim=1)  # [BS, num_proposals, 2]
            else:
                pts_query_feat = self.pts_query_feat.repeat(batch_size, 1, 1)  # [BS, C, num_proposals]
                pts_query_pos = self.pts_query_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)  # [BS, num_proposals, 2]

        if self.with_img:
            img_feat = self.shared_conv_img(img_inputs)  # [BS * n_views, C, H, W]

            img_h, img_w, num_channel = img_inputs.shape[-2], img_inputs.shape[-1], img_feat.shape[1]
            raw_img_feat = img_feat.view(batch_size, self.num_views, num_channel, img_h, img_w).permute(0, 2, 3, 1, 4) # [BS, C, H, n_views, W]
            img_feat_collapsed = torch.max(raw_img_feat, dim=2)[0]  # [BS, C, n_views, W]

            if self.initialize_by_heatmap:
                img_query_pos = pts_query_pos.clone()  # [BS, num_proposals, 2]
                img_query_feat = nn.Parameter(torch.randn(batch_size, num_channel, self.num_proposals)).to(img_query_pos.device)

                query_pos_realmetric = img_query_pos.permute(0, 2, 1) * self.test_cfg['out_size_factor'] * \
                                       self.test_cfg['voxel_size'][0] + self.test_cfg['pc_range'][0]  # [BS, 2, num_proposals]
                query_pos_3d = torch.cat([query_pos_realmetric, torch.ones_like(query_pos_realmetric[:,:1])], dim=1).detach().clone()  # [BS, 3, num_proposals]
                # on_the_image_mask = torch.ones([batch_size, self.num_proposals]).to(query_pos_3d.device) * -1

                for sample_idx in range(batch_size):
                    lidar2img_rt = query_pos_3d.new_tensor(img_metas[sample_idx]['lidar2img'])
                    img_scale_factor = (
                        query_pos_3d.new_tensor(img_metas[sample_idx]['scale_factor'][:2]
                                                if 'scale_factor' in img_metas[sample_idx].keys() else [1.0, 1.0])
                    )
                    img_flip = img_metas[sample_idx]['flip'] if 'flip' in img_metas[sample_idx].keys() else False
                    img_crop_offset = (
                        query_pos_3d.new_tensor(img_metas[sample_idx]['img_crop_offset'])
                        if 'img_crop_offset' in img_metas[sample_idx].keys() else 0)
                    img_shape = img_metas[sample_idx]['img_shape'][:2]
                    img_pad_shape = img_metas[sample_idx]['input_shape'][:2]
                    # transform point clouds back to original coordinate system by reverting the data augmentation
                    points = apply_3d_transformation(query_pos_3d[sample_idx].T, 'LIDAR', img_metas[sample_idx],
                                                         reverse=True).detach()  # [num_proposals, 3]
                    num_points = points.shape[0]
                    for view_idx in range(self.num_views):
                        pts_4d = torch.cat([points, points.new_ones(size=(num_points, 1))], dim=-1)  # # [num_proposals, 4]
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

                        center_xs = coor_x / self.out_size_factor_img
                        center_xs = torch.round(center_xs).long().squeeze()  # [num_proposal_on_img, ]

                        on_the_image = (center_xs >= 0) & (center_xs < img_feat_collapsed.shape[-1])  # [num_proposals, ]
                        # skip the following computation if no object query fall on current image
                        if on_the_image.sum() <= 1:
                            continue
                        # on_the_image_mask[sample_idx, on_the_image] = view_idx
                        center_xs = center_xs[on_the_image]

                        img_feat_on_mask = img_feat_collapsed[sample_idx, :, view_idx, center_xs]  # [C, W]
                        img_query_feat[sample_idx, :, on_the_image] = img_feat_on_mask
                img_query_feat += query_cat_encoding
            else:
                img_query_feat = self.img_query_feat.repeat(batch_size, 1, 1)  # [BS, C, num_proposals]
                img_query_pos = self.img_query_pos.repeat(batch_size, 1, 1).to(img_feat.device)  # [BS, num_proposals, 2]

        #################################
        # transformer decoder layer (LiDAR feature as K,V)
        #################################
        ret_dicts = []
        if self.with_pts:
            pts_query_feat, pts_query_pos, pts_ret_dicts = self.point_transformer(pts_query_feat, pts_query_pos, lidar_feat_flatten, bev_pos)
            if self.pts_prediction:
                ret_dicts.extend(pts_ret_dicts)

        #################################
        # transformer decoder layer (img feature as K,V)
        #################################
        if self.with_img:
            # positional encoding for image fusion
            img_feat = raw_img_feat.permute(0, 3, 1, 2, 4)  # [BS, n_views, C, H, W]
            img_feat_flatten = img_feat.view(batch_size, self.num_views, num_channel, -1)  # [BS, n_views, C, H*W]
            if self.img_feat_pos is None:
                (h, w) = img_inputs.shape[-2], img_inputs.shape[-1]
                img_feat_pos = self.img_feat_pos = self.create_2D_grid(h, w).to(img_feat_flatten.device)  # (1, h*w, 2)
            else:
                img_feat_pos = self.img_feat_pos

            img_query_feat, img_query_pos, img_ret_dicts, self.on_the_image_mask = self.img_transformer(img_query_feat, img_query_pos, img_feat_flatten, img_feat_pos, img_metas)
            if self.img_prediction:
                ret_dicts.extend(img_ret_dicts)

        if self.with_img and self.with_pts:
            all_query_feat, all_query_pos, fusion_ret_dicts = self.fusion_transformer(pts_query_feat, pts_query_pos, img_query_feat, img_query_pos)
            ret_dicts.extend(fusion_ret_dicts)

        if self.initialize_by_heatmap:
            ret_dicts[0]['query_heatmap_score'] = heatmap.gather(index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1), dim=-1)  # [bs, num_classes, num_proposals]
            ret_dicts[0]['dense_heatmap'] = dense_heatmap

        if self.auxiliary is False:
            # only return the results of last decoder layer
            return [ret_dicts[-1]]

        # return all the layer's results for auxiliary superivison
        new_res = {}
        for key in ret_dicts[0].keys():
            if key not in ['dense_heatmap', 'dense_heatmap_old', 'query_heatmap_score']:
                new_res[key] = torch.cat([ret_dict[key] for ret_dict in ret_dicts], dim=-1)
            else:
                new_res[key] = ret_dicts[0][key]

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
        # matched_ious = np.mean(res_tuple[6])
        matched_ious = torch.cat(res_tuple[6], dim=0)

        if self.initialize_by_heatmap:
            heatmap = torch.cat(res_tuple[7], dim=0)
            return labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious, heatmap
        else:
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

            # mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
            # return labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], ious[None], int(pos_inds.shape[0]), float(mean_iou), heatmap[None]
            matched_ious = torch.ones_like(ious) * -1
            matched_ious[pos_inds] = ious[pos_inds]
            return labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], ious[None], int(pos_inds.shape[0]), matched_ious[None], heatmap[None]
        else:
            # mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
            # return labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], ious[None], int(pos_inds.shape[0]), float(mean_iou)
            matched_ious = torch.ones_like(ious) * -1
            matched_ious[pos_inds] = ious[pos_inds]
            return labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], ious[None], int(pos_inds.shape[0]), matched_ious[None]

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

        if self.initialize_by_heatmap:
            labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious, heatmap = self.get_targets(gt_bboxes_3d, gt_labels_3d, preds_dicts[0])
        else:
            labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious = self.get_targets(gt_bboxes_3d, gt_labels_3d, preds_dicts[0])        # if hasattr(self, 'on_the_image_mask'):
        #     label_weights = label_weights * self.on_the_image_mask
        #     bbox_weights = bbox_weights * self.on_the_image_mask[:, :, None]
        #     num_pos = bbox_weights.max(-1).values.sum()
        preds_dict = preds_dicts[0][0]
        loss_dict = dict()

        if self.initialize_by_heatmap:
            # compute heatmap loss
            loss_heatmap = self.loss_heatmap(clip_sigmoid(preds_dict['dense_heatmap']), heatmap, avg_factor=max(heatmap.eq(1).float().sum().item(), 1))
            loss_dict['loss_heatmap'] = loss_heatmap

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

            layer_match_ious = matched_ious[..., start:start + layer_num_proposals]
            layer_match_ious = torch.sum(layer_match_ious*(layer_match_ious>=0), dim=-1) / torch.sum(layer_match_ious>=0, dim=-1)
            layer_match_ious = torch.mean(layer_match_ious)
            start += layer_num_proposals

            # layer_iou = preds_dict['iou'][..., idx_layer*self.num_proposals:(idx_layer+1)*self.num_proposals].squeeze(1)
            # layer_iou_target = ious[..., idx_layer*self.num_proposals:(idx_layer+1)*self.num_proposals]
            # layer_loss_iou = self.loss_iou(layer_iou, layer_iou_target, layer_bbox_weights.max(-1).values, avg_factor=max(num_pos, 1))

            loss_dict[f'{prefix}_loss_cls'] = layer_loss_cls
            loss_dict[f'{prefix}_loss_bbox'] = layer_loss_bbox
            loss_dict[f'{prefix}_loss_matched_ious'] = layer_match_ious

            # loss_dict[f'{prefix}_loss_iou'] = layer_loss_iou

        # loss_dict[f'matched_ious'] = layer_loss_cls.new_tensor(matched_ious)

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
            if self.with_img and self.with_pts:
                layer_num_proposal = self.num_proposals * 2
            else:
                layer_num_proposal = self.num_proposals
            batch_size = preds_dict[0]['heatmap'].shape[0]

            batch_score = preds_dict[0]['heatmap'][..., -layer_num_proposal:].sigmoid()

            one_hot = F.one_hot(self.query_labels, num_classes=self.num_classes).permute(0, 2, 1)
            if self.with_img and self.with_pts:
                query_heatmap_score = preds_dict[0]['query_heatmap_score'] * one_hot
                query_heatmap_score = torch.cat([query_heatmap_score, query_heatmap_score], dim=2)
            else:
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