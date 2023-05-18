import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear
import math
import warnings
from typing import Optional, no_type_check
from torch.autograd.function import Function, once_differentiable

from mmdet3d.models.utils import MultiheadAttention
from mmcv.runner import BaseModule
from mmcv import deprecated_api_warning
from mmcv.cnn import constant_init, xavier_init
from mmcv.runner import BaseModule
from mmdet3d.models.utils.ops.modules import MSDeformAttn


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, level_num=4, dim_feedforward=2048, dropout=0.1, activation="relu",
                 self_posembed=None, cross_posembed=None, cross_only=False, n_points=4):
        super().__init__()
        self.cross_only = cross_only
        if not self.cross_only:
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MSDeformAttn(d_model, level_num, nhead, n_points)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        def _get_activation_fn(activation):
            """Return an activation function given a string"""
            if activation == "relu":
                return F.relu
            if activation == "gelu":
                return F.gelu
            if activation == "glu":
                return F.glu
            raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

        self.activation = _get_activation_fn(activation)

        self.self_posembed = self_posembed
        self.cross_posembed = cross_posembed

    def with_pos_embed(self, tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, query, key, query_pos, key_pos, reference_points, level_start_index, spatial_shapes, query_padding_mask=None, input_padding_mask=None):

        """
        :param query: B C Pq
        :param key: B C Pk
        :param query_pos: B Pq 3/6
        :param key_pos: B Pk 3/6
        :param value_pos: [B Pq 3/6]
        :return:
        """
        # NxCxP to PxNxC
        if self.self_posembed is not None:
            query_pos_embed = self.self_posembed(query_pos).permute(2, 0, 1)
        else:
            query_pos_embed = None
        if self.cross_posembed is not None:
            key_pos_embed = self.cross_posembed(key_pos).permute(2, 0, 1)
        else:
            key_pos_embed = None

        query = query.permute(2, 0, 1)
        key = key.permute(2, 0, 1)

        if not self.cross_only:
            q = k = v = self.with_pos_embed(query, query_pos_embed)
            query2 = self.self_attn(q, k, value=v, key_padding_mask=query_padding_mask)[0]
            query = query + self.dropout1(query2)
            query = self.norm1(query)

        query_d = self.with_pos_embed(query, query_pos_embed)
        input_flatten_d = self.with_pos_embed(key, key_pos_embed)
        query2 = self.multihead_attn(query=query_d.permute(1, 0, 2),
                    input_flatten=input_flatten_d.permute(1, 0, 2), reference_points=reference_points,
                    input_spatial_shapes=spatial_shapes, input_level_start_index=level_start_index,
                    input_padding_mask=input_padding_mask
                )


        query2 = query2.permute(1, 0, 2)
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm3(query)

        # NxCxP to PxNxC
        query = query.permute(1, 2, 0)
        return query
