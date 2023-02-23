import copy
import numpy as np
import torch
from torch import nn


class PETR_Heads(nn.Module):
    def __init__(self, in_channels, heads):
        super(PETR_Heads, self).__init__()
        self.embed_dims = in_channels
        heads = heads.copy()

        class_item = heads['heatmap'] if 'heatmap' in heads else heads['cls']
        class_num = class_item[0]
        class_layer = class_item[1]
        if 'heatmap' in heads:
            self.class_name = 'heatmap'
            del heads['heatmap']
        else:
            self.class_name = 'cls'
            del heads['cls']

        cls_branch = []
        for _ in range(class_layer-1):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(self.embed_dims, class_num))
        self.cls_branch = nn.Sequential(*cls_branch)

        reg_dim = 0
        reg_layer = 0
        for reg_key, reg_item in heads.items():
            reg_dim = reg_dim + reg_item[0]
            reg_layer = max(reg_layer, reg_item[1])

        reg_branch = []
        for _ in range(reg_layer-1):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU(inplace=True))
        reg_branch.append(nn.Linear(self.embed_dims, reg_dim))
        self.reg_branch = nn.Sequential(*reg_branch)
        self.heads = heads

    def forward(self, x):
        # (bs, C, N)
        x = x.transpose(2, 1)  # (bs, N, C)
        result_dict = {}

        class_out = self.cls_branch(x).transpose(2, 1)  # (bs, C, N)

        reg_out = self.reg_branch(x).transpose(2, 1) # (bs, C, N)
        result_dict[self.class_name] = class_out

        dim = 0
        for key, item in self.heads.items():
            result_dict[key] = reg_out[:, dim:dim+item[0], :]
            dim += item[0]

        return result_dict