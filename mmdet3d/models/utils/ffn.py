import copy
import numpy as np
import torch
from mmcv.cnn import ConvModule, build_conv_layer, kaiming_init
from mmcv.runner import force_fp32
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Linear
from torch.nn.init import xavier_uniform_, constant_


class FFN(nn.Module):
    def __init__(self,
                 in_channels,
                 heads,
                 head_conv=64,
                 final_kernel=1,
                 init_bias=-2.19,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 bias='auto',
                 **kwargs):
        super(FFN, self).__init__()

        self.heads = heads
        self.init_bias = init_bias
        for head in self.heads:
            if len(self.heads[head]) == 2:
                classes, num_conv = self.heads[head]
                need_bn = True
            else:
                classes, num_conv, need_bn = self.heads[head]

            conv_layers = []
            c_in = in_channels
            for i in range(num_conv - 1):
                if need_bn:
                    conv_layers.append(
                        ConvModule(
                            c_in,
                            head_conv,
                            kernel_size=final_kernel,
                            stride=1,
                            padding=final_kernel // 2,
                            bias=bias,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg))
                else:
                    conv_layers.append(
                        ConvModule(
                            c_in,
                            head_conv,
                            kernel_size=final_kernel,
                            stride=1,
                            padding=final_kernel // 2,
                            bias=bias,
                            conv_cfg=conv_cfg,
                            norm_cfg=None))
                c_in = head_conv

            conv_layers.append(
                build_conv_layer(
                    conv_cfg,
                    head_conv,
                    classes,
                    kernel_size=final_kernel,
                    stride=1,
                    padding=final_kernel // 2,
                    bias=True))
            conv_layers = nn.Sequential(*conv_layers)

            self.__setattr__(head, conv_layers)

    def init_weights(self):
        """Initialize weights."""
        for head in self.heads:
            if 'heatmap' in head or 'cls' in head:
                self.__getattr__(head)[-1].bias.data.fill_(self.init_bias)
            else:
                for m in self.__getattr__(head).modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m)

    def forward(self, x):
        """Forward function for SepHead.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            dict[str: torch.Tensor]: contains the following keys:

                -reg （torch.Tensor): 2D regression value with the \
                    shape of [B, 2, H, W].
                -height (torch.Tensor): Height value with the \
                    shape of [B, 1, H, W].
                -dim (torch.Tensor): Size value with the shape \
                    of [B, 3, H, W].
                -rot (torch.Tensor): Rotation value with the \
                    shape of [B, 1, H, W].
                -vel (torch.Tensor): Velocity value with the \
                    shape of [B, 2, H, W].
                -heatmap (torch.Tensor): Heatmap with the shape of \
                    [B, N, H, W].
        """
        ret_dict = dict()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict


class FFNLN(nn.Module):
    def __init__(self,
                 in_channels,
                 heads,
                 head_conv=64,
                 init_bias=-2.19,
                 **kwargs):
        super(FFNLN, self).__init__()

        self.heads = heads
        self.init_bias = init_bias
        for head in self.heads:
            if len(self.heads[head]) == 2:
                classes, num_conv = self.heads[head]
                need_norm = True
            else:
                classes, num_conv, need_norm = self.heads[head]

            conv_layers = []
            c_in = in_channels
            for i in range(num_conv - 1):
                if need_norm:
                    conv_layers.append(
                        nn.Linear(
                            c_in,
                            head_conv,
                            bias=False,
                        )
                    )
                    conv_layers.append(nn.LayerNorm(head_conv))
                else:
                    conv_layers.append(
                        nn.Linear(
                            c_in,
                            head_conv,
                            bias=True,
                        )
                    )
                conv_layers.append(nn.ReLU(inplace=True))
                c_in = head_conv

            conv_layers.append(
                nn.Linear(
                    head_conv,
                    classes,
                    bias=True,
                )
            )
            conv_layers = nn.Sequential(*conv_layers)

            self.__setattr__(head, conv_layers)

    def init_weights(self):
        """Initialize weights."""
        for head in self.heads:
            if 'heatmap' in head or 'cls' in head:
                self.__getattr__(head)[-1].bias.data.fill_(self.init_bias)
            else:
                for m in self.__getattr__(head).modules():
                    if isinstance(m, nn.Linear):
                        kaiming_init(m)

    def forward(self, x):
        """Forward function for SepHead.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            dict[str: torch.Tensor]: contains the following keys:

                -reg （torch.Tensor): 2D regression value with the \
                    shape of [B, 2, H, W].
                -height (torch.Tensor): Height value with the \
                    shape of [B, 1, H, W].
                -dim (torch.Tensor): Size value with the shape \
                    of [B, 3, H, W].
                -rot (torch.Tensor): Rotation value with the \
                    shape of [B, 1, H, W].
                -vel (torch.Tensor): Velocity value with the \
                    shape of [B, 2, H, W].
                -heatmap (torch.Tensor): Heatmap with the shape of \
                    [B, N, H, W].
        """
        ret_dict = dict()
        x = x.permute(0, 2, 1).contiguous()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)
            ret_dict[head] = ret_dict[head].permute(0, 2, 1).contiguous()

        return ret_dict

class FFNReg(nn.Module):
    def __init__(self,
                 in_channels,
                 heads,
                 head_conv=64,
                 init_bias=-2.19,
                 **kwargs):
        super(FFNReg, self).__init__()

        self.heads = heads
        self.init_bias = init_bias
        for head in self.heads:
            classes, num_conv = self.heads[head]

            conv_layers = []
            c_in = in_channels
            for i in range(num_conv - 1):
                conv_layers.append(
                    nn.Linear(
                        c_in,
                        head_conv,
                        bias=False,
                    )
                )
                if head == "heatmap" or head == "cls":
                    conv_layers.append(nn.LayerNorm(head_conv))
                conv_layers.append(nn.ReLU(inplace=True))
                c_in = head_conv

            conv_layers.append(
                nn.Linear(
                    head_conv,
                    classes,
                    bias=True,
                )
            )
            conv_layers = nn.Sequential(*conv_layers)

            self.__setattr__(head, conv_layers)

    def init_weights(self):
        """Initialize weights."""
        for head in self.heads:
            if head == 'heatmap' or head == 'cls':
                self.__getattr__(head)[-1].bias.data.fill_(self.init_bias)
            else:
                for m in self.__getattr__(head).modules():
                    if isinstance(m, nn.Linear):
                        kaiming_init(m)

    def forward(self, x):
        """Forward function for SepHead.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            dict[str: torch.Tensor]: contains the following keys:

                -reg （torch.Tensor): 2D regression value with the \
                    shape of [B, 2, H, W].
                -height (torch.Tensor): Height value with the \
                    shape of [B, 1, H, W].
                -dim (torch.Tensor): Size value with the shape \
                    of [B, 3, H, W].
                -rot (torch.Tensor): Rotation value with the \
                    shape of [B, 1, H, W].
                -vel (torch.Tensor): Velocity value with the \
                    shape of [B, 2, H, W].
                -heatmap (torch.Tensor): Heatmap with the shape of \
                    [B, N, H, W].
        """
        ret_dict = dict()
        x = x.permute(0, 2, 1).contiguous()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)
            ret_dict[head] = ret_dict[head].permute(0, 2, 1).contiguous()

        if 'bbox_3d' in ret_dict:
            ret_dict['center'] = ret_dict['bbox_3d'][:, 0:2]
            ret_dict['dim'] = ret_dict['bbox_3d'][:, 2:5]
            ret_dict['height'] = ret_dict['bbox_3d'][:, 5:6]
            ret_dict['rot'] = ret_dict['bbox_3d'][:, 6:8]
            ret_dict['vel'] = ret_dict['bbox_3d'][:, 8:10]
            del ret_dict['bbox_3d']
        return ret_dict