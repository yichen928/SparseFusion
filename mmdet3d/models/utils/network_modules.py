import torch
import torch.nn as nn
import torch.nn.functional as F

from .inverse_sigmoid import inverse_sigmoid

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
