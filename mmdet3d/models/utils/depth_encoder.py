import torch
import torch.nn as nn
from mmdet.models.backbones.resnet import BasicBlock

from mmdet3d.models.utils.network_modules import LayerNorm
from mmcv.cnn import ConvModule


class DepthEncoderResNet(nn.Module):
    def __init__(self, input_channel, input_channel_img, hidden_channel, depth_layers):
        super().__init__()

        self.depth_layers = depth_layers

        self.conv_depth = nn.Sequential(
            nn.Conv2d(input_channel, hidden_channel, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(inplace=True)
        )

        self.inplanes = hidden_channel
        self._norm_layer = nn.BatchNorm2d

        self.layers = nn.ModuleList()
        self.fuse_layers = nn.ModuleList()
        self.output_layers = nn.ModuleList()
        for i in range(len(depth_layers)):
            if i == 0:
                stride = 1
            else:
                stride = 2

            self.layers.append(self._make_layer(BasicBlock, hidden_channel, depth_layers[i], stride=stride))
            self.fuse_layers.append(nn.Conv2d(input_channel_img+hidden_channel, hidden_channel, kernel_size=3, padding=1))


    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, sparse_depth, img_inputs):
        depth = self.conv_depth(sparse_depth)

        img_outputs = []
        for i in range(len(img_inputs)):
            depth = self.layers[i](depth)
            depth = torch.cat([depth, img_inputs[i]], dim=1)
            depth = self.fuse_layers[i](depth)
            img_outputs.append(depth.clone())

        return img_outputs
