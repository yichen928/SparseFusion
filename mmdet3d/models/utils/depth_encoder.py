import torch
import torch.nn as nn
from mmdet.models.backbones.resnet import BasicBlock

from mmdet3d.models.utils import ASPP
from mmdet3d.models.utils.network_modules import LayerNorm
from mmcv.cnn import ConvModule


class DepthEncoder(nn.Module):
    def __init__(self, input_channel, hidden_channel):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channel, hidden_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channel, hidden_channel, kernel_size=3, padding=1),
        )

    def forward(self, input):
        return self.encoder(input)


class DepthEncoderSmall(nn.Module):
    def __init__(self, input_channel, hidden_channel, dcn=False):
        super().__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(input_channel-2, hidden_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(inplace=True),
        )

        self.depth_gt_conv = nn.Sequential(
            nn.Conv2d(2, hidden_channel, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channel, hidden_channel, kernel_size=1, stride=1),
        )

        self.depth_conv = nn.Sequential(
            BasicBlock(hidden_channel, hidden_channel),
            BasicBlock(hidden_channel, hidden_channel),
        )

        if dcn:
            self.dcn = ConvModule(
                    hidden_channel,
                    hidden_channel,
                    kernel_size=3,
                    padding=1,
                    conv_cfg=dict(type='DCNv2'),
                    norm_cfg=dict(type='BN2d'),
                )
        else:
            self.dcn = nn.Identity()

    def forward(self, inputs):
        inputs_feat = inputs[:, :-2]
        inputs_depth = inputs[:, -2:]

        depth_feat = self.depth_gt_conv(inputs_depth)
        inputs_feat = self.reduce_conv(inputs_feat)

        x = self.depth_conv(depth_feat+inputs_feat)
        x = self.dcn(x)
        return x


class DepthEncoderLarge(nn.Module):
    def __init__(self, input_channel, hidden_channel):
        super().__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(input_channel-2, hidden_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(inplace=True)
        )

        self.depth_gt_conv = nn.Sequential(
            nn.Conv2d(2, hidden_channel, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channel, hidden_channel, kernel_size=1, stride=1),
        )

        self.depth_conv = nn.Sequential(
            BasicBlock(hidden_channel, hidden_channel),
            BasicBlock(hidden_channel, hidden_channel),
            BasicBlock(hidden_channel, hidden_channel),
        )

        self.aspp = ASPP(hidden_channel)

    def forward(self, inputs):
        inputs_feat = inputs[:, :-2]
        inputs_depth = inputs[:, -2:]

        depth_feat = self.depth_gt_conv(inputs_depth)
        inputs_feat = self.reduce_conv(inputs_feat)

        x = self.depth_conv(inputs_feat+depth_feat)
        x = self.aspp(x)

        return x

class DepthEncoderResNet(nn.Module):
    def __init__(self, input_channel, input_channel_img, hidden_channel, depth_layers, proj_input=False):
        super().__init__()

        self.depth_layers = depth_layers

        self.conv_depth = nn.Sequential(
            # nn.Conv2d(input_channel, hidden_channel, kernel_size=1),
            # nn.ReLU(inplace=True),
            nn.Conv2d(input_channel, hidden_channel, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(inplace=True)
        )

        # self.conv_depth = nn.Sequential(
        #     nn.Conv2d(input_channel, hidden_channel, kernel_size=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(hidden_channel, hidden_channel, kernel_size=1),
        # )

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
            # self.output_layers.append(nn.Conv2d(hidden_channel, hidden_channel, kernel_size=3, padding=1))

            # self.fuse_layers.append(
            #     ConvModule(
            #         input_channel_img+hidden_channel,
            #         hidden_channel,
            #         kernel_size=3,
            #         padding=1,
            #         conv_cfg=dict(type='Conv2d'),
            #         norm_cfg=dict(type='BN2d'),
            #         act_cfg=dict(type='ReLU')
            #     )
            # )

        self.proj_input = proj_input
        if proj_input:
            self.shared_conv = nn.Conv2d(input_channel_img, hidden_channel, kernel_size=3, padding=1)

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
        new_img_inputs = []
        for i in range(len(img_inputs)):
            depth = self.layers[i](depth)
            depth = torch.cat([depth, img_inputs[i]], dim=1)
            depth = self.fuse_layers[i](depth)
            img_outputs.append(depth.clone())
            # img_outputs.append(self.output_layers[i](depth))

            if self.proj_input:
                new_img_inputs.append(self.shared_conv(img_inputs[i]))

        if self.proj_input:
            return img_outputs, new_img_inputs
        else:
            return img_outputs


#
# class DepthEncoderResNet(nn.Module):
#     def __init__(self, input_channel, input_channel_img, hidden_channel, depth_layers, proj_input=False):
#         super().__init__()
#
#         self.depth_layers = depth_layers
#
#         self.conv_depth = nn.Sequential(
#             nn.Conv2d(input_channel, hidden_channel, kernel_size=3, padding=1, bias=True),
#             nn.BatchNorm2d(hidden_channel),
#             nn.ReLU(inplace=True)
#         )
#
#         self.inplanes = hidden_channel
#         self._norm_layer = nn.BatchNorm2d
#
#         self.layers = nn.ModuleList()
#         self.depth_map_layers = nn.ModuleList()
#         self.img_map_layers = nn.ModuleList()
#         self.output_layers = nn.ModuleList()
#         for i in range(len(depth_layers)):
#             if i == 0:
#                 stride = 1
#             else:
#                 stride = 2
#
#             self.layers.append(self._make_layer(BasicBlock, hidden_channel, depth_layers[i], stride=stride))
#             # self.fuse_layers.append(nn.Conv2d(input_channel_img+hidden_channel, hidden_channel, kernel_size=3, padding=1))
#             self.depth_map_layers.append(nn.Conv2d(hidden_channel, hidden_channel, kernel_size=1))
#             self.img_map_layers.append(nn.Conv2d(input_channel_img, hidden_channel, kernel_size=1))
#
#         self.proj_input = proj_input
#         if proj_input:
#             self.shared_conv = nn.Conv2d(input_channel_img, hidden_channel, kernel_size=3, padding=1)
#
#     def _make_layer(self, block, planes, blocks, stride=1):
#         norm_layer = self._norm_layer
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride),
#                 norm_layer(planes * block.expansion),
#             )
#
#         layers = []
#
#         layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, sparse_depth, img_inputs):
#         depth = self.conv_depth(sparse_depth)
#
#         img_outputs = []
#         new_img_inputs = []
#         for i in range(len(img_inputs)):
#             depth = self.layers[i](depth)
#             # depth = torch.cat([depth, img_inputs[i]], dim=1)
#             # depth = self.fuse_layers[i](depth)
#             output = self.depth_map_layers[i](depth) + self.img_map_layers[i](img_inputs[i])
#             img_outputs.append(output)
#
#             if self.proj_input:
#                 new_img_inputs.append(self.shared_conv(img_inputs[i]))
#
#         if self.proj_input:
#             return img_outputs, new_img_inputs
#         else:
#             return img_outputs


class DepthEncoderResNetSimple(nn.Module):
    def __init__(self, input_channel, input_channel_img, hidden_channel, proj_input=False):
        super().__init__()

        self.conv_depth = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.block1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.fpn_layers = nn.ModuleList()
        self.output_layers = nn.ModuleList()

        for i in range(4):
            if i == 0:
                self.fpn_layers.append(nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1))
            else:
                self.fpn_layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1))

            self.output_layers.append(nn.Conv2d(256, hidden_channel, kernel_size=3, stride=1, padding=1))

        self.proj_input = proj_input
        if proj_input:
            self.shared_conv = nn.Conv2d(input_channel_img, hidden_channel, kernel_size=3, padding=1)

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
        depth = self.block1(depth)

        img_outputs = []
        new_img_inputs = []
        for i in range(len(img_inputs)):
            depth = self.fpn_layers[i](depth) + img_inputs[i]
            img_outputs.append(self.output_layers[i](depth))
            if self.proj_input:
                new_img_inputs.append(self.shared_conv(img_inputs[i]))

        if self.proj_input:
            return img_outputs, new_img_inputs
        else:
            return img_outputs


# class ConvNextBlock(nn.Module):
#     def __init__(self, input_channel):
#         super().__init__()
#
#         self.conv1 = nn.Conv2d(input_channel, input_channel, kernel_size=7, padding=3, stride=1, groups=input_channel)
#         self.norm = LayerNorm(input_channel, data_format="channels_first")
#         self.conv2 = nn.Conv2d(input_channel, input_channel, kernel_size=1)
#         self.act = nn.ReLU(inplace=True)
#         self.conv3 = nn.Conv2d(input_channel, input_channel, kernel_size=1)
#
#     def forward(self, input):
#         x = self.conv1(input)
#         x = self.norm(x)
#         x = self.conv2(x)
#         x = self.act(x)
#         x = self.conv3(x)
#
#         x = x + input
#         return x
#
#
# class DepthEncoderResNet(nn.Module):
#     def __init__(self, input_channel, input_channel_img, hidden_channel, depth_layers, proj_input=False):
#         super().__init__()
#
#         self.depth_layers = depth_layers
#
#         self.conv_depth = nn.Sequential(
#             nn.Conv2d(input_channel, 32, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             LayerNorm(64, data_format="channels_first")
#         )
#
#         self.downsample = nn.ModuleList()
#         self.blocks = nn.ModuleList()
#         self.outputs = nn.ModuleList()
#         self.norms = nn.ModuleList()
#         self.laterals = nn.ModuleList()
#         in_channel = 64
#         for i in range(len(depth_layers)):
#             self.blocks.append(ConvNextBlock(in_channel))
#             self.laterals.append(nn.Conv2d(input_channel_img, in_channel, kernel_size=1))
#             downsample_block =  nn.Conv2d(in_channel+in_channel, in_channel*2, kernel_size=2, stride=2, padding=0)
#             self.norms.append(LayerNorm(in_channel+in_channel, data_format="channels_first"))
#             if i != len(depth_layers) - 1:
#                 self.downsample.append(downsample_block)
#             self.outputs.append(nn.Conv2d(in_channel+in_channel, hidden_channel, kernel_size=3, stride=1, padding=1))
#             in_channel = in_channel * 2
#
#     def forward(self, sparse_depth, img_inputs):
#         depth = self.conv_depth(sparse_depth)
#
#         img_outputs = []
#         for i in range(len(img_inputs)):
#             depth = self.blocks[i](depth)
#             depth = self.norms[i](torch.cat([depth, self.laterals[i](img_inputs[i])], dim=1))
#             img_outputs.append(self.outputs[i](depth))
#             if i != len(img_inputs) - 1:
#                 depth = self.downsample[i](depth)
#
#         return img_outputs

#
# class DepthEncoderResNet(nn.Module):
#     def __init__(self, input_channel, input_channel_img, hidden_channel, depth_layers, proj_input=False):
#         super().__init__()
#
#         self.depth_layers = depth_layers
#
#         self.conv_depth = nn.Sequential(
#             nn.Conv2d(input_channel, hidden_channel, kernel_size=3, padding=1),
#             nn.BatchNorm2d(hidden_channel),
#             nn.ReLU(inplace=True)
#         )
#
#         self.inplanes = hidden_channel
#         self._norm_layer = nn.BatchNorm2d
#
#         self.layers = nn.ModuleList()
#         self.fuse_layers = nn.ModuleList()
#         self.output_layers = nn.ModuleList()
#         for i in range(len(depth_layers)):
#             if i == 0:
#                 stride = 1
#             else:
#                 stride = 2
#             self.layers.append(
#                 nn.Sequential(
#                     nn.Conv2d(hidden_channel, hidden_channel, kernel_size=3, padding=1, stride=stride),
#                     nn.BatchNorm2d(hidden_channel),
#                     nn.ReLU(inplace=True)
#                 )
#             )
#             self.fuse_layers.append(nn.Conv2d(input_channel_img+hidden_channel, hidden_channel, kernel_size=3, padding=1))
#
#         self.proj_input = proj_input
#         if proj_input:
#             self.shared_conv = nn.Conv2d(input_channel_img, hidden_channel, kernel_size=3, padding=1)
#
#     def forward(self, sparse_depth, img_inputs):
#         depth = self.conv_depth(sparse_depth)
#
#         img_outputs = []
#         new_img_inputs = []
#         for i in range(len(img_inputs)):
#             depth = self.layers[i](depth)
#             depth = torch.cat([depth, img_inputs[i]], dim=1)
#             depth = self.fuse_layers[i](depth)
#             img_outputs.append(depth.clone())
#             # img_outputs.append(self.output_layers[i](depth))
#
#             if self.proj_input:
#                 new_img_inputs.append(self.shared_conv(img_inputs[i]))
#
#         if self.proj_input:
#             return img_outputs, new_img_inputs
#         else:
#             return img_outputs
