import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import os
import random
from typing import Type, Any, Union, List
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1

random.seed(1)
cwd = os.getcwd()


def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        elif 'module.' in name:
            name = 'encoder.' + name[15:]
        new_state_dict[name] = v
    return new_state_dict


########### Image Encoder start#########
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.add_layer = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.add_layer(x)

        x = torch.flatten(x, 1)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def resnet18(
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    return model
########### Image Encoder fin#########


########### Image Decoder start#########
class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decode = nn.Sequential(
            View((-1, 128, 7, 7)),
            nn.ConvTranspose2d(128, 512, 1, 1),
            nn.ConvTranspose2d(512, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 26, 4, 2, 1)
        )

    def forward(self, x_):
        x = self.decode(x_)
        return x
########### Image Decoder fin#########


########### Image AE model (Encoder + Decoder) start#########
class Image_AE_model(nn.Module):

    def __init__(self, encoder, decoder):
        super(Image_AE_model, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        y = self.encoder(x)
        y = torch.flatten(y, 1)
        z = self.decoder(y)
        return y, z
########### Image AE model (Encoder + Decoder) fin#########


########### Impression Encoder start#########
class DeepSets(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc0 = torch.nn.Linear(300, 300)
        self.fc1 = torch.nn.Linear(300, 300)

    def pooling(self, x, mode='mean'):
        if mode == 'sum':
            x = x.sum(0)
        elif mode == 'mean':
            x = x.mean(0)
        elif mode == 'max':
            x = x.max(1)[0]
        return x

    def forward(self, x, flag):
        # [batch, sets, channels, x, y]
        # x = x.to(device)
        x = F.relu(self.fc0(x))
        x = self.fc1(x)
        if flag:
            # if input set
            x = self.pooling(x)
        return x


class imp_Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc0 = torch.nn.Linear(300, 1024)
        self.fc1 = torch.nn.Linear(1024, 2048)
        self.fc2 = torch.nn.Linear(2048, 512)
        # self.fc2 = torch.nn.Linear(512, 300)

    def forward(self, x):
        # [batch, sets, channels, x, y]
        # x = x.to(device)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
########### Impression Encoder fin#########


########### Impression Decoder start#########
class imp_Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc0 = torch.nn.Linear(512, 2048)
        self.fc1 = torch.nn.Linear(2048, 1024)
        self.fc2 = torch.nn.Linear(1024, 300)
        # self.fc2 = torch.nn.Linear(512, 300)

    def forward(self, x):
        # [batch, sets, channels, x, y]
        # x = x.to(device)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
########### Impression Decoder fin#########


########### Impression AE model start#########
class IntegratedModel(nn.Module):

    def __init__(self, deepsets, imp_encoder, imp_decoder):
        super().__init__()

        self.deepsets = deepsets
        self.imp_encoder = imp_encoder
        self.imp_decoder = imp_decoder

    def forward(self, inputs):
        output = []
        if len(inputs) != 1:
            # batch size == len(inputs)
            for input in inputs:
                input = input.to('cuda')
                y = self.deepsets(input, True)
                output.append(y)
        else:
            input = inputs[0]
            input = input.to('cuda')
            if len(input) == 300:
                # input an impression
                y = self.deepsets(input, False)
            else:
                # input set impressions
                y = self.deepsets(input, True)
            output.append(y)
        output = torch.stack(output)
        imp_feature = self.imp_encoder(output)
        output2 = self.imp_decoder(imp_feature)
        return output, imp_feature, output2
########### Impression AE model fin#########
