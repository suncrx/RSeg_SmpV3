# -*- coding: utf-8 -*-
"""
Created on Sun May 19 20:42:54 2024

@author: renxi
"""

from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch import Tensor

from torchvision.models.resnet import conv1x1
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck

from .base import resnet_urls


class ResNetEnc(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        in_channels: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.in_channels = in_channels
        
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
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
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)


    def load_state_dict(self, state_dict):
        #fc layer is removed from the resnet, and the weights need to be
        #removed from the dict correspondingly.
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        
        #The default in_channels of conv1 layer is 3. 
        #If the in_channels is changed to 1,2,or others, we need to
        #modify the structure of the new conv1 layer.
        if self.in_channels < 3:
            new_weight = state_dict['conv1.weight']
            new_weight = new_weight[:,0:self.in_channels]
            state_dict['conv1.weight'] = nn.parameter.Parameter(new_weight)
            
        elif self.in_channels > 3:
            weight = state_dict['conv1.weight']
            out_channels, ks1, ks2 = weight.shape[0],weight.shape[2],weight.shape[3]
            
            new_weight = torch.Tensor(out_channels, self.in_channels, ks1, ks2)
            for i in range(self.in_channels):
                new_weight[:, i] = weight[:, i % 3]

            new_weight = new_weight * (3 / self.in_channels)
            state_dict['conv1.weight'] = nn.parameter.Parameter(new_weight)
            
        super(ResNetEnc, self).load_state_dict(state_dict, strict=True)
        
        
    def forward(self, x: Tensor) -> Tensor:
        '''
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #x = self.avgpool(x)
        #x = torch.flatten(x, 1)
        #x = self.fc(x)

        return x
        '''
        '''
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        '''
        features = [x]
                
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)
        
        x = self.maxpool(x)                
        
        x = self.layer1(x)
        features.append(x)
        
        x = self.layer2(x)
        features.append(x)
        
        x = self.layer3(x)
        features.append(x)
        
        x = self.layer4(x)
        features.append(x)

        return features



def resnet18_enc(pretrained=True, **kwargs):
    model = ResNetEnc(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(resnet_urls['resnet18'])
        model.load_state_dict(state_dict)
    return model


def resnet34_enc(pretrained=True, **kwargs):
    model = ResNetEnc(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(resnet_urls['resnet34'])
        model.load_state_dict(state_dict)
    return model


def resnet50_enc(pretrained=True, **kwargs):
    model = ResNetEnc(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(resnet_urls['resnet50'])
        model.load_state_dict(state_dict)
    return model


def resnet101_enc(pretrained=True, **kwargs):
    ''' ResNet-101 Model'''
    model = ResNetEnc(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        #state_dict = model_zoo.load_url(model_urls['resnet101'])
        #model.load_state_dict(state_dict)
        pass
    return model


def resnet152_enc(pretrained=True, **kwargs):
    model = ResNetEnc(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        #state_dict = model_zoo.load_url(model_urls['resnet152'])
        #model.load_state_dict(state_dict)
        pass
    return model
