# -*- coding: utf-8 -*-
"""
Created on Sat May 18 19:59:35 2024

@author: renxi
"""
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from .attcbam import CBAM
from .attse import SELayer

from .base import BasicBlock, BottleneckBlock, conv1x1
from .base import resnet_urls


class ResNet_Att2(nn.Module):
    def __init__(self, block, layers, 
                 in_channels = 3, 
                 num_classes=1000, 
                 att_type = 'se',
                 zero_init_residual=False,
                 groups=1, width_per_group=64, 
                 replace_stride_with_dilation=None,
                 norm_layer=None):

        super(ResNet_Att2, self).__init__()

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
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, 
                               stride=2, padding=3,  bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        ############################################
        if att_type.lower() == 'cbam':
            # CBAM Attention
            self.att1 = CBAM(in_channels=self.inplanes)
        elif att_type.lower() == 'se':
            self.att1 = SELayer(in_channels=self.inplanes)
        else:
            self.att1 = nn.Identity()
        

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block,  64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilation=self.dilation) 
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilation=self.dilation) 
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilation=self.dilation) 
        
        
        ####################################
        #self.ca1 = ChannelAttention(self.inplanes)
        #self.sa1 = SpatialAttention()        
        if att_type.lower() == 'cbam':
            # CBAM Attention
            self.att2 = CBAM(in_channels=self.inplanes)
        elif att_type.lower() == 'se':
            self.att2 = SELayer(in_channels=self.inplanes)            
        else:
            self.att2 = nn.Identity()

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
                if isinstance(m, BottleneckBlock):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        norm_layer = self._norm_layer
        downsample = None
        #previous_dilation = self.dilation
        #if dilate:
        #    self.dilation *= stride
        #    stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        #layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
        #                    self.base_width, previous_dilation, norm_layer))
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=1))
        
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            #layers.append(block(self.inplanes, planes, groups=self.groups,
            #                    base_width=self.base_width, dilation=self.dilation,
            #                    norm_layer=norm_layer))
            layers.append(block(self.inplanes, planes, dilation=1))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        #x = self.ca(x) * x
        #x = self.sa(x) * x
        x = self.att1(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #x = self.ca1(x) * x
        #x = self.sa1(x) * x
        x = self.att2(x)


        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x

    def load_state_dict(self, state_dict):        
        #The default in_channels of conv1 layer is 3. 
        #If the in_channels is changed to 1,2,or others, state_dict needs to
        #be modified to match the structure of the new conv1 layer.
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
        

        # !!!!!!! Set strict to False when you load pretrained weights to 
        # your modified network, or it raises errors. 
        # This keyword (False) ignores non-matching keys. 
        super().load_state_dict(state_dict, strict=False)

        
def resnet18_att2(pretrained=True, **kwargs):
    m = ResNet_Att2(BasicBlock, [2, 2, 2, 2], **kwargs)
    if(pretrained):
        state_dict = model_zoo.load_url(resnet_urls['resnet18'])
        m.load_state_dict(state_dict)
    return m


def resnet34_att2(pretrained=True, **kwargs):
    m = ResNet_Att2(BasicBlock, [3, 4, 6, 3], **kwargs)
    if(pretrained):
        state_dict = model_zoo.load_url(resnet_urls['resnet34'])
        m.load_state_dict(state_dict)
    return m

def resnet50_att2(pretrained=True, **kwargs):
    m = ResNet_Att2(BottleneckBlock, [3, 4, 6, 3], **kwargs)
    if(pretrained):
        state_dict = model_zoo.load_url(resnet_urls['resnet50'])
        m.load_state_dict(state_dict)
    return m


def resnet101_att2(pretrained=True, **kwargs):
    ''' ResNet-101 Model'''
    m = ResNet_Att2(BottleneckBlock, [3, 4, 23, 3], **kwargs)
    if(pretrained):
        state_dict = model_zoo.load_url(resnet_urls['resnet101'])
        m.load_state_dict(state_dict)
    return m


def resnet152_att2(pretrained=True, **kwargs):
    m = ResNet_Att2(BottleneckBlock, [3, 8, 36, 3], **kwargs)
    if(pretrained):
        state_dict = model_zoo.load_url(resnet_urls['resnet152'])
        m.load_state_dict(state_dict)
    return m


    

        