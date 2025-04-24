# -*- coding: utf-8 -*-
"""
Created on Thu May  2 19:11:16 2024

@author: renxi
"""

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from .attbam import conv1x1, conv3x3
from .attbam import BAM
from .attcbam import CBAM
from .base import resnet_urls

class BasicBlock_Att(nn.Module):
    expansion = 1
    def __init__(self, in_channels, hid_channels, att_type='cbam', 
                 ratio=16, stride=1, downsample=None):
        super(BasicBlock_Att, self).__init__()
        self.conv1 = conv3x3(in_channels, hid_channels, stride)
        self.bn1 = nn.BatchNorm2d(hid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(hid_channels, hid_channels)
        self.bn2 = nn.BatchNorm2d(hid_channels)
        self.downsample = downsample

        if att_type.lower() == 'cbam':
            self.atte = CBAM(hid_channels, ratio)
        elif att_type.lower() == 'bam':
            self.atte = BAM(hid_channels, ratio)            
        else:
            self.atte = nn.Identity()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # CBAM
        out = self.atte(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckBlock_Att(nn.Module): # bottelneck-block, over the 50 layers.
    expansion = 4
    def __init__(self, in_channels, hid_channels, att_type='bam', ratio=16, 
                 stride=1, downsample=None):
        super(BottleneckBlock_Att, self).__init__()
        self.downsample = downsample
        out_channels = hid_channels * self.expansion
        self.conv1 = conv1x1(in_channels, hid_channels)
        self.bn1 = nn.BatchNorm2d(hid_channels)

        self.conv2 = conv3x3(hid_channels, hid_channels, stride)
        self.bn2 = nn.BatchNorm2d(hid_channels)

        self.conv3 = conv1x1(hid_channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        if att_type == 'cbam':
            self.atte = CBAM(out_channels, ratio)
        else:
            self.atte = nn.Identity()

    def forward(self, x):
        residual = x # indentity
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        #if not self.atte is None:
        out = self.atte(out)

        out += residual
        out = self.relu(out)

        return out


class ResNet_Att1(nn.Module):
    '''
    *50-layer
        conv1 (output: 112x112)
            7x7, 64, stride 2
        conv2 (output: 56x56)
            3x3 max pool, stride 2
            [ 1x1, 64  ]
            [ 3x3, 64  ] x 3
            [ 1x1, 256 ]
        cov3 (output: 28x28)
            [ 1x1, 128 ]
            [ 3x3, 128 ] x 4
            [ 1x1, 512 ]
        cov4 (output: 14x14)
            [ 1x1, 256 ]
            [ 3x3, 256 ] x 6
            [ 1x1, 1024]
        cov5 (output: 28x28)
            [ 1x1, 512 ]
            [ 3x3, 512 ] x 3
            [ 1x1, 2048]
        _ (output: 1x1)
            average pool, 100-d fc, softmax
        FLOPs 3.8x10^9
    '''
    '''
    *101-layer
        conv1 (output: 112x112)
            7x7, 64, stride 2
        conv2 (output: 56x56)
            3x3 max pool, stride 2
            [ 1x1, 64  ]
            [ 3x3, 64  ] x 3
            [ 1x1, 256 ]
        cov3 (output: 28x28)
            [ 1x1, 128 ]
            [ 3x3, 128 ] x 4
            [ 1x1, 512 ]
        cov4 (output: 14x14)
            [ 1x1, 256 ]
            [ 3x3, 256 ] x 23
            [ 1x1, 1024]
        cov5 (output: 28x28)
            [ 1x1, 512 ]
            [ 3x3, 512 ] x 3
            [ 1x1, 2048]
        _ (output: 1x1)
            average pool, 100-d fc, softmax
        FLOPs 7.6x10^9
    '''
    def __init__(self, block, layers, 
                 in_channels=3,
                 num_classes=1000, 
                 att_type='bam', 
                 ratio=16, 
                 dilation=4):
        super(ResNet_Att1, self).__init__()
        
        self.layers = layers
        self.in_channels = 64
        self.att_type = att_type
        self.ratio = ratio
        self.dilation = dilation
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        if num_classes == 1000:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, 
                          stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, 
                          stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )

        if self.att_type.lower() == 'bam':
            self.bam1 = BAM(64*block.expansion, self.ratio, self.dilation)
            self.bam2 = BAM(128*block.expansion, self.ratio, self.dilation)
            self.bam3 = BAM(256*block.expansion, self.ratio, self.dilation)
        else:
            self.bam1 = nn.Identity()
            self.bam2 = nn.Identity()
            self.bam3 = nn.Identity()
            
            
        self.conv2 = self._make_layers(block, 64, self.layers[0])
        self.conv3 = self._make_layers(block, 128, self.layers[1], stride=2)
        self.conv4 = self._make_layers(block, 256, self.layers[2], stride=2)
        self.conv5 = self._make_layers(block, 512, self.layers[3], stride=2)
        
        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        torch.nn.init.kaiming_normal_(self.fc.weight)
        for m in self.state_dict():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        torch.nn.init.kaiming_normal_(self.fc.weight)
        for m in self.state_dict():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layers(self, block, hid_channels, n_layers, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != hid_channels * block.expansion:
            downsample = nn.Sequential(
                    conv1x1(self.in_channels, hid_channels * block.expansion, stride),
                    nn.BatchNorm2d(hid_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, hid_channels, self.att_type, 
                            self.ratio, stride, downsample))
        self.in_channels = hid_channels * block.expansion

        for _ in range(1, n_layers):
            layers.append(block(self.in_channels, hid_channels, 
                                self.att_type, self.ratio))
        return nn.Sequential(*layers)
    
    
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
        
        
    def forward(self, x):
        '''
            Example tensor shape based on resnet101
        '''
        x = self.conv1(x)

        x = self.conv2(x)
        x = self.bam1(x)

        x = self.conv3(x)
        x = self.bam2(x)

        x = self.conv4(x)
        x = self.bam3(x)

        x = self.conv5(x)
 
        x = self.avgPool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnet18_att1(pretrained=True, **kwargs):
    m = ResNet_Att1(BasicBlock_Att, [2, 2, 2, 2], **kwargs)
    if(pretrained):
        state_dict = model_zoo.load_url(resnet_urls['resnet18'])
        m.load_state_dict(state_dict)
    return m

def resnet34_att1(**kwargs):
    return ResNet_Att1(BasicBlock_Att, [3, 4, 6, 3], **kwargs)

def resnet50_att1(**kwargs):
    return ResNet_Att1(BottleneckBlock_Att, [3, 4, 6, 3], **kwargs)

def resnet101_att1(**kwargs):
    ''' ResNet-101 Model'''
    return ResNet_Att1(BottleneckBlock_Att, [3, 4, 23, 3], **kwargs)

def resnet152_att1(**kwargs):
    return ResNet_Att1(BottleneckBlock_Att, [3, 8, 36, 3], **kwargs)

