# -*- coding: utf-8 -*-
'''
Created on Thu May 16 21:23:00 2024
@author: renxi

Resnet encoder 
'''

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from .base import BasicBlock, BottleneckBlock
from .base import resnet_urls


class ResNetEnc(nn.Module):
    def __init__(self, block, layers, strides=(2, 2, 2, 2),                  
                 dilations=(1, 1, 1, 1),
                 in_channels = 3,
                 zero_init_residual = False):
        self.in_channels = in_channels
        self.inplanes = 64
        super(ResNetEnc, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=strides[0], padding=3,
                               bias=False)
        #self.bn1 = FixedBatchNorm(64)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, dilation=dilations[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3])
        self.inplanes = 1024
        
        #remove avgpooling and fc layers 
        #self.avgpool = nn.AvgPool2d(7, stride=1)
        #self.fc = nn.Linear(512 * block.expansion, 1000)
        
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
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, dilation=1)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

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
        

    def forward(self, x):
        '''
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        '''
        features = []
                
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

        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        #x = self.fc(x)
        #return x
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
    model = ResNetEnc(BottleneckBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(resnet_urls['resnet50'])
        model.load_state_dict(state_dict)
    return model


def resnet101_enc(pretrained=True, **kwargs):
    ''' ResNet-101 Model'''
    model = ResNetEnc(BottleneckBlock, [3, 4, 23, 3], **kwargs)
    if pretrained:
        #state_dict = model_zoo.load_url(model_urls['resnet101'])
        #model.load_state_dict(state_dict)
        pass
    return model


def resnet152_enc(pretrained=True, **kwargs):
    model = ResNetEnc(BottleneckBlock, [3, 8, 36, 3], **kwargs)
    if pretrained:
        #state_dict = model_zoo.load_url(model_urls['resnet152'])
        #model.load_state_dict(state_dict)
        pass
    return model


   
    
    