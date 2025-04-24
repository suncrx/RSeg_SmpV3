# -*- coding: utf-8 -*-
"""
Created on Sun May 19 21:16:39 2024
@author: renxi

Multiple-band Resnet encoder with attention version 1. 

This encode can handle inputs with more than 3 channels.

e.g.:
    nchns = 6
    m = mresnet18_enc(in_channels=nchns, att_type='cbam',  pretrained=True)

"""

from typing import Type, List, Union

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

#from torchvision.models.resnet import conv1x1
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck
from torchvision.models.resnet import ResNet

from .attcbam import CBAM
from .attse import SELayer

resnet_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': "https://download.pytorch.org/models/resnet34-b627a593.pth",
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
}

class MResNet_Enc1(ResNet):
    def __init__(self, 
                block: Type[Union[BasicBlock, Bottleneck]],
                layers: List[int],
                in_channels: int=3, 
                att_type: str='',
                **kwargs):

        super().__init__(block, layers, **kwargs)
                       
        #remove the avgpool and fc layers
        del self.avgpool
        del self.fc

        self.in_channels = in_channels        
        #modify the first conv layer: the input channels is 3 in the original net,
        #and our input channels can be any number. 
        self.conv1 = nn.Conv2d(in_channels, 64, 
                               kernel_size=7, stride=2, padding=3, bias=False)
        
        
        ############################################
        #att1 is after first relu, and the in_channels is 64
        #att2 is after layer4, and the in_channels is the out_channels of the 
        #last conv layer, depending on ...
        
        if isinstance(self.layer4[-1], BasicBlock):
            nchans = self.layer4[-1].conv2.out_channels
        else: 
            nchans = self.layer4[-1].conv3.out_channels
            
        if att_type.lower() == 'cbam':
            # CBAM Attention
            self.att1 = CBAM(in_channels=self.conv1.out_channels)# 64) #self.inplanes)
            self.att2 = CBAM(in_channels = nchans)
        elif att_type.lower() == 'se':
            # SE Attention 
            self.att1 = SELayer(in_channels=self.conv1.out_channels)#64) #self.inplanes)
            self.att2 = SELayer(in_channels = nchans)
        else:
            # without attention module
            self.att1 = nn.Identity()            
            self.att2 = nn.Identity()            


    def load_state_dict(self, state_dict, **kwargs):
        #fc layer is removed from the resnet, and the weights need to be
        #removed from the dict correspondingly.
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        
        #The default in_channels of conv1 layer is 3. 
        #If the in_channels is changed to 1,2,or others, we need to
        #modify the structure of the new conv1 layer in the state_dict.
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
            
        super().load_state_dict(state_dict, strict = False)


    def forward(self, x):
        features = [x]
                
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #attention 1
        x = self.att1(x)
        features.append(x)
        
        x = self.maxpool(x)                
        
        x = self.layer1(x)
        features.append(x)
        
        x = self.layer2(x)
        features.append(x)
        
        x = self.layer3(x)
        features.append(x)
        
        x = self.layer4(x)
        #attention 2
        x = self.att2(x)
        features.append(x)

        return features
    

def mresnet18_enc(in_channels=3, pretrained=True, 
                  att_type='se', **kwargs):
    model = MResNet_Enc1(BasicBlock, [2, 2, 2, 2], 
                      in_channels=in_channels, 
                      att_type=att_type, **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(resnet_urls['resnet18'])
        #we add attention layers to the original resnet, and so 
        #strcit should be set to False when load pretrained weight
        model.load_state_dict(state_dict, strict=False)
    return model


def mresnet34_enc(in_channels=3, pretrained=True, 
                  att_type='se', **kwargs):
    model = MResNet_Enc1(BasicBlock, [3, 4, 6, 3], 
                       in_channels=in_channels,
                       att_type=att_type, 
                       **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(resnet_urls['resnet34'])
        model.load_state_dict(state_dict)
    return model


def mresnet50_enc(in_channels=3, pretrained=True, 
                  att_type='se', **kwargs):
    model = MResNet_Enc1(Bottleneck, [3, 4, 6, 3], 
                       in_channels=in_channels,
                       att_type=att_type, 
                       **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(resnet_urls['resnet50'])
        model.load_state_dict(state_dict)
    return model


def mresnet101_enc(in_channels=3, pretrained=True, 
                   att_type='se', **kwargs):
    ''' ResNet-101 Model'''
    model = MResNet_Enc1(Bottleneck, [3, 4, 23, 3], 
                       in_channels=in_channels,
                       att_type=att_type, 
                       **kwargs)
    if pretrained:
        #state_dict = model_zoo.load_url(model_urls['resnet101'])
        #model.load_state_dict(state_dict)
        pass
    return model


def mresnet152_enc(in_channels=3, pretrained=True, 
                   att_type='se', **kwargs):
    model = MResNet_Enc1(Bottleneck, [3, 8, 36, 3], 
                       in_channels=in_channels,
                       att_type=att_type, 
                       **kwargs)
    if pretrained:
        #state_dict = model_zoo.load_url(model_urls['resnet152'])
        #model.load_state_dict(state_dict)
        pass
    return model

