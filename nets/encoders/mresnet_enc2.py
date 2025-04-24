# -*- coding: utf-8 -*-
"""
Created on Sun May 19 21:16:39 2024
@author: renxi

Customed Resnet encoder with attention version 2

In this network, the attention is added to BasicBlock and Bottleneck,
seeing the derived class BasicBlockAtt and BottleneckAtt.


"""

from typing import Type, List, Union, Optional, Callable

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch import Tensor

from torchvision.models.resnet import conv1x1, conv3x3
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck
from torchvision.models.resnet import ResNet

from .attcbam import CBAM
from .attbam import BAM

resnet_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': "https://download.pytorch.org/models/resnet34-b627a593.pth",
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
}

#derived block with attention
class BasicBlockAtt(BasicBlock):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        att_type = 'cbam',
        reduction = 8,
    ) -> None:
        super().__init__(inplanes, planes, stride, downsample,
                         groups, base_width, dilation, norm_layer)
        
        if att_type.lower() == 'cbam':
            self.atte = CBAM(planes, reduction)
        elif att_type.lower() == 'bam':
            self.atte = BAM(planes, reduction)            
        else:
            self.atte = nn.Identity()
            

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        #att
        out = self.atte(out)
            
        out += identity
        out = self.relu(out)

        return out
        

class BottleneckAtt(Bottleneck):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        att_type = 'cbam',
        reduction = 8,
    ) -> None:
        super().__init__()
        
        if att_type.lower() == 'cbam':
            self.atte = CBAM(planes, reduction)
        elif att_type.lower() == 'bam':
            self.atte = BAM(planes, reduction)                
        else:
            self.atte = nn.Identity()
        

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.atte(out)
        
        out += identity
        out = self.relu(out)

        return out
    

#Our resnet encoder with attention blocks    
class MResNetAttEnc(ResNet):
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
        

    def forward(self, x):
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
    

def mresnet18_enc2(in_channels=3, pretrained=True, 
                  att_type='cbam', **kwargs):
    model = MResNetAttEnc(BasicBlockAtt, [2, 2, 2, 2], 
                      in_channels=in_channels, 
                      att_type=att_type, **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(resnet_urls['resnet18'])
        #we add attention layers to the original resnet, and so 
        #strcit should be set to False when load pretrained weight
        model.load_state_dict(state_dict, strict=False)
    return model


def mresnet34_enc2(in_channels=3, pretrained=True, 
                  att_type='cbam', **kwargs):
    model = MResNetAttEnc(BasicBlockAtt, [3, 4, 6, 3], 
                       in_channels=in_channels,
                       att_type=att_type, 
                       **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(resnet_urls['resnet34'])
        model.load_state_dict(state_dict, strict=False)
    return model


def mresnet50_enc2(in_channels=3, pretrained=True, 
                  att_type='cbam', **kwargs):
    model = MResNetAttEnc(BottleneckAtt, [3, 4, 6, 3], 
                       in_channels=in_channels,
                       att_type=att_type, 
                       **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(resnet_urls['resnet50'])
        model.load_state_dict(state_dict, strict=False)
    return model


def mresnet101_enc2(in_channels=3, pretrained=True, 
                   att_type='cbam', **kwargs):
    ''' ResNet-101 Model'''
    model = MResNetAttEnc(BottleneckAtt, [3, 4, 23, 3], 
                       in_channels=in_channels,
                       att_type=att_type, 
                       **kwargs)
    if pretrained:
        #state_dict = model_zoo.load_url(model_urls['resnet101'])
        #model.load_state_dict(state_dict)
        pass
    return model


def mresnet152_enc2(in_channels=3, pretrained=True, 
                   att_type='cbam', **kwargs):
    model = MResNetAttEnc(BottleneckAtt, [3, 8, 36, 3], 
                       in_channels=in_channels,
                       att_type=att_type, 
                       **kwargs)
    if pretrained:
        #state_dict = model_zoo.load_url(model_urls['resnet152'])
        #model.load_state_dict(state_dict)
        pass
    return model

