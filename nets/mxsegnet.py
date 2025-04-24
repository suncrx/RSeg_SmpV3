# -*- coding: utf-8 -*-
"""
Created on Fri May 17 09:18:14 2024

@author: renxi

My model for semantic segmentation, derived from Unet from 
segmentation_models_pytorch packages.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationModel
from segmentation_models_pytorch.base import modules as md


from .attention import MAttention

# decoder block with attention
class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        att_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        
        #attention module
        self.attention1 = MAttention(att_type, 
                                     in_channels=in_channels + skip_channels)
        
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        
        #attention module
        self.attention2 = MAttention(att_type, 
                                     in_channels=out_channels)



    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x
    


# Customized decoder
class MDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels,
                 n_blocks = 5, use_bn=True, att_type=None):
        super().__init__()
        
        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )
            
        # remove first skip with same spatial resolution    
        encoder_channels = encoder_channels[1:]
        # reverse channels
        encoder_channels = encoder_channels[::-1]                    
            
        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels       
        
        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_bn, att_type=att_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
            ]
        self.blocks = nn.ModuleList(blocks)
    
    
    def forward(self, *features):
        # remove first skip with same spatial resolution
        features = features[1:]  
        # reverse channels to start from head of encoder
        features = features[::-1]  

        head = features[0]
        skips = features[1:]

        #x = self.center(head)
        x = head
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x
    


class SegHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, 
                 kernel_size=3, 
                 activation=None, 
                 upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, 
                           kernel_size=kernel_size, 
                           padding=kernel_size // 2)
        
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        
        #activation = Activation(activation)
        activation = nn.Sigmoid() if out_channels==1 else nn.Softmax() 
        
        super().__init__(conv2d, upsampling, activation)
    
    
    def forward(self, x, saliency_map=None):
        o = super().forward(x)
        if saliency_map is not None:
            o = (o + saliency_map)/2
        return o
        
    
    
class MXSegNet(SegmentationModel):
    def __init__(self,
                 encoder_name = 'resnet50',
                 in_channels=3, 
                 n_classes=1,
                 att_type = None,
                 **keywords):
        super().__init__(**keywords)
        
        encoder_depth = 5
        decoder_channels = (256,128,64,32,16)
        
        #encoder (from segmentation_modelspytorch)
        self.encoder = get_encoder(encoder_name, 
                                   in_channels=in_channels,
                                   depth = encoder_depth,
                                   weights='imagenet')
        
        #customized decoder
        self.decoder = MDecoder(self.encoder.out_channels, 
                                decoder_channels,
                                n_blocks=encoder_depth,
                                use_bn=True,
                                att_type=att_type)
        
        #customized segmentation header        
        self.segmentation_head = SegHead(in_channels=decoder_channels[-1], 
                                         out_channels=n_classes)
        
        
        self.classification_head = None
        
        self.name = 'mxsegnet'
        self.initialize()
        
        
    def forward1(self, x):
        self.check_input_shape(x)

        x = self.encoder(x)
        o = self.decoder(*x)
        o = self.segmentation_head(o)
        
        return o


    # with auxillary input (saliency_map)
    def forward(self, x, saliency_map=None):
        self.check_input_shape(x)
        if saliency_map is not None:
            self.check_input_shape(saliency_map)
            
        x = self.encoder(x)
        o = self.decoder(*x)
        o = self.segmentation_head(o, saliency_map=saliency_map)
        
        return o
        
    

# if __name__ == '__main__':
#     chn = 6
#     ms = MXSegNet(in_channels=chn, n_classes=1, att_type='se')
#     #print(ms)
#     print(ms.encoder.conv1.weight.shape)
#     print(ms.encoder.conv1.weight[0,0])
#     d = torch.rand(4, chn, 256,256)
#     print(d.shape)
#     o=ms(d)
#     print(o.shape)
    
    