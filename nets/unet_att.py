# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 21:31:56 2024

@author: renxi


Attention Unet 

The Unet is created from segmentation_models_pytorch and then
modified by adding attention modules. 

supported attention:
['se', 'sk', 'cbam', 'bam', 'eca', 'coord']
 
"""

#import torch
import segmentation_models_pytorch as smp

from .attention import SELayer, CBAM, BAM, ECA, SKLayer, CoordAtt

# Unet model with attentions
class Unet_Attention(smp.Unet):
    def __init__(self, decoder_attention_type = 'se', **kwargs):
        #print(kwargs)
        super(Unet_Attention, self).__init__(**kwargs)  
        
        att_type = decoder_attention_type.lower()
        assert att_type in ['se', 'sk', 'cbam', 'bam', 'eca', 'coord']
        
        #Adding attention modules. 
        #This can be done by replacing the 
        #Identity modules attention1 and attention2 with SE, CBAM, and ...
        for blk in self.decoder.blocks:
            out_channels = blk.conv1[0].out_channels
            in_channels = blk.conv1[0].in_channels            
            if att_type == 'se':
                att1 = SELayer(in_channels, reduction=8)
            elif att_type == 'cbam':
                att1 = CBAM(in_channels, reduction=8)
            elif att_type == 'bam':
                att1 = BAM(in_channels, reduction=8)                
            elif att_type == 'eca':
                att1 = ECA(in_channels)                
            elif att_type == 'sk':
                att1 = SKLayer(in_channels, in_channels)
            elif att_type == 'coord':
                att1 = CoordAtt(in_channels, reduction=32)
                    
            blk.attention1 = att1
            
            
            out_channels = blk.conv2[0].out_channels
            in_channels = blk.conv2[0].in_channels            
            if att_type.lower() == 'se':
                att2 = SELayer(in_channels, reduction=8)
            elif att_type.lower() == 'cbam':
                att2 = CBAM(in_channels, reduction=8)
            elif att_type == 'bam':
                att2 = BAM(in_channels, reduction=8)                                
            elif att_type == 'eca':
                att2 = ECA(in_channels)                
            elif att_type == 'sk':
                att2 = SKLayer(in_channels, in_channels)
            elif att_type == 'coord':
                att2 = CoordAtt(in_channels, reduction=32)
                
            blk.attention2 = att2
            

