# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:44:47 2024

attention module class, supporting scse, se, cbam, etc. 

@author: renxi
"""
import torch.nn as nn
from segmentation_models_pytorch.base.modules import SCSEModule

'''
import attse
import attsk
import attcbam
import attbam
import atteca

'''

from . import attse
from . import attsk
from . import attbam
from . import attcbam
from . import atteca
from . import attcoord


class MAttention(nn.Module):
    def __init__(self, att_type, in_channels, **params):
        super().__init__()

        if att_type is None:
            self.attention = nn.Identity(**params)
        elif att_type == "scse":
            self.attention = SCSEModule(in_channels, **params)            
        elif att_type == 'se':
            self.attention = attse.SELayer(in_channels, reduction=8)
        elif att_type == 'cbam':
            self.attention = attcbam.CBAM(in_channels, reduction=8)
        elif att_type == 'bam':
            self.attention = attbam.BAM(in_channels, reduction=8)                
        elif att_type == 'eca':
            self.attention = atteca.ECA(in_channels)                
        elif att_type == 'sk':
            self.attention = attsk.SKLayer(in_channels, in_channels)
        elif att_type == 'coord':
            self.attention = attcoord.CoordAtt(in_channels, reduction=32)    
        else:
            raise ValueError("Attention {} is not implemented".format(att_type))

    def forward(self, x):
        return self.attention(x)
    
    

    