# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:24:14 2025

@author: renxi

This is an Densenet encoder that can handle multiple band images

Similar to ResNet encoder.

e.g.:
inputs: (batches, nchannels, height, width)
torch.Size([4, 6, 256, 256])

features:
torch.Size([4, 6, 256, 256])
torch.Size([4, 64, 128, 128])
torch.Size([4, 64, 64, 64])
torch.Size([4, 128, 32, 32])
torch.Size([4, 256, 16, 16])
torch.Size([4, 512, 8, 8])


"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


#%%
class MDenseNet_Enc(nn.Module):
    def __init__(self, in_channels: int=3, **kwargs):
        super(MDenseNet_Enc, self).__init__(**kwargs)
        
        # Using a pre-trained DenseNet-121
        self.densenet = models.densenet121(pretrained=True)
        
        #modify the first convolutional layer
        self.densenet.features[0] = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), 
                                             stride=(2, 2), padding=(3, 3), 
                                             bias=False)
        
     
    def forward(self, x):    
        feats = [x]
        
        x = self.densenet.features[0](x)
        x = self.densenet.features[1](x)
        x = self.densenet.features[2](x)        
        feats.append(x)
        
        # maxpooling
        x = self.densenet.features[3](x)
        feats.append(x)
        
        # dense block and transition 
        x = self.densenet.features[4](x)
        x = self.densenet.features[5](x)
        feats.append(x)

        # dense block and transition 
        x = self.densenet.features[6](x)
        x = self.densenet.features[7](x)
        feats.append(x)               

        # dense block and transition 
        x = self.densenet.features[8](x)
        x = self.densenet.features[9](x)
        feats.append(x)               
        
        # dense block and transition 
        #x = self.densenet.features[10](x)
        #x = self.densenet.features[11](x)
        #feats.append(x)               
        
        return feats
        
    
if __name__ =='__main__':
    # Example usage
    nchs = 6
    enc = MDenseNet_Enc(in_channels=nchs)    
    print(enc)    
    d = torch.rand(4, nchs, 256, 256)
    fts = enc(d)
    print(d.shape)
    print('features:')
    for ft in fts:
        print(ft.shape)
