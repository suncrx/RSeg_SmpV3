# -*- coding: utf-8 -*-
"""
Created on Fri May 24 20:53:43 2024

@author: renxi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

class SegBCELoss(torch.nn.BCELoss):
    def __init__(self):
        super().__init__()
        self.__name__ = 'BCE_loss'
        
        