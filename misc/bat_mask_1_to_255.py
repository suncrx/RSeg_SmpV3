# -*- coding: utf-8 -*-
"""
Created on Sat May 18 22:30:01 2024

@author: renxi

Convert 0-1 binary mask images to 0-255 binary mask images

"""


import os
import cv2

#############################################################
mask_dir = 'D:/GeoData/DLData/Waters/WaterTiles/train/masks01'
out_folder = 'masks_255'
#############################

parent_dir = os.path.abspath(os.path.join(mask_dir, os.pardir))
out_dir = os.path.join(parent_dir, out_folder)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# test model on test images
maskfiles = os.listdir(mask_dir)
for fn in maskfiles:
    fpath = os.path.join(mask_dir, fn)
    if not os.path.isfile(fpath):
        continue
    
    print('Converting '+fpath)    
    mask = cv2.imread(fpath)    
    if mask.max()==1:
        mask = mask * 255
    
    cv2.imwrite(os.path.join(out_dir, fn), mask)
    
print('Done!')    