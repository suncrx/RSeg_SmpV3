# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 12:42:51 2023

@author: renxi
"""

#run
# python visualize_data.py --cfg configs/waters.yaml
# python visualize_data.py -c configs/waters.yaml

# %% import installed packages
import yaml
import os

import numpy as np
import matplotlib.pyplot as plt
import cv2

#import my modules
from dataset import SegDataset

# %% parameters

# default configuration file path
# configuration file specifies the dataset path and parameters 
CFG_PATH = 'data/waters_nj.yaml'

W, H = 256, 256
       
print('Loading parameters from configuration file : ', CFG_PATH)        
with open(CFG_PATH) as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)
print('Configs:')
print(cfg)

#sys.exit(0)

# %% prepare datasets
# init train, val, test sets
print('Prepare data ...')
train_dataset = SegDataset(cfg['path'], "train", 
                           n_classes=1, imgH=H, imgW=W, apply_aug = False)
valid_dataset = SegDataset(cfg['path'], "val", 
                           n_classes=1, imgH=H, imgW=W, apply_aug = False)
#test_dataset  = SegDataset(cfg['root_dir'], "test", 
#                           n_classes=1, imgH=H, imgW=W, apply_aug = False)

# It is a good practice to check datasets don't intersects with each other
train_imgs = train_dataset.get_image_filepaths()
val_imgs = valid_dataset.get_image_filepaths()
#test_imgs = test_dataset.get_image_filepaths()
#assert set(test_imgs).isdisjoint(set(train_imgs))
#assert set(test_imgs).isdisjoint(set(val_imgs))
assert set(val_imgs).isdisjoint(set(train_imgs))
print(f"Train size: {len(train_dataset)}")
print(f"Valid size: {len(valid_dataset)}")
#print(f"Test size: {len(test_dataset)}")


#------------------------------------------------------------------------------
out_dir = os.path.join(cfg['path'], 'vis')
os.makedirs(out_dir, exist_ok=True)

# lets look at some randomly selected samples
num = 5
ds = train_dataset

idxs = np.int32(np.random.random(num)*(len(ds)-1))
for i in idxs:
    img, msk = ds[i]
    oimg, omsk = ds.get_image_and_mask(i)
    imgpath, mskpath = ds.get_image_and_mask_path(i)
    plt.subplot(1,2,1)
    plt.imshow(oimg) # for visualization we have to transpose back to HWC
    plt.subplot(1,2,2)
    plt.imshow(omsk)  # for visualization we have to remove 3rd dimension of mask
    plt.show()
    
    # overlap boundaries on images
    cnt,hit = cv2.findContours(omsk, cv2.RETR_EXTERNAL, #cv2.RETR_TREE,
                               cv2.CHAIN_APPROX_TC89_KCOS)
                               #cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(oimg, cnt, -1, (0,255,0), 2)
    
    basname = os.path.basename(imgpath)
    opath = os.path.join(out_dir, basname)
    cv2.imwrite(opath, oimg)    


# ----
