# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 19:43:35 2024

@author: renxi

Converting 32 bits RGBA png image to 8 bits image 

"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 20:51:38 2023

@author: renxi
"""

import os
#import imutils
#import shutil
import numpy as np
import skimage
import cv2

###############################################################

IMG_DIR = "D:\\GeoData\\DLData\\Waters\\Wat_nj\\val\\masks"


parentDir = os.path.abspath(IMG_DIR)

out_dir = os.path.join(parentDir, 'converted')
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

files = os.listdir(IMG_DIR)
for fn in files:
    fpath = os.path.join(parentDir, fn)
    print('Converting '+fpath)
    img = skimage.io.imread(fpath)
    
    if len(img.shape)>=3:
        m8u = img[:,:,0]        
    else:
        m8u = np.uint8(img*255)
       
    bfn, ext = os.path.splitext(fn)
    out_path = os.path.join(out_dir, bfn+'.png')
    print(out_path)
    cv2.imwrite(out_path, m8u)
    
print('Done!')    
