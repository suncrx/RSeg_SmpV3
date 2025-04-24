# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:06:18 2022

splitting train data randomly into two parts: train and val

@author: renxi
"""

## split data into train and val according to text file ids

import os
import shutil

###################################################################
ROOT_DIR = 'D:\\dlwater\\train_data\\wat_nj_mb'

# rgb image folder
SRC_IMG_DIR = os.path.join(ROOT_DIR, 'images')
# 0 and 255 mask image folder
SRC_MSK_DIR = os.path.join(ROOT_DIR, 'masks')


train_ids_file = 'train.txt'
val_ids_file = 'val.txt'

###################################################################
TRAIN_DIR = os.path.join(ROOT_DIR, 'train')
VAL_DIR = os.path.join(ROOT_DIR, 'val')
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

TRAIN_IMG_DIR = os.path.join(TRAIN_DIR, 'images')
TRAIN_MSK_DIR = os.path.join(TRAIN_DIR, 'masks')
VAL_IMG_DIR = os.path.join(VAL_DIR, 'images')
VAL_MSK_DIR = os.path.join(VAL_DIR, 'masks')

os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
os.makedirs(TRAIN_MSK_DIR, exist_ok=True)
os.makedirs(VAL_IMG_DIR, exist_ok=True)
os.makedirs(VAL_MSK_DIR, exist_ok=True)


textfiles = [os.path.join(ROOT_DIR, train_ids_file), 
            os.path.join(ROOT_DIR, val_ids_file)]

out_dirs = [[TRAIN_IMG_DIR, TRAIN_MSK_DIR], 
            [VAL_IMG_DIR, VAL_MSK_DIR]]

print(textfiles)
print(out_dirs)

for txtf, out_dir in zip(textfiles, out_dirs):
    if not os.path.exists(txtf):
        raise Exception('train.txt or val.txt does not exist')
        
    fo = open(txtf)
    fids = fo.read().splitlines()        
    for fid in fids:
        #print(fid)        

        src_img = os.path.join(SRC_IMG_DIR, fid)
        des_img = os.path.join(out_dir[0], fid)
        
        fid_msk = os.path.splitext(fid)[0]+'.png'
        src_msk = os.path.join(SRC_MSK_DIR, fid_msk)
        des_msk = os.path.join(out_dir[1], fid_msk)
        
        
        if os.path.exists(src_img) and os.path.exists(src_msk):
            shutil.copy(src_msk, des_msk)
            shutil.copy(src_img, des_img)
            print(fid, fid_msk, 'copied')
            
        
    fo.close()        
    '''
    mskfn = os.path.splitext(imgfn)[0] + MSK_EXT    
    mskfn2 = os.path.splitext(imgfn)[0] + MSK_EXT2    
    src_msk_path = os.path.join(SRC_MSK_DIR, mskfn)
    src_msk_path2 = os.path.join(SRC_MSK_DIR2, mskfn2)
    if os.path.exists(src_msk_path):
        src_img_path = os.path.join(SRC_IMG_DIR, imgfn)
        
        des_img_path = os.path.join(DES_IMG_DIR, imgfn)
        des_msk_path = os.path.join(DES_MSK_DIR, mskfn)
        des_msk_path2 = os.path.join(DES_MSK_DIR2, mskfn)
        
        shutil.move(src_img_path, des_img_path)
        shutil.move(src_msk_path, des_msk_path)
        shutil.move(src_msk_path2, des_msk_path2)
        
        print(src_img_path, ' -> ', des_img_path)
        print(src_msk_path, ' -> ', des_msk_path)        
        print(src_msk_path2, ' -> ', des_msk_path2)        
        count = count + 1
        '''
print('Done!')

