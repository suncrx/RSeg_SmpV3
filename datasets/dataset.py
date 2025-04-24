'''
Dataset class for semantic segmentation, 
using OpenCV to read images and masks

directory template
--data
     |--train
            |--images
            |--masks
     |----val
            |--images
            |--masks
          
'''
# import the necessary packages
import os, sys
import random
import cv2

import numpy as np
import matplotlib.pylab as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from dataug import get_train_aug, get_val_aug

'''
torch.utils.data.Dataset is an abstract class representing a dataset. 
Your custom dataset should inherit Dataset and override the following 
methods:

    __len__ : so that len(dataset) returns the size of the dataset.

    __getitem__: to support the indexing such that dataset[i] can be 
                 used to get i-th sample.

'''

# Dataset for segmentation
# return numpy image (H, W, 3)
# and numpy mask (H, W)
class SegDataset(Dataset):
    def __init__(self, img_dir, mode="train", 
                 n_classes=1, imgH=None, imgW=None,
                 #preprocess=None,
                 apply_aug=False, sub_size=-1):
        
        assert mode in {"train", "val", "test"}
        self.mode = mode
        self.imgW = imgW
        self.imgH = imgH        
        
        # binary segmentation : n_classes = 1
        # multi-class segmentation : n_classes > 1
        self.n_classes = n_classes
       
        if apply_aug:
            if mode == 'train':
                self.aug = get_train_aug(height=imgH, width=imgW)
            else:
                self.aug = get_val_aug(height=imgH, width=imgW)
        else:
            self.aug = None
        
        # search image and mask filepaths
        self.imgs_dir = img_dir        
        if not os.path.exists(self.imgs_dir):
            print("ERROR: Cannot find directory " + self.imgs_dir)
            sys.exit()
        # mask-image directory
        self.msks_dir = os.path.join(os.path.dirname(self.imgs_dir), "masks")
        
        # search image and mask files
        print('Scanning files in %s ... ' % self.mode)
        print(' ' + self.imgs_dir)
        print(' ' + self.msks_dir)        
        self.imgPairs = self._list_files()
        print(" #image pairs: ", len(self.imgPairs))
        
        #subset the dataset
        #randomly select num items
        if sub_size > 0 and sub_size <= 1:
            num = np.int32(len(self.imgPairs)*sub_size)
            self.imgPairs = random.sample(self.imgPairs, num)
        elif sub_size > 1:
            num = min(len(self.imgPairs), np.int32(sub_size))
            self.imgPairs = random.sample(self.imgPairs, num)
        
        print(" #image pairs: ", len(self.imgPairs))



    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.imgPairs)


    # return a tuple  (image, mask)
    # image: tensor image with shape (3, H, W), and data range (0 ~ 1.0)
    # mask: binary mask image of size (H, W), with value 0 and 1.0.
    def __getitem__(self, idx):
        # read image and mask
        imagePath = self.imgPairs[idx]['image']
        maskPath = self.imgPairs[idx]['mask']
        oimage, omask = self.get_image_and_mask(idx)
        

        # resize image and mask if necessary        
        if (self.imgW is not None) and (self.imgH is not None):            
            if oimage.shape[0:2] != (self.imgH, self.imgW):
                oimage = cv2.resize(oimage, (self.imgW, self.imgH),
                                   interpolation=cv2.INTER_NEAREST)            
            if omask.shape != (self.imgH, self.imgW):
                omask = cv2.resize(omask, (self.imgW, self.imgH), 
                                  interpolation=cv2.INTER_NEAREST)           
        
        # apply augmentation
        # image, mask = copy.deepcopy(oimage), copy.deepcopy(omask)
        image, mask = oimage, omask
        if self.aug:
            sample = self.aug(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
       
        # apply preprocessing on image (not on mask)
        # "mean": [0.485, 0.456, 0.406],
        # "std":  [0.229, 0.224, 0.225],
       
        # [1] convert image to tensor of shape [3, H, W], with 
        # values between(0, 1.0)
        image = transforms.ToTensor()(image)
        
        # [2] transform mask to tensor
        # binary segmentation            
        if self.n_classes < 2:            
            # convert to (0, 1) float 
            mask = transforms.ToTensor()(mask > 0).float()
        # multi-class segmentation
        else:           
            # convert label mask to one-hot tensor
            mask = torch.squeeze(transforms.ToTensor()(mask).long())
            mask = F.one_hot(mask, num_classes=self.n_classes)
            mask = torch.movedim(mask, 2, 0)
        
        return image, mask     
        #return {'image':image, 'mask':mask,
        #        'ori_image':oimage, 'ori_mask':omask,
        #        'image_path':imagePath,'mask_path':maskPath}
            

    
    # read the image and mask at index idx
    def get_image_and_mask(self, idx):
        # grab the image and mask path from the current index
        imagePath = self.imgPairs[idx]['image']
        maskPath = self.imgPairs[idx]['mask']
        
       
        # Using OpenCV to load image from disk, swap its channels 
        # from BGR to RGB. format [H, W, C]
        image = cv2.imread(imagePath)
        if image is None:
            raise "ERROR: can not read image: " + imagePath
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
                            
        # read the associated mask from disk in grayscale mode
        if os.path.exists(maskPath):
            mask = cv2.imread(maskPath, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise "ERROR: can not read mask image: " + maskPath
        else:
            mask = np.zeros(image.shape[0:2], dtype=np.uint8)        
        
        return image, mask
    
    
    # get all image paths
    def get_image_filepaths(self):
        return [item['image'] for item in self.imgPairs]
    
    
    def _list_files(self):
        #EXTS = ['.png', '.bmp', '.gif', '.jpg', '.jpeg']
        img_files = os.listdir(self.imgs_dir)
        msk_files = os.listdir(self.msks_dir) if os.path.exists(self.msks_dir) else [] 
                
        def get_bname_exts(parent_dir, filenames):
            bnames, exts = [], []
            for fn in filenames:
                if os.path.isfile(os.path.join(parent_dir, fn)):
                    fname, ext = os.path.splitext(fn)
                    bnames.append(fname)
                    exts.append(ext)
            return bnames, exts

        msk_bnames, msk_extnames = get_bname_exts(self.msks_dir, msk_files)
                   
        # extract image, mask and aux data file pairs        
        imgpaths, mskpaths = [], []
        for imgf in img_files:
            path_img = os.path.join(self.imgs_dir, imgf)
            if not os.path.isfile(path_img):
                continue
            
            fname, ext = os.path.splitext(imgf)
            
            # if finding a matched mask file in msk_names
            path_msk = ''
            if fname in msk_bnames:
                idx = msk_bnames.index(fname)
                path_msk = os.path.join(self.msks_dir, fname + msk_extnames[idx])                
                
            imgpaths.append(path_img)
            mskpaths.append(path_msk)                        
                    
        #make image pairs list
        imgPairs = [{'image':fp1, 'mask':fp2} 
                    for fp1, fp2 in zip(imgpaths, mskpaths)]    
        
        return imgPairs                                  



# check data integrity
def check_data(root_dir):
    subdirs = ['train', 'val']
    for sd in subdirs:
        img_dir = os.path.join(root_dir, sd, 'images') 
        msk_dir = os.path.join(root_dir, sd, 'masks') 
    
        if not os.path.exists(img_dir):
            print('ERROR: '+img_dir+' does not exist.')
            return False
        if not os.path.exists(msk_dir):
            print('ERROR: '+msk_dir+' does not exist.')
            return False

    return True    

    
if __name__ == '__main__':
    import matplotlib.pylab as plt

    img_dir = 'D:\\dlwater\\train_data\\wat_nj_nirg256\\train\\images'     
    #img_dir = 'D:\\GeoData\\DLData\\AerialImages'      
    ds = SegDataset(img_dir, 'train', n_classes=1, imgW=256, imgH=256, 
                    apply_aug=False, sub_size=1.0)        
    for i in range(5):        
        # get image and mask tensor
        samp = ds[i]        
        img, msk = samp 
        
        oimg, omsk = ds.get_image_and_mask(i)
        
        print('original image: ', oimg.shape, omsk.shape)
        print('transformed image: ', img.shape, msk.shape)
        
        plt.figure()
        plt.subplot(221)        
        plt.imshow(oimg)        
        plt.subplot(222)        
        plt.imshow(omsk)                
        plt.subplot(223)        
        plt.imshow(img[0])        
        plt.subplot(224)        
        plt.imshow(msk[0])             
        plt.show()

