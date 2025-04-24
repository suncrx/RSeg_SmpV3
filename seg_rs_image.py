# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 21:40:58 2024

@author: renxi


This script segments large RS images using rasterio and the trained 
segmentation models.

"""

import os
import argparse

import numpy as np
import cv2
import rasterio
from rasterio.windows import Window

import torch
import torch.nn.functional as F

from ultralytics import YOLO

import models


# determine the device to be used for training and evaluation
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print('Device : ', DEV)



#%% parse arguments from command line
def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_file', type=str,                        
                        #default = "D:/dlwater/train_data/wat_nj_rgb/out/trained_models/unet_resnet50/unet_resnet50_best.pt",
                        required=True,
                        help='model filepath')

    parser.add_argument('--image', type=str,
                        #default = "D:/dlwater/test_data/st2_nj_small/S2B_MSIL2A_20210406T024549_N0300_R132_T50SPA_20210406T054125_RGB.jpg",
                        required=True,                        
                        help='test image directory')

    parser.add_argument('--tile_sz', type=int,
                        default=256, help='tile size (pixels)')
    
    parser.add_argument('--is_yolo', type=int, default=0,
                        help='the model is yolo or not')

    parser.add_argument('--out_filepath', type=str, 
                        default = '',
                        help='output filepath')    

    parser.add_argument('--conf', type=float, default=0.5,
                        help='test confidence')

    parser.add_argument('--save_tiles', type=int, default=1,
                        help='save tiles or not')
    
    return parser.parse_args()


#======================================================================
# padding image to the size of 32*X
# The shape of input image should be (Channel, Height, Width), e.g. 3*213*2455
def pad_image_32x(image):
    c, h, w = image.shape
    if 32 * int(w / 32) != w:
        padx = 32 * int(w / 32 + 1) - w
        image = F.pad(image, (0, padx, 0, 0, 0, 0))
    if 32 * int(h / 32) != h:
        pady = 32 * int(h / 32 + 1) - h
        image = F.pad(image, (0, 0, 0, pady, 0, 0))
    return image


#======================================================================
'''
model: trained model object
image: input image, which should be in tensor format ( values 0 ~ 1.0)
out_H, out_W: output confidence map should be resized into size (outH, out_W)
binary: binary segmentation or not
conf: 0.5d efault 
'''
def make_prediction(model, image, out_H, out_W, binary=False, conf=0.5):
    # set model to evaluation mode
    model.eval()
    # turn off gradient tracking
    with torch.no_grad():
        #padding image size to 32*M
        c, h, w = image.shape
        image = pad_image_32x(image)

        # apply image transformation. This step turns the image into a tensor.
        # with the shape (1, 3, H, W). See IMG_TRANS in dataset.py
        image = torch.unsqueeze(image, 0)
        # make the prediction, pass the results through the sigmoid
        # function, and convert the result to a NumPy array
        pred = model.forward(image).squeeze()

        # crop prediction size to the original size
        pred = pred[0:h, 0:w]

        # Sigmoid or softmax has been performed in the net
        if binary:
            pred = cv2.resize(pred.numpy(), (out_W, out_H))
            pred = np.uint8(pred >= conf)
            #pred = np.uint8(pred*255)            
        else:
            # determine the class by the index with the maximum
            pred = np.uint8(torch.argmax(pred, dim=0))
            # resize to the original size
            pred = cv2.resize(pred, (out_W, out_H),
                              interpolation=cv2.INTER_NEAREST)
            # print('Found classes: ', np.unique(pred))
    return pred



'''
image: input image (RGB, uint8)
'''
def make_prediction_yolo(model, image, out_H, out_W, conf=0.5):
        results = model.predict(source=image, verbose=False)
        #annotator = Annotator(image, line_width=2)
        result = results[0]       
        
        maskimg = np.zeros((out_H, out_W), dtype=np.uint8)
        
        if result.masks is not None:
            clss = result.boxes.cls.cpu().tolist()
            masks = result.masks
            # Each mask is an object that has a set of properties.
            # 1)data - the segmentation mask of the object, which is a black 
            #   and white image matrix, in which 0 elements are black pixels and 
            #   1 elements are white pixels.
            # 2)xy - the polygon of object, which is an array of points. 
            for mask, cls in zip(masks, clss):
                #get binary mask image for each object
                msk = mask.data[0].cpu().numpy().astype(np.uint8)*255
                # conbine maskes by union operation 
                maskimg = maskimg | cv2.resize(msk, (out_W, out_H), 
                                                 interpolation=cv2.INTER_NEAREST)               
                #get polygon for each object
                #poly = mask.xy[0]                
                #color = colors(int(cls), True)            
                #txt_color = annotator.get_txt_color(color)                
                #annotator.seg_bbox(mask=poly, 
                #                   mask_color=color, 
                #                   label=names[int(cls)])#, txt_color=txt_color)                
 
        return maskimg



'''
img_path: input image path
out_img_path: output image path
model_file: trained model file path
tile_size: a big image is predicted tile by tile, and the tile_size specifies
           the width and height of the tile.
conf: 0.5 default
save_tile: if True, predicted tiles are saved in a subfolder.           
'''
def run(img_path, model_file, 
        is_yolo=False,
        out_img_path='', 
        tile_size=256,  conf=0.5, save_tiles=True):
    #--------------------------------------------------------------------    
    if not os.path.exists(model_file):
        raise Exception('Can not find model path: %s' % model_file)
    # load our model from disk and flash it to the current device
    print("Loading model: %s" % model_file)
    if (is_yolo):
        model = YOLO(model_file)
        names = model.model.names        
        print(names)    
        # our trained yolo model is a binary segmentation model 
        n_classes = 2
    else:    
        (model, model_name,
         n_classes, class_names,
         in_channels, model_img_sz) = models.utils.load_seg_model(model_file)
     
    # generate output file path    
    if out_img_path == '':    
        model_name = os.path.basename(model_file)
        #parent_dir = os.path.abspath(img_path)    
        fname = os.path.basename(img_path) 
        out_img_path = os.path.splitext(img_path)[0] + '_' + model_name + '.png'
        
    if save_tiles:
        out_dir = os.path.join(os.path.dirname(out_img_path), 'tile_imgs')
        out_dir_msk = os.path.join(os.path.dirname(out_img_path), 'tile_msks')
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(out_dir_msk, exist_ok=True)        


    with rasterio.open(img_path) as src:
        print('Image information: ')
        print('Image name:', src.name)
        print('Image size (Width x Height):', src.width, src.height)
        print('Number of bands:', src.count)
        print('Band indexes:', src.indexes)
        print('Data types:', src.dtypes)
        print('Bounds:', src.bounds)
        print('Affine transform:\n', src.transform)
        print('CRS:\n', src.crs)
        
        # open mask image object (png format)
        with rasterio.open(out_img_path, 'w', driver='png',
                    width=src.width, height=src.height,
                    count = 1,
                    dtype = np.uint8,
                    description = src.descriptions,                           
                    transform = src.transform,                   
                    crs = src.crs,
                    #interleave = 'bip',     #bip, bsq
                    ) as dst:
            
            rows = src.height//tile_size 
            if rows * tile_size < src.height: rows = rows + 1  
            
            cols = src.width//tile_size 
            if cols * tile_size < src.width: cols = cols + 1  
            
            for i in range(rows):
                #print('Tile row: ', i)
                print('Percent: %d%%' % int(100*float(i+1)/rows))
                offy = i*tile_size
                height = tile_size if offy+tile_size<src.height else src.height-offy                
                for j in range(cols):
                    offx = j*tile_size
                    width = tile_size if offx+tile_size<src.width else src.width-offx 

                    # read image tile
                    subw = Window(offx, offy, width, height)                
                    
                    subimg = src.read(src.indexes, window=subw)
                    #print('shape of subset data: ')
                    #print(subimg.shape)
                    # image size (C, H, W)
                    nbands, h, w = subimg.shape
                    if (h==0 or w==0): 
                        continue           
                    
                    is_binary = (n_classes < 2)                                        
                    
                    if is_yolo:
                        #convert (C,W,H) to (H,W,C)
                        subimg2 = np.transpose(subimg,[1,2,0])
                        # feed the image tile to yolo
                        pred = make_prediction_yolo(model, subimg2, h, w,
                                                    conf=conf)
                        #pred = np.zeros((h,w))
                        msk = pred
                    else:
                        # convert the image tile to tensor and predict it 
                        inp_img = torch.Tensor(subimg / 255.0)                    
                        pred = make_prediction(model, inp_img, h, w, 
                                               binary=is_binary, conf=conf)                                            
                        # convert the results into mask image
                        if is_binary:
                            msk = np.uint8(pred * 255)
                        else:
                            print('Not implemented')
                    
                    # write the result tile                    
                    dst.write(np.expand_dims(msk, axis=0), window=subw)
                                                     
                    if save_tiles:
                        fname = 'r%03d_c%03d.jpg' % (i, j) 
                        fname_msk = 'r%03d_c%03d.png' % (i, j)
                        
                        #save image tile
                        fp = os.path.join(out_dir, fname)
                        cv_img = cv2.cvtColor(np.transpose(subimg)[:,:,0:3], cv2.COLOR_RGB2BGR)
                        cv2.imwrite(fp, cv_img)
                        
                        #save mask tile
                        fp_msk = os.path.join(out_dir_msk, fname_msk)                        
                        cv2.imwrite(fp_msk, msk)




if __name__ == '__main__':
    opt = parse_opt()
    print(opt)
    run(opt.image, opt.model_file,
        out_img_path = opt.out_filepath,
        is_yolo=opt.is_yolo, 
        tile_size=opt.tile_sz, 
        conf=opt.conf, 
        save_tiles=opt.save_tiles)
    
    
'''
if __name__ =='__main__':
    # image (RGB)
    img_path = "D:/Work/DLearn/DLWater/st2_nj_small/S2B_MSIL2A_20210406T024549_N0300_R132_T50SPA_20210406T054125_RGB.jpg"
    img_path = "D:/Work/DLearn/DLWater/st2_nj_apr/S2B_MSIL2A_20210406T024549_N0300_R132_T50SPA_20210406T054125_RGB.jpg"
    img_path = "D:/Work/DLearn/DLWater/st2_nj_feb/S2A_MSIL2A_20210220T024731_N0214_R132_T50SPA_20210220T055831_RGB.jpg"
    img_path = "D:/Work/DLearn/DLWater/st2_nj_jul/S2A_MSIL2A_20210730T024551_N0301_R132_T50SPA_20210730T045020_RGB.jpg"
    img_path = "D:/Work/DLearn/DLWater/st2_nj_oct/S2B_MSIL2A_20211023T024749_N0301_R132_T50SPA_20211023T054927_RGB.jpg"
    
    is_yolo = False
    if is_yolo:    
        # yolov8 model
        model = "D:/Work/DLearn/DLWater/trained_models/yolov8l_seg/weights/best.pt"        
        model_name = 'yolov8l_seg'
    else:
        # unet model
        #model = "D:/Work/DLearn/DLWater/trained_models/unet_resnet50/unet_resnet50_best.pt"
        model = "D:/GeoData/DLData/Waters/wat_nj/out/trained_models/unet_resnet50/unet_resnet50_best.pt"
        
        model_name = os.path.basename(model)

    parent_dir = os.path.abspath(img_path)    
    bfname = os.path.basename(img_path) 
    
    out_img_path = os.path.splitext(img_path)[0] + '_' + model_name + '.png'
    
    print('input file: ', img_path)
    print('output file: ', out_img_path)
       
    run(img_path, out_img_path, model, 
              is_yolo=is_yolo, 
              tile_size=256, 
              conf=0.5)
    
    print('Finished!')
'''