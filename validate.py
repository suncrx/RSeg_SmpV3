# USAGE
# python validate.py --model_file 
# "D:/GeoData/DLData/saltern/10bands/out/mxsegnet_resnet34_best.pt" 
# --img_dir "D:/GeoData/DLData/saltern/10bands/val/images" --conf 0.5 --img_sz 256
# --plot True
#

# import the necessary packages
# %% import installed packages
import os
import sys
import argparse
import shutil
from pathlib import Path
import tqdm
import cv2
import numpy as np
#import matplotlib.pylab as plt
import torch
import torch.nn.functional as F

from segmentation_models_pytorch import utils as smp_utils

import models
from datasetx import SegDatasetX
from misc.plot import plot_prediction


#%% parse arguments from command line
def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_file', type=str,
                        #default = 'D:/dlwater/train_data/wat_hr/out/trained_models/unet_resnet50/unet_resnet50_best.pt',
                        default = 'D:/dlwater/train_data/wat_nj_nirg256/out/trained_models/mxsegnet_resnet50/mxsegnet_resnet50_best.pt',
                        help='model filepath')

    parser.add_argument('--img_dir', type=str,
                        default='D:/dlwater/train_data/wat_nj_nirg256/val/images',
                        help='root directory where the test folder resides.')

    parser.add_argument('--img_sz', type=int,
                        default=512, help='input image size (pixels)')

    #parser.add_argument('--out_dir', type=str, default=ROOT / 'out', 
    #                    help='training output path')    

    parser.add_argument('--conf', type=float, default=0.5,
                        help = 'confidence')
    
    parser.add_argument('--smooth_mode', type=int, default=0,
                        help = 'apply thresholding when binary segmentation or not')

    parser.add_argument('--plot', type=int, default=1,
                        help='plot the results or not')
    
    parser.add_argument('--verbose', type=int, default=0,
                        help='show verbose information or not')
    
    return parser.parse_args()


#%% get current directory
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # FasterRCNN root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# determine the device to be used for training and evaluation
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print('Device : ', DEV)


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
def make_prediction(model, image, out_H, out_W, binary=False, conf=0.5,
                    smooth_mode=False):
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
            conf_map = cv2.resize(pred.numpy(), (out_W, out_H), 
                              interpolation=cv2.INTER_LINEAR)
            # apply thresholding when smooth mode is False
            #if not smooth_mode:
            pred = np.uint8(conf_map >= conf)
        else:
            # determine the class by the index with the maximum
            pred = np.uint8(torch.argmax(pred, dim=0))
            # resize to the original size
            pred = cv2.resize(pred, (out_W, out_H),
                              interpolation=cv2.INTER_NEAREST)
            # print('Found classes: ', np.unique(pred))
            conf_map = pred
    return pred, conf_map


# calculate the evaluation metrics between the predicted mask and ground-truth
def eval_metrics(predMask, gtMask, n_classes):
    predMask = np.int32(predMask)
    gtMask = np.int32(gtMask)
    if n_classes > 1:
        pm = [predMask == v for v in range(n_classes)]
        gm = [gtMask == v for v in range(n_classes)]
        pm, gm = np.array(pm), np.array(gm)
    else:
        pm, gm = np.uint8(predMask > 0), np.uint8(gtMask > 0)
        # iou = smp.utils.metrics.IoU()  #for smp 0.2.1
    iou = smp_utils.metrics.IoU()  #for smp >= 0.3.2
    acc = smp_utils.metrics.Accuracy()
    pre = smp_utils.metrics.Precision()
    rec = smp_utils.metrics.Recall()
    fsc = smp_utils.metrics.Fscore()

    iouv = iou.forward(torch.as_tensor(pm), torch.as_tensor(gm))
    accv = acc.forward(torch.as_tensor(pm), torch.as_tensor(gm))
    prev = pre.forward(torch.as_tensor(pm), torch.as_tensor(gm))
    recv = rec.forward(torch.as_tensor(pm), torch.as_tensor(gm))
    fscv = fsc.forward(torch.as_tensor(pm), torch.as_tensor(gm))

    #return iouv
    return {'iou': iouv, 'acc': accv, 'pre': prev, 'rec': recv, 'fsc': fscv}

def load_val_ids(fpath):
    # read validation file ids        
    if os.path.exists(fpath):
        print('loaded val file name ids.')
        with open(fpath) as fo:
            lines = fo.readlines()
            val_bnames = [ss[0:-1] for ss in lines]
    else:
        val_bnames = []
    return val_bnames

    
def run(opt):
    #print(opt)
    # get parameters
    model_file = opt.model_file
    img_dir = opt.img_dir
    #out_dir = opt.out_dir
    img_sz = opt.img_sz
    conf, plot = opt.conf, opt.plot
    verb = opt.verbose
    if not os.path.exists(model_file):
        raise Exception(f"Can not find model path: {model_file}")
        
    # make output folders
    model_basename = os.path.basename(model_file)
    # make output folder for predicting images
    data_dir = os.path.dirname(os.path.dirname(img_dir))
    outpred_dir = os.path.join(data_dir, 'out', 'val_pred', model_basename)
    if os.path.exists(outpred_dir):
        shutil.rmtree(outpred_dir)
    os.makedirs(outpred_dir)

    #--------------------------------------------------------------------    
    # load our model from disk and flash it to the current device
    print(f'Loading model: {model_file}')
    (model, model_name,
     n_classes, class_names,
     in_channels, model_img_sz) = models.utils.load_seg_model(model_file, device=DEV)
    if model_img_sz is not None:
        pass
        #img_sz = model_img_sz
        #print('NOTE: The parameter img_sz is replaced with the value %d from trained model file' % model_img_sz)

    #---------------------------------------------------------------------------
    # load the image paths in our testing directory and
    # randomly select 10 image paths
    print("Loading validation image ...")    
    dset = SegDatasetX(img_dir, mode="val",  n_classes=n_classes,
                        imgH=img_sz,  imgW=img_sz,
                        apply_aug=False)

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    nm = len(dset)
    #nm = min(len(testDS), 10)
    metrics = []
    imgnames = []
    val_bnames = load_val_ids(os.path.join(os.path.dirname(img_dir), 'val_ids2.txt'))
    print('Predicting ...')
    for i in tqdm.tqdm(range(nm)):
        samp = dset[i]
        #get the image filepath
        imgPath = samp['image_path']
        bname = os.path.splitext(os.path.basename(imgPath))[0]
        if len(val_bnames)>0 and not bname in val_bnames:
            continue

        #get original image and mask
        ori_img, ori_gtMask = samp['ori_image'], samp['ori_mask']
                
        #get preprocessed image and mask
        img = samp['image'] #, samp['mask']
        
        # make predictions and visualize the results
        if verb:
            print('Predicting ' + imgPath)
        #for binary segmentation, pred is a uint8-type mask with 0, 1;
        #for multi-class segmentation, pred is a uint8-type mask with
        #class labels: 0, 1, 2, 3, ... , n_class-1
        is_binary = (n_classes < 2)
        outH, outW = ori_img.shape[0:2]
        pred, conf_map = make_prediction(model, img, outH, outW,
                               binary=is_binary, conf=conf,
                               smooth_mode = True)

        #evaluation
        #iouv = Cal_IoU(pred, ori_gtMask, n_classes=n_classes)    
        res = eval_metrics(pred, ori_gtMask, n_classes=n_classes)
        iouv, accv, prev = res['iou'], res['acc'], res['pre']
        recv, fscv = res['rec'], res['fsc']
        metrics.append([iouv, accv, prev, recv, fscv])
        if verb:
            print(f'IoU: {iouv:.3f} Acc: {accv:.3f} Prec: {prev:.3f} Rec: {recv:.3f} Fscore: {fscv:.3f}')
        image_basename = os.path.basename(imgPath)
        imgnames.append(image_basename)

        #------------------------------------
        #save and convert results to rgb label for visualization 
        image_basename = os.path.basename(imgPath)
        bname, ext = os.path.splitext(image_basename)
        out_mskPath = os.path.join(outpred_dir, bname + '_msk.png')
        fig_path = os.path.join(outpred_dir, 'plot_' + bname + '.png')
        stitle = f'{image_basename} IoU {iouv:.3f}' 
        #save image
        bgrimg = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(outpred_dir, image_basename), bgrimg)
        if is_binary:
            #save predicted mask (one channel)    
            Mask = np.uint8(pred * 255)
            cv2.imwrite(out_mskPath, Mask)
            #save confidence image
            conf_img = np.uint8(conf_map*255)
            cv2.imwrite(os.path.join(outpred_dir, bname + '_conf.jpg'), conf_img)
            #save gt mask
            if ori_gtMask.max() == 1:
                ori_gtMask = ori_gtMask * 255
            cv2.imwrite(os.path.join(outpred_dir, bname + '_gt.png'), ori_gtMask)
            #plot results
            if plot:
                plot_prediction(ori_img, ori_gtMask, Mask,
                                sup_title=stitle, save_path=fig_path,
                                auto_close=True)
        else:
            #save predicted mask
            Mask = np.uint8(pred)
            cv2.imwrite(out_mskPath, Mask)
            Mask_rgb = Mask
            out_rgbMskPath = os.path.join(outpred_dir, bname + '_rgb.png')
            cv2.imwrite(out_rgbMskPath, Mask_rgb)
            #save gt mask            
            ori_gtMask_rgb = ori_gtMask
            cv2.imwrite(os.path.join(outpred_dir, bname + '_gt.png'), ori_gtMask_rgb)
            #plot results
            if plot:
                plot_prediction(ori_img, ori_gtMask_rgb, Mask_rgb,
                                sup_title=stitle, save_path=fig_path,
                                auto_close=True)

    #Evaluation metrics----------------------------------------------------- 
    def output_statistic(maxv, minv, meanv, fo=None):
        print('\nname,  Iou,  Accuracy,  Precision, Recall, Fscore', file=fo)
        print('Max,  %.3f, %.3f, %.3f, %.3f, %.3f' %
              (maxv[0], maxv[1], maxv[2], maxv[3], maxv[4]), file=fo)
        print('Min,  %.3f, %.3f, %.3f, %.3f, %.3f' %
              (minv[0], minv[1], minv[2], minv[3], minv[4]), file=fo)
        print('Mean,  %.3f, %.3f, %.3f, %.3f, %.3f' %
              (meanv[0], meanv[1], meanv[2], meanv[3], meanv[4]), file=fo)
        
    Mm = np.array(metrics)
    maxv, minv, meanv = Mm.max(axis=0), Mm.min(axis=0), Mm.mean(axis=0)
    output_statistic(maxv, minv, meanv)   
    print('Done!')
    print('Results saved: %s' % outpred_dir)

    #write metrics to log file
    logfn = os.path.join(outpred_dir, model_basename + '_log.txt')
    with open(logfn, 'w') as fo:
        output_statistic(maxv, minv, meanv, fo=fo)
        for i in range(Mm.shape[0]):
            print('%s, %.3f, %.3f, %.3f, %.3f, %.3f' %
                  (imgnames[i], Mm[i][0], Mm[i][1], Mm[i][2], Mm[i][3], Mm[i][4]),
                  file=fo)


if __name__ == '__main__':
    opt = parse_opt()
    run(opt)
