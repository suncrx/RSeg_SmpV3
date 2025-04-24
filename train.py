# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 13:38:14 2023

@author: renxi
"""

#!pip install segmentation-models-pytorch

#usage example:
# python train.py --data ./data/waters.yaml -checkpoint_file ./checkpoint.chk 
# --out_dir './output' --arct unet --encoder resnet34
# --imgsz 512 --epochs 2 --batch_size 4 --lr 0.001 
# --momentum 0.9 --loss dice --resume 1   

# %% import installed packages
import os
import sys
import time
import yaml
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

#import segmentation_models_pytorch as smp
# explictly import utils if segmentation_models_pytorch >= 0.3.2
from segmentation_models_pytorch import utils as smp_utils

import nets
import nets.utils
from nets.loss import RMILoss, FCM_label, BoundaryLoss, SegBCELoss, FocalLoss


from datasets.datasetx import SegDatasetX
from misc.common import log_csv, plot_train_val_info
from epoch_run import MyTrainEpoch, MyValidEpoch

#%% get current directory
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  #  root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# determine the device to be used for training and evaluation
DEV = "cuda" if torch.cuda.is_available() else "cpu"
#DEV = "cpu"
print('Device: ', DEV)
# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEV == "cuda" else False


#%% parse arguments from command line
def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str,
                        #default=ROOT / 'data/airimages.yaml',
                        #default=ROOT/'data/buildings.yaml', 
                        default=ROOT / 'data/waters_nj_nirg256.yaml' ,
                        #default=ROOT / 'data/waters.yaml',
                        help='dataset yaml path')

    parser.add_argument('--checkpoint_file', type=str,                        
                        #default="D:/GeoData/DLData/Vehicle/Vehicle_seg/out/unet_resnet50.ckp",
                        #default = 'D:/dlwater/train_data/wat_nj_rgb/out/trained_models/unet_resnet50/unet_resnet50.ckp',
                        default = '',
                        help='checkpoint file path from which the model is trained')

    parser.add_argument('--img_sz', '--img', '--img-size', type=int,
                        default=256, help='train, val image size (pixels)')

    parser.add_argument('--out_dir', type=str, default='',
                        help='training output path')

    parser.add_argument('--arct', type=str,
                        default='unet',
                        help='model architecture (options: unet, unetplusplus, \
                        manet, linknet, fpn, pspnet, deeplabv3, deeplabv3plus, \
                        pan, unet_scse, unet_se, unet_cbam, \
                        munet, munet_ag, munet_cbam, munet2, mxsegnet')

    parser.add_argument('--encoder', type=str, default='timm-resnest50d',
                        help='encoder for the net (options: resnet18, \
                            resnet34, resnet50, vgg16, vgg19, mobilenet_v2')

    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8,
                        help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--weight_decay', type=float, default=0.005, help='weight decay')

    parser.add_argument('--loss', type=str, default='dice', help='loss function: \
                        ["dice","jacard","focal","rmi"]')

    parser.add_argument('--aug', type=int, default=0,
                        help='Apply data augmentation or not')
                        
    parser.add_argument('--resume', type=int, default=0,
                        help='continue the training from the saving check point or not')                      

    parser.add_argument('--sub_size', type=float, default=1.0,
                        help='subsize of training data')

    parser.add_argument('--save_period', type=int, default=5,
                        help='Check point saving period')

    return parser.parse_args()


#%% run training
def run(opt):
    #%% parameters
    print('Parameters: ')
    print(opt, '\n')
    data_yaml_file = opt.data
    arct, encoder = opt.arct, opt.encoder
    img_sz = opt.img_sz
    batch_size, epochs = opt.batch_size, opt.epochs
    lr, momentum, w_decay = opt.lr, opt.momentum, opt.weight_decay
    loss = opt.loss
                
    #checkpoint = opt.checkpoint
    checkpoint_file = opt.checkpoint_file
    save_period = opt.save_period
    resume_mode = opt.resume
    print('Resuming training ? ', resume_mode)

    out_dir = opt.out_dir
    #applying data augumentation or not
    bAug = opt.aug
    
    # read data information from yaml file          
    assert os.path.exists(data_yaml_file)
    with open(data_yaml_file) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    # cfg is a dictionary
    if 'exp_name' in cfg.keys():
        print(cfg['exp_name'])
    print('Data information:', cfg)
    n_classes = cfg['nclasses']
    bands = cfg['bands']
    class_names = cfg['names']
    root_data_dir = cfg['path']
    
    train_img_dir = os.path.join(root_data_dir, cfg['train'])
    val_img_dir = os.path.join(root_data_dir, cfg['val'])

    # make an output dir in the root_data_dir if it is not specified.
    if out_dir == '':
        out_dir = os.path.join(root_data_dir, 'out')
    os.makedirs(out_dir, exist_ok=True)
    print('\nOutput directory: ', out_dir)

    # %% prepare datasets
    # init train, val, test sets
    print('\nPreparing data ...')    
    # Dataset with rasterio (can read multiband images)
    train_dataset = SegDatasetX(train_img_dir, "train", n_classes=n_classes, 
                                imgH=img_sz, imgW=img_sz, channel_indice=bands,
                                apply_aug=bAug)
    val_dataset = SegDatasetX(val_img_dir, "val", n_classes=n_classes, 
                              imgH=img_sz, imgW=img_sz, channel_indice=bands)
    '''
    # Dataset with OpenCV (read only RGB images)
    train_dataset = SegDataset(train_img_dir, "train",
                               n_classes=n_classes, 
                               imgH=img_sz, imgW=img_sz,
                               apply_aug=bAug, sub_size=sub_size)
    val_dataset = SegDataset(val_img_dir, "val",
                             n_classes=n_classes, 
                             imgH=img_sz, imgW=img_sz)
    '''
    
    # It is a good practice to check datasets don't intersect with each other
    train_imgs = train_dataset.get_image_filepaths()
    val_imgs = val_dataset.get_image_filepaths()
    assert set(val_imgs).isdisjoint(set(train_imgs))
    print(f"Train size: {len(train_dataset)}")
    print(f"Valid size: {len(val_dataset)}")
    #print(f"Test size: {len(test_dataset)}")

    n_cpu = 0
    #n_cpu = os.cpu_count()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=n_cpu)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=n_cpu)

    #%% initialize model
    n_channels = len(bands)
    model, model_name = nets.utils.create_model(arct=arct,
                                                  encoder=encoder,
                                                  n_classes=n_classes,
                                                  in_channels=n_channels)
    if model is None:
        print("ERROR: cannot create a model named '%s'" % model_name)
        sys.exit(0)

    # initialize variants
    start_epoch = 0
    best_score = -np.inf
    train_losses, val_losses = [], []
    train_scores, val_scores = [], []

    # optimizer
    opti_name = 'adamw'
    opti = optim.AdamW(model.parameters(), lr=lr, betas=[0.9, 0.999],
                       eps=1e-7, amsgrad=False)   
    print('Optimizer: ')
    print(opti)

    # load check point file if a checkpoint file is specified or found
    # in the output folder.
    if checkpoint_file!='' and os.path.exists(checkpoint_file):
        chk = torch.load(checkpoint_file, map_location=DEV)

        n_classes = chk['n_classes']
        n_channels = chk['n_channels']
        class_names = chk['class_names']
        if 'img_sz' in chk.keys():
            img_sz = chk['img_sz']

        #batch_size = chk[]
        arct = chk['arct']
        encoder = chk['encoder']

        # load model from checkpoint
        model.load_state_dict(chk['model_state_dict'])

        # load optimizer from checkpoint
        opti.load_state_dict(chk['opti_state_dict'])
        #!!! transfer optimizer to DEV
        for state in opti.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(DEV)
        
        # get variants from check point when resuming the training
        if resume_mode:
            print('Resume training from the check point.')
            start_epoch = chk['epochs']
            train_losses = chk['train_losses']
            val_losses = chk['val_losses']
            train_scores = chk['train_scores']
            val_scores = chk['val_scores']    
            best_score = np.array(val_scores).max()
        # if resume is False, set checkpoint file path to the out_dir     
        else:
            checkpoint_file = os.path.join(out_dir, model_name+'.ckp')

        print('\nModel loaded from check point: %s' % checkpoint_file)
        print('Model name: %s' % model_name)
        print('trained epochs: %d' % start_epoch)
    else:
        checkpoint_file = os.path.join(out_dir, model_name+'.ckp')
        print('No checkpoint found, training a new model ...')
        print('Model name: %s' % model_name)

    #-----------------------------------------------------------------------       
    # loss function    
    if loss.lower() == 'rmi':
        lossFunc = RMILoss(with_logits=False)
    elif loss.lower() == 'fcm':
        lossFunc = FCM_label()
    elif loss.lower() == 'boundary':
        lossFunc = BoundaryLoss()
    elif loss.lower() == 'bce':
        lossFunc = SegBCELoss()
    elif loss.lower() == 'focal':
        lossFunc = FocalLoss()
    elif loss.lower() == 'ce':
        lossFunc = torch.nn.CrossEntropyLoss()
    elif loss.lower() == 'jaccard':
        lossFunc = smp_utils.losses.JaccardLoss()
        # default is dice-loss
    else:
        lossFunc = smp_utils.losses.DiceLoss()  
    print('Loss function: ', lossFunc.__name__)

    # metrics
    metrics = [smp_utils.metrics.IoU(threshold=0.5)]

    # learning rate scheduler
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    #%% training
    print("\nTraining network: %s ..." % model_name)

    train_epoch = MyTrainEpoch(model, loss=lossFunc,  metrics=metrics,
                               optimizer=opti, device=DEV, verbose=True)

    val_epoch = MyValidEpoch(model, loss=lossFunc, metrics=metrics,
                             device=DEV, verbose=True)

    startTime = time.time()
    best_epoch = start_epoch
    loss_name = lossFunc.__name__
    for i in range(start_epoch, start_epoch + epochs):
        time_se = time.time()
        print('\nEpoch: %d/%d' % (i + 1, start_epoch + epochs))
        train_logs = train_epoch.run(train_dataloader)
        val_logs = val_epoch.run(val_dataloader)

        train_losses.append(train_logs[loss_name])
        val_losses.append(val_logs[loss_name])

        tsc = train_logs['iou_score']
        vsc = val_logs['iou_score']
        train_scores.append(tsc)
        val_scores.append(vsc)
        
        #save losses and scores to log file
        log_csv(train_losses, val_losses, train_scores, val_scores,
                os.path.join(out_dir, model_name + '_log.csv'))       

        #plot train and val information
        plot_train_val_info([train_losses, val_losses], ['train_loss','val_loss'],
                            xlab='epoch', ylab='loss', 
                            title=f'train and val loss - {model_name}', 
                            save_path=os.path.join(out_dir, model_name + '_loss.png'))
        plot_train_val_info([train_scores, val_scores], ['train_IoU','val_IoU'],
                            xlab='epoch', ylab='IoU', 
                            title=f'train and val IoU - {model_name}', 
                            save_path=os.path.join(out_dir, model_name + '_IoU.png'))
        
        endTime = time.time()
        print("Elapsed time : {:.3f}s".format(endTime - time_se))
        # save the model that obtains the best score 
        if vsc > best_score:
            best_score = vsc
            best_epoch = i + 1
            bestmdl_path = os.path.join(out_dir, model_name + '_best.pt')
            nets.utils.save_seg_model(bestmdl_path, model, arct, encoder,
                                        n_classes, class_names, n_channels)
            print('Best model saved: %s' % bestmdl_path)

        # save the check point 
        if (i % save_period == 0):
            nets.utils.save_checkpoint(checkpoint_file, model, arct, encoder,
                                         opti_name, opti,
                                         n_classes, class_names,
                                         n_channels, img_sz,
                                         len(train_scores), batch_size,
                                         lr, momentum, w_decay,
                                         train_losses, val_losses, train_scores, val_scores)
            print('check point saved: %s' % checkpoint_file)

    endTime = time.time()
    print("\nTotal time : {:.2f}s".format(endTime - startTime))
    print('Best model at epoch: %d' % best_epoch)
    print('Best score: %.3f' % best_score)

    #save the last check point
    nets.utils.save_checkpoint(checkpoint_file, model, arct, encoder,
                                 opti_name, opti,
                                 n_classes, class_names,
                                 n_channels, img_sz,
                                 len(train_scores), batch_size,
                                 lr, momentum, w_decay,
                                 train_losses, val_losses, train_scores, val_scores)
    print('The last check point saved: %s' % checkpoint_file)
    print('Finished')


#%% calling
if __name__ == '__main__':
    opt = parse_opt()
    #print(opt)
    run(opt)
