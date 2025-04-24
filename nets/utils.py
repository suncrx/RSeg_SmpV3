# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 19:06:47 2023

@author: renxi

The following is a list of supported encoders in the SMP. Select the 
appropriate family of encoders and click to expand the table and select a 
specific encoder and its pre-trained weights 
(encoder_name and encoder_weights parameters).

ResNet
ResNeXt
ResNeSt
Res2Ne(X)t
RegNet(x/y)
GERNet
SE-Net
SK-ResNe(X)t
DenseNet
Inception
EfficientNet
MobileNet
DPN
VGG
Mix Vision Transformer
MobileOne


"""
# import the necessary packages
import torch
import segmentation_models_pytorch as smp

from .munet  import MUNet
from .munet_ag  import MUNet_AG
from .munet_cbam  import MUNet_CBAM

#from .munet2 import MUNet2
from .munetx import MUNetX

from .unet_att import Unet_Attention
from .mxsegnet import MXSegNet

########################################################################
#options of arct:
#'unet': standard unet
#'unet_mini':   mini unet (with shallow layers)
#'unet_scse':   unet with scse attention  
#'unet_se':     unet with SE attention
#'unet_sk':     unet with SK attention
#'unet_cbam':   unet with CBAM attention
#'unet_bam':    unet with BAM attention                
#'unet_eca':    unet with ECA attention

#'unetplusplus':  unet++
#'deeplabv3':     deeplabv3
#'deeplabv3plus': deeplabv3++ 
#'linknet':       
#'pan':             
#'manet':
#'pspnet':
#'fpn':
    
#------------------------
#'munet':         
#'munet_ag':
#'munet_cbam':                    
#-----------------------
#'mxsegnet': my segmentation model
#-----------------------------------------------------------------------------


# options of encoders: https://segmentation-models-pytorch.readthedocs.io/en/latest/encoders.html
def create_model(arct='unet', encoder='resnet34',
                 n_classes=1, in_channels=3):

    m_fullname = arct + '_' + encoder     
    
    activation_name = 'softmax' if n_classes>1 else "sigmoid"
    
    ENCODER_WEIGHTS = 'imagenet'
    ENCODER = encoder

    # standard unet
    if arct.lower()=='unet':
        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7 
        # use `imagenet` pre-trained weights for encoder initialization
        # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        # model output channels (number of classes in your dataset)
        MODEL = smp.Unet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,     
                         in_channels=in_channels,  classes=n_classes,
                         activation=activation_name)
                         #activation='softmax')
                         
    #shallow Unet with less depth                          
    elif arct.lower()=='unet_mini':
        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7 
        # use `imagenet` pre-trained weights for encoder initialization
        # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        # model output channels (number of classes in your dataset)
        MODEL = smp.Unet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,     
                         in_channels=in_channels,  classes=n_classes,
                         activation=activation_name,
                         encoder_depth=2, decoder_channels=(256,128))
                         #activation='softmax')
                         
    # unet with attention -----------------------------------------------------                        
    elif arct.lower()=='unet_scse':
        MODEL = smp.Unet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,     
                         in_channels=in_channels,  classes=n_classes,
                         activation=activation_name,
                         decoder_attention_type='scse') 
        
    elif arct.lower()=='unet_se':                        
        MODEL = Unet_Attention(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,     
                         in_channels=in_channels,  classes=n_classes,
                         activation=activation_name,
                         decoder_attention_type='se')

    elif arct.lower()=='unet_sk':                        
        MODEL = Unet_Attention(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,     
                         in_channels=in_channels,  classes=n_classes,
                         activation=activation_name,
                         decoder_attention_type='sk')
    
    elif arct.lower()=='unet_cbam':                        
         MODEL = Unet_Attention(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,     
                          in_channels=in_channels,  classes=n_classes,
                          activation=activation_name,
                          decoder_attention_type='cbam') 

    elif arct.lower()=='unet_bam':                        
         MODEL = Unet_Attention(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,     
                          in_channels=in_channels,  classes=n_classes,
                          activation=activation_name,
                          decoder_attention_type='bam') 

    elif arct.lower()=='unet_eca':                        
         MODEL = Unet_Attention(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,     
                          in_channels=in_channels,  classes=n_classes,
                          activation=activation_name,
                          decoder_attention_type='eca') 

    elif arct.lower()=='unet_coord':                        
         MODEL = Unet_Attention(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,     
                          in_channels=in_channels,  classes=n_classes,
                          activation=activation_name,
                          decoder_attention_type='coord')
    # unet with attention -----------------------------------------------------                        
    
    
                         
    elif arct.lower()=='unetplusplus':
        MODEL = smp.UnetPlusPlus(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,     
                         in_channels=in_channels,  classes=n_classes,
                         activation=activation_name)

                         
    elif arct.lower()=='linknet':
        MODEL = smp.Linknet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,
                         in_channels=in_channels,  classes=n_classes,
                         activation=activation_name)
                         #activation='softmax')
    
    elif arct.lower()=='fpn':
        MODEL = smp.FPN(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,
                        in_channels=in_channels,  classes=n_classes,
                        activation=activation_name)
    
    elif arct.lower()=='deeplabv3':
        MODEL = smp.DeepLabV3(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,
                        in_channels=in_channels,  classes=n_classes,
                        activation=activation_name)
        
    elif arct.lower()=='deeplabv3plus':
        MODEL = smp.DeepLabV3Plus(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,
                              in_channels=in_channels,  classes=n_classes,
                              activation=activation_name)  
    
    elif arct.lower()=='manet':
        MODEL = smp.MAnet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,
                          in_channels=in_channels,  classes=n_classes,
                          activation=activation_name)  

    elif arct.lower()=='pan':
        MODEL = smp.PAN(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,
                        in_channels=in_channels,  classes=n_classes,
                        activation=activation_name)     

    elif arct.lower()=='pspnet':
        MODEL = smp.PSPNet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,
                        in_channels=in_channels,  classes=n_classes,
                        activation=activation_name)              
    
    
    #-------------------------------------------------------------
    # customized segmentation models are listed below
    
    # Unet that can handle multiple band images
    elif arct.lower()=='munet':    
        MODEL = MUNet(in_channels=in_channels, n_classes=n_classes, 
                      activation=activation_name)
        m_fullname = arct
    
    # Attention Unet that can handle multiple band images    
    elif arct.lower()=='munet_ag':    
        MODEL = MUNet_AG(in_channels=in_channels, n_classes=n_classes, 
                      activation=activation_name)
        m_fullname = arct
    
    # Unet with CBAM that can handle multiple band images
    elif arct.lower()=='munet_cbam':    
        MODEL = MUNet_CBAM(in_channels=in_channels, n_classes=n_classes, 
                      activation=activation_name)
        m_fullname = arct
    
    # Another Unet that can handle multiple band images     
    #elif arct.lower()=='munet1':        
    #    MODEL = MUNet1(n_classes=n_classes, activation=activation_name)
    #    m_fullname = arct

    # Another Unet that can handle multiple band images          
    #elif arct.lower()=='munet2':        
    #    MODEL = MUNet2(n_classes=n_classes, activation=activation_name)
    #    m_fullname = arct
    
    # Another Unet that can handle multiple band images          
    elif arct.lower()=='munetx':        
        MODEL = MUNetX(n_classes=n_classes, activation=activation_name)
        m_fullname = arct
        
    #-------------------------------------------------------------        
    elif arct.lower()=='mxsegnet':        
        MODEL = MXSegNet(encoder_name=ENCODER, in_channels=in_channels, 
                        n_classes=n_classes)

    else:
        MODEL = None
        m_fullname = arct
        
    return MODEL, m_fullname        


# generate a model file name 
def generate_model_filename(model_name):
    mfname = 'seg4_%s_best.pth' % model_name  
    return mfname


# save model and auxiliary information
def save_seg_model(fpath, model, arct, encoder, 
                   n_classes, class_names,
                   in_channels = 3,
                   img_sz = 512):
    torch.save({
            'n_classes': n_classes,
            'class_names': class_names, 
            'in_channels': in_channels,
            'img_sz': img_sz,
            'arct': arct,
            'encoder': encoder,
            'model_state_dict': model.state_dict(),                        
            },  fpath)


# save model from model filepath
def load_seg_model(fpath, device='cuda'):
    # load the model and the trained weights
    mdict = torch.load(fpath, map_location=device)
    
    n_classes = mdict['n_classes']
    class_names = mdict['class_names']
    in_channels = mdict['in_channels']
    arct = mdict['arct']
    encoder = mdict['encoder']

    if 'img_sz' in mdict.keys():
        img_sz = mdict['img_sz']
    else:
        img_sz = None
    
    model, model_name = create_model(arct=arct, encoder=encoder, 
                                     n_classes=n_classes, 
                                     in_channels=in_channels)
    
    model.load_state_dict(mdict['model_state_dict'])
    
    return (model, model_name, n_classes, class_names,
            in_channels, img_sz)


# save training check point
def save_checkpoint(fpath, model, arct, encoder, 
               optimizer_name, optimizer,
               n_classes, class_names,
               n_channels, img_sz,
               epochs, batch_size,
               lr, momentum, weight_decay,
               train_losses, val_losses, train_scores, val_scores):
    torch.save({
            'n_classes': n_classes,
            'class_names': class_names, 
            'n_channels': n_channels,
            'img_sz': img_sz,
            
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'momentum': momentum,
            'weight_decay': weight_decay,
            
            'arct': arct,
            'encoder': encoder,
            'model_state_dict': model.state_dict(),            
 
            'opti_name': optimizer_name,
            'opti_state_dict': optimizer.state_dict(),            

            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_scores': train_scores,
            'val_scores': val_scores,
            },  fpath)
    

    
