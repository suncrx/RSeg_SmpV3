# -*- coding: utf-8 -*-
"""
Created on Fri May 17 22:23:34 2024

@author: renxi
"""
import torch



# %% test MxSegNet---------------------------------------------
def test_mxsegnet():
    from models.mxsegnet import MXSegNet
    chn = 6
    ms = MXSegNet(in_channels=chn, n_classes=1, att_type='coord')
    #print(ms)
    print(ms.encoder.conv1.weight.shape)
    print(ms.encoder.conv1.weight[0,0])
    d = torch.rand(4, chn, 256,256)
    sal = torch.rand(4, 1, 256,256)
    print(d.shape)
    o1 = ms(d)
    o2 = ms(d, sal)
    print(o1.shape)
    print(o2.shape)



def test_munetx():
    from models.munetx import MUNetX
    m = MUNetX(in_channels=5, n_classes=1, activation='softmax')
    print(m)
    
    d = torch.rand(1, 5, 256,256)
    print(d.shape)
    o=m(d)
    print(o.shape)


# %% test resnet encoder
def test_resnet_enc():
    from models.resnet_enc import resnet18_enc
    from models.resnet_enc import resnet50_enc
    nchns = 6
    m = resnet18_enc(in_channels=nchns, pretrained=True)
    #print(m)
    print(m.conv1.weight[0,0])
    
    d = torch.rand(4, nchns, 256, 256)
    fts = m(d)
    print(d.shape)
    print('features:')
    for ft in fts:
        print(ft.shape)
        

# %% test my resnet encoder
def test_mresnet_enc():
    from models.mresnet_enc1 import mresnet18_enc
    from models.mresnet_enc1 import mresnet34_enc
    from models.mresnet_enc1 import mresnet50_enc
    nchns = 6
    #m = mresnet18_enc(in_channels=nchns, att_type='cbam',  pretrained=True)
    m = mresnet34_enc(in_channels=nchns, att_type='cbam',  pretrained=True)
    #m = mresnet50_enc(in_channels=nchns, att_type='se',  pretrained=True)
    print(m)
    print(m.conv1.weight[0,0])
    d = torch.rand(4,nchns,256,256)
    fts = m(d)
    for ft in fts:
        print(ft.shape)
                
# %% test my resnet encoder 2
def test_mresnet_enc2():
    from models.mresnet_enc2 import mresnet18_enc2
    from models.mresnet_enc2 import mresnet50_enc2
    nchns =3
    m = mresnet18_enc2(in_channels=nchns, att_type='cbam',
                      pretrained=True)
    #m = mresnet50_enc(in_channels=6, att_type='se',
    #                  pretrained=True)
    #print(m)
    print(m.conv1.weight[0,0])
    d = torch.rand(4,nchns,256,256)
    fts = m(d)
    for ft in fts:
        print(ft.shape)

        
# %% test resnet attention
def test_resnet_att2():
    from models.resnet_att2 import resnet18_att2
    from models.resnet_att2 import resnet34_att2
    from models.resnet_att2 import resnet50_att2
    nchns = 6
    m = resnet18_att2(in_channels=nchns, att_type='cbam', pretrained=True)
    print(m.conv1.weight[0,0])
    m = resnet50_att2(in_channels=nchns, att_type='cbam', pretrained=True)
    #print(m)
    print(m.conv1.weight[0,0])
    
    d = torch.rand(4, nchns, 256, 256)
    o = m(d)
    print(d.shape)
    print(o.shape)
    
        
# %% test resnet attention
def test_resnet_att1():
    from models.resnet_att1 import resnet18_att1
    m = resnet18_att1(att_type='bam')    
    #print(m)
    print(m.conv1[0].weight[0,0])



#%% test attention
def test_mattention():
    from models.mattention import MAttention
    a = MAttention('cbam', in_channels=16)
    print(a)



# %% test Unet attention
def test_unet_attention():
    from models.unet_att import Unet_Attention   
    m = Unet_Attention(encoder_name='resnet34', 
                       encoder_weights='imagenet',
                       in_channels=3, classes=6, 
                       decoder_attention_type='coord')
    #print(un)
    d = torch.rand(4, 3, 256,256)
    print(d.shape)
    o=m(d)
    print(o.shape)


# %% test create_model
def test_create_model():
    from models import utils
    m, mname = utils.create_model(arct='unet')
    print(m)
    print(mname)
    mfname = utils.generate_model_filename(mname)
    print(mfname)
    

#test_mxsegnet()
#test_resnet_att1()
#test_resnet_enc()
test_mresnet_enc()
#test_mresnet_enc2()
#test_resnet_att2()
#test_unet_attention()