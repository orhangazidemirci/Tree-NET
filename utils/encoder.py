# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 00:35:04 2024

@author: Orhan
"""
import os

# current_directory = os.getcwd()

from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
from torch.autograd import Variable

import datetime

from torch.utils.data import DataLoader , TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import torch.optim as optim
import torch
import gc
import cv2
import imageio
import numpy as np
import json
#from PIL import Image
#from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
import argparse
# import models
#### CONFIGURATIONS

os.chdir('C:\\Users\\Orhan\\Documents\\GitHub\\Polyp-PVT')
pretrained=False
dataset_name='CVC-ClinicDB'  # CVC-ClinicDB , Kvasir ,ISAC2018
model_selected='Encoder_x4' # Encoder, Res_Encoder, Encoder_x4,Res_Encoder_x4, Dual, Decoder, Decoder_x4, Decoder_x16, FCC, Conv_attention
# if dataset_name=='ISAC2018':
#     dim = (32, 32)
# else:
#     dim = (256, 256)
dim = (512, 512)

batch_size=8
l_rate=0.001
epochs=90
#path='/content/gdrive/MyDrive/thesis/datasets/CVC-ClinicDB'

class Encoder_x4(nn.Module):
   def __init__(self):

       super(Encoder_x4, self).__init__()
       # self.encoder=nn.Sequential(nn.Conv2d(3,8, 3),nn.ELU(),nn.MaxPool2d(2,2),
       #                 nn.ELU(),nn.Conv2d(8, 24, 3),nn.ELU(),nn.Conv2d(24, 32, 5,stride=3)
       #                 ##nn.Conv2d(8, 8, 3)
       #                 ,nn.ELU(),nn.Conv2d(32, 10, 3,stride=2),nn.ELU(),nn.ConvTranspose2d(10, 10, 3),nn.ELU())
       self.encoder = nn.Sequential(
           nn.Conv2d(3, 8, 3, padding=2),
           nn.BatchNorm2d(8),
           nn.ELU(),
           nn.Conv2d(8, 24, 3, padding=1, stride=2),
           nn.BatchNorm2d(24),
           nn.ELU(),
           nn.Conv2d(24, 32, 5, padding=1),
           nn.BatchNorm2d(32),
           nn.ReLU(),
           nn.Conv2d(32, 18, 3, padding=1, stride=2),
           nn.BatchNorm2d(18),
           nn.Tanh(),
           nn.Conv2d(18, 3, 3, padding=1),  # Adjusted padding
           nn.BatchNorm2d(3),
           nn.ELU()
       )

   #        self.decoder=nn.Sequential(nn.ConvTranspose2d(10, 32, 5,stride=3),nn.ELU(),nn.Conv2d(32, 24, 3),nn.ELU(),nn.Upsample(scale_factor=2, mode='nearest'),
   # nn.Conv2d(24, 16, 3),nn.ELU(),nn.Upsample(scale_factor=2, mode='nearest'),
   # nn.Conv2d(16, 8, 3),nn.ELU(), nn.Conv2d(8, 3, 3),nn.ELU())
       self.decoder = nn.Sequential(
           nn.ConvTranspose2d(3, 10, 3, padding=1),  # Output: (N, 10, 64, 64)
           nn.BatchNorm2d(10),
           nn.ReLU(),
           nn.ConvTranspose2d(10, 24, 3, padding=1, stride=2),  # Output: (N, 24, 128, 128)
           nn.BatchNorm2d(24),
           nn.ELU(),
           nn.ConvTranspose2d(24, 32, 3, padding=1),  # Output: (N, 32, 128, 128)
           nn.BatchNorm2d(32),
           nn.Tanh(),
           nn.ConvTranspose2d(32, 32, 3, padding=1),  # Output: (N, 32, 256, 256)
           nn.BatchNorm2d(32),
           nn.ReLU(),
           nn.Upsample(scale_factor=2, mode='nearest'),
           nn.ConvTranspose2d(32, 24, 3, padding=1),  # Output: (N, 24, 256, 256)
           nn.BatchNorm2d(24),
           nn.ELU(),
           nn.ConvTranspose2d(24, 8, 3, padding=1),  # Output: (N, 18, 256, 256)
           nn.BatchNorm2d(8),
           nn.ReLU(),
           nn.ConvTranspose2d(8, 3, 3),  # Output: (N, 3, 256, 256)
           nn.ELU()
           )

   def forward(self, inp):
       x=self.encoder(inp)
       ## x=self.bridge(x)+x
       
       ## x=x-self.res(inp)
      # # x=self.encoder2(x)
       
       x=self.decoder(x)
       
     #  # out=self.decoder(x)+F.interpolate(self.pooling(inp), scale_factor=8, mode='nearest')
       return x

# class Encoder_x4(nn.Module):
#     def __init__(self, base_channels=24, bottleneck_channels=3):
#         """
#         A simple Autoencoder without skip connections.

#         Args:
#             base_channels (int): Number of channels after the first convolution.
#                                   Subsequent layers will multiply this.
#             bottleneck_channels (int): Number of channels in the bottleneck layer.
#         """
#         super().__init__()

#         # --- Encoder ---
#         # Input: (N, 3, 256, 256)

#         self.enc_conv1 = nn.Sequential(
#             # Output: (N, base_channels, 256, 256)
#             nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(base_channels),
#             nn.ReLU(inplace=True)
#         )

#         self.enc_conv2 = nn.Sequential(
#             # Downsample 1 -> (N, base_channels*2, 128, 128)
#             nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(base_channels * 2),
#             nn.ReLU(inplace=True)
#         )

#         self.enc_conv3 = nn.Sequential(
#             # Downsample 2 -> (N, base_channels*4, 64, 64)
#             nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(base_channels * 4),
#             nn.ReLU(inplace=True)
#         )

#         # Bottleneck Layer
#         self.bottleneck = nn.Sequential(
#               # Reduce channels slightly if desired, maintain spatial size -> (N, bottleneck_channels, 64, 64)
#             nn.Conv2d(base_channels * 4, bottleneck_channels, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(bottleneck_channels),
#             nn.ReLU(inplace=True)
#             # Alternative: Could use a 1x1 conv here: nn.Conv2d(base_channels*4, bottleneck_channels, kernel_size=1, bias=False)
#         )

#         # --- Decoder ---
#         # Input: (N, bottleneck_channels, 64, 64)

#         self.dec_upconv1_prep = nn.Sequential(
#             # Prepare for upsampling, potentially increase channels back -> (N, base_channels*4, 64, 64)
#             nn.Conv2d(bottleneck_channels, base_channels * 3, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(base_channels * 3),
#             nn.ReLU(inplace=True)
#         )

#         self.dec_upconv2 = nn.Sequential(
#             # Upsample 1 -> (N, base_channels*2, 128, 128)
#             nn.ConvTranspose2d(base_channels * 3, base_channels * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
#             nn.BatchNorm2d(base_channels * 2),
#             nn.ReLU(inplace=True)
#         )

#         self.dec_upconv3 = nn.Sequential(
#             # Upsample 2 -> (N, base_channels, 256, 256)
#             nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
#             nn.BatchNorm2d(base_channels),
#             nn.ReLU(inplace=True)
#         )

#         # Final Layer to produce 3 output channels
#         self.final_conv = nn.Sequential(
#               # Output: (N, 3, 256, 256)
#             nn.Conv2d(base_channels, 3, kernel_size=3, stride=1, padding=1), # Use padding=1 for 3x3 kernel to keep size
#             nn.Sigmoid() # Output pixel values between 0 and 1
#         )


#     def forward(self, x):
#         # Encode
#         x = self.enc_conv1(x)
#         x = self.enc_conv2(x)
#         x = self.enc_conv3(x)
#         x = self.bottleneck(x)

#         # Decode
#         x = self.dec_upconv1_prep(x)
#         x = self.dec_upconv2(x)
#         x = self.dec_upconv3(x)
#         x = self.final_conv(x)

#         return x

# model_x=Encoder_x4()
# x=torch.randn(10,3,224,224)
# model_x(x).shape    
# encoder_path='./pretrained_pth/encoder_x4_CVC-ClinicDB.pt'

# for param in in_model.parameters():
#     param.requires_grad = False  

class Decoder_x16(nn.Module):
    def __init__(self):

        super(Decoder_x16, self).__init__()
        self.encoder_out=nn.Sequential(
              nn.Conv2d(1,8, 3),
              nn.ELU(),
              nn.MaxPool2d(2,2, padding=0, return_indices=False, ceil_mode=False),
              nn.Conv2d(8, 24, 3),
              nn.ReLU(),
              nn.MaxPool2d(2,2, padding=0, return_indices=False, ceil_mode=False),
              nn.Conv2d(24, 32, 5,stride=2),
              nn.ReLU(),

                  ##nn.Conv2d(8, 8, 3)
              nn.ELU(),
              nn.Conv2d(32, 24, 3,stride=2),
              nn.ReLU(),
              nn.ConvTranspose2d(24, 16, 3),
              nn.Sigmoid())
   #        self.decoder=nn.Sequential(nn.ConvTranspose2d(10, 32, 5,stride=3),nn.ELU(),nn.Conv2d(32, 24, 3),nn.ELU(),nn.Upsample(scale_factor=2, mode='nearest'),
   # nn.Conv2d(24, 16, 3),nn.ELU(),nn.Upsample(scale_factor=2, mode='nearest'),
   # nn.Conv2d(16, 8, 3),nn.ELU(), nn.Conv2d(8, 3, 3),nn.ELU())
        self.decoder_out=nn.Sequential(
              nn.ConvTranspose2d(16, 24, 3,padding=1),
              nn.ELU(),
              nn.Upsample(scale_factor=2, mode='nearest'),
              nn.ConvTranspose2d(24, 32, 3,padding=1),
              nn.ELU(),
              nn.Upsample(scale_factor=2, mode='nearest'),
              nn.ConvTranspose2d(32, 24, 3,padding=1),
              nn.ELU(),
              nn.Upsample(scale_factor=2, mode='nearest'),
              nn.ConvTranspose2d(24, 24, 3,padding=2),
              nn.ReLU(),
              nn.Upsample(scale_factor=2, mode='nearest'),
              nn.ConvTranspose2d(24, 16, 5,padding=1),
              nn.ELU(),
              nn.ConvTranspose2d(16, 8, 3,padding=1),
              nn.ReLU(),
              nn.ConvTranspose2d(8, 1, 3),
              nn.Sigmoid())
 #        self.encoder_out=nn.Sequential(nn.Conv2d(1,8, 3),nn.ELU(),nn.MaxPool2d(2,2, padding=0, return_indices=False, ceil_mode=False),
 #                nn.ELU(),nn.Conv2d(8, 24, 3),nn.ELU(),nn.Conv2d(24, 32, 5,stride=2)
 #                ##nn.Conv2d(8, 8, 3)
 #                ,nn.ELU(),nn.Conv2d(32, 5, 3,stride=2),nn.ELU(),nn.ConvTranspose2d(5, 3, 3),nn.Sigmoid())
 # #        self.decoder=nn.Sequential(nn.ConvTranspose2d(10, 32, 5,stride=3),nn.ELU(),nn.Conv2d(32, 24, 3),nn.ELU(),nn.Upsample(scale_factor=2, mode='nearest'),
 # # nn.Conv2d(24, 16, 3),nn.ELU(),nn.Upsample(scale_factor=2, mode='nearest'),
 # # nn.Conv2d(16, 8, 3),nn.ELU(), nn.Conv2d(8, 3, 3),nn.ELU())
 #        self.decoder_out=nn.Sequential(nn.Conv2d(3, 24, 3),nn.ELU(),nn.Upsample(scale_factor=2, mode='nearest'),nn.ConvTranspose2d(24, 32, 3,padding=1),nn.ELU(),nn.Upsample(scale_factor=2, mode='nearest'),
 # nn.ConvTranspose2d(32, 24, 3),nn.ELU(),nn.ConvTranspose2d(24, 16, 5),nn.ELU(),nn.Upsample(scale_factor=2, mode='nearest'),
 # nn.ConvTranspose2d(16, 8, 3),nn.ELU(),nn.ConvTranspose2d(8, 1, 3),nn.ELU())        
    def forward(self, x):
        x=self.encoder_out(x)
        x=self.decoder_out(x)
        return x

        
class Decoder_x8(nn.Module):
    def __init__(self):

        super(Decoder_x8, self).__init__()
        self.encoder_out=nn.Sequential(
              nn.Conv2d(1,8, 3),
              nn.ELU(),
              nn.MaxPool2d(2,2, padding=0, return_indices=False, ceil_mode=False),
              nn.Conv2d(8, 24, 3),
              nn.ReLU(),
              nn.MaxPool2d(2,2, padding=0, return_indices=False, ceil_mode=False),
              nn.Conv2d(24, 32, 5,stride=2),
              nn.ReLU(),

                  ##nn.Conv2d(8, 8, 3)
              nn.ELU(),
              nn.Conv2d(32, 24, 3,stride=2),
              nn.ReLU(),
              nn.ConvTranspose2d(24, 16, 3),
              nn.Sigmoid())
   #        self.decoder=nn.Sequential(nn.ConvTranspose2d(10, 32, 5,stride=3),nn.ELU(),nn.Conv2d(32, 24, 3),nn.ELU(),nn.Upsample(scale_factor=2, mode='nearest'),
   # nn.Conv2d(24, 16, 3),nn.ELU(),nn.Upsample(scale_factor=2, mode='nearest'),
   # nn.Conv2d(16, 8, 3),nn.ELU(), nn.Conv2d(8, 3, 3),nn.ELU())
        self.decoder_out=nn.Sequential(
              nn.ConvTranspose2d(16, 24, 3,padding=1),
              nn.ELU(),
              nn.Upsample(scale_factor=2, mode='nearest'),
              nn.ConvTranspose2d(24, 32, 3,padding=1),
              nn.ELU(),
              nn.Upsample(scale_factor=2, mode='nearest'),
              nn.ConvTranspose2d(32, 24, 3,padding=1),
              nn.ELU(),
              nn.Upsample(scale_factor=2, mode='nearest'),
              nn.ConvTranspose2d(24, 24, 3,padding=2),
              nn.ReLU(),
              nn.Upsample(scale_factor=2, mode='nearest'),
              nn.ConvTranspose2d(24, 16, 5,padding=1),
              nn.ELU(),
              nn.ConvTranspose2d(16, 8, 3,padding=1),
              nn.ReLU(),
              nn.ConvTranspose2d(8, 1, 3),
              nn.Sigmoid())
 #        self.encoder_out=nn.Sequential(nn.Conv2d(1,8, 3),nn.ELU(),nn.MaxPool2d(2,2, padding=0, return_indices=False, ceil_mode=False),
 #                nn.ELU(),nn.Conv2d(8, 24, 3),nn.ELU(),nn.Conv2d(24, 32, 5,stride=2)
 #                ##nn.Conv2d(8, 8, 3)
 #                ,nn.ELU(),nn.Conv2d(32, 5, 3,stride=2),nn.ELU(),nn.ConvTranspose2d(5, 3, 3),nn.Sigmoid())
 # #        self.decoder=nn.Sequential(nn.ConvTranspose2d(10, 32, 5,stride=3),nn.ELU(),nn.Conv2d(32, 24, 3),nn.ELU(),nn.Upsample(scale_factor=2, mode='nearest'),
 # # nn.Conv2d(24, 16, 3),nn.ELU(),nn.Upsample(scale_factor=2, mode='nearest'),
 # # nn.Conv2d(16, 8, 3),nn.ELU(), nn.Conv2d(8, 3, 3),nn.ELU())
 #        self.decoder_out=nn.Sequential(nn.Conv2d(3, 24, 3),nn.ELU(),nn.Upsample(scale_factor=2, mode='nearest'),nn.ConvTranspose2d(24, 32, 3,padding=1),nn.ELU(),nn.Upsample(scale_factor=2, mode='nearest'),
 # nn.ConvTranspose2d(32, 24, 3),nn.ELU(),nn.ConvTranspose2d(24, 16, 5),nn.ELU(),nn.Upsample(scale_factor=2, mode='nearest'),
 # nn.ConvTranspose2d(16, 8, 3),nn.ELU(),nn.ConvTranspose2d(8, 1, 3),nn.ELU())        
    def forward(self, x):
        x=self.encoder_out(x)
        x=self.decoder_out(x)
        return x
    
class Decoder_x4(nn.Module):
    def __init__(self):

        super(Decoder_x4, self).__init__()
        self.encoder_out= nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, 24, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 32, 5, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 18, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(18, 3, 3, padding=1), 
            nn.Sigmoid()
        )

        self.decoder_out = nn.Sequential(
            nn.ConvTranspose2d(3, 18, 3, padding=1, stride=2),  # Output: (N, 24, 128, 128)
            nn.ReLU(),
            nn.ConvTranspose2d(18, 32, 3, padding=1),  # Output: (N, 32, 128, 128)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, padding=1),  # Output: (N, 32, 256, 256)
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(32, 24, 3, padding=1),  # Output: (N, 24, 256, 256)
            nn.ReLU(),
            nn.ConvTranspose2d(24, 8, 3, padding=1),  # Output: (N, 18, 256, 256)
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 3),  # Output: (N, 3, 256, 256)
            nn.Sigmoid()
            )
    def forward(self, x):
        x=self.encoder_out(x)
        x=self.decoder_out(x)
        return x
# train=True
# model_mapping = {
#     # "FCC": models.FCC_Encoder(),
#     "Encoder": models.Encoder(),
#     "Res_Encoder": models.Res_Encoder(),
#     "Encoder_x4": models.Encoder_x4(),
#     "Res_Encoder_x4": models.Res_Encoder_x4(),
#     "Decoder": models.Decoder(),
#     "Decoder_x4": models.Decoder_x4()
#     }
if model_selected=='Encoder_x4':
    pretrained_path='./model_pth/encoder_x4_Polyp.pt'

    model=Encoder_x4().cuda()

elif model_selected=='Decoder_x4':
    model=Decoder_x4().cuda()
elif model_selected=='Decoder_x8':
    model=Decoder_x8().cuda()

elif model_selected=='Decoder_x16':
    model=Decoder_x16().cuda()
    pretrained_path='./model_pth/decoder_x16_Polyp.pt'


if pretrained==True:
   model.load_state_dict(torch.load(pretrained_path))
 
image_path = 'C:/Users/Orhan/Desktop/python/thesis/datasets/{}/Original'.format(dataset_name)
gtruth_path = 'C:/Users/Orhan/Desktop/python/thesis/datasets/{}/Ground Truth'.format(dataset_name)
path_indices='C:/Users/Orhan/Desktop/python/thesis/output_file/path_indices_{}.png'.format(dataset_name)
indices_saved=True  
# save_norm='C:/Users/Orhan/Desktop/python/thesis/output_file/data_norm_{}.json'.format(dataset_name)
save_model = './model_pth/{}_new_{}.pt'.format(model_selected.lower(),dataset_name)
save_model = './model_pth/encoder_x4_Polyp.pt'.format(model_selected.lower(),dataset_name)

#### LOADING THE DATA...


# x=torch.randn(10,3,224,224)    
# model_x=SimpleAE_NoSkips()
# model_x(x).shape

# image_name = [file for file in os.listdir(image_path) ]
# gtruth_name = [file for file in os.listdir(gtruth_path)]
# data=torch.empty(len(image_name),dim[0],dim[1], 3)
# label=torch.empty(len(image_name),dim[0],dim[1],1)
# Read each TIFF image
# for i,file in enumerate(image_name):
#     im_path = os.path.join(image_path, file)
#     label_path = os.path.join(gtruth_path, file)

#     data[i] =torch.tensor( cv2.resize(cv2.imread(im_path, cv2.IMREAD_COLOR ), (dim[1],dim[0]), interpolation = cv2.INTER_AREA))
#   #  torch.tensor(imageio.imread(im_path))
#     lab=cv2.resize(cv2.imread( label_path), (dim[1],dim[0]), interpolation = cv2.INTER_AREA)
#     lab=cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)

#     #imageio.imread(label_path)
model_name = 'Decoder_Skin'
###############################################
parser = argparse.ArgumentParser()

parser.add_argument('--epoch', type=int,
                    default=100, help='epoch number')

parser.add_argument('--lr', type=float,
                    default=1e-3, help='learning rate')

parser.add_argument('--optimizer', type=str,
                    default='AdamW', help='choosing optimizer AdamW or SGD')

parser.add_argument('--augmentation',
                    default=True, help='choose to do random flip rotation')

parser.add_argument('--batchsize', type=int,
                    default=32, help='training batch size')

parser.add_argument('--trainsize', type=int,
                    default=384, help='training dataset size') #Change here

parser.add_argument('--clip', type=float,
                    default=0.5, help='gradient clipping margin')

parser.add_argument('--decay_rate', type=float,
                    default=0.1, help='decay rate of learning rate')

parser.add_argument('--decay_epoch', type=int,
                    default=50, help='every n epochs decay learning rate')

parser.add_argument('--train_path', type=str,
                    default='./dataset/ISAC2018/train/',
                    help='path to train dataset')

parser.add_argument('--test_path', type=str,
                    default='./dataset/ISAC2018/', #'./dataset/TestDataset/'
                    help='path to testing Kvasir dataset')

parser.add_argument('--train_save', type=str,
                    default='./model_pth/'+model_name+'/')

opt = parser.parse_args()

#     label[i] = torch.tensor(lab).unsqueeze(-1)
test = 'C:/Users/Orhan/Desktop/python/thesis/datasets/test/'
train = 'C:/Users/Orhan/Desktop/python/thesis/datasets/train/'
val= 'C:/Users/Orhan/Desktop/python/thesis/datasets/valid/' 

inp_path='images/' # 'images/' 
# Read each TIFF image
train_loader = get_loader(opt.train_path+inp_path, opt.train_path+inp_path, batchsize=opt.batchsize, trainsize=opt.trainsize,
                          augmentation=opt.augmentation)
test_loader=get_loader(opt.test_path+'test/{}'.format(inp_path), opt.test_path+'test/{}'.format(inp_path), batchsize=1, trainsize=opt.trainsize,
                          augmentation=False) # CVC-ClinicDB
valid_loader=get_loader( opt.test_path+'valid/{}'.format(inp_path), opt.test_path+'valid/{}'.format(inp_path), batchsize=opt.batchsize, trainsize=opt.trainsize,
                          augmentation=opt.augmentation)
# test_loader=get_loader('./dataset/test/'+inp_path, './dataset/test/'+inp_path, batchsize=1, trainsize=opt.trainsize,
#                           augmentation=False)
# valid_loader=get_loader( opt.test_path+'test/{}'.format(inp_path), opt.test_path+'test/{}'.format(inp_path), batchsize=opt.batchsize, trainsize=opt.trainsize,
#                           augmentation=opt.augmentation)

# dataset = TensorDataset(data , label)


# train_loader = DataLoader(dataset, batch_size=1, sampler=train_indices)
# valid_loader = DataLoader(dataset, batch_size=1, sampler=val_indices)
# test_loader = DataLoader(dataset, batch_size=1, sampler=test_indices)


# test = 'C:/Users/Orhan/Desktop/python/thesis/datasets/test/'
# train = 'C:/Users/Orhan/Desktop/python/thesis/datasets/train/'
# val= 'C:/Users/Orhan/Desktop/python/thesis/datasets/valid/'  

# for i,data in enumerate(train_loader):
#     image,lab=data
#     image=image.view(image.size(1),image.size(2),image.size(3))
#     # image=torch.transpose(image,0,2)
#     image=image.detach().cpu()
#     image=image.numpy()
#     # lab=torch.transpose(lab,0,2)
#     lab=lab.view(lab.size(1),lab.size(2))

#     lab=lab.detach().cpu().numpy()
#     # cv2.imwrite(join('C:/Users/Orhan/Desktop',"{}.jpg".format(i)),image)
# # 
#     cv2.imwrite(train+'images/{}.jpg'.format(i), image)
#     cv2.imwrite(train+'masks/{}.jpg'.format(i), lab)

# for i,data in enumerate(test_loader):
#     image,lab=data
#     image=image.view(image.size(1),image.size(2),image.size(3))
#     # image=torch.transpose(image,0,2)
#     image=image.detach().cpu()
#     image=image.numpy()
#     # lab=torch.transpose(lab,0,2)
#     lab=lab.view(lab.size(1),lab.size(2))

#     lab=lab.detach().cpu().numpy()
#     # cv2.imwrite(join('C:/Users/Orhan/Desktop',"{}.jpg".format(i)),image)
# # 
#     cv2.imwrite(test+'images/{}.jpg'.format(i), image)
#     cv2.imwrite(test+'masks/{}.jpg'.format(i), lab)
    
# for i,data in enumerate(valid_loader):
#     image,lab=data
#     image=image.view(image.size(1),image.size(2),image.size(3))
#     # image=torch.transpose(image,0,2)
#     image=image.detach().cpu()
#     image=image.numpy()
#     # lab=torch.transpose(lab,0,2)
#     lab=lab.view(lab.size(1),lab.size(2))

#     lab=lab.detach().cpu().numpy()
#     # cv2.imwrite(join('C:/Users/Orhan/Desktop',"{}.jpg".format(i)),image)
# # 
#     cv2.imwrite(val+'images/{}.jpg'.format(i), image)
#     cv2.imwrite(val+'masks/{}.jpg'.format(i), lab)

    
# model=model_mapping["{}".format(model_selected)]
# if model_selected=='Res_Encoder':
#   model.load_state_dict(torch.load(save_model))

# if torch.cuda.is_available():
#     model =model.cuda()
model =model.cuda()
criterion = nn.MSELoss()

class EuclideanLoss(nn.Module):
    def __init__(self):
        super(EuclideanLoss, self).__init__()

    def forward(self, predicted, true):
        # Compute Euclidean loss (L2 loss)
        loss = torch.mean((predicted - true) ** 2)
        return loss

criterion = EuclideanLoss()

criterion = nn.MSELoss()

running_loss = 0.0
optimizer= optim.Adam(model.parameters(), lr=l_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, amsgrad=False)

from torch.optim.lr_scheduler import ReduceLROnPlateau

# Adjust parameters as needed
# mode='min': monitor metric to be minimized (like validation loss)
# factor=0.1: reduce LR by factor of 10
# patience=5: wait 5 epochs with no improvement before reducing
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)


trainsizes= [256,384,512]

if  "Decoder" in model_selected:
    trainsizes= [128,192,256]

len_train=len(train_loader)
for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss=0
    start_time = datetime.datetime.now()
    model.train()
    
    for img,lab in train_loader:
        for size in trainsizes:
            gc.collect()
            torch.cuda.empty_cache()
            # if torch.cuda.is_available():
            img= img.cuda()
            lab= lab.cuda()

            if "Res_Encoder" in model_selected:
              lab,output_res,in_out = model(img)
              lab=torch.tensor(lab, requires_grad=True)
              
              # loss= criterion(lab, output_res)
              loss= criterion(output_res+in_out,img)
    
              # loss_res= criterion(output_res, (img-output))
              total_loss=loss
    
            else:
                trainsize=size
                if size != trainsizes[1]:
                    img = F.upsample(img, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    lab = F.upsample(lab, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

                output = model(img)
                total_loss= criterion(output, lab)
        
    
            optimizer.zero_grad()
            # optimizer_res.zero_grad()
            # print("training loss=",loss.item())
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            # optimizer_res.step()
            # running_loss += torch.mean(total_loss)
    
            torch.cuda.empty_cache()
            # if torch.cuda.is_available():
            img= img.cuda()
            if size == trainsizes[1]:
            #     # loss_P2_record.update(loss_P2.data, opt.batchsize)
                running_loss += torch.mean(total_loss)


        # running_loss += torch.mean(loss)
  #      print("training loss=",loss.item())
        #running_loss += loss.data[0]
      #  scheduler.step()
          # if (running_loss>loss_prev):
          #     l_rate = l_rate/200


    print()
    model.eval()
    val_loss=0.0
    loss_val_res=0.0
    with torch.no_grad():
        for val_in, val_lab in valid_loader:
            optimizer.zero_grad()
            gc.collect()
            torch.cuda.empty_cache()
    
            # if torch.cuda.is_available():
            val_in= val_in.cuda()
            val_lab= val_lab.cuda()
    
            # val_lab=torch.transpose(val_lab, 2, 3)
          #  val_in=val_in.to(device)
            # val_in=val_in.view(-1, 96*96)
            if "Res_Encoder" in model_selected:
              lab,output_res,in_out = model(val_in)
              lab=torch.tensor(lab, requires_grad=True)
    
              loss_val= criterion(output_res+in_out , val_in)
              # loss_val= criterion(output_res, lab)
              # loss_val_res= criterion(output_res, (img-output))
              val_tot=loss_val
    
            else:
              val_out = model(val_in)
              val_tot= criterion(val_out, val_lab)
              
            val_loss +=val_tot.item()
            scheduler.step(val_loss)
            
  #  scheduler.step(loss_val)

    #    val_values.append(loss_val.detach())
      #  running_loss += loss.item()
      #   print(loss.item())
          #zero the parameter gradients
  ####  scheduler.step(loss_val)
    if isinstance(val_loss, tuple):
        
        avg_val_loss = sum(val_loss) / len(valid_loader)
    else:
        avg_val_loss = val_loss / len(valid_loader)


    if epoch==0:
        best_score=avg_val_loss
        print('Model saved at epoch: ', epoch+1 )

    if isinstance(best_score, tuple):
        
        best_score = sum(best_score)
        
    if avg_val_loss < best_score:
        best_score=avg_val_loss,
        torch.save(model.state_dict(), save_model)
        print('Model saved at epoch: ', epoch+1 )
    

    print('validation loss=',avg_val_loss)
    val_loss=0
    end_time = datetime.datetime.now()
    print("duration",end_time - start_time)

    #indices = torch.randperm(len(train_loader))
    # train_indices = indices[:train_size]
  #  train_loader = DataLoader(data, batch_size=batch_size, sampler=train_indices)
#del train_dataset

        #     loss_prev=running_loss
      #    ld_prev=loss_prev-running_loss
#    print(optimizer)
    print()
    print("epoch",epoch,"training loss=",running_loss/len_train)
    
    
    
# trainsizes= [256,384,512]
# loss_P2_record = AvgMeter()
# loss_P2_val= AvgMeter()
# len_train=len(train_loader)
# for epoch in range(epochs):  # loop over the dataset multiple times

#     running_loss=0
#     start_time = datetime.datetime.now()
#     model.train()
    

#     for i, pack in enumerate(train_loader, start=1):
#         for size in trainsizes:
#             optimizer.zero_grad()
#             # ---- data prepare ----
#             images,label = pack
#             images = Variable(images).cuda()
#             # ---- rescale ----
#             # trainsize = int(round(opt.trainsize * rate / 32) * 32)
#             trainsize=size
#             if size != 384:
#                 images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
#             # ---- forward ----
            
#             output=model(images)
  
#             # ---- loss function ----
#             loss = criterion(output, images)
#             # ---- backward ----
#             loss.backward()
#             clip_gradient(optimizer, opt.clip)
#             optimizer.step()
#             # ---- recording loss ----
#             if size == 384:
#                 # loss_P2_record.update(loss_P2.data, opt.batchsize)
#                 running_loss += torch.mean(loss)
#   #   for img in train_loader:

#   #       gc.collect()
#   #       torch.cuda.empty_cache()
#   #       # if torch.cuda.is_available():
#   #       img= img.cuda()

#   #       # if "Res_Encoder" in model_selected:
#   #       #   lab,output_res,in_out = model(img)
#   #       #   lab=torch.tensor(lab, requires_grad=True)
          
#   #       #   # loss= criterion(lab, output_res)
#   #       #   loss= criterion(output_res+in_out,img)

#   #       #   # loss_res= criterion(output_res, (img-output))
#   #       #   total_loss=loss

#   #       # else:
#   #       output = model(img)
#   #       total_loss= criterion(output, img)


#   #       optimizer.zero_grad()
#   #       # optimizer_res.zero_grad()
#   #      # print("training loss=",loss.item())
#   #       total_loss.backward()
#   #       nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
#   #       optimizer.step()
#   #       # optimizer_res.step()

#   #       running_loss += torch.mean(total_loss)

#   #       torch.cuda.empty_cache()
#   #       # if torch.cuda.is_available():
#   #       img= img.cuda()



#   #       # running_loss += torch.mean(loss)
#   # #      print("training loss=",loss.item())
#   #       #running_loss += loss.data[0]
#   #     #  scheduler.step()
#   #        # if (running_loss>loss_prev):
#   #        #     l_rate = l_rate/200


#     print()
#     model.eval()
#     val_loss=0.0
#     loss_val_res=0.0
#     for val_in, val_o in valid_loader:
#         optimizer.zero_grad()
#         gc.collect()
#         torch.cuda.empty_cache()

#         # if torch.cuda.is_available():
#         val_in= val_in.cuda()
#         # val_lab=torch.transpose(val_lab, 2, 3)
#       #  val_in=val_in.to(device)
#        # val_in=val_in.view(-1, 96*96)

#         val_out = model(val_in)
#         val_tot= criterion(val_out, val_in)
          
#         val_loss +=val_tot.item()
#         # loss_P2_record.update(loss_P2.data, opt.batchsize)
#   #  scheduler.step(loss_val)

#     #    val_values.append(loss_val.detach())
#       #  running_loss += loss.item()
#      #   print(loss.item())
#          #zero the parameter gradients
#   ####  scheduler.step(loss_val)
#     if isinstance(val_loss, tuple):
        
#         avg_val_loss = sum(val_loss) / len(valid_loader)
#     else:
#         avg_val_loss = val_loss / len(valid_loader)


#     if epoch==0:
#         best_score=avg_val_loss
#         print('Model saved at epoch: ', epoch+1 )

#     if isinstance(best_score, tuple):
        
#         best_score = sum(best_score)
        
#     if avg_val_loss < best_score:
#         best_score=avg_val_loss,
#         torch.save(model.state_dict(), save_model)
#         print('Model saved at epoch: ', epoch+1 )
    

#     print('validation loss=',avg_val_loss)
#     val_loss=0
#     end_time = datetime.datetime.now()
#     print("duration",end_time - start_time)

#     #indices = torch.randperm(len(train_loader))
#    # train_indices = indices[:train_size]
#   #  train_loader = DataLoader(data, batch_size=batch_size, sampler=train_indices)
# #del train_dataset

#         #     loss_prev=running_loss
#      #    ld_prev=loss_prev-running_loss
# #    print(optimizer)
#     print()
#     print("epoch",epoch,"training loss=",running_loss/len_train)

# torch.save(model.state_dict(), save_model)

from os import listdir
from os.path import isfile, join
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
# Sample data (replace with your actual data)
# input_data = np.random.rand(10, 50)  # 10 samples of input with 50 features
# real_values = np.random.rand(10)  # Real values
# your_algorithm_predictions = np.random.rand(10)  # Predictions from your algorithm
# other_algorithm_predictions = np.random.rand(10)  # Predictions from another algorithm
def unnormalize(tensor, mean, std):
    """
    Un-normalize a normalized tensor image and clip to [0, 1] range.
    
    :param tensor: The normalized tensor image (shape: [C, H, W])
    :param mean: List of mean values for each channel
    :param std: List of std values for each channel
    :return: Clipped un-normalized tensor image
    """
    # Reverse the normalization (out-of-place)
    unnormalized_tensor = torch.empty_like(tensor)  # Create a new tensor to store the result
    for i, (t, m, s) in enumerate(zip(tensor, mean, std)):
        unnormalized_tensor[i] = t * s + m  # Out-of-place operation

    # Clip the values to be in [0, 1] range
    unnormalized_tensor = torch.clamp(unnormalized_tensor, 0, 1)

    return unnormalized_tensor


def add_gaussian_noise_func(tensor, mean=0.4, std=0.5):
    """Adds noise to a tensor."""
    # Ensure noise is generated on the same device as the tensor
    noise = torch.randn(tensor.size(), device=tensor.device) * std + mean
    return tensor + noise

model.load_state_dict(torch.load(save_model, map_location=torch.device('cpu')))
# tr= net(i.unsqueeze(0))
num_samples = 15


# Plotting
num_rows = num_samples
num_columns = 2  #  Real, Your Prediction, Other Prediction, Output from Unet

fig, axs = plt.subplots(num_rows, num_columns, figsize=(7,3 * num_rows))
fig.subplots_adjust(hspace=0.1, wspace=0.01)  # Adjust the value as needed

model=model.cpu()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
model.eval()
threshold=0.5
for i,inp in enumerate(test_loader):
    im, lab =inp
    out=model(im.cpu())
    # im=add_gaussian_noise_func(im)

    # out = (out * std) + mean
    if "Decoder" in model_selected:
        out=torch.transpose(out,1, 3).view(out.size(3),out.size(2),1)
    else:
        out=torch.transpose(out,1, 3).view(out.size(3),out.size(2),3)
        # out = unnormalize(out, mean, std)
    # out=torch.transpose(out,1, 3)
    out = out.data.cpu().numpy().squeeze()
    out = (out - out.min()) / (out.max() - out.min() )*255

    im = (im - im.min()) / (im.max() - im.min() )

    # # # out=out.detach().numpy()
    # if "Encoder" in model_selected:
    #     out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    # # out = (out - out.min()) / (out.max() - out.min())

    # out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)*255
    out = out.astype(np.uint8)
  #  out = (out - out.min()) / (out.max() - out.min())*256
    # out = (out - out.min()) / (out.max() - out.min())

    lab=torch.transpose(im,1,3)
    # lab = unnormalize(lab, mean, std)
    if "Encoder" in model_selected:
        lab=lab.view(lab.size(1),lab.size(2),lab.size(3))
        
    else:
        lab=lab.view(lab.size(1),lab.size(2),1)

    lab=lab.detach().numpy()
    lab = cv2.cvtColor(lab, cv2.COLOR_BGR2RGB)

    lab = cv2.cvtColor(lab, cv2.COLOR_BGR2RGB)*255
    lab = lab.astype(np.uint8)

    # axs[i, 0].imshow(input_images[i], cmap='viridis')
    # axs[i, 0].set_title('Input')
    # cv2.imwrite(join('C:/Users/Orhan/Desktop',"{}.jpg".format(i)),resized_image)
# torch.tensor( cv2.resize(cv2.imread(im_path, cv2.IMREAD_COLOR ), (dim[1],dim[0]), interpolation = cv2.INTER_AREA))
#  torch.tensor(imageio.imread(im_path))
    axs[i, 0].imshow(lab, cmap='viridis')
    axs[i, 0].set_xticks([])  # Remove x-axis ticks
    axs[i, 0].set_yticks([])  # Remove y-axis ticks
    if i==0:
        axs[i, 0].set_title('Ground Truth', fontsize=25, fontweight='bold', ha='center', va='center', fontname='Times New Roman')


    axs[i, 1].imshow(out, cmap='viridis')
    axs[i, 1].set_xticks([])  # Remove x-axis ticks
    axs[i, 1].set_yticks([])  # Remove y-axis ticks
    if i==0:

        axs[i, 1].set_title('Output', fontsize=25, fontweight='bold', ha='center', va='center', fontname='Times New Roman')
    # if i==9:
    #     break


plt.tight_layout()
plt.show()





    
    
# # out.shape
# # Read the image using OpenCV
# image = cv2.imread(im_path, cv2.IMREAD_COLOR)

# # Convert BGR to RGB
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Resize the image
# resized_image = cv2.resize(image_rgb, (dim[1], dim[0]), interpolation=cv2.INTER_AREA)
# cv2.imwrite(join('C:/Users/Orhan/Desktop',"{}.jpg".format(i)),resized_image)
# # _,out=net(data[i][np.newaxis].cpu())
# # out=torch.transpose(out,1, 3).reshape(dim[0],dim[1])

# # data[i].shape

# # out.shape

# torch.save(net.state_dict(), save_model)
import torchsummary
input_size= (3, 256, 256)
torchsummary.summary(model.cuda(), input_size=input_size)

from torch.utils.data import DataLoader # Assumin g you use DataLoader
# Import your model definition and dataset
# from your_model_file import YourAutoencoderModel
# from your_dataset_file import YourTestDataset

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
# from sklearn.metrics import mean_squared_error, mean_absolute_error # Alternative for numpy arrays

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_path = 'path/to/your/trained_autoencoder.pth' # Path to your saved model weights
# batch_size = 32 # Or your preferred batch size for evaluation

# --- Load Model ---
# model = YourAutoencoderModel().to(device)
# model.load_state_dict(torch.load(model_path, map_location=device))
# print("Model loaded successfully.")

# --- Load Test Data ---
# test_dataset = YourTestDataset(root='./data', train=False, transform=your_transforms) # Adjust as needed
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# print(f"Test dataset loaded with {len(test_dataset)} images.")

# --- Placeholder for demonstration if you don't have model/data loaded ---
# Example: Create a dummy model and data loader
def denormalize_manual(tensor, mean, std):
    """
    Denormalizes a tensor image with mean and standard deviation.

    Args:
    tensor (torch.Tensor): Normalized tensor image, expected shape (..., C, H, W)
    mean (list or tuple): Mean values for each channel.
    std (list or tuple): Standard deviation values for each channel.

    Returns:
    torch.Tensor: Denormalized tensor image.
    """
    # Clone to avoid modifying the original tensor in place
    denormalized_tensor = tensor.clone()

    # Convert mean and std to tensors and reshape for broadcasting
    # Ensure they are on the same device as the tensor
    device = denormalized_tensor.device
    mean = torch.tensor(mean, device=device) # Shape [C]
    std = torch.tensor(std, device=device)   # Shape [C]

    # Reshape mean and std to (1, C, 1, 1) to broadcast across N, H, and W
    # This assumes tensor has shape (N, C, H, W).
    # If tensor can be (C, H, W), this needs adjustment, but (1, C, 1, 1)
    # usually broadcasts correctly to (C, H, W) as well.
    view_shape = [1, -1, 1, 1] # Target shape (1, C, 1, 1)
    # Adjust view_shape if tensor dimensions are different, e.g., for (C, H, W) use [-1, 1, 1]
    if len(tensor.shape) == 3: # Shape (C, H, W)
         view_shape = [-1, 1, 1]
    elif len(tensor.shape) != 4: # Handle unexpected shapes
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")


    mean = mean.view(view_shape)
    std = std.view(view_shape)

    # Apply the denormalization formula: tensor * std + mean
    # Perform out-of-place first to avoid potential inplace operation issues
    denormalized_tensor = denormalized_tensor * std + mean
    # Or using inplace if preferred, now that shapes match:
    # denormalized_tensor.mul_(std).add_(mean)
    return denormalized_tensor


model=model.cpu()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
model.eval()
threshold=0.5
pnsr_val=[]
ssim_val=[]
mse_val=[]

pnsr_org=[]
ssim_org=[]
mse_org=[]
model=model.cuda()

def add_gaussian_noise_func(tensor, mean=0.0, std=0.5):
    """Adds noise to a tensor."""
    # Ensure noise is generated on the same device as the tensor
    noise = torch.randn(tensor.size(), device=tensor.device) * std + mean
    return tensor + noise

channel_axis_val = -1
for i,inp in enumerate(test_loader):
    im, lab =inp
    im=im.cuda()
    # im=add_gaussian_noise_func(im)
    lab=lab.cuda()
    # break
    out=model(im)
    # # out = (out * std) + mean
    im=denormalize_manual(im,mean,std)
    im = (im - im.min()) / (im.max() - im.min() )

    mse= F.mse_loss(lab,im,   reduction='none').mean()
    mse_org.append(mse.cpu().numpy())
    out = (out - out.min()) / (out.max() - out.min() )

    mse= F.mse_loss(lab,out  ,   reduction='none').mean()
    mse_val.append(mse.cpu().detach().numpy())
    
       # out = unnormalize(out, mean, std)
    # out=torch.transpose(out,1, 3)
    lab=torch.transpose(lab,1, 3)
    im=torch.transpose(im,1, 3)
    out=torch.transpose(out,1, 3)

    # out = out.data.cpu().numpy().squeeze()
    im=im.data.cpu().numpy().squeeze()
    lab=lab.data.cpu().numpy().squeeze()
    out=out.data.cpu().numpy().squeeze()

    pnsr_val.append(psnr(lab,out, data_range=1.0))
    ssim_val.append(ssim(lab, out, data_range=1.0, channel_axis=channel_axis_val, win_size=7)) # Adjust win_size if needed

    pnsr_org.append(psnr(lab,im, data_range=1.0))
    ssim_org.append(ssim(lab, im, data_range=1.0, channel_axis=channel_axis_val, win_size=7)) # Adjust win_size if needed

    
avg_ssim = np.mean(ssim_val)
avg_psnr = np.mean(pnsr_val)
avg_mse = np.mean(mse_val)

avg_mse_org = np.mean(mse_org)
avg_ssim_org = np.mean(ssim_org)
avg_psnr_org = np.mean(pnsr_org)
print("Using dummy model and data for demonstration.")
# --- End Placeholder ---


# --- Evaluation Function ---
def evaluate_autoencoder(model, dataloader, device):
    model.eval()  # Set model to evaluation mode
    all_psnr = []
    all_ssim = []
    all_mse = []
    all_mae = []

    with torch.no_grad(): # Disable gradient calculations
        for batch_idx, (inputs, _) in enumerate(dataloader): # We don't need labels for AE eval
            # inputs = inputs.to(device)

            # Get reconstructions
            reconstructions = model(inputs)

            # --- Calculate Metrics ---
            # Move tensors to CPU and convert to NumPy for skimage
            # Ensure data range is correct for metrics (e.g., 0-1 or 0-255)
            # Assuming inputs/reconstructions are floats in [0, 1] range
            inputs_np = inputs.cpu().numpy()
            reconstructions_np = reconstructions.cpu().detach().numpy()

            # Important: skimage metrics often expect images as (H, W, C) or (H, W)
            # PyTorch tensors are often (N, C, H, W). Adjust accordingly.
            # If grayscale (N, 1, H, W), you might need squeeze().
            # If color (N, 3, H, W), you need transpose(0, 2, 3, 1).

            # Ensure clamping if model output might exceed expected range [0,1]
            reconstructions_np = np.clip(reconstructions_np, 0, 1)

            batch_size_current = inputs.size(0)
            for i in range(batch_size_current):
                original_img = inputs_np[i]
                reconstructed_img = reconstructions_np[i]

                # Handle channel dimension for skimage
                if original_img.shape[0] == 1: # Grayscale (C, H, W) -> (H, W)
                    original_img_sk = np.squeeze(original_img, axis=0)
                    reconstructed_img_sk = np.squeeze(reconstructed_img, axis=0)
                    channel_axis_val = None # No channel axis for grayscale ssim
                elif original_img.shape[0] == 3: # Color (C, H, W) -> (H, W, C)
                     original_img_sk = np.transpose(original_img, (1, 2, 0))
                     reconstructed_img_sk = np.transpose(reconstructed_img, (1, 2, 0))
                     channel_axis_val = -1 # Channel axis is the last one
                else:
                     # Handle other cases or raise error
                     raise ValueError(f"Unsupported image shape: {original_img.shape}")

                # --- PSNR ---
                # data_range is the max possible pixel value (1.0 for float [0,1])
                psnr_val = psnr(original_img_sk, reconstructed_img_sk, data_range=1.0)
                all_psnr.append(psnr_val)

                # --- SSIM ---
                # Also needs data_range. Specify channel_axis for multichannel (color) images.
                # win_size might need adjustment based on image size (must be odd, <= min(H,W))
                # Default win_size=7 is usually fine unless images are very small.
                ssim_val = ssim(original_img_sk, reconstructed_img_sk, data_range=1.0, channel_axis=channel_axis_val, win_size=7) # Adjust win_size if needed
                all_ssim.append(ssim_val)

            # --- MSE & MAE (using PyTorch is efficient) ---
            # Calculate per-image loss, then average later
            mse_batch = F.mse_loss(reconstructions.view(batch_size_current, -1),
                                   inputs.view(batch_size_current, -1),
                                   reduction='none').mean(dim=1) # Mean over pixels for each image
            mae_batch = F.l1_loss(reconstructions.view(batch_size_current, -1),
                                  inputs.view(batch_size_current, -1),
                                  reduction='none').mean(dim=1) # Mean over pixels for each image

            all_mse.extend(mse_batch.cpu().tolist())
            all_mae.extend(mae_batch.cpu().tolist())

            if batch_idx % 50 == 0: # Print progress
                 print(f"Processed batch {batch_idx+1}/{len(dataloader)}")

    # --- Calculate Average Metrics ---
    avg_psnr = np.mean(all_psnr)
    avg_ssim = np.mean(all_ssim)
    avg_mse = np.mean(all_mse)
    avg_mae = np.mean(all_mae)

    return avg_psnr, avg_ssim, avg_mse, avg_mae

# --- Run Evaluation ---
avg_psnr, avg_ssim, avg_mse, avg_mae = evaluate_autoencoder(model, test_loader, device)

print("\n--- Evaluation Results ---")
print(f"Average PSNR: {avg_psnr:.4f} dB")
print(f"Average SSIM: {avg_ssim:.4f}")
print(f"Average MSE:  {avg_mse:.6f}")
print(f"Average MAE:  {avg_mae:.6f}")

# --- How to include in your paper (same text as before applies) ---
# "To quantitatively evaluate the reconstruction quality after 16x compression,
# we computed standard image fidelity metrics comparing the original inputs to the
# reconstructed outputs from our Encoder-Net on the test dataset. The average
# Peak Signal-to-Noise Ratio (PSNR) was calculated as [Your Avg PSNR value] dB,
# and the average Structural Similarity Index Measure (SSIM) was [Your Avg SSIM value].
# Additionally, the mean squared error (MSE) was [Your Avg MSE value] and the mean
# absolute error (MAE) was [Your Avg MAE value]. These metrics indicate a high degree
# of fidelity in the reconstructed images, supporting the visual evidence in Figures 8-9
# and demonstrating that the encoder effectively preserves salient features despite
# the high compression ratio."
# You could also present these in a table.


# import torch
# from torch.autograd import Variable
# import os
# import argparse
# from datetime import datetime
# from lib.pvt import PolypPVT
# from utils.dataloader import get_loader, test_dataset
# from utils.utils import clip_gradient, adjust_lr, AvgMeter
# import torch.nn.functional as F
# import numpy as np
# import logging
# import torch.nn as nn

# import matplotlib.pyplot as plt

# class EuclideanLoss(nn.Module):
#     def __init__(self):
#         super(EuclideanLoss, self).__init__()

#     def forward(self, predicted, true):
#         # Compute Euclidean loss (L2 loss)
#         loss = torch.mean((predicted - true) ** 2)
#         return loss
    
# class Encoder_x4(nn.Module):
#     def __init__(self):

#         super(Encoder_x4, self).__init__()
#         # self.encoder=nn.Sequential(nn.Conv2d(3,8, 3),nn.ELU(),nn.MaxPool2d(2,2),
#         #                 nn.ELU(),nn.Conv2d(8, 24, 3),nn.ELU(),nn.Conv2d(24, 32, 5,stride=3)
#         #                 ##nn.Conv2d(8, 8, 3)
#         #                 ,nn.ELU(),nn.Conv2d(32, 10, 3,stride=2),nn.ELU(),nn.ConvTranspose2d(10, 10, 3),nn.ELU())
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 8, 3, padding=2),
#             nn.ELU(),
#             nn.Conv2d(8, 24, 3, padding=1, stride=2),
#             nn.ELU(),
#             nn.Conv2d(24, 32, 5, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 18, 3, padding=1, stride=2),
#             nn.Tanh(),
#             nn.Conv2d(18, 3, 3, padding=1),  # Adjusted padding
#             nn.ELU()
#         )

#     #        self.decoder=nn.Sequential(nn.ConvTranspose2d(10, 32, 5,stride=3),nn.ELU(),nn.Conv2d(32, 24, 3),nn.ELU(),nn.Upsample(scale_factor=2, mode='nearest'),
#     # nn.Conv2d(24, 16, 3),nn.ELU(),nn.Upsample(scale_factor=2, mode='nearest'),
#     # nn.Conv2d(16, 8, 3),nn.ELU(), nn.Conv2d(8, 3, 3),nn.ELU())
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(3, 10, 3, padding=1),  # Output: (N, 10, 64, 64)
#             nn.ReLU(),
#             nn.ConvTranspose2d(10, 24, 3, padding=1, stride=2),  # Output: (N, 24, 128, 128)
#             nn.Tanh(),
#             nn.ConvTranspose2d(24, 32, 3, padding=1),  # Output: (N, 32, 128, 128)
#             nn.Tanh(),
#             nn.ConvTranspose2d(32, 32, 3, padding=1),  # Output: (N, 32, 256, 256)
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode='nearest'),
#             nn.ConvTranspose2d(32, 24, 3, padding=1),  # Output: (N, 24, 256, 256)
#             nn.Tanh(),
#             nn.ConvTranspose2d(24, 8, 3, padding=1),  # Output: (N, 18, 256, 256)
#             nn.ReLU(),
#             nn.ConvTranspose2d(8, 3, 3),  # Output: (N, 3, 256, 256)
#             nn.Sigmoid()
#             )

#     def forward(self, inp):
#         x=self.encoder(inp)
#         ## x=self.bridge(x)+x
       
#         ## x=x-self.res(inp)
#       # # x=self.encoder2(x)
       
#         x=self.decoder(x)
       
#       #  # out=self.decoder(x)+F.interpolate(self.pooling(inp), scale_factor=8, mode='nearest')
#         return x
    
# # encoder_path='./pretrained_pth/encoder_x4_CVC-ClinicDB.pt'
# model=Encoder_x4().cuda()

# # for param in in_model.parameters():
# #     param.requires_grad = False  
    
# class Decoder_x4(nn.Module):
#     def __init__(self):

#         super(Decoder_x4, self).__init__()
#         self.encoder_out= nn.Sequential(
#             nn.Conv2d(1, 8, 3, padding=2),
#             nn.ReLU(),
#             nn.Conv2d(8, 24, 3, padding=1, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(24, 32, 5, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 18, 3, padding=1, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(18, 3, 3, padding=1), 
#             nn.Sigmoid()
#         )

#         self.decoder_out = nn.Sequential(
#             nn.ConvTranspose2d(3, 18, 3, padding=1, stride=2),  # Output: (N, 24, 128, 128)
#             nn.ReLU(),
#             nn.ConvTranspose2d(18, 32, 3, padding=1),  # Output: (N, 32, 128, 128)
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, 32, 3, padding=1),  # Output: (N, 32, 256, 256)
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode='nearest'),
#             nn.ConvTranspose2d(32, 24, 3, padding=1),  # Output: (N, 24, 256, 256)
#             nn.ReLU(),
#             nn.ConvTranspose2d(24, 8, 3, padding=1),  # Output: (N, 18, 256, 256)
#             nn.ReLU(),
#             nn.ConvTranspose2d(8, 1, 3),  # Output: (N, 3, 256, 256)
#             nn.Sigmoid()
#             )
#     def forward(self, x):
#         x=self.encoder_out(x)
#         x=self.decoder_out(x)
#         return x
    
# # decoder_path='./pretrained_pth/decoder_x4_CVC-ClinicDB.pt'

# # out_model=Decoder_x4().cuda()
# # save_model = torch.load(decoder_path)
# # model_dict = out_model.state_dict()
# # state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
# # model_dict.update(state_dict)
# # out_model.load_state_dict(model_dict)

