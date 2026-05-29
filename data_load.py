# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 15:09:37 2024

@author: Orhan
"""
from torch.autograd import Variable

from transformers import AutoModelForImageSegmentation
import torchvision.transforms as T
import random
from torchvision.transforms import functional as F

import datetime
from torch.utils.data import DataLoader , TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import gc
import cv2
import imageio
import numpy as np
import json
#from PIL import Image
#from matplotlib import pyplot as plt
import os
from os import listdir
from os.path import isfile, join
from torchsummary import summary
from torch.cuda.amp import autocast
# import models
from utils.dataloader import get_loader, test_dataset
import argparse


# # Inside your training loop
# #### CONFIGURATIONS
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Set to the GPU index you want to use
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"  # Replace with the indices of your GPUs

# mod=archs.UNet(3)
# Check the number of available GPU devices
num_gpus = torch.cuda.device_count()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
  
if num_gpus > 0:
    print(f"Number of available GPU devices: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPU devices available.")

  
dim = (256, 256)
batch_size=1
h_channels=64
l_rate=0.001

epochs=30 


#Increase the layer of discriminator
#Try it with retraining
#Try less epoch

parameters = {"ORIGINAL": [3,1],"Res_Encoder": [5,3],"Res_Encoder_x4": [3,3],"Encoder": [5,3],"Dual": [10,3],"Encoder_x4": [3,3]}


encoder_selected= "Encoder_x4"  # Encoder, Res_Encoder, Encoder_x4,Res_Encoder_x4, Dual, ORIGINAL
decoder_selected="Decoder_x4"   #Decoder, Decoder_x4

in_channels,out_channels=parameters[encoder_selected]
    



# Assuming `model` is your HFUnetPlusPlus model
# num_classes = 1  # The number of classes you want

# Replace the last layer of the segmentation head
# model.segmentation_head = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=1)
# # Access the configuration
# config = model.config

# # Print the configuration attributes
# print(config)


        
#path='/content/gdrive/MyDrive/thesis/datasets/CVC-ClinicDB'
#path='/content/gdrive/MyDrive/thesis/datasets/CVC-ClinicDB'



bridge_d='./bridge_dataset/data'
bridge_l='./bridge_dataset/labels'
bridge_data_saved=False


class Encoder_x4(nn.Module):
   def __init__(self):

       super(Encoder_x4, self).__init__()
       # self.encoder=nn.Sequential(nn.Conv2d(3,8, 3),nn.ELU(),nn.MaxPool2d(2,2),
       #                 nn.ELU(),nn.Conv2d(8, 24, 3),nn.ELU(),nn.Conv2d(24, 32, 5,stride=3)
       #                 ##nn.Conv2d(8, 8, 3)
       #                 ,nn.ELU(),nn.Conv2d(32, 10, 3,stride=2),nn.ELU(),nn.ConvTranspose2d(10, 10, 3),nn.ELU())
       self.encoder = nn.Sequential(
           nn.Conv2d(3, 8, 3, padding=2),
           nn.ELU(),
           nn.Conv2d(8, 24, 3, padding=1, stride=2),
           nn.ELU(),
           nn.Conv2d(24, 32, 5, padding=1),
           nn.ReLU(),
           nn.Conv2d(32, 18, 3, padding=1, stride=2),
           nn.Tanh(),
           nn.Conv2d(18, 3, 3, padding=1),  # Adjusted padding
           nn.ELU()
       )

   #        self.decoder=nn.Sequential(nn.ConvTranspose2d(10, 32, 5,stride=3),nn.ELU(),nn.Conv2d(32, 24, 3),nn.ELU(),nn.Upsample(scale_factor=2, mode='nearest'),
   # nn.Conv2d(24, 16, 3),nn.ELU(),nn.Upsample(scale_factor=2, mode='nearest'),
   # nn.Conv2d(16, 8, 3),nn.ELU(), nn.Conv2d(8, 3, 3),nn.ELU())
       self.decoder = nn.Sequential(
           nn.ConvTranspose2d(3, 10, 3, padding=1),  # Output: (N, 10, 64, 64)
           nn.ReLU(),
           nn.ConvTranspose2d(10, 24, 3, padding=1, stride=2),  # Output: (N, 24, 128, 128)
           nn.Tanh(),
           nn.ConvTranspose2d(24, 32, 3, padding=1),  # Output: (N, 32, 128, 128)
           nn.Tanh(),
           nn.ConvTranspose2d(32, 32, 3, padding=1),  # Output: (N, 32, 256, 256)
           nn.ReLU(),
           nn.Upsample(scale_factor=2, mode='nearest'),
           nn.ConvTranspose2d(32, 24, 3, padding=1),  # Output: (N, 24, 256, 256)
           nn.Tanh(),
           nn.ConvTranspose2d(24, 8, 3, padding=1),  # Output: (N, 18, 256, 256)
           nn.ReLU(),
           nn.ConvTranspose2d(8, 3, 3),  # Output: (N, 3, 256, 256)
           nn.Sigmoid()
           )

   def forward(self, inp):
       x=self.encoder(inp)
       ## x=self.bridge(x)+x
       
       ## x=x-self.res(inp)
      # # x=self.encoder2(x)
       
       # x=self.decoder(x)
       
     #  # out=self.decoder(x)+F.interpolate(self.pooling(inp), scale_factor=8, mode='nearest')
       return x
    
encoder_path='./pretrained_pth/encoder_x4_CVC-ClinicDB.pt'
in_model=Encoder_x4().cuda()
save_model = torch.load(encoder_path)
model_dict = in_model.state_dict()
state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
model_dict.update(state_dict)
in_model.load_state_dict(model_dict)
for param in in_model.parameters():
    param.requires_grad = False  
    
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
        # x=self.decoder_out(x)
        return x
    
decoder_path='./pretrained_pth/decoder_x4_CVC-ClinicDB.pt'


out_model=Decoder_x4().cuda()
save_model = torch.load(decoder_path)
model_dict = out_model.state_dict()
state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
model_dict.update(state_dict)
out_model.load_state_dict(model_dict)


dict_plot = {'CVC-300':[], 'CVC-ClinicDB':[], 'Kvasir':[], 'CVC-ColonDB':[], 'ETIS-LaribPolypDB':[], 'test':[]}
name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test']
##################model_name#############################
model_name = 'PolypPVT'
###############################################
parser = argparse.ArgumentParser()

parser.add_argument('--epoch', type=int,
                    default=100, help='epoch number')

parser.add_argument('--lr', type=float,
                    default=1e-4, help='learning rate')

parser.add_argument('--optimizer', type=str,
                    default='AdamW', help='choosing optimizer AdamW or SGD')

parser.add_argument('--augmentation',
                    default=False, help='choose to do random flip rotation')

parser.add_argument('--batchsize', type=int,
                    default=8, help='training batch size')

parser.add_argument('--trainsize', type=int,
                    default=256, help='training dataset size')

parser.add_argument('--clip', type=float,
                    default=0.5, help='gradient clipping margin')

parser.add_argument('--decay_rate', type=float,
                    default=0.1, help='decay rate of learning rate')

parser.add_argument('--decay_epoch', type=int,
                    default=50, help='every n epochs decay learning rate')

parser.add_argument('--train_path', type=str,
                    default='./dataset/TrainDataset/',
                    help='path to train dataset')

parser.add_argument('--test_path', type=str,
                    default='./dataset/TestDataset/',
                    help='path to testing Kvasir dataset')

parser.add_argument('--train_save', type=str,
                    default='./model_pth/'+model_name+'/')

opt = parser.parse_args()

image_root = '{}/images/'.format(opt.train_path)
gt_root = '{}/masks/'.format(opt.train_path)


train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize, args=opt
                          augmentation=opt.augmentation)
out_model.eval()
in_model.eval()
# count=0
for i,pack in enumerate(train_loader):
    images, gts = pack
    images = Variable(images).cuda()
    gts = Variable(gts).cuda()

    out=out_model(gts)

    torch.save(out, join(bridge_l, str(i)))
    
    # label=label.cpu()
      # cv2.imwrite(join(bridge_l,values_2d[n][0]), lab[n].reshape(400,400).detach().numpy())
      # print('Image ',dataname[n])
    
    
      # tr=torch.transpose(tr, 1, 3)
      # tr=tr.cuda()
    inp=in_model(images)
    #  inp = torch.cat((inp[0].reshape(in_channels,out.size(1),out.size(2))), inp[1].reshape(in_channels,out.size(1),out.size(2))), dim=0)
       # if  "BRIDGE"==algorithm_selected:
       #   inp = torch.cat((inp[0].view(in_channels,inp[0].size(2),inp[0].size(3)), inp[1].view(in_channels,inp[1].size(2),inp[0].size(3))), dim=0)
    
    # in_out[n]=inp
    torch.save(inp, join(bridge_d,str(i))) 
    
    
import os
import torch
from torch.utils.data import DataLoader, TensorDataset

# Directory containing the batch files

# Initialize lists to hold all the loaded data
all_inputs = []
all_labels = []

# Iterate through the directory to load each batch file
for batch_file in os.listdir(bridge_d):
    # if batch_file.endswith('.pt'):  # Only load .pt files (if you're using torch tensors)
    full_path = os.path.join(bridge_d, batch_file)
    
    # Load the batch (assume each file contains a tuple (inputs, labels))
    inputs = torch.load(full_path)
    
    # Append to the lists
    all_inputs.append(inputs)
        # all_labels.append(labels)

# Combine all batches into a single tensor for inputs and labels
for batch_file in os.listdir(bridge_l):
    # if batch_file.endswith('.pt'):  # Only load .pt files (if you're using torch tensors)
    full_path = os.path.join(bridge_l, batch_file)
    
    # Load the batch (assume each file contains a tuple (inputs, labels))
    labels = torch.load(full_path)
    
    # Append to the lists
    all_labels.append(labels)
all_inputs_combined = torch.cat(all_inputs, dim=0)
all_labels_combined = torch.cat(all_labels, dim=0)

    # Create a TensorDataset from combined data
combined_dataset = TensorDataset(all_inputs_combined, all_labels_combined)

# Create a DataLoader from the combined dataset
trainloader = DataLoader(combined_dataset, batch_size=8, shuffle=True)  # Adjust batch_size as needed

    
def test(model, path, dataset):

    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 256)
    DSC = 0.0
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res, res1  = model(image)
        # eval Dice
        res = F.upsample(res + res1 , size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice




    
def test_data(bridge_d,bridge_l):
    all_inputs = []
    all_labels = []

    # Iterate through the directory to load each batch file
    for batch_file in os.listdir(bridge_d):
        # if batch_file.endswith('.pt'):  # Only load .pt files (if you're using torch tensors)
        full_path = os.path.join(bridge_d, batch_file)
        
        # Load the batch (assume each file contains a tuple (inputs, labels))
        inputs = torch.load(full_path)
        
        # Append to the lists
        all_inputs.append(inputs)
            # all_labels.append(labels)

    # Combine all batches into a single tensor for inputs and labels
    for batch_file in os.listdir(bridge_l):
        # if batch_file.endswith('.pt'):  # Only load .pt files (if you're using torch tensors)
        full_path = os.path.join(bridge_l, batch_file)
        
        # Load the batch (assume each file contains a tuple (inputs, labels))
        labels = torch.load(full_path)
        
        # Append to the lists
        all_labels.append(labels)
    all_inputs_combined = torch.cat(all_inputs, dim=0)
    all_labels_combined = torch.cat(all_labels, dim=0)
    combined_dataset = TensorDataset(all_inputs_combined, all_labels_combined)
    
    # Create a DataLoader from the combined dataset
    testloader = DataLoader(combined_dataset, batch_size=8, shuffle=True)  # Adjust batch_size as needed
    return testloader

# test1path = './dataset/TestDataset/'
# for dataset in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
#     bridge_d='./bridge_dataset/test/{}/images'.format(dataset)
#     bridge_l='./bridge_dataset/test/{}/masks'.format(dataset)
    
#     test_loader = test_data(bridge_d, bridge_l)
#     DSC = 0.0
#     for i,pack in enumerate(test_loader):
#         image, gt = pack
#         gt = np.asarray(gt, np.float32)
#         gt /= (gt.max() + 1e-8)
#         image = image.cuda()

#         res, res1  = model(image)
#         # eval Dice
#         res = F.upsample(res + res1 , size=gt.shape, mode='bilinear', align_corners=False)
#         res = res.sigmoid().data.cpu().numpy().squeeze()
#         res = (res - res.min()) / (res.max() - res.min() + 1e-8)
#         input = res
#         target = np.array(gt)
#         N = gt.shape
#         smooth = 1
#         input_flat = np.reshape(input, (-1))
#         target_flat = np.reshape(target, (-1))
#         intersection = (input_flat * target_flat)
#         dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
#         dice = '{:.4f}'.format(dice)
#         dice = float(dice)
#         DSC = DSC + dice

#     return DSC / num1




# def test(model, path, dataset):

#     for dataset in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
#         bridge_d='./bridge_dataset/test/{}/images'.format(dataset)
#         bridge_l='./bridge_dataset/test/{}/masks'.format(dataset)
        
#         test_loader = test_data(bridge_d, bridge_l)
#         DSC = 0.0
#         for i,pack in enumerate(test_loader):
#             image, gt = pack
#             gt = np.asarray(gt.detach().cpu(), np.float32)
#             gt /= (gt.max() + 1e-8)
#             image = image.cuda()
    
#             res, res1  = model(image)
#             # eval Dice
#             target_size = gt.shape[2:]  # This will give us (64, 64)

#             res = F.upsample(res + res1 , size=target_size, mode='bilinear', align_corners=False)
#             res = res.sigmoid().data.cpu().numpy().squeeze()
#             res = (res - res.min()) / (res.max() - res.min() + 1e-8)
#             input = res
#             target = np.array(gt)
#             N = gt.shape
#             smooth = 1
#             input_flat = np.reshape(input, (-1))
#             target_flat = np.reshape(target, (-1))
#             intersection = (input_flat * target_flat)
#             dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
#             dice = '{:.4f}'.format(dice)
#             dice = float(dice)
#             DSC = DSC + dice
    
#         return DSC / len(test_loader)
    
    
    
# dataset_dice = test(model, test1path, dataset)
