# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:29:06 2024

@author: Orhan
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


import sys

# sys.path.append('C:/Users/Orhan/Desktop/python/Okul/paper_cifar')
# from fcc import FCC

seed = 42  #seed value. Try 0

def set_random_seed(seed_value):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set random seed for reproducibility
set_random_seed(seed)
# import os
dataset_name='CVC-ClinicDB'  # CVC-ClinicDB , Kvasir

load_model = 'C:/Users/Orhan/Desktop/python/thesis/output_file/save_models/encoder_{}.pt'.format(dataset_name)
load_x4_model = 'C:/Users/Orhan/Desktop/python/thesis/output_file/save_models/encoder_x4_{}.pt'.format(dataset_name)
seg_model = 'C:/Users/Orhan/Desktop/python/thesis/output_file/save_models/seg_{}.pt'.format(dataset_name)
res_path = 'C:/Users/Orhan/Desktop/python/thesis/output_file/save_models/res_encoder_{}.pt'.format(dataset_name)
dual_path = 'C:/Users/Orhan/Desktop/python/thesis/output_file/save_models/dual_{}.pt'.format(dataset_name)

    
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
       
       # x=self.decoder(x)
       
     #  # out=self.decoder(x)+F.interpolate(self.pooling(inp), scale_factor=8, mode='nearest')
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
        # x=self.encoder_out(x)
        x=self.decoder_out(x)
        return x
    

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
        # x=self.encoder_out(x)
        x=self.decoder_out(x)
        return x

    



class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UNET(nn.Module):
    def __init__(self,in_channels,h_channels,out_channels,algorithm_selected):
        super().__init__() 
        if algorithm_selected !='ORIGINAL':
            encoder_path='./model_pth/encoder_x4_ISAC2018.pt'
            # encoder_path='./model_pth/encoder_x4_Polyp.pt'

            self.in_model=Encoder_x4().cuda()
            save_model = torch.load(encoder_path)
            model_dict = self.in_model.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.in_model.load_state_dict(model_dict)
            for param in self.in_model.parameters():
                param.requires_grad = False  
                
        self.algorithm_selected=algorithm_selected
    
        
        nb_filter = [32, 64, 128, 256, 512]
        self.in_channels=in_channels
        self.l1=nn.Parameter(torch.randn(1))
        self.l2=nn.Parameter(torch.randn(1))      
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(in_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        
        # self.btl_net= nn.Conv2d(nb_filter[3],3,3,padding=1)

        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        
        self.btl_net= nn.Conv2d(nb_filter[2],3,3,padding=1)
   
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.final = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)


        if algorithm_selected !='ORIGINAL':
            # encoder_path='./model_pth/encoder_x4_ISAC2018.pt'
            #decoder_path='./model_pth/decoder_x4_Polyp.pt'
            decoder_path='./model_pth/decoder_x4_ISAC2018.pt'
    
            self.out_model=Decoder_x4().cuda() # We can create 2 out_model for both outputs and train on them
            
            #Also create an encoder out that gets an additional loss
            save_model = torch.load(decoder_path)
            model_dict = self.out_model.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.out_model.load_state_dict(model_dict)
            
            for param in self.out_model.parameters():
                param.requires_grad = False   
            
            # self.relu = nn.ReLU(inplace=True)

            # self.convf = nn.Conv2d(1, 1 ,3,padding=1)


    def forward(self, x):
        if self.algorithm_selected!="ORIGINAL":
            # print(x.shape)

            x=self.in_model(x)
            # print(x.shape)

        
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        
        # btl=self.btl_net(x3_1)
        
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
       
        # btl=self.btl_net(x2_2)

        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        # return btl,output
        # if self.algorithm_selected!="ORIGINAL":
        #     # print(output.shape)
        #     output=self.out_model(output)
        #     # print(output.shape)

        # # output=self.relu(output)
        
        # # output=self.convf(output)
        #     # print(output.shape)

        return output



class UNET(nn.Module):
    def __init__(self,in_channels,h_channels,out_channels,algorithm_selected):
        super().__init__()
        if algorithm_selected !='ORIGINAL':
            encoder_path='./model_pth/encoder_x4_ISAC2018.pt'
            # encoder_path='./model_pth/encoder_x4_Polyp.pt'

            self.in_model=Encoder_x4().cuda()
            save_model = torch.load(encoder_path)
            model_dict = self.in_model.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.in_model.load_state_dict(model_dict)
            for param in self.in_model.parameters():
                param.requires_grad = False  
                
        self.algorithm_selected=algorithm_selected


        nb_filter = [32, 64, 128, 256, 512]
        self.in_channels=in_channels
        self.l1=nn.Parameter(torch.randn(1))
        self.l2=nn.Parameter(torch.randn(1))
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(in_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.btl_net= nn.Conv2d(nb_filter[3],3,3,padding=1)

        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])

        self.btl_net= nn.Conv2d(nb_filter[2],3,3,padding=1)

        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        


        if algorithm_selected !='ORIGINAL':
            # encoder_path='./model_pth/encoder_x4_ISAC2018.pt'
            decoder_path='./model_pth/decoder_x4_Polyp.pt'
            # decoder_path='./model_pth/decoder_x4_ISAC2018.pt'

            self.out_model=Decoder_x4().cuda() # We can create 2 out_model for both outputs and train on them

            #Also create an encoder out that gets an additional loss
            save_model = torch.load(decoder_path)
            model_dict = self.out_model.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.out_model.load_state_dict(model_dict)

            for param in self.out_model.parameters():
                param.requires_grad = False


            self.final = nn.Conv2d(nb_filter[0], 3, kernel_size=1)

        else:
            self.final = nn.Conv2d(nb_filter[0],1, kernel_size=1)


            # self.relu = nn.ReLU(inplace=True)

            # self.convf = nn.Conv2d(1, 1 ,3,padding=1)


    def forward(self, x):

        if self.algorithm_selected!="ORIGINAL":
            # print(output.shape)
            x=self.in_model(x)
            # print(output.shape)

        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))

       # btl=self.btl_net(x3_1)

        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))

        btl=self.btl_net(x2_2)
        btl=self.out_model(btl)
        
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        # return output, btl
        if self.algorithm_selected!="ORIGINAL":
            # print(output.shape)
            output=self.out_model(output)
            # print(output.shape)

        # output=self.relu(output)

        # output=self.convf(output)
            # print(output.shape)

        return output



 ## NESTED 
class NestedUNet(nn.Module):
    def __init__(self,in_channels,h_channels,out_channels,algorithm_selected):
        super().__init__()

        if algorithm_selected !='ORIGINAL':
            encoder_path='./model_pth/encoder_x4_ISAC2018.pt'
            # encoder_path='./model_pth/encoder_x4_Polyp.pt'

            self.in_model=Encoder_x4().cuda()
            save_model = torch.load(encoder_path)
            model_dict = self.in_model.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.in_model.load_state_dict(model_dict)
            for param in self.in_model.parameters():
                param.requires_grad = False  
                
        nb_filter = [h_channels//2, h_channels, h_channels*2, h_channels*4, h_channels*8]

        # self.deep_supervision = deep_supervision
        self.algorithm_selected=algorithm_selected
        self.in_channels=in_channels
        
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.l1=nn.Parameter(torch.randn(1))
        self.l2=nn.Parameter(torch.randn(1))
        self.conv0_0 = VGGBlock(in_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        
        self.btl_net= nn.Conv2d(nb_filter[3],3,3,padding=1)

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        # self.btl_net= nn.Conv2d(nb_filter[2],3,3,padding=1)


        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        # if self.deep_supervision:
        #     self.final1 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        #     self.final2 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        #     self.final3 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        #     self.final4 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        # else:
        self.final = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)

        if algorithm_selected !='ORIGINAL':
            # encoder_path='./model_pth/encoder_x4_ISAC2018.pt'
            decoder_path='./model_pth/decoder_x4_Polyp.pt'
            # decoder_path='./model_pth/decoder_x4_ISAC2018.pt'
    
            self.out_model=Decoder_x4().cuda() # We can create 2 out_model for both outputs and train on them
            
            #Also create an encoder out that gets an additional loss
            save_model = torch.load(decoder_path)
            model_dict = self.out_model.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.out_model.load_state_dict(model_dict)
            
            for param in self.out_model.parameters():
                param.requires_grad = False   
            
            # self.relu = nn.ReLU(inplace=True)

            # self.convf = nn.Conv2d(1, 1 ,3,padding=1)

    def forward(self, x):
        if self.algorithm_selected!="ORIGINAL":
            # print(x.shape)

            x=self.in_model(x)
            
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        
        # btl=self.btl_net(x3_1)
        
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        
        # btl=self.btl_net(x2_2)

        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        # if self.deep_supervision:
        #     output1 = self.final1(x0_1)
        #     output2 = self.final2(x0_2)
        #     output3 = self.final3(x0_3)
        #     output4 = self.final4(x0_4)
        #     return [output1, output2, output3, output4]

        # else:
        output = self.final(x0_4)
        
        if self.algorithm_selected!="ORIGINAL":
            # print(x.shape)

            output=self.out_model(output)
            
        return output
        # return btl,output
 