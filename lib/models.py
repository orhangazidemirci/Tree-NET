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

from lib.pvt_v1 import Decoder_x8

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

   
class Encoder_x4(nn.Module):
    def __init__(self, input_channel, bottleneck_channel, inference_mode):
        self.inference_mode=inference_mode
        super(Encoder_x4, self).__init__()
        # self.encoder=nn.Sequential(nn.Conv2d(input_channel,8, 3),nn.ELU(),nn.MaxPool2d(2,2),
        #                 nn.ELU(),nn.Conv2d(8, 24, 3),nn.ELU(),nn.Conv2d(24, 32, 5,stride=3)
        #                 ##nn.Conv2d(8, 8, 3)
        #                 ,nn.ELU(),nn.Conv2d(32, 10, 3,stride=2),nn.ELU(),nn.ConvTranspose2d(10, 10, 3),nn.ELU())
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channel, 8, 3, padding=2),
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
            nn.Conv2d(18, bottleneck_channel, 3, padding=1),  # Adjusted padding
            nn.BatchNorm2d(bottleneck_channel),
            nn.ELU()
        )

    #        self.decoder=nn.Sequential(nn.ConvTranspose2d(10, 32, 5,stride=3),nn.ELU(),nn.Conv2d(32, 24, 3),nn.ELU(),nn.Upsample(scale_factor=2, mode='nearest'),
    # nn.Conv2d(24, 16, 3),nn.ELU(),nn.Upsample(scale_factor=2, mode='nearest'),
    # nn.Conv2d(16, 8, 3),nn.ELU(), nn.Conv2d(8, 3, 3),nn.ELU())
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(bottleneck_channel, 10, 3, padding=1),  # Output: (N, 10, 64, 64)
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
            nn.ConvTranspose2d(8, input_channel, 3),  # Output: (N, 3, 256, 256)
            nn.ELU()
            )

    def forward(self, inp):
        x=self.encoder(inp)
        if not self.inference_mode:
            x=self.decoder(x)
        return x
  
    
class Decoder_x4(nn.Module):
    def __init__(self, input_channel, bottleneck_channel, inference_mode):

        super(Decoder_x4, self).__init__()
        self.inference_mode=inference_mode

        self.encoder_out= nn.Sequential(
            nn.Conv2d(input_channel, 8, 3, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, 24, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 32, 5, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 18, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(18, bottleneck_channel, 3, padding=1), 
            nn.Sigmoid()
        )

        self.decoder_out = nn.Sequential(
            nn.ConvTranspose2d(bottleneck_channel, 18, 3, padding=1, stride=2),  # Output: (N, 24, 128, 128)
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
            nn.ConvTranspose2d(8, input_channel, 3),  # Output: (N, 3, 256, 256)
            nn.Sigmoid()
            )
    def forward(self, x):
        if not self.inference_mode:
            x=self.encoder_out(x)
        
        x=self.decoder_out(x)
        return x
    
    
class Decoder_x16(nn.Module):
    def __init__(self, input_channel, bottleneck_channel, inference_mode):

        super(Decoder_x16, self).__init__()
        self.inference_mode=inference_mode

        self.encoder_out=nn.Sequential(
              nn.Conv2d(input_channel,8, 3),
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
              nn.ConvTranspose2d(24, bottleneck_channel, 3),
              nn.Sigmoid())
   #        self.decoder=nn.Sequential(nn.ConvTranspose2d(10, 32, 5,stride=3),nn.ELU(),nn.Conv2d(32, 24, 3),nn.ELU(),nn.Upsample(scale_factor=2, mode='nearest'),
   # nn.Conv2d(24, 16, 3),nn.ELU(),nn.Upsample(scale_factor=2, mode='nearest'),
   # nn.Conv2d(16, 8, 3),nn.ELU(), nn.Conv2d(8, 3, 3),nn.ELU())
        self.decoder_out=nn.Sequential(
              nn.ConvTranspose2d(bottleneck_channel, 24, 3,padding=1),
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
              nn.ConvTranspose2d(8, input_channel, 3),
              nn.Sigmoid())

    def forward(self, x):
        if not self.inference_mode:
            x = self.encoder_out(x)
        
        x = self.decoder_out(x)
        return x


class Decoder_x16(nn.Module):
    def __init__(self, input_channel, bottleneck_channel, inference_mode):

        super(Decoder_x16, self).__init__()
        self.inference_mode = inference_mode
        self.encoder_out=nn.Sequential(
              nn.Conv2d(input_channel,8, 3),
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
              nn.ConvTranspose2d(24, bottleneck_channel, 3),
              nn.Sigmoid())
   #        self.decoder=nn.Sequential(nn.ConvTranspose2d(10, 32, 5,stride=3),nn.ELU(),nn.Conv2d(32, 24, 3),nn.ELU(),nn.Upsample(scale_factor=2, mode='nearest'),
   # nn.Conv2d(24, 16, 3),nn.ELU(),nn.Upsample(scale_factor=2, mode='nearest'),
   # nn.Conv2d(16, 8, 3),nn.ELU(), nn.Conv2d(8, 3, 3),nn.ELU())
        self.decoder_out=nn.Sequential(
              nn.ConvTranspose2d(bottleneck_channel, 24, 3,padding=1),
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
              nn.ConvTranspose2d(8, input_channel, 3),
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
        if not self.inference_mode:
            x = self.encoder_out(x)
        
        x = self.decoder_out(x)
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
    def __init__(self,in_channels,h_channels,out_channels,args):
        super().__init__()
        self.args=args
        self.treenet=args.treenet
        if args.treenet:
            # encoder_path='./model_pth/encoder_x4_ISAC2018.pt'
            encoder_path=args.encoder_path
            # encoder_path='./model_pth/encoder_x4_Polyp.pt'
            component=args.component_selected
            args.component_selected=args.encoder_component
            args.component=args.encoder_path
            self.in_model=call_model(args, encoder_path).cuda()
            args.component_selected=component

            for param in self.in_model.parameters():
                param.requires_grad = False  
            

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
        


        if args.treenet:
            # encoder_path='./model_pth/encoder_x4_ISAC2018.pt'
            decoder_path= args.decoder_path
            # './model_pth/decoder_x4_Polyp.pt'
            # decoder_path='./model_pth/decoder_x4_ISAC2018.pt'
            component=args.component_selected
            args.component_selected=args.decoder_component
            self.out_model=call_model(args, decoder_path).cuda() # We can create 2 out_model for both outputs and train on them
            args.component_selected=component

            for param in self.out_model.parameters():
                param.requires_grad = False



        
        self.final = nn.Conv2d(nb_filter[0],18, kernel_size=1)


            # self.relu = nn.ReLU(inplace=True)

            # self.convf = nn.Conv2d(1, 1 ,3,padding=1)


    def forward(self, x):

        if self.treenet:
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

        # btl=self.btl_net(x2_2)
        # btl=self.out_model(btl)
        
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        # return output, btl
        if self.treenet:
            # print(output.shape)
            output=self.out_model(output)
            # print(output.shape)

        # output=self.relu(output)

        # output=self.convf(output)
            # print(output.shape)

        return output



 ## NESTED 
class NestedUNet(nn.Module):
    def __init__(self,in_channels,h_channels,out_channels,args):
        super().__init__()


        self.args=args
        self.treenet=args.treenet

        if args.treenet:
            # encoder_path='./model_pth/encoder_x4_ISAC2018.pt'
            encoder_path=args.encoder_path

            component=args.component_selected
            args.component_selected=args.encoder_component

            self.in_model=call_model(args, encoder_path).cuda()

            args.component_selected=component


            for param in self.in_model.parameters():
                param.requires_grad = False  
                
        nb_filter = [h_channels//2, h_channels, h_channels*2, h_channels*4, h_channels*8]

        # self.deep_supervision = deep_supervision
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

        if args.treenet:
            # encoder_path='./model_pth/encoder_x4_ISAC2018.pt'
            decoder_path=args.decoder_path
            # './model_pth/decoder_x4_Polyp.pt'
            # decoder_path='./model_pth/decoder_x4_ISAC2018.pt'

            component=args.component_selected
            args.component_selected=args.decoder_component
            
            self.out_model=call_model(args, decoder_path).cuda() # We can create 2 out_model for both outputs and train on them       
            
            args.component_selected=component

            for param in self.out_model.parameters():
                param.requires_grad = False   
            

            # self.relu = nn.ReLU(inplace=True)

            # self.convf = nn.Conv2d(1, 1 ,3,padding=1)
        self.final = nn.Conv2d(nb_filter[0], 18, kernel_size=1)

    def forward(self, x):
        if self.treenet:
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
        
        if self.treenet:
            # print(x.shape)
            output=self.out_model(output)
            
        return output
        # return btl,output





def call_model(args,path=None):
    input_channel=3
    print("Selected component is:", args.component_selected)
    if args.component_selected=='ORIGINAL':
        args.inference_mode=True
        if args.arch=="pvt":
            args.decoder_component= 'Decoder_x16'

            from lib.pvt import PolypPVT
            model=PolypPVT(args).cuda()
        else:
            parameters = {"ORIGINAL": [3,1],"Res_Encoder": [5,3],"Res_Encoder_x4": [3,3],"Encoder": [5,3],"Dual": [10,3],"Encoder_x4": [3,18]}

            h_channels=64

            backbone_selected="UNET"  #UNET,  NestedUNet, CARANET ,FR_UNET, TransUnet++ ,DuckNet , pvt
            
            if args.treenet:
                args.decoder_component= 'Decoder_x4'
                component_selected= "Encoder_x4"  # Encoder, Res_Encoder, Encoder_x4,Res_Encoder_x4, Dual, ORIGINAL
            else:
                component_selected=args.component_selected
            input_channels,out_channels=parameters[component_selected]

            mode_selected='no' + backbone_selected  #SEG,BS

            if args.arch=='UNET':
                model=UNET(input_channels,h_channels, out_channels,args).cuda()
            elif args.arch=='NestedUNET':
                model=NestedUNet(input_channels,h_channels, out_channels,args).cuda()


    else:
        if args.component_selected=='Encoder_x4':
            model=Encoder_x4(input_channel,args.bottleneck_size[args.component_selected], args.inference_mode).cuda()
        else:
            input_channel=1
            if args.component_selected=='Decoder_x4':
                model=Decoder_x4(input_channel,args.bottleneck_size[args.component_selected], args.inference_mode).cuda()
            if args.component_selected=='Decoder_x8':
                model=Decoder_x8(input_channel,args.bottleneck_size[args.component_selected], args.inference_mode).cuda()
            if args.component_selected=='Decoder_x16':
                model=Decoder_x16(input_channel,args.bottleneck_size[args.component_selected], args.inference_mode).cuda()
        
  
  # if args.is_pretrain==True:
    if path is not None:
        save_model = torch.load(path)
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    #     model.load_state_dict(torch.load(args.pretrained_path))

    return model


if __name__ == "__main__":
    dummo_input = torch.randn(1, 1, 256, 256)  # Example input tensor
    x=Encoder_x4(input_channel=1, bottleneck_channel=16, inference_mode=True)
    output = x(dummo_input)
    print(output.shape)  # Should be (1, 1, 256, 256

    import torchsummary
    torchsummary.summary(x.cuda(), (1, 256, 256))      


    x=Decoder_x4(input_channel=1, bottleneck_channel=16, inference_mode=True)
    output = x(dummo_input)     
    print(output.shape)  # Should be (1, 1, 256, 256)
    torchsummary.summary(x.cuda(), (1, 256, 256))

    x=Decoder_x16(input_channel=1, bottleneck_channel=16, inference_mode=True)
    output = x(dummo_input)
    print(output.shape)  # Should be (1, 1, 256,
    torchsummary.summary(x.cuda(), (1, 256, 256))
    