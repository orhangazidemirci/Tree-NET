# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 02:22:38 2024

@author: Orhan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvtv2 import pvt_v2_b2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


from torchvision import transforms


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
    
       
class Decoder_x8(nn.Module):
    def __init__(self):

        super(Decoder_x8, self).__init__()
        self.encoder_out=nn.Sequential(
              nn.Conv2d(1,8, 3),
              nn.ELU(),
              nn.MaxPool2d(2,2, padding=0, return_indices=False, ceil_mode=False),
              nn.ReLU(),
              nn.Conv2d(8, 24, 3),
              nn.ReLU(),
              nn.Conv2d(24, 32, 5,stride=2),
                  ##nn.Conv2d(8, 8, 3)
              nn.ELU(),nn.Conv2d(32, 18, 3,stride=2),
              nn.ReLU(),
              nn.ConvTranspose2d(18, 3, 3),
              nn.Sigmoid())
   #        self.decoder=nn.Sequential(nn.ConvTranspose2d(10, 32, 5,stride=3),nn.ELU(),nn.Conv2d(32, 24, 3),nn.ELU(),nn.Upsample(scale_factor=2, mode='nearest'),
   # nn.Conv2d(24, 16, 3),nn.ELU(),nn.Upsample(scale_factor=2, mode='nearest'),
   # nn.Conv2d(16, 8, 3),nn.ELU(), nn.Conv2d(8, 3, 3),nn.ELU())
        self.decoder_out=nn.Sequential(
              nn.ConvTranspose2d(3, 24, 3,padding=1),
              nn.ELU(),
              nn.Upsample(scale_factor=2, mode='nearest'),
              nn.ConvTranspose2d(24, 32, 3,padding=1),
              nn.ELU(),
              nn.Upsample(scale_factor=2, mode='nearest'),
              nn.ConvTranspose2d(32, 24, 3,padding=2),
              nn.ReLU(),
              nn.Upsample(scale_factor=2, mode='nearest'),
              nn.ConvTranspose2d(24, 16, 5,padding=1),
              nn.ELU(),
              nn.ConvTranspose2d(16, 8, 3,padding=1),
              nn.ReLU(),
              nn.ConvTranspose2d(8, 1, 3),
              nn.ELU())
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

    
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1 ,groups=1):
        super(BasicConv2d, self).__init__()
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups,bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        # x = self.relu(x)
        x = self.bn(x)
        return x


class CFM(nn.Module):
    def __init__(self, channel):
        super(CFM, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x1 = self.conv4(x3_2)

        return x1




class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        return h


class SAM(nn.Module):
    def __init__(self, num_in=32, plane_mid=16, mids=4, normalize=False):
        super(SAM, self).__init__()

        self.normalize = normalize
        self.num_s = int(plane_mid)
        self.num_n = (mids) * (mids)
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))

        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)

    def forward(self, x, edge):
        edge = F.upsample(edge, (x.size()[-2], x.size()[-1]))

        n, c, h, w = x.size()
        edge = torch.nn.functional.softmax(edge, dim=1)[:, 1, :, :].unsqueeze(1)

        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)
        x_proj = self.conv_proj(x)
        x_mask = x_proj * edge

        x_anchor1 = self.priors(x_mask)
        x_anchor2 = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)
        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)

        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(n, self.num_s, -1))
        x_proj_reshaped = torch.nn.functional.softmax(x_proj_reshaped, dim=1)

        x_rproj_reshaped = x_proj_reshaped

        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))
        x_n_rel = self.gcn(x_n_state)

        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])
        out = x + (self.conv_extend(x_state))

        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class PolypPVT(nn.Module):
    def __init__(self, channel=32):
        super(PolypPVT, self).__init__()

        encoder_path='./model_pth/encoder_x4_Polyp.pt'
        self.in_model=Encoder_x4().cuda()
        save_model = torch.load(encoder_path)
        model_dict = self.in_model.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.in_model.load_state_dict(model_dict)
        for param in self.in_model.parameters():
            param.requires_grad = False  

        self.relu = nn.ReLU()


        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pth/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.uplayer2_0 = BasicConv2d(64 , 64 , 3, padding=1, groups=64 )
        self.uplayer2_1 = BasicConv2d(128, 128, 3, padding=1, groups=128)
        self.uplayer3_1 = BasicConv2d(320, 320, 3, padding=1, groups=320)
        self.uplayer4_1 = BasicConv2d(512, 512, 3, padding=1, groups=512)

        self.Translayer2_0 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)

        self.CFM = CFM(channel)
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()
        self.SAM = SAM()
        
        # hidden_channel=32
        self.up05 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.down05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.out_SAM = nn.Conv2d(channel,3, 1)
        self.out_CFM = nn.Conv2d(channel, 3, 1)
        
        # self.skip_layer_1= BasicConv2d(3, channel, 3 , padding=1)
        # self.skip_layer_2= BasicConv2d(channel, channel, 3 , padding=1)
        # self.skip_layer_3= BasicConv2d(channel, 3, 3 , padding=1)


        self.out_SAM2 = nn.Conv2d(1,3, 3,padding=1)
        self.out_CFM2 = nn.Conv2d(1,3, 3,padding=1)

        self.out_SAM3 = nn.Conv2d(3,1, 3,padding=1)
        self.out_CFM3 = nn.Conv2d(3,1, 3,padding=1)

        # self.out_SAM4 = nn.Conv2d(1,1, 3,padding=1)
        # self.out_CFM4 = nn.Conv2d(1,1, 3,padding=1)




        decoder_path='./model_pth/decoder_x8_Polyp.pt'
        
        self.out_model=Decoder_x8().cuda() # We can create 2 out_model for both outputs and train on them
        
        #Also create an encoder out that gets an additional loss
        save_model = torch.load(decoder_path)
        model_dict = self.out_model.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.out_model.load_state_dict(model_dict)
        
        for param in self.out_model.parameters():
            param.requires_grad = False   

    def forward(self, x):
        x=self.in_model(x)
        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]
        # print(len(pvt))
        # print('x1',x1.shape)
        # print('x2',x2.shape)
        # print('x3',x3.shape)
        # print('x4',x4.shape)

        # CIM
        x1 = self.ca(x1) * x1 # channel attention
        cim_feature = self.sa(x1) * x1 # spatial attention
        # print('x1',x1.shape)
        
        ## UPSAMPLING
        cim_feature=self.up05(cim_feature)
        cim_feature=self.uplayer2_0(cim_feature)
        
        x2=self.up05(x2)
        x2=self.relu(self.uplayer2_1(x2))

        x3=self.up05(x3)
        x3=self.relu(self.uplayer3_1(x3))

        x4=self.up05(x4)
        x4=self.relu(self.uplayer4_1(x4))

    

        # CFM
        x2_t = self.Translayer2_1(x2) 
        x3_t = self.Translayer3_1(x3)  
        x4_t = self.Translayer4_1(x4)
    
        cfm_feature = self.CFM(x4_t, x3_t, x2_t)
        
        # print('cfm_feature',cfm_feature.shape)


        
        # cfm_feature=x+cfm_feature
        

        # SAM
        T2 = self.Translayer2_0(cim_feature)
        # print('T2',T2.shape)

        # T2 = self.down05(T2)
        # print('T2',T2.shape)

        sam_feature = self.SAM(cfm_feature, T2)
        # print('sam_feature',sam_feature.shape)
        # print('-----------------')

        prediction1 = self.out_CFM(cfm_feature)
        prediction2 = self.out_SAM(sam_feature)

        # prediction1_8 = F.interpolate(prediction1, scale_factor=8, mode='bilinear') 
        # prediction2_8 = F.interpolate(prediction2, scale_factor=8, mode='bilinear')  


#### Skip 

        # x = self.skip_layer_1(x)  
        # x = self.skip_layer_2(x)  
        # x = self.skip_layer_3(x)  
        # # x = self.out_model(x)
        
        ### Output 
        prediction1_8 = self.out_model(prediction1) 
        
        prediction1_8 = F.interpolate(prediction1_8, scale_factor=2, mode='bilinear') 
        prediction1_8 = self.out_CFM2(self.relu(prediction1_8))
        
        # prediction1_8 = F.interpolate(prediction1_8, scale_factor=2, mode='bilinear') 
        prediction1_8 = self.out_CFM3(self.relu(prediction1_8))

        # prediction1_8 = F.interpolate(prediction1_8, scale_factor=2, mode='bilinear') 
        # prediction1_8 = self.out_CFM4(self.relu(prediction1_8))
        

        
        prediction2_8 = self.out_model(prediction2) 
        
        prediction2_8 = F.interpolate(prediction2_8, scale_factor=2, mode='bilinear')  
        prediction2_8 = self.out_SAM2(self.relu(prediction2_8))
        
        # prediction2_8 = F.interpolate(prediction2_8, scale_factor=2, mode='bilinear')  
        prediction2_8 = self.out_SAM3(self.relu(prediction2_8))

        # prediction2_8 = F.interpolate(prediction2_8, scale_factor=2, mode='bilinear')  
        # prediction2_8 = self.out_SAM4(self.relu(prediction2_8))


        # # ### Output 
        
        # prediction1_8 = F.interpolate(prediction1, scale_factor=2, mode='bilinear') 
        # prediction1_8 = self.out_CFM2(self.relu(prediction1_8))
        
        # prediction1_8 = F.interpolate(prediction1_8, scale_factor=2, mode='bilinear') 
        # prediction1_8 = self.out_CFM3(self.relu(prediction1_8)+x)

        # prediction1_8 = self.out_model(prediction1_8) 
        # prediction1_8 = self.out_CFM4(self.relu(prediction1_8)) 

        # # prediction1_8 = F.interpolate(prediction1_8, scale_factor=2, mode='bilinear') 
        # # prediction1_8 = self.out_CFM4(self.relu(prediction1_8))
        

        
        
        # prediction2_8 = F.interpolate(prediction2, scale_factor=2, mode='bilinear')  
        # prediction2_8 = self.out_SAM2(self.relu(prediction2_8))
        
        # prediction2_8 = F.interpolate(prediction2_8, scale_factor=2, mode='bilinear')  
        # prediction2_8 = self.out_SAM3(self.relu(prediction2_8)+x)

        # prediction2_8 = self.out_model(prediction2_8) 
        # prediction2_8 = self.out_SAM4(self.relu(prediction2_8)) 




        return prediction1_8, prediction2_8


if __name__ == '__main__':
    model = PolypPVT().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    prediction1, prediction2 = model(input_tensor)
    print(prediction1.size(), prediction2.size())
