# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 00:00:51 2024

@author: Orhan
"""

import torch
import torch.nn as nn
from thop import profile
# from lib.pvt import PolypPVT
from lib import models


parameters = {"ORIGINAL": [3,1],"Res_Encoder": [5,3],"Res_Encoder_x4": [3,3],"Encoder": [5,3],"Dual": [10,3],"Encoder_x4": [3,3]}

h_channels=64

backbone_selected="UNET"  #UNET,  NestedUNet, CARANET ,FR_UNET, TransUnet++ ,DuckNet , pvt
    
encoder_selected= "Encoder_x4"  # Encoder, Res_Encoder, Encoder_x4,Res_Encoder_x4, Dual, ORIGINAL
in_channels,out_channels=parameters[encoder_selected]

mode_selected='BS' + backbone_selected  #SEG,BS


if backbone_selected=='UNET':
    model = models.UNET(in_channels, h_channels, out_channels, encoder_selected).cuda() 
else:
    model = models.NestedUNet(in_channels, h_channels, out_channels, encoder_selected).cuda() 


# model = PolypPVT().cuda() 

# Generate a random input tensor with batch size 1
input_tensor = torch.randn(8, 3, 384, 384).cuda()

# Compute FLOPs and number of parameters
flops, params = profile(model, inputs=(input_tensor,))

# print(f"FLOPs: {flops}")
# print(f"Parameters: {params}")

print(f"{flops}")
print(f"{params}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.cuda.reset_peak_memory_stats(device)
output = model(input_tensor)

# Memory stats
allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

# print(f"Memory Allocated: {allocated:.2f} MB")
# print(f"Memory Reserved: {reserved:.2f} MB")
# print(f"Peak Memory Allocated: {peak_allocated:.2f} MB")

# print(f"{allocated:.2f}")
# print(f"{reserved:.2f}")
print(f"{peak_allocated:.2f}")

