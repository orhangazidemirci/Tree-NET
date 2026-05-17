# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 22:05:17 2024

@author: Orhan
"""

import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from lib.pvt import PolypPVT
from utils.dataloader import test_dataset
import cv2
from lib import models
parameters = {"ORIGINAL": [3,1],"Res_Encoder": [5,3],"Res_Encoder_x4": [3,3],"Encoder": [5,3],"Dual": [10,3],"Encoder_x4": [3,3]}

h_channels=64

backbone_selected="UNET"  #UNET,  NestedUNet, CARANET ,FR_UNET, TransUnet++ ,DuckNet , pvt
    
encoder_selected= "ORIGINAL"  # Encoder, Res_Encoder, Encoder_x4,Res_Encoder_x4, Dual, ORIGINAL
in_channels,out_channels=parameters[encoder_selected]

mode_selected='no' + backbone_selected  #SEG,BS


from Train_UNET import out_model2
def test(res,gt):

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
        
        iou = dice / (2 - dice)
        return dice,iou
    # mIoU=mIoU/num1
    # DSC = DSC / num1

    # return DSC,mIoU

def accuracy(res,gt):
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)

    # Threshold to obtain binary prediction (you can adjust the threshold if needed)
    threshold = 0.5
    res_binary = (res > threshold).astype(np.uint8)
    
    # Ground truth is already normalized; threshold it if required
    gt_binary = (gt > 0.5).astype(np.uint8)
    
    # Calculate segmentation accuracy
    correct_predictions = np.sum(res_binary == gt_binary)
    total_pixels = gt_binary.size
    accuracy = correct_predictions / total_pixels
    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=384, help='testing size')
    # parser.add_argument('--pth_path', type=str, default='./model_pth/Bests/Encoder_x4_87PolypPVT-best.pth') # Encoder_x4_87PolypPVT-best, Original_CVC_66PolypPVT-best, Original_12PolypPVT-best, Treenet_Isic_54PolypPVT-best

    parser.add_argument('--pth_path', type=str, default='./model_pth/UNET/UNET_79PolypPVT-best.pth')# Treenet_UNET_93PolypPVT-best, Treenet_NestedUNET_87PolypPVT-best, Treenet_NestedUNET_Isic_49-best.pth, Treenet_Unet_74Isic-best, Original_Isic_Unet_69-best ,NestedUnet_90ISAC-best, UNET_79PolypPVT-best, Original_NestetUnet_74PolypPVT-best.pth, 98BSUnet-Polyp-best
    # parser.add_argument('--pth_path', type=str, default='./model_pth/UNET/BSUnet_44ISAC-best.pth')# Treenet_UNET_93PolypPVT-best, Treenet_NestedUNET_87PolypPVT-best, Treenet_NestedUNET_Isic_49-best.pth, Treenet_Unet_74Isic-best, Original_Isic_Unet_69-best ,NestedUnet_90ISAC-best, UNET_79PolypPVT-best, Original_NestetUnet_74PolypPVT-best.pth, 98BSUnet-Polyp-best
    opt = parser.parse_args()
    # model = PolypPVT()
    
    if backbone_selected=='UNET':
        model = models.UNET(in_channels, h_channels, out_channels, encoder_selected).cuda() 
    else:
        model = models.NestedUNet(in_channels, h_channels, out_channels, encoder_selected).cuda() 
        
        
    model.load_state_dict(torch.load(opt.pth_path))
    
    # with torch.no_grad():  # Ensure gradients are not tracked for this operation
    #     if model.out_CFM.bias is not None:  # Check if the layer has a bias
    #         print("Original bias:", model.out_CFM.bias)
    #         model.out_CFM.bias += 0.02  # Adjust the bias by adding 0.01
    #         print("Modified bias:", model.out_CFM.bias)
    #     # for i in range(len(model.SAM.conv_proj.bias)):
    #     #     if i%2==0:
    #     #         model.SAM.conv_proj.bias[i]+= 0.1
    #     #     else:
    #     #         model.SAM.conv_proj.bias[i]-= 0.1     
    #     model.SAM.conv_proj.bias+= 1
    #     model.SAM.conv_state.bias+=0.2
    #     # model.CFM.conv_upsample3.bias=True
    #     # model.CFM.conv_upsample3.bias=1
    
    model.cuda()
    model.eval()
    for _data_name in ['CVC-ClinicDB']: #ISAC2018 ,CVC-ClinicDB

        ##### put data_path here #####
        data_path = './dataset/TestDataset/{}'.format(_data_name)
        # data_path = './dataset/{}/test'.format(_data_name)

        ##### save_path #####
        save_path = './result_map/NUnet/{}/tree/'.format(_data_name)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        num1 = len(os.listdir(gt_root))
        test_loader = test_dataset(image_root, gt_root, 384)

        mIoU=0.0
        DSC=0.0
        ACC=0.0
        for i in range(num1):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            # P1,P2 = model(image)
            # res=P1+P2
           
            # res=out_model2(model(image))
            res=(model(image))

            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            cv2.imwrite(save_path+name, res*255)
            dice,iou=test(res,gt)
            acc= accuracy(res,gt)
            ACC +=acc
            DSC = DSC + dice
            mIoU +=iou
        mIoU=mIoU/num1
        DSC = DSC / num1
        ACC = ACC / num1
        print(_data_name, 'Finish!')
