# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 14:44:56 2024

@author: Orhan
"""
import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from utils.dataloader import test_dataset
import cv2
import matplotlib.pyplot as plt

def overlay(gt,pred,i, col, key):
    threshold=0.5
    
    pred[pred>threshold]=1
    pred[pred<=threshold]=0
    overlay = np.zeros((384, 384, 3))  # RGB color space for the overlay
    
    # Green: Correct prediction (gt AND pred)
    correct_prediction = (gt == 1) & (pred == 1)
    overlay[correct_prediction] = [0, 1, 0]  # Green
    
    # Yellow: Missed polyps (gt and NOT pred)
    missed_polyp = (gt == 1) & (pred == 0)
    overlay[missed_polyp] = [1, 1, 0]  # Yellow
    
    # Red: Wrong predictions (NOT gt and pred)
    wrong_prediction = (gt == 0) & (pred == 1)
    overlay[wrong_prediction] = [1, 0, 0]  # Red
    
    # Final visualization: use the overlay where it's non-zero
    visualization = image.copy()
    
    # Apply the overlay colors where overlay is non-zero
    non_zero_overlay = np.any(overlay != 0, axis=-1)  # Identify non-zero overlay pixels
    visualization[non_zero_overlay] = overlay[non_zero_overlay]
    axs[i, col].imshow(visualization, cmap='viridis')
    axs[i, col].set_xticks([])  # Remove x-axis ticks
    axs[i, col].set_yticks([])  # Remove y-axis ticks
    if i == 0:
        axs[i,  col].set_title('{}'.format(key), fontsize=25, fontweight='bold', ha='center', va='center', fontname='Times New Roman')


def pred_op(pred):
    pred = np.array(pred, dtype=np.float32)
    pred = cv2.resize(pred, (384, 384), interpolation=cv2.INTER_LINEAR)

    # Normalize the array
    pred = pred / 255.0
    return pred

num_samples = 5
s_start=0
num_columns = 8

fig, axs = plt.subplots(num_samples, num_columns, figsize=(20, 3 * num_samples))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=384, help='testing size')
    # parser.add_argument('--pth_path', type=str, default='./model_pth/Bests/Encoder_x4_87PolypPVT-best.pth') # Encoder_x4_87PolypPVT-best, Original_CVC_66PolypPVT-best, Original_12PolypPVT-best, Treenet_Isic_54PolypPVT-best
    parser.add_argument('--pth_path', type=str, default='./model_pth/UNET/Original_NestetUnet_74PolypPVT-best.pth')# Treenet_UNET_93PolypPVT-best, Treenet_NestedUNET_87PolypPVT-best, Treenet_NestedUNET_Isic_49-best.pth, Treenet_Unet_74Isic-best, Original_Isic_Unet_69-best ,NestedUnet_90ISAC-best, UNET_79PolypPVT-best
    opt = parser.parse_args()
    # model = PolypPVT()
    
   
    
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

    for _data_name in ['CVC-ClinicDB']: #ISAC_2018 ,CVC-ClinicDB

        ##### put data_path here #####
        if _data_name=="CVC-ClinicDB":
            data_path = './dataset/TestDataset/{}'.format(_data_name)
        else:
            data_path = './dataset/ISAC2018/test'

        ##### save_path #####
        
        ##PVT
        
        path_pvt_ours = f'./result_map/PolypPVT/{_data_name}/Tree_NET/'
        path_pvt_org = f'./result_map/PolypPVT/{_data_name}/Original/'

        path_unet_ours = f'./result_map/Unet/{_data_name}/Tree_NET/'
        path_unet_org = f'./result_map/Unet/{_data_name}/Original/'

        path_nunet_ours = f'./result_map/NUnet/{_data_name}/Tree_NET/'
        path_nunet_org = f'./result_map/NUnet/{_data_name}/Original/'

# paths= {
#     'Tree-NET\n(U-NET BB)'      : 'Treenet_UNET_93PolypPVT-best',
#     'Tree-NET\n(U-NET++ BB)'   : 'Treenet_NestedUNET_87PolypPVT-best',
#     'Tree-NET\n(Poyp-PVT BB)'   : 'Encoder_x4_87PolypPVT-best',
#     'UNET\n'                    : ' UNET_79PolypPVT-best',
#   #  'BS\nU-NET'                 : 'BS\nU-NET',
#     'UNET++\n'                  : 'Original_NestetUnet_74PolypPVT',
#     'Poyp-PVT\n'                :'Original_CVC_66PolypPVT-best',
# }
        
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        num1 = len(os.listdir(gt_root))
        test_loader = test_dataset(image_root, gt_root, 384)
        
        test_loader2 = test_dataset(image_root, path_pvt_ours, 384)
        test_loader3 = test_dataset(image_root, path_pvt_org, 384)
        
        test_loader4 = test_dataset(image_root, path_unet_ours, 384)
        test_loader5 = test_dataset(image_root, path_unet_org, 384)
        
        test_loader6 = test_dataset(image_root, path_nunet_ours, 384)
        test_loader7 = test_dataset(image_root, path_nunet_org, 384)

        for n in range(num1):
            image, gt, name = test_loader.load_data()
            _, pred2, name_check = test_loader2.load_data()
            _, pred3, name_check = test_loader3.load_data()
            _, pred4, name_check = test_loader4.load_data()
            _, pred5, name_check = test_loader5.load_data()
            _, pred6, name_check = test_loader6.load_data()
            _, pred7, name_check = test_loader7.load_data()

            if name==name_check and n>=s_start:
                i=n-s_start
                
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)
                gt = cv2.resize(gt, (384, 384), interpolation=cv2.INTER_LINEAR)

                                # Convert the image to a NumPy array

                pred2=pred_op(pred2)
                pred3=pred_op(pred3)
                pred4=pred_op(pred4)
                pred5=pred_op(pred5)
                pred6=pred_op(pred6)
                pred7=pred_op(pred7)
                
                image = np.squeeze(image, axis=0)  # Remove batch dimension -> (3, 384, 384)
                image = np.transpose(image, (1, 2, 0))  # Transpose to (384, 384, 3)

                image = (image - image.min()) / (image.max() - image.min())
                image = np.array(image, dtype=np.float32)
                image = cv2.resize(image, (384, 384), interpolation=cv2.INTER_LINEAR)
                
                axs[i, 0].imshow(image, cmap='viridis')
                axs[i, 0].set_xticks([])  # Remove x-axis ticks
                axs[i, 0].set_yticks([])  # Remove y-axis ticks
                
                if i == 0:
                    axs[i, 0].set_title('Image\n',  fontsize=25, fontweight='bold', ha='center', va='center', fontname='Times New Roman')
                    
                # # Original input image
                # plt.subplot(1, 3, 1)
                # plt.title("Input Image (RGB)")
                # plt.imshow(image)
                # plt.axis("off")

                overlay(gt,gt,i,1,'GT\n')
                
                overlay(gt,pred2,i,2,'Tree-NET\n(Poyp-PVT BB)')
                overlay(gt,pred3,i,3,'Poyp-PVT\n')
                
                overlay(gt,pred4,i,4,'Tree-NET\n(U-NET BB)')
                overlay(gt,pred5,i,5,'UNET\n')
                
                overlay(gt,pred6,i,6,'Tree-NET\n(U-NET++ BB)')
                overlay(gt,pred7,i,7,'UNET++\n')

                # axs[i, col+1].imshow(algorithm_result, cmap='viridis')

            # P1,P2 = model(image)
            # res=P1+P2
           
            # res=out_model2(model(image))
                if i==num_samples-1+s_start:
                    break

model_selected=[ 'Encoder_x4_87PolypPVT-best','Original_CVC_66PolypPVT-best','Original_12PolypPVT-best','Treenet_Isic_54PolypPVT-best'] # Encoder, Res_Encoder, Encoder_x4, Dual, Decoder, Decoder_x4
# Treenet_UNET_93PolypPVT-best, Treenet_NestedUNET_87PolypPVT-best, Treenet_NestedUNET_Isic_49-best.pth, Treenet_Unet_74Isic-best, Original_Isic_Unet_69-best ,NestedUnet_90ISAC-best, UNET_79PolypPVT-best, Original_NestetUnet_74PolypPVT

# model_selected=[  'ORIGINAL_unet','Encoder_x4_nestedunet','Retrained_Encoder_x4_nestedunet', 'ORIGINAL_nestedunet'] # Encoder, Res_Encoder, Encoder_x4, Dual, Decoder, Decoder_x4


# model_selected=['Encoder_x4', 'ORIGINAL'] # Encoder, Res_Encoder, Encoder_x4, Dual, Decoder, Decoder_x4

# model_selected=['Encoder_x4', 'ORIGINAL'] # Encoder, Res_Encoder, Encoder_x4, Dual, Decoder, Decoder_x4
# backbone_selected=['nestedunet','unet']




paths= {
    'Tree-NET\n(U-NET BB)'      : 'Treenet_UNET_93PolypPVT-best',
    'Tree-NET\n(U-NET++ BB)'   : 'Treenet_NestedUNET_87PolypPVT-best',
    'Tree-NET\n(Poyp-PVT BB)'   : 'Encoder_x4_87PolypPVT-best',
    'UNET\n'                    : ' UNET_79PolypPVT-best',
  #  'BS\nU-NET'                 : 'BS\nU-NET',
    'UNET++\n'                  : 'Original_NestetUnet_74PolypPVT',
    'Poyp-PVT\n'                :'Original_CVC_66PolypPVT-best',
}

paths_isic= {
    'Tree-NET\n(U-NET BB)'      : 'Treenet_Unet_74Isic-best',
    'Tree-NET\n(U-NET++ BB)'   : 'Treenet_NestedUNET_Isic_49-best',
    'Tree-NET\n(Poyp-PVT BB)'   : 'Treenet_Isic_54PolypPVT-best',
    'UNET\n'                    : 'Original_Isic_Unet_69-best',
   # 'BS\nU-NET'                 : 'BS\nU-NET',
    'UNET++\n'                  : 'NestedUnet_90ISAC-best',
    'Poyp-PVT\n'                :'Original_12PolypPVT-best',
}

# label_paths=  {}
# # for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
    
# #     path = './result_map/PolypPVT/{}/'.format(_data_name)
# #     label_path= './dataset/TestDataset/{}'.format(_data_name)

# # paths['BS U-NET'] = 'C:/Users/Orhan/Desktop/python/thesis/output_file/save_data/{}_Output/output_bsunet'.format(dataset_name)
# paths['Tree-Net']='./result_map/PolypPVT'
# key_mapping = {
#     'Encoder_x4_unet': 'Tree-NET\n(U-NET BB)',
#     'Retrained_Encoder_x4_unet': 'Tree-NET\n(U-NET BB ST)',
#     'bsunet': 'BS\nU-NET',
#     'ORIGINAL_unet': 'UNET\n',
#     'Encoder_x4_nestedunet':  'Tree-NET\n(U-NET++ BB)',
#     'Retrained_Encoder_x4_nestedunet':'Tree-NET\n(U-NET++ BB ST)',
#     'ORIGINAL_nestedunet': 'UNET++\n'
# }

# paths = {key_mapping.get(k, k): v for k, v in paths.items()}


# for i, n in enumerate(label_res_name):
#     if i==num_samples:
#         break
    
#     lab = cv2.imread(join(label_res, label_res_name[i]))
#     norm_lab = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)
#     lab = normalizing(lab)

#     axs[i, 0].imshow(lab, cmap='viridis')
#     axs[i, 0].set_xticks([])  # Remove x-axis ticks
#     axs[i, 0].set_yticks([])  # Remove y-axis ticks
#     if i == 0:
#         axs[i, 0].set_title('Ground Truth\n',  fontsize=25, fontweight='bold', ha='center', va='center', fontname='Times New Roman')
        
#     for col, (key, path) in enumerate(paths.items()):
#         # if col>0:
#         #     break
        
#         prediction = cv2.imread(join(path, n))
#         algorithm_result = coloring(prediction, norm_lab)
#         prediction = normalizing(prediction)
        
#         # dice_scores[key].append(dice_score)
    
#         dice_scores[key].append(dice_coeff(prediction, lab))

#         if torch.isnan(dice_coeff(prediction, lab)):
#             break
    
#         iou_scores[key].append(iou(prediction, lab))
#         accuracy_scores[key].append(accuracy_score(prediction, lab))
    
#         axs[i, col+1].imshow(algorithm_result, cmap='viridis')
#         axs[i, col+1].set_xticks([])  # Remove x-axis ticks
#         axs[i, col+1].set_yticks([])  # Remove y-axis ticks
#         if i == 0:
#             axs[i,  col+1].set_title('{}'.format(key), fontsize=25, fontweight='bold', ha='center', va='center', fontname='Times New Roman')

# plt.tight_layout()
# plt.show()



# # Simulated example 3D input image in RGB
# # image = np.ones((256, 256, 3)) * 0.5  # Uniform gray RGB background

# # # Synthetic ground truth mask (gt)
# # gt = np.zeros((256, 256))
# # gt[100:150, 100:150] = 1  # Ground truth polyp

# # # # Synthetic prediction mask (pred)
# # # pred = np.zeros((256, 256))
# # pred[110:160, 110:160] = 1  # Predicted polyp

# # Create the overlay mask


#     # # Final visualization
#     # plt.subplot(1, 3, i)
#     # plt.title("Final Visualization")
#     # plt.imshow(visualization)
#     # plt.axis("off")
    
# # # Plot the images
# # plt.figure(figsize=(15, 5))



# # # Overlay
# # plt.subplot(1, 3, 2)
# # plt.title("Overlay")
# # plt.imshow(overlay)
# # plt.axis("off")

# # # Final visualization
# # plt.subplot(1, 3, 3)
# # plt.title("Final Visualization")
# # plt.imshow(visualization)
# # plt.axis("off")

# plt.tight_layout()
# plt.show()

