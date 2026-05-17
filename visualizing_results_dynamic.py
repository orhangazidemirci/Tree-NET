# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 18:03:35 2023
Despite the accuracy and performance superiority of Tree-Net model compared with other approaches, there are several limitations and challenges in implementation of the algorithm.

One limitation is in creating the comparative testing environment. Since it is a novel approach, there are no pre-trained models to be fine-tuned in evaluation. Thus, it makes it harder to compare the algorithm with the pre-trained models.
 
One challenge is tuning the component parameters and layers well. As both the Encoder-net and Bridge-net shrinks the size, the bottleneck of Bridge-net could become quite small that most of the learning parameters might become useless. If autoencoders are designed with lesser downsizing, there will be a worse performance.

The complexity of algorithm is another challenge while designing the algorithm. Since all the components are dependent on each other, applying a change on one could require a restoration in the other components. 

--

In conclusion, the implementation of Tree-Net enhanced by self-tuning approach applied in polyp and skin cancer segmentation has demonstrated remarkable advancements over traditional methods, showcasing superior efficiency, accuracy and robustness. By integrating this algorithm, significant improvements have been achieved while maintaining a lower computational cost compared to conventional techniques. This approach harnesses the power of deep learning to effectively capture multi-scale contextual information, enabling precise and efficient polyp and skin cancer segmentation. 
This algorithm could be applied to all types of image segmentation tasks with the promise hope due to the robustness in Bridge-Net layer type. Any sort of segmentation algorithm could be used as Bridge-Net which will reduce the cost of the algorithm.

As a result, bottleneck feature supervision stands out as a promising strategy for enhancing segmentation tasks, particularly in medical imaging, by achieving unprecedented levels of performance and accuracy.

@author: odemirci
"""

from os import listdir
from os.path import isfile, join
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from torch import Tensor

def numeric_score(predictions, labels):
    predictions = predictions.numpy() if isinstance(predictions, torch.Tensor) else predictions
    labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels

    # Ensure predictions and labels are binary (0 or 1)
    predictions = np.round(predictions).astype(int)
    labels = np.round(labels).astype(int)

    # True Positives
    TP = np.sum((predictions == 1) & (labels == 1))
    # True Negatives
    TN = np.sum((predictions == 0) & (labels == 0))
    # False Positives
    FP = np.sum((predictions == 1) & (labels == 0))
    # False Negatives
    FN = np.sum((predictions == 0) & (labels == 1))

    return FP, FN, TP, TN

def iou_score(predictions, labels):
    FP, FN, TP, TN = numeric_score(predictions, labels)

    return TP / (TP + FP + FN)

def accuracy_score(predictions, labels):
    """Getting the accuracy of the model"""

    FP, FN, TP, TN = numeric_score(predictions, labels)
    N = FP + FN + TP + TN
    if N == 0:
        return 0  # To avoid division by zero
    else:
        return (TP + TN) / N


# # Example usage
# predictions = torch.tensor([0.1, 0.4, 0.35, 0.8])
# labels = torch.tensor([0, 0, 1, 1])



def dice_score(output, target):
    output=torch.tensor(output)
    target=torch.tensor(target)
    intersection = torch.sum(output * target)
    union = torch.sum(output) + torch.sum(target)
    dice = (2. * intersection) / (union + 1e-8)  # Adding a small epsilon to avoid division by zero
    return dice

def iou(output, target):
    output=torch.tensor(output)
    target=torch.tensor(target)
    intersection = torch.sum(output * target)
    union = torch.sum(output) + torch.sum(target) - intersection
    iou = (intersection) / (union + 1e-8)  # Adding a small epsilon to avoid division by zero
    return iou

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    input=torch.tensor(input)
    target=torch.tensor(target)
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def dice_score(pred, target, smooth=1e-6):
    """
    Calculate the Dice score.
    
    Args:
    pred: Predicted binary mask (numpy array).
    target: Ground truth binary mask (numpy array).
    smooth: Small value to avoid division by zero.
    
    Returns:
    Dice score as a float.
    """
    intersection = np.sum(pred * target)
    return (2. * intersection) / (np.sum(pred) + np.sum(target) + smooth)

def iou_score(pred, target, smooth=1e-6):
    """
    Calculate the Intersection over Union (IoU).
    
    Args:
    pred: Predicted binary mask (numpy array).
    target: Ground truth binary mask (numpy array).
    smooth: Small value to avoid division by zero.
    
    Returns:
    IoU score as a float.
    """
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target) - intersection
    return (intersection ) / (union + smooth)



# Sample data (replace with your actual data)
# input_data = np.random.rand(10, 50)  # 10 samples of input with 50 features
# real_values = np.random.rand(10)  # Real values
# your_algorithm_predictions = np.random.rand(10)  # Predictions from your algorithm
# other_algorithm_predictions = np.random.rand(10)  # Predictions from another algorithm

# # Plotting
# label_res='C:/Users/odemirci/Desktop/python/thesis/output_file/label_res'
# out_res='C:/Users/odemirci/Desktop/python/thesis/output_file/out_res'
# out='C:/Users/odemirci/Desktop/python/thesis/output_file/out'
# out_unet='C:/Users/odemirci/Desktop/python/thesis/output_file/Out_Unet/'

# Plotting
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

dataset_name='CVC-ClinicDB'  # CVC-ClinicDB , Kvasir, ISAC2018
 

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

# Define the function to dynamically set the number of columns
def set_num_columns(*args):
    return len(args)

# Example inputs (you can modify these)
backbone_selected='unet'
backbone_selected2='nestedunet'

# model_selected=['Encoder_x4_unet','Retrained_Encoder_x4_unet','bsunet', 'ORIGINAL_unet','Encoder_x4_nestedunet','Retrained_Encoder_x4_nestedunet', 'ORIGINAL_nestedunet'] # Encoder, Res_Encoder, Encoder_x4, Dual, Decoder, Decoder_x4
# model_selected=[ 'Encoder_x4_unet','Retrained_Encoder_x4_unet','Encoder_x4_nestedunet','Retrained_Encoder_x4_nestedunet'] # Encoder, Res_Encoder, Encoder_x4, Dual, Decoder, Decoder_x4
model_selected=[ 'Retrained_Encoder_x4_nestedunet','ORIGINAL_DuckNet','ORIGINAL_unet','ORIGINAL_nestedunet'] # Encoder, Res_Encoder, Encoder_x4, Dual, Decoder, Decoder_x4

# model_selected=[  'ORIGINAL_unet','Encoder_x4_nestedunet','Retrained_Encoder_x4_nestedunet', 'ORIGINAL_nestedunet'] # Encoder, Res_Encoder, Encoder_x4, Dual, Decoder, Decoder_x4


# model_selected=['Encoder_x4', 'ORIGINAL'] # Encoder, Res_Encoder, Encoder_x4, Dual, Decoder, Decoder_x4

# model_selected=['Encoder_x4', 'ORIGINAL'] # Encoder, Res_Encoder, Encoder_x4, Dual, Decoder, Decoder_x4
# backbone_selected=['nestedunet','unet']




label_res='C:/Users/Orhan/Desktop/python/thesis/output_file/save_data/{}_Output/gnd_truth_original_unet'.format(dataset_name)
paths= {}
label_paths=  {}
# for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
    
#     path = './result_map/PolypPVT/{}/'.format(_data_name)
#     label_path= './dataset/TestDataset/{}'.format(_data_name)

# paths['BS U-NET'] = 'C:/Users/Orhan/Desktop/python/thesis/output_file/save_data/{}_Output/output_bsunet'.format(dataset_name)
paths['Tree-Net']='./result_map/PolypPVT'
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



num_samples =10
num_columns = len(paths.keys()) +1

# fig, axs = plt.subplots(num_samples, num_columns, figsize=(15, 3 * num_samples))

def normalizing(normalized_array):
    threshold=0.5
    if not (normalized_array.max() - normalized_array.min()) == 0:
        normalized_array = (normalized_array - normalized_array.min()) / (normalized_array.max() - normalized_array.min())
        normalized_array[normalized_array >= threshold] = 1
        normalized_array[normalized_array < threshold] = 0
    return normalized_array    

def coloring(inp, normalized_lab):
    threshold=0.5
    normalized_array = cv2.cvtColor(inp, cv2.COLOR_BGR2GRAY)
    if not (normalized_array.max() - normalized_array.min()) == 0:
        normalized_array = (normalized_array - normalized_array.min()) / (normalized_array.max() - normalized_array.min())
        normalized_array[normalized_array >= threshold] = 1
        normalized_array[normalized_array < threshold] = 0

    tt = normalized_array * normalized_lab
    ff = np.logical_and(normalized_lab, np.logical_not(normalized_array))
    ft = np.logical_and(np.logical_not(normalized_lab), normalized_array)
    
    rgb_image = np.stack((tt, ff, ft), axis=-1) * 255
    return rgb_image

# dice_scores = {"Res": [], "no_res": [], "Unet": []}
# iou_scores = {"Res": [], "no_res": [], "Unet": []}
# accuracy_scores = {"Res": [], "no_res": [], "Unet": []}

# dice_scores = {key: [] for key in paths.keys()}
# iou_scores = {key: [] for key in paths.keys()}
# accuracy_scores = {key: [] for key in paths.keys()}

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
        
#        # dice_scores[key].append(dice_score)
    
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



for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
    path = './result_map/PolypPVT/{}/'.format(_data_name)
    label_path= './dataset/TestDataset/{}/masks'.format(_data_name)
    dice_scores = {key: [] for key in paths.keys()}
    iou_scores = {key: [] for key in paths.keys()}
    accuracy_scores = {key: [] for key in paths.keys()}
    label_res_name = [ f for f in listdir(join(label_path)) if isfile(join(label_path,f)) ] 
    inp_name = [ f for f in listdir(join(path)) if isfile(join(path,f)) ] 
    common_items = list(set(inp_name) & set(label_res_name))

    for i, n in enumerate(common_items):
    
        
        lab = cv2.imread(join(label_path, common_items[i]))
        norm_lab = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)
        lab = normalizing(lab)
    
        for col, (key, _) in enumerate(paths.items()):
            # if col>0:
            #     break
            
            prediction = cv2.imread(join(path, common_items[i]))
            lab=cv2.resize(lab, (np.shape(prediction)[1],np.shape(prediction)[0]), interpolation = cv2.INTER_AREA)

            # algorithm_result = coloring(prediction, norm_lab)
            prediction = normalizing(prediction)
            
           # dice_scores[key].append(dice_score)
        
            dice_scores[key].append(dice_score(prediction, lab))
    
            if torch.isnan(dice_coeff(prediction, lab)):
                break
        
            iou_scores[key].append(iou_score(prediction, lab))
            accuracy_scores[key].append(accuracy_score(prediction, lab))
        for col, (key, path) in enumerate(paths.items()):
            # print(f"Dice {key}: {np.mean(dice_scores[key])}")
            print('---------------------')
            print('Dice {}'.format(key),np.mean(dice_scores["{}".format(key)]))
            print('IOU {}'.format(key),np.mean(iou_scores["{}".format(key)]))
            print('Accuracy {}'.format(key),np.mean(accuracy_scores["{}".format(key)]))
        print('---------------------')


# # Perform element-wise XNOR
# tt = normalized_array*normalized_lab
# ff =np.logical_and(normalized_lab, np.logical_not(normalized_array))
# ft= np.logical_and(np.logical_not(normalized_lab), normalized_array)

# # Create an RGB image
# rgb_image = np.stack((tt, ff, ft), axis=-1)*256 



     
# # Plot the RGB image
# plt.figure(figsize=(8, 4))


# plt.imshow(rgb_image, aspect='auto')
# plt.title('Combined Visualization of Ground Truth and Prediction')
# plt.xlabel('Index')
# plt.ylabel('Color Channels')
# plt.show()