# -*- coding: utf-8 -*-
"""
Created on Fri May 23 03:53:11 2025

@author: Orhan
"""

import os
import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare
import cv2 # Using OpenCV for image loading and basic operations
from sklearn.metrics import jaccard_score # For IoU (Jaccard)
# You might need: pip install opencv-python scikit-learn
from utils.dataloader import test_dataset

# ---------------------------------------------------------------------------
# 0. CONFIGURATION
# ---------------------------------------------------------------------------
BASE_OUTPUT_DIR = r"Tree-NET\result_map" # REPLACE with the path to NUnet, PolypPVT, Unet
# Define how to find ground truth images.
# Option 1: GT is in a parallel structure (e.g., a 'gt' folder at the same level as 'NUnet')
# GT_BASE_DIR = r"C:\path\to\your\gt_folder"
# Example: if output is BASE_OUTPUT_DIR/NUnet/CVC-ClinicDB/Original/img1.png
# GT could be GT_BASE_DIR/CVC-ClinicDB/gt/img1.png (or similar)

# Option 2: GT path can be constructed or is fixed for each dataset
# For simplicity, let's assume for now a function will get the GT path based on image_name and dataset
# This needs to be adapted to your actual GT structure
def get_gt_path(dataset_name, image_filename):
    # THIS IS A PLACEHOLDER - MODIFY IT TO MATCH YOUR GT STRUCTURE
    if dataset_name == "CVC-ClinicDB":
        gt_dir = r"Tree-NET\dataset\TestDataset\{}\masks".format(dataset_name) # REPLACE
        # data_path = './dataset/TestDataset/{}'.format(_data_name)
        # data_path = './dataset/{}/test'.format(dataset_name)
    elif dataset_name == "ISAC_2018": # Note: you wrote ISAC, assuming ISIC
        gt_dir = r"Tree-NET\dataset\ISAC2018\test\masks".format(dataset_name) # REPLACE
        # data_path = './dataset/{}/test'.format(dataset_name)

    else:
        raise ValueError(f"Unknown dataset for GT: {dataset_name}")
    return os.path.join(gt_dir, image_filename)


ALPHA = 0.05 # Significance level

# ---------------------------------------------------------------------------
# 1. METRIC CALCULATION FUNCTIONS
# ---------------------------------------------------------------------------
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Computes Dice coefficient."""
    y_true_f = y_true.flatten().astype(np.float32) / 255.0 # Assuming 0-255, convert to 0-1
    y_pred_f = y_pred.flatten().astype(np.float32) / 255.0

    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def iou_score(y_true, y_pred):
    """Computes IoU (Jaccard) score."""
    y_true_f = (y_true.flatten() > 128).astype(np.uint8) # Binarize if not already
    y_pred_f = (y_pred.flatten() > 128).astype(np.uint8)
    if np.sum(y_true_f) == 0 and np.sum(y_pred_f) == 0: # Handle empty masks case
        return 1.0
    return jaccard_score(y_true_f, y_pred_f, average='binary', zero_division=0)

def calculate_metrics_for_image(gt_path, pred_path):
    """Loads GT and prediction, calculates Dice and IoU."""
    try:
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        if gt_img is None:
            print(f"Warning: Could not load GT image: {gt_path}")
            return None, None
        if pred_img is None:
            print(f"Warning: Could not load prediction image: {pred_path}")
            return None, None

        # Ensure masks are binary (0 or 255) - adjust threshold if needed
        _, gt_img_bin = cv2.threshold(gt_img, 127, 255, cv2.THRESH_BINARY)
        _, pred_img_bin = cv2.threshold(pred_img, 127, 255, cv2.THRESH_BINARY)

        dice = dice_coefficient(gt_img_bin, pred_img_bin)
        iou = iou_score(gt_img_bin, pred_img_bin)
        return dice, iou
    except Exception as e:
        print(f"Error processing {pred_path} against {gt_path}: {e}")
        return None, None

# ---------------------------------------------------------------------------
# 2. STATISTICAL TEST FUNCTION (same as before)
# ---------------------------------------------------------------------------
def perform_wilcoxon_test(scores_model1, scores_model2, model1_name, model2_name, metric_name, comparison_group_name, alpha=0.05):
    if not scores_model1 or not scores_model2:
        print(f"Skipping test for {comparison_group_name} - {metric_name}: Empty score list(s).")
        return

    if len(scores_model1) != len(scores_model2):
        print(f"Error: Score lists for {model1_name} and {model2_name} must have the same length for {comparison_group_name}.")
        return
    if len(scores_model1) < 2 : # Wilcoxon needs at least a few pairs
        print(f"Skipping test for {comparison_group_name} - {metric_name}: Not enough data points (N={len(scores_model1)}).")
        return

    try:
        # stat, p_value = wilcoxon(scores_model1, scores_model2, alternative='two-sided', zero_division='wilcox')
        # Inside perform_wilcoxon_test function:
        # stat, p_value = wilcoxon(scores_model1, scores_model2, alternative='two-sided', zero_division='wilcox') # OLD
        stat, p_value = wilcoxon(scores_model1, scores_model2, alternative='two-sided') # NEW for older SciPy
        print(f"\n--- Wilcoxon Signed-Rank Test ({comparison_group_name}) ---")
        print(f"Metric: {metric_name}")
        print(f"Comparing: {model1_name} (N={len(scores_model1)}) vs. {model2_name} (N={len(scores_model2)})")
        print(f"  Mean Scores: {model1_name}={np.mean(scores_model1):.4f}, {model2_name}={np.mean(scores_model2):.4f}")
        print(f"  Median Scores: {model1_name}={np.median(scores_model1):.4f}, {model2_name}={np.median(scores_model2):.4f}")
        print(f"  Wilcoxon statistic: {stat:.4f}")
        print(f"  P-value (two-sided): {p_value:.4f}")

        if p_value < alpha:
            print(f"  Result: Statistically significant difference at alpha = {alpha} (p < {alpha}).")
            if np.median(scores_model2) > np.median(scores_model1):
                 print(f"    Indication: {model2_name} performs significantly better (based on median).")
            elif np.median(scores_model2) < np.median(scores_model1):
                 print(f"    Indication: {model1_name} performs significantly better (based on median).")
            else:
                 print(f"    Indication: Significant difference, but medians are equal (check distributions/means).")
        else:
            print(f"  Result: No statistically significant difference at alpha = {alpha} (p >= {alpha}).")

    except ValueError as e:
        print(f"Error performing Wilcoxon test for {model1_name} vs {model2_name} in {comparison_group_name}: {e}")


def perform_friedman_test(all_scores_map, metric_name, comparison_group_name, alpha=0.05):
    """
    Performs Friedman test on multiple related groups.
    Args:
        all_scores_map (dict): A dictionary where keys are model names and
                               values are lists of scores for that model.
                               Example: {'ModelA': [s1,s2,..], 'ModelB': [s1,s2,..]}
        metric_name (str): Name of the metric.
        comparison_group_name (str): Name of the overall comparison group.
        alpha (float): Significance level.
    Returns:
        bool: True if a significant difference is found, False otherwise.
    """
    model_names = list(all_scores_map.keys())
    if len(model_names) < 3:
        # print(f"Skipping Friedman test for {comparison_group_name} - {metric_name}: Needs at least 3 groups, got {len(model_names)}.")
        return False # Friedman needs at least 3 groups

    # Ensure all score lists have the same length
    score_lengths = [len(scores) for scores in all_scores_map.values()]
    if len(set(score_lengths)) > 1:
        print(f"Error: Friedman score lists for {comparison_group_name} - {metric_name} have inconsistent lengths: {score_lengths}")
        return False
    if score_lengths[0] < 2: # Need at least a few observations
        print(f"Skipping Friedman for {comparison_group_name} - {metric_name}: Not enough data points (N={score_lengths[0]}) per group.")
        return False


    # Friedman test expects samples as separate arguments *args
    scores_for_friedman = [np.array(all_scores_map[name]) for name in model_names]

    try:
        stat, p_value = friedmanchisquare(*scores_for_friedman)

        print(f"\n--- Friedman Rank Test ({comparison_group_name}) ---")
        print(f"Metric: {metric_name}")
        print(f"Comparing models: {', '.join(model_names)} (N={score_lengths[0]} per model)")
        for name in model_names:
            print(f"  Mean {name}: {np.mean(all_scores_map[name]):.4f}, Median {name}: {np.median(all_scores_map[name]):.4f}")
        print(f"  Friedman statistic: {stat:.4f}")
        print(f"  P-value: {p_value:.6e}") # More precision

        if p_value < alpha:
            print(f"  Result: Statistically significant difference among models at alpha = {alpha} (p < {alpha}).")
            print(f"  Indication: At least one model's {metric_name} distribution is different. Consider post-hoc tests.")
            return True # Significant
        else:
            print(f"  Result: No statistically significant difference among models at alpha = {alpha} (p >= {alpha}).")
            return False # Not significant
    except Exception as e: # Broad exception for any issue during Friedman
        print(f"Error performing Friedman test for {comparison_group_name} - {metric_name}: {e}")
        return False
    
    
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


import torch.nn.functional as F
from torchvision import transforms

to_tensor = transforms.ToTensor()         # or your existing pipeline

def scores(pred_path,gt_path):

    image_root = '{}/'.format(pred_path)
    gt_root = '{}/masks/'.format(gt_path)
    test_loader = test_dataset(image_root, gt_root, 384)
    num1 = len(os.listdir(image_root))
    
    
    
    mIoU=[]
    DSC=[]
    ACC=[]
    
    for i in range(num1):
        res, gt, name = test_loader.load_data()
        
        # res = np.asarray(res, np.float32)
        # res /= (res.max() + 1e-8)
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        # P1,P2 = model(image)
        # res=P1+P2
       
        # res=out_model2(model(image))
        res = to_tensor(res).unsqueeze(0)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    
        dice,iou=test(res,gt)
        acc= accuracy(res,gt)
        
        ACC.append(acc)
        DSC.append(dice)
        mIoU.append(iou)

    return DSC, mIoU, ACC




# ---------------------------------------------------------------------------
# 3. MAIN PROCESSING LOGIC
# ---------------------------------------------------------------------------

dict_scores_dice = {
    "NUnet": {
        "CVC-ClinicDB": { "Original": [],  "Tree_NET": [] },
        "ISAC_2018":    { "Original": [],  "Tree_NET": [] }
    },
    "PolypPVT": {
        "CVC-ClinicDB": { "Original": [],  "Tree_NET": [] },
        "ISAC_2018":    { "Original": [],  "Tree_NET": [] }
    },
    "Unet": { # Unet has BSU-NET as an additional baseline to compare against Tree_NET
        "CVC-ClinicDB": { "Original": [], "BSU-NET": [], "Tree_NET": [] },
        "ISAC_2018":    { "Original": [], "BSU-NET": [], "Tree_NET": [] }
    }
}
dict_scores_iou = {
    "NUnet": {
        "CVC-ClinicDB": { "Original": [],  "Tree_NET": [] },
        "ISAC_2018":    { "Original": [],  "Tree_NET": [] }
    },
    "PolypPVT": {
        "CVC-ClinicDB": { "Original": [],  "Tree_NET": [] },
        "ISAC_2018":    { "Original": [],  "Tree_NET": [] }
    },
    "Unet": { # Unet has BSU-NET as an additional baseline to compare against Tree_NET
        "CVC-ClinicDB": { "Original": [], "BSU-NET": [], "Tree_NET": [] },
        "ISAC_2018":    { "Original": [], "BSU-NET": [], "Tree_NET": [] }
    }
}

dict_scores_acc = {
    "NUnet": {
        "CVC-ClinicDB": { "Original": [],  "Tree_NET": [] },
        "ISAC_2018":    { "Original": [],  "Tree_NET": [] }
    },
    "PolypPVT": {
        "CVC-ClinicDB": { "Original": [],  "Tree_NET": [] },
        "ISAC_2018":    { "Original": [],  "Tree_NET": [] }
    },
    "Unet": { # Unet has BSU-NET as an additional baseline to compare against Tree_NET
        "CVC-ClinicDB": { "Original": [], "BSU-NET": [], "Tree_NET": [] },
        "ISAC_2018":    { "Original": [], "BSU-NET": [], "Tree_NET": [] }
    }
}

def process_comparison_group(base_model_path, tree_net_model_path, gt_path,dataset_name, backbone_name, base_model_label):
    global dict_scores_dice,dict_scores_iou
    """
    Processes a single comparison group (e.g., Unet/CVC-ClinicDB/Original vs. Tree_NET).
    """
    comparison_group_name = f"{backbone_name}/{dataset_name}/{base_model_label}_vs_TreeNET"
    print(f"\nProcessing: {comparison_group_name}")

    base_scores_dice = []
    tree_net_scores_dice = []
    base_scores_iou = []
    tree_net_scores_iou = []

    tree_net_scores_dice,tree_net_scores_iou,tree_net_scores_acc=scores(tree_net_model_path,gt_path)
    base_scores_dice,base_scores_iou,base_scores_acc=scores(base_model_path,gt_path)
    
    dict_scores_dice[f"{backbone_name}"][f"{dataset_name}"]["Original"]=tree_net_scores_dice
    dict_scores_dice[f"{backbone_name}"][f"{dataset_name}"]["Tree_NET"]=base_scores_dice

    dict_scores_iou[f"{backbone_name}"][f"{dataset_name}"]["Original"]=tree_net_scores_iou
    dict_scores_iou[f"{backbone_name}"][f"{dataset_name}"]["Tree_NET"]=base_scores_iou
    
    # dict_scores_acc[f"{backbone_name}"][f"{dataset_name}"]["Original"]=tree_net_scores_acc
    # dict_scores_acc[f"{backbone_name}"][f"{dataset_name}"]["Tree_NET"]=base_scores_acc
    # # Perform statistical tests
    # perform_wilcoxon_test(base_scores_dice, tree_net_scores_dice, base_model_label, "Tree_NET", "Dice Score", comparison_group_name, ALPHA)
    # perform_wilcoxon_test(base_scores_iou, tree_net_scores_iou, base_model_label, "Tree_NET", "IoU Score", comparison_group_name, ALPHA)


def save_scores(base_model_path, gt_path,dataset_name, backbone_name, base_model_label):
    global dict_scores_dice,dict_scores_iou, dict_scores_acc
    """
    Processes a single comparison group (e.g., Unet/CVC-ClinicDB/Original vs. Tree_NET).
    """
    comparison_group_name = f"{backbone_name}/{dataset_name}/{base_model_label}_vs_TreeNET"
    print(f"\nProcessing: {comparison_group_name}")

    base_scores_dice = []
    base_scores_iou = []
    base_scores_acc = []

    # tree_net_scores_dice,tree_net_scores_iou,_=scores(tree_net_model_path,gt_path)
    base_scores_dice,base_scores_iou,base_scores_acc=scores(base_model_path,gt_path)
    
    # dict_scores_dice[f"{backbone_name}"][f"{dataset_name}"][f"{base_model_label}"]=base_scores_dice
    # dict_scores_iou[f"{backbone_name}"][f"{dataset_name}"][f"{base_model_label}"]=base_scores_iou

    dict_scores_acc[f"{backbone_name}"][f"{dataset_name}"][f"{base_model_label}"]=base_scores_acc
# --- Iterate through your directory structure ---
# Note: You might need to adjust dataset names if "ISAC_2018" should be "ISIC_2018" etc.
structure = {
    "NUnet": {
        "CVC-ClinicDB": ["Original", "Tree_NET"],
        "ISAC_2018": ["Original", "Tree_NET"] # Assuming ISIC_2018 for GT mapping
    },
    "PolypPVT": {
        "CVC-ClinicDB": ["Original", "Tree_NET"],
        "ISAC_2018": ["Original", "Tree_NET"]
    },
    "Unet": { # Unet has BSU-NET as an additional baseline to compare against Tree_NET
        "CVC-ClinicDB": ["Original", "Tree_NET", "BSU-NET"],
        "ISAC_2018": ["Original", "Tree_NET", "BSU-NET"]
    }
}
# import pandas as pd

# rows = []
# for backbone, dsets in dict_scores_dice.items():
#     for dataset, models in dsets.items():
#         for base_model_type, dice_scores in models.items():
#             dice_mean = np.mean(dice_scores)
#             iou_mean  = np.mean(dict_scores_iou[backbone][dataset][base_model_type])
#             # append two rows: one for dice, one for iou
#             rows.append({
#                 "backbone":        backbone,
#                 "dataset":         dataset,
#                 "base_model_type": base_model_type,
#                 "score_type":      "dice",
#                 "value":           dice_mean
#             })
#             rows.append({
#                 "backbone":        backbone,
#                 "dataset":         dataset,
#                 "base_model_type": base_model_type,
#                 "score_type":      "iou",
#                 "value":           iou_mean
#             })

# # build DataFrame
# df = pd.DataFrame(rows)

# # print it out
# print(df.to_string(index=False))
# df.to_excel("scores_long.xlsx", index=False, engine="openpyxl")
# for backbone, datasets in structure.items():
#     for dataset, base_model_types in datasets.items():
#         for base_model_type in base_model_types: # "Original" or "BSU-NET"
#             print(f"{backbone}, {dataset}, {base_model_type} dice score is:",  np.mean(dict_scores_dice[f"{backbone}"][f"{dataset}"][f"{base_model_type}"]))
#             print(f"{backbone}, {dataset}, {base_model_type} iou score is:",  np.mean(dict_scores_iou[f"{backbone}"][f"{dataset}"][f"{base_model_type}"]))

for backbone, datasets in structure.items():
    for dataset, base_model_types in datasets.items():
        for base_model_type in base_model_types: # "Original" or "BSU-NET"
            base_model_path = os.path.join(BASE_OUTPUT_DIR, backbone, dataset, base_model_type)
            # tree_net_model_path = os.path.join(BASE_OUTPUT_DIR, backbone, dataset, "Tree_NET")

            # Important: `dataset` variable used here must match keys in `get_gt_path`
            # e.g., if folder is ISAC_2018 but GT key is ISIC_2018, adjust `dataset`
            # or make `get_gt_path` more flexible.
            current_dataset_name_for_gt = dataset # Default
            if dataset == "ISAC_2018": # Example adjustment
                 current_dataset_name_for_gt = "ISAC_2018"
                 
                 gt_path = './dataset/ISAC2018/test'

            else:
                 gt_path = './dataset/TestDataset/{}'.format(current_dataset_name_for_gt)
            
            # gt_path = './dataset/{}/test'.format(_data_name)
            save_scores(base_model_path, gt_path,current_dataset_name_for_gt, backbone, base_model_type)
print("\n\nProcessing Complete.")



for backbone, datasets in structure.items():
    for dataset, base_model_types in datasets.items():
        for base_model_type in base_model_types: # "Original" or "BSU-NET"
            base_model_path = os.path.join(BASE_OUTPUT_DIR, backbone, dataset, base_model_type)
            tree_net_model_path = os.path.join(BASE_OUTPUT_DIR, backbone, dataset, "Tree_NET")

            # Important: `dataset` variable used here must match keys in `get_gt_path`
            # e.g., if folder is ISAC_2018 but GT key is ISIC_2018, adjust `dataset`
            # or make `get_gt_path` more flexible.
            current_dataset_name_for_gt = dataset # Default
            if dataset == "ISAC_2018": # Example adjustment
                 current_dataset_name_for_gt = "ISAC_2018"
                 
                 gt_path = './dataset/ISAC2018/test'

            else:
                 gt_path = './dataset/TestDataset/{}'.format(current_dataset_name_for_gt)
            
            # gt_path = './dataset/{}/test'.format(_data_name)
            process_comparison_group(base_model_path, tree_net_model_path, gt_path,current_dataset_name_for_gt ,backbone, base_model_type)
            # process_comparison_group(base_model_path, tree_net_model_path, gt_path,dataset_name, backbone_name, base_model_label)
print("\n\nProcessing Complete.")


# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 22:05:17 2024

@author: Orhan
"""

import pickle 

# with open("results.pkl", "wb") as f: pickle.dump(dict_scores_dice, f)

pickle.dump(dict_scores_dice, open("results_dice.pkl", "wb"))
pickle.dump(dict_scores_iou, open("results_iou.pkl", "wb"))


my_dict = pickle.load(open("results_iou.pkl", "rb"))
