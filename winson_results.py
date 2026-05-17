# -*- coding: utf-8 -*-
"""
Created on Sun May 25 04:52:40 2025

@author: Orhan
"""

import pandas as pd
from scipy import stats
import numpy as np

def analyze_dice_scores(dict_scores_dice):
    results_list = []
    alpha = 0.05

    # Placeholder for your actual data.
    # Ensure these lists are populated with your real dice scores.
    # Example data:
    example_data = dict_scores_dice
    # Use the provided dict_scores_dice. If it's the one with empty lists,
    # this will use the example_data for demonstration.
    # In a real scenario, the user's dict_scores_dice should be pre-filled.
    # This check is for demonstration if the input dict is the template with empty lists.
    is_input_empty = True
    for model_type_key in dict_scores_dice:
        for dataset_key in dict_scores_dice[model_type_key]:
            for method_key in dict_scores_dice[model_type_key][dataset_key]:
                if dict_scores_dice[model_type_key][dataset_key][method_key]: # if list is not empty
                    is_input_empty = False
                    break
            if not is_input_empty:
                break
        if not is_input_empty:
            break
    
    current_data_source = dict_scores_dice
    if is_input_empty:
        print("INFO: The provided 'dict_scores_dice' contains empty lists. Using example data for demonstration.\n")
        current_data_source = example_data


    for model_type, datasets in current_data_source.items():
        for dataset_name, methods in datasets.items():
            
            original_scores = methods.get("Original")
            bsu_net_scores = methods.get("BSU-NET")
            tree_net_scores = methods.get("Tree_NET")

            # Basic data validation
            if not original_scores or not tree_net_scores:
                results_list.append({
                    "Model_Type": model_type, "Dataset": dataset_name, "Test_Type": "N/A",
                    "Comparison": "Data Check", "Statistic": "N/A", "P_Value": "N/A",
                    "Conclusion": "Error: Original or Tree_NET scores missing or empty."
                })
                continue

            min_len = len(original_scores)
            if bsu_net_scores:
                min_len = min(min_len, len(bsu_net_scores))
            min_len = min(min_len, len(tree_net_scores))
            
            # Ensure all lists being compared have this minimum length by truncation (if needed, though ideally they are already same length)
            # More robustly, one should check if all intended lists for a test have the same length.
            # For simplicity here, we'll assume they should all be of `min_len` if they are to be compared.
            # This is a simplistic way to handle potentially ragged inputs if they were not already curated.
            # A better check is done before each test.

            if min_len < 1: # Or some other threshold like 5 for meaningful test
                 results_list.append({
                    "Model_Type": model_type, "Dataset": dataset_name, "Test_Type": "N/A",
                    "Comparison": "Data Check", "Statistic": "N/A", "P_Value": "N/A",
                    "Conclusion": f"Error: Not enough data points (min samples found: {min_len})."
                })
                 continue


            # Case 1: Three groups (Original, BSU-NET, Tree_NET) - typically for "Unet"
            if bsu_net_scores:
                if not (len(original_scores) == len(bsu_net_scores) == len(tree_net_scores)):
                    results_list.append({
                        "Model_Type": model_type, "Dataset": dataset_name, "Test_Type": "Friedman",
                        "Comparison": "Original vs BSU-NET vs Tree_NET", "Statistic": "N/A", "P_Value": "N/A",
                        "Conclusion": "Error: Unequal sample sizes for 3-group comparison."
                    })
                elif len(original_scores) == 0: # Also handles other lists due to above check
                     results_list.append({
                        "Model_Type": model_type, "Dataset": dataset_name, "Test_Type": "Friedman",
                        "Comparison": "Original vs BSU-NET vs Tree_NET", "Statistic": "N/A", "P_Value": "N/A",
                        "Conclusion": "Error: Empty score lists for 3-group comparison."
                    })
                else:
                    try:
                        stat_friedman, p_friedman = stats.friedmanchisquare(original_scores, bsu_net_scores, tree_net_scores)
                        conclusion_friedman = f"Overall significant difference (p={p_friedman:.4f})" if p_friedman < alpha else f"No overall significant difference (p={p_friedman:.4f})"
                        results_list.append({
                            "Model_Type": model_type, "Dataset": dataset_name, "Test_Type": "Friedman",
                            "Comparison": "Original vs BSU-NET vs Tree_NET", "Statistic": f"{stat_friedman:.4f}", "P_Value": f"{p_friedman:.4f}",
                            "Conclusion": conclusion_friedman
                        })
                    except ValueError as e:
                         results_list.append({
                            "Model_Type": model_type, "Dataset": dataset_name, "Test_Type": "Friedman",
                            "Comparison": "Original vs BSU-NET vs Tree_NET", "Statistic": "N/A", "P_Value": "N/A",
                            "Conclusion": f"Error during Friedman test: {e}"
                        })


                # Pairwise Wilcoxon tests for 3-group case
                comparisons = [
                    ("Original", original_scores, "Tree_NET", tree_net_scores),
                    ("Original", original_scores, "BSU-NET", bsu_net_scores),
                    ("BSU-NET", bsu_net_scores, "Tree_NET", tree_net_scores)
                ]
                for name1, scores1, name2, scores2 in comparisons:
                    if not scores1 or not scores2: continue # Should be caught by earlier checks but good to have
                    if len(scores1) != len(scores2):
                        results_list.append({
                            "Model_Type": model_type, "Dataset": dataset_name, "Test_Type": "Wilcoxon",
                            "Comparison": f"{name1} vs {name2}", "Statistic": "N/A", "P_Value": "N/A",
                            "Conclusion": "Error: Unequal sample sizes for Wilcoxon."
                        })
                        continue
                    if len(scores1) == 0:
                        results_list.append({
                            "Model_Type": model_type, "Dataset": dataset_name, "Test_Type": "Wilcoxon",
                            "Comparison": f"{name1} vs {name2}", "Statistic": "N/A", "P_Value": "N/A",
                            "Conclusion": "Error: Empty score lists for Wilcoxon."
                        })
                        continue
                    
                    # Check for constant array for Wilcoxon
                    if np.all(np.array(scores1) == np.array(scores1)[0]) and np.all(np.array(scores2) == np.array(scores2)[0]) and np.array(scores1)[0] == np.array(scores2)[0]:
                         results_list.append({
                            "Model_Type": model_type, "Dataset": dataset_name, "Test_Type": "Wilcoxon",
                            "Comparison": f"{name1} vs {name2}", "Statistic": "N/A", "P_Value": "N/A",
                            "Conclusion": "Skipped: Data are constant and identical across groups."
                        })
                         continue
                    # Check if differences are all zero
                    diff = np.array(scores1) - np.array(scores2)
                    if np.all(diff == 0):
                        results_list.append({
                            "Model_Type": model_type, "Dataset": dataset_name, "Test_Type": "Wilcoxon",
                            "Comparison": f"{name1} vs {name2}", "Statistic": "N/A", "P_Value": "1.0000 (Identical Samples)",
                            "Conclusion": "Samples are identical."
                        })
                        continue


                    try:
                        stat_wilcoxon, p_wilcoxon = stats.wilcoxon(scores1, scores2, alternative='two-sided', zero_method='wilcox')
                        
                        median1 = np.median(scores1)
                        median2 = np.median(scores2)
                        
                        if p_wilcoxon < alpha:
                            if median1 > median2:
                                conclusion_wilcoxon = f"{name1} significantly better than {name2} (p={p_wilcoxon:.4f}; Medians: {median1:.3f} vs {median2:.3f})"
                            elif median2 > median1:
                                conclusion_wilcoxon = f"{name2} significantly better than {name1} (p={p_wilcoxon:.4f}; Medians: {median2:.3f} vs {median1:.3f})"
                            else:
                                conclusion_wilcoxon = f"Significant difference, but medians are equal (p={p_wilcoxon:.4f}; Medians: {median1:.3f})"
                        else:
                            conclusion_wilcoxon = f"No significant difference (p={p_wilcoxon:.4f}; Medians: {median1:.3f} vs {median2:.3f})"
                        
                        results_list.append({
                            "Model_Type": model_type, "Dataset": dataset_name, "Test_Type": "Wilcoxon",
                            "Comparison": f"{name1} vs {name2}", "Statistic": f"{stat_wilcoxon:.4f}", "P_Value": f"{p_wilcoxon:.4f}",
                            "Conclusion": conclusion_wilcoxon
                        })
                    except ValueError as e: # Catches "zero_method 'wilcox' and 'pratt' do not handle ties." or "samples are identical"
                        results_list.append({
                            "Model_Type": model_type, "Dataset": dataset_name, "Test_Type": "Wilcoxon",
                            "Comparison": f"{name1} vs {name2}", "Statistic": "N/A", "P_Value": "N/A",
                            "Conclusion": f"Error/Skipped: {e}"
                        })


            # Case 2: Two groups (Original, Tree_NET) - typically for "NUnet", "PolypPVT"
            else:
                if len(original_scores) != len(tree_net_scores):
                    results_list.append({
                        "Model_Type": model_type, "Dataset": dataset_name, "Test_Type": "Wilcoxon",
                        "Comparison": "Original vs Tree_NET", "Statistic": "N/A", "P_Value": "N/A",
                        "Conclusion": "Error: Unequal sample sizes for Wilcoxon."
                    })
                    continue
                if len(original_scores) == 0:
                     results_list.append({
                        "Model_Type": model_type, "Dataset": dataset_name, "Test_Type": "Wilcoxon",
                        "Comparison": "Original vs Tree_NET", "Statistic": "N/A", "P_Value": "N/A",
                        "Conclusion": "Error: Empty score lists for Wilcoxon."
                    })
                     continue

                # Check for constant array for Wilcoxon
                if np.all(np.array(original_scores) == np.array(original_scores)[0]) and \
                   np.all(np.array(tree_net_scores) == np.array(tree_net_scores)[0]) and \
                   np.array(original_scores)[0] == np.array(tree_net_scores)[0]:
                     results_list.append({
                        "Model_Type": model_type, "Dataset": dataset_name, "Test_Type": "Wilcoxon",
                        "Comparison": "Original vs Tree_NET", "Statistic": "N/A", "P_Value": "N/A",
                        "Conclusion": "Skipped: Data are constant and identical across groups."
                    })
                     continue
                # Check if differences are all zero
                diff = np.array(original_scores) - np.array(tree_net_scores)
                if np.all(diff == 0):
                    results_list.append({
                        "Model_Type": model_type, "Dataset": dataset_name, "Test_Type": "Wilcoxon",
                        "Comparison": "Original vs Tree_NET", "Statistic": "N/A", "P_Value": "1.0000 (Identical Samples)",
                        "Conclusion": "Samples are identical."
                    })
                    continue

                try:
                    stat_wilcoxon, p_wilcoxon = stats.wilcoxon(original_scores, tree_net_scores, alternative='two-sided', zero_method='wilcox')
                    median_orig = np.median(original_scores)
                    median_tree = np.median(tree_net_scores)

                    if p_wilcoxon < alpha:
                        if median_orig > median_tree:
                            conclusion_wilcoxon = f"Original significantly better than Tree_NET (p={p_wilcoxon:.4f}; Medians: {median_orig:.3f} vs {median_tree:.3f})"
                        elif median_tree > median_orig:
                            conclusion_wilcoxon = f"Tree_NET significantly better than Original (p={p_wilcoxon:.4f}; Medians: {median_tree:.3f} vs {median_orig:.3f})"
                        else:
                            conclusion_wilcoxon = f"Significant difference, but medians are equal (p={p_wilcoxon:.4f}; Medians: {median_tree:.3f})"
                    else:
                        conclusion_wilcoxon = f"No significant difference (p={p_wilcoxon:.4f}; Medians: {median_orig:.3f} vs {median_tree:.3f})"
                    
                    results_list.append({
                        "Model_Type": model_type, "Dataset": dataset_name, "Test_Type": "Wilcoxon",
                        "Comparison": "Original vs Tree_NET", "Statistic": f"{stat_wilcoxon:.4f}", "P_Value": f"{p_wilcoxon:.4f}",
                        "Conclusion": conclusion_wilcoxon
                    })
                except ValueError as e:
                     results_list.append({
                        "Model_Type": model_type, "Dataset": dataset_name, "Test_Type": "Wilcoxon",
                        "Comparison": "Original vs Tree_NET", "Statistic": "N/A", "P_Value": "N/A",
                        "Conclusion": f"Error/Skipped: {e}"
                    })


    return pd.DataFrame(results_list)

# This is the structure you provided.
# YOU NEED TO FILL THIS WITH YOUR ACTUAL DICE SCORES.
# For example: "Original": [0.85, 0.92, 0.78, ...], "Tree_NET": [0.90, 0.91, 0.82, ...]
# The lists for comparison (e.g. Original, Tree_NET for a given dataset) must have the same number of scores.


# Perform the analysis
# If dict_scores_dice_user remains empty, the function will use internal example data.
# Otherwise, it will use the data you provide in dict_scores_dice_user.
results_df = analyze_dice_scores(dict_scores_dice)

# Print the results table
print(results_df.to_string())
results_df = analyze_dice_scores(dict_scores_iou)
print(results_df.to_string())


results_df = analyze_dice_scores(dict_scores_acc)
print(results_df.to_string())
