# -*- coding: utf-8 -*-
"""
Created on Sat May 24 21:29:39 2025

@author: Orhan
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare
import os # For path joining

# --- 1. Populate with Example Data (Replace with your actual data) ---
np.random.seed(42) # for reproducibility
num_samples_cvc = 15
num_samples_isac = 20

# dict_scores_dice = {
#     "NUnet": {
#         "CVC-ClinicDB": {
#             "Original": np.random.rand(num_samples_cvc).tolist(),
#             "Tree_NET": (np.random.rand(num_samples_cvc) * 0.9 + 0.05).tolist()
#         },
#         "ISAC_2018":    {
#             "Original": np.random.rand(num_samples_isac).tolist(),
#             "Tree_NET": (np.random.rand(num_samples_isac) * 0.95 + 0.02).tolist()
#         }
#     },
#     "PolypPVT": {
#         "CVC-ClinicDB": {
#             "Original": np.random.rand(num_samples_cvc).tolist(),
#             "Tree_NET": (np.random.rand(num_samples_cvc) * 0.8 + 0.15).tolist()
#         },
#         "ISAC_2018":    {
#             "Original": np.random.rand(num_samples_isac).tolist(),
#             "Tree_NET": (np.random.rand(num_samples_isac) * 0.85 + 0.1).tolist()
#         }
#     },
#     "Unet": {
#         "CVC-ClinicDB": {
#             "Original": np.random.rand(num_samples_cvc).tolist(),
#             "BSU-NET":  (np.random.rand(num_samples_cvc) * 0.9 + 0.03).tolist(),
#             "Tree_NET": (np.random.rand(num_samples_cvc) * 0.8 + 0.18).tolist()
#         },
#         "ISAC_2018":    {
#             "Original": np.random.rand(num_samples_isac).tolist(),
#             "BSU-NET":  (np.random.rand(num_samples_isac) * 0.92 + 0.04).tolist(),
#             "Tree_NET": (np.random.rand(num_samples_isac) * 0.82 + 0.15).tolist()
#         }
#     }
# }

# --- 2. Perform Statistical Tests and Store Results ---
results_list = []

for model_type, datasets in dict_scores_dice.items():
    for dataset_name, methods_scores in datasets.items():
        branch_name = f"{model_type} - {dataset_name}"
        print(f"Processing branch: {branch_name}")

        scores_original = methods_scores.get("Original")
        scores_tree_net = methods_scores.get("Tree_NET")
        scores_bsu_net = methods_scores.get("BSU-NET")

        p_wilcoxon_orig_vs_tree = np.nan
        p_wilcoxon_orig_vs_bsu = np.nan
        p_wilcoxon_bsu_vs_tree = np.nan
        p_friedman = np.nan

        if scores_original and scores_tree_net:
            if len(scores_original) == len(scores_tree_net) and len(scores_original) > 0:
                if not np.array_equal(scores_original, scores_tree_net):
                    try:
                        _, p_wilcoxon_orig_vs_tree = wilcoxon(scores_original, scores_tree_net)
                    except ValueError as e:
                        print(f"  Wilcoxon (Orig vs Tree) Warning for {branch_name}: {e}")
                        p_wilcoxon_orig_vs_tree = 1.0
                else:
                    p_wilcoxon_orig_vs_tree = 1.0
            else:
                print(f"  Skipping Wilcoxon (Orig vs Tree) for {branch_name} due to mismatched/empty lists.")

        if scores_bsu_net:
            if scores_original:
                if len(scores_original) == len(scores_bsu_net) and len(scores_original) > 0:
                    if not np.array_equal(scores_original, scores_bsu_net):
                        try:
                            _, p_wilcoxon_orig_vs_bsu = wilcoxon(scores_original, scores_bsu_net)
                        except ValueError as e:
                            print(f"  Wilcoxon (Orig vs BSU) Warning for {branch_name}: {e}")
                            p_wilcoxon_orig_vs_bsu = 1.0
                    else:
                        p_wilcoxon_orig_vs_bsu = 1.0
                else:
                    print(f"  Skipping Wilcoxon (Orig vs BSU) for {branch_name} due to mismatched/empty lists.")

            if scores_tree_net:
                if len(scores_bsu_net) == len(scores_tree_net) and len(scores_bsu_net) > 0:
                    if not np.array_equal(scores_bsu_net, scores_tree_net):
                        try:
                            _, p_wilcoxon_bsu_vs_tree = wilcoxon(scores_bsu_net, scores_tree_net)
                        except ValueError as e:
                            print(f"  Wilcoxon (BSU vs Tree) Warning for {branch_name}: {e}")
                            p_wilcoxon_bsu_vs_tree = 1.0
                    else:
                        p_wilcoxon_bsu_vs_tree = 1.0
                else:
                    print(f"  Skipping Wilcoxon (BSU vs Tree) for {branch_name} due to mismatched/empty lists.")

            if scores_original and scores_tree_net:
                if (len(scores_original) == len(scores_bsu_net) == len(scores_tree_net) and
                        len(scores_original) > 0):
                    # Ensure lists are not all identical for friedman
                    data_for_friedman = [scores_original, scores_bsu_net, scores_tree_net]
                    all_identical = True
                    for i in range(len(data_for_friedman[0])):
                        if not (data_for_friedman[0][i] == data_for_friedman[1][i] == data_for_friedman[2][i]):
                            all_identical = False
                            break
                    if not all_identical:
                        try:
                            _, p_friedman = friedmanchisquare(*data_for_friedman)
                        except ValueError as e:
                            print(f"  Friedman Warning for {branch_name}: {e}")
                            p_friedman = 1.0 # Or np.nan if preferred
                    else:
                        print(f"  Skipping Friedman for {branch_name} as all groups have identical scores.")
                        p_friedman = 1.0 # All groups identical, no difference
                else:
                    print(f"  Skipping Friedman for {branch_name} due to mismatched/empty lists for all three groups.")
        
        results_list.append({
            "Branch": branch_name,
            "Wilcoxon (Orig vs Tree) p-value": p_wilcoxon_orig_vs_tree,
            "Wilcoxon (Orig vs BSU) p-value": p_wilcoxon_orig_vs_bsu,
            "Wilcoxon (BSU vs Tree) p-value": p_wilcoxon_bsu_vs_tree,
            "Friedman (Orig, BSU, Tree) p-value": p_friedman
        })

# --- 3. Create and Display the Table ---
results_df = pd.DataFrame(results_list)
results_df = results_df.set_index("Branch")

def format_p_value(p):
    if pd.isna(p):
        return "-"
    return f"{p:.4f}" if p >= 0.0001 else "<0.0001"

results_df_formatted = results_df.applymap(format_p_value)

print("\n--- Statistical Test Results (Raw p-values) ---")
print(results_df)

print("\n--- Formatted Statistical Test Results ---")
print(results_df_formatted)

# --- 4. Save the Tables ---
output_dir = "statistical_results" # You can change this directory name
os.makedirs(output_dir, exist_ok=True) # Create directory if it doesn't exist

# Save raw p-values to CSV
csv_path_raw = os.path.join(output_dir, "dice_stats_raw_p_values.csv")
results_df.to_csv(csv_path_raw)
print(f"\nRaw p-value results saved to: {csv_path_raw}")

# Save formatted p-values to a text file
txt_path_formatted = os.path.join(output_dir, "dice_stats_formatted.txt")
with open(txt_path_formatted, 'w') as f:
    f.write("Formatted Statistical Test Results\n")
    f.write("----------------------------------\n")
    f.write(results_df_formatted.to_string()) # to_string() gives a nice text representation
print(f"Formatted results saved to: {txt_path_formatted}")

# Optional: Save formatted p-values to CSV (less common for pre-formatted strings but possible)
csv_path_formatted = os.path.join(output_dir, "dice_stats_formatted_p_values.csv")
results_df_formatted.to_csv(csv_path_formatted)
print(f"Formatted p-value results (as strings) saved to: {csv_path_formatted}")

# Optional: Save formatted table to Markdown for easy pasting into reports
md_path_formatted = os.path.join(output_dir, "dice_stats_formatted.md")
with open(md_path_formatted, 'w') as f:
    f.write("# Formatted Statistical Test Results\n\n")
    f.write(results_df_formatted.to_markdown()) # to_markdown() for GitHub-flavored markdown
print(f"Formatted results (Markdown) saved to: {md_path_formatted}")