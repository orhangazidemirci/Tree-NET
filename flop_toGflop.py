# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 02:33:01 2025

@author: Orhan
"""
import pandas as pd

file_path = 'C:/Users/Orhan/Documents/GitHub/Polyp-PVT/result_map/Flops/Book2.xlsx'  # Replace with actual file path if provided
df = pd.read_excel(file_path)
# Convert values
# Adjusting the column names to match the dataset structure based on the provided keys
df.columns = [
    "Model", "Variant", "Batch", "FLOPs", "Parameters", "Peak Memory Allocated (MB)"
]

# Applying the conversions
df["FLOPs (GFLOP)"] = df["FLOPs"] / 1e9  # Convert FLOPs to GFLOPs
df["Parameters (M)"] = df["Parameters"] / 1e6  # Convert Parameters to Million
df["Peak Memory Allocated (GB)"] = df["Peak Memory Allocated (MB)"] / 1024  # Convert MB to GB

# Dropping the original columns that have been converted
df_transformed = df.drop(columns=["FLOPs", "Parameters", "Peak Memory Allocated (MB)"])

# Displaying the updated DataFrame to the user

output_file_path = 'C:/Users/Orhan/Documents/GitHub/Polyp-PVT/result_map/Flops/flops.xlsx'
df_transformed.to_excel(output_file_path, index=False)

# Informing the user about the saved file
output_file_path