"""
Run this once to generate:
  dataset/Synapse/lists/lists_Synapse/train.txt
  dataset/Synapse/lists/lists_Synapse/test_vol.txt

Usage:
  python generate_lists.py
  python generate_lists.py --data_root /custom/path/to/Synapse
"""

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str,
                    default='dataset/Synapse',
                    help='Path to your Synapse folder')
args = parser.parse_args()

train_npz = os.path.join(args.data_root, 'train_npz')
test_h5   = os.path.join(args.data_root, 'test_vol_h5')
out_dir   = os.path.join(args.data_root, 'lists', 'lists_Synapse')

os.makedirs(out_dir, exist_ok=True)

# train.txt — one slice name per line (no extension)
train_slices = sorted([
    f.replace('.npz', '')
    for f in os.listdir(train_npz)
    if f.endswith('.npz')
])
with open(os.path.join(out_dir, 'train.txt'), 'w') as f:
    f.write('\n'.join(train_slices))

# test_vol.txt — one volume name per line (no extension)
test_vols = sorted([
    f.replace('.npy.h5', '')
    for f in os.listdir(test_h5)
    if f.endswith('.h5')
])
with open(os.path.join(out_dir, 'test_vol.txt'), 'w') as f:
    f.write('\n'.join(test_vols))

print("Done!")
print("  train.txt  : {} slices  →  {}".format(
    len(train_slices), os.path.join(out_dir, 'train.txt')))
print("  test_vol.txt: {} volumes →  {}".format(
    len(test_vols), os.path.join(out_dir, 'test_vol.txt')))
