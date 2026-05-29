# Tree-NET — Efficient Medical Image Segmentation via Dual Bottleneck Supervision

Tree-NET is a novel deep learning framework for medical image segmentation that leverages **dual bottleneck supervision** to enhance both segmentation accuracy and computational efficiency. It compresses both input images and ground-truth label masks into low-dimensional representations, training the central segmentation network entirely on these compact feature spaces — reducing FLOPs by 4–13× while maintaining or improving accuracy.

---

## Architecture Overview

Tree-NET consists of three independently trained components assembled end-to-end:

- **Encoder-Net**: A convolutional autoencoder trained on input images. Only its encoder half is retained at inference to compress inputs into bottleneck features.
- **Bridge-Net**: The central segmentation model, trained on compressed inputs and supervised by compressed labels. Any existing segmentation backbone (U-NET, U-NET++, Polyp-PVT) can serve as Bridge-Net.
- **Decoder-Net**: A convolutional autoencoder trained on label masks. Only its decoder half is retained at inference to upsample Bridge-Net outputs back to full resolution.

Training proceeds in three sequential stages:
1. Train Encoder-Net on input images (Euclidean loss)
2. Train Decoder-Net on label masks (Euclidean loss)
3. Extract bottleneck features from both; train Bridge-Net on compressed inputs supervised by compressed labels (weighted IoU + weighted BCE)

At inference, the pipeline is: **Encoder-Net encoder → Bridge-Net → Decoder-Net decoder**, assembled end-to-end.

---

## Features

- **Dual bottleneck supervision**: Compresses both inputs and labels, reducing spatial resolution by 16× (384×384 → 96×96) with minimal semantic loss
- **Backbone-agnostic**: Bridge-Net supports `pvt`, `UNET`, and `NestedUNET` without modifying their internal structures
- **Multi-scale training**: Trains at 256×256, 384×384, and 512×512 per batch
- **Significant efficiency gains**: 4–13× fewer FLOPs and lower peak memory vs. baselines
- **Competitive accuracy**: Matches or exceeds U-NET, U-NET++, BS U-NET, and Polyp-PVT on ISIC-2018 and CVC-ClinicDB
- **Automated benchmarking**: End-to-end inference evaluation with `run_benchmark`

---

## Project Structure

```
.
├── train.py                  # Main training script
├── inference_test.py         # Benchmark runner
├── lib/
│   ├── pvt.py                # PolypPVT model definition
│   └── models.py             # Model factory (call_model)
└── utils/
    ├── dataloader.py         # get_loader, test_dataset
    ├── encoder.py            # Encoder-Net / Decoder-Net, components_exp
    └── utils.py              # clip_gradient, adjust_lr, AvgMeter
```

---

## Installation

```bash
git clone https://github.com/orhangazidemirci/Tree-NET.git
cd Tree-NET
pip install -r requirements.txt
```

**Requirements** (key dependencies):
- Python ≥ 3.8
- PyTorch ≥ 1.10
- torchvision
- numpy
- matplotlib

---

## Dataset Setup

Organize your datasets under `./dataset/` as follows:

```
dataset/
├── train/
│   ├── images/
│   └── masks/
├── TestDataset/
│   └── CVC-ClinicDB/
│       ├── images/
│       └── masks/
├── valid/
│   └── CVC-ClinicDB/
│       ├── images/
│       └── masks/
└── ISIC2018/
    ├── train/
    │   ├── images/
    │   └── masks/
    ├── test/
    │   ├── images/
    │   └── masks/
    └── valid/
        ├── images/
        └── masks/
```

### Datasets Used

**CVC-ClinicDB** — 612 colonoscopy images (384×288×3) with polyp segmentation masks. Official training dataset for the MICCAI 2015 Automatic Polyp Detection Challenge. Split: 80% train / 10% validation / 10% test.

**ISIC-2018** — 2,594 dermoscopic images (1022×767) for training, 100 for validation, and 1,000 for testing, with ground-truth skin lesion masks. Split provided by default.

All images are resized to 384×384 and normalized to [0, 1] during preprocessing.

---

## Usage

### Stage 1 & 2 — Train Encoder-Net and Decoder-Net

```bash
python train.py \
  --component_selected Encoder_x4 \
  --dataset ISIC2018 \
  --epoch 100 \
  --batchsize 8 \
  --lr 1e-3 \
  --optimizer AdamW
```

```bash
python train.py \
  --component_selected Decoder_x16 \
  --dataset ISIC2018 \
  --epoch 100 \
  --batchsize 8 \
  --lr 1e-3 \
  --optimizer AdamW
```

### Stage 3 — Train Bridge-Net (Tree-NET mode)

```bash
python train.py \
  --arch pvt \
  --dataset ISIC2018 \
  --epoch 100 \
  --batchsize 8 \
  --lr 1e-4 \
  --optimizer AdamW \
  --treenet True
```

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--arch` | `UNET` | Bridge-Net backbone: `pvt`, `UNET`, `NestedUNET` |
| `--dataset` | `ISAC2018` | Dataset: `CVC-ClinicDB` or `ISAC2018` |
| `--epoch` | `100` | Number of training epochs |
| `--batchsize` | `8` | Training batch size |
| `--lr` | `1e-4` | Learning rate (Bridge-Net); use `1e-3` for Encoder/Decoder-Net |
| `--optimizer` | `AdamW` | `AdamW` or `SGD` |
| `--trainsize` | `384` | Base training image size |
| `--clip` | `0.5` | Gradient clipping margin |
| `--augmentation` | `True` | Enable random flip/rotation augmentation |
| `--encoder_component` | `Encoder_x4` | Encoder variant (4× spatial reduction) |
| `--decoder_component` | `Decoder_x16` | Decoder variant: `Decoder_x4`, `Decoder_x8`, `Decoder_x16` |
| `--component_selected` | `ORIGINAL` | Experiment mode: `ORIGINAL`, `Encoder_x4`, `Decoder_x4`, etc. |
| `--treenet` | `False` | Enable Tree-NET mode (train Bridge-Net on bottleneck features) |

---

## Training Parameters

| Parameter | Encoder-Net | Bridge-Net | Decoder-Net |
|---|---|---|---|
| Input Size | 3 × N × N | 3 × (N/e) × (N/e) | 3 × N × N |
| Bottleneck Size | 3 × (N/e) × (N/e) | B × L × L | D × (N/d) × (N/d) |
| Output Size | 3 × N × N | D × (N/d) × (N/d) | 1 × N × N |
| Batch Size | 8 | 8 | 8 |
| Learning Rate | 0.001 | 0.0001 | 0.001 |
| Loss Function | Euclidean | wIoU + wBCE | Euclidean |
| Optimizer | AdamW | AdamW | AdamW |
| Epochs | 100 | 100 | 100 |
| Random Seed | 42 | 42 | 42 |

---

## Experiment Grid

When run as `__main__`, the script automatically executes a full experiment grid across:

- **Batch sizes**: `[8, 16]`
- **Architectures**: `pvt`, `UNET`, `NestedUNET`
- **Components**: `Encoder_x4`, `Decoder_x4`, `Decoder_x16`, `ORIGINAL`
- **TreeNet modes**: `[True, False]`

Followed by a final benchmark across all configured test datasets.

---

## Loss Functions

**Structure Loss** (Bridge-Net with PVT backbone):
Weighted BCE + weighted IoU. Edge-aware pixel weights are computed via average pooling on the ground-truth mask, prioritizing difficult boundary pixels.

$$\mathcal{L} = \mathcal{L}_{wIoU} + \mathcal{L}_{wBCE}$$

**Euclidean Loss** (Encoder-Net and Decoder-Net):
Mean squared error between predicted and ground-truth values, used to train the autoencoder components.

---

## Outputs

| Output | Location |
|---|---|
| Model checkpoints | `./model_pth/<arch>/` |
| Best model | `./model_pth/<arch>/_<dataset>best.pth` |
| Training logs | `train_log<component><dataset><arch><mode>.log` |
| Benchmark results | Printed to stdout / logged |

---

## Evaluation Metrics

| Metric | Formula |
|---|---|
| Dice (DSC) | $\frac{2 \cdot tp}{2 \cdot tp + fp + fn}$ |
| IoU | $\frac{tp}{tp + fp + fn}$ |
| Accuracy | $\frac{tp + tn}{tp + tn + fp + fn}$ |

The best model checkpoint is saved whenever validation Dice improves. Statistical significance of performance differences is assessed using a **paired Wilcoxon signed-rank test** on per-image Dice scores (threshold: p < 0.01).

---

## Results

### Accuracy (Dice / IoU / Accuracy)

| Model | Variant | ISIC-2018 Dice | ISIC-2018 IoU | CVC-ClinicDB Dice | CVC-ClinicDB IoU |
|---|---|---|---|---|---|
| U-NET | Original | 0.807 | 0.700 | 0.936 | 0.891 |
| BS U-NET | Original | 0.822 | 0.723 | 0.928 | 0.883 |
| U-NET++ | Original | 0.829 | 0.736 | 0.940 | 0.893 |
| Polyp-PVT | Original | 0.903 | 0.839 | **0.959** | **0.924** |
| **Tree-NET** | U-NET BB | 0.867 | 0.790 | 0.923 | 0.872 |
| **Tree-NET** | U-NET++ BB | 0.862 | 0.787 | 0.925 | 0.875 |
| **Tree-NET** | Polyp-PVT BB | **0.886** | **0.811** | 0.946 | 0.901 |

Tree-NET achieves the best results on ISIC-2018 (statistically significant, p < 0.01 vs. all baselines). On CVC-ClinicDB, Polyp-PVT retains a significant edge, while Tree-NET remains competitive across all backbones.

### Computational Efficiency (Batch Size 1)

| Model | Variant | FLOPs (GFLOP) | Params (M) | Peak Memory (GB) |
|---|---|---|---|---|
| Polyp-PVT BB | Tree-NET | **2.54** | 25.17 | **1.402** |
| Polyp-PVT BB | Original | 11.92 | 25.11 | 1.440 |
| U-NET | Tree-NET | **2.85** | 7.88 | **1.081** |
| U-NET | BS U-NET | 33.58 | 7.88 | 1.976 |
| U-NET | Original | 31.73 | 7.85 | 4.386 |
| U-NET++ | Tree-NET | **5.77** | 9.19 | **1.322** |
| U-NET++ | Original | 78.53 | 9.16 | 2.516 |

### Inference Speed (NVIDIA RTX 5070 Ti)

| Model | Variant | Batch1 Latency (ms) | Batch8 Throughput (img/s) |
|---|---|---|---|
| Polyp-PVT BB | Tree-NET | 20.77 | 297.77 |
| Polyp-PVT BB | Original | 18.09 | 252.97 |
| U-NET | Tree-NET | 4.07 | **772.14** |
| U-NET | Original | 4.04 | 242.25 |
| U-NET++ | Tree-NET | 6.13 | **570.81** |
| U-NET++ | Original | 10.29 | 87.50 |

---

## Citation

If you use this codebase in your research, please cite:

```bibtex
@article{treenet2025,
  title   = {Tree-NET: Enhancing 2D Medical Image Segmentation through Efficient Low-Level Feature Training},
  author  = {Anonymous},
  journal = {Under Review},
  year    = {2025}
}
```

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.