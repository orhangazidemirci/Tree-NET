# PolypPVT — Polyp Segmentation Training Framework

A multi-architecture training and benchmarking framework for medical image segmentation, with a focus on polyp detection in colonoscopy images. Supports PVT (Pyramid Vision Transformer), U-Net, and NestedU-Net backbones, along with encoder/decoder component experiments.

---

## Features

- **Multi-architecture support**: Train with `pvt`, `UNET`, or `NestedUNET`
- **Multi-scale training**: Automatically trains at resolutions 256×256, 384×384, and 512×512 per batch
- **Structure loss**: Weighted BCE + IoU loss for boundary-aware segmentation
- **Component experiments**: Isolated encoder/decoder experiments via `components_exp`
- **Automated benchmarking**: End-to-end inference evaluation with `run_benchmark`
- **Flexible datasets**: Supports `CVC-ClinicDB` and `ISAC2018` out of the box

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
    ├── encoder.py            # components_exp for ablations
    └── utils.py              # clip_gradient, adjust_lr, AvgMeter
```

---

## Installation

```bash
git clone https://github.com/your-username/polyp-pvt.git
cd polyp-pvt
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
└── ISAC2018/
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

---

## Usage

### Basic Training

```bash
python train.py \
  --arch pvt \
  --dataset ISAC2018 \
  --epoch 100 \
  --batchsize 8 \
  --lr 1e-4 \
  --optimizer AdamW
```

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--arch` | `UNET` | Model architecture: `pvt`, `UNET`, `NestedUNET` |
| `--dataset` | `ISAC2018` | Dataset: `CVC-ClinicDB` or `ISAC2018` |
| `--epoch` | `100` | Number of training epochs |
| `--batchsize` | `8` | Training batch size |
| `--lr` | `1e-4` | Learning rate |
| `--optimizer` | `AdamW` | `AdamW` or `SGD` |
| `--trainsize` | `384` | Base training image size |
| `--clip` | `0.5` | Gradient clipping margin |
| `--augmentation` | `True` | Enable random flip/rotation augmentation |
| `--encoder_component` | `Encoder_x4` | Encoder variant for component experiments |
| `--decoder_component` | `Decoder_x16` | Decoder variant: `Decoder_x4`, `Decoder_x8`, `Decoder_x16` |
| `--component_selected` | `ORIGINAL` | Experiment mode: `ORIGINAL`, `Encoder_x4`, `Decoder_x4`, etc. |

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

**Structure Loss** (used for PVT):  
Weighted BCE + weighted IoU, with edge-aware weighting computed via average pooling on the ground truth mask.

**Euclidean Loss** (used for U-Net / NestedU-Net):  
Mean squared error between predicted and ground truth masks.

---

## Outputs

| Output | Location |
|---|---|
| Model checkpoints | `./model_pth/<arch>/` |
| Best model | `./model_pth/<arch>/_<dataset>best.pth` |
| Training logs | `train_log<component><dataset><arch><mode>.log` |
| Benchmark results | Printed to stdout / logged |

---

## Evaluation Metric

Dice Similarity Coefficient (DSC) is used as the primary metric:

$$\text{DSC} = \frac{2 \cdot |P \cap G| + \epsilon}{|P| + |G| + \epsilon}$$

The best model checkpoint is saved whenever validation DSC improves.

---

## Citation

If you use this codebase in your research, please cite the original PolypPVT paper:

```bibtex
@article{tang2023polyppvt,
  title   = {PolypPVT: Polyp Segmentation with Pyramid Vision Transformers},
  author  = {Tang, Bo and others},
  journal = {arXiv preprint},
  year    = {2023}
}
```

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
