# Brain Tumor AI

A production-ready deep learning pipeline for brain tumor detection and classification from MRI scans. Based on the Neuro-HyMamba theoretical framework, this system implements a **Segmentation-then-Classification** workflow for accurate, clinically-validated tumor diagnosis.

## 🎯 Overview

This pipeline achieves high-accuracy brain tumor classification through:

1. **Preprocessing**: N4 bias field correction + Z-score normalization
2. **Segmentation**: 3D UNet for tumor localization (WT/TC/ET)
3. **ROI Extraction**: Bounding box cropping of tumor regions
4. **Classification**: ConvNeXt/Swin Transformer for tumor type prediction

```
Raw MRI → Preprocessing → Segmentation → ROI Extraction → Classification → Diagnosis
```

## ⚙️ Features

- **Patient-Level Splitting**: Strict data separation to prevent leakage
- **Mixed Precision Training**: Optimized for GPU memory efficiency
- **Multi-Modal Support**: T1, T2, FLAIR, T1ce MRI sequences
- **MONAI Integration**: Medical imaging best practices
- **timm Backbones**: State-of-the-art ConvNeXt and Swin models
- **Comprehensive Metrics**: Dice, AUC-ROC, confusion matrices, per-patient analysis

## 📁 Project Structure

```
brain-tumor-ai/
├── data/
│   ├── raw/                     # Original MRI per patient
│   ├── processed/               # Preprocessed volumes
│   ├── masks/                   # Segmentation outputs
│   └── roi/                     # Cropped tumor ROIs
├── models/
│   ├── classification_best.pt   # Best classification checkpoint
│   └── segmentation_best.pt     # Best segmentation checkpoint (finetuned)
├── docs/                        # Consolidated documentation
│   ├── classification_architecture.md
│   ├── segmentation_architecture.md
│   ├── inference.md             # Inference module details
│   ├── deployment.md            # Deployment guides
│   └── results.md               # Detailed results and methodology
├── preprocessing/
│   ├── n4_bias.py               # N4 bias field correction
│   ├── normalize.py             # Z-score normalization
│   └── preprocess_pipeline.py   # Full preprocessing pipeline
├── segmentation/
│   ├── dataset.py               # 3D MRI dataset with MONAI transforms
│   ├── model.py                 # 3D UNet / UNETR / Swin UNETR
│   └── train.py                 # Training script
├── classification/
│   ├── dataset.py               # ROI dataset (2.5D/3D modes)
│   ├── model.py                 # ConvNeXt/Swin classifier
│   └── evaluate.py              # Evaluation script
├── inference/
│   ├── engine.py                # Production inference engine
│   └── run_inference.py         # CLI tool for inference
├── configs/
│   ├── seg.yaml                 # Segmentation config
│   └── cls.yaml                 # Classification config
├── requirements.txt
├── README.md                    # Entry point
├── demo.py                      # Interactive demo script
└── test_engine.py               # Inference engine test suite
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/your-repo/brain-tumor-ai.git
cd brain-tumor-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Organize your MRI data in the following structure:

```
data/raw/
├── patient_001/
│   ├── t1.nii.gz
│   ├── t2.nii.gz
│   ├── flair.nii.gz
│   └── t1ce.nii.gz
├── patient_002/
│   └── ...
```

Create patient split files:

```bash
# data/splits/train.txt
patient_001
patient_002
...

# data/splits/val.txt
patient_021
...

# data/splits/test.txt
patient_026
...
```

Create labels file (`data/labels.csv`):

```csv
patient_id,label
patient_001,0
patient_002,1
patient_003,2
...
```

### 3. Run Full Pipeline

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh all

# Run Interactive Demo
python demo.py

# Run Inference Engine Test
python test_engine.py
```

Or run individual stages:

```bash
./run_pipeline.sh preprocess   # Preprocessing only
./run_pipeline.sh seg-train    # Train segmentation
./run_pipeline.sh seg-infer    # Run segmentation inference
./run_pipeline.sh roi          # Extract ROIs
./run_pipeline.sh cls-train    # Train classifier
./run_pipeline.sh evaluate     # Evaluate on test set
```

## 📖 Detailed Usage

### Preprocessing

```bash
python -m preprocessing.preprocess_pipeline \
    --raw-dir data/raw \
    --output-dir data/processed \
    --modalities t1 t2 flair t1ce \
    --normalize-method zscore \
    --num-workers 4
```

### Segmentation Training

```bash
python -m segmentation.train --config configs/seg.yaml
```

Resume from checkpoint:

```bash
python -m segmentation.train \
    --config configs/seg.yaml \
    --resume checkpoints/segmentation/checkpoint_epoch_50.pt
```

### Segmentation Inference

```bash
python -m segmentation.infer \
    --checkpoint models/segmentation_best.pt \
    --patient-list data/splits/test.txt \
    --processed-dir data/processed \
    --output-dir data/masks
```

### ROI Extraction

```bash
python -m roi_extraction.extract_roi \
    --patient-list data/splits/train.txt \
    --processed-dir data/processed \
    --masks-dir data/masks \
    --output-dir data/roi \
    --margin 10
```

### Classification Training

```bash
python -m classification.train --config configs/cls.yaml
```

### Evaluation

```bash
python -m classification.evaluate \
    --checkpoint models/classification_best.pt \
    --patient-list data/splits/test.txt \
    --roi-dir data/roi \
    --labels-file data/labels.csv \
    --output-dir results/evaluation
```

## ⚙️ Configuration

### Segmentation Config (`configs/seg.yaml`)

Key parameters:

```yaml
model:
  name: "UNet"
  in_channels: 4       # T1, T2, FLAIR, T1ce
  out_channels: 4      # Background, WT, TC, ET
  channels: [32, 64, 128, 256, 512]

training:
  batch_size: 2
  epochs: 300
  learning_rate: 1.0e-4

loss:
  name: "DiceFocalLoss"
  gamma: 2.0

augmentation:
  roi_size: [128, 128, 128]
```

### Classification Config (`configs/cls.yaml`)

Key parameters:

```yaml
model:
  backbone: "convnext_base"  # or swin_base_patch4_window7_224
  in_channels: 4
  mode: "2.5d"               # or "3d"

training:
  batch_size: 8
  epochs: 100

loss:
  name: "FocalLoss"
  gamma: 2.0
  label_smoothing: 0.1

augmentation:
  num_slices: 16
  input_size: [224, 224]
```

## 💻 Hardware Requirements

### Target Platform: AWS EC2 g5.12xlarge

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA A10G (24 GB VRAM) |
| CPU | 48 vCPUs |
| RAM | 192 GB |
| Storage | 500+ GB SSD |

### GPU Memory Usage (Approximate)

| Model | Batch Size | VRAM Usage |
|-------|------------|------------|
| 3D UNet (Segmentation) | 2 | ~12 GB |
| ConvNeXt-Base (Classification) | 8 | ~8 GB |
| Swin UNETR (Segmentation) | 1 | ~20 GB |

### Memory Optimization Tips

1. **Use gradient checkpointing** for large models
2. **Reduce batch size** if OOM occurs
3. **Enable mixed precision** (default)
4. **Use sliding window inference** for full-volume segmentation

## ⚠️ Common Issues

### 1. CUDA Out of Memory

```bash
# Reduce batch size
# In configs/seg.yaml:
training:
  batch_size: 1

# Or use gradient checkpointing (for Swin UNETR)
model:
  use_checkpoint: true
```

### 2. Data Leakage Warning

Ensure patient-level splits:

```python
from utils import validate_no_leakage

validate_no_leakage(
    "data/splits/train.txt",
    "data/splits/val.txt",
    "data/splits/test.txt"
)
```

### 3. Missing Modalities

If some patients lack certain modalities:

```yaml
# Use modality dropout during training
augmentation:
  modality_dropout: 0.2
```

### 4. Slow Preprocessing

Use parallel processing:

```bash
python -m preprocessing.preprocess_pipeline \
    --raw-dir data/raw \
    --output-dir data/processed \
    --num-workers 8
```

## 📊 Expected Results

With proper training on BraTS-like data:

| Metric | Segmentation | Classification |
|--------|-------------|----------------|
| Dice (WT) | 0.88+ | - |
| Dice (TC) | 0.82+ | - |
| Dice (ET) | 0.78+ | - |
| Accuracy | - | 95%+ |
| AUC-ROC | - | 0.98+ |

## 🔬 Evaluation Outputs

After running evaluation, you'll find:

```
results/evaluation/
├── metrics.json              # All numerical metrics
├── classification_report.txt # Per-class precision/recall/F1
├── predictions.csv           # Per-patient predictions
├── confusion_matrix.png      # Visualization
├── roc_curves.png            # ROC curves per class
└── patient_analysis.png      # Confidence analysis
```

## 📚 References

Based on the Neuro-HyMamba framework principles:

- **Segmentation**: MONAI 3D UNet with Dice+Focal loss
- **Classification**: timm ConvNeXt/Swin with Focal loss
- **Preprocessing**: N4ITK bias correction, Z-score normalization
- **Evaluation**: Patient-level cross-validation, no slice leakage

## 📝 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request
