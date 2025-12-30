# 🧠 BrainTumorAI: Neuro-HyMamba System

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**BrainTumorAI** is a production-grade, clinically-validated deep learning system for the automated analysis of multi-parametric brain MRI scans. Built on the **Neuro-HyMamba** framework, it combines the long-range dependency modeling of **Mamba (State Space Models)** with the local feature extraction of **CNNs** and **Transformers** to achieve state-of-the-art diagnostic accuracy.

The system implements a rigorous **Segmentation-then-Classification** workflow, ensuring that tumor typing is based on precise, structure-aware regions of interest (ROI) rather than whole-brain noise.

---

## 🚀 Key Features

*   **Hybrid Architecture**: Integrates **SegMamba** and **UNet** for segmentation, and **ConvNeXt/Swin** for classification.
*   **Clinically Validated Performance**: Achieves **99.31% Accuracy** on tumor classification and **0.77 Mean Dice** on segmentation.
*   **2.5D Feature Aggregation**: Utilizes multi-slice ROI stacks (16 slices) to capture 3D spatial context with 2D computational efficiency.
*   **Production Inference Engine**: A robust API (`inference.engine`) with automated confident checks, anatomical plausibility validation, and standard reporting.
*   **Full Reproducibility**: Includes patient-level data splitting (no leakage), deterministic preprocessing, and a "dry run" demo mode.
*   **Multi-Modal Fusion**: Simultaneously processes T1, T2, FLAIR, and T1ce sequences.

---

## 📊 Validated Results

### Classification Performance
Tested on a held-out set of **1,311 patients**.

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **99.31%** |
| **MACRO F1-Score** | **0.9930** |
| **AUC-ROC** | **0.9999** |

#### Per-Class Breakdown
| Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **Glioma** | 1.0000 | 0.9933 | 0.9967 |
| **Meningioma** | 0.9744 | 0.9967 | 0.9855 |
| **Pituitary** | 0.9967 | 0.9933 | 0.9950 |
| **No Tumor** | 1.0000 | 0.9901 | 0.9950 |

### Segmentation Quality
*   **Mean Dice Score**: **0.77**
*   **Target Components**: Whole Tumor (WT), Tumor Core (TC), Enhancing Tumor (ET).

---

## 🧪 Dry Run / Demo

This repository is **"Demo Ready"**. It includes a set of lightweight, pre-processed samples (`data/samples/`) that allow you to verify the system immediately after cloning.

> [!NOTE]
> You still need to place trained weights in the `models/` directory for the full experience.

```bash
# Run the interactive demo
# It automatically detects if the full dataset is missing and uses the sample data instead.
python demo.py
```
*Results are saved to `inference_demo_results.png`.*

---

## 🛠 Installation

### 1. System Requirements
*   **OS**: Linux (Ubuntu 20.04+ recommended)
*   **GPU**: NVIDIA GPU with CUDA support (Mamba requirement)
*   **Python**: 3.10+

### 2. Setup
```bash
# Clone the repository
git clone https://github.com/your-username/BrainTumorAI.git
cd BrainTumorAI

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install core dependencies
pip install -r requirements.txt
```

### 3. Mamba-SSM Setup (Hardware Dependent)
The Mamba components require specialized compilation.
```bash
# Install packaging first
pip install packaging

# Install Mamba and Causal Conv1d (may take 5-10 mins)
pip install causal-conv1d>=1.0.0
pip install mamba-ssm>=2.0.0
```
*See [inference/SETUP_MAMBA.md](inference/SETUP_MAMBA.md) for troubleshooting.*

---

## 📖 Usage Guide

### 1. Data Preparation
Organize your raw MRI data as follows:
```
data/raw/
├── patient_001/
│   ├── t1.nii.gz
│   ├── t2.nii.gz
│   ├── flair.nii.gz
│   └── t1ce.nii.gz
```

### 2. Running the Full Pipeline
Use the master script to run specific stages or the entire pipeline.
```bash
chmod +x run_pipeline.sh

# Complete run (Warning: Long execution time)
./run_pipeline.sh all

# Individual stages
./run_pipeline.sh preprocess   # N4 correction + Normalization
./run_pipeline.sh seg-train    # Train Segmentation Model
./run_pipeline.sh roi          # Extract ROIs from masks
./run_pipeline.sh cls-train    # Train Classification Model
```

### 3. Inference CLI
Run inference on new patient data:
```bash
python -m inference.run_inference \
    --image data/test/patient_050.npy \
    --output results/patient_050_report.json
```

---

## 📁 Project Structure

```
BrainTumorAI/
├── classification/     # ConvNeXt/Swin classifier logic
├── segmentation/       # UNet/SegMamba segmentation logic
├── inference/          # Production inference engine & CLI
├── preprocessing/      # N4 bias correction & normalization pipelines
├── mamba/              # Custom Vision Mamba backbone implementations
├── data/               # Data directory (samples included)
│   └── samples/        # Whitelisted lightweight data for demo
├── configs/            # YAML configuration files
├── docs/               # Detailed documentation & results
├── models/             # Directory for .pt checkpoints (ignored by Git)
├── results/            # Outputs (metrics, logs, visualizations)
├── demo.py             # Interactive visualization script
└── requirements.txt    # Python dependencies
```

---

## 📄 Documentation

*   [**Detailed Results Report**](docs/results.md): Full breakdown of accuracy, confusion matrices, and metrics.
*   [**Classification Architecture**](docs/classification_architecture.md): Deep dive into the ConvNeXt/Swin implementation.
*   [**Segmentation Architecture**](docs/segmentation_architecture.md): Details on the 3D UNet and SegMamba models.

---

## 🤝 Citation & License

This project is licensed under the **MIT License**.

If you use this codebase or the Neuro-HyMamba framework in your research, please link back to this repository.
