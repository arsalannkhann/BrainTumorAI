# Brain Tumor Inference Module

Production-grade inference engine for brain tumor MRI analysis.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Module Overview](#module-overview)
3. [CLI Usage](#cli-usage)
4. [Python API](#python-api)
5. [REST API](#rest-api)
6. [Explainability (XAI)](#explainability-xai)
7. [Deployment](#deployment)
8. [Checkpoints](#checkpoints)
9. [Input/Output Formats](#inputoutput-formats)
10. [Troubleshooting](#troubleshooting)

---

## Quick Start

### CLI Inference (Recommended)

```bash
# Single image inference
python -m inference.run_inference single \
  --input data/processed/patient001/patient001.npy \
  --output results/patient001.json

# Batch inference
python -m inference.run_inference batch \
  --input-dir data/processed \
  --output-dir results \
  --save-masks
```

### Python API

```python
from inference import create_inference_engine

# Create engine (auto-loads checkpoints)
engine = create_inference_engine()

# Run inference
import numpy as np
image = np.load("patient001.npy")  # (4, 128, 128, 128)
report = engine.run_inference(image, image_id="patient001")

# Access results
print(f"Classification: {report.classification.predicted_class.name}")
print(f"Confidence: {report.classification.confidence_score:.2%}")
print(f"Tumor area: {report.segmentation.tumor_area_percentage:.2f}%")
print(f"Needs review: {report.validation.requires_manual_review}")
```

---

## Module Overview

| File | Description |
|------|-------------|
| `engine.py` | Core inference engine with classification + segmentation |
| `run_inference.py` | CLI tool for single/batch inference |
| `api.py` | FastAPI REST endpoints |
| `xai.py` | Explainability tools (Grad-CAM, uncertainty) |
| `__init__.py` | Module exports |

### Core Classes

```
BrainTumorInferenceEngine
├── classify(image) → ClassificationResult
├── segment(image) → SegmentationResult
├── validate(cls_result, seg_result) → ValidationResult
└── run_inference(image) → InferenceReport
```

---

## CLI Usage

### Single Image

```bash
python -m inference.run_inference single \
  --input <path/to/image.npy> \
  --output <path/to/report.json> \
  [--seg-only]              # Skip classification
  [--cls-only]              # Skip segmentation
  [--save-mask]             # Export segmentation mask
  [--format json|text]      # Report format
```

### Batch Processing

```bash
python -m inference.run_inference batch \
  --input-dir <path/to/images/> \
  --output-dir <path/to/results/> \
  [--save-masks]            # Export all masks
  [--pattern "*.npy"]       # File pattern
  [--format json|text]      # Report format
```

### Examples

```bash
# Process test set with mask export
python -m inference.run_inference batch \
  --input-dir data/processed \
  --output-dir data/masks \
  --save-masks \
  --format json

# Single ROI classification only
python -m inference.run_inference single \
  --input data/roi/glioma_001.npy \
  --output results/glioma_001.json \
  --cls-only
```

---

## Python API

### Basic Usage

```python
from inference import BrainTumorInferenceEngine, TumorClass

# Initialize with custom checkpoints
engine = BrainTumorInferenceEngine(
    cls_checkpoint="checkpoints/classification/best_model.pt",
    seg_checkpoint="checkpoints/segmentation/best_model.pt",
    seg_config="configs/seg.yaml",
    device="cuda",
)

# Classification only
import numpy as np
roi_image = np.load("roi_image.npy")  # (3, 224, 224)
cls_result = engine.classify(roi_image)
print(f"Class: {cls_result.predicted_class.name}")
print(f"Probabilities: {cls_result.class_probabilities}")

# Segmentation only
mri_volume = np.load("mri_volume.npy")  # (4, 128, 128, 128)
seg_result = engine.segment(mri_volume, use_sliding_window=True)
print(f"Tumor area: {seg_result.tumor_area_percentage:.2f}%")
print(f"Edema voxels: {seg_result.edema_mask.sum()}")

# Combined with validation
report = engine.run_inference(
    mri_volume,
    image_id="patient001",
    run_classification=True,
    run_segmentation=True,
)

# Check validation
if report.validation.requires_manual_review:
    print("⚠️ Requires manual review:")
    for note in report.validation.notes:
        print(f"  - {note}")
```

### Result Classes

```python
from inference import (
    ClassificationResult,  # Classification output
    SegmentationResult,    # Segmentation masks
    ValidationResult,      # Quality checks
    InferenceReport,       # Combined report
    TumorClass,            # Enum: GLIOMA, MENINGIOMA, PITUITARY, NO_TUMOR
    SegmentationClass,     # Enum: BACKGROUND, NECROTIC_CORE, EDEMA, ENHANCING
)

# Report serialization
json_str = report.to_dict()  # → dict
text_str = report.to_text()  # → formatted string
```

---

## REST API

### Start Server

```bash
# Development
uvicorn inference.api:app --host 0.0.0.0 --port 8000 --reload

# Production
gunicorn inference.api:app -w 4 -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 --timeout 300
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/classify` | Classification only |
| POST | `/segment` | Segmentation only |
| POST | `/infer` | Full inference |
| POST | `/batch` | Batch inference |

### Request Format

```json
{
  "image_base64": "<base64-encoded numpy array>",
  "image_id": "patient001",
  "run_classification": true,
  "run_segmentation": true
}
```

### Example with curl

```bash
# Health check
curl http://localhost:8000/health

# Full inference
python -c "
import numpy as np
import base64
import requests

# Prepare image
image = np.load('patient001.npy')
image_bytes = image.tobytes()
image_base64 = base64.b64encode(image_bytes).decode()

# Send request
response = requests.post(
    'http://localhost:8000/infer',
    json={
        'image_base64': image_base64,
        'image_id': 'patient001',
        'run_classification': True,
        'run_segmentation': True,
    }
)
print(response.json())
"
```

---

## Explainability (XAI)

### Grad-CAM Visualization

```python
from inference import GradCAM, visualize_gradcam
import torch

# Create Grad-CAM
gradcam = GradCAM(engine.cls_model)

# Generate heatmap
input_tensor = torch.from_numpy(image).unsqueeze(0).cuda()
heatmap, pred_class, confidence = gradcam(input_tensor)

# Visualize
overlay = visualize_gradcam(image[1], heatmap, alpha=0.5)  # Use T1 channel
```

### Uncertainty Estimation

```python
from inference import MCDropoutUncertainty

# MC Dropout
mc_estimator = MCDropoutUncertainty(engine.cls_model, num_samples=20)
result = mc_estimator.predict_with_uncertainty(input_tensor)

print(f"Mean prediction: {result['mean_prediction']}")
print(f"Uncertainty: {result['uncertainty']:.4f}")
print(f"Entropy: {result['entropy']:.4f}")
```

### Test-Time Augmentation

```python
from inference import TestTimeAugmentation

# TTA for robust prediction
tta = TestTimeAugmentation(engine.cls_model, augmentations=["flip_h", "flip_v"])
result = tta.predict(input_tensor, aggregate="mean")

print(f"Aggregated prediction: {result['prediction']}")
print(f"Prediction variance: {result['variance']:.4f}")
```

---

## Deployment

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download/copy checkpoints
COPY checkpoints/ checkpoints/

EXPOSE 8000
CMD ["uvicorn", "inference.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  brain-tumor-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./checkpoints:/app/checkpoints
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CLS_CHECKPOINT` | `checkpoints/classification/best_model.pt` | Classification model path |
| `SEG_CHECKPOINT` | `checkpoints/segmentation/best_model.pt` | Segmentation model path |
| `SEG_CONFIG` | `configs/seg.yaml` | Segmentation config path |
| `DEVICE` | `cuda` | Inference device |
| `CONFIDENCE_THRESHOLD` | `0.85` | Low confidence threshold |

---

## Checkpoints

### Required Files

```
checkpoints/
├── classification/
│   └── best_model.pt      # Classification model
└── segmentation/
    └── best_model.pt      # Segmentation model
```

### Checkpoint Format

```python
# Classification checkpoint
{
    "model_state_dict": {...},
    "model_config": {
        "num_classes": 4,
        "backbone": "resnet50",
        ...
    },
    "val_metrics": {"accuracy": 0.95, ...},
}

# Segmentation checkpoint
{
    "model_state_dict": {...},
    "model_config": {
        "name": "UNet",
        "in_channels": 4,
        "out_channels": 4,
        ...
    },
    "val_metrics": {"mean_dice": 0.74, ...},
}
```

---

## Input/Output Formats

### Input

| Task | Shape | Dtype | Description |
|------|-------|-------|-------------|
| Classification | `(3, H, W)` | float32 | RGB ROI image, normalized 0-1 |
| Segmentation | `(4, H, W, D)` | float32 | 4-channel MRI (T1, T1ce, T2, FLAIR) |

### Output

#### Classification Result
```python
ClassificationResult(
    predicted_class=TumorClass.GLIOMA,
    confidence_score=0.95,
    class_probabilities={
        "Glioma": 0.95,
        "Meningioma": 0.03,
        "Pituitary": 0.01,
        "No Tumor": 0.01,
    },
    is_low_confidence=False,
    raw_logits=np.array([...]),
)
```

#### Segmentation Result
```python
SegmentationResult(
    edema_mask=np.array(...),       # (H, W, D) uint8
    enhancing_mask=np.array(...),   # (H, W, D) uint8
    necrotic_mask=np.array(...),    # (H, W, D) uint8
    combined_mask=np.array(...),    # (H, W, D) uint8, values 0-3
    tumor_area_percentage=12.5,
    raw_logits=np.array(...),       # (1, 4, H, W, D) float32
)
```

#### Validation Result
```python
ValidationResult(
    confidence_passed=True,
    segmentation_plausible=True,
    mask_alignment_valid=True,
    anatomical_plausibility=True,
    notes=[],
)
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `Model not found` | Check checkpoint paths and environment variables |
| `CUDA OOM` | Reduce batch size or use `use_sliding_window=True` |
| `Low confidence` | Check input normalization and channel order |
| `Empty segmentation` | Verify input shape is `(4, H, W, D)` |

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
# export LOG_LEVEL=DEBUG
```

### Validation Notes

The engine returns validation notes when issues are detected:

| Note | Meaning |
|------|---------|
| `"Low classification confidence"` | Confidence < 85% |
| `"Tumor classified but segmentation empty"` | Mismatch between cls/seg |
| `"Tumor area exceeds plausible threshold"` | > 50% tumor area |

---

## License

See repository LICENSE file.
