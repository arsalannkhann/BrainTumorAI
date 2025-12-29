# Deployment Quick Reference

## Pre-Deployment Checklist

- [ ] Checkpoints copied to `checkpoints/` directory
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] GPU available (or set `DEVICE=cpu`)
- [ ] Configs match checkpoint model architecture

---

## Minimum Requirements

- Python 3.10+
- PyTorch 2.0+ (with CUDA for GPU)
- ~16GB RAM
- ~8GB VRAM (for GPU inference)

---

## 1-Minute Deploy (Docker)

```bash
# Build
docker build -t brain-tumor-api .

# Run (GPU)
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  brain-tumor-api

# Test
curl http://localhost:8000/health
```

---

## Manual Deploy

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment
export CLS_CHECKPOINT=checkpoints/classification/best_model.pt
export SEG_CHECKPOINT=checkpoints/segmentation_finetuned/best_model.pt
export SEG_CONFIG=configs/seg_finetune.yaml

# Run API
uvicorn inference.api:app --host 0.0.0.0 --port 8000

# Or CLI batch
python -m inference.run_inference batch \
  --input-dir /data/input \
  --output-dir /data/output \
  --save-masks
```

---

## Key Files

```
inference/
├── README.md           # Full documentation (you are here)
├── DEPLOY.md           # This quickstart
├── engine.py           # Core inference (import this)
├── run_inference.py    # CLI tool
├── api.py              # REST API
└── xai.py              # Explainability

checkpoints/            # Required!
├── classification/
│   └── best_model.pt
└── segmentation_finetuned/
    └── best_model.pt
```

---

## Test Before Deploy

```bash
# Quick sanity check
python test_inference_engine.py

# Expected output:
# ✓ Classification inference PASSED
# ✓ Segmentation inference PASSED
# ✓ Validation logic PASSED
# ALL TESTS PASSED ✓
```

---

## Common Issues

| Problem | Solution |
|---------|----------|
| No GPU | Set `DEVICE=cpu` env var |
| Wrong shape | Classification: `(3,H,W)`, Segmentation: `(4,H,W,D)` |
| OOM | Enable sliding window: `use_sliding_window=True` |

---

## API Endpoints Summary

```
GET  /health     → {"status": "healthy", ...}
POST /classify   → {"predicted_class": "Glioma", ...}
POST /segment    → {"tumor_area_percentage": 12.5, ...}
POST /infer      → Full report with validation
POST /batch      → Multiple images
```
