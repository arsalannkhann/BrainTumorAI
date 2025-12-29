# 🐍 Setting Up Mamba-SSM & Causal-Conv1d

The `BrainTumorAI` project utilizes State Space Models (Mamba) for advanced feature extraction. These dependencies require compilation from source and are hardware-dependent.

## Prerequisites

- **NVIDIA GPU** with CUDA support.
- **CUDA Toolkit** installed (matching your PyTorch CUDA version).
- **PyTorch** >= 2.1.0.

## Installation Steps

If you are setting up a fresh environment, follow these steps:

1. **Install PyTorch** (if not already present):
   ```bash
   pip install torch torchvision
   ```

2. **Install Compilation Dependencies**:
   ```bash
   pip install packaging
   ```

3. **Install Mamba Dependencies**:
   > [!IMPORTANT]
   > These commands may take several minutes as they compile CUDA kernels.
   
   ```bash
   pip install causal-conv1d>=1.0.0
   pip install mamba-ssm>=2.0.0
   ```

## Common Issues

### "NameError: name 'bare_metal_version' is not defined"
This usually happens when `nvcc` is not in your `PATH` or the CUDA Toolkit is missing. Verify with:
```bash
nvcc --version
```

### CUDA Version Mismatch
Ensure `torch.version.cuda` matches the version of `nvcc`.

## Verification
You can verify the installation by running:
```bash
python -c "import mamba_ssm; import causal_conv1d; print('Mamba is ready!')"
```
