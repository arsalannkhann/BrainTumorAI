"""
Inference script for brain tumor segmentation.
Generates segmentation masks for new patients.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from torch.amp import autocast
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from segmentation.dataset import create_inference_dataset
from segmentation.model import load_model_from_checkpoint, SegmentationModel
from utils import load_config, load_patient_list, set_seed


def run_inference(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    roi_size: Tuple[int, int, int] = (128, 128, 128),
    sw_batch_size: int = 4,
    overlap: float = 0.5,
    use_amp: bool = True,
) -> np.ndarray:
    """
    Run inference on a single volume.
    
    Args:
        model: Segmentation model
        image: Input tensor (1, C, H, W, D)
        device: Inference device
        roi_size: ROI size for sliding window
        sw_batch_size: Batch size for sliding window
        overlap: Overlap ratio
        use_amp: Whether to use AMP
        
    Returns:
        Segmentation mask (H, W, D)
    """
    model.eval()
    image = image.to(device)
    
    with torch.no_grad():
        with autocast("cuda", enabled=use_amp):
            outputs = sliding_window_inference(
                image,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap,
            )
    
    # Get predicted class
    pred = torch.argmax(outputs, dim=1).squeeze(0)
    return pred.cpu().numpy().astype(np.uint8)


def run_inference_with_tta(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    roi_size: Tuple[int, int, int] = (128, 128, 128),
    sw_batch_size: int = 4,
    overlap: float = 0.5,
    use_amp: bool = True,
) -> np.ndarray:
    """
    Run inference with Test-Time Augmentation (TTA).
    
    Applies flip augmentations along each axis and averages predictions.
    Typically provides ~1-2% Dice improvement.
    
    Args:
        model: Segmentation model
        image: Input tensor (1, C, H, W, D)
        device: Inference device
        roi_size: ROI size for sliding window
        sw_batch_size: Batch size for sliding window
        overlap: Overlap ratio
        use_amp: Whether to use AMP
        
    Returns:
        Segmentation mask (H, W, D)
    """
    model.eval()
    image = image.to(device)
    
    # TTA: original + 3 axis flips
    flip_dims = [None, (2,), (3,), (4,)]  # No flip, flip H, flip W, flip D
    
    accumulated_probs = None
    
    with torch.no_grad():
        for flip_dim in flip_dims:
            # Apply flip
            if flip_dim is not None:
                aug_image = torch.flip(image, dims=flip_dim)
            else:
                aug_image = image
            
            # Run inference
            with autocast("cuda", enabled=use_amp):
                outputs = sliding_window_inference(
                    aug_image,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=overlap,
                )
            
            # Apply softmax to get probabilities
            probs = torch.softmax(outputs, dim=1)
            
            # Reverse flip
            if flip_dim is not None:
                probs = torch.flip(probs, dims=flip_dim)
            
            # Accumulate
            if accumulated_probs is None:
                accumulated_probs = probs
            else:
                accumulated_probs = accumulated_probs + probs
    
    # Average and get predicted class
    averaged_probs = accumulated_probs / len(flip_dims)
    pred = torch.argmax(averaged_probs, dim=1).squeeze(0)
    
    return pred.cpu().numpy().astype(np.uint8)


def infer_patient(
    model: torch.nn.Module,
    patient_id: str,
    processed_dir: Path,
    output_dir: Path,
    device: torch.device,
    roi_size: Tuple[int, int, int] = (128, 128, 128),
    sw_batch_size: int = 4,
    overlap: float = 0.5,
    use_amp: bool = True,
    use_tta: bool = False,
    affine: Optional[np.ndarray] = None,
) -> Path:
    """
    Run inference for a single patient.
    
    Args:
        model: Segmentation model
        patient_id: Patient identifier
        processed_dir: Directory with preprocessed volumes
        output_dir: Output directory for masks
        device: Inference device
        roi_size: ROI size for sliding window
        sw_batch_size: Batch size for sliding window
        overlap: Overlap ratio
        use_amp: Whether to use AMP
        use_tta: Whether to use Test-Time Augmentation
        affine: Optional affine matrix for output
        
    Returns:
        Path to saved segmentation mask
    """
    # Load volume
    volume_path = processed_dir / patient_id / f"{patient_id}.npy"
    if not volume_path.exists():
        volume_path = processed_dir / f"{patient_id}.npy"
    
    if not volume_path.exists():
        raise FileNotFoundError(f"Volume not found for patient: {patient_id}")
    
    volume = np.load(volume_path).astype(np.float32)
    image = torch.from_numpy(volume).unsqueeze(0)  # (1, C, H, W, D)
    
    # Run inference (with or without TTA)
    inference_fn = run_inference_with_tta if use_tta else run_inference
    mask = inference_fn(
        model=model,
        image=image,
        device=device,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        use_amp=use_amp,
    )
    
    # Save mask
    patient_output = output_dir / patient_id
    patient_output.mkdir(parents=True, exist_ok=True)
    
    mask_path = patient_output / "seg.nii.gz"
    
    if affine is None:
        affine = np.eye(4)
    
    mask_img = nib.Nifti1Image(mask.astype(np.uint8), affine)
    nib.save(mask_img, str(mask_path))
    
    return mask_path


def run_batch_inference(
    checkpoint_path: str,
    patient_ids: List[str],
    processed_dir: Union[str, Path],
    output_dir: Union[str, Path],
    config_path: Optional[str] = None,
    roi_size: Tuple[int, int, int] = (128, 128, 128),
    sw_batch_size: int = 4,
    overlap: float = 0.5,
    use_amp: bool = True,
    use_tta: bool = False,
    device: Optional[str] = None,
) -> None:
    """
    Run inference for multiple patients.
    
    Args:
        checkpoint_path: Path to model checkpoint
        patient_ids: List of patient identifiers
        processed_dir: Directory with preprocessed volumes
        output_dir: Output directory for masks
        config_path: Optional path to config for model settings
        roi_size: ROI size for sliding window
        sw_batch_size: Batch size for sliding window
        overlap: Overlap ratio
        use_amp: Whether to use AMP
        device: Device to use (auto-detect if None)
    """
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device setup
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # Load model configuration
    model_kwargs = {}
    if config_path:
        config = load_config(config_path)
        model_kwargs = config.get("model", {})
        roi_size = tuple(config.get("augmentation", {}).get("roi_size", roi_size))
    
    # Load model
    print(f"Loading model from: {checkpoint_path}")
    model = load_model_from_checkpoint(
        checkpoint_path,
        device=str(device),
        **model_kwargs,
    )
    
    print(f"Processing {len(patient_ids)} patients...")
    
    # Run inference
    successful = 0
    failed = []
    
    for patient_id in tqdm(patient_ids, desc="Inference"):
        try:
            mask_path = infer_patient(
                model=model,
                patient_id=patient_id,
                processed_dir=processed_dir,
                output_dir=output_dir,
                device=device,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                overlap=overlap,
                use_amp=use_amp,
                use_tta=use_tta,
            )
            successful += 1
        except Exception as e:
            print(f"Error processing {patient_id}: {e}")
            failed.append(patient_id)
    
    # Summary
    print(f"\nInference complete!")
    print(f"Successful: {successful}/{len(patient_ids)}")
    if failed:
        print(f"Failed: {failed}")


def main():
    parser = argparse.ArgumentParser(
        description="Brain Tumor Segmentation Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run inference on test patients
  python -m segmentation.infer --checkpoint checkpoints/segmentation/best_model.pt \\
      --patient-list data/splits/test.txt \\
      --processed-dir data/processed \\
      --output-dir data/masks

  # Run inference on a single patient
  python -m segmentation.infer --checkpoint checkpoints/segmentation/best_model.pt \\
      --patients patient_001 patient_002 \\
      --processed-dir data/processed \\
      --output-dir data/masks
        """,
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--patient-list",
        type=Path,
        help="Path to text file with patient IDs",
    )
    parser.add_argument(
        "--patients",
        nargs="+",
        help="List of patient IDs (alternative to --patient-list)",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        required=True,
        help="Directory containing preprocessed volumes",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for segmentation masks",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (for model settings)",
    )
    parser.add_argument(
        "--roi-size",
        type=int,
        nargs=3,
        default=[128, 128, 128],
        help="ROI size for sliding window inference",
    )
    parser.add_argument(
        "--sw-batch-size",
        type=int,
        default=4,
        help="Batch size for sliding window inference",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Overlap ratio for sliding window",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable Test-Time Augmentation (~1-2%% Dice boost, 4x slower)",
    )
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Get patient list
    if args.patient_list:
        patient_ids = load_patient_list(args.patient_list)
    elif args.patients:
        patient_ids = args.patients
    else:
        raise ValueError("Must provide either --patient-list or --patients")
    
    # Run inference
    run_batch_inference(
        checkpoint_path=args.checkpoint,
        patient_ids=patient_ids,
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        roi_size=tuple(args.roi_size),
        sw_batch_size=args.sw_batch_size,
        overlap=args.overlap,
        use_amp=not args.no_amp,
        use_tta=args.tta,
        device=args.device,
    )


if __name__ == "__main__":
    main()
