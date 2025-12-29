"""
Intensity normalization for MRI volumes.
Implements Z-score normalization per patient and per modality.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np


def zscore_normalize(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    clip_range: Optional[Tuple[float, float]] = (-5.0, 5.0),
) -> np.ndarray:
    """
    Apply Z-score normalization to a 3D volume.
    
    Normalizes using the mean and standard deviation of non-zero voxels
    (or masked voxels if mask is provided).
    
    Args:
        image: Input 3D volume
        mask: Optional brain mask (non-zero = brain)
        clip_range: Optional (min, max) to clip normalized values
        
    Returns:
        Z-score normalized image
    """
    if mask is not None:
        brain_voxels = image[mask > 0]
    else:
        brain_voxels = image[image > 0]
    
    if len(brain_voxels) == 0:
        return image
    
    mean = np.mean(brain_voxels)
    std = np.std(brain_voxels)
    
    if std < 1e-8:
        # Avoid division by zero
        return image - mean
    
    normalized = (image - mean) / std
    
    # Apply clipping if specified
    if clip_range is not None:
        normalized = np.clip(normalized, clip_range[0], clip_range[1])
    
    # Zero out background
    if mask is not None:
        normalized = normalized * (mask > 0)
    else:
        normalized = normalized * (image > 0)
    
    return normalized.astype(np.float32)


def percentile_normalize(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    output_range: Tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """
    Apply percentile-based intensity normalization.
    
    Args:
        image: Input 3D volume
        mask: Optional brain mask
        lower_percentile: Lower percentile for clipping
        upper_percentile: Upper percentile for clipping
        output_range: Desired output intensity range
        
    Returns:
        Normalized image
    """
    if mask is not None:
        brain_voxels = image[mask > 0]
    else:
        brain_voxels = image[image > 0]
    
    if len(brain_voxels) == 0:
        return image
    
    p_low = np.percentile(brain_voxels, lower_percentile)
    p_high = np.percentile(brain_voxels, upper_percentile)
    
    if p_high - p_low < 1e-8:
        return np.zeros_like(image)
    
    # Clip and scale
    clipped = np.clip(image, p_low, p_high)
    normalized = (clipped - p_low) / (p_high - p_low)
    normalized = normalized * (output_range[1] - output_range[0]) + output_range[0]
    
    # Zero out background
    if mask is not None:
        normalized = normalized * (mask > 0)
    else:
        normalized = normalized * (image > 0)
    
    return normalized.astype(np.float32)


def normalize_multimodal(
    volumes: np.ndarray,
    mask: Optional[np.ndarray] = None,
    method: str = "zscore",
    **kwargs,
) -> np.ndarray:
    """
    Normalize multi-modal MRI data (each modality independently).
    
    Args:
        volumes: Multi-modal volume (C, H, W, D)
        mask: Optional brain mask (H, W, D)
        method: "zscore" or "percentile"
        **kwargs: Additional arguments for normalization function
        
    Returns:
        Normalized multi-modal volume
    """
    num_modalities = volumes.shape[0]
    normalized = np.zeros_like(volumes)
    
    normalize_fn = zscore_normalize if method == "zscore" else percentile_normalize
    
    for i in range(num_modalities):
        normalized[i] = normalize_fn(volumes[i], mask=mask, **kwargs)
    
    return normalized


def normalize_patient(
    patient_dir: Union[str, Path],
    output_dir: Union[str, Path],
    modalities: List[str] = ["t1", "t2", "flair", "t1ce"],
    suffix: str = ".nii.gz",
    method: str = "zscore",
    **kwargs,
) -> None:
    """
    Normalize all modalities for a single patient.
    
    Args:
        patient_dir: Directory containing patient's MRI modalities
        output_dir: Output directory for normalized images
        modalities: List of modality names to process
        suffix: File suffix
        method: Normalization method
        **kwargs: Additional arguments for normalization
    """
    import nibabel as nib
    
    patient_dir = Path(patient_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to find a brain mask
    mask = None
    mask_names = ["brain_mask.nii.gz", "mask.nii.gz", "brainmask.nii.gz"]
    for mask_name in mask_names:
        mask_path = patient_dir / mask_name
        if mask_path.exists():
            mask = nib.load(str(mask_path)).get_fdata()
            break
    
    for modality in modalities:
        # Find the modality file
        possible_names = [
            f"{modality}{suffix}",
            f"{patient_dir.name}_{modality}{suffix}",
            f"{modality.upper()}{suffix}",
        ]
        
        input_path = None
        for name in possible_names:
            path = patient_dir / name
            if path.exists():
                input_path = path
                break
        
        if input_path is None:
            print(f"Warning: Modality '{modality}' not found for {patient_dir.name}")
            continue
        
        # Load and normalize
        img = nib.load(str(input_path))
        data = img.get_fdata().astype(np.float32)
        
        if method == "zscore":
            normalized = zscore_normalize(data, mask=mask, **kwargs)
        else:
            normalized = percentile_normalize(data, mask=mask, **kwargs)
        
        # Save
        output_path = output_dir / f"{modality}{suffix}"
        normalized_img = nib.Nifti1Image(normalized, img.affine, img.header)
        nib.save(normalized_img, str(output_path))
        print(f"  Normalized {modality}")


def save_as_numpy(
    patient_dir: Union[str, Path],
    output_path: Union[str, Path],
    modalities: List[str] = ["t1", "t2", "flair", "t1ce"],
    suffix: str = ".nii.gz",
) -> None:
    """
    Stack and save multi-modal data as a single numpy file.
    
    Args:
        patient_dir: Directory containing normalized NIfTI files
        output_path: Output .npy or .npz file path
        modalities: List of modality names in order
        suffix: File suffix for NIfTI files
    """
    import nibabel as nib
    
    patient_dir = Path(patient_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    volumes = []
    affine = None
    
    for modality in modalities:
        path = patient_dir / f"{modality}{suffix}"
        if not path.exists():
            raise FileNotFoundError(f"Modality file not found: {path}")
        
        img = nib.load(str(path))
        data = img.get_fdata().astype(np.float32)
        volumes.append(data)
        
        if affine is None:
            affine = img.affine
    
    stacked = np.stack(volumes, axis=0)  # (C, H, W, D)
    
    if output_path.suffix == ".npz":
        np.savez_compressed(output_path, data=stacked, affine=affine)
    else:
        np.save(output_path, stacked)
    
    print(f"Saved multi-modal volume: {output_path} (shape: {stacked.shape})")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MRI Intensity Normalization")
    parser.add_argument("--input", "-i", required=True, help="Input patient directory")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument(
        "--method",
        choices=["zscore", "percentile"],
        default="zscore",
        help="Normalization method",
    )
    parser.add_argument("--save-numpy", action="store_true", help="Also save as numpy array")
    
    args = parser.parse_args()
    
    normalize_patient(
        args.input,
        args.output,
        method=args.method,
    )
    
    if args.save_numpy:
        patient_name = Path(args.input).name
        save_as_numpy(args.output, Path(args.output) / f"{patient_name}.npy")
    
    print("Done!")
