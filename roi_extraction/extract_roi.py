"""
ROI Extraction from segmentation masks.
Extracts bounding box crops of tumor regions from multi-modal MRI volumes.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
from scipy import ndimage
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import load_patient_list


def get_bounding_box(
    mask: np.ndarray,
    margin: int = 10,
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """
    Get bounding box coordinates from a binary mask.
    
    Args:
        mask: Binary mask array (H, W, D)
        margin: Margin to add around the bounding box
        
    Returns:
        Tuple of ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    """
    # Find non-zero coordinates
    coords = np.where(mask > 0)
    
    if len(coords[0]) == 0:
        # Return full volume if no tumor found
        return (
            (0, mask.shape[0]),
            (0, mask.shape[1]),
            (0, mask.shape[2]),
        )
    
    # Get bounding box with margin
    x_min = max(0, np.min(coords[0]) - margin)
    x_max = min(mask.shape[0], np.max(coords[0]) + margin + 1)
    y_min = max(0, np.min(coords[1]) - margin)
    y_max = min(mask.shape[1], np.max(coords[1]) + margin + 1)
    z_min = max(0, np.min(coords[2]) - margin)
    z_max = min(mask.shape[2], np.max(coords[2]) + margin + 1)
    
    return ((x_min, x_max), (y_min, y_max), (z_min, z_max))


def extract_roi(
    volume: np.ndarray,
    mask: np.ndarray,
    margin: int = 10,
    min_size: Optional[Tuple[int, int, int]] = None,
    target_size: Optional[Tuple[int, int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Extract ROI from volume using segmentation mask.
    
    Args:
        volume: Multi-modal volume (C, H, W, D)
        mask: Segmentation mask (H, W, D)
        margin: Margin around bounding box
        min_size: Minimum output size (pad if smaller)
        target_size: Optional target size to resize to
        
    Returns:
        Tuple of (cropped_volume, cropped_mask, metadata)
    """
    # Get bounding box
    bbox = get_bounding_box(mask, margin=margin)
    
    # Crop volume and mask
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = bbox
    
    cropped_volume = volume[:, x_min:x_max, y_min:y_max, z_min:z_max]
    cropped_mask = mask[x_min:x_max, y_min:y_max, z_min:z_max]
    
    # Store metadata
    metadata = {
        "bbox": bbox,
        "original_shape": volume.shape,
        "cropped_shape": cropped_volume.shape,
    }
    
    # Pad to minimum size if needed
    if min_size is not None:
        cropped_volume, cropped_mask = pad_to_min_size(
            cropped_volume, cropped_mask, min_size
        )
        metadata["padded_shape"] = cropped_volume.shape
    
    # Resize if target size specified
    if target_size is not None:
        cropped_volume = resize_volume(cropped_volume, target_size)
        cropped_mask = resize_mask(cropped_mask, target_size)
        metadata["target_shape"] = cropped_volume.shape
    
    return cropped_volume, cropped_mask, metadata


def pad_to_min_size(
    volume: np.ndarray,
    mask: np.ndarray,
    min_size: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pad volume and mask to minimum size.
    
    Args:
        volume: Volume to pad (C, H, W, D)
        mask: Mask to pad (H, W, D)
        min_size: Minimum size (H, W, D)
        
    Returns:
        Padded volume and mask
    """
    current_size = volume.shape[1:]  # (H, W, D)
    
    # Calculate padding needed
    pad_h = max(0, min_size[0] - current_size[0])
    pad_w = max(0, min_size[1] - current_size[1])
    pad_d = max(0, min_size[2] - current_size[2])
    
    if pad_h == 0 and pad_w == 0 and pad_d == 0:
        return volume, mask
    
    # Pad symmetrically
    pad_h1, pad_h2 = pad_h // 2, pad_h - pad_h // 2
    pad_w1, pad_w2 = pad_w // 2, pad_w - pad_w // 2
    pad_d1, pad_d2 = pad_d // 2, pad_d - pad_d // 2
    
    # Pad volume (channels, H, W, D)
    volume_padding = ((0, 0), (pad_h1, pad_h2), (pad_w1, pad_w2), (pad_d1, pad_d2))
    padded_volume = np.pad(volume, volume_padding, mode="constant", constant_values=0)
    
    # Pad mask (H, W, D)
    mask_padding = ((pad_h1, pad_h2), (pad_w1, pad_w2), (pad_d1, pad_d2))
    padded_mask = np.pad(mask, mask_padding, mode="constant", constant_values=0)
    
    return padded_volume, padded_mask


def resize_volume(
    volume: np.ndarray,
    target_size: Tuple[int, int, int],
    order: int = 1,
) -> np.ndarray:
    """
    Resize volume to target size using interpolation.
    
    Args:
        volume: Input volume (C, H, W, D)
        target_size: Target size (H, W, D)
        order: Interpolation order (1=linear, 0=nearest)
        
    Returns:
        Resized volume
    """
    current_size = volume.shape[1:]
    
    if current_size == target_size:
        return volume
    
    # Calculate zoom factors
    zoom_factors = [1.0]  # Don't zoom channels
    zoom_factors.extend([t / c for t, c in zip(target_size, current_size)])
    
    resized = ndimage.zoom(volume, zoom_factors, order=order)
    return resized


def resize_mask(
    mask: np.ndarray,
    target_size: Tuple[int, int, int],
) -> np.ndarray:
    """
    Resize mask to target size using nearest neighbor.
    
    Args:
        mask: Input mask (H, W, D)
        target_size: Target size (H, W, D)
        
    Returns:
        Resized mask
    """
    current_size = mask.shape
    
    if current_size == target_size:
        return mask
    
    zoom_factors = [t / c for t, c in zip(target_size, current_size)]
    resized = ndimage.zoom(mask, zoom_factors, order=0)  # Nearest neighbor
    return resized.astype(mask.dtype)


def extract_roi_for_patient(
    patient_id: str,
    processed_dir: Path,
    masks_dir: Path,
    output_dir: Path,
    margin: int = 10,
    min_size: Optional[Tuple[int, int, int]] = None,
    target_size: Optional[Tuple[int, int, int]] = None,
) -> Optional[Path]:
    """
    Extract ROI for a single patient.
    
    Args:
        patient_id: Patient identifier
        processed_dir: Directory with preprocessed volumes
        masks_dir: Directory with segmentation masks
        output_dir: Output directory for ROIs
        margin: Margin around bounding box
        min_size: Minimum output size
        target_size: Optional target size
        
    Returns:
        Path to saved ROI or None if failed
    """
    # Find volume
    volume_path = processed_dir / patient_id / f"{patient_id}.npy"
    if not volume_path.exists():
        volume_path = processed_dir / f"{patient_id}.npy"
    
    if not volume_path.exists():
        print(f"Warning: Volume not found for {patient_id}")
        return None
    
    # Find mask
    mask_path = masks_dir / patient_id / "seg.nii.gz"
    if not mask_path.exists():
        mask_path = masks_dir / f"{patient_id}_seg.nii.gz"
    
    if not mask_path.exists():
        print(f"Warning: Mask not found for {patient_id}")
        return None
    
    # Load data
    volume = np.load(volume_path).astype(np.float32)
    mask = nib.load(str(mask_path)).get_fdata().astype(np.uint8)
    
    # Extract ROI
    roi_volume, roi_mask, metadata = extract_roi(
        volume=volume,
        mask=mask,
        margin=margin,
        min_size=min_size,
        target_size=target_size,
    )
    
    # Save ROI
    patient_output = output_dir / patient_id
    patient_output.mkdir(parents=True, exist_ok=True)
    
    roi_path = patient_output / f"{patient_id}_roi.npy"
    mask_roi_path = patient_output / f"{patient_id}_roi_mask.npy"
    metadata_path = patient_output / "metadata.npy"
    
    np.save(roi_path, roi_volume)
    np.save(mask_roi_path, roi_mask)
    np.save(metadata_path, metadata)
    
    return roi_path


def run_roi_extraction(
    patient_ids: List[str],
    processed_dir: Union[str, Path],
    masks_dir: Union[str, Path],
    output_dir: Union[str, Path],
    margin: int = 10,
    min_size: Optional[Tuple[int, int, int]] = None,
    target_size: Optional[Tuple[int, int, int]] = None,
    num_workers: int = 1,
) -> None:
    """
    Run ROI extraction for multiple patients.
    
    Args:
        patient_ids: List of patient identifiers
        processed_dir: Directory with preprocessed volumes
        masks_dir: Directory with segmentation masks
        output_dir: Output directory for ROIs
        margin: Margin around bounding box
        min_size: Minimum output size
        target_size: Optional target size
        num_workers: Number of parallel workers
    """
    processed_dir = Path(processed_dir)
    masks_dir = Path(masks_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting ROIs for {len(patient_ids)} patients...")
    print(f"  Processed dir: {processed_dir}")
    print(f"  Masks dir: {masks_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Margin: {margin}")
    print(f"  Min size: {min_size}")
    print(f"  Target size: {target_size}")
    print()
    
    successful = 0
    failed = []
    
    for patient_id in tqdm(patient_ids, desc="ROI Extraction"):
        result = extract_roi_for_patient(
            patient_id=patient_id,
            processed_dir=processed_dir,
            masks_dir=masks_dir,
            output_dir=output_dir,
            margin=margin,
            min_size=min_size,
            target_size=target_size,
        )
        
        if result is not None:
            successful += 1
        else:
            failed.append(patient_id)
    
    print(f"\nROI extraction complete!")
    print(f"Successful: {successful}/{len(patient_ids)}")
    if failed:
        print(f"Failed: {failed}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract ROI from Segmentation Masks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract ROIs for all patients
  python -m roi_extraction.extract_roi \\
      --patient-list data/splits/train.txt \\
      --processed-dir data/processed \\
      --masks-dir data/masks \\
      --output-dir data/roi

  # Extract ROIs with specific size
  python -m roi_extraction.extract_roi \\
      --patient-list data/splits/train.txt \\
      --processed-dir data/processed \\
      --masks-dir data/masks \\
      --output-dir data/roi \\
      --target-size 96 96 96
        """,
    )
    
    parser.add_argument(
        "--patient-list",
        type=Path,
        help="Path to text file with patient IDs",
    )
    parser.add_argument(
        "--patients",
        nargs="+",
        help="List of patient IDs",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        required=True,
        help="Directory containing preprocessed volumes",
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        required=True,
        help="Directory containing segmentation masks",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for ROIs",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=10,
        help="Margin around bounding box",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        nargs=3,
        default=None,
        help="Minimum output size (H W D)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=3,
        default=None,
        help="Target size to resize to (H W D)",
    )
    
    args = parser.parse_args()
    
    # Get patient list
    if args.patient_list:
        patient_ids = load_patient_list(args.patient_list)
    elif args.patients:
        patient_ids = args.patients
    else:
        raise ValueError("Must provide either --patient-list or --patients")
    
    # Parse size arguments
    min_size = tuple(args.min_size) if args.min_size else None
    target_size = tuple(args.target_size) if args.target_size else None
    
    run_roi_extraction(
        patient_ids=patient_ids,
        processed_dir=args.processed_dir,
        masks_dir=args.masks_dir,
        output_dir=args.output_dir,
        margin=args.margin,
        min_size=min_size,
        target_size=target_size,
    )


if __name__ == "__main__":
    main()
