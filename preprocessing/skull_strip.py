"""
Skull Stripping for brain MRI.

This module provides a placeholder for skull stripping functionality.
For production use, integrate one of:
- SynthStrip (recommended): https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/
- HD-BET: https://github.com/MIC-DKFZ/HD-BET
- FSL BET: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET

SynthStrip is recommended for its robustness to pathologies.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False


def simple_threshold_skull_strip(
    image: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple threshold-based skull stripping (NOT recommended for production).
    
    This is a placeholder that performs basic Otsu thresholding.
    For production, use SynthStrip or HD-BET.
    
    Args:
        image: Input 3D volume
        lower_percentile: Lower intensity percentile
        upper_percentile: Upper intensity percentile
        
    Returns:
        Tuple of (stripped image, brain mask)
    """
    # Clip intensities
    p_low = np.percentile(image[image > 0], lower_percentile)
    p_high = np.percentile(image[image > 0], upper_percentile)
    clipped = np.clip(image, p_low, p_high)
    
    # Simple Otsu thresholding
    if SITK_AVAILABLE:
        sitk_image = sitk.GetImageFromArray(clipped.astype(np.float32))
        mask = sitk.OtsuThreshold(sitk_image, 0, 1, 200)
        brain_mask = sitk.GetArrayFromImage(mask).astype(np.uint8)
    else:
        # Fallback to simple threshold
        threshold = np.mean(clipped[clipped > 0])
        brain_mask = (clipped > threshold).astype(np.uint8)
    
    # Apply morphological operations to clean mask
    if SITK_AVAILABLE:
        mask_sitk = sitk.GetImageFromArray(brain_mask)
        
        # Binary closing to fill holes
        mask_sitk = sitk.BinaryMorphologicalClosing(mask_sitk, [3, 3, 3])
        
        # Keep largest connected component
        cc = sitk.ConnectedComponent(mask_sitk)
        stats = sitk.LabelIntensityStatisticsImageFilter()
        stats.Execute(cc, sitk.GetImageFromArray(clipped.astype(np.float32)))
        
        largest_label = max(
            range(1, stats.GetNumberOfLabels() + 1),
            key=lambda l: stats.GetNumberOfPixels(l),
            default=1,
        )
        
        brain_mask = sitk.GetArrayFromImage(cc) == largest_label
        brain_mask = brain_mask.astype(np.uint8)
    
    # Apply mask
    stripped = image * brain_mask
    
    return stripped, brain_mask


def skull_strip_synthstrip(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    mask_path: Optional[Union[str, Path]] = None,
    gpu: bool = True,
) -> None:
    """
    Skull stripping using SynthStrip (requires FreeSurfer installation).
    
    Args:
        input_path: Path to input NIfTI file
        output_path: Path to save stripped brain
        mask_path: Optional path to save brain mask
        gpu: Whether to use GPU acceleration
        
    Raises:
        RuntimeError: If SynthStrip is not installed
    """
    import subprocess
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = ["mri_synthstrip", "-i", str(input_path), "-o", str(output_path)]
    
    if mask_path:
        mask_path = Path(mask_path)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        cmd.extend(["-m", str(mask_path)])
    
    if gpu:
        cmd.append("--gpu")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"SynthStrip completed: {output_path}")
    except FileNotFoundError:
        raise RuntimeError(
            "SynthStrip not found. Please install FreeSurfer or use HD-BET. "
            "See: https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"SynthStrip failed: {e.stderr}")


def skull_strip_hdbet(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    mode: str = "fast",
    device: str = "cuda",
) -> Tuple[Path, Path]:
    """
    Skull stripping using HD-BET (requires HD-BET installation).
    
    Args:
        input_path: Path to input NIfTI file
        output_dir: Output directory
        mode: "fast" or "accurate"
        device: "cuda" or "cpu"
        
    Returns:
        Tuple of (stripped brain path, mask path)
        
    Raises:
        RuntimeError: If HD-BET is not installed
    """
    import subprocess
    
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{input_path.stem.replace('.nii', '')}_bet"
    
    cmd = [
        "hd-bet",
        "-i", str(input_path),
        "-o", str(output_path),
        "-device", device,
        "-mode", mode,
        "-tta", "0",  # Disable test-time augmentation for speed
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        brain_path = Path(str(output_path) + ".nii.gz")
        mask_path = Path(str(output_path) + "_mask.nii.gz")
        print(f"HD-BET completed: {brain_path}")
        return brain_path, mask_path
    except FileNotFoundError:
        raise RuntimeError(
            "HD-BET not found. Install via: pip install hd-bet "
            "See: https://github.com/MIC-DKFZ/HD-BET"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"HD-BET failed: {e.stderr}")


def skull_strip(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    mask_path: Optional[Union[str, Path]] = None,
    method: str = "threshold",
    **kwargs,
) -> None:
    """
    Unified skull stripping interface.
    
    Args:
        input_path: Path to input NIfTI file
        output_path: Path to save stripped brain
        mask_path: Optional path to save brain mask
        method: "synthstrip", "hdbet", or "threshold"
        **kwargs: Additional arguments for the chosen method
    """
    import nibabel as nib
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if method == "synthstrip":
        skull_strip_synthstrip(input_path, output_path, mask_path, **kwargs)
    
    elif method == "hdbet":
        output_dir = output_path.parent
        brain_path, hdbet_mask_path = skull_strip_hdbet(input_path, output_dir, **kwargs)
        
        # Rename to desired output names
        import shutil
        shutil.move(brain_path, output_path)
        if mask_path:
            shutil.move(hdbet_mask_path, mask_path)
    
    elif method == "threshold":
        # Fallback simple method
        img = nib.load(str(input_path))
        data = img.get_fdata().astype(np.float32)
        
        stripped, brain_mask = simple_threshold_skull_strip(data)
        
        # Save stripped brain
        stripped_img = nib.Nifti1Image(stripped, img.affine, img.header)
        nib.save(stripped_img, str(output_path))
        
        # Save mask if requested
        if mask_path:
            mask_path = Path(mask_path)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            mask_img = nib.Nifti1Image(brain_mask.astype(np.uint8), img.affine)
            nib.save(mask_img, str(mask_path))
        
        print(f"Threshold skull stripping completed: {output_path}")
        print("WARNING: Threshold method is not recommended for production. Use SynthStrip or HD-BET.")
    
    else:
        raise ValueError(f"Unknown skull stripping method: {method}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Skull Stripping")
    parser.add_argument("--input", "-i", required=True, help="Input NIfTI file")
    parser.add_argument("--output", "-o", required=True, help="Output stripped brain path")
    parser.add_argument("--mask", "-m", help="Output brain mask path")
    parser.add_argument(
        "--method",
        choices=["synthstrip", "hdbet", "threshold"],
        default="threshold",
        help="Skull stripping method",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU (for SynthStrip/HD-BET)")
    
    args = parser.parse_args()
    
    skull_strip(
        args.input,
        args.output,
        mask_path=args.mask,
        method=args.method,
        gpu=args.gpu,
    )
