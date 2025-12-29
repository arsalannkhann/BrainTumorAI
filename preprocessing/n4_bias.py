"""
N4 Bias Field Correction for MRI volumes.
Uses SimpleITK implementation of the N4ITK algorithm.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk


def n4_bias_field_correction(
    image: Union[np.ndarray, sitk.Image],
    mask: Optional[Union[np.ndarray, sitk.Image]] = None,
    shrink_factor: int = 4,
    num_iterations: Tuple[int, ...] = (50, 50, 50, 50),
    convergence_threshold: float = 0.001,
    num_fitting_levels: int = 4,
    num_control_points: Tuple[int, ...] = (4, 4, 4),
    spline_order: int = 3,
) -> np.ndarray:
    """
    Apply N4 bias field correction to a 3D MRI volume.
    
    The N4ITK algorithm corrects intensity non-uniformity (bias field)
    in MRI images caused by magnetic field inhomogeneities.
    
    Args:
        image: Input 3D volume (numpy array or SimpleITK image)
        mask: Optional brain mask to focus correction (non-zero = brain)
        shrink_factor: Factor to downsample image for faster processing
        num_iterations: Iterations at each resolution level
        convergence_threshold: Convergence threshold for optimization
        num_fitting_levels: Number of multi-resolution fitting levels
        num_control_points: Number of B-spline control points
        spline_order: B-spline order for bias field modeling
        
    Returns:
        Bias-corrected image as numpy array
    """
    # Convert to SimpleITK if needed
    if isinstance(image, np.ndarray):
        sitk_image = sitk.GetImageFromArray(image.astype(np.float32))
    else:
        sitk_image = image
    
    # Handle mask
    if mask is not None:
        if isinstance(mask, np.ndarray):
            sitk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
        else:
            sitk_mask = mask
        sitk_mask.CopyInformation(sitk_image)
    else:
        # Create mask from non-zero voxels
        sitk_mask = sitk.OtsuThreshold(sitk_image, 0, 1, 200)
    
    # Cast to float32
    sitk_image = sitk.Cast(sitk_image, sitk.sitkFloat32)
    
    # Shrink image for faster processing
    if shrink_factor > 1:
        shrunk_image = sitk.Shrink(
            sitk_image,
            [shrink_factor] * sitk_image.GetDimension()
        )
        shrunk_mask = sitk.Shrink(
            sitk_mask,
            [shrink_factor] * sitk_mask.GetDimension()
        )
    else:
        shrunk_image = sitk_image
        shrunk_mask = sitk_mask
    
    # Configure N4 corrector
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(list(num_iterations))
    corrector.SetConvergenceThreshold(convergence_threshold)
    # corrector.SetNumberOfFittingLevels(num_fitting_levels)  # Implicitly set by iterations
    corrector.SetSplineOrder(spline_order)
    corrector.SetNumberOfControlPoints(list(num_control_points))
    
    # Run correction on shrunk image
    _ = corrector.Execute(shrunk_image, shrunk_mask)
    
    # Get the bias field and resample to original size
    log_bias_field = corrector.GetLogBiasFieldAsImage(sitk_image)
    
    # Apply correction to original resolution image
    corrected = sitk_image / sitk.Exp(log_bias_field)
    
    return sitk.GetArrayFromImage(corrected).astype(np.float32)


def correct_modality(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    mask_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> None:
    """
    Apply N4 correction to a single NIfTI file.
    
    Args:
        input_path: Path to input NIfTI file
        output_path: Path to save corrected image
        mask_path: Optional path to brain mask
        **kwargs: Additional arguments for n4_bias_field_correction
    """
    import nibabel as nib
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Load image
    img = nib.load(str(input_path))
    data = img.get_fdata().astype(np.float32)
    
    # Load mask if provided
    mask = None
    if mask_path is not None:
        mask_path = Path(mask_path)
        if mask_path.exists():
            mask = nib.load(str(mask_path)).get_fdata()
    
    # Apply correction
    corrected = n4_bias_field_correction(data, mask=mask, **kwargs)
    
    # Save result
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corrected_img = nib.Nifti1Image(corrected, img.affine, img.header)
    nib.save(corrected_img, str(output_path))


def correct_patient(
    patient_dir: Union[str, Path],
    output_dir: Union[str, Path],
    modalities: list[str] = ["t1", "t2", "flair", "t1ce"],
    suffix: str = ".nii.gz",
    **kwargs,
) -> None:
    """
    Apply N4 correction to all modalities for a single patient.
    
    Args:
        patient_dir: Directory containing patient's MRI modalities
        output_dir: Output directory for corrected images
        modalities: List of modality names to process
        suffix: File suffix
        **kwargs: Additional arguments for n4_bias_field_correction
    """
    patient_dir = Path(patient_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
        
        output_path = output_dir / f"{modality}{suffix}"
        print(f"  Processing {modality}...")
        correct_modality(input_path, output_path, **kwargs)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="N4 Bias Field Correction")
    parser.add_argument("--input", "-i", required=True, help="Input NIfTI file or patient directory")
    parser.add_argument("--output", "-o", required=True, help="Output path")
    parser.add_argument("--mask", "-m", help="Optional brain mask")
    parser.add_argument("--shrink-factor", type=int, default=4, help="Shrink factor for speed")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        correct_modality(
            input_path,
            args.output,
            mask_path=args.mask,
            shrink_factor=args.shrink_factor,
        )
    else:
        correct_patient(
            input_path,
            args.output,
            shrink_factor=args.shrink_factor,
        )
    
    print("Done!")
