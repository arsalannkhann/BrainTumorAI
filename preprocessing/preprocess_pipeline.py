"""
End-to-end preprocessing pipeline for brain MRI.
Orchestrates N4 bias correction, skull stripping (optional), and normalization.
"""

import argparse
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from .n4_bias import correct_patient as n4_correct_patient
from .normalize import normalize_patient, save_as_numpy
from .skull_strip import skull_strip


def preprocess_patient(
    patient_id: str,
    raw_dir: Path,
    output_dir: Path,
    modalities: List[str],
    do_n4: bool = True,
    do_skull_strip: bool = False,
    skull_strip_method: str = "threshold",
    do_normalize: bool = True,
    normalize_method: str = "zscore",
    save_numpy: bool = True,
    suffix: str = ".nii.gz",
) -> bool:
    """
    Preprocess a single patient's MRI data.
    
    Args:
        patient_id: Patient identifier
        raw_dir: Base directory containing raw patient data
        output_dir: Base output directory for processed data
        modalities: List of modality names
        do_n4: Whether to apply N4 bias correction
        do_skull_strip: Whether to apply skull stripping
        skull_strip_method: Method for skull stripping
        do_normalize: Whether to apply intensity normalization
        normalize_method: Method for normalization
        save_numpy: Whether to save as numpy array
        suffix: File suffix for NIfTI files
        
    Returns:
        True if successful, False otherwise
    """
    try:
        patient_raw = raw_dir / patient_id
        if not patient_raw.exists():
            print(f"Warning: Patient directory not found: {patient_raw}")
            return False
        
        # Create working directories
        patient_output = output_dir / patient_id
        patient_output.mkdir(parents=True, exist_ok=True)
        
        # Stage 1: N4 Bias Correction
        if do_n4:
            n4_output = patient_output / "n4_corrected"
            n4_output.mkdir(exist_ok=True)
            n4_correct_patient(
                patient_raw,
                n4_output,
                modalities=modalities,
                suffix=suffix,
            )
            current_dir = n4_output
        else:
            current_dir = patient_raw
        
        # Stage 2: Skull Stripping (optional)
        if do_skull_strip:
            stripped_dir = patient_output / "skull_stripped"
            stripped_dir.mkdir(exist_ok=True)
            
            # Apply skull stripping to T1 (reference modality)
            t1_path = current_dir / f"t1{suffix}"
            if t1_path.exists():
                skull_strip(
                    t1_path,
                    stripped_dir / f"t1{suffix}",
                    mask_path=stripped_dir / f"brain_mask{suffix}",
                    method=skull_strip_method,
                )
            
            # Copy other modalities (they share the same brain mask)
            import shutil
            for modality in modalities:
                if modality != "t1":
                    src = current_dir / f"{modality}{suffix}"
                    dst = stripped_dir / f"{modality}{suffix}"
                    if src.exists():
                        shutil.copy(src, dst)
            
            current_dir = stripped_dir
        
        # Stage 3: Normalization
        if do_normalize:
            normalized_dir = patient_output / "normalized"
            normalized_dir.mkdir(exist_ok=True)
            normalize_patient(
                current_dir,
                normalized_dir,
                modalities=modalities,
                suffix=suffix,
                method=normalize_method,
            )
            current_dir = normalized_dir
        
        # Stage 4: Save as numpy array
        if save_numpy:
            numpy_path = patient_output / f"{patient_id}.npy"
            save_as_numpy(
                current_dir,
                numpy_path,
                modalities=modalities,
                suffix=suffix,
            )
        
        return True
    
    except Exception as e:
        print(f"Error processing {patient_id}: {e}")
        return False


def run_preprocessing_pipeline(
    raw_dir: Path,
    output_dir: Path,
    patient_list: Optional[List[str]] = None,
    modalities: List[str] = ["t1", "t2", "flair", "t1ce"],
    do_n4: bool = True,
    do_skull_strip: bool = False,
    skull_strip_method: str = "threshold",
    do_normalize: bool = True,
    normalize_method: str = "zscore",
    save_numpy: bool = True,
    num_workers: int = 1,
    suffix: str = ".nii.gz",
) -> None:
    """
    Run the full preprocessing pipeline on all patients.
    
    Args:
        raw_dir: Directory containing raw patient data
        output_dir: Output directory for processed data
        patient_list: List of patient IDs (if None, discover from raw_dir)
        modalities: List of modality names
        do_n4: Whether to apply N4 bias correction
        do_skull_strip: Whether to apply skull stripping
        skull_strip_method: Method for skull stripping
        do_normalize: Whether to apply normalization
        normalize_method: Method for normalization
        save_numpy: Whether to save as numpy
        num_workers: Number of parallel workers
        suffix: File suffix for NIfTI files
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Discover patients if not provided
    if patient_list is None:
        patient_list = [d.name for d in raw_dir.iterdir() if d.is_dir()]
    
    print(f"Processing {len(patient_list)} patients...")
    print(f"  Raw directory: {raw_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Modalities: {modalities}")
    print(f"  N4 correction: {do_n4}")
    print(f"  Skull stripping: {do_skull_strip} ({skull_strip_method})")
    print(f"  Normalization: {do_normalize} ({normalize_method})")
    print(f"  Save numpy: {save_numpy}")
    print()
    
    # Create worker function
    worker_fn = partial(
        preprocess_patient,
        raw_dir=raw_dir,
        output_dir=output_dir,
        modalities=modalities,
        do_n4=do_n4,
        do_skull_strip=do_skull_strip,
        skull_strip_method=skull_strip_method,
        do_normalize=do_normalize,
        normalize_method=normalize_method,
        save_numpy=save_numpy,
        suffix=suffix,
    )
    
    # Process patients
    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(worker_fn, patient_list),
                total=len(patient_list),
                desc="Preprocessing",
            ))
    else:
        results = []
        for patient_id in tqdm(patient_list, desc="Preprocessing"):
            results.append(worker_fn(patient_id))
    
    # Summary
    success_count = sum(results)
    print(f"\nPreprocessing complete: {success_count}/{len(patient_list)} patients successful")


def main():
    parser = argparse.ArgumentParser(
        description="Brain MRI Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic preprocessing (N4 + normalization)
  python -m preprocessing.preprocess_pipeline --raw-dir data/raw --output-dir data/processed
  
  # Full pipeline with skull stripping
  python -m preprocessing.preprocess_pipeline --raw-dir data/raw --output-dir data/processed --skull-strip --skull-strip-method synthstrip
  
  # Parallel processing
  python -m preprocessing.preprocess_pipeline --raw-dir data/raw --output-dir data/processed --num-workers 4
        """,
    )
    
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Directory containing raw patient data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--patient-list",
        type=Path,
        help="Optional text file with patient IDs (one per line)",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["t1", "t2", "flair", "t1ce"],
        help="List of modality names",
    )
    parser.add_argument(
        "--no-n4",
        action="store_true",
        help="Skip N4 bias field correction",
    )
    parser.add_argument(
        "--skull-strip",
        action="store_true",
        help="Apply skull stripping",
    )
    parser.add_argument(
        "--skull-strip-method",
        choices=["synthstrip", "hdbet", "threshold"],
        default="threshold",
        help="Skull stripping method",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip intensity normalization",
    )
    parser.add_argument(
        "--normalize-method",
        choices=["zscore", "percentile"],
        default="zscore",
        help="Normalization method",
    )
    parser.add_argument(
        "--no-numpy",
        action="store_true",
        help="Skip saving as numpy array",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--suffix",
        default=".nii.gz",
        help="File suffix for NIfTI files",
    )
    
    args = parser.parse_args()
    
    # Load patient list if provided
    patient_list = None
    if args.patient_list:
        with open(args.patient_list, "r") as f:
            patient_list = [line.strip() for line in f if line.strip()]
    
    run_preprocessing_pipeline(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        patient_list=patient_list,
        modalities=args.modalities,
        do_n4=not args.no_n4,
        do_skull_strip=args.skull_strip,
        skull_strip_method=args.skull_strip_method,
        do_normalize=not args.no_normalize,
        normalize_method=args.normalize_method,
        save_numpy=not args.no_numpy,
        num_workers=args.num_workers,
        suffix=args.suffix,
    )


if __name__ == "__main__":
    main()
