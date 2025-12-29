"""
I/O utilities for medical imaging file operations.
Handles NIfTI files, patient lists, checkpoints, and configurations.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import torch
import yaml


# ============================================================================
# NIfTI File Operations
# ============================================================================


def load_nifti(
    path: Union[str, Path],
    return_affine: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Load a NIfTI file (.nii or .nii.gz).
    
    Args:
        path: Path to NIfTI file
        return_affine: If True, also return the affine matrix
        
    Returns:
        Volume data as numpy array, optionally with affine matrix
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {path}")
    
    img = nib.load(str(path))
    data = img.get_fdata().astype(np.float32)
    
    if return_affine:
        return data, img.affine
    return data


def save_nifti(
    data: np.ndarray,
    path: Union[str, Path],
    affine: Optional[np.ndarray] = None,
) -> None:
    """
    Save a numpy array as a NIfTI file.
    
    Args:
        data: Volume data to save
        path: Output path
        affine: Affine transformation matrix (identity if not provided)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if affine is None:
        affine = np.eye(4)
    
    img = nib.Nifti1Image(data, affine)
    nib.save(img, str(path))


def load_multi_modal_nifti(
    patient_dir: Union[str, Path],
    modalities: List[str] = ["t1", "t2", "flair", "t1ce"],
    suffix: str = ".nii.gz",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load multi-modal MRI data for a patient.
    
    Args:
        patient_dir: Directory containing modality files
        modalities: List of modality names
        suffix: File suffix
        
    Returns:
        Stacked volume (C, H, W, D) and affine matrix
    """
    patient_dir = Path(patient_dir)
    volumes = []
    affine = None
    
    for modality in modalities:
        # Try different naming conventions
        possible_names = [
            f"{modality}{suffix}",
            f"{patient_dir.name}_{modality}{suffix}",
            f"{modality.upper()}{suffix}",
        ]
        
        found = False
        for name in possible_names:
            path = patient_dir / name
            if path.exists():
                data, aff = load_nifti(path, return_affine=True)
                volumes.append(data)
                if affine is None:
                    affine = aff
                found = True
                break
        
        if not found:
            raise FileNotFoundError(
                f"Modality '{modality}' not found in {patient_dir}"
            )
    
    # Stack along channel dimension
    stacked = np.stack(volumes, axis=0)
    return stacked, affine


# ============================================================================
# Patient List Operations
# ============================================================================


def load_patient_list(path: Union[str, Path]) -> List[str]:
    """
    Load a list of patient IDs from a text file.
    
    Args:
        path: Path to text file (one patient ID per line)
        
    Returns:
        List of patient IDs
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Patient list not found: {path}")
    
    with open(path, "r") as f:
        patients = [line.strip() for line in f if line.strip()]
    
    return patients


def save_patient_list(
    patients: List[str],
    path: Union[str, Path],
) -> None:
    """
    Save a list of patient IDs to a text file.
    
    Args:
        patients: List of patient IDs
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        for patient in patients:
            f.write(f"{patient}\n")


def validate_no_leakage(
    train_file: Union[str, Path],
    val_file: Union[str, Path],
    test_file: Union[str, Path],
) -> bool:
    """
    Validate that there is no patient overlap between splits.
    
    Args:
        train_file: Path to training patient list
        val_file: Path to validation patient list
        test_file: Path to test patient list
        
    Returns:
        True if no overlap (valid), raises AssertionError otherwise
    """
    train_patients = set(load_patient_list(train_file))
    val_patients = set(load_patient_list(val_file))
    test_patients = set(load_patient_list(test_file))
    
    train_val_overlap = train_patients & val_patients
    train_test_overlap = train_patients & test_patients
    val_test_overlap = val_patients & test_patients
    
    if train_val_overlap:
        raise AssertionError(
            f"Data leakage detected! Train-Val overlap: {train_val_overlap}"
        )
    if train_test_overlap:
        raise AssertionError(
            f"Data leakage detected! Train-Test overlap: {train_test_overlap}"
        )
    if val_test_overlap:
        raise AssertionError(
            f"Data leakage detected! Val-Test overlap: {val_test_overlap}"
        )
    
    print(f"✓ No data leakage detected.")
    print(f"  Train: {len(train_patients)} patients")
    print(f"  Val: {len(val_patients)} patients")
    print(f"  Test: {len(test_patients)} patients")
    
    return True


# ============================================================================
# Configuration Operations
# ============================================================================


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        path: Path to YAML file
        
    Returns:
        Configuration dictionary
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(
    config: Dict[str, Any],
    path: Union[str, Path],
) -> None:
    """
    Save a configuration dictionary to YAML.
    
    Args:
        config: Configuration dictionary
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# ============================================================================
# Checkpoint Operations
# ============================================================================


def save_checkpoint(
    state: Dict[str, Any],
    path: Union[str, Path],
    is_best: bool = False,
) -> None:
    """
    Save a training checkpoint.
    
    Args:
        state: Checkpoint state dict
        path: Output path
        is_best: If True, also save as 'best_model.pt'
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(state, path)
    
    if is_best:
        best_path = path.parent / "best_model.pt"
        torch.save(state, best_path)


def load_checkpoint(
    path: Union[str, Path],
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Load a training checkpoint.
    
    Args:
        path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        device: Device to load tensors to
        
    Returns:
        Checkpoint state dict
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    return checkpoint


# ============================================================================
# Label Operations
# ============================================================================


def load_labels(
    path: Union[str, Path],
    patient_col: str = "patient_id",
    label_col: str = "label",
) -> Dict[str, int]:
    """
    Load patient labels from a CSV file.
    
    Args:
        path: Path to CSV file
        patient_col: Column name for patient IDs
        label_col: Column name for labels
        
    Returns:
        Dictionary mapping patient ID to label
    """
    import pandas as pd
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")
    
    df = pd.read_csv(path)
    labels = dict(zip(df[patient_col], df[label_col]))
    
    return labels


def create_labels_file(
    patient_labels: Dict[str, int],
    path: Union[str, Path],
    class_names: Optional[List[str]] = None,
) -> None:
    """
    Create a labels CSV file.
    
    Args:
        patient_labels: Dictionary mapping patient ID to label
        path: Output path
        class_names: Optional class name mapping
    """
    import pandas as pd
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame([
        {"patient_id": pid, "label": label}
        for pid, label in patient_labels.items()
    ])
    
    if class_names:
        df["class_name"] = df["label"].map(
            {i: name for i, name in enumerate(class_names)}
        )
    
    df.to_csv(path, index=False)
