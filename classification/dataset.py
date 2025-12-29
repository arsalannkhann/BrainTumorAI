"""
Dataset for brain tumor classification from ROI volumes.
Supports both 3D and 2.5D (slice-based) input modes.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from utils import load_labels, load_patient_list


class BrainTumorClassificationDataset(Dataset):
    """
    Dataset for brain tumor classification from extracted ROIs.
    
    Supports two modes:
    - 3D: Use full volumetric ROI
    - 2.5D: Sample central slices across z-axis
    """
    
    def __init__(
        self,
        patient_ids: List[str],
        roi_dir: Union[str, Path],
        labels: Dict[str, int],
        mode: str = "2.5d",
        num_slices: int = 16,
        slice_sampling: str = "uniform",
        input_size: Tuple[int, int] = (224, 224),
        is_train: bool = True,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize the classification dataset.
        
        Args:
            patient_ids: List of patient identifiers
            roi_dir: Directory containing ROI files
            labels: Dict mapping patient_id to class label
            mode: "3d" or "2.5d"
            num_slices: Number of slices to sample (for 2.5D mode)
            slice_sampling: "uniform", "random", or "center"
            input_size: Input size for 2D slices
            is_train: Whether this is training data
            transform: Optional transform function
        """
        self.patient_ids = patient_ids
        self.roi_dir = Path(roi_dir)
        self.labels = labels
        self.mode = mode.lower()
        self.num_slices = num_slices
        self.slice_sampling = slice_sampling
        self.input_size = input_size
        self.is_train = is_train
        
        # Build transform if not provided
        if transform is None:
            self.transform = self._build_transform()
        else:
            self.transform = transform
        
        # Filter out patients without ROI files
        self.valid_patients = self._validate_patients()
        
        print(f"Dataset initialized: {len(self.valid_patients)}/{len(patient_ids)} patients valid")
    
    def _validate_patients(self) -> List[str]:
        """Validate that patients have required files."""
        valid = []
        for patient_id in self.patient_ids:
            roi_path = self.roi_dir / patient_id / f"{patient_id}_roi.npy"
            if not roi_path.exists():
                roi_path = self.roi_dir / f"{patient_id}_roi.npy"
            
            if roi_path.exists() and patient_id in self.labels:
                valid.append(patient_id)
        return valid
    
    def _build_transform(self) -> A.Compose:
        """Build albumentations transform pipeline."""
        if self.is_train:
            return A.Compose([
                A.Resize(self.input_size[0], self.input_size[1]),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5,
                ),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.GaussNoise(var_limit=(10, 50), p=0.2),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(self.input_size[0], self.input_size[1]),
                ToTensorV2(),
            ])
    
    def _sample_slices(self, volume: np.ndarray) -> np.ndarray:
        """
        Sample slices from a 3D volume.
        
        Args:
            volume: Volume (C, H, W, D)
            
        Returns:
            Sampled slices (num_slices, C, H, W)
        """
        depth = volume.shape[3]
        
        if self.slice_sampling == "uniform":
            # Uniformly spaced slices
            indices = np.linspace(0, depth - 1, self.num_slices, dtype=int)
        elif self.slice_sampling == "random":
            # Random slices
            indices = np.random.choice(depth, min(self.num_slices, depth), replace=False)
            indices = np.sort(indices)
        elif self.slice_sampling == "center":
            # Center slices
            center = depth // 2
            half = self.num_slices // 2
            start = max(0, center - half)
            end = min(depth, center + half + self.num_slices % 2)
            indices = np.arange(start, end)
            # Pad if needed
            if len(indices) < self.num_slices:
                pad = self.num_slices - len(indices)
                indices = np.pad(indices, (pad // 2, pad - pad // 2), mode="edge")
        else:
            raise ValueError(f"Unknown slice sampling: {self.slice_sampling}")
        
        # Extract slices (C, H, W, D) -> (num_slices, C, H, W)
        slices = volume[:, :, :, indices].transpose(3, 0, 1, 2)
        
        return slices
    
    def __len__(self) -> int:
        return len(self.valid_patients)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        patient_id = self.valid_patients[idx]
        
        # Load ROI
        roi_path = self.roi_dir / patient_id / f"{patient_id}_roi.npy"
        if not roi_path.exists():
            roi_path = self.roi_dir / f"{patient_id}_roi.npy"
        
        volume = np.load(roi_path).astype(np.float32)  # (C, H, W, D)
        label = self.labels[patient_id]
        
        if self.mode == "3d":
            # Return full 3D volume
            # Resize to fixed size if needed
            from scipy import ndimage
            
            target_size = (self.input_size[0], self.input_size[1], self.num_slices)
            zoom_factors = [1.0]  # Keep channels
            zoom_factors.extend([t / c for t, c in zip(target_size, volume.shape[1:])])
            volume = ndimage.zoom(volume, zoom_factors, order=1)
            
            return {
                "image": torch.from_numpy(volume),
                "label": label,
                "patient_id": patient_id,
            }
        
        else:  # 2.5D mode
            # Sample slices
            slices = self._sample_slices(volume)  # (num_slices, C, H, W)
            
            # Apply transforms to each slice
            transformed_slices = []
            for i in range(slices.shape[0]):
                # Convert to HWC for albumentations
                slice_hwc = slices[i].transpose(1, 2, 0)  # (H, W, C)
                
                # Normalize to 0-255 range for albumentations
                slice_min = slice_hwc.min()
                slice_max = slice_hwc.max()
                if slice_max - slice_min > 1e-6:
                    slice_hwc = (slice_hwc - slice_min) / (slice_max - slice_min) * 255
                slice_hwc = slice_hwc.astype(np.uint8)
                
                # Apply transform
                if self.transform:
                    transformed = self.transform(image=slice_hwc)
                    slice_tensor = transformed["image"].float() / 255.0
                else:
                    slice_tensor = torch.from_numpy(slice_hwc.transpose(2, 0, 1)).float() / 255.0
                
                transformed_slices.append(slice_tensor)
            
            # Stack slices: (num_slices, C, H, W)
            stacked = torch.stack(transformed_slices, dim=0)
            
            return {
                "image": stacked,
                "label": label,
                "patient_id": patient_id,
            }


def create_classification_dataloaders(
    train_patients: List[str],
    val_patients: List[str],
    roi_dir: Union[str, Path],
    labels_file: Union[str, Path],
    batch_size: int = 8,
    num_workers: int = 4,
    mode: str = "2.5d",
    num_slices: int = 16,
    input_size: Tuple[int, int] = (224, 224),
    use_weighted_sampler: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders for classification.
    
    Args:
        train_patients: Training patient IDs
        val_patients: Validation patient IDs
        roi_dir: Directory with ROI files
        labels_file: Path to labels CSV file
        batch_size: Batch size
        num_workers: Number of data loading workers
        mode: "3d" or "2.5d"
        num_slices: Number of slices for 2.5D mode
        input_size: Input image size
        use_weighted_sampler: Use weighted sampling for class imbalance
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Load labels
    labels = load_labels(labels_file)
    
    # Create datasets
    train_dataset = BrainTumorClassificationDataset(
        patient_ids=train_patients,
        roi_dir=roi_dir,
        labels=labels,
        mode=mode,
        num_slices=num_slices,
        input_size=input_size,
        is_train=True,
    )
    
    val_dataset = BrainTumorClassificationDataset(
        patient_ids=val_patients,
        roi_dir=roi_dir,
        labels=labels,
        mode=mode,
        num_slices=num_slices,
        input_size=input_size,
        is_train=False,
    )
    
    # Weighted sampler for class imbalance
    sampler = None
    shuffle = True
    
    if use_weighted_sampler:
        train_labels = [labels[p] for p in train_dataset.valid_patients]
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[label] for label in train_labels]
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


def get_class_weights(
    labels_file: Union[str, Path],
    patient_ids: List[str],
) -> torch.Tensor:
    """
    Compute class weights for imbalanced dataset.
    
    Args:
        labels_file: Path to labels CSV
        patient_ids: List of patient IDs to consider
        
    Returns:
        Tensor of class weights
    """
    labels = load_labels(labels_file)
    
    patient_labels = [labels[p] for p in patient_ids if p in labels]
    class_counts = np.bincount(patient_labels)
    
    # Inverse frequency weighting
    total = sum(class_counts)
    weights = total / (len(class_counts) * class_counts)
    
    return torch.FloatTensor(weights)
