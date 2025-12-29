"""
Dataset for 3D brain tumor segmentation.
Uses MONAI transforms for medical imaging-specific augmentations.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from monai.data import CacheDataset, Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    ToTensord,
    MapLabelValued,
)
from torch.utils.data import DataLoader


class BrainTumorSegDataset:
    """
    Dataset for brain tumor segmentation from multi-modal MRI.
    
    Expected directory structure:
    processed_dir/
    ├── patient_001/
    │   ├── patient_001.npy  # (4, H, W, D) normalized multi-modal volume
    │   └── ...
    └── ...
    
    masks_dir/
    ├── patient_001/
    │   └── seg.nii.gz  # Segmentation mask
    └── ...
    """
    
    def __init__(
        self,
        patient_ids: List[str],
        processed_dir: Union[str, Path],
        masks_dir: Union[str, Path],
        roi_size: Tuple[int, int, int] = (128, 128, 128),
        is_train: bool = True,
        cache_rate: float = 0.0,
        num_samples: int = 4,
    ):
        """
        Initialize the segmentation dataset.
        
        Args:
            patient_ids: List of patient identifiers
            processed_dir: Directory containing preprocessed volumes
            masks_dir: Directory containing segmentation masks
            roi_size: Size of random crops for training
            is_train: Whether this is for training (enables augmentation)
            cache_rate: Fraction of data to cache (0-1)
            num_samples: Number of random crops per volume for training
        """
        self.patient_ids = patient_ids
        self.processed_dir = Path(processed_dir)
        self.masks_dir = Path(masks_dir)
        self.roi_size = roi_size
        self.is_train = is_train
        self.cache_rate = cache_rate
        self.num_samples = num_samples
        
        # Build data list
        self.data_list = self._build_data_list()
        
        # Build transforms
        self.transforms = self._build_transforms()
        
        # Create MONAI dataset
        if cache_rate > 0:
            self.dataset = CacheDataset(
                data=self.data_list,
                transform=self.transforms,
                cache_rate=cache_rate,
            )
        else:
            self.dataset = Dataset(
                data=self.data_list,
                transform=self.transforms,
            )
    
    def _build_data_list(self) -> List[Dict[str, str]]:
        """Build list of data dictionaries for MONAI."""
        data_list = []
        
        for patient_id in self.patient_ids:
            # Find volume file
            volume_path = self.processed_dir / patient_id / f"{patient_id}.npy"
            if not volume_path.exists():
                # Try alternative path
                volume_path = self.processed_dir / f"{patient_id}.npy"
            
            # Find mask file
            mask_path = self.masks_dir / patient_id / "seg.nii.gz"
            if not mask_path.exists():
                mask_path = self.masks_dir / f"{patient_id}_seg.nii.gz"
            
            if volume_path.exists() and mask_path.exists():
                data_list.append({
                    "image": str(volume_path),
                    "label": str(mask_path),
                    "patient_id": patient_id,
                })
            else:
                print(f"Warning: Missing data for {patient_id}")
        
        return data_list
    
    def _build_transforms(self) -> Compose:
        """Build MONAI transform pipeline."""
        
        # Custom loader for numpy files
        class LoadNpyd:
            def __init__(self, keys: List[str]):
                self.keys = keys
            
            def __call__(self, data: Dict) -> Dict:
                d = dict(data)
                for key in self.keys:
                    if key in d and str(d[key]).endswith(".npy"):
                        d[key] = np.load(d[key]).astype(np.float32)
                return d
        
        if self.is_train:
            transforms = Compose([
                # Load data
                LoadNpyd(keys=["image"]),
                LoadImaged(keys=["label"]),
                
                # Ensure proper shapes
                EnsureChannelFirstd(keys=["label"]),
                
                # Map label 4 to 3
                MapLabelValued(keys=["label"], orig_labels=[4], target_labels=[3]),
                
                # Random crop with positive/negative balance
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=self.roi_size,
                    pos=1,
                    neg=1,
                    num_samples=self.num_samples,
                    image_key="image",
                ),
                
                # Spatial augmentations
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                
                RandAffined(
                    keys=["image", "label"],
                    prob=0.3,
                    rotate_range=[0.3, 0.3, 0.3],
                    scale_range=[0.1, 0.1, 0.1],
                    mode=["bilinear", "nearest"],
                    padding_mode="zeros",
                ),
                
                # Intensity augmentations (image only)
                RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
                RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.5, 1.0)),
                RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
                RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
                
                # Convert to tensor
                ToTensord(keys=["image", "label"]),
            ])
        else:
            transforms = Compose([
                # Load data
                LoadNpyd(keys=["image"]),
                LoadImaged(keys=["label"]),
                
                # Ensure proper shapes
                EnsureChannelFirstd(keys=["label"]),
                
                # Map label 4 to 3
                MapLabelValued(keys=["label"], orig_labels=[4], target_labels=[3]),
                
                # Convert to tensor
                ToTensord(keys=["image", "label"]),
            ])
        
        return transforms
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.dataset[idx]


def create_dataloaders(
    train_patients: List[str],
    val_patients: List[str],
    processed_dir: Union[str, Path],
    masks_dir: Union[str, Path],
    batch_size: int = 2,
    num_workers: int = 4,
    roi_size: Tuple[int, int, int] = (128, 128, 128),
    cache_rate: float = 0.0,
    num_samples: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        train_patients: List of training patient IDs
        val_patients: List of validation patient IDs
        processed_dir: Directory with preprocessed volumes
        masks_dir: Directory with segmentation masks
        batch_size: Batch size
        num_workers: Number of data loading workers
        roi_size: Size of random crops
        cache_rate: Fraction of data to cache
        num_samples: Samples per volume for training
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = BrainTumorSegDataset(
        patient_ids=train_patients,
        processed_dir=processed_dir,
        masks_dir=masks_dir,
        roi_size=roi_size,
        is_train=True,
        cache_rate=cache_rate,
        num_samples=num_samples,
    )
    
    val_dataset = BrainTumorSegDataset(
        patient_ids=val_patients,
        processed_dir=processed_dir,
        masks_dir=masks_dir,
        roi_size=roi_size,
        is_train=False,
        cache_rate=cache_rate,
    )
    
    train_loader = DataLoader(
        train_dataset.dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset.dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


def create_inference_dataset(
    patient_ids: List[str],
    processed_dir: Union[str, Path],
) -> Dataset:
    """
    Create dataset for inference (no masks required).
    
    Args:
        patient_ids: List of patient IDs
        processed_dir: Directory with preprocessed volumes
        
    Returns:
        MONAI Dataset for inference
    """
    class LoadNpyd:
        def __init__(self, keys: List[str]):
            self.keys = keys
        
        def __call__(self, data: Dict) -> Dict:
            d = dict(data)
            for key in self.keys:
                if key in d and str(d[key]).endswith(".npy"):
                    d[key] = np.load(d[key]).astype(np.float32)
            return d
    
    processed_dir = Path(processed_dir)
    data_list = []
    
    for patient_id in patient_ids:
        volume_path = processed_dir / patient_id / f"{patient_id}.npy"
        if not volume_path.exists():
            volume_path = processed_dir / f"{patient_id}.npy"
        
        if volume_path.exists():
            data_list.append({
                "image": str(volume_path),
                "patient_id": patient_id,
            })
    
    transforms = Compose([
        LoadNpyd(keys=["image"]),
        ToTensord(keys=["image"]),
    ])
    
    return Dataset(data=data_list, transform=transforms)
