"""
Evaluation metrics for brain tumor segmentation and classification.
Includes both volumetric (3D) and slice-level metrics.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ============================================================================
# Segmentation Metrics
# ============================================================================


def compute_dice(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    smooth: float = 1e-5,
) -> float:
    """
    Compute Dice Similarity Coefficient.
    
    Args:
        pred: Binary prediction mask
        target: Binary ground truth mask
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice coefficient (0-1)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    pred = pred.astype(bool).flatten()
    target = target.astype(bool).flatten()
    
    intersection = np.sum(pred & target)
    return (2.0 * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)


def compute_iou(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    smooth: float = 1e-5,
) -> float:
    """
    Compute Intersection over Union (Jaccard Index).
    
    Args:
        pred: Binary prediction mask
        target: Binary ground truth mask
        smooth: Smoothing factor
        
    Returns:
        IoU score (0-1)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    pred = pred.astype(bool).flatten()
    target = target.astype(bool).flatten()
    
    intersection = np.sum(pred & target)
    union = np.sum(pred | target)
    return (intersection + smooth) / (union + smooth)


def compute_hausdorff_distance(
    pred: np.ndarray,
    target: np.ndarray,
    percentile: float = 95,
) -> float:
    """
    Compute Hausdorff Distance between two binary masks.
    Uses the 95th percentile by default for robustness.
    
    Args:
        pred: Binary prediction mask
        target: Binary ground truth mask
        percentile: Percentile for robust HD computation
        
    Returns:
        Hausdorff distance in voxels
    """
    from scipy.ndimage import distance_transform_edt
    
    pred = pred.astype(bool)
    target = target.astype(bool)
    
    if not np.any(pred) or not np.any(target):
        return float("inf")
    
    # Distance transform of inverted masks
    pred_dist = distance_transform_edt(~pred)
    target_dist = distance_transform_edt(~target)
    
    # Surface distances
    pred_surface = pred_dist[target]
    target_surface = target_dist[pred]
    
    all_distances = np.concatenate([pred_surface, target_surface])
    
    return float(np.percentile(all_distances, percentile))


def compute_segmentation_metrics(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    num_classes: int = 4,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive segmentation metrics for multi-class masks.
    
    Args:
        pred: Predicted segmentation (H, W, D) or (C, H, W, D)
        target: Ground truth segmentation
        num_classes: Number of classes including background
        class_names: Names for each class
        
    Returns:
        Dictionary of metrics per class and mean
    """
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]
    
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    # If one-hot encoded, convert to class indices
    if pred.ndim == 4 and pred.shape[0] == num_classes:
        pred = np.argmax(pred, axis=0)
    if target.ndim == 4 and target.shape[0] == num_classes:
        target = np.argmax(target, axis=0)
    
    metrics = {}
    dice_scores = []
    iou_scores = []
    
    # Per-class metrics (skip background at index 0)
    for i in range(1, num_classes):
        pred_i = (pred == i)
        target_i = (target == i)
        
        dice = compute_dice(pred_i, target_i)
        iou = compute_iou(pred_i, target_i)
        
        metrics[f"{class_names[i]}_dice"] = dice
        metrics[f"{class_names[i]}_iou"] = iou
        
        dice_scores.append(dice)
        iou_scores.append(iou)
    
    metrics["mean_dice"] = float(np.mean(dice_scores))
    metrics["mean_iou"] = float(np.mean(iou_scores))
    
    return metrics


# ============================================================================
# Classification Metrics
# ============================================================================


def compute_accuracy(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
) -> float:
    """
    Compute classification accuracy.
    
    Args:
        pred: Predicted class labels
        target: Ground truth class labels
        
    Returns:
        Accuracy score (0-1)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    return float(accuracy_score(target, pred))


def compute_classification_metrics(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    pred_probs: Optional[Union[np.ndarray, torch.Tensor]] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        pred: Predicted class labels
        target: Ground truth class labels
        pred_probs: Prediction probabilities for AUC-ROC
        class_names: Names for each class
        
    Returns:
        Dictionary of metrics
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    if pred_probs is not None and isinstance(pred_probs, torch.Tensor):
        pred_probs = pred_probs.cpu().numpy()
    
    metrics = {
        "accuracy": float(accuracy_score(target, pred)),
        "precision_macro": float(precision_score(target, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(target, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(target, pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(target, pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(target, pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(target, pred, average="weighted", zero_division=0)),
    }
    
    # AUC-ROC if probabilities provided
    if pred_probs is not None:
        try:
            num_classes = pred_probs.shape[1]
            if num_classes == 2:
                metrics["auc_roc"] = float(roc_auc_score(target, pred_probs[:, 1]))
            else:
                metrics["auc_roc_macro"] = float(
                    roc_auc_score(target, pred_probs, multi_class="ovr", average="macro")
                )
                metrics["auc_roc_weighted"] = float(
                    roc_auc_score(target, pred_probs, multi_class="ovr", average="weighted")
                )
        except ValueError:
            # Not all classes present in target
            pass
    
    return metrics


def get_confusion_matrix(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    class_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, Optional[List[str]]]:
    """
    Compute confusion matrix.
    
    Args:
        pred: Predicted class labels
        target: Ground truth class labels
        class_names: Names for each class
        
    Returns:
        Confusion matrix and class names
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    cm = confusion_matrix(target, pred)
    return cm, class_names


def get_roc_curves(
    target: Union[np.ndarray, torch.Tensor],
    pred_probs: Union[np.ndarray, torch.Tensor],
    class_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute ROC curves for each class (one-vs-rest).
    
    Args:
        target: Ground truth class labels
        pred_probs: Prediction probabilities (N, num_classes)
        class_names: Names for each class
        
    Returns:
        Dictionary with FPR, TPR, and AUC for each class
    """
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    if isinstance(pred_probs, torch.Tensor):
        pred_probs = pred_probs.cpu().numpy()
    
    num_classes = pred_probs.shape[1]
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]
    
    roc_data = {}
    
    for i, name in enumerate(class_names):
        binary_target = (target == i).astype(int)
        fpr, tpr, _ = roc_curve(binary_target, pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        
        roc_data[name] = {
            "fpr": fpr,
            "tpr": tpr,
            "auc": roc_auc,
        }
    
    return roc_data


def get_classification_report(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    class_names: Optional[List[str]] = None,
) -> str:
    """
    Generate a text classification report.
    
    Args:
        pred: Predicted class labels
        target: Ground truth class labels
        class_names: Names for each class
        
    Returns:
        Formatted classification report string
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    return classification_report(target, pred, target_names=class_names, zero_division=0)
