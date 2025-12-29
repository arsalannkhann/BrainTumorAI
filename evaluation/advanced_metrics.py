"""
Advanced evaluation metrics for publication-grade analysis.
Extends base metrics with calibration, statistical, and distance-based metrics.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from scipy import stats
from scipy.ndimage import distance_transform_edt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


# ============================================================================
# Calibration Metrics
# ============================================================================


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures the difference between predicted probabilities and actual
    outcome frequencies, weighted by the number of samples in each bin.
    
    Args:
        y_true: Binary ground truth labels
        y_prob: Predicted probabilities for the positive class
        n_bins: Number of bins for calibration
        
    Returns:
        Tuple of (ECE value, bin accuracies, bin confidences, bin counts)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            bin_accs.append(accuracy_in_bin)
            bin_confs.append(avg_confidence_in_bin)
            bin_counts.append(in_bin.sum())
        else:
            bin_accs.append(0)
            bin_confs.append((bin_lower + bin_upper) / 2)
            bin_counts.append(0)
    
    return ece, np.array(bin_accs), np.array(bin_confs), np.array(bin_counts)


def maximum_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Maximum Calibration Error (MCE).
    
    MCE is the maximum absolute difference between predicted probabilities
    and actual outcome frequencies across all bins.
    
    Args:
        y_true: Binary ground truth labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        
    Returns:
        MCE value
    """
    _, bin_accs, bin_confs, bin_counts = expected_calibration_error(
        y_true, y_prob, n_bins
    )
    
    # Only consider non-empty bins
    mask = np.array(bin_counts) > 0
    if not mask.any():
        return 0.0
    
    return float(np.max(np.abs(np.array(bin_accs)[mask] - np.array(bin_confs)[mask])))


def multiclass_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute ECE for multi-class classification.
    
    Uses the confidence (max probability) approach.
    
    Args:
        y_true: Ground truth class labels
        y_prob: Predicted probabilities (N, num_classes)
        n_bins: Number of bins
        
    Returns:
        Multi-class ECE value
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return float(ece)


def multiclass_brier_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """
    Compute multi-class Brier score.
    
    Args:
        y_true: Ground truth class labels
        y_prob: Predicted probabilities (N, num_classes)
        
    Returns:
        Brier score (lower is better)
    """
    n_samples = len(y_true)
    n_classes = y_prob.shape[1]
    
    # One-hot encode targets
    y_true_onehot = np.zeros((n_samples, n_classes))
    y_true_onehot[np.arange(n_samples), y_true] = 1
    
    return float(np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1)))


# ============================================================================
# Per-Class Specificity
# ============================================================================


def compute_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute per-class specificity (true negative rate).
    
    Specificity = TN / (TN + FP)
    
    Args:
        y_true: Ground truth class labels
        y_pred: Predicted class labels
        num_classes: Number of classes (inferred if not provided)
        
    Returns:
        Dictionary with per-class specificity
    """
    if num_classes is None:
        num_classes = max(y_true.max(), y_pred.max()) + 1
    
    specificities = {}
    for c in range(num_classes):
        # Binary conversion: class c vs rest
        y_true_binary = (y_true == c).astype(int)
        y_pred_binary = (y_pred == c).astype(int)
        
        # True negatives and false positives
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities[f"class_{c}"] = float(specificity)
    
    specificities["macro"] = float(np.mean(list(specificities.values())))
    return specificities


# ============================================================================
# Distance-Based Segmentation Metrics
# ============================================================================


def compute_average_surface_distance(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """
    Compute Average Symmetric Surface Distance (ASD).
    
    Args:
        pred: Binary prediction mask
        target: Binary ground truth mask
        spacing: Voxel spacing in mm
        
    Returns:
        ASD in mm
    """
    pred = pred.astype(bool)
    target = target.astype(bool)
    
    if not np.any(pred) or not np.any(target):
        return float("inf")
    
    # Distance transforms
    pred_dist = distance_transform_edt(~pred, sampling=spacing)
    target_dist = distance_transform_edt(~target, sampling=spacing)
    
    # Surface distances
    pred_surface_distances = pred_dist[target]
    target_surface_distances = target_dist[pred]
    
    # Average of all surface distances
    all_distances = np.concatenate([pred_surface_distances, target_surface_distances])
    return float(np.mean(all_distances))


def compute_hausdorff_distance_95(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """
    Compute 95th percentile Hausdorff Distance (HD95).
    
    More robust than max HD as it's less sensitive to outliers.
    
    Args:
        pred: Binary prediction mask
        target: Binary ground truth mask
        spacing: Voxel spacing in mm
        
    Returns:
        HD95 in mm
    """
    pred = pred.astype(bool)
    target = target.astype(bool)
    
    if not np.any(pred) or not np.any(target):
        return float("inf")
    
    # Distance transforms with spacing
    pred_dist = distance_transform_edt(~pred, sampling=spacing)
    target_dist = distance_transform_edt(~target, sampling=spacing)
    
    # Surface distances
    pred_surface_distances = pred_dist[target]
    target_surface_distances = target_dist[pred]
    
    all_distances = np.concatenate([pred_surface_distances, target_surface_distances])
    return float(np.percentile(all_distances, 95))


# ============================================================================
# Volume Metrics
# ============================================================================


def compute_volume_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, float]:
    """
    Compute volume-based metrics.
    
    Args:
        pred: Binary prediction mask
        target: Binary ground truth mask
        spacing: Voxel spacing in mm
        
    Returns:
        Dictionary with volume metrics
    """
    voxel_volume = np.prod(spacing)  # mm^3 per voxel
    
    pred_volume = np.sum(pred.astype(bool)) * voxel_volume
    target_volume = np.sum(target.astype(bool)) * voxel_volume
    
    abs_error = abs(pred_volume - target_volume)
    rel_error = abs_error / target_volume if target_volume > 0 else float("inf")
    
    return {
        "pred_volume_mm3": float(pred_volume),
        "target_volume_mm3": float(target_volume),
        "absolute_volume_error_mm3": float(abs_error),
        "relative_volume_error_pct": float(rel_error * 100),
    }


# ============================================================================
# Statistical Analysis
# ============================================================================


def bootstrap_confidence_interval(
    values: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.
    
    Args:
        values: Array of metric values
        confidence: Confidence level (default 95%)
        n_bootstrap: Number of bootstrap samples
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (mean, lower bound, upper bound)
    """
    rng = np.random.RandomState(seed)
    n = len(values)
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
    mean = np.mean(values)
    
    return float(mean), float(lower), float(upper)


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size.
    
    Args:
        group1: First group of values
        group2: Second group of values
        
    Returns:
        Cohen's d value
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def wilcoxon_test(group1: np.ndarray, group2: np.ndarray) -> Dict[str, float]:
    """
    Perform Wilcoxon signed-rank test for paired samples.
    
    Args:
        group1: First group of values
        group2: Second group of values
        
    Returns:
        Dictionary with test statistic and p-value
    """
    statistic, p_value = stats.wilcoxon(group1, group2, alternative="two-sided")
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
    }


# ============================================================================
# ROC/PR Curve Metrics
# ============================================================================


def compute_micro_macro_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """
    Compute micro-averaged and macro-averaged ROC curves.
    
    Args:
        y_true: Ground truth class labels
        y_prob: Predicted probabilities (N, num_classes)
        class_names: Optional class names
        
    Returns:
        Dictionary with per-class, micro, and macro ROC data
    """
    n_classes = y_prob.shape[1]
    if class_names is None:
        class_names = [f"class_{i}" for i in range(n_classes)]
    
    # Binarize labels
    y_true_bin = np.zeros_like(y_prob)
    y_true_bin[np.arange(len(y_true)), y_true] = 1
    
    roc_data = {}
    
    # Per-class ROC
    all_fpr = []
    all_tpr = []
    all_auc = []
    
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        
        roc_data[name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": float(roc_auc),
        }
        
        all_fpr.append(fpr)
        all_tpr.append(tpr)
        all_auc.append(roc_auc)
    
    # Micro-averaged ROC
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    
    roc_data["micro_average"] = {
        "fpr": fpr_micro.tolist(),
        "tpr": tpr_micro.tolist(),
        "auc": float(roc_auc_micro),
    }
    
    # Macro-averaged ROC (average of class ROCs)
    # First, aggregate all unique FPR points
    all_fpr_unique = np.unique(np.concatenate(all_fpr))
    
    # Interpolate all TPR at these FPR points
    mean_tpr = np.zeros_like(all_fpr_unique)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr_unique, all_fpr[i], all_tpr[i])
    mean_tpr /= n_classes
    
    roc_auc_macro = auc(all_fpr_unique, mean_tpr)
    
    roc_data["macro_average"] = {
        "fpr": all_fpr_unique.tolist(),
        "tpr": mean_tpr.tolist(),
        "auc": float(roc_auc_macro),
    }
    
    # Summary statistics
    roc_data["summary"] = {
        "mean_auc": float(np.mean(all_auc)),
        "std_auc": float(np.std(all_auc)),
        "micro_auc": float(roc_auc_micro),
        "macro_auc": float(roc_auc_macro),
    }
    
    return roc_data


def compute_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """
    Compute Precision-Recall curves with PR-AUC.
    
    Args:
        y_true: Ground truth class labels
        y_prob: Predicted probabilities (N, num_classes)
        class_names: Optional class names
        
    Returns:
        Dictionary with per-class PR data
    """
    n_classes = y_prob.shape[1]
    if class_names is None:
        class_names = [f"class_{i}" for i in range(n_classes)]
    
    # Binarize labels
    y_true_bin = np.zeros_like(y_prob)
    y_true_bin[np.arange(len(y_true)), y_true] = 1
    
    pr_data = {}
    all_ap = []
    
    for i, name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_prob[:, i])
        
        pr_data[name] = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "ap": float(ap),
        }
        all_ap.append(ap)
    
    pr_data["summary"] = {
        "mean_ap": float(np.mean(all_ap)),
        "std_ap": float(np.std(all_ap)),
    }
    
    return pr_data


# ============================================================================
# BraTS Aggregate Metrics
# ============================================================================


def compute_brats_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, Dict[str, float]]:
    """
    Compute BraTS challenge aggregate metrics.
    
    Classes: 0=Background, 1=Edema, 2=Enhancing, 3=Necrotic
    
    Regions:
    - Whole Tumor (WT): Edema + Enhancing + Necrotic (1, 2, 3)
    - Tumor Core (TC): Enhancing + Necrotic (2, 3)
    - Enhancing Tumor (ET): Enhancing only (2)
    
    Args:
        pred: Predicted segmentation (H, W, D) with class indices
        target: Ground truth segmentation
        spacing: Voxel spacing in mm
        
    Returns:
        Dictionary with Dice, HD95, and ASD for each region
    """
    from utils.metrics import compute_dice, compute_iou
    
    # Region masks
    # Whole Tumor: classes 1, 2, 3
    pred_wt = (pred == 1) | (pred == 2) | (pred == 3)
    target_wt = (target == 1) | (target == 2) | (target == 3)
    
    # Tumor Core: classes 2, 3
    pred_tc = (pred == 2) | (pred == 3)
    target_tc = (target == 2) | (target == 3)
    
    # Enhancing Tumor: class 2
    pred_et = (pred == 2)
    target_et = (target == 2)
    
    results = {}
    
    for name, (p, t) in [
        ("whole_tumor", (pred_wt, target_wt)),
        ("tumor_core", (pred_tc, target_tc)),
        ("enhancing_tumor", (pred_et, target_et)),
    ]:
        results[name] = {
            "dice": compute_dice(p, t),
            "iou": compute_iou(p, t),
            "hd95": compute_hausdorff_distance_95(p, t, spacing),
            "asd": compute_average_surface_distance(p, t, spacing),
        }
        
        # Add volume metrics
        vol_metrics = compute_volume_metrics(p, t, spacing)
        results[name].update(vol_metrics)
    
    return results


def compute_comprehensive_segmentation_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    class_names: List[str] = ["background", "edema", "enhancing", "necrotic"],
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, Dict[str, float]]:
    """
    Compute all segmentation metrics for publication.
    
    Args:
        pred: Predicted segmentation (H, W, D)
        target: Ground truth segmentation
        class_names: Names for each class
        spacing: Voxel spacing in mm
        
    Returns:
        Comprehensive metrics dictionary
    """
    from utils.metrics import compute_dice, compute_iou
    
    results = {}
    
    # Per-class metrics (skip background)
    dice_scores = []
    iou_scores = []
    
    for i in range(1, len(class_names)):
        pred_i = (pred == i)
        target_i = (target == i)
        
        dice = compute_dice(pred_i, target_i)
        iou = compute_iou(pred_i, target_i)
        hd95 = compute_hausdorff_distance_95(pred_i, target_i, spacing)
        asd = compute_average_surface_distance(pred_i, target_i, spacing)
        vol = compute_volume_metrics(pred_i, target_i, spacing)
        
        # Sensitivity (recall) and specificity
        tp = np.sum(pred_i & target_i)
        fp = np.sum(pred_i & ~target_i)
        fn = np.sum(~pred_i & target_i)
        tn = np.sum(~pred_i & ~target_i)
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        results[class_names[i]] = {
            "dice": float(dice),
            "iou": float(iou),
            "hd95": float(hd95) if hd95 != float("inf") else None,
            "asd": float(asd) if asd != float("inf") else None,
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "precision": float(precision),
            **vol,
        }
        
        dice_scores.append(dice)
        iou_scores.append(iou)
    
    # Aggregate metrics
    results["mean"] = {
        "dice": float(np.mean(dice_scores)),
        "iou": float(np.mean(iou_scores)),
    }
    
    # BraTS aggregate regions
    brats = compute_brats_metrics(pred, target, spacing)
    results["whole_tumor"] = brats["whole_tumor"]
    results["tumor_core"] = brats["tumor_core"]
    results["enhancing_tumor"] = brats["enhancing_tumor"]
    
    return results
