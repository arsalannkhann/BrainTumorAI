"""
Publication-quality visualizations for brain tumor evaluation.
Generates 300 DPI figures suitable for IEEE/Springer/MICCAI submission.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


# Set publication-quality defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Color palette for consistent styling
COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "tertiary": "#F18F01",
    "quaternary": "#C73E1D",
    "success": "#2ECC71",
    "warning": "#F39C12",
    "danger": "#E74C3C",
    "info": "#3498DB",
}

CLASS_COLORS = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]


def plot_roc_curves_publication(
    roc_data: Dict,
    output_path: Path,
    title: str = "ROC Curves (One-vs-Rest)",
    figsize: Tuple[int, int] = (8, 7),
) -> None:
    """
    Generate publication-quality ROC curves.
    
    Args:
        roc_data: Dictionary with per-class ROC data
        output_path: Path to save figure
        title: Figure title
        figsize: Figure size in inches
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot per-class ROC curves
    for i, (name, data) in enumerate(roc_data.items()):
        if name in ["micro_average", "macro_average", "summary"]:
            continue
        
        fpr = np.array(data["fpr"]) if isinstance(data["fpr"], list) else data["fpr"]
        tpr = np.array(data["tpr"]) if isinstance(data["tpr"], list) else data["tpr"]
        
        ax.plot(
            fpr, tpr,
            color=CLASS_COLORS[i % len(CLASS_COLORS)],
            lw=2,
            label=f'{name.replace("_", " ").title()} (AUC = {data["auc"]:.3f})',
        )
    
    # Plot micro and macro averages if available
    if "micro_average" in roc_data:
        data = roc_data["micro_average"]
        fpr = np.array(data["fpr"]) if isinstance(data["fpr"], list) else data["fpr"]
        tpr = np.array(data["tpr"]) if isinstance(data["tpr"], list) else data["tpr"]
        ax.plot(
            fpr, tpr,
            color="navy",
            lw=2,
            linestyle="--",
            label=f'Micro-average (AUC = {data["auc"]:.3f})',
        )
    
    if "macro_average" in roc_data:
        data = roc_data["macro_average"]
        fpr = np.array(data["fpr"]) if isinstance(data["fpr"], list) else data["fpr"]
        tpr = np.array(data["tpr"]) if isinstance(data["tpr"], list) else data["tpr"]
        ax.plot(
            fpr, tpr,
            color="deeppink",
            lw=2,
            linestyle=":",
            label=f'Macro-average (AUC = {data["auc"]:.3f})',
        )
    
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random Classifier")
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_aspect("equal")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.savefig(output_path.with_suffix(".pdf"), format="pdf")  # Vector format
    plt.close()


def plot_pr_curves_publication(
    pr_data: Dict,
    output_path: Path,
    title: str = "Precision-Recall Curves",
    figsize: Tuple[int, int] = (8, 7),
) -> None:
    """
    Generate publication-quality Precision-Recall curves.
    
    Args:
        pr_data: Dictionary with per-class PR data
        output_path: Path to save figure
        title: Figure title
        figsize: Figure size in inches
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, (name, data) in enumerate(pr_data.items()):
        if name == "summary":
            continue
        
        precision = np.array(data["precision"]) if isinstance(data["precision"], list) else data["precision"]
        recall = np.array(data["recall"]) if isinstance(data["recall"], list) else data["recall"]
        
        ax.plot(
            recall, precision,
            color=CLASS_COLORS[i % len(CLASS_COLORS)],
            lw=2,
            label=f'{name.replace("_", " ").title()} (AP = {data["ap"]:.3f})',
        )
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left", framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.savefig(output_path.with_suffix(".pdf"), format="pdf")
    plt.close()


def plot_confusion_matrix_publication(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 7),
    cmap: str = "Blues",
) -> None:
    """
    Generate publication-quality confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix
        class_names: Class names
        output_path: Path to save figure
        normalize: Whether to normalize
        title: Figure title
        figsize: Figure size
        cmap: Colormap
    """
    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_display = np.nan_to_num(cm_display)
        fmt = ".2f"
        vmax = 1.0
    else:
        cm_display = cm
        fmt = "d"
        vmax = None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(cm_display, interpolation="nearest", cmap=cmap, vmax=vmax)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Proportion" if normalize else "Count", rotation=-90, va="bottom")
    
    # Set ticks
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels([n.replace("_", "\n") for n in class_names])
    ax.set_yticklabels([n.replace("_", " ").title() for n in class_names])
    
    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm_display.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm_display[i, j]
            color = "white" if val > thresh else "black"
            text = f"{val:{fmt}}" if normalize else f"{int(val)}"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=11)
    
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.savefig(output_path.with_suffix(".pdf"), format="pdf")
    plt.close()


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    output_path: Path = None,
    title: str = "Reliability Diagram",
    figsize: Tuple[int, int] = (8, 7),
) -> None:
    """
    Generate reliability (calibration) diagram.
    
    Args:
        y_true: Ground truth (binary or class indices)
        y_prob: Predicted probabilities
        n_bins: Number of calibration bins
        output_path: Path to save figure
        title: Figure title
        figsize: Figure size
    """
    from evaluation.advanced_metrics import expected_calibration_error
    
    # For multi-class, use max probability
    if y_prob.ndim == 2:
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
        accuracies = (predictions == y_true).astype(float)
    else:
        confidences = y_prob
        accuracies = y_true
    
    # Compute calibration bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bin_centers = (bin_lowers + bin_uppers) / 2
    
    bin_accs = []
    bin_confs = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if in_bin.sum() > 0:
            bin_accs.append(accuracies[in_bin].mean())
            bin_confs.append(confidences[in_bin].mean())
            bin_counts.append(in_bin.sum())
        else:
            bin_accs.append(0)
            bin_confs.append((bin_lower + bin_upper) / 2)
            bin_counts.append(0)
    
    bin_accs = np.array(bin_accs)
    bin_confs = np.array(bin_confs)
    bin_counts = np.array(bin_counts)
    
    # Compute ECE
    ece = np.sum(np.abs(bin_accs - bin_confs) * bin_counts) / len(confidences)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]})
    
    # Main reliability diagram
    bar_width = 0.08
    ax1.bar(
        bin_centers,
        bin_accs,
        width=bar_width,
        color=COLORS["primary"],
        edgecolor="black",
        linewidth=0.5,
        label="Model Accuracy",
        alpha=0.8,
    )
    
    # Gap bars (calibration error)
    gap = bin_accs - bin_confs
    gap_colors = [COLORS["danger"] if g < 0 else COLORS["success"] for g in gap]
    ax1.bar(
        bin_centers,
        gap,
        width=bar_width,
        bottom=bin_confs,
        color=gap_colors,
        alpha=0.4,
        label="Calibration Gap",
    )
    
    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], "k--", lw=2, label="Perfect Calibration")
    
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"{title}\n(ECE = {ece:.4f})")
    ax1.legend(loc="upper left", framealpha=0.9)
    
    # Histogram of predictions
    ax2.bar(
        bin_centers,
        bin_counts / len(confidences),
        width=bar_width,
        color=COLORS["secondary"],
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.set_xlim([0, 1])
    ax2.set_xlabel("Mean Predicted Probability")
    ax2.set_ylabel("Fraction of\nSamples")
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
        plt.savefig(output_path.with_suffix(".pdf"), format="pdf")
    plt.close()


def plot_dice_boxplot(
    dice_scores: Dict[str, List[float]],
    output_path: Path,
    title: str = "Dice Score Distribution",
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    Generate boxplot of Dice scores per class.
    
    Args:
        dice_scores: Dictionary mapping class names to lists of Dice scores
        output_path: Path to save figure
        title: Figure title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data
    labels = list(dice_scores.keys())
    data = [dice_scores[label] for label in labels]
    
    # Create boxplot
    bp = ax.boxplot(
        data,
        labels=[l.replace("_", " ").title() for l in labels],
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black"},
    )
    
    # Color boxes
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(CLASS_COLORS[i % len(CLASS_COLORS)])
        patch.set_alpha(0.7)
    
    # Add individual points
    for i, d in enumerate(data):
        x = np.random.normal(i + 1, 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.4, s=10, color="black")
    
    # Add mean and std annotations
    for i, d in enumerate(data):
        mean = np.mean(d)
        std = np.std(d)
        ax.annotate(
            f"μ={mean:.3f}\nσ={std:.3f}",
            xy=(i + 1, max(d) + 0.02),
            ha="center",
            fontsize=8,
        )
    
    ax.set_ylabel("Dice Score")
    ax.set_title(title)
    ax.set_ylim([0, 1.1])
    ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.5, label="Clinical Threshold (0.80)")
    ax.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.savefig(output_path.with_suffix(".pdf"), format="pdf")
    plt.close()


def plot_segmentation_overlay(
    image: np.ndarray,
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    slice_idx: Optional[int] = None,
    output_path: Path = None,
    class_names: List[str] = ["Background", "Edema", "Enhancing", "Necrotic"],
    title: str = "Segmentation Results",
    figsize: Tuple[int, int] = (15, 5),
) -> None:
    """
    Generate side-by-side segmentation overlay visualization.
    
    Args:
        image: Input MRI volume (H, W, D) or (C, H, W, D)
        prediction: Predicted segmentation
        ground_truth: Ground truth segmentation
        slice_idx: Slice index to visualize (auto-select if None)
        output_path: Path to save figure
        class_names: Names for each class
        title: Figure title
        figsize: Figure size
    """
    # Handle multi-channel input
    if image.ndim == 4:
        image = image[0]  # Use first modality (T1c typically)
    
    # Auto-select slice with most tumor
    if slice_idx is None:
        tumor_per_slice = np.sum(ground_truth > 0, axis=(0, 1))
        slice_idx = np.argmax(tumor_per_slice)
    
    # Extract slices
    img_slice = image[:, :, slice_idx]
    pred_slice = prediction[:, :, slice_idx]
    gt_slice = ground_truth[:, :, slice_idx]
    
    # Normalize image for display
    img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
    
    # Create color overlays
    colors = np.array([
        [0, 0, 0, 0],        # Background - transparent
        [0, 1, 0, 0.5],      # Edema - green
        [1, 1, 0, 0.5],      # Enhancing - yellow
        [1, 0, 0, 0.5],      # Necrotic - red
    ])
    
    def create_overlay(mask, colors):
        overlay = np.zeros((*mask.shape, 4))
        for i in range(len(colors)):
            overlay[mask == i] = colors[i]
        return overlay
    
    pred_overlay = create_overlay(pred_slice, colors)
    gt_overlay = create_overlay(gt_slice, colors)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Input image
    axes[0].imshow(img_slice, cmap="gray")
    axes[0].set_title("Input MRI")
    axes[0].axis("off")
    
    # Ground truth
    axes[1].imshow(img_slice, cmap="gray")
    axes[1].imshow(gt_overlay)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")
    
    # Prediction
    axes[2].imshow(img_slice, cmap="gray")
    axes[2].imshow(pred_overlay)
    axes[2].set_title("Prediction")
    axes[2].axis("off")
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[i][:3], alpha=colors[i][3], label=class_names[i])
        for i in range(1, len(class_names))
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=len(class_names) - 1, bbox_to_anchor=(0.5, 0.0))
    
    fig.suptitle(f"{title} (Slice {slice_idx})", fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_threshold_analysis(
    confidences: np.ndarray,
    correct: np.ndarray,
    output_path: Path,
    title: str = "Confidence Threshold Analysis",
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    Plot accuracy vs confidence threshold curve.
    
    Args:
        confidences: Array of prediction confidences
        correct: Boolean array indicating correct predictions
        output_path: Path to save figure
        title: Figure title
        figsize: Figure size
    """
    thresholds = np.linspace(0, 1, 101)
    accuracies = []
    coverages = []
    
    for thresh in thresholds:
        mask = confidences >= thresh
        if mask.sum() > 0:
            acc = correct[mask].mean()
            cov = mask.mean()
        else:
            acc = 0
            cov = 0
        accuracies.append(acc)
        coverages.append(cov)
    
    fig, ax1 = plt.subplots(figsize=figsize)
    
    ax1.plot(thresholds, accuracies, color=COLORS["primary"], lw=2, label="Accuracy")
    ax1.set_xlabel("Confidence Threshold")
    ax1.set_ylabel("Accuracy", color=COLORS["primary"])
    ax1.tick_params(axis="y", labelcolor=COLORS["primary"])
    ax1.set_ylim([0, 1.05])
    
    ax2 = ax1.twinx()
    ax2.plot(thresholds, coverages, color=COLORS["secondary"], lw=2, linestyle="--", label="Coverage")
    ax2.set_ylabel("Coverage (Fraction Retained)", color=COLORS["secondary"])
    ax2.tick_params(axis="y", labelcolor=COLORS["secondary"])
    ax2.set_ylim([0, 1.05])
    
    # Find optimal threshold (maximize accuracy * coverage)
    optimal_idx = np.argmax(np.array(accuracies) * np.array(coverages))
    optimal_thresh = thresholds[optimal_idx]
    ax1.axvline(x=optimal_thresh, color="gray", linestyle=":", alpha=0.7)
    ax1.annotate(
        f"Optimal: {optimal_thresh:.2f}",
        xy=(optimal_thresh, accuracies[optimal_idx]),
        xytext=(optimal_thresh + 0.1, accuracies[optimal_idx] - 0.1),
        arrowprops=dict(arrowstyle="->", color="gray"),
    )
    
    fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88))
    ax1.set_title(title)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.savefig(output_path.with_suffix(".pdf"), format="pdf")
    plt.close()


def plot_training_curves(
    history: Dict[str, List[float]],
    output_path: Path,
    title: str = "Training History",
    figsize: Tuple[int, int] = (12, 5),
) -> None:
    """
    Plot training and validation curves.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', and optional metrics
        output_path: Path to save figure
        title: Figure title
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    
    # Loss curves
    if "train_loss" in history:
        axes[0].plot(epochs, history["train_loss"], label="Training Loss", color=COLORS["primary"], lw=2)
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], label="Validation Loss", color=COLORS["secondary"], lw=2)
    
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    
    # Metric curves
    metric_keys = [k for k in history.keys() if k not in ["train_loss", "val_loss"]]
    for i, key in enumerate(metric_keys[:4]):  # Max 4 metrics
        axes[1].plot(epochs[:len(history[key])], history[key], label=key.replace("_", " ").title(), lw=2)
    
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Metric Value")
    axes[1].set_title("Metric Curves")
    if metric_keys:
        axes[1].legend()
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.savefig(output_path.with_suffix(".pdf"), format="pdf")
    plt.close()


def generate_all_classification_figures(
    results: Dict,
    output_dir: Path,
    class_names: List[str],
) -> None:
    """
    Generate all classification figures.
    
    Args:
        results: Dictionary with all classification results
        output_dir: Output directory
        class_names: Class names
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ROC curves
    if "roc_data" in results:
        plot_roc_curves_publication(
            results["roc_data"],
            output_dir / "roc_curves.png",
        )
    
    # PR curves
    if "pr_data" in results:
        plot_pr_curves_publication(
            results["pr_data"],
            output_dir / "pr_curves.png",
        )
    
    # Confusion matrix
    if "confusion_matrix" in results:
        cm = np.array(results["confusion_matrix"])
        plot_confusion_matrix_publication(
            cm,
            class_names,
            output_dir / "confusion_matrix.png",
            normalize=True,
        )
        plot_confusion_matrix_publication(
            cm,
            class_names,
            output_dir / "confusion_matrix_raw.png",
            normalize=False,
            title="Confusion Matrix (Raw Counts)",
        )
    
    # Reliability diagram
    if "probabilities" in results and "targets" in results:
        plot_reliability_diagram(
            np.array(results["targets"]),
            np.array(results["probabilities"]),
            output_path=output_dir / "reliability_diagram.png",
        )
    
    # Threshold analysis
    if "confidences" in results and "correct" in results:
        plot_threshold_analysis(
            np.array(results["confidences"]),
            np.array(results["correct"]),
            output_dir / "threshold_analysis.png",
        )
