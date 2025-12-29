"""
Evaluation script for brain tumor classification.
Generates comprehensive metrics, confusion matrices, and ROC curves.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from classification.dataset import BrainTumorClassificationDataset
from classification.model import load_classifier_from_checkpoint
from utils import (
    compute_classification_metrics,
    get_classification_report,
    get_confusion_matrix,
    get_roc_curves,
    load_config,
    load_labels,
    load_patient_list,
    set_seed,
)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Classification model
        dataloader: Data loader
        device: Device
        use_amp: Whether to use AMP
        
    Returns:
        Tuple of (predictions, targets, probabilities, patient_ids)
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = []
    all_patient_ids = []
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch["image"].to(device)
        labels = batch["label"]
        patient_ids = batch["patient_id"]
        
        with autocast("cuda", enabled=use_amp):
            outputs = model(images)
        
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
        all_patient_ids.extend(patient_ids)
    
    return (
        np.array(all_preds),
        np.array(all_targets),
        np.array(all_probs),
        all_patient_ids,
    )


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path,
    normalize: bool = True,
) -> None:
    """
    Plot and save confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: Class names
        output_path: Path to save figure
        normalize: Whether to normalize values
    """
    if normalize:
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_display = cm_normalized
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved confusion matrix to {output_path}")


def plot_roc_curves(
    roc_data: Dict,
    output_path: Path,
) -> None:
    """
    Plot and save ROC curves.
    
    Args:
        roc_data: Dictionary with ROC data per class
        output_path: Path to save figure
    """
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(roc_data)))
    
    for i, (class_name, data) in enumerate(roc_data.items()):
        plt.plot(
            data["fpr"],
            data["tpr"],
            color=colors[i],
            lw=2,
            label=f'{class_name} (AUC = {data["auc"]:.3f})',
        )
    
    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved ROC curves to {output_path}")


def plot_per_patient_analysis(
    predictions: np.ndarray,
    targets: np.ndarray,
    probabilities: np.ndarray,
    patient_ids: List[str],
    class_names: List[str],
    output_path: Path,
) -> None:
    """
    Plot per-patient prediction confidence analysis.
    
    Args:
        predictions: Model predictions
        targets: Ground truth
        probabilities: Prediction probabilities
        patient_ids: Patient identifiers
        class_names: Class names
        output_path: Path to save figure
    """
    # Get confidence of predicted class
    confidences = probabilities.max(axis=1)
    correct = predictions == targets
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Confidence distribution
    plt.subplot(1, 2, 1)
    plt.hist(
        [confidences[correct], confidences[~correct]],
        bins=20,
        label=["Correct", "Incorrect"],
        alpha=0.7,
    )
    plt.xlabel("Prediction Confidence")
    plt.ylabel("Count")
    plt.title("Confidence Distribution")
    plt.legend()
    
    # Plot 2: Per-class accuracy
    plt.subplot(1, 2, 2)
    class_accuracies = []
    for i, name in enumerate(class_names):
        mask = targets == i
        if mask.sum() > 0:
            acc = (predictions[mask] == targets[mask]).mean()
            class_accuracies.append(acc)
        else:
            class_accuracies.append(0)
    
    plt.bar(class_names, class_accuracies, color="steelblue")
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy")
    plt.xticks(rotation=45, ha="right")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved per-patient analysis to {output_path}")


def run_evaluation(
    checkpoint_path: str,
    patient_list_path: str,
    roi_dir: str,
    labels_file: str,
    output_dir: str,
    config_path: Optional[str] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    use_amp: bool = True,
    device: Optional[str] = None,
) -> Dict:
    """
    Run full evaluation pipeline.
    
    Args:
        checkpoint_path: Path to model checkpoint
        patient_list_path: Path to patient list file
        roi_dir: Directory with ROI files
        labels_file: Path to labels CSV
        output_dir: Output directory for results
        config_path: Optional path to config
        batch_size: Batch size
        num_workers: Number of workers
        use_amp: Whether to use AMP
        device: Device to use
        
    Returns:
        Dictionary of evaluation metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device setup
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {checkpoint_path}")
    model = load_classifier_from_checkpoint(checkpoint_path, device=str(device))
    
    # Get model config from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint.get("model_config", {})
    class_names = checkpoint.get("class_names", None)
    
    # Load configuration for dataset settings
    if config_path:
        config = load_config(config_path)
        aug_config = config.get("augmentation", {})
        num_slices = aug_config.get("num_slices", 16)
        input_size = tuple(aug_config.get("input_size", [224, 224]))
        if class_names is None:
            class_names = config["classes"].get("names")
    else:
        num_slices = model_config.get("num_slices", 16)
        input_size = (224, 224)
    
    # Load patient list and labels
    patient_ids = load_patient_list(patient_list_path)
    labels = load_labels(labels_file)
    
    num_classes = len(set(labels.values()))
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]
    
    print(f"Evaluating {len(patient_ids)} patients")
    print(f"Classes: {class_names}")
    
    # Create dataset
    dataset = BrainTumorClassificationDataset(
        patient_ids=patient_ids,
        roi_dir=roi_dir,
        labels=labels,
        mode=model_config.get("mode", "2.5d"),
        num_slices=num_slices,
        input_size=input_size,
        is_train=False,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    # Run evaluation
    predictions, targets, probabilities, eval_patient_ids = evaluate(
        model=model,
        dataloader=dataloader,
        device=device,
        use_amp=use_amp,
    )
    
    # Compute metrics
    metrics = compute_classification_metrics(
        pred=predictions,
        target=targets,
        pred_probs=probabilities,
        class_names=class_names,
    )
    
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Classification report
    report = get_classification_report(predictions, targets, class_names)
    print("\nClassification Report:")
    print(report)
    
    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")
    
    # Save classification report
    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    
    # Save per-patient predictions
    predictions_path = output_dir / "predictions.csv"
    import pandas as pd
    
    pred_df = pd.DataFrame({
        "patient_id": eval_patient_ids,
        "true_label": targets,
        "predicted_label": predictions,
        "true_class": [class_names[t] for t in targets],
        "predicted_class": [class_names[p] for p in predictions],
        "correct": predictions == targets,
    })
    
    # Add per-class probabilities
    for i, name in enumerate(class_names):
        pred_df[f"prob_{name}"] = probabilities[:, i]
    
    pred_df.to_csv(predictions_path, index=False)
    print(f"Saved predictions to {predictions_path}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Confusion matrix
    cm, _ = get_confusion_matrix(predictions, targets, class_names)
    plot_confusion_matrix(
        cm,
        class_names,
        output_dir / "confusion_matrix.png",
        normalize=True,
    )
    
    # ROC curves
    roc_data = get_roc_curves(targets, probabilities, class_names)
    plot_roc_curves(roc_data, output_dir / "roc_curves.png")
    
    # Per-patient analysis
    plot_per_patient_analysis(
        predictions,
        targets,
        probabilities,
        eval_patient_ids,
        class_names,
        output_dir / "patient_analysis.png",
    )
    
    print(f"\nAll results saved to {output_dir}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Brain Tumor Classification Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on test set
  python -m classification.evaluate \\
      --checkpoint checkpoints/classification/best_model.pt \\
      --patient-list data/splits/test.txt \\
      --roi-dir data/roi \\
      --labels-file data/labels.csv \\
      --output-dir results/evaluation
        """,
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--patient-list",
        type=str,
        required=True,
        help="Path to patient list file",
    )
    parser.add_argument(
        "--roi-dir",
        type=str,
        required=True,
        help="Directory containing ROI files",
    )
    parser.add_argument(
        "--labels-file",
        type=str,
        required=True,
        help="Path to labels CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    run_evaluation(
        checkpoint_path=args.checkpoint,
        patient_list_path=args.patient_list,
        roi_dir=args.roi_dir,
        labels_file=args.labels_file,
        output_dir=args.output_dir,
        config_path=args.config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
        device=args.device,
    )


if __name__ == "__main__":
    main()
