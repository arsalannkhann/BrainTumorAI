"""
Comprehensive classification evaluation for publication.
Generates all metrics, visualizations, and tables required for peer-reviewed papers.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from classification.dataset import BrainTumorClassificationDataset
from classification.model import load_classifier_from_checkpoint
from evaluation.advanced_metrics import (
    bootstrap_confidence_interval,
    compute_micro_macro_roc,
    compute_pr_curves,
    compute_specificity,
    multiclass_brier_score,
    multiclass_ece,
)
from evaluation.visualizations import (
    generate_all_classification_figures,
    plot_confusion_matrix_publication,
    plot_pr_curves_publication,
    plot_reliability_diagram,
    plot_roc_curves_publication,
    plot_threshold_analysis,
)
from evaluation.latex_tables import (
    generate_classification_table,
    generate_per_class_table,
    generate_roc_auc_table,
)
from utils import (
    compute_classification_metrics,
    get_classification_report,
    get_confusion_matrix,
    load_config,
    load_labels,
    load_patient_list,
    set_seed,
)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], float]:
    """
    Evaluate model on a dataset with timing.
    
    Args:
        model: Classification model
        dataloader: Data loader
        device: Device
        use_amp: Whether to use AMP
        
    Returns:
        Tuple of (predictions, targets, probabilities, patient_ids, inference_time_ms)
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = []
    all_patient_ids = []
    total_time = 0
    total_samples = 0
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch["image"].to(device)
        labels = batch["label"]
        patient_ids = batch["patient_id"]
        
        start = time.perf_counter()
        with autocast("cuda", enabled=use_amp):
            outputs = model(images)
        torch.cuda.synchronize() if device.type == "cuda" else None
        end = time.perf_counter()
        
        total_time += (end - start) * 1000  # ms
        total_samples += len(images)
        
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
        all_patient_ids.extend(patient_ids)
    
    avg_inference_time = total_time / total_samples
    
    return (
        np.array(all_preds),
        np.array(all_targets),
        np.array(all_probs),
        all_patient_ids,
        avg_inference_time,
    )


def compute_per_class_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_names: List[str],
) -> Dict[str, Dict]:
    """
    Compute detailed per-class metrics.
    
    Args:
        predictions: Model predictions
        targets: Ground truth
        probabilities: Prediction probabilities
        class_names: Class names
        
    Returns:
        Dictionary with per-class metrics
    """
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    per_class = {}
    specificities = compute_specificity(targets, predictions, len(class_names))
    
    for i, name in enumerate(class_names):
        # Binary mask for this class
        y_true = (targets == i).astype(int)
        y_pred = (predictions == i).astype(int)
        
        # Metrics
        tp = np.sum(y_true & y_pred)
        fp = np.sum(~y_true.astype(bool) & y_pred.astype(bool))
        fn = np.sum(y_true.astype(bool) & ~y_pred.astype(bool))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # AUC for this class
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(y_true, probabilities[:, i])
        except ValueError:
            auc = 0.0
        
        per_class[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "specificity": float(specificities[f"class_{i}"]),
            "auc": float(auc),
            "support": int(np.sum(y_true)),
        }
    
    return per_class


def compute_statistical_analysis(
    predictions: np.ndarray,
    targets: np.ndarray,
    probabilities: np.ndarray,
    n_bootstrap: int = 1000,
) -> Dict:
    """
    Compute statistical analysis with confidence intervals.
    
    Args:
        predictions: Model predictions
        targets: Ground truth
        probabilities: Prediction probabilities
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Statistical analysis dictionary
    """
    correct = (predictions == targets).astype(float)
    
    # Bootstrap accuracy
    acc_mean, acc_lower, acc_upper = bootstrap_confidence_interval(
        correct, confidence=0.95, n_bootstrap=n_bootstrap
    )
    
    # Bootstrap per-class
    stats = {
        "accuracy": {
            "mean": acc_mean,
            "ci_lower": acc_lower,
            "ci_upper": acc_upper,
            "std": float(np.std(correct)),
        }
    }
    
    # Per-class F1 bootstrap
    from sklearn.metrics import f1_score
    
    num_classes = probabilities.shape[1]
    for c in range(num_classes):
        binary_correct = (predictions == c) == (targets == c)
        mean, lower, upper = bootstrap_confidence_interval(
            binary_correct.astype(float), n_bootstrap=n_bootstrap
        )
        stats[f"class_{c}_accuracy"] = {
            "mean": mean,
            "ci_lower": lower,
            "ci_upper": upper,
        }
    
    return stats


def run_comprehensive_evaluation(
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
    n_bootstrap: int = 1000,
) -> Dict:
    """
    Run comprehensive classification evaluation.
    
    Args:
        checkpoint_path: Path to model checkpoint
        patient_list_path: Path to patient list
        roi_dir: Directory with ROI files
        labels_file: Path to labels CSV
        output_dir: Output directory
        config_path: Optional config path
        batch_size: Batch size
        num_workers: Number of workers
        use_amp: Use automatic mixed precision
        device: Device to use
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Complete evaluation results dictionary
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
    
    # Get config
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint.get("model_config", {})
    class_names = checkpoint.get("class_names", ["glioma", "meningioma", "pituitary", "no_tumor"])
    
    # Model info
    num_params = sum(p.numel() for p in model.parameters())
    
    # Dataset setup
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
    
    # Load data
    patient_ids = load_patient_list(patient_list_path)
    labels = load_labels(labels_file)
    
    print(f"Evaluating {len(patient_ids)} patients")
    print(f"Classes: {class_names}")
    
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
    predictions, targets, probabilities, patient_ids_eval, inference_time_ms = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        use_amp=use_amp,
    )
    
    # Compute all metrics
    print("\n" + "=" * 60)
    print("COMPUTING COMPREHENSIVE METRICS")
    print("=" * 60)
    
    # Basic metrics
    basic_metrics = compute_classification_metrics(
        pred=predictions,
        target=targets,
        pred_probs=probabilities,
        class_names=class_names,
    )
    
    # Per-class metrics
    per_class_metrics = compute_per_class_metrics(
        predictions, targets, probabilities, class_names
    )
    
    # ROC curves (micro/macro)
    roc_data = compute_micro_macro_roc(targets, probabilities, class_names)
    
    # PR curves
    pr_data = compute_pr_curves(targets, probabilities, class_names)
    
    # Calibration metrics
    ece = multiclass_ece(targets, probabilities)
    brier = multiclass_brier_score(targets, probabilities)
    
    # Confusion matrix
    cm, _ = get_confusion_matrix(predictions, targets, class_names)
    
    # Statistical analysis
    print("Computing bootstrap confidence intervals...")
    statistics = compute_statistical_analysis(
        predictions, targets, probabilities, n_bootstrap=n_bootstrap
    )
    
    # Compile results
    results = {
        "basic_metrics": basic_metrics,
        "per_class_metrics": per_class_metrics,
        "roc_data": roc_data,
        "pr_data": pr_data,
        "calibration": {
            "ece": ece,
            "brier_score": brier,
        },
        "confusion_matrix": cm.tolist(),
        "statistics": statistics,
        "inference_time_ms": inference_time_ms,
        "model_parameters": num_params,
        "test_samples": len(targets),
        "class_names": class_names,
        # For visualization
        "probabilities": probabilities.tolist(),
        "targets": targets.tolist(),
        "predictions": predictions.tolist(),
        "confidences": np.max(probabilities, axis=1).tolist(),
        "correct": (predictions == targets).tolist(),
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"Accuracy: {basic_metrics['accuracy']:.4f}")
    print(f"F1 (Macro): {basic_metrics['f1_macro']:.4f}")
    print(f"AUC-ROC (Macro): {roc_data['summary']['macro_auc']:.4f}")
    print(f"ECE: {ece:.4f}")
    print(f"Brier Score: {brier:.4f}")
    print(f"Inference Time: {inference_time_ms:.2f} ms/sample")
    
    # Classification report
    report = get_classification_report(predictions, targets, class_names)
    print("\nClassification Report:")
    print(report)
    
    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    
    # Save JSON (excluding large arrays for JSON)
    json_results = {k: v for k, v in results.items() 
                    if k not in ["probabilities", "targets", "predictions", "confidences", "correct"]}
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(json_results, f, indent=2)
    
    # Save predictions CSV
    pred_df = pd.DataFrame({
        "patient_id": patient_ids_eval,
        "true_label": targets,
        "predicted_label": predictions,
        "true_class": [class_names[t] for t in targets],
        "predicted_class": [class_names[p] for p in predictions],
        "correct": predictions == targets,
        "confidence": np.max(probabilities, axis=1),
    })
    for i, name in enumerate(class_names):
        pred_df[f"prob_{name}"] = probabilities[:, i]
    pred_df.to_csv(output_dir / "predictions.csv", index=False)
    
    # Save classification report
    with open(output_dir / "classification_report.txt", "w") as f:
        f.write(report)
    
    # Generate visualizations
    print("Generating visualizations...")
    
    plot_roc_curves_publication(roc_data, output_dir / "roc_curves.png")
    plot_pr_curves_publication(pr_data, output_dir / "pr_curves.png")
    plot_confusion_matrix_publication(cm, class_names, output_dir / "confusion_matrix.png")
    plot_confusion_matrix_publication(cm, class_names, output_dir / "confusion_matrix_raw.png", normalize=False)
    plot_reliability_diagram(targets, probabilities, output_path=output_dir / "reliability_diagram.png")
    plot_threshold_analysis(
        np.max(probabilities, axis=1),
        predictions == targets,
        output_dir / "threshold_analysis.png"
    )
    
    # Generate LaTeX tables
    print("Generating LaTeX tables...")
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    
    generate_classification_table(basic_metrics, class_names, tables_dir / "table_classification.tex")
    generate_per_class_table(per_class_metrics, class_names, tables_dir / "table_per_class.tex")
    generate_roc_auc_table(roc_data, tables_dir / "table_roc_auc.tex")
    
    print(f"\nAll results saved to {output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Classification Evaluation for Publication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--patient-list", type=str, required=True, help="Path to patient list file")
    parser.add_argument("--roi-dir", type=str, required=True, help="Directory containing ROI files")
    parser.add_argument("--labels-file", type=str, required=True, help="Path to labels CSV file")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Number of bootstrap samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    run_comprehensive_evaluation(
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
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == "__main__":
    main()
