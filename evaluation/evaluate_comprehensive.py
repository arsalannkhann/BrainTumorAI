"""
Master evaluation script that orchestrates all evaluation tasks.
Runs both classification and segmentation evaluation and generates complete paper artifacts.
"""

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import set_seed


def get_system_info() -> Dict:
    """Get system and environment information for reproducibility."""
    import torch
    
    info = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
    
    try:
        import monai
        info["monai_version"] = monai.__version__
    except ImportError:
        info["monai_version"] = "N/A"
    
    return info


def run_classification_evaluation(
    checkpoint_path: str,
    patient_list_path: str,
    roi_dir: str,
    labels_file: str,
    output_dir: Path,
    config_path: Optional[str] = None,
    device: Optional[str] = None,
    n_bootstrap: int = 1000,
) -> Dict:
    """Run classification evaluation."""
    from evaluation.classification_eval import run_comprehensive_evaluation
    
    cls_output_dir = output_dir / "classification"
    cls_output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("CLASSIFICATION EVALUATION")
    print("=" * 70)
    
    return run_comprehensive_evaluation(
        checkpoint_path=checkpoint_path,
        patient_list_path=patient_list_path,
        roi_dir=roi_dir,
        labels_file=labels_file,
        output_dir=str(cls_output_dir),
        config_path=config_path,
        device=device,
        n_bootstrap=n_bootstrap,
    )


def run_segmentation_evaluation(
    checkpoint_path: str,
    patient_list_path: str,
    processed_dir: str,
    masks_dir: str,
    output_dir: Path,
    config_path: Optional[str] = None,
    device: Optional[str] = None,
    n_bootstrap: int = 1000,
) -> Dict:
    """Run segmentation evaluation."""
    from evaluation.segmentation_eval import run_comprehensive_evaluation
    
    seg_output_dir = output_dir / "segmentation"
    seg_output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("SEGMENTATION EVALUATION")
    print("=" * 70)
    
    return run_comprehensive_evaluation(
        checkpoint_path=checkpoint_path,
        patient_list_path=patient_list_path,
        processed_dir=processed_dir,
        masks_dir=masks_dir,
        output_dir=str(seg_output_dir),
        config_path=config_path,
        device=device,
        n_bootstrap=n_bootstrap,
    )


def generate_results_section(
    cls_results: Optional[Dict],
    seg_results: Optional[Dict],
    output_path: Path,
) -> str:
    """Generate paper-ready results section text."""
    
    lines = [
        "# Results Section (Draft)\n",
        "## Classification Performance\n",
    ]
    
    if cls_results:
        basic = cls_results.get("basic_metrics", {})
        roc = cls_results.get("roc_data", {}).get("summary", {})
        cal = cls_results.get("calibration", {})
        
        lines.extend([
            f"The classification model achieved an overall accuracy of **{basic.get('accuracy', 0)*100:.2f}%** ",
            f"on the test set of {cls_results.get('test_samples', 0)} patients. ",
            f"The macro-averaged F1-score was **{basic.get('f1_macro', 0):.4f}**, ",
            f"with precision of {basic.get('precision_macro', 0):.4f} and recall of {basic.get('recall_macro', 0):.4f}.\n\n",
            
            f"For multi-class discrimination, the model achieved a macro-averaged AUC-ROC of **{roc.get('macro_auc', 0):.4f}** ",
            f"(mean per-class AUC: {roc.get('mean_auc', 0):.4f} ± {roc.get('std_auc', 0):.4f}). ",
            f"The micro-averaged AUC was {roc.get('micro_auc', 0):.4f}.\n\n",
            
            f"Calibration analysis revealed an Expected Calibration Error (ECE) of **{cal.get('ece', 0):.4f}** ",
            f"and a Brier score of **{cal.get('brier_score', 0):.4f}**, indicating well-calibrated probability estimates.\n\n",
            
            f"Average inference time was **{cls_results.get('inference_time_ms', 0):.2f} ms** per sample.\n\n",
        ])
    else:
        lines.append("*Classification evaluation not available.*\n\n")
    
    lines.append("## Segmentation Performance\n")
    
    if seg_results:
        agg = seg_results.get("aggregated_metrics", {})
        
        # Per-class metrics
        for region in ["edema", "enhancing", "necrotic"]:
            if region in agg:
                dice = agg[region].get("dice", {})
                lines.append(
                    f"- **{region.title()}**: Dice = {dice.get('mean', 0):.4f} ± {dice.get('std', 0):.4f} "
                    f"(95% CI: [{dice.get('ci_lower', 0):.4f}, {dice.get('ci_upper', 0):.4f}])\n"
                )
        
        lines.append("\n### BraTS Aggregate Regions\n")
        
        for region in ["whole_tumor", "tumor_core", "enhancing_tumor"]:
            if region in agg:
                dice = agg[region].get("dice", {})
                hd95 = agg[region].get("hd95", {})
                lines.append(
                    f"- **{region.replace('_', ' ').title()}**: Dice = {dice.get('mean', 0):.4f} ± {dice.get('std', 0):.4f}, "
                    f"HD95 = {hd95.get('mean', 0):.2f} ± {hd95.get('std', 0):.2f} mm\n"
                )
        
        if "mean" in agg:
            mean_dice = agg["mean"].get("dice", {})
            lines.append(
                f"\nOverall mean Dice score: **{mean_dice.get('mean', 0):.4f}** ± {mean_dice.get('std', 0):.4f}\n\n"
            )
        
        lines.append(
            f"Average inference time was **{seg_results.get('inference_time_ms', 0):.2f} ms** per volume.\n"
        )
    else:
        lines.append("*Segmentation evaluation not available.*\n")
    
    lines.extend([
        "\n## Limitations\n",
        "- The evaluation was performed on a single dataset split; cross-validation results may vary.\n",
        "- Distance-based metrics (HD95, ASD) may be infinite for cases with no ground truth or prediction.\n",
        "- Further validation on external datasets is recommended.\n",
    ])
    
    text = "".join(lines)
    
    with open(output_path, "w") as f:
        f.write(text)
    
    return text


def generate_reproducibility_details(
    sys_info: Dict,
    cls_results: Optional[Dict],
    seg_results: Optional[Dict],
    output_path: Path,
) -> Dict:
    """Generate reproducibility details."""
    
    details = {
        "random_seed": 42,
        "hardware": sys_info.get("gpu_name", "CPU"),
        "pytorch_version": sys_info.get("pytorch_version", "N/A"),
        "monai_version": sys_info.get("monai_version", "N/A"),
        "cuda_version": sys_info.get("cuda_version", "N/A"),
        "timestamp": sys_info.get("timestamp", "N/A"),
    }
    
    if cls_results:
        details["cls_test_samples"] = cls_results.get("test_samples", 0)
        details["cls_model_parameters"] = cls_results.get("model_parameters", 0)
        details["cls_inference_time_ms"] = cls_results.get("inference_time_ms", 0)
    
    if seg_results:
        details["seg_test_samples"] = seg_results.get("test_samples", 0)
        details["seg_model_parameters"] = seg_results.get("model_parameters", 0)
        details["seg_inference_time_ms"] = seg_results.get("inference_time_ms", 0)
    
    with open(output_path, "w") as f:
        json.dump(details, f, indent=2)
    
    # Also write markdown version
    md_lines = [
        "# Reproducibility Details\n\n",
        "| Parameter | Value |\n",
        "|-----------|-------|\n",
    ]
    
    for key, value in details.items():
        display_key = key.replace("_", " ").title()
        if isinstance(value, int) and value > 1000:
            value = f"{value:,}"
        elif isinstance(value, float):
            value = f"{value:.4f}"
        md_lines.append(f"| {display_key} | {value} |\n")
    
    with open(output_path.with_suffix(".md"), "w") as f:
        f.writelines(md_lines)
    
    return details


def main():
    parser = argparse.ArgumentParser(
        description="Master Evaluation Pipeline for Brain Tumor Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run both classification and segmentation evaluation
  python -m evaluation.evaluate_comprehensive \\
      --mode all \\
      --cls-checkpoint checkpoints/classification/best_model.pt \\
      --seg-checkpoint checkpoints/segmentation/best_model.pt \\
      --data-dir data \\
      --output-dir results/evaluation
      
  # Run classification only
  python -m evaluation.evaluate_comprehensive \\
      --mode classification \\
      --cls-checkpoint checkpoints/classification/best_model.pt \\
      --data-dir data \\
      --output-dir results/evaluation
        """,
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["classification", "segmentation", "all"],
        default="all",
        help="Evaluation mode",
    )
    parser.add_argument("--cls-checkpoint", type=str, help="Classification checkpoint path")
    parser.add_argument("--seg-checkpoint", type=str, help="Segmentation checkpoint path")
    parser.add_argument("--data-dir", type=str, required=True, help="Data directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--cls-config", type=str, help="Classification config path")
    parser.add_argument("--seg-config", type=str, help="Segmentation config path")
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Number of bootstrap samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = Path(args.data_dir)
    
    # Get system info
    sys_info = get_system_info()
    
    print("=" * 70)
    print("BRAIN TUMOR EVALUATION PIPELINE")
    print("=" * 70)
    print(f"Timestamp: {sys_info['timestamp']}")
    print(f"PyTorch: {sys_info['pytorch_version']}")
    if sys_info['cuda_available']:
        print(f"GPU: {sys_info['gpu_name']}")
    print("=" * 70)
    
    cls_results = None
    seg_results = None
    
    # Run classification evaluation
    if args.mode in ["classification", "all"]:
        if args.cls_checkpoint:
            try:
                cls_results = run_classification_evaluation(
                    checkpoint_path=args.cls_checkpoint,
                    patient_list_path=str(data_dir / "splits_cls" / "test.txt"),
                    roi_dir=str(data_dir / "roi"),
                    labels_file=str(data_dir / "labels_cls.csv"),
                    output_dir=output_dir,
                    config_path=args.cls_config,
                    device=args.device,
                    n_bootstrap=args.n_bootstrap,
                )
            except Exception as e:
                print(f"Classification evaluation failed: {e}")
        else:
            print("Skipping classification: no checkpoint provided")
    
    # Run segmentation evaluation
    if args.mode in ["segmentation", "all"]:
        if args.seg_checkpoint:
            try:
                seg_results = run_segmentation_evaluation(
                    checkpoint_path=args.seg_checkpoint,
                    patient_list_path=str(data_dir / "splits" / "test.txt"),
                    processed_dir=str(data_dir / "processed"),
                    masks_dir=str(data_dir / "masks"),
                    output_dir=output_dir,
                    config_path=args.seg_config,
                    device=args.device,
                    n_bootstrap=args.n_bootstrap,
                )
            except Exception as e:
                print(f"Segmentation evaluation failed: {e}")
        else:
            print("Skipping segmentation: no checkpoint provided")
    
    # Generate paper artifacts
    print("\n" + "=" * 70)
    print("GENERATING PAPER ARTIFACTS")
    print("=" * 70)
    
    paper_dir = output_dir / "paper_text"
    paper_dir.mkdir(exist_ok=True)
    
    # Results section
    results_text = generate_results_section(
        cls_results, seg_results,
        paper_dir / "results_section.md"
    )
    print(f"Generated results section: {paper_dir / 'results_section.md'}")
    
    # Reproducibility details
    repro_details = generate_reproducibility_details(
        sys_info, cls_results, seg_results,
        paper_dir / "reproducibility_details.json"
    )
    print(f"Generated reproducibility details: {paper_dir / 'reproducibility_details.json'}")
    
    # Summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"All results saved to: {output_dir}")
    print("\nGenerated artifacts:")
    
    for p in output_dir.rglob("*"):
        if p.is_file():
            print(f"  - {p.relative_to(output_dir)}")


if __name__ == "__main__":
    main()
