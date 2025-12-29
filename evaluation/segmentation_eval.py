"""
Comprehensive segmentation evaluation for publication.
Generates BraTS-style metrics, visualizations, and tables for peer-reviewed papers.
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.inferers import sliding_window_inference
from torch.amp import autocast
from tqdm import tqdm

# Suppress noisy nibabel warnings
warnings.filterwarnings("ignore", category=UserWarning, module="nibabel")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.advanced_metrics import (
    bootstrap_confidence_interval,
    compute_average_surface_distance,
    compute_brats_metrics,
    compute_comprehensive_segmentation_metrics,
    compute_hausdorff_distance_95,
    compute_volume_metrics,
)
from evaluation.visualizations import (
    plot_dice_boxplot,
    plot_segmentation_overlay,
)
from evaluation.latex_tables import generate_segmentation_table
from segmentation.model import load_model_from_checkpoint
from utils import load_config, load_patient_list, set_seed
from utils.metrics import compute_dice, compute_iou


def load_nifti(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load NIfTI file and return data with affine."""
    nii = nib.load(str(path))
    # Using ascontiguousarray to fix common writeable array warnings
    return np.ascontiguousarray(nii.get_fdata()), nii.affine


def get_spacing_from_affine(affine: np.ndarray) -> Tuple[float, float, float]:
    """Extract voxel spacing from affine matrix."""
    return tuple(np.abs(np.diag(affine)[:3]).tolist())


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    roi_size: Tuple[int, int, int] = (128, 128, 128),
    sw_batch_size: int = 4,
    overlap: float = 0.5,
    use_amp: bool = True,
) -> np.ndarray:
    """
    Run inference with sliding window.
    
    Args:
        model: Segmentation model
        image: Input tensor (1, C, H, W, D)
        device: Device
        roi_size: ROI size for sliding window
        sw_batch_size: Batch size for sliding window
        overlap: Overlap between windows
        use_amp: Use automatic mixed precision
        
    Returns:
        Predicted segmentation mask
    """
    model.eval()
    image = image.to(device)
    
    with autocast("cuda", enabled=use_amp):
        outputs = sliding_window_inference(
            inputs=image,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
            mode="gaussian",
        )
    
    pred = torch.argmax(outputs, dim=1).cpu().numpy()[0]
    return pred


def evaluate_patient(
    model: torch.nn.Module,
    patient_id: str,
    processed_dir: Path,
    masks_dir: Path,
    device: torch.device,
    roi_size: Tuple[int, int, int] = (128, 128, 128),
    sw_batch_size: int = 4,
    overlap: float = 0.5,
    use_amp: bool = True,
    class_names: List[str] = ["background", "edema", "enhancing", "necrotic"],
) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Evaluate a single patient.
    
    Args:
        model: Segmentation model
        patient_id: Patient ID
        processed_dir: Directory with preprocessed volumes
        masks_dir: Directory with ground truth masks
        device: Device
        roi_size: ROI size for sliding window
        sw_batch_size: Sliding window batch size
        overlap: Sliding window overlap
        use_amp: Use automatic mixed precision
        class_names: Class names
        
    Returns:
        Tuple of (metrics, prediction, ground_truth, image, inference_time_ms)
    """
    # Load preprocessed volume
    patient_dir = processed_dir / patient_id
    modalities = []
    
    # Determine search directory and patterns
    norm_dir = patient_dir / "normalized"
    if norm_dir.exists():
        search_dir = norm_dir
        # Files are likely t1.nii, etc.
        patterns = ["{mod}.nii", "{mod}.nii.gz", "{patient_id}_{mod}.nii", "{patient_id}_{mod}.nii.gz"]
    else:
        search_dir = patient_dir
        patterns = ["{patient_id}_{mod}.nii.gz", "{patient_id}_{mod}.nii"]
        
    # Standardize modality names
    mod_names = ["t1", "t1ce", "t2", "flair"]
    
    affine = None
    
    for mod in mod_names:
        found = False
        # Try lowercase and uppercase variations
        for m in [mod, mod.upper()]:
            for pat in patterns:
                fname = pat.format(mod=m, patient_id=patient_id)
                mod_path = search_dir / fname
                if mod_path.exists():
                    data, a = load_nifti(mod_path)
                    modalities.append(data)
                    affine = a
                    found = True
                    break
            if found:
                break
    
    if len(modalities) != 4:
        raise FileNotFoundError(f"No modalities found for {patient_id}")
    
    # Stack modalities
    image = np.stack(modalities, axis=0)
    image_tensor = torch.from_numpy(image).float().unsqueeze(0)  # (1, C, H, W, D)
    
    # Get spacing
    spacing = get_spacing_from_affine(affine) if affine is not None else (1.0, 1.0, 1.0)
    
    # Run inference with timing
    start = time.perf_counter()
    prediction = run_inference(
        model=model,
        image=image_tensor,
        device=device,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        use_amp=use_amp,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()
    inference_time_ms = (end - start) * 1000

    # Load ground truth mask
    mask_patterns = [
        masks_dir / patient_id / "seg.nii.gz",
        masks_dir / patient_id / "seg.nii",
        masks_dir / f"{patient_id}_seg.nii.gz",
        masks_dir / f"{patient_id}.nii.gz",
    ]
    
    mask_path = None
    for p in mask_patterns:
        if p.exists():
            mask_path = p
            break
            
    if mask_path is None:
        # If no mask found, return empty metrics but valid prediction/image
        print(f"Warning: No mask found for {patient_id}")
        return {}, prediction, np.zeros_like(prediction), image, inference_time_ms

    ground_truth, _ = load_nifti(mask_path)
    ground_truth = ground_truth.astype(np.int64)
    
    # Compute metrics
    metrics = compute_comprehensive_segmentation_metrics(
        pred=prediction,
        target=ground_truth,
        class_names=class_names,
        spacing=spacing,
    )
    
    return metrics, prediction, ground_truth, image, inference_time_ms


def aggregate_metrics(
    all_metrics: List[Dict],
    n_bootstrap: int = 1000,
) -> Dict:
    """
    Aggregate metrics across patients with statistical analysis.
    
    Args:
        all_metrics: List of per-patient metrics
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Aggregated metrics with mean, std, and 95% CI
    """
    # Collect per-class dice scores
    regions = set()
    for m in all_metrics:
        regions.update(m.keys())
    
    aggregated = {}
    
    for region in regions:
        region_values = {}
        
        for metric_name in ["dice", "iou", "hd95", "asd", "sensitivity", "specificity", "precision"]:
            values = []
            for m in all_metrics:
                if region in m and metric_name in m[region]:
                    val = m[region][metric_name]
                    if val is not None and val != float("inf"):
                        values.append(val)
            
            if values:
                values = np.array(values)
                mean, lower, upper = bootstrap_confidence_interval(values, n_bootstrap=n_bootstrap)
                region_values[metric_name] = {
                    "mean": float(mean),
                    "std": float(np.std(values)),
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "n": len(values),
                }
        
        if region_values:
            aggregated[region] = region_values
    
    return aggregated


def run_comprehensive_evaluation(
    checkpoint_path: str,
    patient_list_path: str,
    processed_dir: str,
    masks_dir: str,
    output_dir: str,
    config_path: Optional[str] = None,
    roi_size: Tuple[int, int, int] = (128, 128, 128),
    sw_batch_size: int = 4,
    overlap: float = 0.5,
    use_amp: bool = True,
    device: Optional[str] = None,
    n_bootstrap: int = 1000,
    save_overlays: bool = True,
    max_overlays: int = 10,
) -> Dict:
    """
    Run comprehensive segmentation evaluation.
    
    Args:
        checkpoint_path: Path to model checkpoint
        patient_list_path: Path to patient list
        processed_dir: Directory with preprocessed volumes
        masks_dir: Directory with ground truth masks
        output_dir: Output directory
        config_path: Optional config path
        roi_size: ROI size for sliding window
        sw_batch_size: Sliding window batch size
        overlap: Sliding window overlap
        use_amp: Use automatic mixed precision
        device: Device to use
        n_bootstrap: Number of bootstrap samples
        save_overlays: Whether to save overlay visualizations
        max_overlays: Maximum number of overlays to save
        
    Returns:
        Complete evaluation results dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processed_dir = Path(processed_dir)
    masks_dir = Path(masks_dir)
    
    # Device setup
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {checkpoint_path}")
    model = load_model_from_checkpoint(checkpoint_path, device=str(device))
    
    # Model info
    num_params = sum(p.numel() for p in model.parameters())
    
    # Load patient list
    patient_ids = load_patient_list(patient_list_path)
    print(f"Evaluating {len(patient_ids)} patients")
    
    # Evaluate each patient
    all_metrics = []
    all_per_patient = []
    total_inference_time = 0
    
    overlay_dir = output_dir / "overlays"
    if save_overlays:
        overlay_dir.mkdir(exist_ok=True)
    
    for idx, patient_id in enumerate(tqdm(patient_ids, desc="Evaluating patients")):
        try:
            metrics, prediction, ground_truth, image, inference_time = evaluate_patient(
                model=model,
                patient_id=patient_id,
                processed_dir=processed_dir,
                masks_dir=masks_dir,
                device=device,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                overlap=overlap,
                use_amp=use_amp,
                class_names=["background", "edema", "enhancing", "necrotic"],
            )
            
            # If metrics empty (no mask), skip aggregation but keep stats?
            if not metrics:
                continue

            all_metrics.append(metrics)
            total_inference_time += inference_time
            
            # Per-patient record
            patient_record = {"patient_id": patient_id, "inference_time_ms": inference_time}
            for region, region_metrics in metrics.items():
                if isinstance(region_metrics, dict):
                    for metric_name, value in region_metrics.items():
                        patient_record[f"{region}_{metric_name}"] = value
            all_per_patient.append(patient_record)
            
            # Save overlay for first few patients
            if save_overlays and idx < max_overlays:
                plot_segmentation_overlay(
                    image=image,
                    prediction=prediction,
                    ground_truth=ground_truth,
                    output_path=overlay_dir / f"{patient_id}_overlay.png",
                    class_names=["background", "edema", "enhancing", "necrotic"],
                    title=f"Segmentation: {patient_id}",
                )
        
        except Exception as e:
            print(f"Error evaluating {patient_id}: {e}")
            continue
    
    if not all_metrics:
        print("No patients were successfully evaluated with ground truth.")
        return {}
    
    # Aggregate metrics
    print("\nAggregating metrics with bootstrap confidence intervals...")
    aggregated_metrics = aggregate_metrics(all_metrics, n_bootstrap=n_bootstrap)
    
    avg_inference_time = total_inference_time / len(all_metrics) if all_metrics else 0
    
    # Compile results
    results = {
        "aggregated_metrics": aggregated_metrics,
        "inference_time_ms": avg_inference_time,
        "model_parameters": num_params,
        "test_samples": len(all_metrics),
        "class_names": ["background", "edema", "enhancing", "necrotic"],
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("SEGMENTATION EVALUATION RESULTS")
    print("=" * 60)
    
    # Print per-region metrics
    for region in ["edema", "enhancing", "necrotic", "whole_tumor", "tumor_core", "enhancing_tumor", "mean"]:
        if region in aggregated_metrics:
            m = aggregated_metrics[region]
            dice_info = m.get("dice", {})
            print(f"\n{region.replace('_', ' ').title()}:")
            if dice_info:
                print(f"  Dice: {dice_info.get('mean', 0):.4f} ± {dice_info.get('std', 0):.4f} "
                      f"[{dice_info.get('ci_lower', 0):.4f}, {dice_info.get('ci_upper', 0):.4f}]")
    
    print(f"\nAverage Inference Time: {avg_inference_time:.2f} ms/volume")
    
    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    
    # Save JSON
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save per-patient CSV
    per_patient_df = pd.DataFrame(all_per_patient)
    per_patient_df.to_csv(output_dir / "per_patient_metrics.csv", index=False)
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # Dice boxplot
    dice_scores = {}
    for region in ["edema", "enhancing", "necrotic"]:
        dice_scores[region] = [
            m[region]["dice"] for m in all_metrics 
            if region in m and "dice" in m[region]
        ]
    
    plot_dice_boxplot(dice_scores, output_dir / "dice_boxplot.png")
    
    # BraTS regions boxplot
    brats_dice = {}
    for region in ["whole_tumor", "tumor_core", "enhancing_tumor"]:
        brats_dice[region] = [
            m[region]["dice"] for m in all_metrics 
            if region in m and "dice" in m[region]
        ]
    
    plot_dice_boxplot(brats_dice, output_dir / "brats_dice_boxplot.png", title="BraTS Region Dice Scores")
    
    # Generate LaTeX table
    print("Generating LaTeX tables...")
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    
    # Flatten metrics for table
    table_metrics = {}
    for region, metrics in aggregated_metrics.items():
        table_metrics[region] = {
            metric: data.get("mean", 0) 
            for metric, data in metrics.items()
        }
    
    generate_segmentation_table(table_metrics, tables_dir / "table_segmentation.tex")
    
    print(f"\nAll results saved to {output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Segmentation Evaluation for Publication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--patient-list", type=str, required=True, help="Path to patient list file")
    parser.add_argument("--processed-dir", type=str, required=True, help="Directory with preprocessed volumes")
    parser.add_argument("--masks-dir", type=str, required=True, help="Directory with ground truth masks")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--roi-size", type=int, nargs=3, default=[128, 128, 128], help="ROI size")
    parser.add_argument("--sw-batch-size", type=int, default=4, help="Sliding window batch size")
    parser.add_argument("--overlap", type=float, default=0.5, help="Sliding window overlap")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Number of bootstrap samples")
    parser.add_argument("--no-overlays", action="store_true", help="Skip saving overlay visualizations")
    parser.add_argument("--max-overlays", type=int, default=10, help="Maximum overlays to save")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    run_comprehensive_evaluation(
        checkpoint_path=args.checkpoint,
        patient_list_path=args.patient_list,
        processed_dir=args.processed_dir,
        masks_dir=args.masks_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        roi_size=tuple(args.roi_size),
        sw_batch_size=args.sw_batch_size,
        overlap=args.overlap,
        use_amp=not args.no_amp,
        device=args.device,
        n_bootstrap=args.n_bootstrap,
        save_overlays=not args.no_overlays,
        max_overlays=args.max_overlays,
    )


if __name__ == "__main__":
    main()
