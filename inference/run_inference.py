"""
Production Inference CLI Tool

Command-line interface for brain tumor inference:
- Single image inference
- Batch inference from directory
- Report generation
- Mask export

Usage:
    python run_inference.py --image path/to/image.npy --output report.json
    python run_inference.py --batch data/test/ --output results/
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.engine import (
    BrainTumorInferenceEngine,
    create_inference_engine,
    InferenceReport,
)


def save_report(report: InferenceReport, output_path: Path, format: str = "json"):
    """Save inference report to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
    elif format == "text":
        with open(output_path, "w") as f:
            f.write(report.to_text())
    else:
        raise ValueError(f"Unknown format: {format}")


def save_masks(report: InferenceReport, output_dir: Path, image_id: str):
    """Save segmentation masks to files."""
    if report.segmentation is None:
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    seg = report.segmentation
    
    # Save individual masks
    np.save(output_dir / f"{image_id}_edema.npy", seg.edema_mask)
    np.save(output_dir / f"{image_id}_enhancing.npy", seg.enhancing_mask)
    np.save(output_dir / f"{image_id}_necrotic.npy", seg.necrotic_mask)
    np.save(output_dir / f"{image_id}_combined.npy", seg.combined_mask)


def run_single_inference(
    engine: BrainTumorInferenceEngine,
    image_path: Path,
    output_path: Path,
    save_mask: bool = False,
    output_format: str = "json",
) -> InferenceReport:
    """Run inference on a single image."""
    print(f"[CLI] Loading image: {image_path}")
    
    # Load image
    image = np.load(image_path, allow_pickle=False)
    image_id = image_path.stem
    
    print(f"[CLI] Image shape: {image.shape}")
    print(f"[CLI] Running inference...")
    
    # Run inference
    report = engine.run_inference(image, image_id=image_id)
    
    # Save report
    save_report(report, output_path, format=output_format)
    print(f"[CLI] Report saved to: {output_path}")
    
    # Save masks
    if save_mask and report.segmentation is not None:
        mask_dir = output_path.parent / "masks"
        save_masks(report, mask_dir, image_id)
        print(f"[CLI] Masks saved to: {mask_dir}")
    
    return report


def run_batch_inference(
    engine: BrainTumorInferenceEngine,
    input_dir: Path,
    output_dir: Path,
    save_mask: bool = False,
    output_format: str = "json",
    pattern: str = "*.npy",
) -> List[InferenceReport]:
    """Run inference on all images in a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all matching files
    image_files = list(input_dir.glob(pattern))
    
    if not image_files:
        print(f"[CLI] No files matching '{pattern}' found in {input_dir}")
        return []
    
    print(f"[CLI] Found {len(image_files)} images")
    
    reports = []
    summary = {
        "total": len(image_files),
        "successful": 0,
        "failed": 0,
        "requires_review": 0,
        "class_distribution": {},
    }
    
    for i, image_path in enumerate(image_files):
        print(f"\n[CLI] Processing {i+1}/{len(image_files)}: {image_path.name}")
        
        try:
            # Load image
            image = np.load(image_path, allow_pickle=False)
            image_id = image_path.stem
            
            # Run inference
            report = engine.run_inference(image, image_id=image_id)
            reports.append(report)
            
            # Save individual report
            ext = "json" if output_format == "json" else "txt"
            report_path = output_dir / f"{image_id}_report.{ext}"
            save_report(report, report_path, format=output_format)
            
            # Save masks if requested
            if save_mask and report.segmentation is not None:
                mask_dir = output_dir / "masks"
                save_masks(report, mask_dir, image_id)
            
            # Update summary
            summary["successful"] += 1
            if report.validation.requires_manual_review:
                summary["requires_review"] += 1
            
            if report.classification:
                class_name = report.classification.predicted_class.name
                summary["class_distribution"][class_name] = (
                    summary["class_distribution"].get(class_name, 0) + 1
                )
            
            print(f"    Status: {'Ready' if not report.validation.requires_manual_review else 'Needs Review'}")
            
        except Exception as e:
            print(f"    [ERROR] Failed: {e}")
            summary["failed"] += 1
    
    # Save summary report
    summary_path = output_dir / "batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("BATCH INFERENCE SUMMARY")
    print(f"{'='*60}")
    print(f"Total images: {summary['total']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Requires manual review: {summary['requires_review']}")
    print(f"\nClass Distribution:")
    for class_name, count in summary["class_distribution"].items():
        print(f"  {class_name}: {count}")
    print(f"\nSummary saved to: {summary_path}")
    
    return reports


def main():
    parser = argparse.ArgumentParser(
        description="Brain Tumor MRI Inference CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image inference
  python run_inference.py --image data/test/patient_001.npy --output results/report.json
  
  # Batch inference
  python run_inference.py --batch data/test/ --output results/
  
  # With mask export
  python run_inference.py --image data/test/patient_001.npy --output results/ --save-masks
  
  # Text format output
  python run_inference.py --image data/test/patient_001.npy --output report.txt --format text
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image", "-i",
        type=Path,
        help="Path to single image (.npy file)"
    )
    input_group.add_argument(
        "--batch", "-b",
        type=Path,
        help="Directory containing images for batch inference"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output path (file for single, directory for batch)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["json", "text"],
        default="json",
        help="Output format (default: json)"
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Save segmentation masks as .npy files"
    )
    
    # Model options
    parser.add_argument(
        "--cls-checkpoint",
        type=Path,
        default=Path("checkpoints/classification/best_model.pt"),
        help="Classification model checkpoint"
    )
    parser.add_argument(
        "--seg-checkpoint",
        type=Path,
        default=Path("checkpoints/segmentation/best_model.pt"),
        help="Segmentation model checkpoint"
    )
    parser.add_argument(
        "--seg-config",
        type=Path,
        default=Path("configs/seg.yaml"),
        help="Segmentation config file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto)"
    )
    
    # Inference options
    parser.add_argument(
        "--skip-classification",
        action="store_true",
        help="Skip classification inference"
    )
    parser.add_argument(
        "--skip-segmentation",
        action="store_true",
        help="Skip segmentation inference"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.npy",
        help="File pattern for batch mode (default: *.npy)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate paths
    if args.image and not args.image.exists():
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)
    
    if args.batch and not args.batch.is_dir():
        print(f"[ERROR] Directory not found: {args.batch}")
        sys.exit(1)
    
    # Initialize engine
    print("="*60)
    print("Brain Tumor Inference Engine")
    print("="*60)
    
    engine = create_inference_engine(
        cls_checkpoint=str(args.cls_checkpoint),
        seg_checkpoint=str(args.seg_checkpoint),
        seg_config=str(args.seg_config),
        device=args.device,
    )
    
    # Run inference
    if args.image:
        # Single image
        report = run_single_inference(
            engine=engine,
            image_path=args.image,
            output_path=args.output,
            save_mask=args.save_masks,
            output_format=args.format,
        )
        
        # Print report
        print("\n" + report.to_text())
        
    else:
        # Batch
        reports = run_batch_inference(
            engine=engine,
            input_dir=args.batch,
            output_dir=args.output,
            save_mask=args.save_masks,
            output_format=args.format,
            pattern=args.pattern,
        )


if __name__ == "__main__":
    main()
