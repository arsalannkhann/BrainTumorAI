"""
Generate sample visualizations for Classification (GradCAM) and Segmentation.
Selects 20 random samples from the test set.
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from classification.dataset import BrainTumorClassificationDataset
from classification.model import load_classifier_from_checkpoint
from evaluation.explainability import GradCAM, overlay_cam_on_image
from evaluation.segmentation_eval import evaluate_patient, load_model_from_checkpoint, run_inference
from evaluation.visualizations import plot_segmentation_overlay
from utils import load_config, load_labels, load_patient_list, set_seed

def generate_classification_samples(
    checkpoint_path: str,
    patient_list_path: str,
    roi_dir: str,
    labels_file: str,
    output_dir: str,
    num_samples: int = 20,
    device: str = "cuda",
):
    """Generate GradCAM visualizations for classification."""
    print(f"\nGenerating {num_samples} classification GradCAM samples...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(device)
    
    # Load model
    model = load_classifier_from_checkpoint(checkpoint_path, device=str(device))
    model.eval()
    
    # Setup GradCAM
    gradcam = GradCAM(model)
    
    # Load data
    patient_ids = load_patient_list(patient_list_path)
    labels = load_labels(labels_file)
    
    # Select random samples
    if len(patient_ids) > num_samples:
        selected_ids = random.sample(patient_ids, num_samples)
    else:
        selected_ids = patient_ids
        
    dataset = BrainTumorClassificationDataset(
        patient_ids=selected_ids,
        roi_dir=roi_dir,
        labels=labels,
        mode="2.5d",
        num_slices=16,
        is_train=False,
    )
    
    class_names = ["glioma", "meningioma", "pituitary", "no_tumor"]
    
    for i in tqdm(range(len(dataset)), desc="GradCAM"):
        sample = dataset[i]
        img = sample["image"].unsqueeze(0).to(device) # (1, 16, C, H, W)
        
        # Handle label type (int vs tensor)
        if isinstance(sample["label"], torch.Tensor):
            label_idx = sample["label"].item()
        else:
            label_idx = sample["label"]
            
        pid = sample["patient_id"]
        
        # Compute CAM
        try:
            cams, pred_idx, best_slice_idx = gradcam(img)
            
            # Get best slice image
            # Image is (16, 4, H, W) - use T1CE (channel 1) or T2 (channel 2)
            # Typically inputs are normalized. We need to denormalize for viz?
            # Assuming standard normalization, but for heatmap overlay let's just use raw intensity range 0-1
            
            # Extract slice: (C, H, W)
            slice_img = img[0, best_slice_idx].cpu().numpy()
            
            # Use T1CE (index 1) if available, else T1 (0)
            viz_img = slice_img[1] 
            
            # Normalize to 0-1 for visualization
            if viz_img.max() > viz_img.min():
                viz_img = (viz_img - viz_img.min()) / (viz_img.max() - viz_img.min())
            else:
                viz_img = np.zeros_like(viz_img)
            
            # Overlay
            cam_slice = cams[best_slice_idx]
            overlay = overlay_cam_on_image(viz_img, cam_slice, alpha=0.4)
            
            # Plot
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original
            axes[0].imshow(viz_img, cmap="gray")
            axes[0].set_title(f"Patient: {pid}\nSlice: {best_slice_idx}")
            axes[0].axis("off")
            
            # Heatmap
            axes[1].imshow(cam_slice, cmap="jet")
            axes[1].set_title("GradCAM Heatmap")
            axes[1].axis("off")
            
            # Overlay
            axes[2].imshow(overlay)
            gt_name = class_names[label_idx]
            pred_name = class_names[pred_idx]
            color = "green" if label_idx == pred_idx else "red"
            axes[2].set_title(f"Pred: {pred_name}\nGT: {gt_name}", color=color, fontweight="bold")
            axes[2].axis("off")
            
            plt.tight_layout()
            plt.savefig(output_dir / f"{pid}_gradcam.png", dpi=150)
            plt.close()
            
        except Exception as e:
            print(f"Failed GradCAM for {pid}: {e}")
            continue

def generate_segmentation_samples(
    checkpoint_path: str,
    patient_list_path: str,
    processed_dir: str,
    masks_dir: str,
    output_dir: str,
    num_samples: int = 20,
    device: str = "cuda",
):
    """Generate segmentation overlays."""
    print(f"\nGenerating {num_samples} segmentation samples...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(device)
    
    model = load_model_from_checkpoint(checkpoint_path, device=str(device))
    
    patient_ids = load_patient_list(patient_list_path)
    
    if len(patient_ids) > num_samples:
        selected_ids = random.sample(patient_ids, num_samples)
    else:
        selected_ids = patient_ids
        
    class_names = ["background", "edema", "enhancing", "necrotic"]
    
    for pid in tqdm(selected_ids, desc="Segmentation"):
        try:
            # Use existing evaluation function but just for viz
            _, prediction, ground_truth, image, _ = evaluate_patient(
                model=model,
                patient_id=pid,
                processed_dir=Path(processed_dir),
                masks_dir=Path(masks_dir),
                device=device,
            )
            
            # Determine best slice (most tumor area)
            tumor_mask = ground_truth > 0
            if tumor_mask.sum() == 0:
                # If no tumor in GT, use prediction
                tumor_mask = prediction > 0
            
            if tumor_mask.sum() > 0:
                slice_sums = tumor_mask.sum(axis=(0, 1)) # Sum across H, W
                best_slice = np.argmax(slice_sums)
            else:
                best_slice = image.shape[-1] // 2
            
            plot_segmentation_overlay(
                image=image,
                prediction=prediction,
                ground_truth=ground_truth,
                output_path=output_dir / f"{pid}_seg.png",
                class_names=class_names,
                slice_idx=best_slice,
                title=f"Sample: {pid} (Slice {best_slice})",
            )
            
        except Exception as e:
            print(f"Failed segmentation viz for {pid}: {e}")
            continue

def main():
    parser = argparse.ArgumentParser(description="Generate sample visualizations")
    parser.add_argument("--cls-checkpoint", type=str, help="Classification checkpoint")
    parser.add_argument("--seg-checkpoint", type=str, help="Segmentation checkpoint")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)
    
    if args.cls_checkpoint:
        generate_classification_samples(
            checkpoint_path=args.cls_checkpoint,
            patient_list_path=str(data_dir / "splits_cls" / "test.txt"),
            roi_dir=str(data_dir / "roi"),
            labels_file=str(data_dir / "labels_cls.csv"),
            output_dir=output_dir / "visualizations" / "gradcam",
            num_samples=args.num_samples,
        )
        
    if args.seg_checkpoint:
        generate_segmentation_samples(
            checkpoint_path=args.seg_checkpoint,
            patient_list_path=str(data_dir / "splits" / "test.txt"),
            processed_dir=str(data_dir / "processed"),
            masks_dir=str(data_dir / "masks"),
            output_dir=output_dir / "visualizations" / "segmentation",
            num_samples=args.num_samples,
        )

if __name__ == "__main__":
    main()
