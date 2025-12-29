"""
Interactive Inference Demo with Visualization

This demo showcases the production inference engine with:
- Classification visualization
- Segmentation overlay visualization
- Validation status display
- Report generation
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from inference import (
    create_inference_engine,
    BrainTumorInferenceEngine,
    TumorClass,
)


def visualize_classification(engine: BrainTumorInferenceEngine, ax_list: list, num_samples: int = 4):
    """Visualize classification results."""
    labels_file = Path("data/labels_cls.csv")
    if not labels_file.exists():
        print("[Demo] Classification labels file not found")
        return
    
    labels_df = pd.read_csv(labels_file)
    class_names = {0: "glioma", 1: "meningioma", 2: "pituitary", 3: "notumor"}
    
    # Pick random samples
    samples = labels_df.sample(min(num_samples, len(labels_df)))
    
    # Check if we need to use fallback samples
    roi_dir = Path("data/roi")
    use_fallback = not roi_dir.exists() or not any(roi_dir.glob("*.npy"))
    if use_fallback:
        print("[Demo] Using fallback classification samples from data/samples/roi")
        roi_dir = Path("data/samples/roi")
    
    for i, (_, row) in enumerate(samples.iterrows()):
        if i >= len(ax_list):
            break
            
        patient_id = row["patient_id"]
        true_label = row["label"]
        
        # Load ROI
        roi_path = roi_dir / f"{patient_id}_roi.npy"
        if not roi_path.exists():
            continue
        
        img = np.load(roi_path)
        
        # Handle shape
        if img.ndim == 4:
            img = img[..., 0]  # (C, H, W)
        
        # Run classification
        result = engine.classify(img)
        
        # Prepare display image
        display_img = img.transpose(1, 2, 0)  # (H, W, C)
        if display_img.max() > 1.0:
            display_img = display_img / 255.0
        display_img = np.clip(display_img, 0, 1)
        
        # Plot
        ax = ax_list[i]
        ax.imshow(display_img)
        
        pred_label = result.predicted_class.value
        correct = true_label == pred_label
        
        title = f"True: {class_names[true_label]}\nPred: {result.predicted_class.display_name}"
        title += f"\nConf: {result.confidence_score:.2%}"
        if result.is_low_confidence:
            title += " ⚠️"
        
        ax.set_title(title, color="green" if correct else "red", fontsize=9)
        ax.axis("off")


def visualize_segmentation(engine: BrainTumorInferenceEngine, ax_list: list, num_samples: int = 4):
    """Visualize segmentation results."""
    processed_dir = Path("data/processed")
    
    try:
        import nibabel as nib
    except ImportError:
        print("[Demo] nibabel not installed, skipping segmentation demo")
        return
    
    # Find processed files
    files = list(processed_dir.glob("*/BraTS20_*.npy"))
    
    # Check if we need to use fallback samples
    if not files:
        processed_dir = Path("data/samples/processed")
        files = list(processed_dir.glob("*.npy"))
        if files:
            print("[Demo] Using fallback segmentation samples from data/samples/processed")
    
    if not files:
        print("[Demo] No processed data found for segmentation")
        return
    
    samples = random.sample(files, min(num_samples, len(files)))
    
    for i, mri_path in enumerate(samples):
        if i >= len(ax_list):
            break
        
        patient_id = mri_path.stem
        mask_path = Path("data/masks") / patient_id / "seg.nii.gz"
        
        try:
            mri = np.load(mri_path)  # (C, H, W, D)
            
            if mask_path.exists():
                mask_img = nib.load(mask_path)
                gt_mask = mask_img.get_fdata()
            else:
                # Check fallback masks
                fallback_mask = Path("data/samples/masks") / f"{patient_id}_seg.nii.gz"
                if fallback_mask.exists():
                    mask_img = nib.load(fallback_mask)
                    gt_mask = mask_img.get_fdata()
                else:
                    gt_mask = None
                
        except Exception as e:
            print(f"[Demo] Error loading {patient_id}: {e}")
            continue
        
        # Run segmentation
        result = engine.segment(mri)
        pred_mask = result.combined_mask
        
        # Find a good slice to visualize (with tumor)
        tumor_slices = np.where(pred_mask.max(axis=(0, 1)) > 0)[0]
        if len(tumor_slices) > 0:
            z = tumor_slices[len(tumor_slices) // 2]
        else:
            z = pred_mask.shape[2] // 2
        
        # Extract visualization slice
        # Crop the MRI to match the segmentation patch size
        patch_size = 128
        C, H, W, D = mri.shape
        pad_h = max(0, patch_size - H)
        pad_w = max(0, patch_size - W)
        pad_d = max(0, patch_size - D)
        
        if pad_h or pad_w or pad_d:
            mri = np.pad(mri, ((0, 0), (0, pad_h), (0, pad_w), (0, pad_d)))
        
        _, H, W, D = mri.shape
        s_h = (H - patch_size) // 2
        s_w = (W - patch_size) // 2
        s_d = (D - patch_size) // 2
        
        img_slice = mri[0, s_h:s_h+patch_size, s_w:s_w+patch_size, s_d+z]  # T1 channel
        pred_slice = pred_mask[:, :, z]
        
        # Plot
        ax = ax_list[i]
        ax.imshow(img_slice, cmap="gray")
        
        # Create overlay
        overlay = np.zeros((patch_size, patch_size, 4))
        overlay[pred_slice == 1] = [1, 0, 0, 0.5]    # NCR: Red
        overlay[pred_slice == 2] = [0, 1, 0, 0.5]    # ED: Green
        overlay[pred_slice == 3] = [0, 0, 1, 0.5]    # ET: Blue
        
        ax.imshow(overlay)
        
        title = f"{patient_id[:20]}"
        title += f"\nTumor: {result.tumor_area_percentage:.2f}%"
        ax.set_title(title, fontsize=9)
        ax.axis("off")


def create_legend(ax):
    """Create legend for segmentation colors."""
    patches = [
        mpatches.Patch(color='red', alpha=0.5, label='Necrotic Core'),
        mpatches.Patch(color='green', alpha=0.5, label='Edema'),
        mpatches.Patch(color='blue', alpha=0.5, label='Enhancing Tumor'),
    ]
    ax.legend(handles=patches, loc='lower right', fontsize=8)


def main():
    """Run interactive demo."""
    print("="*60)
    print("BRAIN TUMOR INFERENCE DEMO")
    print("Production Inference Engine Test")
    print("="*60)
    
    # Initialize engine
    engine = create_inference_engine()
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Brain Tumor Inference Demo", fontsize=16, fontweight='bold')
    
    # Classification row
    cls_axes = []
    for i in range(4):
        ax = fig.add_subplot(2, 4, i + 1)
        cls_axes.append(ax)
        if i == 0:
            ax.text(-0.2, 0.5, "Classification", transform=ax.transAxes,
                    rotation=90, verticalalignment='center', fontsize=12, fontweight='bold')
    
    # Segmentation row  
    seg_axes = []
    for i in range(4):
        ax = fig.add_subplot(2, 4, i + 5)
        seg_axes.append(ax)
        if i == 0:
            ax.text(-0.2, 0.5, "Segmentation", transform=ax.transAxes,
                    rotation=90, verticalalignment='center', fontsize=12, fontweight='bold')
    
    # Run demos
    if engine.cls_model is not None:
        print("\n[Demo] Running classification inference...")
        visualize_classification(engine, cls_axes)
    else:
        for ax in cls_axes:
            ax.text(0.5, 0.5, "Classification\nModel Not Loaded", 
                    ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    
    if engine.seg_model is not None:
        print("[Demo] Running segmentation inference...")
        visualize_segmentation(engine, seg_axes)
        create_legend(seg_axes[-1])
    else:
        for ax in seg_axes:
            ax.text(0.5, 0.5, "Segmentation\nModel Not Loaded",
                    ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    
    plt.tight_layout()
    
    # Save
    output_path = "inference_demo_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[Demo] Results saved to: {output_path}")
    
    # Print sample report
    print("\n" + "="*60)
    print("SAMPLE INFERENCE REPORT (Classification)")
    print("="*60)
    
    # Run classification report on ROI data (correct data source)
    roi_dir = Path("data/roi")
    roi_files = list(roi_dir.glob("*_roi.npy"))
    if roi_files and engine.cls_model:
        sample_roi = np.load(roi_files[0])
        if sample_roi.ndim == 4:
            sample_roi = sample_roi[..., 0]
        report = engine.run_inference(
            sample_roi, 
            image_id=roi_files[0].stem,
            run_classification=True,
            run_segmentation=False  # Don't run segmentation on 2D classification data
        )
        print(report.to_text())
    
    print("\n" + "="*60)
    print("SAMPLE INFERENCE REPORT (Segmentation)")  
    print("="*60)
    
    # Run segmentation report on BraTS data (correct data source)
    processed_dir = Path("data/processed")
    sample_files = list(processed_dir.glob("*/BraTS20_Training_*.npy"))  # Training data has masks
    if sample_files and engine.seg_model:
        sample = np.load(sample_files[0])
        report = engine.run_inference(
            sample, 
            image_id=sample_files[0].stem,
            run_classification=False,  # Don't run classification on BraTS data
            run_segmentation=True
        )
        print(report.to_text())
    
    plt.close()
    print("\n[Demo] Complete!")


if __name__ == "__main__":
    main()
