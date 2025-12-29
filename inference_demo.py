
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from pathlib import Path
import random
import yaml
from segmentation.model import SegmentationModel
from classification.model import load_classifier_from_checkpoint
from utils import load_config

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_segmentation_model(checkpoint_path, config_path):
    config = load_config(config_path)
    model_config = config["model"]
    model = SegmentationModel(**model_config)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    return model

def visualize_inference():
    # Paths
    cls_checkpoint = "checkpoints/classification/best_model.pt"
    seg_checkpoint = "checkpoints/segmentation/best_model.pt"
    seg_config = "configs/seg.yaml"
    
    # Check if models exist
    if not os.path.exists(cls_checkpoint) or not os.path.exists(seg_checkpoint):
        print("Model checkpoints not found!")
        return

    # Load Models
    print("Loading classification model...")
    # Load using helper, assuming config in checkpoint or default
    # Note: Using default architecture config if not in checkpoint might fail if architecture differs
    # But we used standard training script which saves config.
    try:
        cls_model = load_classifier_from_checkpoint(cls_checkpoint, device=str(device))
    except Exception as e:
        print(f"Error loading classification model: {e}")
        cls_model = None

    print("Loading segmentation model...")
    try:
        seg_model = load_segmentation_model(seg_checkpoint, seg_config)
    except Exception as e:
        print(f"Error loading segmentation model: {e}")
        seg_model = None

    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    plt.suptitle("Model Inference Demo", fontsize=16)
    
    # --- Classification Demo ---
    if cls_model:
        # Load Labels
        labels_df = pd.read_csv("data/labels_cls.csv")
        class_names = {0: "glioma", 1: "meningioma", 2: "pituitary", 3: "notumor"}
        
        # Pick 4 random samples
        samples = labels_df.sample(4)
        
        for i, (_, row) in enumerate(samples.iterrows()):
            patient_id = row["patient_id"]
            true_label = row["label"]
            
            # Load ROI
            roi_path = f"data/roi/{patient_id}_roi.npy"
            if not os.path.exists(roi_path):
                continue
                
            img_tensor = np.load(roi_path) # (C, H, W, 1) or (C, H, W)
            
            # Preprocess
            # Model expects (B, C, H, W) or (B, num_slices, C, H, W) depending on mode
            # Current cls config has mode='2.5d', num_slices=1
            # Input shape on disk: (3, H, W, 1) or similar. 
            # Let's inspect shape
            if img_tensor.ndim == 4: # (C, H, W, D)
               # Take 1st slice? or D dim is 1
               img_slice = img_tensor[..., 0] # (C, H, W)
            else:
               img_slice = img_tensor # (C, H, W)
            
            # Normalize 0-1 for display
            display_img = img_slice.transpose(1, 2, 0) # H, W, C
            if display_img.max() > 1.0:
                display_img = display_img / 255.0
            
            # Inference
            # 2.5d mode expects (B, num_slices, C, H, W)
            input_tensor = torch.from_numpy(img_slice).float().to(device)
            if input_tensor.max() > 1.0:
                 input_tensor = input_tensor / 255.0
            
            # Add dims: (C, H, W) -> (1, 1, C, H, W)
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
            
            with torch.no_grad():
                logits = cls_model(input_tensor)
                pred_label = torch.argmax(logits, dim=1).item()
            
            # Plot
            ax = axes[0, i]
            ax.imshow(display_img)
            ax.set_title(f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}", 
                         color="green" if true_label == pred_label else "red")
            ax.axis("off")
            if i == 0:
                ax.text(-0.2, 0.5, "Classification", transform=ax.transAxes, 
                        rotation=90, verticalalignment='center', fontsize=12, weight='bold')

    # --- Segmentation Demo ---
    if seg_model:
        # Load processed data
        # Load processed data
        processed_dir = Path("data/processed")
        # Find all .npy files that match the patient ID pattern (avoiding nested folder noise if any)
        # Assuming pattern BraTS20_Training_XXX/BraTS20_Training_XXX.npy
        files = list(processed_dir.glob("*/BraTS20_*.npy"))
        
        if files:
            # Pick 4 random samples
            samples = random.sample(files, min(4, len(files)))
            
            for i, mri_path in enumerate(samples):
                patient_id = mri_path.stem # BraTS20_Training_XXX
                
                # Mask path: data/masks/patient_id/seg.nii.gz
                mask_path = Path("data/masks") / patient_id / "seg.nii.gz"
                
                try:
                    mri = np.load(mri_path) # (C, H, W, D)
                    
                    # Load mask using nibabel (it's .nii.gz)
                    import nibabel as nib
                    mask_img = nib.load(mask_path)
                    mask = mask_img.get_fdata() # (H, W, D)
                except Exception as e:
                    print(f"Error loading data for {patient_id}: {e}")
                    continue
                
                # Pick a slice with some tumor if possible
                tumor_indices = np.where(mask > 0)[2]
                if len(tumor_indices) > 0:
                    z = tumor_indices[len(tumor_indices)//2]
                else:
                    z = mri.shape[3] // 2
                    
                # Input to model (whole volume or sliding window? Demo uses simple crop or resize?)
                # Model expects (B, C, H, W, D). We can pass a crop.
                # Let's take a crop around the center or just resize?
                # Or simply run inference on a patch?
                # Sliding window is best but slow for demo.
                # Let's simple create a patch of 128x128x128 centered
                
                D, H, W = mri.shape[3], mri.shape[1], mri.shape[2]
                # Center crop 128
                patch_size = 128
                cx, cy, cz = H//2, W//2, D//2
                
                # Pad if needed
                pad_h = max(0, patch_size - H)
                pad_w = max(0, patch_size - W)
                pad_d = max(0, patch_size - D)
                
                if pad_h or pad_w or pad_d:
                    mri = np.pad(mri, ((0,0), (0,pad_h), (0,pad_w), (0,pad_d)))
                
                # Crop
                # Re-calculate center after padding? No, just crop 0:128 if small
                s_h = 0 if H <= patch_size else (H - patch_size)//2
                s_w = 0 if W <= patch_size else (W - patch_size)//2
                s_d = 0 if D <= patch_size else (D - patch_size)//2
                
                image_patch = mri[:, s_h:s_h+patch_size, s_w:s_w+patch_size, s_d:s_d+patch_size]
                
                # Prepare input
                input_tensor = torch.from_numpy(image_patch).float().unsqueeze(0).to(device) # (1, C, H, W, D)
                
                with torch.no_grad():
                    # Direct inference on patch
                    output = seg_model(input_tensor)
                    pred_mask = torch.argmax(output, dim=1).cpu().numpy()[0] # (H, W, D)
                    
                print(f"Sample {i}: Pred Mask Unique Values: {np.unique(pred_mask)}")
                
                # Visualization Slice (middle of the patch)
                # Try to pick a slice where the tumor is present in the ground truth MASK
                tumor_indices_crop = np.where(mask[s_h:s_h+patch_size, s_w:s_w+patch_size, s_d:s_d+patch_size] > 0)[2]
                if len(tumor_indices_crop) > 0:
                     viz_z = tumor_indices_crop[len(tumor_indices_crop)//2]
                else:
                     viz_z = patch_size // 2
                
                # Show T1ce (channel 3)
                img_slice = image_patch[3, :, :, viz_z]
                
                # Use ground truth mask crop for comparison (optional, but good for debug)
                # mask_slice = mask[s_h:s_h+patch_size, s_w:s_w+patch_size, s_d:s_d+patch_size][:, :, viz_z]
                
                pred_slice = pred_mask[:, :, viz_z]
                
                ax = axes[1, i]
                ax.imshow(img_slice, cmap="gray")
                
                # Overlay prediction
                # Create RGBA overlay
                overlay = np.zeros((patch_size, patch_size, 4))
                # Class 1 (NCR): Red, 2 (ED): Green, 3 (ET): Blue
                overlay[pred_slice==1] = [1, 0, 0, 0.4]
                overlay[pred_slice==2] = [0, 1, 0, 0.4]
                overlay[pred_slice==3] = [0, 0, 1, 0.4]
                
                ax.imshow(overlay)
                ax.set_title(f"Seg Pred Slice (Pat: {patient_id})")
                ax.axis("off")
                
                if i == 0:
                   ax.text(-0.2, 0.5, "Segmentation", transform=ax.transAxes, 
                           rotation=90, verticalalignment='center', fontsize=12, weight='bold')

    plt.tight_layout()
    plt.savefig("inference_results.png")
    print("\nInference complete! Results saved to 'inference_results.png'")

if __name__ == "__main__":
    visualize_inference()
