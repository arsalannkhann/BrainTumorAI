
import os
import cv2
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch

def prepare_classification_data():
    """
    Prepare classification data from raw JPEGs.
    """
    # Paths
    root_dir = Path("data/classification_raw")
    roi_dir = Path("data/roi")
    roi_dir.mkdir(parents=True, exist_ok=True)
    
    splits_dir = Path("data/splits_cls")
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    labels_file = Path("data/labels_cls.csv")
    
    classes = ["glioma", "meningioma", "pituitary", "notumor"]
    class_map = {name: i for i, name in enumerate(classes)}
    
    patient_data = []
    
    # process Training and Testing sets
    for split_name in ["Training", "Testing"]:
        split_path = root_dir / split_name
        if not split_path.exists():
            print(f"Warning: {split_path} not found")
            continue
            
        print(f"Processing {split_name} set...")
        
        for class_name in classes:
            class_dir = split_path / class_name
            if not class_dir.exists():
                print(f"Warning: {class_dir} not found")
                continue
                
            image_files = glob.glob(str(class_dir / "*.jpg"))
            print(f"  {class_name}: {len(image_files)} images")
            
            for img_path in tqdm(image_files, desc=f"Processing {class_name}"):
                # Read image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                # Resize to standard size if needed (e.g. 224x224)
                # But we will save as 3D volume with 1 slice to map to the 2.5D/3D pipeline
                # Pipeline expects (C, H, W, D) or (C, H, W) depending on dataset loading
                # Dataset loader expects .npy volume. 2.5d mode samples slices.
                # Since we have single slices, we can save as (C, H, W, 1) or just treat as 2D?
                # Dataset.py logic:
                # volume = np.load(roi_path) # (C, H, W, D)
                # if mode == "3d": ...
                # else: slices = volume[:, :, :, indices]
                
                # So we must save as 4D shape: (Channels, H, W, Depth)
                # Channels=4 for BraTS (t1,t2,flair,t1ce). Here we have RGB or Grayscale.
                # We can replicate channels or just use 1/3.
                # Let's save as (3, H, W, 1) assuming RGB
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # H, W, C
                img = img.transpose(2, 0, 1) # C, H, W
                img = img[:, :, :, np.newaxis] # C, H, W, D=1
                
                # normalize 0-1? Dataset seems to expect it? 
                # Dataset.py lines 186-190 normalize to 0-255 then / 255.0
                # So we can keep as uint8 0-255
                
                patient_id = Path(img_path).stem
                save_path = roi_dir / f"{patient_id}_roi.npy"
                np.save(save_path, img)
                
                patient_data.append({
                    "patient_id": patient_id,
                    "label": class_map[class_name],
                    "split": split_name
                })

    # Create DataFrame
    df = pd.DataFrame(patient_data)
    print(f"Total processed: {len(df)}")
    
    # Save labels
    df[["patient_id", "label"]].to_csv(labels_file, index=False)
    print(f"Saved labels to {labels_file}")
    
    # Create splits
    # Use 'Training' folder for train/val split, 'Testing' folder for test split
    train_val_df = df[df["split"] == "Training"]
    test_df = df[df["split"] == "Testing"]
    
    # 80/20 train/val split
    train_ids, val_ids = train_test_split(
        train_val_df["patient_id"].values,
        test_size=0.2,
        stratify=train_val_df["label"].values,
        random_state=42
    )
    
    # Save split files
    with open(splits_dir / "train.txt", "w") as f:
        f.write("\n".join(train_ids))
        
    with open(splits_dir / "val.txt", "w") as f:
        f.write("\n".join(val_ids))
    
    if not test_df.empty:
        with open(splits_dir / "test.txt", "w") as f:
            f.write("\n".join(test_df["patient_id"].values))
            
    print("Fold splits created in", splits_dir)

if __name__ == "__main__":
    prepare_classification_data()
