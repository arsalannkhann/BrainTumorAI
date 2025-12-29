
import os
import argparse
from pathlib import Path
import nibabel as nib
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

def process_mask(patient_path, raw_dir, output_dir):
    try:
        patient_id = patient_path.name
        
        # Look for the segmentation file
        seg_file = list(patient_path.glob("*_seg.nii"))
        if not seg_file:
            # Try .nii.gz
            seg_file = list(patient_path.glob("*_seg.nii.gz"))
            
        if not seg_file:
            print(f"No segmentation file found for {patient_id}")
            return False
            
        seg_file = seg_file[0]
        
        # Create output directory
        patient_output_dir = output_dir / patient_id
        patient_output_dir.mkdir(parents=True, exist_ok=True)
        
        target_file = patient_output_dir / "seg.nii.gz"
        
        if target_file.exists():
            return True
            
        # Load and save as compressed nifti
        img = nib.load(str(seg_file))
        nib.save(img, str(target_file))
        
        return True
    except Exception as e:
        print(f"Error processing {patient_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Prepare masks for training")
    parser.add_argument("--raw-dir", type=Path, default="data/raw", help="Path to raw data")
    parser.add_argument("--output-dir", type=Path, default="data/masks", help="Path to output masks")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers")
    
    args = parser.parse_args()
    
    if not args.raw_dir.exists():
        print(f"Raw directory does not exist: {args.raw_dir}")
        return
        
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all patient directories
    # Assuming structure data/raw/BraTS20_Training_XXX
    patient_dirs = [d for d in args.raw_dir.iterdir() if d.is_dir() and "Training" in d.name]
    
    print(f"Found {len(patient_dirs)} training patients.")
    
    worker_fn = partial(process_mask, raw_dir=args.raw_dir, output_dir=args.output_dir)
    
    with mp.Pool(args.num_workers) as pool:
        results = list(tqdm(
            pool.imap(worker_fn, patient_dirs),
            total=len(patient_dirs),
            desc="Preparing masks"
        ))
        
    print(f"Successfully processed {sum(results)} masks.")

if __name__ == "__main__":
    main()
