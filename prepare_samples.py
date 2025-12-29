import numpy as np
from pathlib import Path

def crop_volume(path, output_path, size=128):
    data = np.load(path)
    C, H, W, D = data.shape
    
    # Center crop
    sh = (H - size) // 2
    sw = (W - size) // 2
    sd = (D - size) // 2
    
    cropped = data[:, sh:sh+size, sw:sw+size, sd:sd+size]
    np.save(output_path, cropped)
    print(f"Saved {output_path} with shape {cropped.shape}")

def crop_mask(path, output_path, size=128):
    import nibabel as nib
    img = nib.load(str(path))
    data = img.get_fdata()
    H, W, D = data.shape
    
    # Center crop
    sh = (H - size) // 2
    sw = (W - size) // 2
    sd = (D - size) // 2
    
    cropped = data[sh:sh+size, sw:sw+size, sd:sd+size]
    new_img = nib.Nifti1Image(cropped, img.affine)
    nib.save(new_img, str(output_path))
    print(f"Saved {output_path} with shape {cropped.shape}")

dest = Path("data/samples/processed")
dest.mkdir(parents=True, exist_ok=True)
mask_dest = Path("data/samples/masks")
mask_dest.mkdir(parents=True, exist_ok=True)

for i in [1, 2]:
    patient = f"BraTS20_Training_00{i}"
    crop_volume(f"data/processed/{patient}/{patient}.npy", dest / f"{patient}.npy")
    crop_mask(f"data/masks/{patient}/seg.nii.gz", mask_dest / f"{patient}_seg.nii.gz")
