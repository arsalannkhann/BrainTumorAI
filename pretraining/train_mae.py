"""
BM-MAE Pretraining Script.

Self-supervised pretraining for brain tumor detection using
masked autoencoder reconstruction objective.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pretraining.bm_mae import BM_MAE, create_bm_mae
from utils.seed import set_seed
from utils.logger import Logger


class UnlabeledMRIDataset(Dataset):
    """
    Dataset for unlabeled MRI volumes for self-supervised pretraining.
    """
    
    def __init__(
        self,
        data_dir: str,
        patient_list: str,
        modalities: list = ["t1", "t2", "flair", "t1ce"],
        img_size: tuple = (128, 128, 128),
    ):
        self.data_dir = Path(data_dir)
        self.modalities = modalities
        self.img_size = img_size
        
        # Load patient IDs
        with open(patient_list, "r") as f:
            self.patient_ids = [line.strip() for line in f if line.strip()]
    
    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        
        # Load modalities
        volumes = []
        for mod in self.modalities:
            vol_path = self.data_dir / patient_id / f"{patient_id}_{mod}.nii.gz"
            
            if vol_path.exists():
                import nibabel as nib
                vol = nib.load(str(vol_path)).get_fdata()
                vol = torch.tensor(vol, dtype=torch.float32)
            else:
                # Create zero volume if modality missing
                vol = torch.zeros(self.img_size, dtype=torch.float32)
            
            volumes.append(vol)
        
        # Stack modalities: (C, D, H, W)
        volume = torch.stack(volumes, dim=0)
        
        # Normalize per volume
        mean = volume.mean()
        std = volume.std() + 1e-8
        volume = (volume - mean) / std
        
        # Random crop to target size
        volume = self._random_crop(volume, self.img_size)
        
        return volume
    
    def _random_crop(self, volume, target_size):
        """Random crop to target size."""
        c, d, h, w = volume.shape
        td, th, tw = target_size
        
        if d >= td and h >= th and w >= tw:
            sd = torch.randint(0, d - td + 1, (1,)).item() if d > td else 0
            sh = torch.randint(0, h - th + 1, (1,)).item() if h > th else 0
            sw = torch.randint(0, w - tw + 1, (1,)).item() if w > tw else 0
            
            volume = volume[:, sd:sd+td, sh:sh+th, sw:sw+tw]
        else:
            # Pad if smaller
            pad_d = max(0, td - d)
            pad_h = max(0, th - h)
            pad_w = max(0, tw - w)
            
            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                volume = torch.nn.functional.pad(
                    volume,
                    (0, pad_w, 0, pad_h, 0, pad_d),
                    mode='constant',
                    value=0,
                )
            
            # Crop to exact size
            volume = volume[:, :td, :th, :tw]
        
        return volume


def train_mae(config: dict):
    """
    Main MAE pretraining loop.
    
    Args:
        config: Training configuration dictionary
    """
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    set_seed(config.get("seed", 42))
    
    # Create model
    model_config = config["model"]
    model = create_bm_mae(
        img_size=tuple(model_config["img_size"]),
        variant=model_config.get("variant", "base"),
        patch_size=tuple(model_config.get("patch_size", [16, 16, 16])),
        in_channels=model_config.get("in_channels", 4),
        mask_ratio=model_config.get("mask_ratio", 0.75),
        modality_dropout=model_config.get("modality_dropout", 0.1),
    )
    model = model.to(device)
    print(f"Model parameters: {model.get_num_parameters():,}")
    
    # Create dataset
    data_config = config["data"]
    dataset = UnlabeledMRIDataset(
        data_dir=data_config["processed_dir"],
        patient_list=data_config["train_list"],
        img_size=tuple(model_config["img_size"]),
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"].get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    
    # Optimizer
    train_config = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config.get("weight_decay", 0.05),
        betas=tuple(train_config.get("betas", [0.9, 0.95])),
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_config["epochs"],
        eta_min=train_config.get("min_lr", 1e-7),
    )
    
    # Mixed precision
    scaler = torch.amp.GradScaler("cuda") if config.get("amp", {}).get("enabled", True) else None
    
    # Logging
    logger = Logger(
        use_wandb=config.get("logging", {}).get("use_wandb", False),
        project_name=config.get("logging", {}).get("project_name", "bm-mae-pretrain"),
        config=config,
    )
    
    # Checkpoint directory
    ckpt_dir = Path(config.get("checkpoint", {}).get("save_dir", "checkpoints/mae"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_loss = float("inf")
    
    for epoch in range(train_config["epochs"]):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{train_config['epochs']}")
        for batch_idx, volume in enumerate(pbar):
            volume = volume.to(device)
            
            optimizer.zero_grad()
            
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    loss = model(volume)
                
                scaler.scale(loss).backward()
                
                # Gradient clipping
                if train_config.get("grad_clip", 0) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        train_config["grad_clip"],
                    )
                
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = model(volume)
                loss.backward()
                
                if train_config.get("grad_clip", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        train_config["grad_clip"],
                    )
                
                optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
            # Log
            if batch_idx % config.get("logging", {}).get("log_interval", 10) == 0:
                logger.log({
                    "train/loss": loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                })
        
        scheduler.step()
        
        # Epoch summary
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}: avg_loss = {avg_loss:.4f}")
        
        logger.log({
            "epoch": epoch + 1,
            "train/epoch_loss": avg_loss,
        })
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "config": config,
            }
            
            torch.save(checkpoint, ckpt_dir / "best_mae.pt")
            print(f"Saved best checkpoint (loss: {avg_loss:.4f})")
        
        # Save periodic checkpoint
        if (epoch + 1) % config.get("checkpoint", {}).get("save_interval", 50) == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "config": config,
            }
            torch.save(checkpoint, ckpt_dir / f"mae_epoch_{epoch + 1}.pt")
    
    # Save final encoder weights for downstream tasks
    encoder_state = model.encoder.state_dict()
    torch.save(encoder_state, ckpt_dir / "encoder_weights.pt")
    print(f"Saved encoder weights to {ckpt_dir / 'encoder_weights.pt'}")
    
    logger.finish()


def main():
    parser = argparse.ArgumentParser(description="BM-MAE Pretraining")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mae.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    train_mae(config)


if __name__ == "__main__":
    main()
