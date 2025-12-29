"""
Training script for brain tumor segmentation.
Implements complete training loop with AMP, checkpointing, and metrics logging.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings

# Suppress MONAI internal warnings
warnings.filterwarnings("ignore", category=UserWarning, module="monai.inferers.utils")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from segmentation.dataset import create_dataloaders
from segmentation.loss import get_loss_function
from segmentation.model import SegmentationModel
from utils import (
    Logger,
    MetricTracker,
    load_checkpoint,
    load_config,
    load_patient_list,
    save_checkpoint,
    set_seed,
    validate_no_leakage,
)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    logger: Logger,
    log_interval: int = 10,
    grad_clip: float = 1.0,
    use_amp: bool = True,
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: Segmentation model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        scaler: Gradient scaler for AMP
        device: Training device
        epoch: Current epoch number
        logger: Logger instance
        log_interval: Steps between logging
        grad_clip: Gradient clipping value
        use_amp: Whether to use automatic mixed precision
        
    Returns:
        Dictionary of training metrics
    """
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, batch in enumerate(pbar):
        if isinstance(batch, list):
            images = torch.cat([d["image"] for d in batch], dim=0).to(device)
            labels = torch.cat([d["label"] for d in batch], dim=0).to(device)
        else:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with AMP
        # Forward pass with AMP
        with autocast("cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({"loss": loss.item()})
        
        # Log periodically
        if batch_idx % log_interval == 0:
            step = epoch * len(train_loader) + batch_idx
            logger.log_metrics({"loss": loss.item()}, step=step, prefix="train/")
    
    avg_loss = total_loss / num_batches
    return {"loss": avg_loss}


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    roi_size: Tuple[int, int, int] = (128, 128, 128),
    sw_batch_size: int = 4,
    overlap: float = 0.5,
    use_amp: bool = True,
) -> Dict[str, float]:
    """
    Validate the model.
    
    Uses sliding window inference for full-volume evaluation.
    
    Args:
        model: Segmentation model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device
        epoch: Current epoch
        roi_size: ROI size for sliding window
        sw_batch_size: Batch size for sliding window
        overlap: Overlap ratio for sliding window
        use_amp: Whether to use AMP
        
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
    
    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        with autocast("cuda", enabled=use_amp):
            # Sliding window inference for full volume
            outputs = sliding_window_inference(
                images,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap,
            )
            loss = criterion(outputs, labels)
        
        # Compute Dice
        pred_argmax = torch.argmax(outputs, dim=1, keepdim=True)
        
        # Convert to one-hot for dice computation
        num_classes = outputs.shape[1]
        pred_onehot = torch.zeros_like(outputs)
        pred_onehot.scatter_(1, pred_argmax, 1)
        
        label_onehot = torch.zeros_like(outputs)
        label_onehot.scatter_(1, labels.long(), 1)
        
        dice_metric(pred_onehot, label_onehot)
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({"loss": loss.item()})
    
    avg_loss = total_loss / num_batches
    dice_scores = dice_metric.aggregate()
    dice_metric.reset()
    
    # Per-class Dice
    metrics = {"loss": avg_loss}
    class_names = ["ncr", "ed", "et"]  # Necrotic, Edema, Enhancing
    
    for i, name in enumerate(class_names):
        if i < len(dice_scores):
            metrics[f"dice_{name}"] = dice_scores[i].item()
    
    metrics["mean_dice"] = dice_scores.mean().item()
    
    return metrics


def train(
    config_path: str,
    resume_checkpoint: Optional[str] = None,
) -> None:
    """
    Main training function.
    
    Args:
        config_path: Path to configuration YAML file
        resume_checkpoint: Optional path to checkpoint to resume from
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set seed for reproducibility
    set_seed(config.get("seed", 42))
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load patient lists
    splits_dir = Path(config["data"]["splits_dir"])
    train_patients = load_patient_list(splits_dir / "train.txt")
    val_patients = load_patient_list(splits_dir / "val.txt")
    
    # Validate no data leakage
    validate_no_leakage(
        splits_dir / "train.txt",
        splits_dir / "val.txt",
        splits_dir / "test.txt",
    )
    
    print(f"Training patients: {len(train_patients)}")
    print(f"Validation patients: {len(val_patients)}")
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        train_patients=train_patients,
        val_patients=val_patients,
        processed_dir=config["data"]["processed_dir"],
        masks_dir=config["data"]["masks_dir"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        roi_size=tuple(config["augmentation"]["roi_size"]),
        cache_rate=config.get("cache_rate", 0.0),
        num_samples=config.get("num_samples", 4),
    )
    
    # Create model
    model_config = config["model"]
    model = SegmentationModel(**model_config)
    model = model.to(device)
    
    print(f"Model: {model_config['name']}")
    print(f"Parameters: {model.get_num_parameters():,}")
    
    # Create loss function
    loss_config = config["loss"]
    criterion = get_loss_function(
        loss_name=loss_config["name"],
        include_background=loss_config.get("include_background", False),
        softmax=loss_config.get("softmax", True),
        dice_weight=loss_config.get("dice_weight", 1.0),
        focal_weight=loss_config.get("focal_weight", 1.0),
        gamma=loss_config.get("gamma", 2.0),
    )
    
    # Create optimizer
    train_config = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
    )
    
    # Create scheduler
    scheduler_config = train_config["scheduler"]
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=scheduler_config["T_0"],
        T_mult=scheduler_config.get("T_mult", 2),
        eta_min=scheduler_config.get("eta_min", 1e-7),
    )
    
    # AMP scaler
    use_amp = config["amp"]["enabled"]
    scaler = GradScaler("cuda", enabled=use_amp)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if resume_checkpoint:
        checkpoint = load_checkpoint(
            resume_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=str(device),
        )
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Create logger
    log_config = config.get("logging", {})
    checkpoint_dir = Path(config["checkpoint"]["save_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    logger = Logger(
        name="segmentation_training",
        log_dir=checkpoint_dir / "logs",
        use_wandb=log_config.get("use_wandb", False),
        wandb_project=log_config.get("project_name", "brain-tumor-segmentation"),
        wandb_config=config,
    )
    
    logger.log_config(config)
    
    # Metric tracker
    metric_tracker = MetricTracker(
        metrics=["mean_dice", "loss"],
        mode="max",  # Higher dice is better
    )
    
    # Training loop
    num_epochs = train_config["epochs"]
    early_stopping_patience = train_config["early_stopping"]["patience"]
    epochs_without_improvement = 0
    
    print(f"\nStarting training for {num_epochs} epochs...")
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            logger=logger,
            log_interval=log_config.get("log_interval", 10),
            grad_clip=train_config.get("grad_clip", 1.0),
            use_amp=use_amp,
        )
        
        # Validate
        val_metrics = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            roi_size=tuple(config["augmentation"]["roi_size"]),
            sw_batch_size=config["validation"].get("sliding_window_batch_size", 4),
            overlap=config["validation"].get("overlap", 0.5),
            use_amp=use_amp,
        )
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        epoch_time = time.time() - epoch_start
        
        logger.info(
            f"Epoch {epoch} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Dice: {val_metrics['mean_dice']:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        logger.log_metrics(train_metrics, step=epoch, prefix="train/")
        logger.log_metrics(val_metrics, step=epoch, prefix="val/")
        
        # Update metric tracker
        improved = metric_tracker.update(val_metrics, epoch)
        
        # Save checkpoint
        is_best = improved.get("mean_dice", False)
        save_best_only = config.get("checkpoint", {}).get("save_best_only", False)
        
        if save_best_only:
            save_path = checkpoint_dir / "latest_model.pt"
        else:
            save_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        
        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "val_metrics": val_metrics,
            "model_config": model_config,
        }
        
        save_checkpoint(
            checkpoint_state,
            save_path,
            is_best=is_best,
        )
        
        # Early stopping
        if is_best:
            epochs_without_improvement = 0
            logger.info(f"New best model! Dice: {val_metrics['mean_dice']:.4f}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break
    
    # Final summary
    best_dice, best_epoch = metric_tracker.get_best("mean_dice")
    logger.info(f"\nTraining complete!")
    logger.info(f"Best Dice: {best_dice:.4f} at epoch {best_epoch}")
    
    logger.finish()


def main():
    parser = argparse.ArgumentParser(description="Train Brain Tumor Segmentation Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/seg.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    
    args = parser.parse_args()
    
    train(config_path=args.config, resume_checkpoint=args.resume)


if __name__ == "__main__":
    main()
