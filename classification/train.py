"""
Training script for brain tumor classification.
Implements complete training loop with patient-level evaluation.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from classification.dataset import create_classification_dataloaders, get_class_weights
from classification.loss import get_classification_loss
from classification.model import BrainTumorClassifier
from utils import (
    Logger,
    MetricTracker,
    compute_classification_metrics,
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
        model: Classification model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        scaler: Gradient scaler for AMP
        device: Training device
        epoch: Current epoch number
        logger: Logger instance
        log_interval: Steps between logging
        grad_clip: Gradient clipping value
        use_amp: Whether to use AMP
        
    Returns:
        Dictionary of training metrics
    """
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        
        with autocast("cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        
        accuracy = 100.0 * correct / total
        pbar.set_postfix({"loss": loss.item(), "acc": f"{accuracy:.2f}%"})
        
        if batch_idx % log_interval == 0:
            step = epoch * len(train_loader) + batch_idx
            logger.log_metrics(
                {"loss": loss.item(), "accuracy": accuracy},
                step=step,
                prefix="train/",
            )
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return {"loss": avg_loss, "accuracy": accuracy}


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    use_amp: bool = True,
) -> Tuple[Dict[str, float], list, list, list]:
    """
    Validate the model.
    
    Args:
        model: Classification model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device
        epoch: Current epoch
        use_amp: Whether to use AMP
        
    Returns:
        Tuple of (metrics dict, predictions, targets, probabilities)
    """
    model.eval()
    
    total_loss = 0.0
    
    all_preds = []
    all_targets = []
    all_probs = []
    all_patient_ids = []
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
    
    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        patient_ids = batch["patient_id"]
        
        with autocast("cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_patient_ids.extend(patient_ids)
        
        pbar.set_postfix({"loss": loss.item()})
    
    avg_loss = total_loss / len(val_loader)
    
    # Compute metrics
    import numpy as np
    
    metrics = compute_classification_metrics(
        pred=np.array(all_preds),
        target=np.array(all_targets),
        pred_probs=np.array(all_probs),
    )
    metrics["loss"] = avg_loss
    
    return metrics, all_preds, all_targets, all_probs


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
    
    # Set seed
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
    
    # Get class configuration
    classes_config = config["classes"]
    num_classes = classes_config["num_classes"]
    class_names = classes_config.get("names", [f"class_{i}" for i in range(num_classes)])
    
    # Create data loaders
    aug_config = config.get("augmentation", {})
    
    train_loader, val_loader = create_classification_dataloaders(
        train_patients=train_patients,
        val_patients=val_patients,
        roi_dir=config["data"]["roi_dir"],
        labels_file=config["data"]["labels_file"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        mode=config["model"].get("mode", "2.5d"),
        num_slices=aug_config.get("num_slices", 16),
        input_size=tuple(aug_config.get("input_size", [224, 224])),
        use_weighted_sampler=True,
    )
    
    # Create model
    model_config = config["model"]
    model = BrainTumorClassifier(
        backbone=model_config["backbone"],
        pretrained=model_config.get("pretrained", True),
        in_channels=model_config.get("in_channels", 4),
        num_classes=num_classes,
        dropout=model_config.get("dropout", 0.3),
        mode=model_config.get("mode", "2.5d"),
        num_slices=aug_config.get("num_slices", 16),
        aggregation=model_config.get("pooling", "attention"),
    )
    model = model.to(device)
    
    print(f"Model: {model_config['backbone']}")
    print(f"Parameters: {model.get_num_parameters():,}")
    
    # Create loss function
    loss_config = config["loss"]
    
    # Get class weights if specified
    class_weights = None
    if loss_config.get("alpha") is None:
        class_weights = get_class_weights(
            config["data"]["labels_file"],
            train_patients,
        ).to(device)
        print(f"Class weights: {class_weights}")
    
    criterion = get_classification_loss(
        loss_name=loss_config["name"],
        gamma=loss_config.get("gamma", 2.0),
        alpha=class_weights,
        label_smoothing=loss_config.get("label_smoothing", 0.1),
    )
    
    # Create optimizer
    train_config = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
        betas=tuple(train_config.get("betas", [0.9, 0.999])),
    )
    
    # Create scheduler
    scheduler_config = train_config["scheduler"]
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=scheduler_config.get("T_max", train_config["epochs"]),
        eta_min=scheduler_config.get("eta_min", 1e-7),
    )
    
    # AMP scaler
    use_amp = config["amp"]["enabled"]
    scaler = GradScaler("cuda", enabled=use_amp)
    
    # Resume from checkpoint
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
        name="classification_training",
        log_dir=checkpoint_dir / "logs",
        use_wandb=log_config.get("use_wandb", False),
        wandb_project=log_config.get("project_name", "brain-tumor-classification"),
        wandb_config=config,
    )
    
    logger.log_config(config)
    
    # Metric tracker
    metric_tracker = MetricTracker(
        metrics=["accuracy", "f1_macro", "loss"],
        mode="max",
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
        val_metrics, _, _, _ = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            use_amp=use_amp,
        )
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        epoch_time = time.time() - epoch_start
        
        logger.info(
            f"Epoch {epoch} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.2f}% | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1_macro']:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        logger.log_metrics(train_metrics, step=epoch, prefix="train/")
        logger.log_metrics(val_metrics, step=epoch, prefix="val/")
        
        # Update metric tracker
        improved = metric_tracker.update(val_metrics, epoch)
        
        # Save checkpoint
        is_best = improved.get("accuracy", False)
        
        model_config_save = {
            "backbone": model_config["backbone"],
            "in_channels": model_config.get("in_channels", 4),
            "num_classes": num_classes,
            "dropout": model_config.get("dropout", 0.3),
            "mode": model_config.get("mode", "2.5d"),
            "num_slices": aug_config.get("num_slices", 16),
            "aggregation": model_config.get("pooling", "attention"),
        }
        
        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "val_metrics": val_metrics,
            "model_config": model_config_save,
            "class_names": class_names,
        }
        
        save_checkpoint(
            checkpoint_state,
            checkpoint_dir / f"checkpoint_epoch_{epoch}.pt",
            is_best=is_best,
        )
        
        # Early stopping
        if is_best:
            epochs_without_improvement = 0
            logger.info(f"New best model! Accuracy: {val_metrics['accuracy']:.4f}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break
    
    # Final summary
    best_acc, best_epoch = metric_tracker.get_best("accuracy")
    logger.info(f"\nTraining complete!")
    logger.info(f"Best Accuracy: {best_acc:.4f} at epoch {best_epoch}")
    
    logger.finish()


def main():
    parser = argparse.ArgumentParser(description="Train Brain Tumor Classification Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cls.yaml",
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
