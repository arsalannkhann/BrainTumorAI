"""
Loss functions for brain tumor segmentation.
Implements Dice, Focal, and combined losses for handling class imbalance.
"""

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceFocalLoss, DiceLoss, FocalLoss, TverskyLoss


class DiceFocalCombinedLoss(nn.Module):
    """
    Combined Dice and Focal loss for segmentation.
    
    This loss combines:
    - Dice Loss: Handles class imbalance by measuring overlap
    - Focal Loss: Focuses on hard-to-classify examples
    """
    
    def __init__(
        self,
        include_background: bool = False,
        softmax: bool = True,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        gamma: float = 2.0,
        alpha: Optional[Union[float, Sequence[float]]] = None,
        reduction: str = "mean",
    ):
        """
        Initialize the combined loss.
        
        Args:
            include_background: Whether to include background in loss computation
            softmax: Whether to apply softmax to predictions
            dice_weight: Weight for Dice loss component
            focal_weight: Weight for Focal loss component
            gamma: Focal loss focusing parameter
            alpha: Class weights for Focal loss
            reduction: Reduction method ("mean", "sum", "none")
        """
        super().__init__()
        
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        self.dice_focal = DiceFocalLoss(
            include_background=include_background,
            softmax=softmax,
            to_onehot_y=True,
            gamma=gamma,
            alpha=alpha,
            lambda_dice=dice_weight,
            lambda_focal=focal_weight,
            reduction=reduction,
        )
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            pred: Predictions (B, C, H, W, D)
            target: Ground truth (B, 1, H, W, D) with class indices
            
        Returns:
            Combined loss value
        """
        return self.dice_focal(pred, target)


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for handling extreme class imbalance.
    
    Generalizes Dice loss with tunable false positive/negative weights
    and focal parameter for hard example mining.
    
    FTL = (1 - TI)^gamma
    TI = TP / (TP + alpha*FP + beta*FN)
    """
    
    def __init__(
        self,
        include_background: bool = False,
        softmax: bool = True,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 0.75,
        reduction: str = "mean",
    ):
        """
        Initialize Focal Tversky Loss.
        
        Args:
            include_background: Whether to include background
            softmax: Whether to apply softmax
            alpha: Weight for false positives
            beta: Weight for false negatives
            gamma: Focal parameter (lower = more focus on hard examples)
            reduction: Reduction method
        """
        super().__init__()
        
        self.include_background = include_background
        self.softmax = softmax
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Focal Tversky Loss.
        
        Args:
            pred: Predictions (B, C, H, W, D)
            target: Ground truth (B, 1, H, W, D) with class indices
            
        Returns:
            Loss value
        """
        n_classes = pred.shape[1]
        
        if self.softmax:
            pred = F.softmax(pred, dim=1)
        
        # Convert target to one-hot
        target_one_hot = F.one_hot(
            target.squeeze(1).long(),
            num_classes=n_classes,
        ).permute(0, 4, 1, 2, 3).float()
        
        # Compute Tversky Index per class
        dims = (2, 3, 4)  # Spatial dimensions
        
        if not self.include_background:
            pred = pred[:, 1:]
            target_one_hot = target_one_hot[:, 1:]
        
        tp = (pred * target_one_hot).sum(dim=dims)
        fp = (pred * (1 - target_one_hot)).sum(dim=dims)
        fn = ((1 - pred) * target_one_hot).sum(dim=dims)
        
        tversky_index = (tp + 1e-6) / (tp + self.alpha * fp + self.beta * fn + 1e-6)
        
        # Focal Tversky Loss
        focal_tversky = torch.pow(1 - tversky_index, self.gamma)
        
        # Reduce
        if self.reduction == "mean":
            return focal_tversky.mean()
        elif self.reduction == "sum":
            return focal_tversky.sum()
        else:
            return focal_tversky


class BraTSLoss(nn.Module):
    """
    BraTS-style hierarchical loss for brain tumor segmentation.
    
    Computes loss on three nested regions:
    - Whole Tumor (WT): All tumor regions
    - Tumor Core (TC): Active tumor + necrotic core
    - Enhancing Tumor (ET): Active enhancing tumor
    """
    
    def __init__(
        self,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        gamma: float = 2.0,
    ):
        """
        Initialize BraTS loss.
        
        Args:
            dice_weight: Weight for Dice component
            focal_weight: Weight for Focal component
            gamma: Focal loss gamma
        """
        super().__init__()
        
        # Loss for each region
        self.loss_fn = DiceFocalCombinedLoss(
            include_background=False,
            softmax=False,  # We handle softmax manually
            dice_weight=dice_weight,
            focal_weight=focal_weight,
            gamma=gamma,
        )
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute BraTS hierarchical loss.
        
        Assumes BraTS labels:
        - 0: Background
        - 1: Necrotic tumor core (NCR)
        - 2: Peritumoral edema (ED)
        - 3/4: Enhancing tumor (ET)
        
        Args:
            pred: Predictions (B, 4, H, W, D)
            target: Ground truth (B, 1, H, W, D)
            
        Returns:
            Combined loss
        """
        # Apply softmax
        pred_soft = F.softmax(pred, dim=1)
        
        target_squeezed = target.squeeze(1).long()
        
        # Convert to binary masks for each region
        # Label 4 is sometimes used for ET in BraTS
        target_squeezed = torch.where(target_squeezed == 4, torch.ones_like(target_squeezed) * 3, target_squeezed)
        
        # Whole Tumor (WT): labels 1, 2, 3
        wt_pred = pred_soft[:, 1:].sum(dim=1, keepdim=True)
        wt_target = (target_squeezed > 0).unsqueeze(1).float()
        
        # Tumor Core (TC): labels 1, 3 (NCR + ET)
        tc_pred = pred_soft[:, 1:2] + pred_soft[:, 3:4] if pred_soft.shape[1] > 3 else pred_soft[:, 1:2]
        tc_target = ((target_squeezed == 1) | (target_squeezed == 3)).unsqueeze(1).float()
        
        # Enhancing Tumor (ET): label 3
        et_pred = pred_soft[:, 3:4] if pred_soft.shape[1] > 3 else pred_soft[:, 1:2]
        et_target = (target_squeezed == 3).unsqueeze(1).float()
        
        # Dice loss for each region
        dice_wt = self._dice_loss(wt_pred, wt_target)
        dice_tc = self._dice_loss(tc_pred, tc_target)
        dice_et = self._dice_loss(et_pred, et_target)
        
        # Average
        return (dice_wt + dice_tc + dice_et) / 3.0
    
    def _dice_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        smooth: float = 1e-6,
    ) -> torch.Tensor:
        """Compute Dice loss."""
        dims = (2, 3, 4)
        
        intersection = (pred * target).sum(dim=dims)
        union = pred.sum(dim=dims) + target.sum(dim=dims)
        
        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()


def get_loss_function(
    loss_name: str = "DiceFocalLoss",
    **kwargs,
) -> nn.Module:
    """
    Get a loss function by name.
    
    Args:
        loss_name: Name of loss function
        **kwargs: Additional arguments for the loss
        
    Returns:
        Loss function module
    """
    loss_name_lower = loss_name.lower()
    
    if loss_name_lower == "dicefocalloss" or loss_name_lower == "dicefocal":
        return DiceFocalCombinedLoss(**kwargs)
    elif loss_name_lower == "diceloss" or loss_name_lower == "dice":
        return DiceLoss(
            include_background=kwargs.get("include_background", False),
            softmax=kwargs.get("softmax", True),
            to_onehot_y=True,
        )
    elif loss_name_lower == "focalloss" or loss_name_lower == "focal":
        return FocalLoss(
            include_background=kwargs.get("include_background", False),
            gamma=kwargs.get("gamma", 2.0),
            to_onehot_y=True,
        )
    elif loss_name_lower == "focaltversky" or loss_name_lower == "focaltversktloss":
        return FocalTverskyLoss(
            include_background=kwargs.get("include_background", False),
            softmax=kwargs.get("softmax", True),
            alpha=kwargs.get("alpha", 0.3),
            beta=kwargs.get("beta", 0.7),
            gamma=kwargs.get("gamma", 0.75),
            reduction=kwargs.get("reduction", "mean"),
        )
    elif loss_name_lower == "bratsloss" or loss_name_lower == "brats":
        return BraTSLoss(
            dice_weight=kwargs.get("dice_weight", 1.0),
            focal_weight=kwargs.get("focal_weight", 1.0),
            gamma=kwargs.get("gamma", 2.0),
        )
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")
    
    pred = torch.randn(2, 4, 32, 32, 32)
    target = torch.randint(0, 4, (2, 1, 32, 32, 32))
    
    # Test DiceFocal
    loss_fn = DiceFocalCombinedLoss()
    loss = loss_fn(pred, target)
    print(f"DiceFocal Loss: {loss.item():.4f}")
    
    # Test FocalTversky
    loss_fn = FocalTverskyLoss()
    loss = loss_fn(pred, target)
    print(f"Focal Tversky Loss: {loss.item():.4f}")
    
    # Test BraTS
    loss_fn = BraTSLoss()
    loss = loss_fn(pred, target)
    print(f"BraTS Loss: {loss.item():.4f}")
    
    print("Loss function tests passed!")
