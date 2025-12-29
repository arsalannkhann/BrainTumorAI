"""
Loss functions for brain tumor classification.
Implements Focal Loss and Label Smoothing for class imbalance handling.
"""

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in classification.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    The focal loss down-weights well-classified examples and focuses
    on hard, misclassified examples.
    """
    
    def __init__(
        self,
        alpha: Optional[Union[float, torch.Tensor]] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Class weights. Can be:
                   - None: No weighting
                   - float: Weight for positive class (binary)
                   - Tensor: Per-class weights
            gamma: Focusing parameter (0 = CE, higher = more focus on hard examples)
            reduction: Reduction method ("mean", "sum", "none")
            label_smoothing: Label smoothing factor (0-1)
        """
        super().__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.FloatTensor(alpha)
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predicted logits (B, num_classes)
            targets: Ground truth labels (B,)
            
        Returns:
            Focal loss value
        """
        num_classes = inputs.shape[1]
        
        # Apply label smoothing
        if self.label_smoothing > 0:
            with torch.no_grad():
                targets_smooth = torch.zeros_like(inputs)
                targets_smooth.fill_(self.label_smoothing / (num_classes - 1))
                targets_smooth.scatter_(1, targets.unsqueeze(1), 1 - self.label_smoothing)
        else:
            targets_smooth = F.one_hot(targets, num_classes).float()
        
        # Compute probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Gather probabilities of correct class
        if self.label_smoothing > 0:
            pt = (probs * targets_smooth).sum(dim=1)
        else:
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Compute cross entropy
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        # Apply class weights
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha = self.alpha.to(inputs.device)
                alpha_t = alpha.gather(0, targets)
            else:
                alpha_t = self.alpha
            focal_loss = alpha_t * focal_loss
        
        # Reduce
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy with Label Smoothing.
    
    Prevents overconfident predictions by softening the target distribution.
    """
    
    def __init__(
        self,
        smoothing: float = 0.1,
        reduction: str = "mean",
        weight: Optional[torch.Tensor] = None,
    ):
        """
        Initialize Label Smoothing CE.
        
        Args:
            smoothing: Smoothing factor (0-1)
            reduction: Reduction method
            weight: Class weights
        """
        super().__init__()
        
        self.smoothing = smoothing
        self.reduction = reduction
        self.weight = weight
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute label smoothing cross entropy.
        
        Args:
            inputs: Predicted logits (B, num_classes)
            targets: Ground truth labels (B,)
            
        Returns:
            Loss value
        """
        return F.cross_entropy(
            inputs,
            targets,
            weight=self.weight.to(inputs.device) if self.weight is not None else None,
            reduction=self.reduction,
            label_smoothing=self.smoothing,
        )


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon).
    
    Pulls together embeddings of the same class while pushing apart
    embeddings of different classes.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
    ):
        """
        Initialize SupCon loss.
        
        Args:
            temperature: Temperature for scaling
            base_temperature: Base temperature for normalization
        """
        super().__init__()
        
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.
        
        Args:
            features: Feature embeddings (B, feature_dim)
            labels: Class labels (B,)
            
        Returns:
            Contrastive loss value
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(features, features.T) / self.temperature
        
        # Create masks
        labels = labels.contiguous().view(-1, 1)
        mask_positive = torch.eq(labels, labels.T).float().to(device)
        
        # Mask out self-similarity
        logits_mask = torch.ones_like(mask_positive)
        logits_mask.fill_diagonal_(0)
        
        mask_positive = mask_positive * logits_mask
        
        # Compute log-softmax
        exp_logits = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-6)
        
        # Compute mean of log-likelihood over positives
        mean_log_prob = (mask_positive * log_prob).sum(dim=1) / (mask_positive.sum(dim=1) + 1e-6)
        
        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob
        loss = loss.mean()
        
        return loss


class CombinedClassificationLoss(nn.Module):
    """
    Combined loss with Focal Loss and optional Contrastive Loss.
    """
    
    def __init__(
        self,
        focal_weight: float = 1.0,
        contrastive_weight: float = 0.0,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.1,
        temperature: float = 0.07,
    ):
        """
        Initialize combined loss.
        
        Args:
            focal_weight: Weight for focal loss
            contrastive_weight: Weight for contrastive loss
            gamma: Focal loss gamma
            alpha: Class weights
            label_smoothing: Label smoothing factor
            temperature: Temperature for contrastive loss
        """
        super().__init__()
        
        self.focal_weight = focal_weight
        self.contrastive_weight = contrastive_weight
        
        self.focal_loss = FocalLoss(
            alpha=alpha,
            gamma=gamma,
            label_smoothing=label_smoothing,
        )
        
        if contrastive_weight > 0:
            self.contrastive_loss = SupervisedContrastiveLoss(
                temperature=temperature,
            )
        else:
            self.contrastive_loss = None
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            logits: Predicted logits
            targets: Ground truth labels
            features: Optional feature embeddings for contrastive loss
            
        Returns:
            Combined loss value
        """
        loss = self.focal_weight * self.focal_loss(logits, targets)
        
        if self.contrastive_loss is not None and features is not None:
            loss += self.contrastive_weight * self.contrastive_loss(features, targets)
        
        return loss


def get_classification_loss(
    loss_name: str = "FocalLoss",
    **kwargs,
) -> nn.Module:
    """
    Get a classification loss function by name.
    
    Args:
        loss_name: Name of loss function
        **kwargs: Additional arguments
        
    Returns:
        Loss function module
    """
    loss_name_lower = loss_name.lower()
    
    if loss_name_lower == "focalloss" or loss_name_lower == "focal":
        return FocalLoss(**kwargs)
    elif loss_name_lower == "labelsmoothing" or loss_name_lower == "labelsmoothingce":
        return LabelSmoothingCrossEntropy(**kwargs)
    elif loss_name_lower == "crossentropy" or loss_name_lower == "ce":
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_name_lower == "supcon" or loss_name_lower == "contrastive":
        return SupervisedContrastiveLoss(**kwargs)
    elif loss_name_lower == "combined":
        return CombinedClassificationLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


if __name__ == "__main__":
    # Test loss functions
    print("Testing classification losses...")
    
    logits = torch.randn(8, 4)
    targets = torch.randint(0, 4, (8,))
    features = torch.randn(8, 512)
    
    # Test Focal Loss
    loss_fn = FocalLoss(gamma=2.0, label_smoothing=0.1)
    loss = loss_fn(logits, targets)
    print(f"Focal Loss: {loss.item():.4f}")
    
    # Test with class weights
    weights = torch.tensor([1.0, 2.0, 1.5, 0.8])
    loss_fn = FocalLoss(alpha=weights, gamma=2.0)
    loss = loss_fn(logits, targets)
    print(f"Focal Loss (weighted): {loss.item():.4f}")
    
    # Test Label Smoothing CE
    loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
    loss = loss_fn(logits, targets)
    print(f"Label Smoothing CE: {loss.item():.4f}")
    
    # Test SupCon
    loss_fn = SupervisedContrastiveLoss()
    loss = loss_fn(features, targets)
    print(f"SupCon Loss: {loss.item():.4f}")
    
    # Test Combined
    loss_fn = CombinedClassificationLoss(
        focal_weight=1.0,
        contrastive_weight=0.1,
    )
    loss = loss_fn(logits, targets, features)
    print(f"Combined Loss: {loss.item():.4f}")
    
    print("Classification loss tests passed!")
