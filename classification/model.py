"""
Classification models for brain tumor diagnosis.
Uses timm for ConvNeXt and Swin Transformer backbones.
Extended with Vision Mamba and TransMIL aggregation.
"""

from typing import Optional, Tuple, Union

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional imports for Mamba and TransMIL
try:
    from mamba.vim_backbone import VimClassifier, create_vim_classifier, MAMBA_AVAILABLE
    VIM_AVAILABLE = True
except ImportError:
    VIM_AVAILABLE = False
    MAMBA_AVAILABLE = False

try:
    from classification.transmil import TransMILAggregator, TransMILEncoder
    TRANSMIL_AVAILABLE = True
except ImportError:
    TRANSMIL_AVAILABLE = False


class MultiSliceEncoder(nn.Module):
    """
    Encodes multiple 2D slices into a single feature vector.
    Aggregates slice-level features using attention or pooling.
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        aggregation: str = "attention",
        num_slices: int = 16,
    ):
        """
        Initialize multi-slice encoder.
        
        Args:
            backbone: 2D CNN backbone
            feature_dim: Feature dimension from backbone
            aggregation: "attention", "mean", or "max"
            num_slices: Expected number of input slices
        """
        super().__init__()
        
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.aggregation = aggregation
        self.num_slices = num_slices
        
        if aggregation == "attention":
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 4),
                nn.Tanh(),
                nn.Linear(feature_dim // 4, 1),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, num_slices, C, H, W)
            
        Returns:
            Aggregated features (B, feature_dim)
        """
        batch_size, num_slices, channels, height, width = x.shape
        
        # Reshape for batch processing
        x = x.view(batch_size * num_slices, channels, height, width)
        
        # Extract features for all slices
        features = self.backbone(x)  # (B*num_slices, feature_dim)
        
        # Reshape back
        features = features.view(batch_size, num_slices, -1)  # (B, num_slices, feature_dim)
        
        # Aggregate
        if self.aggregation == "attention":
            # Attention-based aggregation
            attention_weights = self.attention(features)  # (B, num_slices, 1)
            attention_weights = F.softmax(attention_weights, dim=1)
            aggregated = (features * attention_weights).sum(dim=1)  # (B, feature_dim)
        elif self.aggregation == "mean":
            aggregated = features.mean(dim=1)
        elif self.aggregation == "max":
            aggregated = features.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        return aggregated


class BrainTumorClassifier(nn.Module):
    """
    Brain tumor classification model using ConvNeXt or Swin Transformer.
    """
    
    def __init__(
        self,
        backbone: str = "convnext_base",
        pretrained: bool = True,
        in_channels: int = 4,
        num_classes: int = 4,
        dropout: float = 0.3,
        mode: str = "2.5d",
        num_slices: int = 16,
        aggregation: str = "attention",
    ):
        """
        Initialize the classifier.
        
        Args:
            backbone: timm model name
            pretrained: Use ImageNet pretrained weights
            in_channels: Number of input channels
            num_classes: Number of output classes
            dropout: Dropout rate
            mode: "3d" or "2.5d"
            num_slices: Number of slices for 2.5D mode
            aggregation: Aggregation method for 2.5D mode
        """
        super().__init__()
        
        self.backbone_name = backbone
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.mode = mode.lower()
        self.num_slices = num_slices
        
        # Create backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier head
            in_chans=in_channels,
        )
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        
        if self.mode == "2.5d":
            # Multi-slice encoder
            self.encoder = MultiSliceEncoder(
                backbone=self.backbone,
                feature_dim=self.feature_dim,
                aggregation=aggregation,
                num_slices=num_slices,
            )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(self.feature_dim // 2, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
               - 3D mode: (B, C, H, W, D)
               - 2.5D mode: (B, num_slices, C, H, W)
               
        Returns:
            Class logits (B, num_classes)
        """
        if self.mode == "3d":
            # 3D mode: pool across depth dimension
            # (B, C, H, W, D) -> (B, C, H, W) via adaptive pooling
            batch_size = x.shape[0]
            
            # Global average pool across depth
            x = x.mean(dim=-1)  # (B, C, H, W)
            
            features = self.backbone(x)
        else:
            # 2.5D mode
            features = self.encoder(x)
        
        logits = self.classifier(features)
        return logits
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EnsembleClassifier(nn.Module):
    """
    Ensemble of multiple classifiers for robust prediction.
    """
    
    def __init__(
        self,
        backbones: list[str],
        in_channels: int = 4,
        num_classes: int = 4,
        pretrained: bool = True,
        mode: str = "2.5d",
    ):
        """
        Initialize ensemble classifier.
        
        Args:
            backbones: List of timm model names
            in_channels: Number of input channels
            num_classes: Number of output classes
            pretrained: Use pretrained weights
            mode: Input mode
        """
        super().__init__()
        
        self.models = nn.ModuleList([
            BrainTumorClassifier(
                backbone=backbone,
                pretrained=pretrained,
                in_channels=in_channels,
                num_classes=num_classes,
                mode=mode,
            )
            for backbone in backbones
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with averaging.
        
        Args:
            x: Input tensor
            
        Returns:
            Averaged logits
        """
        outputs = [model(x) for model in self.models]
        return torch.stack(outputs).mean(dim=0)


def create_classifier(
    backbone: str = "convnext_base",
    pretrained: bool = True,
    in_channels: int = 4,
    num_classes: int = 4,
    dropout: float = 0.3,
    mode: str = "2.5d",
    num_slices: int = 16,
    aggregation: str = "attention",
    **kwargs,
) -> nn.Module:
    """
    Create a brain tumor classifier.
    
    Args:
        backbone: Model name. Options:
            - timm models: "convnext_base", "swin_base_patch4_window7_224", etc.
            - Vision Mamba: "vim_tiny", "vim_small", "vim_base"
        pretrained: Use pretrained weights
        in_channels: Number of input channels
        num_classes: Number of output classes
        dropout: Dropout rate
        mode: "3d" or "2.5d"
        num_slices: Number of slices
        aggregation: "attention", "mean", "max", or "transmil"
        
    Returns:
        Classifier model
    """
    # Check for Vision Mamba backbone
    if backbone.startswith("vim_"):
        if not VIM_AVAILABLE:
            raise ImportError(
                "Vision Mamba not available. Install mamba-ssm package or "
                "check mamba/ directory for vim_backbone.py"
            )
        variant = backbone.split("_")[1]  # e.g., "vim_base" -> "base"
        return create_vim_classifier(
            variant=variant,
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=dropout,
            mode=mode,
            num_slices=num_slices,
            aggregation=aggregation if aggregation != "transmil" else "attention",
            **kwargs,
        )
    
    # Standard timm-based classifier
    return BrainTumorClassifier(
        backbone=backbone,
        pretrained=pretrained,
        in_channels=in_channels,
        num_classes=num_classes,
        dropout=dropout,
        mode=mode,
        num_slices=num_slices,
        aggregation=aggregation,
    )


def load_classifier_from_checkpoint(
    checkpoint_path: str,
    device: str = "cuda",
    **model_kwargs,
) -> BrainTumorClassifier:
    """
    Load classifier from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        device: Device to load to
        **model_kwargs: Model configuration overrides
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get model config from checkpoint
    if "model_config" in checkpoint:
        config = checkpoint["model_config"]
        config.update(model_kwargs)
    else:
        config = model_kwargs
    
    model = BrainTumorClassifier(**config)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


# Available backbone options
AVAILABLE_BACKBONES = [
    # ConvNeXt family
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "convnext_large",
    "convnextv2_tiny",
    "convnextv2_base",
    # Swin Transformer family
    "swin_tiny_patch4_window7_224",
    "swin_small_patch4_window7_224",
    "swin_base_patch4_window7_224",
    "swin_base_patch4_window12_384",
    # EfficientNet
    "efficientnet_b0",
    "efficientnet_b3",
    "efficientnet_b4",
    # ResNet
    "resnet50",
    "resnet101",
]


if __name__ == "__main__":
    # Test model creation
    print("Testing classifier creation...")
    
    # Test 2.5D mode
    model = BrainTumorClassifier(
        backbone="convnext_tiny",
        pretrained=False,
        in_channels=4,
        num_classes=4,
        mode="2.5d",
        num_slices=16,
    )
    print(f"2.5D Model parameters: {model.get_num_parameters():,}")
    
    # Test forward pass
    x = torch.randn(2, 16, 4, 224, 224)  # (B, num_slices, C, H, W)
    with torch.no_grad():
        y = model(x)
    print(f"2.5D Input shape: {x.shape}")
    print(f"2.5D Output shape: {y.shape}")
    
    # Test 3D mode
    model_3d = BrainTumorClassifier(
        backbone="convnext_tiny",
        pretrained=False,
        in_channels=4,
        num_classes=4,
        mode="3d",
    )
    
    x_3d = torch.randn(2, 4, 224, 224, 16)  # (B, C, H, W, D)
    with torch.no_grad():
        y_3d = model_3d(x_3d)
    print(f"3D Input shape: {x_3d.shape}")
    print(f"3D Output shape: {y_3d.shape}")
    
    print("Classifier tests passed!")
