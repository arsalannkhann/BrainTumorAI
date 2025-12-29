"""
Vision Mamba (Vim) backbone for brain tumor classification.

Implements bidirectional state-space modeling with linear O(N) complexity.
Provides global context capture for high-resolution 3D MRI data.

Reference: "Vision Mamba: Efficient Visual Representation Learning with
           Bidirectional State Space Model" (Zhu et al., 2024)
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# Try to import mamba-ssm
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


class PatchEmbed(nn.Module):
    """
    Patch embedding layer for Vision Mamba.
    Converts 2D image into sequence of patch embeddings.
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 4,
        embed_dim: int = 768,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, num_patches, embed_dim)
        """
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        return x


class Mamba2DBlock(nn.Module):
    """
    Bidirectional Mamba block for 2D vision.
    Processes sequence in forward and backward directions.
    """
    
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
        if MAMBA_AVAILABLE:
            # Forward Mamba
            self.mamba_fwd = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            # Backward Mamba
            self.mamba_bwd = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            # Fallback: Use simple linear layers when mamba-ssm unavailable
            self.mamba_fwd = nn.Sequential(
                nn.Linear(dim, dim * expand),
                nn.GELU(),
                nn.Linear(dim * expand, dim),
            )
            self.mamba_bwd = nn.Sequential(
                nn.Linear(dim, dim * expand),
                nn.GELU(),
                nn.Linear(dim * expand, dim),
            )
        
        # Merge forward and backward
        self.merge = nn.Linear(dim * 2, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
        Returns:
            (B, L, D)
        """
        residual = x
        x = self.norm(x)
        
        # Forward direction
        x_fwd = self.mamba_fwd(x)
        
        # Backward direction
        x_bwd = self.mamba_bwd(x.flip(dims=[1])).flip(dims=[1])
        
        # Merge bidirectional features
        x = torch.cat([x_fwd, x_bwd], dim=-1)
        x = self.merge(x)
        x = self.dropout(x)
        
        return x + residual


class VisionMamba(nn.Module):
    """
    Vision Mamba (Vim) backbone.
    
    Replaces self-attention in Vision Transformers with bidirectional
    state-space models for linear complexity O(N) instead of O(N^2).
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 4,
        num_classes: int = 0,
        embed_dim: int = 768,
        depth: int = 12,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        drop_path: float = 0.1,
    ):
        """
        Initialize Vision Mamba.
        
        Args:
            img_size: Input image size
            patch_size: Patch size
            in_channels: Number of input channels
            num_classes: Number of output classes (0 = no classifier head)
            embed_dim: Embedding dimension
            depth: Number of Mamba blocks
            d_state: SSM state dimension
            d_conv: SSM convolution kernel size
            expand: SSM expansion factor
            dropout: Dropout rate
            drop_path: Stochastic depth rate
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        self.num_classes = num_classes
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Mamba blocks with drop path
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            Mamba2DBlock(
                dim=embed_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dpr[i],
            )
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        if num_classes > 0:
            self.head = nn.Linear(embed_dim, num_classes)
        else:
            self.head = nn.Identity()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features without classification head.
        
        Args:
            x: (B, C, H, W)
        Returns:
            (B, embed_dim)
        """
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 1 + num_patches, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply Mamba blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Return class token features
        return x[:, 0]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (B, C, H, W)
        Returns:
            (B, num_classes) if num_classes > 0, else (B, embed_dim)
        """
        x = self.forward_features(x)
        x = self.head(x)
        return x


class VimClassifier(nn.Module):
    """
    Vision Mamba classifier for brain tumor detection.
    Compatible with existing BrainTumorClassifier interface.
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        img_size: int = 224,
        variant: str = "base",
        pretrained: bool = False,
        dropout: float = 0.3,
        mode: str = "2.5d",
        num_slices: int = 16,
        aggregation: str = "attention",
    ):
        """
        Initialize Vim classifier.
        
        Args:
            in_channels: Number of input channels (MRI modalities)
            num_classes: Number of tumor classes
            img_size: Input image size
            variant: Model variant ("tiny", "small", "base")
            pretrained: Use pretrained weights (not available for Vim yet)
            dropout: Dropout rate
            mode: "2.5d" or "3d"
            num_slices: Number of slices for 2.5D mode
            aggregation: Slice aggregation method
        """
        super().__init__()
        
        # Model configurations
        configs = {
            "tiny": {"embed_dim": 192, "depth": 12, "d_state": 8},
            "small": {"embed_dim": 384, "depth": 12, "d_state": 16},
            "base": {"embed_dim": 768, "depth": 12, "d_state": 16},
        }
        
        if variant not in configs:
            raise ValueError(f"Unknown variant: {variant}. Choose from {list(configs.keys())}")
        
        config = configs[variant]
        
        self.mode = mode.lower()
        self.num_slices = num_slices
        self.feature_dim = config["embed_dim"]
        
        # Vision Mamba backbone
        self.backbone = VisionMamba(
            img_size=img_size,
            in_channels=in_channels,
            num_classes=0,
            embed_dim=config["embed_dim"],
            depth=config["depth"],
            d_state=config["d_state"],
            dropout=dropout,
        )
        
        if self.mode == "2.5d":
            # Slice aggregation
            self.aggregation = aggregation
            if aggregation == "attention":
                self.slice_attention = nn.Sequential(
                    nn.Linear(self.feature_dim, self.feature_dim // 4),
                    nn.Tanh(),
                    nn.Linear(self.feature_dim // 4, 1),
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
            # Pool across depth and extract features
            x = x.mean(dim=-1)  # (B, C, H, W)
            features = self.backbone(x)
        else:
            # 2.5D mode: process slices
            batch_size, num_slices, channels, height, width = x.shape
            
            # Reshape for batch processing
            x = x.view(batch_size * num_slices, channels, height, width)
            
            # Extract features for all slices
            features = self.backbone(x)  # (B*num_slices, feature_dim)
            
            # Reshape back
            features = features.view(batch_size, num_slices, -1)
            
            # Aggregate slices
            if self.aggregation == "attention":
                weights = self.slice_attention(features)
                weights = F.softmax(weights, dim=1)
                features = (features * weights).sum(dim=1)
            elif self.aggregation == "mean":
                features = features.mean(dim=1)
            elif self.aggregation == "max":
                features = features.max(dim=1)[0]
            else:
                raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        logits = self.classifier(features)
        return logits
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Factory function
def create_vim_classifier(
    variant: str = "base",
    **kwargs,
) -> VimClassifier:
    """
    Create a Vision Mamba classifier.
    
    Args:
        variant: Model variant ("tiny", "small", "base")
        **kwargs: Additional arguments passed to VimClassifier
        
    Returns:
        VimClassifier model
    """
    return VimClassifier(variant=variant, **kwargs)


if __name__ == "__main__":
    print("Testing Vision Mamba backbone...")
    print(f"Mamba SSM available: {MAMBA_AVAILABLE}")
    
    # Test VisionMamba backbone
    model = VisionMamba(
        img_size=224,
        in_channels=4,
        num_classes=0,
        embed_dim=192,
        depth=6,
    )
    print(f"VisionMamba parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    x = torch.randn(2, 4, 224, 224)
    with torch.no_grad():
        y = model(x)
    print(f"VisionMamba input: {x.shape} -> output: {y.shape}")
    
    # Test VimClassifier in 2.5D mode
    classifier = VimClassifier(
        in_channels=4,
        num_classes=4,
        variant="tiny",
        mode="2.5d",
        num_slices=16,
    )
    print(f"VimClassifier parameters: {classifier.get_num_parameters():,}")
    
    x_25d = torch.randn(2, 16, 4, 224, 224)
    with torch.no_grad():
        y_25d = classifier(x_25d)
    print(f"VimClassifier 2.5D input: {x_25d.shape} -> output: {y_25d.shape}")
    
    # Test 3D mode
    classifier_3d = VimClassifier(
        in_channels=4,
        num_classes=4,
        variant="tiny",
        mode="3d",
    )
    
    x_3d = torch.randn(2, 4, 224, 224, 16)
    with torch.no_grad():
        y_3d = classifier_3d(x_3d)
    print(f"VimClassifier 3D input: {x_3d.shape} -> output: {y_3d.shape}")
    
    print("Vision Mamba tests passed!")
