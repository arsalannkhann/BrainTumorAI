"""
3D UNet model for brain tumor segmentation.
Uses MONAI's implementation with configurable architecture.
Extended with SegMamba for state-space model segmentation.
"""

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from monai.networks.nets import UNet, UNETR, SwinUNETR

# Optional SegMamba import
try:
    from mamba.segmamba import SegMamba, create_segmamba
    SEGMAMBA_AVAILABLE = True
except ImportError:
    SEGMAMBA_AVAILABLE = False


def create_unet(
    spatial_dims: int = 3,
    in_channels: int = 4,
    out_channels: int = 4,
    channels: Sequence[int] = (32, 64, 128, 256, 512),
    strides: Sequence[int] = (2, 2, 2, 2),
    num_res_units: int = 2,
    dropout: float = 0.1,
    norm: str = "instance",
    **kwargs,
) -> nn.Module:
    """
    Create a MONAI 3D UNet for segmentation.
    
    This is an nnU-Net-style architecture with residual units.
    
    Args:
        spatial_dims: Number of spatial dimensions (2 or 3)
        in_channels: Number of input channels (modalities)
        out_channels: Number of output channels (classes)
        channels: Number of channels at each level
        strides: Downsampling strides at each level
        num_res_units: Number of residual units per level
        dropout: Dropout probability
        norm: Normalization type ("batch", "instance", "group")
        
    Returns:
        MONAI UNet model
    """
    return UNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        dropout=dropout,
        norm=norm,
    )


def create_unetr(
    spatial_dims: int = 3,
    in_channels: int = 4,
    out_channels: int = 4,
    img_size: Tuple[int, int, int] = (128, 128, 128),
    feature_size: int = 16,
    hidden_size: int = 768,
    mlp_dim: int = 3072,
    num_heads: int = 12,
    dropout_rate: float = 0.0,
    **kwargs,
) -> nn.Module:
    """
    Create a UNETR (U-Net Transformer) model.
    
    Combines Vision Transformer encoder with U-Net decoder.
    Good for capturing global context but more memory-intensive.
    
    Args:
        spatial_dims: Number of spatial dimensions
        in_channels: Number of input channels
        out_channels: Number of output channels
        img_size: Input image size
        feature_size: Feature size in decoder
        hidden_size: Transformer hidden size
        mlp_dim: MLP dimension in Transformer
        num_heads: Number of attention heads
        dropout_rate: Dropout rate
        
    Returns:
        UNETR model
    """
    return UNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        img_size=img_size,
        feature_size=feature_size,
        hidden_size=hidden_size,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        pos_embed="conv",
        norm_name="instance",
        res_block=True,
        dropout_rate=dropout_rate,
        spatial_dims=spatial_dims,
    )


def create_swin_unetr(
    spatial_dims: int = 3,
    in_channels: int = 4,
    out_channels: int = 4,
    img_size: Tuple[int, int, int] = (128, 128, 128),
    feature_size: int = 48,
    depths: Tuple[int, ...] = (2, 2, 2, 2),
    num_heads: Tuple[int, ...] = (3, 6, 12, 24),
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    use_checkpoint: bool = True,
    **kwargs,
) -> nn.Module:
    """
    Create a Swin UNETR model.
    
    State-of-the-art architecture using Swin Transformer encoder.
    Achieves top performance on BraTS but requires significant GPU memory.
    
    Args:
        spatial_dims: Number of spatial dimensions
        in_channels: Number of input channels
        out_channels: Number of output channels
        img_size: Input image size
        feature_size: Base feature size
        depths: Number of blocks at each stage
        num_heads: Number of attention heads at each stage
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        use_checkpoint: Use gradient checkpointing to save memory
        
    Returns:
        Swin UNETR model
    """
    return SwinUNETR(
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        depths=depths,
        num_heads=num_heads,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        use_checkpoint=use_checkpoint,
        spatial_dims=spatial_dims,
    )


class SegmentationModel(nn.Module):
    """
    Wrapper for segmentation models with standardized interface.
    """
    
    def __init__(
        self,
        model_name: str = "UNet",
        in_channels: int = 4,
        out_channels: int = 4,
        **kwargs,
    ):
        """
        Initialize segmentation model.
        
        Args:
            model_name: Model architecture ("UNet", "UNETR", "SwinUNETR")
            in_channels: Number of input channels
            out_channels: Number of output channels
            **kwargs: Additional model-specific arguments
        """
        super().__init__()
        
        self.model_name = model_name
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Create backbone
        if model_name.lower() == "unet":
            self.backbone = create_unet(
                in_channels=in_channels,
                out_channels=out_channels,
                **kwargs,
            )
        elif model_name.lower() == "unetr":
            self.backbone = create_unetr(
                in_channels=in_channels,
                out_channels=out_channels,
                **kwargs,
            )
        elif model_name.lower() == "swinunetr":
            self.backbone = create_swin_unetr(
                in_channels=in_channels,
                out_channels=out_channels,
                **kwargs,
            )
        elif model_name.lower() == "segmamba":
            if not SEGMAMBA_AVAILABLE:
                raise ImportError(
                    "SegMamba not available. Check mamba/ directory for segmamba.py"
                )
            self.backbone = create_segmamba(
                in_channels=in_channels,
                out_channels=out_channels,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown model: {model_name}. Options: UNet, UNETR, SwinUNETR, SegMamba")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W, D)
            
        Returns:
            Logits tensor (B, num_classes, H, W, D)
        """
        return self.backbone(x)
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str = "cuda",
    **model_kwargs,
) -> SegmentationModel:
    """
    Load a segmentation model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model to
        **model_kwargs: Model configuration
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get model config from checkpoint if available
    if "model_config" in checkpoint:
        model_kwargs.update(checkpoint["model_config"])
    
    model = SegmentationModel(**model_kwargs)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")
    
    # Test UNet
    model = SegmentationModel(
        model_name="UNet",
        in_channels=4,
        out_channels=4,
        channels=[32, 64, 128, 256],
        strides=[2, 2, 2],
    )
    print(f"UNet parameters: {model.get_num_parameters():,}")
    
    # Test forward pass
    x = torch.randn(1, 4, 128, 128, 128)
    with torch.no_grad():
        y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    print("Model creation test passed!")
