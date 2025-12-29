"""
SegMamba: State Space Model for 3D Medical Image Segmentation.

Implements Tri-Oriented Mamba (ToM) for efficient long-range dependency
modeling in 3D volumetric data with linear O(N) complexity.

Reference: "SegMamba: Long-range Sequential Modeling Mamba For 3D
           Medical Image Segmentation" (MICCAI 2024)
"""

import math
from typing import Optional, Tuple, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Try to import mamba-ssm
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


class MambaLayer(nn.Module):
    """
    Single Mamba layer with optional fallback.
    """
    
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        
        if MAMBA_AVAILABLE:
            self.mamba = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            # Fallback: simple MLP
            self.mamba = nn.Sequential(
                nn.Linear(dim, dim * expand),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(dim * expand, dim),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mamba(x)


class TriOrientedMamba(nn.Module):
    """
    Tri-Oriented Mamba (ToM) block.
    
    Processes 3D volume along three orthogonal directions:
    - Axial (depth/z-axis)
    - Coronal (height/y-axis)  
    - Sagittal (width/x-axis)
    
    This captures long-range dependencies across the entire volume
    with linear complexity.
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
        
        self.norm = nn.LayerNorm(dim)
        
        # Mamba for each orientation
        self.mamba_axial = MambaLayer(dim, d_state, d_conv, expand)
        self.mamba_coronal = MambaLayer(dim, d_state, d_conv, expand)
        self.mamba_sagittal = MambaLayer(dim, d_state, d_conv, expand)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Final projection
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W) - 3D feature volume
        Returns:
            (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape
        residual = x
        
        # Reshape for layer norm: (B, D, H, W, C)
        x = rearrange(x, 'b c d h w -> b d h w c')
        x = self.norm(x)
        
        # Axial: process along D dimension
        # (B, D, H, W, C) -> (B*H*W, D, C)
        x_axial = rearrange(x, 'b d h w c -> (b h w) d c')
        x_axial = self.mamba_axial(x_axial)
        x_axial = rearrange(x_axial, '(b h w) d c -> b d h w c', b=B, h=H, w=W)
        
        # Coronal: process along H dimension
        # (B, D, H, W, C) -> (B*D*W, H, C)
        x_coronal = rearrange(x, 'b d h w c -> (b d w) h c')
        x_coronal = self.mamba_coronal(x_coronal)
        x_coronal = rearrange(x_coronal, '(b d w) h c -> b d h w c', b=B, d=D, w=W)
        
        # Sagittal: process along W dimension
        # (B, D, H, W, C) -> (B*D*H, W, C)
        x_sagittal = rearrange(x, 'b d h w c -> (b d h) w c')
        x_sagittal = self.mamba_sagittal(x_sagittal)
        x_sagittal = rearrange(x_sagittal, '(b d h) w c -> b d h w c', b=B, d=D, h=H)
        
        # Fuse three orientations
        x = torch.cat([x_axial, x_coronal, x_sagittal], dim=-1)
        x = self.fusion(x)
        x = self.proj(x)
        x = self.dropout(x)
        
        # Reshape back: (B, D, H, W, C) -> (B, C, D, H, W)
        x = rearrange(x, 'b d h w c -> b c d h w')
        
        return x + residual


class MambaEncoderBlock(nn.Module):
    """
    Encoder block with Tri-Oriented Mamba and convolution.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        d_state: int = 16,
        downsample: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.downsample = downsample
        
        # Convolution block
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        
        # Tri-Oriented Mamba
        self.mamba = TriOrientedMamba(
            dim=out_channels,
            d_state=d_state,
            dropout=dropout,
        )
        
        # Downsampling
        if downsample:
            self.down = nn.Conv3d(out_channels, out_channels, 2, stride=2)
        
        # Skip connection if channels change
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C_in, D, H, W)
        Returns:
            Tuple of (downsampled output, skip connection)
        """
        # Convolution
        residual = self.skip(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        x = x + residual
        
        # Mamba
        x = self.mamba(x)
        
        # Store skip connection
        skip = x
        
        # Downsample if needed
        if self.downsample:
            x = self.down(x)
        
        return x, skip


class MambaDecoderBlock(nn.Module):
    """
    Decoder block with upsampling and skip connection.
    """
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        upsample: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.upsample = upsample
        
        # Upsampling
        if upsample:
            self.up = nn.ConvTranspose3d(in_channels, in_channels, 2, stride=2)
        
        # Convolution
        self.conv1 = nn.Conv3d(in_channels + skip_channels, out_channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout3d(dropout)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, D, H, W)
            skip: Skip connection from encoder
        Returns:
            (B, C_out, D', H', W')
        """
        # Upsample
        if self.upsample:
            x = self.up(x)
        
        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Convolution
        x = self.act(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.act(self.norm2(self.conv2(x)))
        
        return x


class SegMamba(nn.Module):
    """
    SegMamba: U-shaped architecture with Tri-Oriented Mamba.
    
    Combines the efficiency of U-Net skip connections with the
    global context modeling of State Space Models.
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        channels: Sequence[int] = (32, 64, 128, 256, 512),
        d_state: int = 16,
        dropout: float = 0.1,
        **kwargs,
    ):
        """
        Initialize SegMamba.
        
        Args:
            in_channels: Number of input channels (MRI modalities)
            out_channels: Number of output classes
            channels: Feature channels at each level
            d_state: State dimension for Mamba
            dropout: Dropout rate
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        
        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv3d(in_channels, channels[0], 3, padding=1),
            nn.InstanceNorm3d(channels[0]),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Encoder
        self.encoders = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.encoders.append(
                MambaEncoderBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    d_state=d_state,
                    downsample=(i < len(channels) - 2),  # No downsample on last
                    dropout=dropout,
                )
            )
        
        # Bottleneck
        self.bottleneck = TriOrientedMamba(
            dim=channels[-1],
            d_state=d_state,
            dropout=dropout,
        )
        
        # Decoder - match channels to encoder skip outputs
        # Encoder skips (after init_conv): [channels[0], channels[1], channels[2], channels[3]]
        # so decoder needs to match these from bottom up
        self.decoders = nn.ModuleList()
        
        # Number of decoder stages = len(channels) - 1 (same as encoders)
        # Decoder[0] receives bottleneck output + skip from last encoder
        # For channels [16, 32, 64, 128]:
        #   encoders produce skips with [32, 64, 128] channels (index 1 to N-1)
        #   decoder[0]: in=128, skip=128, out=64
        #   decoder[1]: in=64, skip=64, out=32
        #   decoder[2]: in=32, skip=32, out=16
        
        num_decoders = len(channels) - 1
        for i in range(num_decoders):
            # i=0 is top of decoder (receives bottleneck)
            # Calculate indices into channels list
            decoder_level = len(channels) - 2 - i  # Goes from len-2 down to 0
            
            in_ch = channels[decoder_level + 1]  # Input from previous decoder/bottleneck
            skip_ch = channels[decoder_level + 1]  # Skip from encoder at same level
            out_ch = channels[decoder_level]  # Output channels
            
            # Upsample for all but first decoder (which receives non-downsampled bottleneck)
            needs_upsample = (decoder_level < len(channels) - 2)
            
            self.decoders.append(
                MambaDecoderBlock(
                    in_channels=in_ch,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                    upsample=needs_upsample,
                    dropout=dropout,
                )
            )
        
        # Output
        self.output = nn.Conv3d(channels[0], out_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (B, in_channels, D, H, W)
        Returns:
            (B, out_channels, D, H, W)
        """
        # Initial convolution
        x = self.init_conv(x)
        
        # Encoder with skip connections
        # For channels [16, 32, 64, 128] with 3 encoders:
        #   encoder[0]: 16 -> 32, skip at 32 channels, downsample
        #   encoder[1]: 32 -> 64, skip at 64 channels, downsample  
        #   encoder[2]: 64 -> 128, skip at 128 channels, NO downsample
        # Skips stored: [skip@32, skip@64, skip@128]
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)
        
        # Bottleneck - operates on encoder output (no further downsampling)
        x = self.bottleneck(x)
        
        # Decoder - reverse order of skips
        # decoder[0]: receives x@128 + skip@128 -> 64
        # decoder[1]: receives x@64 + skip@64 -> 32
        # decoder[2]: receives x@32 + skip@32 -> 16
        skips = skips[::-1]  # Reverse to match decoder order
        for decoder, skip in zip(self.decoders, skips):
            x = decoder(x, skip)
        
        # Output
        x = self.output(x)
        
        return x
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def create_segmamba(
    in_channels: int = 4,
    out_channels: int = 4,
    **kwargs,
) -> SegMamba:
    """
    Create a SegMamba model.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output classes
        **kwargs: Additional arguments
        
    Returns:
        SegMamba model
    """
    return SegMamba(
        in_channels=in_channels,
        out_channels=out_channels,
        **kwargs,
    )


if __name__ == "__main__":
    print("Testing SegMamba...")
    print(f"Mamba SSM available: {MAMBA_AVAILABLE}")
    
    # Test with small input
    model = SegMamba(
        in_channels=4,
        out_channels=4,
        channels=[16, 32, 64, 128],
        d_state=8,
    )
    print(f"SegMamba parameters: {model.get_num_parameters():,}")
    
    # Test forward pass with smaller size for memory
    x = torch.randn(1, 4, 32, 32, 32)
    with torch.no_grad():
        y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Verify shapes match
    assert y.shape == (1, 4, 32, 32, 32), f"Shape mismatch: {y.shape}"
    
    print("SegMamba tests passed!")
