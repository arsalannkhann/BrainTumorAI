"""
TransMIL: Transformer-based Correlated Multiple Instance Learning.

Replaces attention-based slice aggregation with self-attention over
instance features for improved patient-level classification.

Reference: "TransMIL: Transformer based Correlated Multiple Instance
           Learning for Whole Slide Image Classification" (NeurIPS 2021)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PPEG(nn.Module):
    """
    Pyramid Position Encoding Generator (PPEG).
    
    Generates positional encodings using multi-scale convolutions,
    which is more flexible than fixed sinusoidal encodings for
    variable-length sequences.
    """
    
    def __init__(
        self,
        dim: int,
    ):
        super().__init__()
        
        # Multi-scale convolutions for position encoding
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv2 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        
        self.norm = nn.LayerNorm(dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        spatial_shape: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) - sequence of instance features
            spatial_shape: (H, W) - optional spatial arrangement
            
        Returns:
            (B, N, D) - position-encoded features
        """
        B, N, D = x.shape
        
        # Determine spatial arrangement
        if spatial_shape is None:
            # Arrange instances in a square-ish grid
            H = int(math.sqrt(N))
            W = (N + H - 1) // H
        else:
            H, W = spatial_shape
        
        # Pad if necessary
        padded_N = H * W
        if padded_N > N:
            pad = torch.zeros(B, padded_N - N, D, device=x.device, dtype=x.dtype)
            x_padded = torch.cat([x, pad], dim=1)
        else:
            x_padded = x[:, :padded_N]
        
        # Reshape to 2D grid
        x_2d = rearrange(x_padded, 'b (h w) d -> b d h w', h=H, w=W)
        
        # Multi-scale position encoding
        pos1 = self.conv1(x_2d)
        pos2 = self.conv2(x_2d)
        pos3 = self.conv3(x_2d)
        pos = pos1 + pos2 + pos3
        
        # Reshape back
        pos = rearrange(pos, 'b d h w -> b (h w) d')
        
        # Only take original positions
        pos = pos[:, :N]
        
        # Add to input
        x = x + pos
        x = self.norm(x)
        
        return x


class NystromAttention(nn.Module):
    """
    Nyström-based efficient attention.
    
    Approximates self-attention with O(N) complexity using
    landmark-based approximation.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_landmarks: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_landmarks = num_landmarks
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D)
        Returns:
            (B, N, D)
        """
        B, N, D = x.shape
        
        # Standard attention for small sequences
        if N <= self.num_landmarks * 2:
            return self._standard_attention(x)
        
        # Nyström approximation for large sequences
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, heads, N, head_dim)
        
        # Select landmarks via uniform sampling
        indices = torch.linspace(0, N - 1, self.num_landmarks, dtype=torch.long, device=x.device)
        k_landmarks = k[:, :, indices]  # (B, heads, m, head_dim)
        q_landmarks = q[:, :, indices]  # (B, heads, m, head_dim)
        
        # Kernel computation
        kernel_1 = F.softmax(q @ k_landmarks.transpose(-2, -1) * self.scale, dim=-1)
        kernel_2 = F.softmax(q_landmarks @ k_landmarks.transpose(-2, -1) * self.scale, dim=-1)
        kernel_3 = F.softmax(q_landmarks @ k.transpose(-2, -1) * self.scale, dim=-1)
        
        # Nyström approximation
        kernel_2_inv = torch.linalg.pinv(kernel_2 + 1e-6 * torch.eye(
            self.num_landmarks, device=x.device, dtype=x.dtype
        ).unsqueeze(0).unsqueeze(0))
        
        attn = kernel_1 @ kernel_2_inv @ kernel_3
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        
        return out
    
    def _standard_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Standard scaled dot-product attention."""
        B, N, D = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        
        return out


class TransMILBlock(nn.Module):
    """
    TransMIL Transformer block with PPEG and attention.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_nystrom: bool = True,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        if use_nystrom:
            self.attn = NystromAttention(dim, num_heads, dropout=dropout)
        else:
            self.attn = nn.MultiheadAttention(
                dim, num_heads, dropout=dropout, batch_first=True
            )
            self._use_mha = True
        
        self._use_nystrom = use_nystrom
        
        # MLP
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )
        
        # PPEG for position encoding
        self.ppeg = PPEG(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D)
        Returns:
            (B, N, D)
        """
        # Add positional encoding
        x = self.ppeg(x)
        
        # Self-attention
        if self._use_nystrom:
            x = x + self.attn(self.norm1(x))
        else:
            x_norm = self.norm1(x)
            attn_out, _ = self.attn(x_norm, x_norm, x_norm)
            x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x


class TransMILAggregator(nn.Module):
    """
    TransMIL Aggregator for Multiple Instance Learning.
    
    Aggregates instance-level features (slices) into a patient-level
    representation using transformer self-attention.
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_classes: int = 4,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_nystrom: bool = True,
    ):
        """
        Initialize TransMIL aggregator.
        
        Args:
            feature_dim: Dimension of instance features
            num_classes: Number of output classes
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_nystrom: Use Nyström attention for efficiency
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feature_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransMILBlock(
                dim=feature_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_nystrom=use_nystrom,
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(feature_dim)
        
        # Classification head
        self.head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_classes),
        )
        
        # Initialize
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_features: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (B, N, D) - instance features
            return_features: If True, also return aggregated features
            
        Returns:
            Class logits (B, num_classes)
            If return_features: (logits, features)
        """
        B, N, D = x.shape
        
        # Prepend class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Extract class token
        cls_features = x[:, 0]  # (B, D)
        
        # Classification
        logits = self.head(cls_features)
        
        if return_features:
            return logits, cls_features
        return logits


class TransMILEncoder(nn.Module):
    """
    Full TransMIL encoder that processes slices and aggregates.
    
    Drop-in replacement for MultiSliceEncoder in classification/model.py
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int = 4,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        """
        Initialize TransMIL encoder.
        
        Args:
            backbone: 2D CNN backbone for feature extraction
            feature_dim: Feature dimension from backbone
            num_classes: Number of output classes
            num_layers: Number of TransMIL layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.backbone = backbone
        self.feature_dim = feature_dim
        
        self.aggregator = TransMILAggregator(
            feature_dim=feature_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (B, num_slices, C, H, W) - multi-slice input
            
        Returns:
            Class logits (B, num_classes)
        """
        batch_size, num_slices, channels, height, width = x.shape
        
        # Reshape for batch processing
        x = x.view(batch_size * num_slices, channels, height, width)
        
        # Extract features for all slices
        features = self.backbone(x)  # (B*num_slices, feature_dim)
        
        # Reshape back
        features = features.view(batch_size, num_slices, -1)
        
        # Aggregate with TransMIL
        logits = self.aggregator(features)
        
        return logits


def create_transmil_aggregator(
    feature_dim: int,
    num_classes: int = 4,
    **kwargs,
) -> TransMILAggregator:
    """
    Create a TransMIL aggregator.
    
    Args:
        feature_dim: Feature dimension
        num_classes: Number of classes
        **kwargs: Additional arguments
        
    Returns:
        TransMILAggregator
    """
    return TransMILAggregator(
        feature_dim=feature_dim,
        num_classes=num_classes,
        **kwargs,
    )


if __name__ == "__main__":
    print("Testing TransMIL...")
    
    # Test PPEG
    ppeg = PPEG(dim=512)
    x = torch.randn(2, 16, 512)
    y = ppeg(x)
    print(f"PPEG input: {x.shape} -> output: {y.shape}")
    
    # Test NystromAttention
    attn = NystromAttention(dim=512, num_heads=8)
    y = attn(x)
    print(f"NystromAttention input: {x.shape} -> output: {y.shape}")
    
    # Test TransMILAggregator
    aggregator = TransMILAggregator(
        feature_dim=512,
        num_classes=4,
        num_layers=2,
    )
    print(f"TransMIL parameters: {sum(p.numel() for p in aggregator.parameters()):,}")
    
    # Test with varying sequence lengths
    for num_instances in [16, 64, 128]:
        x = torch.randn(2, num_instances, 512)
        with torch.no_grad():
            logits = aggregator(x)
        print(f"TransMIL {num_instances} instances: input {x.shape} -> output {logits.shape}")
    
    # Test with feature return
    x = torch.randn(2, 32, 512)
    with torch.no_grad():
        logits, features = aggregator(x, return_features=True)
    print(f"TransMIL with features: logits {logits.shape}, features {features.shape}")
    
    print("TransMIL tests passed!")
