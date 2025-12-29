"""
BM-MAE: Brain Multimodal Masked Autoencoder.

Self-supervised pretraining for 3D MRI with modality-invariant
representations and robust handling of missing modalities.

Reference: "Multimodal Masked Autoencoder Pre-training for 3D MRI-Based
           Brain Tumor Analysis with Missing Modalities" (MICCAI 2024)
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class PatchEmbed3D(nn.Module):
    """
    3D Patch embedding for volumetric data.
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int, int] = (128, 128, 128),
        patch_size: Tuple[int, int, int] = (16, 16, 16),
        in_channels: int = 4,
        embed_dim: int = 768,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (
            (img_size[0] // patch_size[0]) *
            (img_size[1] // patch_size[1]) *
            (img_size[2] // patch_size[2])
        )
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        )
        
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W)
        Returns:
            (B, num_patches, embed_dim)
        """
        x = self.proj(x)  # (B, embed_dim, D', H', W')
        x = rearrange(x, 'b c d h w -> b (d h w) c')
        x = self.norm(x)
        return x


class TransformerBlock(nn.Module):
    """
    Standard Transformer block with self-attention and MLP.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class MAEEncoder(nn.Module):
    """
    Masked Autoencoder Encoder.
    Processes only visible (non-masked) patches.
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int, int] = (128, 128, 128),
        patch_size: Tuple[int, int, int] = (16, 16, 16),
        in_channels: int = 4,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with masking.
        
        Args:
            x: (B, C, D, H, W) - input volume
            mask: (B, num_patches) - binary mask (1 = keep, 0 = remove)
            
        Returns:
            Tuple of (encoded visible patches, visible indices)
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Apply mask - keep only visible patches
        B, N, D = x.shape
        visible_indices = []
        visible_patches = []
        
        for i in range(B):
            vis_idx = mask[i].nonzero(as_tuple=False).squeeze(-1)
            visible_indices.append(vis_idx)
            visible_patches.append(x[i, vis_idx])
        
        # Pad to same length for batching
        max_vis = max(len(idx) for idx in visible_indices)
        x_vis = torch.zeros(B, max_vis, D, device=x.device, dtype=x.dtype)
        for i in range(B):
            x_vis[i, :len(visible_indices[i])] = visible_patches[i]
        
        # Transformer blocks
        for block in self.blocks:
            x_vis = block(x_vis)
        
        x_vis = self.norm(x_vis)
        
        return x_vis, visible_indices


class MAEDecoder(nn.Module):
    """
    Masked Autoencoder Decoder.
    Reconstructs full volume from visible patches.
    """
    
    def __init__(
        self,
        num_patches: int,
        patch_size: Tuple[int, int, int] = (16, 16, 16),
        in_channels: int = 4,
        embed_dim: int = 768,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.in_channels = in_channels
        
        # Project encoder output to decoder dimension
        self.enc_to_dec = nn.Linear(embed_dim, decoder_embed_dim)
        
        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # Positional embedding for decoder
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(decoder_depth)
        ])
        
        self.norm = nn.LayerNorm(decoder_embed_dim)
        
        # Prediction head - predict patch pixels
        patch_pixels = patch_size[0] * patch_size[1] * patch_size[2] * in_channels
        self.pred = nn.Linear(decoder_embed_dim, patch_pixels)
        
        # Initialize
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(
        self,
        x_vis: torch.Tensor,
        visible_indices: List[torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x_vis: (B, num_visible, embed_dim) - encoded visible patches
            visible_indices: List of visible indices per sample
            mask: (B, num_patches) - original mask
            
        Returns:
            (B, num_patches, patch_pixels) - reconstructed patches
        """
        B = x_vis.shape[0]
        
        # Project to decoder dimension
        x_vis = self.enc_to_dec(x_vis)
        
        # Create full sequence with mask tokens
        x = self.mask_token.expand(B, self.num_patches, -1).clone()
        
        # Insert visible patches
        for i in range(B):
            vis_idx = visible_indices[i]
            x[i, vis_idx] = x_vis[i, :len(vis_idx)]
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Predict pixels
        pred = self.pred(x)
        
        return pred


class BM_MAE(nn.Module):
    """
    Brain Multimodal Masked Autoencoder.
    
    Self-supervised pretraining with:
    - High mask ratio (75%) for efficient pretraining
    - Modality dropout for robust multimodal learning
    - Patch-wise reconstruction loss
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int, int] = (128, 128, 128),
        patch_size: Tuple[int, int, int] = (16, 16, 16),
        in_channels: int = 4,
        embed_dim: int = 768,
        encoder_depth: int = 12,
        encoder_num_heads: int = 12,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 8,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.75,
        modality_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        """
        Initialize BM-MAE.
        
        Args:
            img_size: Input volume size (D, H, W)
            patch_size: Patch size (d, h, w)
            in_channels: Number of modalities
            embed_dim: Encoder embedding dimension
            encoder_depth: Number of encoder layers
            encoder_num_heads: Encoder attention heads
            decoder_embed_dim: Decoder embedding dimension
            decoder_depth: Number of decoder layers
            decoder_num_heads: Decoder attention heads
            mlp_ratio: MLP expansion ratio
            mask_ratio: Ratio of patches to mask
            modality_dropout: Probability of dropping a modality
            dropout: Dropout rate
        """
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.mask_ratio = mask_ratio
        self.modality_dropout = modality_dropout
        
        # Encoder
        self.encoder = MAEEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        
        num_patches = self.encoder.patch_embed.num_patches
        
        # Decoder
        self.decoder = MAEDecoder(
            num_patches=num_patches,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
    
    def random_masking(
        self, 
        batch_size: int, 
        num_patches: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generate random mask.
        
        Args:
            batch_size: Batch size
            num_patches: Number of patches
            device: Device
            
        Returns:
            (B, num_patches) - binary mask (1 = keep, 0 = remove)
        """
        num_keep = int(num_patches * (1 - self.mask_ratio))
        
        # Random noise for shuffling
        noise = torch.rand(batch_size, num_patches, device=device)
        
        # Sort and keep top-k
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :num_keep]
        
        # Create binary mask
        mask = torch.zeros(batch_size, num_patches, device=device)
        mask.scatter_(1, ids_keep, 1)
        
        return mask
    
    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert volume to patches.
        
        Args:
            x: (B, C, D, H, W)
        Returns:
            (B, num_patches, patch_pixels)
        """
        p = self.patch_size
        d, h, w = self.img_size
        
        x = rearrange(
            x,
            'b c (nd pd) (nh ph) (nw pw) -> b (nd nh nw) (pd ph pw c)',
            pd=p[0], ph=p[1], pw=p[2],
        )
        return x
    
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert patches back to volume.
        
        Args:
            x: (B, num_patches, patch_pixels)
        Returns:
            (B, C, D, H, W)
        """
        p = self.patch_size
        d, h, w = self.img_size
        nd, nh, nw = d // p[0], h // p[1], w // p[2]
        
        x = rearrange(
            x,
            'b (nd nh nw) (pd ph pw c) -> b c (nd pd) (nh ph) (nw pw)',
            nd=nd, nh=nh, nw=nw,
            pd=p[0], ph=p[1], pw=p[2],
            c=self.in_channels,
        )
        return x
    
    def apply_modality_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """
        Randomly drop modalities during training.
        
        Args:
            x: (B, C, D, H, W)
        Returns:
            (B, C, D, H, W) with some channels zeroed
        """
        if self.modality_dropout > 0 and self.training:
            B, C = x.shape[:2]
            # Random mask for each modality
            drop_mask = torch.rand(B, C, 1, 1, 1, device=x.device) > self.modality_dropout
            x = x * drop_mask.float()
        return x
    
    def forward(
        self, 
        x: torch.Tensor,
        return_reconstruction: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (B, C, D, H, W) - input volume
            return_reconstruction: If True, return reconstructed volume
            
        Returns:
            Reconstruction loss, or (loss, reconstruction) if return_reconstruction
        """
        # Apply modality dropout
        x = self.apply_modality_dropout(x)
        
        B = x.shape[0]
        num_patches = self.encoder.patch_embed.num_patches
        
        # Random masking
        mask = self.random_masking(B, num_patches, x.device)
        
        # Encode visible patches
        x_vis, visible_indices = self.encoder(x, mask)
        
        # Decode all patches
        pred = self.decoder(x_vis, visible_indices, mask)
        
        # Compute loss on masked patches only
        target = self.patchify(x)
        
        # Mask for loss computation (1 = masked = compute loss)
        loss_mask = 1 - mask
        
        # MSE loss
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # Mean over patch pixels
        loss = (loss * loss_mask).sum() / (loss_mask.sum() + 1e-6)
        
        if return_reconstruction:
            recon = self.unpatchify(pred)
            return loss, recon
        
        return loss
    
    def get_encoder(self) -> MAEEncoder:
        """Get encoder for downstream tasks."""
        return self.encoder
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def create_bm_mae(
    img_size: Tuple[int, int, int] = (128, 128, 128),
    variant: str = "base",
    **kwargs,
) -> BM_MAE:
    """
    Create a BM-MAE model.
    
    Args:
        img_size: Volume size
        variant: Model variant ("small", "base", "large")
        **kwargs: Additional arguments
        
    Returns:
        BM_MAE model
    """
    configs = {
        "small": {
            "embed_dim": 384,
            "encoder_depth": 6,
            "encoder_num_heads": 6,
            "decoder_embed_dim": 256,
            "decoder_depth": 4,
            "decoder_num_heads": 4,
        },
        "base": {
            "embed_dim": 768,
            "encoder_depth": 12,
            "encoder_num_heads": 12,
            "decoder_embed_dim": 512,
            "decoder_depth": 8,
            "decoder_num_heads": 8,
        },
        "large": {
            "embed_dim": 1024,
            "encoder_depth": 24,
            "encoder_num_heads": 16,
            "decoder_embed_dim": 512,
            "decoder_depth": 8,
            "decoder_num_heads": 8,
        },
    }
    
    config = configs.get(variant, configs["base"])
    config.update(kwargs)
    
    return BM_MAE(img_size=img_size, **config)


if __name__ == "__main__":
    print("Testing BM-MAE...")
    
    # Test with smaller size for memory
    model = create_bm_mae(
        img_size=(64, 64, 64),
        variant="small",
        patch_size=(16, 16, 16),
        in_channels=4,
        mask_ratio=0.75,
        modality_dropout=0.1,
    )
    print(f"BM-MAE parameters: {model.get_num_parameters():,}")
    
    # Test forward pass
    x = torch.randn(2, 4, 64, 64, 64)
    with torch.no_grad():
        loss = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction loss: {loss.item():.4f}")
    
    # Test with reconstruction
    with torch.no_grad():
        loss, recon = model(x, return_reconstruction=True)
    print(f"Reconstructed shape: {recon.shape}")
    
    print("BM-MAE tests passed!")
