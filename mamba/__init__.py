"""
Mamba State Space Models for Vision.
Provides Vision Mamba (Vim) backbone and SegMamba for 3D segmentation.
"""

from typing import Optional

# Optional mamba-ssm import
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    Mamba = None

__all__ = [
    "MAMBA_AVAILABLE",
    "Mamba",
]
