"""
Reproducibility utilities for seeding random number generators.
Ensures deterministic behavior across training runs.
"""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set seed for reproducibility across all random number generators.
    
    Args:
        seed: Random seed value
        deterministic: If True, enforce deterministic algorithms in PyTorch
    """
    # Python random
    random.seed(seed)
    
    # Environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    if deterministic:
        # Enable deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # PyTorch 1.8+ deterministic flag
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True)
            except RuntimeError:
                # Some operations don't have deterministic implementations
                pass


def get_generator(seed: int) -> torch.Generator:
    """
    Create a PyTorch Generator with a specific seed.
    Useful for reproducible data loading.
    
    Args:
        seed: Seed for the generator
        
    Returns:
        Seeded PyTorch Generator
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def worker_init_fn(worker_id: int, base_seed: Optional[int] = None) -> None:
    """
    Worker initialization function for DataLoader.
    Ensures each worker has a unique but reproducible seed.
    
    Args:
        worker_id: Worker ID from DataLoader
        base_seed: Base seed (uses torch initial seed if not provided)
    """
    if base_seed is None:
        base_seed = torch.initial_seed() % 2**32
    
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
