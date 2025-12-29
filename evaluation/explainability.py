"""
Explainability methods for brain tumor models.
Includes GradCAM implementation for 2.5D models.
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from typing import Optional, Tuple, List, Union

class GradCAM:
    """
    GradCAM for 2.5D classification models.
    Handles multi-slice inputs by computing CAM for all slices.
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):
        self.model = model
        self.gradients = None
        self.activations = None
        
        # Auto-detect target layer if not provided
        if target_layer is None:
            target_layer = self._find_target_layer()
            
        print(f"GradCAM target layer: {target_layer}")
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def _find_target_layer(self) -> torch.nn.Module:
        """Attempt to find the last convolutional layer/stage."""
        # Check for encoder.backbone (our structure)
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "backbone"):
            backbone = self.model.encoder.backbone
            # ConvNeXt / Swin
            if hasattr(backbone, "stages"):
                return backbone.stages[-1]
            # ResNet / EfficientNet
            if hasattr(backbone, "layer4"):
                return backbone.layer4
            if hasattr(backbone, "blocks"):
                return backbone.blocks[-1]
            # Fallback to last child
            return list(backbone.children())[-1]
            
        return list(self.model.children())[-1]

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(
        self, 
        x: torch.Tensor, 
        class_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, int, int]:
        """
        Compute GradCAM.
        
        Args:
            x: Input tensor (B, num_slices, C, H, W). B should be 1.
            class_idx: Target class index. If None, uses predicted class.
            
        Returns:
            Tuple of:
            - heatmaps: (num_slices, H, W) numpy array
            - predicted_class: index of predicted class
            - best_slice_idx: index of slice with maximum activation
        """
        self.model.zero_grad()
        
        # Forward pass
        logits = self.model(x)
        
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
            
        # Backward pass
        logits[0, class_idx].backward()
        
        # Generator CAM
        # Gradients: (B*S, C', H', W')
        # Activations: (B*S, C', H', W')
        
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations not captured. Check hooks.")
            
        # Global average pooling on gradients
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)  # (B*S, C', 1, 1)
        
        # Weighted sum of activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (B*S, 1, H', W')
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize per slice
        # (avoid division by zero)
        B, S, C, H, W = x.shape
        
        # Upsample to input size
        cam = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)
        cam = cam.view(S, H, W).detach().cpu().numpy()
        
        # Normalize 0-1 globally or per slice?
        # Global normalization ensures we see relative importance of slices
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
            
        # Find best slice (slice with highest total activation)
        slice_scores = cam.sum(axis=(1, 2))
        best_slice_idx = np.argmax(slice_scores)
        
        return cam, class_idx, best_slice_idx

def overlay_cam_on_image(img: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlay CAM heatmap on image.
    
    Args:
        img: (H, W) or (H, W, 3) image, range 0-1
        mask: (H, W) heatmap, range 0-1
        alpha: Transparency
    
    Returns:
        (H, W, 3) RGB image with overlay
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1] # BGR to RGB
    
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    
    cam = heatmap * alpha + img * (1 - alpha)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
