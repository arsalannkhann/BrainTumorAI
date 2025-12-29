"""
Explainable AI (XAI) Module for Brain Tumor Inference

Provides interpretability tools:
- Grad-CAM for classification
- Attention visualization for segmentation
- Uncertainty estimation (MC Dropout / TTA)
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for CNN interpretability.
    
    Highlights regions that influenced the model's prediction.
    """
    
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        """
        Initialize Grad-CAM.
        
        Args:
            model: The classification model
            target_layer: Layer to compute CAM for (default: last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handles = []
        
        # Find target layer if not specified
        if target_layer is None:
            self._find_target_layer()
        
        self._register_hooks()
    
    def _find_target_layer(self):
        """Find the last convolutional layer in the model."""
        target = None
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                target = module
        self.target_layer = target
        
        if target is None:
            raise ValueError("Could not find a convolutional layer for Grad-CAM")
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.handles.append(
            self.target_layer.register_forward_hook(forward_hook)
        )
        self.handles.append(
            self.target_layer.register_full_backward_hook(backward_hook)
        )
    
    def remove_hooks(self):
        """Remove registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image (B, C, H, W) or (B, N, C, H, W) for 2.5D
            target_class: Class to visualize (default: predicted class)
            
        Returns:
            Tuple of (heatmap, predicted_class, confidence)
        """
        self.model.eval()
        input_tensor.requires_grad = True
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get predicted class if not specified
        probs = F.softmax(output, dim=1)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        confidence = probs[0, target_class].item()
        
        # Backward pass
        self.model.zero_grad()
        target = output[0, target_class]
        target.backward()
        
        # Compute Grad-CAM
        gradients = self.gradients
        activations = self.activations
        
        # Global average pooling of gradients
        if gradients.dim() == 4:  # 2D: (B, C, H, W)
            weights = gradients.mean(dim=(2, 3), keepdim=True)
        else:  # 3D: (B, C, H, W, D)
            weights = gradients.mean(dim=(2, 3, 4), keepdim=True)
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize to input size
        if cam.dim() == 4:
            cam = F.interpolate(
                cam,
                size=input_tensor.shape[-2:] if input_tensor.dim() == 4 else input_tensor.shape[-3:-1],
                mode='bilinear',
                align_corners=False
            )
        else:
            cam = F.interpolate(
                cam,
                size=input_tensor.shape[-3:],
                mode='trilinear',
                align_corners=False
            )
        
        # Normalize to 0-1
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, target_class, confidence
    
    def __del__(self):
        self.remove_hooks()


class MCDropoutUncertainty:
    """
    Monte Carlo Dropout for uncertainty estimation.
    
    Enables dropout during inference to estimate prediction uncertainty.
    """
    
    def __init__(self, model: nn.Module, num_samples: int = 20):
        """
        Initialize MC Dropout.
        
        Args:
            model: Model with dropout layers
            num_samples: Number of forward passes
        """
        self.model = model
        self.num_samples = num_samples
    
    def enable_dropout(self):
        """Enable dropout layers during inference."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        input_tensor: torch.Tensor,
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Make prediction with uncertainty estimation.
        
        Args:
            input_tensor: Input image tensor
            
        Returns:
            Dictionary with predictions and uncertainty metrics
        """
        self.model.eval()
        self.enable_dropout()
        
        predictions = []
        for _ in range(self.num_samples):
            output = self.model(input_tensor)
            probs = F.softmax(output, dim=1)
            predictions.append(probs.cpu().numpy())
        
        predictions = np.stack(predictions)  # (num_samples, B, num_classes)
        
        # Mean prediction
        mean_probs = predictions.mean(axis=0)[0]  # (num_classes,)
        
        # Epistemic uncertainty (model uncertainty)
        epistemic = predictions.var(axis=0).mean()
        
        # Aleatoric uncertainty approximation (entropy of mean)
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10))
        
        # Predictive uncertainty (total)
        predictive = epistemic + entropy
        
        return {
            "mean_probabilities": mean_probs,
            "predicted_class": int(mean_probs.argmax()),
            "confidence": float(mean_probs.max()),
            "epistemic_uncertainty": float(epistemic),
            "aleatoric_uncertainty": float(entropy),
            "total_uncertainty": float(predictive),
            "all_predictions": predictions,
        }


class TestTimeAugmentation:
    """
    Test-Time Augmentation for robust predictions.
    
    Applies augmentations during inference and aggregates predictions.
    """
    
    def __init__(
        self,
        model: nn.Module,
        augmentations: Optional[List[str]] = None,
    ):
        """
        Initialize TTA.
        
        Args:
            model: Model to use
            augmentations: List of augmentation types
        """
        self.model = model
        self.augmentations = augmentations or [
            "original",
            "hflip",
            "vflip",
            "rotate90",
            "rotate180",
            "rotate270",
        ]
    
    def _apply_augmentation(
        self,
        tensor: torch.Tensor,
        aug_type: str,
    ) -> torch.Tensor:
        """Apply augmentation to tensor."""
        if aug_type == "original":
            return tensor
        elif aug_type == "hflip":
            return torch.flip(tensor, dims=[-1])
        elif aug_type == "vflip":
            return torch.flip(tensor, dims=[-2])
        elif aug_type == "rotate90":
            return torch.rot90(tensor, k=1, dims=[-2, -1])
        elif aug_type == "rotate180":
            return torch.rot90(tensor, k=2, dims=[-2, -1])
        elif aug_type == "rotate270":
            return torch.rot90(tensor, k=3, dims=[-2, -1])
        else:
            return tensor
    
    @torch.no_grad()
    def predict(
        self,
        input_tensor: torch.Tensor,
        aggregate: str = "mean",
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Make TTA prediction.
        
        Args:
            input_tensor: Input image tensor
            aggregate: Aggregation method ("mean" or "max")
            
        Returns:
            Dictionary with aggregated prediction
        """
        self.model.eval()
        
        predictions = []
        for aug_type in self.augmentations:
            aug_tensor = self._apply_augmentation(input_tensor, aug_type)
            output = self.model(aug_tensor)
            probs = F.softmax(output, dim=1)
            predictions.append(probs.cpu().numpy())
        
        predictions = np.stack(predictions)  # (num_augs, B, num_classes)
        
        if aggregate == "mean":
            agg_probs = predictions.mean(axis=0)[0]
        else:  # max
            agg_probs = predictions.max(axis=0)[0]
        
        return {
            "probabilities": agg_probs,
            "predicted_class": int(agg_probs.argmax()),
            "confidence": float(agg_probs.max()),
            "num_augmentations": len(self.augmentations),
            "agreement": float((predictions.argmax(axis=-1) == agg_probs.argmax()).mean()),
        }


def visualize_gradcam(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on image.
    
    Args:
        image: Original image (H, W) or (H, W, C)
        heatmap: Grad-CAM heatmap (H, W)
        alpha: Overlay transparency
        
    Returns:
        Blended image with heatmap overlay
    """
    import cv2
    
    # Ensure image is 3-channel
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    
    # Normalize image
    if image.max() > 1.0:
        image = image / 255.0
    
    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Convert to colormap
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized),
        cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0
    
    # Blend
    blended = (1 - alpha) * image + alpha * heatmap_colored
    blended = np.clip(blended, 0, 1)
    
    return blended


if __name__ == "__main__":
    print("XAI Module Test")
    print("="*60)
    
    # Test with dummy model and input
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(32, 4)
            self.dropout = nn.Dropout(0.5)
        
        def forward(self, x):
            # Handle 2.5D input (B, N, C, H, W)
            if x.dim() == 5:
                x = x[:, 0]  # Take first slice
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x).flatten(1)
            x = self.dropout(x)
            return self.fc(x)
    
    model = DummyModel()
    dummy_input = torch.randn(1, 3, 64, 64)
    
    # Test Grad-CAM
    print("\n1. Testing Grad-CAM...")
    gradcam = GradCAM(model)
    heatmap, pred_class, conf = gradcam(dummy_input)
    print(f"   Heatmap shape: {heatmap.shape}")
    print(f"   Predicted class: {pred_class}, Confidence: {conf:.4f}")
    gradcam.remove_hooks()
    
    # Test MC Dropout
    print("\n2. Testing MC Dropout Uncertainty...")
    mc_dropout = MCDropoutUncertainty(model, num_samples=10)
    result = mc_dropout.predict_with_uncertainty(dummy_input)
    print(f"   Predicted class: {result['predicted_class']}")
    print(f"   Confidence: {result['confidence']:.4f}")
    print(f"   Epistemic uncertainty: {result['epistemic_uncertainty']:.4f}")
    print(f"   Total uncertainty: {result['total_uncertainty']:.4f}")
    
    # Test TTA
    print("\n3. Testing Test-Time Augmentation...")
    tta = TestTimeAugmentation(model)
    result = tta.predict(dummy_input)
    print(f"   Predicted class: {result['predicted_class']}")
    print(f"   Confidence: {result['confidence']:.4f}")
    print(f"   Agreement: {result['agreement']:.2%}")
    
    print("\n✓ All XAI tests passed!")
