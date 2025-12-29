"""
Inference module for brain tumor analysis.

This module provides production-grade inference capabilities for:
- Brain tumor classification (Glioma, Meningioma, Pituitary, No Tumor)
- Brain tumor segmentation (Edema, Enhancing Tumor, Necrotic Core)
- Post-inference validation
- Explainability (Grad-CAM, Uncertainty Estimation)
- REST API endpoint
- CLI tools
"""

from inference.engine import (
    BrainTumorInferenceEngine,
    create_inference_engine,
    ClassificationResult,
    SegmentationResult,
    ValidationResult,
    InferenceReport,
    TumorClass,
    SegmentationClass,
)

from inference.xai import (
    GradCAM,
    MCDropoutUncertainty,
    TestTimeAugmentation,
    visualize_gradcam,
)

__all__ = [
    # Core inference
    "BrainTumorInferenceEngine",
    "create_inference_engine",
    "ClassificationResult",
    "SegmentationResult",
    "ValidationResult",
    "InferenceReport",
    "TumorClass",
    "SegmentationClass",
    # XAI
    "GradCAM",
    "MCDropoutUncertainty",
    "TestTimeAugmentation",
    "visualize_gradcam",
]
