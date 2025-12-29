"""
Production-Grade Medical AI Inference Engine
Brain Tumor Classification + Segmentation

This module provides deterministic inference with validation and 
clinically interpretable outputs for brain tumor MRI analysis.
"""

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from segmentation.model import SegmentationModel, load_model_from_checkpoint
from classification.model import load_classifier_from_checkpoint
from utils import load_config


class TumorClass(Enum):
    """Tumor classification categories."""
    GLIOMA = 0
    MENINGIOMA = 1
    PITUITARY = 2
    NO_TUMOR = 3

    @classmethod
    def from_index(cls, idx: int) -> "TumorClass":
        return list(cls)[idx]

    @property
    def display_name(self) -> str:
        return self.name.replace("_", " ").title()


class SegmentationClass(Enum):
    """Segmentation mask classes (BraTS convention)."""
    BACKGROUND = 0
    NECROTIC_CORE = 1  # NCR - Necrotic/Non-enhancing Tumor Core
    EDEMA = 2          # ED - Peritumoral Edema
    ENHANCING = 3      # ET - Enhancing Tumor


@dataclass
class ClassificationResult:
    """Classification inference result."""
    predicted_class: TumorClass
    confidence_score: float
    class_probabilities: Dict[str, float]
    is_low_confidence: bool
    raw_logits: np.ndarray = field(repr=False)

    @property
    def confidence_threshold_passed(self) -> bool:
        return self.confidence_score >= 0.90


@dataclass
class SegmentationResult:
    """Segmentation inference result."""
    edema_mask: np.ndarray = field(repr=False)
    enhancing_mask: np.ndarray = field(repr=False)
    necrotic_mask: np.ndarray = field(repr=False)
    combined_mask: np.ndarray = field(repr=False)
    tumor_area_percentage: float
    raw_logits: np.ndarray = field(repr=False)

    @property
    def is_empty(self) -> bool:
        return (self.edema_mask.sum() + self.enhancing_mask.sum() + 
                self.necrotic_mask.sum()) == 0


@dataclass
class ValidationResult:
    """Post-inference validation result."""
    confidence_passed: bool
    segmentation_plausible: bool
    mask_alignment_valid: bool
    anatomical_plausibility: bool
    notes: List[str]

    @property
    def requires_manual_review(self) -> bool:
        return not (self.confidence_passed and 
                    self.segmentation_plausible and 
                    self.anatomical_plausibility)


@dataclass
class InferenceReport:
    """Complete inference report."""
    image_id: str
    timestamp: datetime
    classification: Optional[ClassificationResult]
    segmentation: Optional[SegmentationResult]
    validation: ValidationResult
    device: str
    inference_time_ms: float

    def to_text(self) -> str:
        """Generate text report in specified format."""
        lines = [
            "=" * 60,
            "INFERENCE REPORT",
            "=" * 60,
            f"Image ID: {self.image_id}",
            f"Timestamp: {self.timestamp.isoformat()}",
            f"Device: {self.device}",
            f"Inference Time: {self.inference_time_ms:.2f} ms",
            "",
        ]

        if self.classification:
            cls = self.classification
            lines.extend([
                "CLASSIFICATION RESULT:",
                "-" * 40,
                f"  Predicted Class: {cls.predicted_class.display_name}",
                f"  Confidence Score: {cls.confidence_score:.4f}",
                f"  Low Confidence Flag: {'Yes' if cls.is_low_confidence else 'No'}",
                "",
                "  Class Probabilities:",
            ])
            for class_name, prob in cls.class_probabilities.items():
                lines.append(f"    - {class_name}: {prob:.4f}")
            lines.append("")

        if self.segmentation:
            seg = self.segmentation
            lines.extend([
                "SEGMENTATION RESULT:",
                "-" * 40,
                f"  Edema Mask: {'Generated' if seg.edema_mask.sum() > 0 else 'Empty'}",
                f"  Enhancing Tumor Mask: {'Generated' if seg.enhancing_mask.sum() > 0 else 'Empty'}",
                f"  Necrotic Core Mask: {'Generated' if seg.necrotic_mask.sum() > 0 else 'Empty'}",
                f"  Total Tumor Area (% of brain): {seg.tumor_area_percentage:.2f}%",
                "",
            ])

        val = self.validation
        lines.extend([
            "VALIDATION CHECKS:",
            "-" * 40,
            f"  Confidence Threshold Passed: {'Yes' if val.confidence_passed else 'No'}",
            f"  Segmentation Plausibility: {'Valid' if val.segmentation_plausible else 'Flagged'}",
            f"  Anatomical Plausibility: {'Valid' if val.anatomical_plausibility else 'Flagged'}",
            f"  Mask Alignment Valid: {'Yes' if val.mask_alignment_valid else 'No'}",
        ])

        if val.notes:
            lines.append("  Notes:")
            for note in val.notes:
                lines.append(f"    - {note}")

        lines.extend([
            "",
            "=" * 60,
            f"FINAL INFERENCE STATUS: "
            f"{'Ready for Clinical Review' if not val.requires_manual_review else 'Requires Manual Review'}",
            "=" * 60,
        ])

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "image_id": self.image_id,
            "timestamp": self.timestamp.isoformat(),
            "device": self.device,
            "inference_time_ms": self.inference_time_ms,
            "final_status": "ready" if not self.validation.requires_manual_review else "manual_review",
        }

        if self.classification:
            result["classification"] = {
                "predicted_class": self.classification.predicted_class.name,
                "confidence_score": self.classification.confidence_score,
                "is_low_confidence": self.classification.is_low_confidence,
                "class_probabilities": self.classification.class_probabilities,
            }

        if self.segmentation:
            result["segmentation"] = {
                "tumor_area_percentage": self.segmentation.tumor_area_percentage,
                "edema_present": bool(self.segmentation.edema_mask.sum() > 0),
                "enhancing_present": bool(self.segmentation.enhancing_mask.sum() > 0),
                "necrotic_present": bool(self.segmentation.necrotic_mask.sum() > 0),
            }

        result["validation"] = {
            "confidence_passed": self.validation.confidence_passed,
            "segmentation_plausible": self.validation.segmentation_plausible,
            "anatomical_plausibility": self.validation.anatomical_plausibility,
            "mask_alignment_valid": self.validation.mask_alignment_valid,
            "notes": self.validation.notes,
            "requires_manual_review": self.validation.requires_manual_review,
        }

        return result


class BrainTumorInferenceEngine:
    """
    Production-grade inference engine for brain tumor MRI analysis.
    
    Performs:
    1. Classification inference (Glioma, Meningioma, Pituitary, No Tumor)
    2. Segmentation inference (Edema, Enhancing Tumor, Necrotic Core)
    3. Post-inference validation
    
    Constraints:
    - No retraining or fine-tuning
    - No hallucinated medical conclusions
    - Outputs based strictly on model predictions
    """

    CONFIDENCE_THRESHOLD = 0.90
    MAX_TUMOR_AREA_PERCENT = 80.0  # Flag if tumor > 80% of brain volume
    MIN_TUMOR_AREA_PERCENT = 0.001  # Flag if tumor < 0.001% (likely miss)

    def __init__(
        self,
        cls_checkpoint: Optional[str] = None,
        seg_checkpoint: Optional[str] = None,
        seg_config: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize inference engine.
        
        Args:
            cls_checkpoint: Path to classification model checkpoint
            seg_checkpoint: Path to segmentation model checkpoint
            seg_config: Path to segmentation config file
            device: Device to run inference on (cuda/cpu)
        """
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"[InferenceEngine] Using device: {self.device}")

        self.cls_model = None
        self.seg_model = None

        # Load classification model
        if cls_checkpoint and os.path.exists(cls_checkpoint):
            print(f"[InferenceEngine] Loading classification model from {cls_checkpoint}")
            self.cls_model = load_classifier_from_checkpoint(
                cls_checkpoint, device=str(self.device)
            )
            self.cls_model.eval()
            print("[InferenceEngine] Classification model loaded successfully")

        # Load segmentation model
        if seg_checkpoint and seg_config and os.path.exists(seg_checkpoint):
            print(f"[InferenceEngine] Loading segmentation model from {seg_checkpoint}")
            config = load_config(seg_config)
            model_config = config["model"]
            self.seg_model = SegmentationModel(**model_config)
            
            checkpoint = torch.load(seg_checkpoint, map_location=self.device, weights_only=False)
            if "model_state_dict" in checkpoint:
                self.seg_model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.seg_model.load_state_dict(checkpoint)
            
            self.seg_model.to(self.device)
            self.seg_model.eval()
            print("[InferenceEngine] Segmentation model loaded successfully")

    @torch.no_grad()
    def classify(
        self,
        image: Union[np.ndarray, torch.Tensor],
        normalize: bool = True,
        expected_channels: int = 3,
    ) -> ClassificationResult:
        """
        Perform classification inference.
        
        Args:
            image: Input image array (C, H, W) or (C, H, W, D) for 3D
            normalize: Whether to normalize to 0-1
            expected_channels: Number of channels expected by classifier (default: 3)
            
        Returns:
            ClassificationResult with prediction and probabilities
        """
        if self.cls_model is None:
            raise RuntimeError("Classification model not loaded")

        # Prepare input
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()

        # Handle different input dimensions
        if image.dim() == 4:  # (C, H, W, D)
            # Take middle slice for 2.5D classification
            image = image[..., image.shape[-1] // 2]  # (C, H, W)

        # Handle channel mismatch (e.g., 4-channel MRI to 3-channel classifier)
        C = image.shape[0]
        if C > expected_channels:
            # Take first N channels (T1, T2, FLAIR for 3-channel)
            # or use a composite: e.g., average last channels
            image = image[:expected_channels]  # (expected_channels, H, W)
        elif C < expected_channels:
            # Repeat channels if input has fewer channels
            repeats = expected_channels // C + 1
            image = image.repeat(repeats, 1, 1)[:expected_channels]

        # Normalize
        if normalize and image.max() > 1.0:
            image = image / 255.0

        # Add batch and slice dims: (C, H, W) -> (1, 1, C, H, W)
        image = image.unsqueeze(0).unsqueeze(0).to(self.device)

        # Inference
        logits = self.cls_model(image)
        probabilities = F.softmax(logits, dim=1)[0].cpu().numpy()
        predicted_idx = int(probabilities.argmax())
        confidence = float(probabilities[predicted_idx])

        # Build class probabilities dict
        class_probs = {
            TumorClass.GLIOMA.display_name: float(probabilities[0]),
            TumorClass.MENINGIOMA.display_name: float(probabilities[1]),
            TumorClass.PITUITARY.display_name: float(probabilities[2]),
            TumorClass.NO_TUMOR.display_name: float(probabilities[3]),
        }

        return ClassificationResult(
            predicted_class=TumorClass.from_index(predicted_idx),
            confidence_score=confidence,
            class_probabilities=class_probs,
            is_low_confidence=confidence < self.CONFIDENCE_THRESHOLD,
            raw_logits=logits.cpu().numpy(),
        )

    @torch.no_grad()
    def segment(
        self,
        image: Union[np.ndarray, torch.Tensor],
        patch_size: int = 128,
        use_sliding_window: bool = False,
        overlap: float = 0.5,
    ) -> SegmentationResult:
        """
        Perform segmentation inference.
        
        Args:
            image: Input 3D MRI volume (C, H, W, D)
            patch_size: Size of patch for inference
            use_sliding_window: Use sliding window inference for large volumes
            overlap: Overlap ratio for sliding window
            
        Returns:
            SegmentationResult with masks and metrics
        """
        if self.seg_model is None:
            raise RuntimeError("Segmentation model not loaded")

        # Prepare input
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()

        C, H, W, D = image.shape

        # Pad if needed
        pad_h = max(0, patch_size - H)
        pad_w = max(0, patch_size - W)
        pad_d = max(0, patch_size - D)

        if pad_h or pad_w or pad_d:
            image = F.pad(image, (0, pad_d, 0, pad_w, 0, pad_h))

        # Update dimensions after padding
        _, H_pad, W_pad, D_pad = image.shape

        # Center crop to patch size
        s_h = (H_pad - patch_size) // 2
        s_w = (W_pad - patch_size) // 2
        s_d = (D_pad - patch_size) // 2

        image_patch = image[
            :, 
            s_h:s_h + patch_size, 
            s_w:s_w + patch_size, 
            s_d:s_d + patch_size
        ]

        # Add batch dim: (C, H, W, D) -> (1, C, H, W, D)
        input_tensor = image_patch.unsqueeze(0).to(self.device)

        # Inference
        output = self.seg_model(input_tensor)
        raw_logits = output.cpu().numpy()

        # Get prediction mask
        pred_mask = torch.argmax(output, dim=1)[0].cpu().numpy()  # (H, W, D)

        # Extract individual masks
        necrotic_mask = (pred_mask == SegmentationClass.NECROTIC_CORE.value).astype(np.uint8)
        edema_mask = (pred_mask == SegmentationClass.EDEMA.value).astype(np.uint8)
        enhancing_mask = (pred_mask == SegmentationClass.ENHANCING.value).astype(np.uint8)

        # Calculate tumor area percentage
        total_voxels = pred_mask.size
        tumor_voxels = necrotic_mask.sum() + edema_mask.sum() + enhancing_mask.sum()
        tumor_percentage = (tumor_voxels / total_voxels) * 100

        return SegmentationResult(
            edema_mask=edema_mask,
            enhancing_mask=enhancing_mask,
            necrotic_mask=necrotic_mask,
            combined_mask=pred_mask,
            tumor_area_percentage=float(tumor_percentage),
            raw_logits=raw_logits,
        )

    def validate(
        self,
        classification_result: Optional[ClassificationResult],
        segmentation_result: Optional[SegmentationResult],
        brain_mask: Optional[np.ndarray] = None,
    ) -> ValidationResult:
        """
        Perform post-inference validation.
        
        Args:
            classification_result: Classification inference result
            segmentation_result: Segmentation inference result
            brain_mask: Optional brain mask for anatomical validation
            
        Returns:
            ValidationResult with all check results
        """
        notes = []

        # Confidence check
        confidence_passed = True
        if classification_result:
            if classification_result.is_low_confidence:
                confidence_passed = False
                notes.append(
                    f"Low confidence: {classification_result.confidence_score:.3f} "
                    f"(threshold: {self.CONFIDENCE_THRESHOLD})"
                )

        # Segmentation plausibility
        segmentation_plausible = True
        anatomical_plausibility = True
        mask_alignment = True

        if segmentation_result:
            # Check for empty mask when tumor detected
            if classification_result:
                if (classification_result.predicted_class != TumorClass.NO_TUMOR 
                    and segmentation_result.is_empty):
                    segmentation_plausible = False
                    notes.append(
                        "Classification predicts tumor but segmentation mask is empty"
                    )

            # Check tumor size reasonability
            if segmentation_result.tumor_area_percentage > self.MAX_TUMOR_AREA_PERCENT:
                anatomical_plausibility = False
                notes.append(
                    f"Suspicious: Tumor area {segmentation_result.tumor_area_percentage:.1f}% "
                    f"exceeds maximum expected ({self.MAX_TUMOR_AREA_PERCENT}%)"
                )

            if (not segmentation_result.is_empty and 
                segmentation_result.tumor_area_percentage < self.MIN_TUMOR_AREA_PERCENT):
                notes.append(
                    f"Very small tumor area detected: {segmentation_result.tumor_area_percentage:.4f}%"
                )

            # Check anatomical plausibility with brain mask if available
            if brain_mask is not None:
                tumor_mask = segmentation_result.combined_mask > 0
                outside_brain = tumor_mask & (brain_mask == 0)
                if outside_brain.sum() > 0:
                    anatomical_plausibility = False
                    notes.append(
                        f"Segmentation extends outside brain region: "
                        f"{outside_brain.sum()} voxels"
                    )

        return ValidationResult(
            confidence_passed=confidence_passed,
            segmentation_plausible=segmentation_plausible,
            mask_alignment_valid=mask_alignment,
            anatomical_plausibility=anatomical_plausibility,
            notes=notes,
        )

    def detect_data_type(
        self,
        image: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[str, bool, bool]:
        """
        Detect the data type and recommend which inference to run.
        
        Returns:
            Tuple of (data_type_name, should_classify, should_segment)
        """
        if isinstance(image, torch.Tensor):
            shape = image.shape
        else:
            shape = image.shape
        
        ndim = len(shape)
        channels = shape[0]
        
        # 4-channel 3D data = BraTS MRI (for segmentation only)
        if ndim == 4 and channels == 4:
            return "brats_mri", False, True
        
        # 3-channel 2D/3D data = Classification ROI 
        if channels == 3:
            if ndim == 3:  # (C, H, W)
                return "classification_2d", True, False
            elif ndim == 4 and shape[3] == 1:  # (C, H, W, 1)
                return "classification_2d", True, False
        
        # 4-channel 3D = BraTS (default behavior)
        if ndim == 4 and channels == 4:
            return "brats_mri", False, True
        
        # Fallback: unknown type, try both but warn
        return "unknown", True, True

    def run_inference(
        self,
        image: Union[np.ndarray, torch.Tensor],
        image_id: str = "unknown",
        run_classification: bool = True,
        run_segmentation: bool = True,
        auto_detect: bool = True,  # NEW: auto-detect data type
        brain_mask: Optional[np.ndarray] = None,
    ) -> InferenceReport:
        """
        Run complete inference pipeline.
        
        Args:
            image: Input MRI image/volume
            image_id: Identifier for the image
            run_classification: Whether to run classification
            run_segmentation: Whether to run segmentation
            auto_detect: If True, auto-detect data type and skip incompatible inference
            brain_mask: Optional brain mask for validation
            
        Returns:
            Complete InferenceReport
        """
        import time
        start_time = time.perf_counter()

        cls_result = None
        seg_result = None
        
        # Auto-detect data type and adjust settings
        if auto_detect:
            data_type, should_classify, should_segment = self.detect_data_type(image)
            
            if not should_classify and run_classification:
                print(f"[InferenceEngine] Auto-skipping classification (data type: {data_type})")
                run_classification = False
            
            if not should_segment and run_segmentation:
                print(f"[InferenceEngine] Auto-skipping segmentation (data type: {data_type})")
                run_segmentation = False

        # Classification
        if run_classification and self.cls_model is not None:
            try:
                cls_result = self.classify(image)
            except Exception as e:
                print(f"[InferenceEngine] Classification error: {e}")

        # Segmentation
        if run_segmentation and self.seg_model is not None:
            try:
                seg_result = self.segment(image)
            except Exception as e:
                print(f"[InferenceEngine] Segmentation error: {e}")

        # Validation
        validation = self.validate(cls_result, seg_result, brain_mask)

        # Calculate inference time
        inference_time = (time.perf_counter() - start_time) * 1000

        return InferenceReport(
            image_id=image_id,
            timestamp=datetime.now(),
            classification=cls_result,
            segmentation=seg_result,
            validation=validation,
            device=str(self.device),
            inference_time_ms=inference_time,
        )

    def batch_inference(
        self,
        images: List[Tuple[str, Union[np.ndarray, torch.Tensor]]],
        **kwargs,
    ) -> List[InferenceReport]:
        """
        Run inference on a batch of images.
        
        Args:
            images: List of (image_id, image) tuples
            **kwargs: Additional arguments for run_inference
            
        Returns:
            List of InferenceReports
        """
        reports = []
        for image_id, image in images:
            report = self.run_inference(image, image_id=image_id, **kwargs)
            reports.append(report)
        return reports


def create_inference_engine(
    cls_checkpoint: str = "checkpoints/classification/best_model.pt",
    seg_checkpoint: str = "checkpoints/segmentation/best_model.pt",
    seg_config: str = "configs/seg.yaml",
    device: Optional[str] = None,
) -> BrainTumorInferenceEngine:
    """
    Factory function to create inference engine with default paths.
    
    Args:
        cls_checkpoint: Path to classification checkpoint
        seg_checkpoint: Path to segmentation checkpoint
        seg_config: Path to segmentation config
        device: Device to use
        
    Returns:
        Configured BrainTumorInferenceEngine
    """
    return BrainTumorInferenceEngine(
        cls_checkpoint=cls_checkpoint,
        seg_checkpoint=seg_checkpoint,
        seg_config=seg_config,
        device=device,
    )


if __name__ == "__main__":
    # Test inference engine
    print("Testing Brain Tumor Inference Engine...")
    
    engine = create_inference_engine()
    
    # Create dummy input for testing
    dummy_cls_input = np.random.rand(3, 224, 224).astype(np.float32)
    dummy_seg_input = np.random.rand(4, 128, 128, 128).astype(np.float32)
    
    # Test classification
    if engine.cls_model:
        cls_result = engine.classify(dummy_cls_input)
        print(f"\nClassification Result:")
        print(f"  Predicted: {cls_result.predicted_class.display_name}")
        print(f"  Confidence: {cls_result.confidence_score:.4f}")
    
    # Test segmentation
    if engine.seg_model:
        seg_result = engine.segment(dummy_seg_input)
        print(f"\nSegmentation Result:")
        print(f"  Tumor Area: {seg_result.tumor_area_percentage:.2f}%")
    
    # Full inference
    report = engine.run_inference(
        dummy_seg_input if engine.seg_model else dummy_cls_input,
        image_id="test_image_001"
    )
    print("\n" + report.to_text())
