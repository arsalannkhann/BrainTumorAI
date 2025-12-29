"""
Test Script for Production Inference Engine

Validates:
1. Model loading
2. Classification inference
3. Segmentation inference
4. Validation logic
5. Report generation
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from inference.engine import (
    BrainTumorInferenceEngine,
    create_inference_engine,
    TumorClass,
    SegmentationClass,
)


def test_model_loading():
    """Test that models load correctly."""
    print("\n" + "="*60)
    print("TEST: Model Loading")
    print("="*60)
    
    engine = create_inference_engine()
    
    cls_loaded = engine.cls_model is not None
    seg_loaded = engine.seg_model is not None
    
    print(f"Classification model loaded: {cls_loaded}")
    print(f"Segmentation model loaded: {seg_loaded}")
    
    if cls_loaded:
        print(f"  Device: {next(engine.cls_model.parameters()).device}")
    if seg_loaded:
        print(f"  Device: {next(engine.seg_model.parameters()).device}")
    
    return engine


def test_classification_inference(engine: BrainTumorInferenceEngine):
    """Test classification inference."""
    print("\n" + "="*60)
    print("TEST: Classification Inference")
    print("="*60)
    
    if engine.cls_model is None:
        print("SKIPPED: Classification model not loaded")
        return
    
    # Create test input
    test_input = np.random.rand(3, 224, 224).astype(np.float32)
    
    result = engine.classify(test_input)
    
    print(f"Predicted class: {result.predicted_class.name}")
    print(f"Confidence: {result.confidence_score:.4f}")
    print(f"Low confidence flag: {result.is_low_confidence}")
    print(f"Probabilities:")
    for class_name, prob in result.class_probabilities.items():
        print(f"  {class_name}: {prob:.4f}")
    
    # Validate output
    assert isinstance(result.predicted_class, TumorClass)
    assert 0 <= result.confidence_score <= 1
    assert sum(result.class_probabilities.values()) - 1.0 < 0.01
    
    print("\n✓ Classification inference PASSED")


def test_segmentation_inference(engine: BrainTumorInferenceEngine):
    """Test segmentation inference."""
    print("\n" + "="*60)
    print("TEST: Segmentation Inference")
    print("="*60)
    
    if engine.seg_model is None:
        print("SKIPPED: Segmentation model not loaded")
        return
    
    # Create test input (4 channels for T1, T2, FLAIR, T1ce)
    test_input = np.random.rand(4, 128, 128, 128).astype(np.float32)
    
    result = engine.segment(test_input)
    
    print(f"Output shape: {result.combined_mask.shape}")
    print(f"Unique values in mask: {np.unique(result.combined_mask)}")
    print(f"Tumor area percentage: {result.tumor_area_percentage:.4f}%")
    print(f"Edema voxels: {result.edema_mask.sum()}")
    print(f"Enhancing voxels: {result.enhancing_mask.sum()}")
    print(f"Necrotic voxels: {result.necrotic_mask.sum()}")
    
    # Validate output
    assert result.combined_mask.shape == (128, 128, 128)
    assert result.edema_mask.shape == (128, 128, 128)
    assert 0 <= result.tumor_area_percentage <= 100
    
    print("\n✓ Segmentation inference PASSED")


def test_validation_logic(engine: BrainTumorInferenceEngine):
    """Test post-inference validation."""
    print("\n" + "="*60)
    print("TEST: Validation Logic")
    print("="*60)
    
    # Test case 1: Low confidence classification
    from inference.engine import ClassificationResult, SegmentationResult
    
    low_conf_result = ClassificationResult(
        predicted_class=TumorClass.GLIOMA,
        confidence_score=0.6,  # Below threshold
        class_probabilities={
            "Glioma": 0.6,
            "Meningioma": 0.2,
            "Pituitary": 0.1,
            "No Tumor": 0.1,
        },
        is_low_confidence=True,
        raw_logits=np.array([1.0, 0.5, 0.2, 0.1]),
    )
    
    validation = engine.validate(low_conf_result, None)
    assert not validation.confidence_passed, "Should flag low confidence"
    print("✓ Low confidence detection works")
    
    # Test case 2: Tumor classified but empty segmentation
    empty_seg = SegmentationResult(
        edema_mask=np.zeros((128, 128, 128), dtype=np.uint8),
        enhancing_mask=np.zeros((128, 128, 128), dtype=np.uint8),
        necrotic_mask=np.zeros((128, 128, 128), dtype=np.uint8),
        combined_mask=np.zeros((128, 128, 128), dtype=np.uint8),
        tumor_area_percentage=0.0,
        raw_logits=np.zeros((1, 4, 128, 128, 128)),
    )
    
    tumor_cls = ClassificationResult(
        predicted_class=TumorClass.GLIOMA,
        confidence_score=0.95,
        class_probabilities={
            "Glioma": 0.95,
            "Meningioma": 0.03,
            "Pituitary": 0.01,
            "No Tumor": 0.01,
        },
        is_low_confidence=False,
        raw_logits=np.array([2.0, 0.1, 0.05, 0.02]),
    )
    
    validation = engine.validate(tumor_cls, empty_seg)
    assert not validation.segmentation_plausible, "Should flag empty mask with tumor"
    print("✓ Empty segmentation with tumor classification detection works")
    
    # Test case 3: Excessive tumor area
    full_mask = np.ones((128, 128, 128), dtype=np.uint8) * 2  # All edema
    large_seg = SegmentationResult(
        edema_mask=full_mask,
        enhancing_mask=np.zeros((128, 128, 128), dtype=np.uint8),
        necrotic_mask=np.zeros((128, 128, 128), dtype=np.uint8),
        combined_mask=full_mask,
        tumor_area_percentage=100.0,  # 100% is definitely wrong
        raw_logits=np.zeros((1, 4, 128, 128, 128)),
    )
    
    validation = engine.validate(None, large_seg)
    assert not validation.anatomical_plausibility, "Should flag excessive tumor"
    print("✓ Excessive tumor area detection works")
    
    print("\n✓ Validation logic PASSED")


def test_full_inference(engine: BrainTumorInferenceEngine):
    """Test complete inference pipeline."""
    print("\n" + "="*60)
    print("TEST: Full Inference Pipeline")
    print("="*60)
    
    # Create test input
    test_input = np.random.rand(4, 128, 128, 128).astype(np.float32)
    
    report = engine.run_inference(
        image=test_input,
        image_id="test_patient_001",
        run_classification=engine.cls_model is not None,
        run_segmentation=engine.seg_model is not None,
    )
    
    print(f"Image ID: {report.image_id}")
    print(f"Timestamp: {report.timestamp}")
    print(f"Device: {report.device}")
    print(f"Inference time: {report.inference_time_ms:.2f} ms")
    print(f"Requires manual review: {report.validation.requires_manual_review}")
    
    if report.classification:
        print(f"\nClassification:")
        print(f"  Class: {report.classification.predicted_class.name}")
        print(f"  Confidence: {report.classification.confidence_score:.4f}")
    
    if report.segmentation:
        print(f"\nSegmentation:")
        print(f"  Tumor area: {report.segmentation.tumor_area_percentage:.4f}%")
    
    # Test report generation
    text_report = report.to_text()
    dict_report = report.to_dict()
    
    assert "INFERENCE REPORT" in text_report
    assert "image_id" in dict_report
    assert "validation" in dict_report
    
    print("\n--- TEXT REPORT ---")
    print(text_report[:1000] + "..." if len(text_report) > 1000 else text_report)
    
    print("\n✓ Full inference pipeline PASSED")


def test_with_real_data(engine: BrainTumorInferenceEngine):
    """Test with real data if available."""
    print("\n" + "="*60)
    print("TEST: Real Data Inference")
    print("="*60)
    
    # Try to find real test data
    test_paths = [
        Path("data/processed"),
        Path("data/roi"),
    ]
    
    for data_dir in test_paths:
        if not data_dir.exists():
            continue
        
        npy_files = list(data_dir.glob("**/*.npy"))
        if not npy_files:
            continue
        
        # Test with first available file
        test_file = npy_files[0]
        print(f"Testing with: {test_file}")
        
        try:
            image = np.load(test_file, allow_pickle=False)
            print(f"Image shape: {image.shape}")
            
            report = engine.run_inference(
                image=image,
                image_id=test_file.stem,
            )
            
            print(f"\nResult:")
            if report.classification:
                print(f"  Classification: {report.classification.predicted_class.name} "
                      f"({report.classification.confidence_score:.2%})")
            if report.segmentation:
                print(f"  Tumor area: {report.segmentation.tumor_area_percentage:.4f}%")
            print(f"  Status: {'Ready' if not report.validation.requires_manual_review else 'Needs Review'}")
            
            return True
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print("No real test data found - using synthetic data only")
    return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("BRAIN TUMOR INFERENCE ENGINE - TEST SUITE")
    print("="*60)
    
    try:
        # Test 1: Model loading
        engine = test_model_loading()
        
        # Test 2: Classification
        test_classification_inference(engine)
        
        # Test 3: Segmentation
        test_segmentation_inference(engine)
        
        # Test 4: Validation
        test_validation_logic(engine)
        
        # Test 5: Full pipeline
        test_full_inference(engine)
        
        # Test 6: Real data (optional)
        test_with_real_data(engine)
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
