"""
FastAPI Inference Endpoint for Brain Tumor Analysis

Production-grade REST API for:
- Single image classification
- Single/batch segmentation
- Combined inference with validation

Endpoints:
- POST /classify - Classify single MRI
- POST /segment - Segment single MRI volume
- POST /infer - Full inference pipeline
- POST /batch - Batch inference
- GET /health - Health check
"""

import base64
import io
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.engine import (
    BrainTumorInferenceEngine,
    create_inference_engine,
    ClassificationResult,
    SegmentationResult,
    InferenceReport,
    TumorClass,
)

# ============================================================================
# Pydantic Models
# ============================================================================

class ClassProbabilities(BaseModel):
    glioma: float = Field(..., description="Probability of glioma")
    meningioma: float = Field(..., description="Probability of meningioma")
    pituitary: float = Field(..., description="Probability of pituitary tumor")
    no_tumor: float = Field(..., description="Probability of no tumor")


class ClassificationResponse(BaseModel):
    predicted_class: str = Field(..., description="Predicted tumor class")
    confidence_score: float = Field(..., description="Confidence of prediction")
    is_low_confidence: bool = Field(..., description="Flag for low confidence")
    class_probabilities: ClassProbabilities
    

class SegmentationStats(BaseModel):
    edema_present: bool = Field(..., description="Edema detected")
    enhancing_present: bool = Field(..., description="Enhancing tumor detected")
    necrotic_present: bool = Field(..., description="Necrotic core detected")
    tumor_area_percentage: float = Field(..., description="Tumor area as % of brain")


class SegmentationResponse(BaseModel):
    stats: SegmentationStats
    mask_shape: List[int] = Field(..., description="Shape of output mask")
    combined_mask_base64: Optional[str] = Field(None, description="Base64 encoded mask")


class ValidationResponse(BaseModel):
    confidence_passed: bool
    segmentation_plausible: bool
    anatomical_plausibility: bool
    mask_alignment_valid: bool
    notes: List[str]
    requires_manual_review: bool


class InferenceResponse(BaseModel):
    image_id: str
    timestamp: str
    device: str
    inference_time_ms: float
    final_status: str = Field(..., description="'ready' or 'manual_review'")
    classification: Optional[ClassificationResponse] = None
    segmentation: Optional[SegmentationResponse] = None
    validation: ValidationResponse


class HealthResponse(BaseModel):
    status: str
    classification_model_loaded: bool
    segmentation_model_loaded: bool
    device: str
    timestamp: str


class InferenceRequest(BaseModel):
    """Request model for inference with numpy array data."""
    image_base64: str = Field(..., description="Base64 encoded numpy array")
    image_id: str = Field(default="uploaded_image", description="Image identifier")
    run_classification: bool = Field(default=True)
    run_segmentation: bool = Field(default=True)


class BatchInferenceRequest(BaseModel):
    """Request for batch inference."""
    images: List[InferenceRequest]


# ============================================================================
# API Setup
# ============================================================================

app = FastAPI(
    title="Brain Tumor MRI Analysis API",
    description="Production-grade inference API for brain tumor classification and segmentation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference engine
engine: Optional[BrainTumorInferenceEngine] = None


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def load_models():
    """Load models on startup."""
    global engine
    
    cls_checkpoint = os.environ.get(
        "CLS_CHECKPOINT", 
        "checkpoints/classification/best_model.pt"
    )
    seg_checkpoint = os.environ.get(
        "SEG_CHECKPOINT",
        "checkpoints/segmentation/best_model.pt"
    )
    seg_config = os.environ.get(
        "SEG_CONFIG",
        "configs/seg.yaml"
    )
    device = os.environ.get("DEVICE", None)
    
    print("[API] Loading inference engine...")
    engine = create_inference_engine(
        cls_checkpoint=cls_checkpoint,
        seg_checkpoint=seg_checkpoint,
        seg_config=seg_config,
        device=device,
    )
    print("[API] Inference engine ready")


# ============================================================================
# Helper Functions
# ============================================================================

def decode_numpy_array(base64_str: str) -> np.ndarray:
    """Decode base64 string to numpy array."""
    try:
        data = base64.b64decode(base64_str)
        buffer = io.BytesIO(data)
        return np.load(buffer, allow_pickle=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {str(e)}")


def encode_numpy_array(arr: np.ndarray) -> str:
    """Encode numpy array to base64 string."""
    buffer = io.BytesIO()
    np.save(buffer, arr, allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def classification_to_response(result: ClassificationResult) -> ClassificationResponse:
    """Convert classification result to response model."""
    return ClassificationResponse(
        predicted_class=result.predicted_class.name,
        confidence_score=result.confidence_score,
        is_low_confidence=result.is_low_confidence,
        class_probabilities=ClassProbabilities(
            glioma=result.class_probabilities.get("Glioma", 0.0),
            meningioma=result.class_probabilities.get("Meningioma", 0.0),
            pituitary=result.class_probabilities.get("Pituitary", 0.0),
            no_tumor=result.class_probabilities.get("No Tumor", 0.0),
        ),
    )


def segmentation_to_response(
    result: SegmentationResult, 
    include_mask: bool = False
) -> SegmentationResponse:
    """Convert segmentation result to response model."""
    response = SegmentationResponse(
        stats=SegmentationStats(
            edema_present=bool(result.edema_mask.sum() > 0),
            enhancing_present=bool(result.enhancing_mask.sum() > 0),
            necrotic_present=bool(result.necrotic_mask.sum() > 0),
            tumor_area_percentage=result.tumor_area_percentage,
        ),
        mask_shape=list(result.combined_mask.shape),
        combined_mask_base64=encode_numpy_array(result.combined_mask) if include_mask else None,
    )
    return response


def report_to_response(
    report: InferenceReport, 
    include_masks: bool = False
) -> InferenceResponse:
    """Convert inference report to response model."""
    classification = None
    if report.classification:
        classification = classification_to_response(report.classification)

    segmentation = None
    if report.segmentation:
        segmentation = segmentation_to_response(report.segmentation, include_masks)

    return InferenceResponse(
        image_id=report.image_id,
        timestamp=report.timestamp.isoformat(),
        device=report.device,
        inference_time_ms=report.inference_time_ms,
        final_status="ready" if not report.validation.requires_manual_review else "manual_review",
        classification=classification,
        segmentation=segmentation,
        validation=ValidationResponse(
            confidence_passed=report.validation.confidence_passed,
            segmentation_plausible=report.validation.segmentation_plausible,
            anatomical_plausibility=report.validation.anatomical_plausibility,
            mask_alignment_valid=report.validation.mask_alignment_valid,
            notes=report.validation.notes,
            requires_manual_review=report.validation.requires_manual_review,
        ),
    )


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if engine else "not_ready",
        classification_model_loaded=engine.cls_model is not None if engine else False,
        segmentation_model_loaded=engine.seg_model is not None if engine else False,
        device=str(engine.device) if engine else "unknown",
        timestamp=datetime.now().isoformat(),
    )


@app.post("/classify", response_model=ClassificationResponse, tags=["Classification"])
async def classify_image(request: InferenceRequest):
    """
    Classify a single brain MRI image.
    
    Input should be a base64-encoded numpy array of shape (C, H, W) or (C, H, W, D).
    """
    if engine is None or engine.cls_model is None:
        raise HTTPException(status_code=503, detail="Classification model not loaded")
    
    image = decode_numpy_array(request.image_base64)
    result = engine.classify(image)
    return classification_to_response(result)


@app.post("/segment", response_model=SegmentationResponse, tags=["Segmentation"])
async def segment_image(request: InferenceRequest, include_mask: bool = False):
    """
    Segment a single brain MRI volume.
    
    Input should be a base64-encoded numpy array of shape (C, H, W, D).
    Set include_mask=true to receive the base64-encoded mask in response.
    """
    if engine is None or engine.seg_model is None:
        raise HTTPException(status_code=503, detail="Segmentation model not loaded")
    
    image = decode_numpy_array(request.image_base64)
    
    if image.ndim != 4:
        raise HTTPException(
            status_code=400, 
            detail=f"Expected 4D input (C, H, W, D), got {image.ndim}D"
        )
    
    result = engine.segment(image)
    return segmentation_to_response(result, include_mask)


@app.post("/infer", response_model=InferenceResponse, tags=["Inference"])
async def full_inference(request: InferenceRequest, include_masks: bool = False):
    """
    Run complete inference pipeline (classification + segmentation + validation).
    
    Input should be a base64-encoded numpy array.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not loaded")
    
    image = decode_numpy_array(request.image_base64)
    
    report = engine.run_inference(
        image=image,
        image_id=request.image_id,
        run_classification=request.run_classification,
        run_segmentation=request.run_segmentation,
    )
    
    return report_to_response(report, include_masks)


@app.post("/batch", response_model=List[InferenceResponse], tags=["Batch"])
async def batch_inference(request: BatchInferenceRequest, include_masks: bool = False):
    """
    Run batch inference on multiple images.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not loaded")
    
    responses = []
    for item in request.images:
        image = decode_numpy_array(item.image_base64)
        report = engine.run_inference(
            image=image,
            image_id=item.image_id,
            run_classification=item.run_classification,
            run_segmentation=item.run_segmentation,
        )
        responses.append(report_to_response(report, include_masks))
    
    return responses


@app.post("/upload", response_model=InferenceResponse, tags=["Upload"])
async def upload_and_infer(
    file: UploadFile = File(...),
    run_classification: bool = True,
    run_segmentation: bool = True,
    include_masks: bool = False,
):
    """
    Upload a .npy file directly and run inference.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not loaded")
    
    if not file.filename.endswith(".npy"):
        raise HTTPException(status_code=400, detail="Only .npy files are supported")
    
    try:
        # Save to temp file and load
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        image = np.load(tmp_path, allow_pickle=False)
        os.unlink(tmp_path)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load file: {str(e)}")
    
    report = engine.run_inference(
        image=image,
        image_id=file.filename,
        run_classification=run_classification,
        run_segmentation=run_segmentation,
    )
    
    return report_to_response(report, include_masks)


@app.get("/report/{image_id}", tags=["Reports"])
async def get_text_report(image_id: str):
    """
    Generate a text report for a previously processed image.
    
    Note: This is a placeholder - in production, implement caching/storage.
    """
    # In production, retrieve from cache/database
    return JSONResponse(
        status_code=501,
        content={"detail": "Report storage not implemented. Use /infer endpoint."}
    )


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
    )
