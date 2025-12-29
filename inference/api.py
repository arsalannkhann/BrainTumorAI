"""
FastAPI Inference Endpoint for Brain Tumor Analysis

Production-grade REST API for:
- Single image classification
- Single/batch segmentation
- Combined inference with validation
- History tracking (SQLite)
- Explainability (Grad-CAM)

Endpoints:
- POST /classify - Classify single MRI
- POST /segment - Segment single MRI volume
- POST /infer - Full inference pipeline
- POST /batch - Batch inference
- GET /health - Health check
- GET /history - Get past predictions
- POST /upload - Upload and process image
- POST /explain/gradcam - Generate Grad-CAM
"""

import base64
import io
import os
import sys
import tempfile
import shutil
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import cv2
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
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
from inference.database import init_db, get_db, PredictionRecord
from inference.xai import GradCAM, visualize_gradcam

# ============================================================================
# Configuration & Directories
# ============================================================================

UPLOAD_DIR = Path("data/uploads")
MASK_DIR = Path("data/masks")
GRADCAM_DIR = Path("data/gradcam")

for d in [UPLOAD_DIR, MASK_DIR, GRADCAM_DIR]:
    d.mkdir(parents=True, exist_ok=True)

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
    mask_path: Optional[str] = None


class ValidationResponse(BaseModel):
    confidence_passed: bool
    segmentation_plausible: bool
    anatomical_plausibility: bool
    mask_alignment_valid: bool
    notes: List[str]
    requires_manual_review: bool


class InferenceResponse(BaseModel):
    id: Optional[int] = None
    image_id: str
    timestamp: str
    device: str
    inference_time_ms: float
    final_status: str = Field(..., description="'ready' or 'manual_review'")
    classification: Optional[ClassificationResponse] = None
    segmentation: Optional[SegmentationResponse] = None
    validation: ValidationResponse
    image_path: Optional[str] = None
    gradcam_path: Optional[str] = None


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
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for accessing images
app.mount("/data", StaticFiles(directory="data"), name="data")

# Global inference engine
engine: Optional[BrainTumorInferenceEngine] = None
gradcam: Optional[GradCAM] = None

# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def load_models():
    """Load models and initialize DB on startup."""
    global engine, gradcam
    
    # Initialize DB
    init_db()
    print("[API] Database initialized")

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
    try:
        engine = create_inference_engine(
            cls_checkpoint=cls_checkpoint,
            seg_checkpoint=seg_checkpoint,
            seg_config=seg_config,
            device=device,
        )
        print("[API] Inference engine ready")
        
        if engine.cls_model:
             gradcam = GradCAM(engine.cls_model)
             print("[API] Grad-CAM initialized")
             
    except Exception as e:
        print(f"[API] Error loading models: {e}")
        # We don't crash app, just report unhealthy


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
    include_mask: bool = False,
    mask_path: str = None
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
        mask_path=mask_path
    )
    return response


def save_prediction_to_db(
    db: Session,
    report: InferenceReport,
    image_path: str,
    mask_path: Optional[str] = None,
    gradcam_path: Optional[str] = None
) -> PredictionRecord:
    """Save prediction result to database."""
    
    probs = {}
    if report.classification:
        probs = report.classification.class_probabilities
        
    record = PredictionRecord(
        image_id=report.image_id,
        timestamp=report.timestamp,
        image_path=image_path,
        mask_path=mask_path,
        gradcam_path=gradcam_path,
        predicted_class=report.classification.predicted_class.name if report.classification else None,
        confidence_score=report.classification.confidence_score if report.classification else None,
        probabilities=probs,
        tumor_area_percentage=report.segmentation.tumor_area_percentage if report.segmentation else None,
        has_tumor=(report.segmentation.tumor_area_percentage > 0) if report.segmentation else False,
        validation_passed=not report.validation.requires_manual_review
    )
    
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


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


@app.get("/history", response_model=List[InferenceResponse], tags=["History"])
async def get_history(limit: int = 50, db: Session = Depends(get_db)):
    """Get history of predictions."""
    records = db.query(PredictionRecord).order_by(PredictionRecord.timestamp.desc()).limit(limit).all()
    
    responses = []
    for r in records:
        responses.append(InferenceResponse(
            id=r.id,
            image_id=r.image_id,
            timestamp=r.timestamp.isoformat(),
            device="saved",
            inference_time_ms=0,
            final_status="ready" if r.validation_passed else "manual_review",
            classification=ClassificationResponse(
                predicted_class=r.predicted_class,
                confidence_score=r.confidence_score,
                is_low_confidence=False, # TODO store this
                class_probabilities=ClassProbabilities(
                    glioma=r.probabilities.get("Glioma", 0.0),
                    meningioma=r.probabilities.get("Meningioma", 0.0),
                    pituitary=r.probabilities.get("Pituitary", 0.0),
                    no_tumor=r.probabilities.get("No Tumor", 0.0),
                )
            ) if r.predicted_class else None,
            segmentation=SegmentationResponse(
                stats=SegmentationStats(
                    edema_present=False, # stored implicitly
                    enhancing_present=False, 
                    necrotic_present=False,
                    tumor_area_percentage=r.tumor_area_percentage or 0.0
                ),
                mask_shape=[],
                mask_path=r.mask_path
            ) if r.tumor_area_percentage is not None else None,
            validation=ValidationResponse(
                confidence_passed=True,
                segmentation_plausible=True,
                anatomical_plausibility=True,
                mask_alignment_valid=True,
                notes=[],
                requires_manual_review=not r.validation_passed
            ),
            image_path=r.image_path,
            gradcam_path=r.gradcam_path
        ))
    return responses


@app.post("/upload", response_model=InferenceResponse, tags=["Upload"])
async def upload_and_infer(
    file: UploadFile = File(...),
    run_classification: bool = True,
    run_segmentation: bool = True,
    db: Session = Depends(get_db)
):
    """
    Upload a file, run inference, save results, and return report.
    Supports .npy, .jpg, .png.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not loaded")
    
    # Save Upload
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized_filename = f"{timestamp}_{file.filename}"
    file_path = UPLOAD_DIR / sanitized_filename
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Load Image
    try:
        if file.filename.endswith(".npy"):
            image = np.load(file_path)
        elif file.filename.endswith((".jpg", ".png", ".jpeg")):
            # Convert simple image to model format
            # This is a bit tricky depending on model expectation (3D vs 2D)
            # For now, assume we handle 2D images by repeating slices or 3 ch
            img_cv = cv2.imread(str(file_path))
            if img_cv is None:
                 raise ValueError("Invalid image")
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            
            # Normalize to 0-1
            img_cv = img_cv.astype(np.float32) / 255.0
            
            # Model expects (C, H, W) or (C, H, W, D)
            # If 2D image, make it (3, H, W)
            image = np.transpose(img_cv, (2, 0, 1))
            
            # If model is 2.5D or 3D, we might need to adjust dim
            # But the engine might handle 2D inputs for classification
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image: {e}")

    # Run Inference
    report = engine.run_inference(
        image=image,
        image_id=sanitized_filename,
        run_classification=run_classification,
        run_segmentation=run_segmentation,
    )
    
    # Save Mask if exists
    mask_rel_path = None
    mask_png_rel_path = None
    if report.segmentation:
        mask_filename = f"mask_{sanitized_filename}.npy"
        mask_full_path = MASK_DIR / mask_filename
        np.save(mask_full_path, report.segmentation.combined_mask)
        mask_rel_path = str(mask_full_path)
        
        # Save PNG preview for frontend
        try:
             # Find slice with most tumor
             mask_data = report.segmentation.combined_mask
             if mask_data.ndim == 3: # (H, W, D)
                 # Sum across spatial dims to find max tumor
                 sums = mask_data.sum(axis=(0, 1))
                 z = np.argmax(sums) if sums.max() > 0 else mask_data.shape[2] // 2
                 mask_slice = mask_data[:, :, z]
             else:
                 mask_slice = mask_data
                 
             # Colorize
             # 0: BG, 1: NCR (Red), 2: ED (Green), 3: ET (Blue)
             # Map for visualization: NCR=Red, ED=Green, ET=Blue
             H, W = mask_slice.shape
             vis_mask = np.zeros((H, W, 4), dtype=np.uint8)
             
             # NCR (1) -> Red
             vis_mask[mask_slice == 1] = [0, 0, 255, 128] # BGRA
             # ED (2) -> Green
             vis_mask[mask_slice == 2] = [0, 255, 0, 128] 
             # ET (3) -> Blue
             vis_mask[mask_slice == 3] = [255, 0, 0, 128]
             
             mask_png_filename = f"mask_{sanitized_filename}.png"
             mask_png_path = MASK_DIR / mask_png_filename
             cv2.imwrite(str(mask_png_path), vis_mask)
             mask_png_rel_path = str(mask_png_path)
             
        except Exception as e:
            print(f"[API] Mask visualization failed: {e}")
        
    # Generate & Save GradCAM if classification was run
    gradcam_rel_path = None
    if report.classification and gradcam:
        try:
            # We need a tensor for GradCAM
            input_tensor = torch.from_numpy(image).unsqueeze(0).to(engine.device)
            # Handle dimension mismatch if needed (e.g. 2.5D)
            # engine.cls_model expects (B, S, C, H, W) or similar
            # This logic depends on dataset conventions. 
            # If loaded from .npy, it usually matches.
            # Only trying if shape matches loosely
            
            heatmap, pred_class, conf = gradcam(input_tensor)
            
            # Visualize on the middle slice or first channel
            if image.ndim == 4: # (C, H, W, D)
                vis_img = image[0, :, :, image.shape[3]//2]
            elif image.ndim == 3 and image.shape[0] > 4: # (S, C, H, W)?
                vis_img = image[0, 0] # First slice, first ch?
            else: # (C, H, W)
                vis_img = image[0] 
                
            overlay = visualize_gradcam(vis_img, heatmap)
            
            # Save overlay image
            gc_filename = f"gradcam_{sanitized_filename}.png"
            gc_full_path = GRADCAM_DIR / gc_filename
            
            # Convert to uint8 0-255 for saving
            overlay_uint8 = (overlay * 255).astype(np.uint8)
            overlay_bgr = cv2.cvtColor(overlay_uint8, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(gc_full_path), overlay_bgr)
            gradcam_rel_path = str(gc_full_path)
            
        except Exception as e:
            print(f"[API] Grad-CAM failed: {e}")

    # Save to DB
    record = save_prediction_to_db(
        db=db,
        report=report,
        image_path=str(file_path),
        mask_path=mask_png_rel_path or mask_rel_path,
        gradcam_path=gradcam_rel_path
    )
    
    # Formulate Response
    response = InferenceResponse(
        id=record.id,
        image_id=report.image_id,
        timestamp=report.timestamp.isoformat(),
        device=report.device,
        inference_time_ms=report.inference_time_ms,
        final_status="ready" if not report.validation.requires_manual_review else "manual_review",
        classification=classification_to_response(report.classification) if report.classification else None,
        segmentation=segmentation_to_response(report.segmentation, mask_path=mask_png_rel_path or mask_rel_path) if report.segmentation else None,
        validation=ValidationResponse(
            confidence_passed=report.validation.confidence_passed,
            segmentation_plausible=report.validation.segmentation_plausible,
            anatomical_plausibility=report.validation.anatomical_plausibility,
            mask_alignment_valid=report.validation.mask_alignment_valid,
            notes=report.validation.notes,
            requires_manual_review=report.validation.requires_manual_review,
        ),
        image_path=str(file_path),
        gradcam_path=gradcam_rel_path
    )
    
    return response

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
