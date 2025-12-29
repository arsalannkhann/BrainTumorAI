#!/bin/bash
# =============================================================================
# Brain Tumor AI Pipeline
# End-to-end brain tumor detection from MRI
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
DATA_RAW="data/raw"
DATA_PROCESSED="data/processed"
DATA_MASKS="data/masks"
DATA_ROI="data/roi"
DATA_SPLITS="data/splits"

SEG_CONFIG="configs/seg.yaml"
CLS_CONFIG="configs/cls.yaml"

SEG_CHECKPOINT_DIR="checkpoints/segmentation"
CLS_CHECKPOINT_DIR="checkpoints/classification"

RESULTS_DIR="results"

# =============================================================================
# Helper functions
# =============================================================================

print_header() {
    echo ""
    echo "============================================================================="
    echo -e "${GREEN}$1${NC}"
    echo "============================================================================="
    echo ""
}

print_step() {
    echo -e "${YELLOW}>>> $1${NC}"
}

check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    else
        echo "Warning: No GPU detected. Training will be slow."
    fi
}

# =============================================================================
# Pipeline stages
# =============================================================================

stage_preprocessing() {
    print_header "Stage 1: Preprocessing"
    
    print_step "Running N4 bias correction and Z-score normalization..."
    
    venv/bin/python -m preprocessing.preprocess_pipeline \
        --raw-dir "$DATA_RAW" \
        --output-dir "$DATA_PROCESSED" \
        --modalities t1 t2 flair t1ce \
        --normalize-method zscore \
        --num-workers 4 \
        --suffix .nii
    
    echo -e "${GREEN}✓ Preprocessing complete${NC}"
}

stage_segmentation_train() {
    print_header "Stage 2: Segmentation Training"
    
    print_step "Training 3D UNet for tumor segmentation..."
    
    venv/bin/python -m segmentation.train \
        --config "$SEG_CONFIG"
    
    echo -e "${GREEN}✓ Segmentation training complete${NC}"
}

stage_segmentation_infer() {
    print_header "Stage 3: Segmentation Inference"
    
    SEG_BEST="$SEG_CHECKPOINT_DIR/best_model.pt"
    
    if [ ! -f "$SEG_BEST" ]; then
        echo -e "${RED}Error: Segmentation checkpoint not found: $SEG_BEST${NC}"
        exit 1
    fi
    
    print_step "Running segmentation inference on all patients..."
    
    # Inference on train, val, and test sets
    for split in train val test; do
        print_step "Processing $split set..."
        venv/bin/python -m segmentation.infer \
            --checkpoint "$SEG_BEST" \
            --patient-list "$DATA_SPLITS/$split.txt" \
            --processed-dir "$DATA_PROCESSED" \
            --output-dir "$DATA_MASKS" \
            --config "$SEG_CONFIG"
    done
    
    echo -e "${GREEN}✓ Segmentation inference complete${NC}"
}

stage_roi_extraction() {
    print_header "Stage 4: ROI Extraction"
    
    print_step "Extracting tumor ROIs from segmentation masks..."
    
    for split in train val test; do
        print_step "Processing $split set..."
        venv/bin/python -m roi_extraction.extract_roi \
            --patient-list "$DATA_SPLITS/$split.txt" \
            --processed-dir "$DATA_PROCESSED" \
            --masks-dir "$DATA_MASKS" \
            --output-dir "$DATA_ROI" \
            --margin 10
    done
    
    echo -e "${GREEN}✓ ROI extraction complete${NC}"
}

stage_classification_train() {
    print_header "Stage 5: Classification Training"
    
    print_step "Training ConvNeXt classifier..."
    
    venv/bin/python -m classification.train \
        --config "$CLS_CONFIG"
    
    echo -e "${GREEN}✓ Classification training complete${NC}"
}

stage_evaluation() {
    print_header "Stage 6: Evaluation"
    
    CLS_BEST="$CLS_CHECKPOINT_DIR/best_model.pt"
    
    if [ ! -f "$CLS_BEST" ]; then
        echo -e "${RED}Error: Classification checkpoint not found: $CLS_BEST${NC}"
        exit 1
    fi
    
    print_step "Evaluating on test set..."
    
    venv/bin/python -m classification.evaluate \
        --checkpoint "$CLS_BEST" \
        --patient-list "$DATA_SPLITS/test.txt" \
        --roi-dir "$DATA_ROI" \
        --labels-file "data/labels.csv" \
        --output-dir "$RESULTS_DIR/test_evaluation" \
        --config "$CLS_CONFIG"
    
    echo -e "${GREEN}✓ Evaluation complete${NC}"
}

stage_mae_pretrain() {
    print_header "Stage: Self-Supervised Pretraining (BM-MAE)"
    
    print_step "Training masked autoencoder on MRI volumes..."
    
    venv/bin/python -m pretraining.train_mae \
        --config configs/mae.yaml
    
    echo -e "${GREEN}✓ MAE pretraining complete${NC}"
    echo "Encoder weights saved to: checkpoints/mae/encoder_weights.pt"
}

# =============================================================================
# Main
# =============================================================================

main() {
    print_header "Brain Tumor AI Pipeline"
    
    echo "Checking environment..."
    check_gpu
    echo ""
    
    # Parse arguments
    STAGE="${1:-all}"
    
    case "$STAGE" in
        preprocess)
            stage_preprocessing
            ;;
        mae-pretrain)
            stage_mae_pretrain
            ;;
        seg-train)
            stage_segmentation_train
            ;;
        seg-infer)
            stage_segmentation_infer
            ;;
        roi)
            stage_roi_extraction
            ;;
        cls-train)
            stage_classification_train
            ;;
        evaluate)
            stage_evaluation
            ;;
        all)
            stage_preprocessing
            stage_segmentation_train
            stage_segmentation_infer
            stage_roi_extraction
            stage_classification_train
            stage_evaluation
            ;;
        all-with-mae)
            stage_preprocessing
            stage_mae_pretrain
            stage_segmentation_train
            stage_segmentation_infer
            stage_roi_extraction
            stage_classification_train
            stage_evaluation
            ;;
        *)
            echo "Usage: $0 {preprocess|mae-pretrain|seg-train|seg-infer|roi|cls-train|evaluate|all|all-with-mae}"
            echo ""
            echo "Stages:"
            echo "  preprocess    - Run N4 bias correction and normalization"
            echo "  mae-pretrain  - Self-supervised MAE pretraining (optional)"
            echo "  seg-train     - Train segmentation model (UNet/SegMamba)"
            echo "  seg-infer     - Run segmentation inference"
            echo "  roi           - Extract tumor ROIs"
            echo "  cls-train     - Train classification model (ConvNeXt/Vim/TransMIL)"
            echo "  evaluate      - Evaluate on test set"
            echo "  all           - Run full pipeline (without MAE)"
            echo "  all-with-mae  - Run full pipeline with MAE pretraining"
            exit 1
            ;;
    esac
    
    print_header "Pipeline Complete"
    echo "Results saved to: $RESULTS_DIR"
}

main "$@"

