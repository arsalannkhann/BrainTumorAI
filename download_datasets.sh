#!/bin/bash
# =============================================================================
# Brain Tumor Dataset Download Script
# Downloads datasets from Kaggle for brain tumor classification and segmentation
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Directories
DATA_DIR="data"
RAW_DIR="$DATA_DIR/raw"
CLASSIFICATION_DIR="$DATA_DIR/classification_raw"

print_header() {
    echo ""
    echo "============================================================================="
    echo -e "${GREEN}$1${NC}"
    echo "============================================================================="
}

check_kaggle() {
    if ! command -v kaggle &> /dev/null; then
        echo -e "${RED}Error: Kaggle CLI not found${NC}"
        echo ""
        echo "Install it with:"
        echo "  pip install kaggle"
        echo ""
        echo "Then configure your API key:"
        echo "  1. Go to kaggle.com -> Account -> Create New API Token"
        echo "  2. Place kaggle.json in ~/.kaggle/"
        echo "  3. chmod 600 ~/.kaggle/kaggle.json"
        exit 1
    fi
    
    # Check if credentials exist
    if [ ! -f ~/.kaggle/kaggle.json ]; then
        echo -e "${RED}Error: Kaggle credentials not found${NC}"
        echo ""
        echo "Setup your Kaggle API:"
        echo "  1. Go to https://www.kaggle.com/account"
        echo "  2. Click 'Create New API Token'"
        echo "  3. Move the downloaded kaggle.json to ~/.kaggle/"
        echo "  4. Run: chmod 600 ~/.kaggle/kaggle.json"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Kaggle CLI configured${NC}"
}

# =============================================================================
# Dataset 1: Brain Tumor Classification (2D MRI Images)
# 7,023 images - Glioma, Meningioma, Pituitary, No Tumor
# =============================================================================
download_classification_dataset() {
    print_header "Downloading Brain Tumor Classification Dataset"
    check_kaggle
    
    echo "Dataset: Brain Tumor MRI Dataset"
    echo "Source: Kaggle (masoudnickparvar/brain-tumor-mri-dataset)"
    echo "Size: ~150 MB"
    echo "Classes: Glioma, Meningioma, Pituitary, No Tumor"
    echo ""
    
    mkdir -p "$CLASSIFICATION_DIR"
    
    kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset \
        -p "$CLASSIFICATION_DIR" --unzip
    
    echo -e "${GREEN}✓ Classification dataset downloaded to $CLASSIFICATION_DIR${NC}"
}

# =============================================================================
# Dataset 2: BraTS 2020 (3D MRI Volumes with Segmentation Masks)
# =============================================================================
download_brats_2020() {
    print_header "Downloading BraTS 2020 Dataset"
    check_kaggle
    
    echo "Dataset: BraTS 2020 Training Data"
    echo "Source: Kaggle (awsaf49/brats20-dataset-training-validation)"
    echo "Size: ~8 GB"
    echo "Contents: Multi-modal MRI (T1, T2, FLAIR, T1ce) + Segmentation masks"
    echo ""
    
    mkdir -p "$RAW_DIR"
    
    kaggle datasets download -d awsaf49/brats20-dataset-training-validation \
        -p "$RAW_DIR" --unzip
    
    echo -e "${GREEN}✓ BraTS 2020 dataset downloaded to $RAW_DIR${NC}"
}

# =============================================================================
# Dataset 3: BraTS 2021 (Alternative)
# =============================================================================
download_brats_2021() {
    print_header "Downloading BraTS 2021 Dataset"
    check_kaggle
    
    echo "Dataset: BraTS 2021 Task 1"
    echo "Source: Kaggle (dschettler8845/brats-2021-task1)"  
    echo "Size: ~10 GB"
    echo ""
    
    mkdir -p "$RAW_DIR"
    
    kaggle datasets download -d dschettler8845/brats-2021-task1 \
        -p "$RAW_DIR" --unzip
    
    echo -e "${GREEN}✓ BraTS 2021 dataset downloaded to $RAW_DIR${NC}"
}

# =============================================================================
# Organize data for pipeline
# =============================================================================
organize_brats_data() {
    print_header "Organizing BraTS Data"
    
    echo "Restructuring data for pipeline compatibility..."
    
    # Find BraTS training data
    BRATS_TRAINING=$(find "$RAW_DIR" -type d -name "BraTS*Training*" 2>/dev/null | head -1)
    
    if [ -z "$BRATS_TRAINING" ]; then
        BRATS_TRAINING=$(find "$RAW_DIR" -type d -name "*training*" -o -name "*Training*" 2>/dev/null | head -1)
    fi
    
    if [ -n "$BRATS_TRAINING" ]; then
        echo "Found BraTS training data at: $BRATS_TRAINING"
        
        # Count patients
        NUM_PATIENTS=$(find "$BRATS_TRAINING" -mindepth 1 -maxdepth 1 -type d | wc -l)
        echo "Found $NUM_PATIENTS patients"
        
        # Create symlinks or copy structure
        # The BraTS format is already compatible with our pipeline
        echo -e "${GREEN}✓ Data organized${NC}"
    else
        echo -e "${YELLOW}Warning: Could not locate BraTS training directory${NC}"
        echo "Please check $RAW_DIR and organize manually"
    fi
}

# =============================================================================
# Generate patient splits
# =============================================================================
generate_splits() {
    print_header "Generating Patient Splits"
    
    SPLITS_DIR="$DATA_DIR/splits"
    mkdir -p "$SPLITS_DIR"
    
    # Find patient directories
    BRATS_DIR=$(find "$RAW_DIR" -type d -name "BraTS*Training*" | head -1)
    
    if [ -z "$BRATS_DIR" ]; then
        echo -e "${YELLOW}Warning: No BraTS data found. Skipping split generation.${NC}"
        return
    fi
    
    # Get all patient IDs
    PATIENTS=($(find "$BRATS_DIR" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort))
    NUM_PATIENTS=${#PATIENTS[@]}
    
    echo "Total patients: $NUM_PATIENTS"
    
    # Split: 70% train, 15% val, 15% test
    TRAIN_END=$((NUM_PATIENTS * 70 / 100))
    VAL_END=$((NUM_PATIENTS * 85 / 100))
    
    # Shuffle patients (reproducible with seed)
    SEED=42
    SHUFFLED=($(printf '%s\n' "${PATIENTS[@]}" | shuf --random-source=<(yes $SEED | head -$NUM_PATIENTS)))
    
    # Write splits
    printf '%s\n' "${SHUFFLED[@]:0:$TRAIN_END}" > "$SPLITS_DIR/train.txt"
    printf '%s\n' "${SHUFFLED[@]:$TRAIN_END:$((VAL_END-TRAIN_END))}" > "$SPLITS_DIR/val.txt"
    printf '%s\n' "${SHUFFLED[@]:$VAL_END}" > "$SPLITS_DIR/test.txt"
    
    echo "Train: $TRAIN_END patients"
    echo "Val: $((VAL_END-TRAIN_END)) patients"
    echo "Test: $((NUM_PATIENTS-VAL_END)) patients"
    
    echo -e "${GREEN}✓ Splits saved to $SPLITS_DIR${NC}"
}

# =============================================================================
# Main
# =============================================================================

usage() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  classification   Download 2D classification dataset only"
    echo "  brats2020        Download BraTS 2020 segmentation dataset"
    echo "  brats2021        Download BraTS 2021 segmentation dataset"
    echo "  all              Download all datasets"
    echo "  organize         Organize downloaded data and generate splits"
    echo ""
    echo "Examples:"
    echo "  $0 classification    # Download classification dataset"
    echo "  $0 brats2020         # Download BraTS 2020"
    echo "  $0 all               # Download everything"
}

main() {
    print_header "Brain Tumor Dataset Downloader"
    
    # check_kaggle (Moved to individual functions)
    
    case "${1:-all}" in
        classification)
            download_classification_dataset
            ;;
        brats2020)
            download_brats_2020
            organize_brats_data
            generate_splits
            ;;
        brats2021)
            download_brats_2021
            organize_brats_data
            generate_splits
            ;;
        all)
            download_classification_dataset
            download_brats_2020
            organize_brats_data
            generate_splits
            ;;
        organize)
            organize_brats_data
            generate_splits
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
    
    print_header "Download Complete"
    
    echo ""
    echo "Dataset locations:"
    echo "  Classification: $CLASSIFICATION_DIR"
    echo "  Segmentation:   $RAW_DIR"
    echo "  Splits:         $DATA_DIR/splits/"
    echo ""
    echo "Next steps:"
    echo "  1. Run preprocessing: ./run_pipeline.sh preprocess"
    echo "  2. Train segmentation: ./run_pipeline.sh seg-train"
}

main "$@"
