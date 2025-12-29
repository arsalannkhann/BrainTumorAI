"""
Utils package initialization.
"""

from .io import (
    create_labels_file,
    load_checkpoint,
    load_config,
    load_labels,
    load_multi_modal_nifti,
    load_nifti,
    load_patient_list,
    save_checkpoint,
    save_config,
    save_nifti,
    save_patient_list,
    validate_no_leakage,
)
from .logger import Logger, MetricTracker, get_logger
from .metrics import (
    compute_accuracy,
    compute_classification_metrics,
    compute_dice,
    compute_hausdorff_distance,
    compute_iou,
    compute_segmentation_metrics,
    get_classification_report,
    get_confusion_matrix,
    get_roc_curves,
)
from .seed import get_generator, set_seed, worker_init_fn

__all__ = [
    # Seed
    "set_seed",
    "get_generator",
    "worker_init_fn",
    # Metrics
    "compute_dice",
    "compute_iou",
    "compute_hausdorff_distance",
    "compute_segmentation_metrics",
    "compute_accuracy",
    "compute_classification_metrics",
    "get_confusion_matrix",
    "get_roc_curves",
    "get_classification_report",
    # Logging
    "Logger",
    "MetricTracker",
    "get_logger",
    # I/O
    "load_nifti",
    "save_nifti",
    "load_multi_modal_nifti",
    "load_patient_list",
    "save_patient_list",
    "validate_no_leakage",
    "load_config",
    "save_config",
    "save_checkpoint",
    "load_checkpoint",
    "load_labels",
    "create_labels_file",
]
