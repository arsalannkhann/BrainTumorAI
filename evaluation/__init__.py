"""
Evaluation module for publication-grade brain tumor analysis.
"""

from evaluation.advanced_metrics import (
    bootstrap_confidence_interval,
    cohens_d,
    compute_average_surface_distance,
    compute_brats_metrics,
    compute_comprehensive_segmentation_metrics,
    compute_hausdorff_distance_95,
    compute_micro_macro_roc,
    compute_pr_curves,
    compute_specificity,
    compute_volume_metrics,
    expected_calibration_error,
    maximum_calibration_error,
    multiclass_brier_score,
    multiclass_ece,
    wilcoxon_test,
)

from evaluation.visualizations import (
    plot_confusion_matrix_publication,
    plot_dice_boxplot,
    plot_pr_curves_publication,
    plot_reliability_diagram,
    plot_roc_curves_publication,
    plot_segmentation_overlay,
    plot_threshold_analysis,
    plot_training_curves,
)

from evaluation.latex_tables import (
    generate_all_tables,
    generate_classification_table,
    generate_per_class_table,
    generate_reproducibility_table,
    generate_roc_auc_table,
    generate_segmentation_table,
    generate_statistical_table,
)

__all__ = [
    # Metrics
    "bootstrap_confidence_interval",
    "cohens_d",
    "compute_average_surface_distance",
    "compute_brats_metrics",
    "compute_comprehensive_segmentation_metrics",
    "compute_hausdorff_distance_95",
    "compute_micro_macro_roc",
    "compute_pr_curves",
    "compute_specificity",
    "compute_volume_metrics",
    "expected_calibration_error",
    "maximum_calibration_error",
    "multiclass_brier_score",
    "multiclass_ece",
    "wilcoxon_test",
    # Visualizations
    "plot_confusion_matrix_publication",
    "plot_dice_boxplot",
    "plot_pr_curves_publication",
    "plot_reliability_diagram",
    "plot_roc_curves_publication",
    "plot_segmentation_overlay",
    "plot_threshold_analysis",
    "plot_training_curves",
    # LaTeX Tables
    "generate_all_tables",
    "generate_classification_table",
    "generate_per_class_table",
    "generate_reproducibility_table",
    "generate_roc_auc_table",
    "generate_segmentation_table",
    "generate_statistical_table",
]
