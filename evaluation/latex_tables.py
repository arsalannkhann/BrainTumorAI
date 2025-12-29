"""
LaTeX table generation for publication.
Generates IEEE/Springer/MICCAI-style tables.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np


def format_metric(value: float, std: Optional[float] = None, precision: int = 3) -> str:
    """Format metric value with optional std."""
    if value is None:
        return "N/A"
    if std is not None:
        return f"{value:.{precision}f} $\\pm$ {std:.{precision}f}"
    return f"{value:.{precision}f}"


def format_ci(mean: float, lower: float, upper: float, precision: int = 3) -> str:
    """Format confidence interval."""
    return f"{mean:.{precision}f} [{lower:.{precision}f}, {upper:.{precision}f}]"


def generate_classification_table(
    metrics: Dict,
    class_names: List[str],
    output_path: Path,
    caption: str = "Classification Performance Metrics",
    label: str = "tab:classification",
) -> str:
    """
    Generate LaTeX table for classification metrics.
    
    Args:
        metrics: Dictionary of metrics
        class_names: List of class names
        output_path: Path to save table
        caption: Table caption
        label: Table label
        
    Returns:
        LaTeX table string
    """
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\textbf{Metric} & \\textbf{Value} \\\\",
        "\\midrule",
    ]
    
    # Overall metrics
    metric_display = {
        "accuracy": "Accuracy",
        "precision_macro": "Precision (Macro)",
        "precision_weighted": "Precision (Weighted)",
        "recall_macro": "Recall (Macro)",
        "recall_weighted": "Recall (Weighted)",
        "f1_macro": "F1-Score (Macro)",
        "f1_weighted": "F1-Score (Weighted)",
    }
    
    for key, display in metric_display.items():
        if key in metrics:
            val = metrics[key]
            lines.append(f"{display} & {format_metric(val)} \\\\")
    
    lines.extend([
        "\\midrule",
        "\\multicolumn{2}{l}{\\textbf{AUC-ROC}} \\\\",
    ])
    
    if "auc_roc_macro" in metrics:
        lines.append(f"Macro Average & {format_metric(metrics['auc_roc_macro'])} \\\\")
    if "auc_roc_weighted" in metrics:
        lines.append(f"Weighted Average & {format_metric(metrics['auc_roc_weighted'])} \\\\")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    
    table = "\n".join(lines)
    
    with open(output_path, "w") as f:
        f.write(table)
    
    return table


def generate_per_class_table(
    per_class_metrics: Dict[str, Dict],
    class_names: List[str],
    output_path: Path,
    caption: str = "Per-Class Classification Metrics",
    label: str = "tab:per_class",
) -> str:
    """
    Generate LaTeX table for per-class metrics.
    
    Args:
        per_class_metrics: Dictionary with per-class metrics
        class_names: List of class names
        output_path: Path to save table
        caption: Table caption
        label: Table label
        
    Returns:
        LaTeX table string
    """
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "\\textbf{Class} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1} & \\textbf{Specificity} & \\textbf{AUC} \\\\",
        "\\midrule",
    ]
    
    for name in class_names:
        if name in per_class_metrics:
            m = per_class_metrics[name]
            lines.append(
                f"{name.replace('_', ' ').title()} & "
                f"{format_metric(m.get('precision', 0))} & "
                f"{format_metric(m.get('recall', 0))} & "
                f"{format_metric(m.get('f1', 0))} & "
                f"{format_metric(m.get('specificity', 0))} & "
                f"{format_metric(m.get('auc', 0))} \\\\"
            )
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    
    table = "\n".join(lines)
    
    with open(output_path, "w") as f:
        f.write(table)
    
    return table


def generate_roc_auc_table(
    roc_data: Dict,
    output_path: Path,
    caption: str = "ROC-AUC Results",
    label: str = "tab:roc_auc",
) -> str:
    """
    Generate LaTeX table for ROC-AUC results.
    
    Args:
        roc_data: Dictionary with ROC data
        output_path: Path to save table
        caption: Table caption
        label: Table label
        
    Returns:
        LaTeX table string
    """
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lc}",
        "\\toprule",
        "\\textbf{Class/Average} & \\textbf{AUC-ROC} \\\\",
        "\\midrule",
    ]
    
    # Per-class AUC
    for name, data in roc_data.items():
        if name not in ["micro_average", "macro_average", "summary"]:
            lines.append(f"{name.replace('_', ' ').title()} & {format_metric(data['auc'])} \\\\")
    
    lines.append("\\midrule")
    
    # Averages
    if "micro_average" in roc_data:
        lines.append(f"Micro-Average & {format_metric(roc_data['micro_average']['auc'])} \\\\")
    if "macro_average" in roc_data:
        lines.append(f"Macro-Average & {format_metric(roc_data['macro_average']['auc'])} \\\\")
    if "summary" in roc_data:
        summary = roc_data["summary"]
        lines.append(f"Mean $\\pm$ Std & {format_metric(summary['mean_auc'], summary['std_auc'])} \\\\")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    
    table = "\n".join(lines)
    
    with open(output_path, "w") as f:
        f.write(table)
    
    return table


def generate_segmentation_table(
    seg_metrics: Dict[str, Dict],
    output_path: Path,
    caption: str = "Segmentation Performance Metrics",
    label: str = "tab:segmentation",
) -> str:
    """
    Generate LaTeX table for segmentation metrics.
    
    Args:
        seg_metrics: Dictionary with per-class segmentation metrics
        output_path: Path to save table
        caption: Table caption
        label: Table label
        
    Returns:
        LaTeX table string
    """
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "\\textbf{Region} & \\textbf{Dice} & \\textbf{IoU} & \\textbf{HD95 (mm)} & \\textbf{ASD (mm)} & \\textbf{Sensitivity} & \\textbf{Precision} \\\\",
        "\\midrule",
    ]
    
    # Per-class metrics
    region_order = ["edema", "enhancing", "necrotic", "whole_tumor", "tumor_core", "enhancing_tumor"]
    display_names = {
        "edema": "Edema (ED)",
        "enhancing": "Enhancing (ET)",
        "necrotic": "Necrotic (NCR)",
        "whole_tumor": "Whole Tumor (WT)",
        "tumor_core": "Tumor Core (TC)",
        "enhancing_tumor": "Enhancing Tumor (ET)*",
    }
    
    for region in region_order:
        if region in seg_metrics:
            m = seg_metrics[region]
            hd95 = format_metric(m.get("hd95")) if m.get("hd95") is not None else "N/A"
            asd = format_metric(m.get("asd")) if m.get("asd") is not None else "N/A"
            
            lines.append(
                f"{display_names.get(region, region)} & "
                f"{format_metric(m.get('dice', 0))} & "
                f"{format_metric(m.get('iou', 0))} & "
                f"{hd95} & "
                f"{asd} & "
                f"{format_metric(m.get('sensitivity', 0))} & "
                f"{format_metric(m.get('precision', 0))} \\\\"
            )
    
    # Mean metrics
    if "mean" in seg_metrics:
        m = seg_metrics["mean"]
        lines.append("\\midrule")
        lines.append(
            f"\\textbf{{Mean}} & "
            f"\\textbf{{{format_metric(m.get('dice', 0))}}} & "
            f"\\textbf{{{format_metric(m.get('iou', 0))}}} & "
            f"-- & -- & -- & -- \\\\"
        )
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\vspace{2mm}",
        "\\footnotesize{*ET in BraTS aggregate regions differs from individual ET class.}",
        "\\end{table}",
    ])
    
    table = "\n".join(lines)
    
    with open(output_path, "w") as f:
        f.write(table)
    
    return table


def generate_statistical_table(
    statistics: Dict,
    output_path: Path,
    caption: str = "Statistical Analysis Summary",
    label: str = "tab:statistics",
) -> str:
    """
    Generate LaTeX table for statistical analysis.
    
    Args:
        statistics: Dictionary with statistical results
        output_path: Path to save table
        caption: Table caption
        label: Table label
        
    Returns:
        LaTeX table string
    """
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "\\textbf{Metric} & \\textbf{Mean} & \\textbf{95\\% CI} & \\textbf{Std} \\\\",
        "\\midrule",
    ]
    
    for metric_name, data in statistics.items():
        if isinstance(data, dict) and "mean" in data:
            ci_str = ""
            if "ci_lower" in data and "ci_upper" in data:
                ci_str = f"[{data['ci_lower']:.3f}, {data['ci_upper']:.3f}]"
            
            lines.append(
                f"{metric_name.replace('_', ' ').title()} & "
                f"{format_metric(data['mean'])} & "
                f"{ci_str} & "
                f"{format_metric(data.get('std', 0))} \\\\"
            )
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    
    table = "\n".join(lines)
    
    with open(output_path, "w") as f:
        f.write(table)
    
    return table


def generate_reproducibility_table(
    details: Dict,
    output_path: Path,
    caption: str = "Reproducibility Details",
    label: str = "tab:reproducibility",
) -> str:
    """
    Generate LaTeX table for reproducibility details.
    
    Args:
        details: Dictionary with reproducibility information
        output_path: Path to save table
        caption: Table caption
        label: Table label
        
    Returns:
        LaTeX table string
    """
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{ll}",
        "\\toprule",
        "\\textbf{Parameter} & \\textbf{Value} \\\\",
        "\\midrule",
    ]
    
    display_order = [
        ("random_seed", "Random Seed"),
        ("train_samples", "Training Samples"),
        ("val_samples", "Validation Samples"),
        ("test_samples", "Test Samples"),
        ("model_parameters", "Model Parameters"),
        ("batch_size", "Batch Size"),
        ("learning_rate", "Learning Rate"),
        ("epochs", "Training Epochs"),
        ("optimizer", "Optimizer"),
        ("loss_function", "Loss Function"),
        ("hardware", "Hardware"),
        ("pytorch_version", "PyTorch Version"),
        ("monai_version", "MONAI Version"),
        ("inference_time_ms", "Inference Time (ms)"),
    ]
    
    for key, display in display_order:
        if key in details:
            value = details[key]
            if isinstance(value, float):
                value = f"{value:.6f}" if value < 0.01 else f"{value:.4f}"
            elif isinstance(value, int) and value > 1000:
                value = f"{value:,}"
            lines.append(f"{display} & {value} \\\\")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    
    table = "\n".join(lines)
    
    with open(output_path, "w") as f:
        f.write(table)
    
    return table


def generate_all_tables(
    classification_metrics: Dict,
    segmentation_metrics: Dict,
    statistics: Dict,
    reproducibility: Dict,
    class_names: List[str],
    output_dir: Path,
) -> Dict[str, str]:
    """
    Generate all LaTeX tables for publication.
    
    Args:
        classification_metrics: Classification metrics dictionary
        segmentation_metrics: Segmentation metrics dictionary
        statistics: Statistical analysis results
        reproducibility: Reproducibility details
        class_names: List of class names
        output_dir: Output directory
        
    Returns:
        Dictionary with table names and content
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tables = {}
    
    # Table 1: Classification Performance
    tables["classification"] = generate_classification_table(
        classification_metrics,
        class_names,
        output_dir / "table1_classification.tex",
    )
    
    # Table 2: ROC-AUC
    if "roc_data" in classification_metrics:
        tables["roc_auc"] = generate_roc_auc_table(
            classification_metrics["roc_data"],
            output_dir / "table2_roc_auc.tex",
        )
    
    # Table 3: Segmentation
    tables["segmentation"] = generate_segmentation_table(
        segmentation_metrics,
        output_dir / "table3_segmentation.tex",
    )
    
    # Table 4: Statistical Analysis
    tables["statistical"] = generate_statistical_table(
        statistics,
        output_dir / "table4_statistical.tex",
    )
    
    # Table 5: Reproducibility
    tables["reproducibility"] = generate_reproducibility_table(
        reproducibility,
        output_dir / "table5_reproducibility.tex",
    )
    
    return tables
