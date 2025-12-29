"""
Script to reorganize the results directory into a clean structure.
"""

import shutil
from pathlib import Path

def reorganize_results():
    root = Path("results")
    if not root.exists():
        print("Results directory does not exist.")
        return

    # Define new structure
    new_dirs = [
        root / "metrics",
        root / "plots" / "classification",
        root / "plots" / "segmentation",
        root / "tables",
        root / "predictions",
        root / "visualizations" / "gradcam",
        root / "visualizations" / "segmentation",
        root / "reports",
    ]
    
    for d in new_dirs:
        d.mkdir(parents=True, exist_ok=True)
        
    print("Created directory structure.")
    
    # 1. Classification Results (from verification run)
    eval_cls = root / "evaluation" / "classification"
    if eval_cls.exists():
        # Metrics
        if (eval_cls / "metrics.json").exists():
            shutil.copy2(eval_cls / "metrics.json", root / "metrics" / "classification_metrics.json")
            print("Moved classification metrics.")
            
        # Predictions
        if (eval_cls / "predictions.csv").exists():
            shutil.copy2(eval_cls / "predictions.csv", root / "predictions" / "classification_predictions.csv")
            print("Moved classification predictions.")
            
        # Plots (png/pdf)
        for ext in ["*.png", "*.pdf"]:
            for f in eval_cls.glob(ext):
                shutil.copy2(f, root / "plots" / "classification" / f.name)
        print("Moved classification plots.")
        
        # Tables
        tables_dir = eval_cls / "tables"
        if tables_dir.exists():
            for f in tables_dir.glob("*.tex"):
                shutil.copy2(f, root / "tables" / f.name)
            print("Moved classification tables.")
            
        # Reports
        if (eval_cls / "classification_report.txt").exists():
            shutil.copy2(eval_cls / "classification_report.txt", root / "reports" / "classification_report.txt")

    # 2. Paper Text
    paper_text = root / "evaluation" / "paper_text"
    if paper_text.exists():
        for f in paper_text.glob("*"):
            shutil.copy2(f, root / "reports" / f.name)
        print("Moved paper text.")

    print("\nReorganization complete. New structure:")
    for d in new_dirs:
        print(f"  {d}")

if __name__ == "__main__":
    reorganize_results()
