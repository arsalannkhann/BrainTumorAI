"""
Logging utilities for training and experiment tracking.
Supports console logging, file logging, and optional W&B integration.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

# Optional W&B import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class Logger:
    """
    Unified logger for console, file, and W&B logging.
    """
    
    def __init__(
        self,
        name: str = "brain_tumor_ai",
        log_dir: Optional[Union[str, Path]] = None,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        level: int = logging.INFO,
    ):
        """
        Initialize the logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            use_wandb: Whether to use W&B logging
            wandb_project: W&B project name
            wandb_config: Configuration dict to log to W&B
            level: Logging level
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers = []  # Clear existing handlers
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_format = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"{name}_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_format = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)
            self.log_file = log_file
        else:
            self.log_file = None
        
        # W&B integration
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            if not WANDB_AVAILABLE:
                self.logger.warning("W&B requested but not installed. Skipping.")
                self.use_wandb = False
            else:
                wandb.init(
                    project=wandb_project or name,
                    config=wandb_config,
                    reinit=True,
                )
                self.logger.info(f"W&B initialized: {wandb.run.name}")
    
    def info(self, message: str) -> None:
        """Log info level message."""
        self.logger.info(message)
    
    def debug(self, message: str) -> None:
        """Log debug level message."""
        self.logger.debug(message)
    
    def warning(self, message: str) -> None:
        """Log warning level message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error level message."""
        self.logger.error(message)
    
    def critical(self, message: str) -> None:
        """Log critical level message."""
        self.logger.critical(message)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        """
        Log metrics to console and W&B.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Training step or epoch
            prefix: Prefix for metric names (e.g., 'train/', 'val/')
        """
        # Format for console
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        step_str = f"[Step {step}] " if step is not None else ""
        self.logger.info(f"{step_str}{prefix}{metrics_str}")
        
        # Log to W&B
        if self.use_wandb:
            wandb_metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
            if step is not None:
                wandb.log(wandb_metrics, step=step)
            else:
                wandb.log(wandb_metrics)
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """
        Log configuration to file and W&B.
        
        Args:
            config: Configuration dictionary
        """
        config_str = yaml.dump(config, default_flow_style=False)
        self.logger.info(f"Configuration:\n{config_str}")
        
        if self.use_wandb:
            wandb.config.update(config)
    
    def log_image(
        self,
        key: str,
        image: Any,
        step: Optional[int] = None,
        caption: Optional[str] = None,
    ) -> None:
        """
        Log image to W&B.
        
        Args:
            key: Image key/name
            image: Image array (numpy or PIL)
            step: Training step
            caption: Image caption
        """
        if self.use_wandb:
            wandb.log({key: wandb.Image(image, caption=caption)}, step=step)
    
    def finish(self) -> None:
        """Finish logging and close W&B run."""
        if self.use_wandb:
            wandb.finish()
        
        self.logger.info("Logging finished.")


def get_logger(name: str = "brain_tumor_ai") -> logging.Logger:
    """
    Get a simple logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] [%(levelname)s] %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class MetricTracker:
    """
    Tracks metrics over training epochs with best value tracking.
    """
    
    def __init__(self, metrics: list[str], mode: str = "max"):
        """
        Initialize metric tracker.
        
        Args:
            metrics: List of metric names to track
            mode: 'max' or 'min' for best value tracking
        """
        self.metrics = metrics
        self.mode = mode
        self.history: Dict[str, list[float]] = {m: [] for m in metrics}
        self.best_values: Dict[str, float] = {}
        self.best_epoch: Dict[str, int] = {}
        
        for m in metrics:
            if mode == "max":
                self.best_values[m] = float("-inf")
            else:
                self.best_values[m] = float("inf")
            self.best_epoch[m] = -1
    
    def update(self, values: Dict[str, float], epoch: int) -> Dict[str, bool]:
        """
        Update metrics with new values.
        
        Args:
            values: Dictionary of metric values
            epoch: Current epoch
            
        Returns:
            Dictionary indicating if each metric improved
        """
        improved = {}
        
        for metric, value in values.items():
            if metric in self.history:
                self.history[metric].append(value)
                
                is_better = (
                    (self.mode == "max" and value > self.best_values[metric])
                    or (self.mode == "min" and value < self.best_values[metric])
                )
                
                if is_better:
                    self.best_values[metric] = value
                    self.best_epoch[metric] = epoch
                    improved[metric] = True
                else:
                    improved[metric] = False
        
        return improved
    
    def get_best(self, metric: str) -> tuple[float, int]:
        """Get best value and epoch for a metric."""
        return self.best_values.get(metric, 0.0), self.best_epoch.get(metric, -1)
