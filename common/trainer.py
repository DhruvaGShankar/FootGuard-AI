"""
FootGuard AI - Reusable Training Infrastructure
Provides:
1. TrainingConfig dataclass.
2. ModelCheckpointManager: atomic saving/loading of model checkpoints with metadata.
3. EarlyStopping: patience-based training termination.
4. ExperimentTracker: JSON / dictionary metric logging per epoch.
5. BaseTrainer: Standardized train/eval loop.
"""

import os
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Callable, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from common.config import CHECKPOINTS_DIR, EXPERIMENTS_DIR, SEED
from common.logging import get_logger
from common.reproducibility import set_seed
from common.metrics import compute_classification_metrics, compute_regression_metrics

logger = get_logger("trainer")


@dataclass
class TrainingConfig:
    experiment_name: str = "baseline"
    model_version: str = "1.0.0"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    batch_size: int = 16
    patience: int = 7
    min_delta: float = 1e-4
    optimizer_type: str = "adamw"  # adamw, adam, sgd
    scheduler_type: str = "plateau"  # plateau, cosine, none
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    grad_clip_norm: Optional[float] = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = SEED
    extra_params: Dict[str, Any] = field(default_factory=dict)


class EarlyStopping:
    """
    Early stopping monitor.
    Terminates training when monitored metric ceases improvement.
    """
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 1e-4,
        mode: str = "min"  # "min" for loss, "max" for F1/AUROC
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, current_score: float) -> bool:
        improved = self.step(current_score)
        return self.early_stop

    def step(self, current_score: float) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            return True  # Improved
            
        if self.mode == "min":
            improved = current_score < (self.best_score - self.min_delta)
        else:
            improved = current_score > (self.best_score + self.min_delta)
            
        if improved:
            self.best_score = current_score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


class ModelCheckpointManager:
    """
    Manages model checkpoint saving and loading with full provenance metadata.
    """
    def __init__(self, checkpoint_dir: Path, experiment_name: str):
        self.exp_dir = checkpoint_dir / experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.best_path = self.exp_dir / "best_model.pt"
        self.latest_path = self.exp_dir / "latest_model.pt"

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, Any],
        config: TrainingConfig,
        is_best: bool = False
    ) -> Path:
        """Saves checkpoint dictionary containing model weights and training state."""
        checkpoint = {
            "epoch": epoch,
            "model_version": config.model_version,
            "experiment_name": config.experiment_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": asdict(config),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        torch.save(checkpoint, self.latest_path)
        if is_best:
            torch.save(checkpoint, self.best_path)
            logger.info(f"New best model checkpoint saved to {self.best_path}")
        return self.best_path if is_best else self.latest_path

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        load_best: bool = True
    ) -> Dict[str, Any]:
        """Loads checkpoint into model and optional optimizer."""
        path = self.best_path if load_best else self.latest_path
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Loaded checkpoint from {path} (Epoch {checkpoint.get('epoch', 'N/A')})")
        return checkpoint


class ExperimentTracker:
    """
    Tracks and persists training history curves and per-epoch metrics.
    """
    def __init__(self, output_dir: Path, experiment_name: str):
        self.exp_dir = output_dir / experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.exp_dir / "training_history.json"
        self.history: Dict[str, List[Any]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "metrics": []
        }

    def record_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_metrics: Dict[str, Any]
    ) -> None:
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(float(train_loss))
        self.history["val_loss"].append(float(val_loss))
        self.history["metrics"].append(val_metrics)
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2)


def build_optimizer_and_scheduler(
    model: nn.Module,
    config: TrainingConfig
) -> Tuple[torch.optim.Optimizer, Optional[Any]]:
    """Builds configured optimizer and learning rate scheduler."""
    if config.optimizer_type.lower() == "adamw":
        optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer_type.lower() == "adam":
        optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer_type.lower() == "sgd":
        optimizer = SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer_type}")

    if config.scheduler_type.lower() == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=1e-6
        )
    elif config.scheduler_type.lower() == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    else:
        scheduler = None

    return optimizer, scheduler

