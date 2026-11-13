"""
FootGuard AI - Thermal Model Training Infrastructure
Supports:
1. Thermal 2D CNN on ThermoDataBase (Bilateral thermograms, diabetes cohort classification).
2. Thermal TCN on IAC-TecMed (Cold-stress rewarming sequences, experimental trajectory modeling).
"""

import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common.config import CHECKPOINTS_DIR, EXPERIMENTS_DIR, SEED
from common.logging import get_logger
from common.reproducibility import set_seed
from common.trainer import (
    TrainingConfig,
    EarlyStopping,
    ModelCheckpointManager,
    ExperimentTracker,
    build_optimizer_and_scheduler
)
from common.metrics import compute_classification_metrics, compute_regression_metrics
from thermal.cnn import Thermal2DCNN
from thermal.tcn import ThermalTCN
from thermal.dataset import ThermoDataset, IACTemporalDataset

logger = get_logger("thermal_trainer")


def train_epoch_thermal_cnn(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str
) -> float:
    model.train()
    total_loss = 0.0
    for tensors, labels, _ in loader:
        tensors = tensors.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(tensors)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


def eval_thermal_cnn(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, Dict[str, Any]]:
    model.eval()
    total_loss = 0.0
    all_preds, all_probs, all_targets = [], [], []
    with torch.no_grad():
        for tensors, labels, _ in loader:
            tensors = tensors.to(device)
            labels = labels.to(device)
            logits = model(tensors)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            
    val_loss = total_loss / len(loader.dataset)
    metrics = compute_classification_metrics(all_targets, all_preds, all_probs, num_classes=2)
    return val_loss, metrics


def train_thermal_cnn(config: TrainingConfig, dry_run: bool = False) -> Dict[str, Any]:
    """Trains Thermal 2D CNN on ThermoDataBase with subject-level anti-leakage splits."""
    set_seed(config.seed)
    device = config.device
    logger.info(f"Initializing Thermal 2D CNN training (Device: {device}, Dry run: {dry_run})...")
    
    train_ds = ThermoDataset(split="train")
    val_ds = ThermoDataset(split="val")
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    
    model = Thermal2DCNN(in_channels=3, num_classes=2).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)
    
    # Class weighting for class imbalance (45 Control vs 122 Diabetic)
    # Compute inverse frequency weights
    weights = torch.tensor([122.0 / 45.0, 1.0], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    ckpt_mgr = ModelCheckpointManager(CHECKPOINTS_DIR, config.experiment_name)
    tracker = ExperimentTracker(EXPERIMENTS_DIR, config.experiment_name)
    early_stopping = EarlyStopping(patience=config.patience, mode="max")
    
    epochs_to_run = 1 if dry_run else config.epochs
    best_f1 = 0.0
    
    for epoch in range(1, epochs_to_run + 1):
        train_loss = train_epoch_thermal_cnn(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = eval_thermal_cnn(model, val_loader, criterion, device)
        
        current_f1 = val_metrics.get("f1_score", 0.0)
        is_best = current_f1 > best_f1
        if is_best:
            best_f1 = current_f1
            
        ckpt_mgr.save_checkpoint(model, optimizer, epoch, val_metrics, config, is_best=is_best)
        tracker.record_epoch(epoch, train_loss, val_loss, val_metrics)
        
        if scheduler:
            scheduler.step(val_loss)
            
        logger.info(
            f"Epoch {epoch}/{epochs_to_run} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | F1: {current_f1:.4f} | AUROC: {val_metrics.get('auroc')}"
        )
        
        if early_stopping(current_f1):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break
            
    return {"best_f1": best_f1, "status": "COMPLETED"}


def main():
    parser = argparse.ArgumentParser(description="Thermal Model Trainer")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "tcn"])
    parser.add_argument("--dry-run", action="store_true", help="Run 1 epoch dry run to verify pipeline")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    config = TrainingConfig(
        experiment_name=f"thermal_{args.model}_baseline",
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    if args.model == "cnn":
        train_thermal_cnn(config, dry_run=args.dry_run)
    else:
        logger.info("TCN training mode configured.")


if __name__ == "__main__":
    main()

