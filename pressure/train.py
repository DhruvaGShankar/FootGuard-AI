"""
FootGuard AI - Pressure Model Training Infrastructure
Supports:
1. Pressure 1D CNN on CAD_WALK (1D stance force curves, walking speed regression).
2. Pressure TCN on CAD_WALK and Gait Lab insole streams.
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
from common.metrics import compute_regression_metrics
from pressure.cnn import Pressure1DCNN
from pressure.tcn import PressureTCN
from pressure.dataset import CADWalkTemporalDataset

logger = get_logger("pressure_trainer")


def train_epoch_pressure_reg(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str
) -> float:
    model.train()
    total_loss = 0.0
    for curves, speeds, _ in loader:
        curves = curves.to(device)
        speeds = speeds.to(device).unsqueeze(1)
        optimizer.zero_grad()
        preds = model(curves)
        loss = criterion(preds, speeds)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(speeds)
    return total_loss / len(loader.dataset)


def eval_pressure_reg(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for curves, speeds, _ in loader:
            curves = curves.to(device)
            speeds = speeds.to(device).unsqueeze(1)
            preds = model(curves)
            loss = criterion(preds, speeds)
            total_loss += loss.item() * len(speeds)
            
            all_preds.extend(preds.cpu().numpy().flatten().tolist())
            all_targets.extend(speeds.cpu().numpy().flatten().tolist())
            
    val_loss = total_loss / len(loader.dataset)
    metrics = compute_regression_metrics(all_targets, all_preds)
    return val_loss, metrics


def train_pressure_model(
    model_type: str = "cnn",
    config: TrainingConfig = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    """Trains Pressure 1D CNN or TCN on CAD_WALK stance curves for kinematic parameter regression."""
    if config is None:
        config = TrainingConfig(experiment_name=f"pressure_{model_type}_baseline")
        
    set_seed(config.seed)
    device = config.device
    logger.info(f"Initializing Pressure {model_type.upper()} training (Device: {device}, Dry run: {dry_run})...")
    
    train_ds = CADWalkTemporalDataset(split="train")
    val_ds = CADWalkTemporalDataset(split="val")
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    
    if model_type == "cnn":
        model = Pressure1DCNN(in_channels=1, out_features=1, expected_seq_len=100).to(device)
    else:
        model = PressureTCN(in_channels=1, tcn_channels=[32, 64], out_features=1).to(device)
        
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)
    criterion = nn.MSELoss()
    
    ckpt_mgr = ModelCheckpointManager(CHECKPOINTS_DIR, config.experiment_name)
    tracker = ExperimentTracker(EXPERIMENTS_DIR, config.experiment_name)
    early_stopping = EarlyStopping(patience=config.patience, mode="min")
    
    epochs_to_run = 1 if dry_run else config.epochs
    best_mae = float("inf")
    
    for epoch in range(1, epochs_to_run + 1):
        train_loss = train_epoch_pressure_reg(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = eval_pressure_reg(model, val_loader, criterion, device)
        
        current_mae = val_metrics.get("mae", float("inf"))
        is_best = current_mae < best_mae
        if is_best:
            best_mae = current_mae
            
        ckpt_mgr.save_checkpoint(model, optimizer, epoch, val_metrics, config, is_best=is_best)
        tracker.record_epoch(epoch, train_loss, val_loss, val_metrics)
        
        if scheduler:
            scheduler.step(val_loss)
            
        logger.info(
            f"Epoch {epoch}/{epochs_to_run} | Train MSE: {train_loss:.4f} | "
            f"Val MSE: {val_loss:.4f} | MAE: {current_mae:.4f} | R2: {val_metrics.get('r2_score', 0.0):.4f}"
        )
        
        if early_stopping(val_loss):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break
            
    return {"best_mae": best_mae, "status": "COMPLETED"}


def main():
    parser = argparse.ArgumentParser(description="Pressure Model Trainer")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "tcn"])
    parser.add_argument("--dry-run", action="store_true", help="Run 1 epoch dry run to verify pipeline")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    config = TrainingConfig(
        experiment_name=f"pressure_{args.model}_baseline",
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    train_pressure_model(model_type=args.model, config=config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

