"""
FootGuard AI - Experiment B: CAD_WALK Kinematic Walking Speed Regression
Models:
1. Baselines: Mean Prediction Baseline & Linear Feature Regression (Stance Duration + Peak Force)
2. Pressure 1D CNN
3. Pressure TCN

Evaluated on:
- 36 Training subjects (1,728 trials)
- 8 Validation subjects (384 trials)
- 7 Held-out Test subjects (288 valid trials)
Participant-level isolation: strictly zero overlap across splits.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge

ML_ROOT = Path(__file__).resolve().parent.parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from common.config import CHECKPOINTS_DIR, EXPERIMENTS_DIR, SEED
from common.reproducibility import set_seed
from common.logging import get_logger
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

logger = get_logger("experiment_b")


def compute_baseline_models(train_ds, val_ds, test_ds) -> dict:
    """
    Computes statistical and simple linear baselines:
    1. Mean Prediction: predicts train mean speed for all trials.
    2. Linear Ridge Regression: uses stance duration (ms) and peak force (N).
    """
    logger.info("Computing simple comparison baselines...")
    
    # Extract tabular features from metadata
    def get_features_and_targets(ds):
        X, y = [], []
        for i in range(len(ds)):
            _, target, meta = ds[i]
            # Features: duration_ms, peak_force_n
            X.append([meta["duration_ms"], meta["peak_force_n"]])
            y.append(target.item())
        return np.array(X), np.array(y)
        
    X_tr, y_tr = get_features_and_targets(train_ds)
    X_va, y_va = get_features_and_targets(val_ds)
    X_te, y_te = get_features_and_targets(test_ds)
    
    # 1. Mean Baseline
    train_mean = float(np.mean(y_tr))
    y_pred_mean_va = np.full_like(y_va, train_mean)
    y_pred_mean_te = np.full_like(y_te, train_mean)
    
    mean_val_metrics = compute_regression_metrics(y_va, y_pred_mean_va)
    mean_test_metrics = compute_regression_metrics(y_te, y_pred_mean_te)
    
    # 2. Linear Ridge Regression Baseline
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr, y_tr)
    y_pred_ridge_va = ridge.predict(X_va)
    y_pred_ridge_te = ridge.predict(X_te)
    
    ridge_val_metrics = compute_regression_metrics(y_va, y_pred_ridge_va)
    ridge_test_metrics = compute_regression_metrics(y_te, y_pred_ridge_te)
    
    return {
        "mean_baseline": {
            "train_mean_speed": train_mean,
            "validation": mean_val_metrics,
            "test": mean_test_metrics
        },
        "ridge_regression_baseline": {
            "validation": ridge_val_metrics,
            "test": ridge_test_metrics
        }
    }


def train_and_eval_neural_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: TrainingConfig
) -> dict:
    set_seed(config.seed)
    device = config.device
    logger.info(f"Training {model_name} on {device} (Epochs: {config.epochs}, BS: {config.batch_size})...")
    
    if model_name == "pressure_1d_cnn":
        model = Pressure1DCNN(in_channels=1, out_features=1, expected_seq_len=100).to(device)
    elif model_name == "pressure_tcn":
        model = PressureTCN(in_channels=1, tcn_channels=[32, 64], out_features=1, pooling="last").to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")
        
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)
    criterion = nn.MSELoss()
    
    ckpt_mgr = ModelCheckpointManager(CHECKPOINTS_DIR, config.experiment_name)
    tracker = ExperimentTracker(EXPERIMENTS_DIR, config.experiment_name)
    early_stopping = EarlyStopping(patience=config.patience, mode="min")
    
    best_val_mae = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_mae": []}
    
    for epoch in range(1, config.epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        for curves, speeds, _ in train_loader:
            curves = curves.to(device)
            speeds = speeds.to(device).unsqueeze(1)
            optimizer.zero_grad()
            preds = model(curves)
            loss = criterion(preds, speeds)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(speeds)
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_val_preds, all_val_targets = [], []
        with torch.no_grad():
            for curves, speeds, _ in val_loader:
                curves = curves.to(device)
                speeds = speeds.to(device).unsqueeze(1)
                preds = model(curves)
                loss = criterion(preds, speeds)
                val_loss += loss.item() * len(speeds)
                all_val_preds.extend(preds.cpu().numpy().flatten().tolist())
                all_val_targets.extend(speeds.cpu().numpy().flatten().tolist())
        val_loss /= len(val_loader.dataset)
        
        val_m = compute_regression_metrics(all_val_targets, all_val_preds)
        val_mae = val_m["mae"]
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        tracker.record_epoch(epoch, train_loss, val_loss, val_m)
        
        is_best = val_mae < best_val_mae
        if is_best:
            best_val_mae = val_mae
            ckpt_mgr.save_checkpoint(model, optimizer, epoch, val_m, config, is_best=True)
            
        if scheduler:
            scheduler.step(val_loss)
            
        logger.info(
            f"[{model_name}] Epoch {epoch:02d} | Train MSE: {train_loss:.4f} | "
            f"Val MSE: {val_loss:.4f} | Val MAE: {val_mae:.4f} | Val R2: {val_m['r2_score']:.4f}"
        )
        
        if early_stopping(val_loss):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break
            
    # Load best checkpoint for final evaluation
    logger.info(f"Evaluating {model_name} best checkpoint on held-out test set...")
    ckpt_mgr.load_checkpoint(model, load_best=True)
    model.eval()
    
    # 1. Validation metrics at best epoch
    val_preds_best, val_targets_best = [], []
    with torch.no_grad():
        for curves, speeds, _ in val_loader:
            preds = model(curves.to(device))
            val_preds_best.extend(preds.cpu().numpy().flatten().tolist())
            val_targets_best.extend(speeds.numpy().flatten().tolist())
    val_final_m = compute_regression_metrics(val_targets_best, val_preds_best)
    
    # 2. Test metrics (evaluated once)
    test_preds, test_targets = [], []
    with torch.no_grad():
        for curves, speeds, _ in test_loader:
            preds = model(curves.to(device))
            test_preds.extend(preds.cpu().numpy().flatten().tolist())
            test_targets.extend(speeds.numpy().flatten().tolist())
    test_m = compute_regression_metrics(test_targets, test_preds)
    
    return {
        "model": model_name,
        "best_val_mae": best_val_mae,
        "validation_metrics": val_final_m,
        "test_metrics": test_m,
        "history": history
    }


def run_experiment_b() -> dict:
    exp_dir = EXPERIMENTS_DIR / "experiment_b_cad_walk"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Initializing CAD_WALK datasets (Participant-level isolation)...")
    train_ds = CADWalkTemporalDataset(split="train", filter_valid_speed=True)
    val_ds = CADWalkTemporalDataset(split="val", filter_valid_speed=True)
    test_ds = CADWalkTemporalDataset(split="test", filter_valid_speed=True)
    
    logger.info(f"Dataset sizes: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    # 1. Compute Baselines
    baseline_results = compute_baseline_models(train_ds, val_ds, test_ds)
    
    # 2. Train Pressure 1D CNN
    config_cnn = TrainingConfig(
        experiment_name="experiment_b_pressure_1d_cnn",
        learning_rate=1e-3,
        weight_decay=1e-4,
        epochs=15,
        batch_size=32,
        patience=5
    )
    cnn_results = train_and_eval_neural_model("pressure_1d_cnn", train_loader, val_loader, test_loader, config_cnn)
    
    # 3. Train Pressure TCN
    config_tcn = TrainingConfig(
        experiment_name="experiment_b_pressure_tcn",
        learning_rate=1e-3,
        weight_decay=1e-4,
        epochs=15,
        batch_size=32,
        patience=5
    )
    tcn_results = train_and_eval_neural_model("pressure_tcn", train_loader, val_loader, test_loader, config_tcn)
    
    combined_results = {
        "experiment": "experiment_b_cad_walk_speed_regression",
        "dataset_summary": {
            "train_trials": len(train_ds),
            "val_trials": len(val_ds),
            "test_trials": len(test_ds),
            "excluded_unlabeled_c51_trials": 13
        },
        "baselines": baseline_results,
        "pressure_1d_cnn": cnn_results,
        "pressure_tcn": tcn_results
    }
    
    out_file = exp_dir / "experiment_b_results.json"
    with open(out_file, "w") as f:
        json.dump(combined_results, f, indent=2)
        
    logger.info(f"Experiment B complete. Results saved to {out_file}")
    return combined_results


if __name__ == "__main__":
    run_experiment_b()

