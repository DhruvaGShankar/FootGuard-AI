"""
FootGuard AI - Experiment A: Thermal 2D CNN on ThermoDataBase
Supervised Diabetic Cohort Classification (NOT Ulcer Detection)
- Training: 116 subjects with class weighting
- Validation: 25 subjects for early stopping and threshold tuning
- Test: 26 held-out subjects evaluated once with 1,000-iteration bootstrap CIs
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

# Ensure ML root is on sys.path
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
from common.metrics import compute_classification_metrics
from thermal.cnn import Thermal2DCNN
from thermal.dataset import ThermoDataset

logger = get_logger("experiment_a")


def bootstrap_classification_ci(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.5,
    n_bootstraps: int = 1000,
    seed: int = 42
) -> dict:
    """
    Computes 95% bootstrap confidence intervals for F1 and AUROC.
    """
    rng = np.random.RandomState(seed)
    f1_list = []
    auroc_list = []
    n = len(y_true)
    
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        boot_y = y_true[idx]
        boot_probs = y_probs[idx]
        
        # Only evaluate if both classes are present in sample
        if len(np.unique(boot_y)) > 1:
            boot_preds = (boot_probs[:, 1] >= threshold).astype(int)
            m = compute_classification_metrics(boot_y, boot_preds, boot_probs, num_classes=2)
            if m["f1_score"] is not None:
                f1_list.append(m["f1_score"])
            if m["auroc"] is not None:
                auroc_list.append(m["auroc"])
                
    f1_ci = [float(np.percentile(f1_list, 2.5)), float(np.percentile(f1_list, 97.5))] if f1_list else [0.0, 0.0]
    auroc_ci = [float(np.percentile(auroc_list, 2.5)), float(np.percentile(auroc_list, 97.5))] if auroc_list else [0.0, 0.0]
    
    return {
        "f1_95_ci": f1_ci,
        "auroc_95_ci": auroc_ci
    }


def run_experiment_a() -> dict:
    exp_name = "experiment_a_thermal_2d_cnn"
    exp_dir = EXPERIMENTS_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    config = TrainingConfig(
        experiment_name=exp_name,
        learning_rate=1e-3,
        weight_decay=1e-4,
        epochs=25,
        batch_size=16,
        patience=8,
        scheduler_patience=3
    )
    
    set_seed(config.seed)
    device = config.device
    logger.info(f"Starting Experiment A on {device} (Seed: {config.seed})...")
    
    train_ds = ThermoDataset(split="train")
    val_ds = ThermoDataset(split="val")
    test_ds = ThermoDataset(split="test")
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)
    
    # Class weighting: inverse frequency on train split only (31 Control, 85 Diabetic)
    weights = torch.tensor([85.0 / 31.0, 1.0], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    model = Thermal2DCNN(in_channels=3, num_classes=2).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)
    
    ckpt_mgr = ModelCheckpointManager(CHECKPOINTS_DIR, exp_name)
    tracker = ExperimentTracker(EXPERIMENTS_DIR, exp_name)
    early_stopping = EarlyStopping(patience=config.patience, mode="max")
    
    best_val_f1 = 0.0
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_auroc": []}
    
    for epoch in range(1, config.epochs + 1):
        # Training loop
        model.train()
        train_loss = 0.0
        for tensors, labels, _ in train_loader:
            tensors, labels = tensors.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(tensors)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
        train_loss /= len(train_loader.dataset)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        val_preds, val_probs, val_targets = [], [], []
        with torch.no_grad():
            for tensors, labels, _ in val_loader:
                tensors, labels = tensors.to(device), labels.to(device)
                logits = model(tensors)
                loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                val_preds.extend(preds.cpu().numpy().tolist())
                val_probs.extend(probs.cpu().numpy().tolist())
                val_targets.extend(labels.cpu().numpy().tolist())
        val_loss /= len(val_loader.dataset)
        
        val_m = compute_classification_metrics(val_targets, val_preds, val_probs, num_classes=2)
        val_f1 = val_m["f1_score"]
        val_auroc = val_m["auroc"]
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["val_auroc"].append(val_auroc)
        tracker.record_epoch(epoch, train_loss, val_loss, val_m)
        
        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            ckpt_mgr.save_checkpoint(model, optimizer, epoch, val_m, config, is_best=True)
            
        if scheduler:
            scheduler.step(val_loss)
            
        logger.info(
            f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val F1: {val_f1:.4f} | Val AUROC: {val_auroc if val_auroc else 'N/A'}"
        )
        
        if early_stopping(val_f1):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break
            
    # Load Best Checkpoint for Threshold Tuning & Test Evaluation
    logger.info("Loading best validation checkpoint...")
    ckpt_mgr.load_checkpoint(model, load_best=True)
    model.eval()
    
    # 1. Validation Threshold Tuning
    val_probs_all, val_targets_all = [], []
    with torch.no_grad():
        for tensors, labels, _ in val_loader:
            probs = model.predict_proba(tensors.to(device))
            val_probs_all.extend(probs.cpu().numpy())
            val_targets_all.extend(labels.numpy())
    val_probs_all = np.array(val_probs_all)
    val_targets_all = np.array(val_targets_all)
    
    best_thresh = 0.5
    best_thresh_f1 = 0.0
    for thresh in np.arange(0.10, 0.95, 0.05):
        t_preds = (val_probs_all[:, 1] >= thresh).astype(int)
        t_f1 = compute_classification_metrics(val_targets_all, t_preds)["f1_score"]
        if t_f1 > best_thresh_f1:
            best_thresh_f1 = t_f1
            best_thresh = float(thresh)
    logger.info(f"Optimal validation classification threshold: {best_thresh:.2f} (Val F1: {best_thresh_f1:.4f})")
    
    # 2. Final Test Set Evaluation (Evaluated Once)
    test_probs_all, test_targets_all = [], []
    with torch.no_grad():
        for tensors, labels, _ in test_loader:
            probs = model.predict_proba(tensors.to(device))
            test_probs_all.extend(probs.cpu().numpy())
            test_targets_all.extend(labels.numpy())
    test_probs_all = np.array(test_probs_all)
    test_targets_all = np.array(test_targets_all)
    
    # Evaluate at default threshold (0.5) and tuned threshold
    test_preds_default = np.argmax(test_probs_all, axis=1)
    test_preds_tuned = (test_probs_all[:, 1] >= best_thresh).astype(int)
    
    test_metrics_default = compute_classification_metrics(test_targets_all, test_preds_default, test_probs_all, num_classes=2)
    test_metrics_tuned = compute_classification_metrics(test_targets_all, test_preds_tuned, test_probs_all, num_classes=2)
    
    # Bootstrap Confidence Intervals (1,000 iterations)
    boot_ci = bootstrap_classification_ci(test_targets_all, test_probs_all, threshold=best_thresh, n_bootstraps=1000)
    
    results = {
        "experiment": exp_name,
        "optimal_threshold": best_thresh,
        "val_metrics_at_optimal_threshold": {
            "f1": best_thresh_f1
        },
        "test_metrics_default_0.5": test_metrics_default,
        "test_metrics_tuned_threshold": test_metrics_tuned,
        "test_bootstrap_95_ci": boot_ci,
        "class_distribution": {
            "train": {"control": 31, "diabetic": 85},
            "val": {"control": 7, "diabetic": 18},
            "test": {"control": 7, "diabetic": 19}
        },
        "history": history
    }
    
    out_results_file = exp_dir / "experiment_a_results.json"
    with open(out_results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Experiment A results saved to {out_results_file}")
    return results


if __name__ == "__main__":
    run_experiment_a()
