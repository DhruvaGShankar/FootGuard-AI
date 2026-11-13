"""
FootGuard AI - Experiment C: IAC-TecMed Thermal Cold-Stress Rewarming Prediction
Cohort: N=19 Healthy Non-Diabetic Participants
- Train: 13 participants
- Val: 3 participants
- Test: 3 participants

Task:
Input: Chronological observations [T0, T5, T10] (Cold-stress induction and early rewarming).
Target: Plantar temperature at genuinely held-out later timepoint T15.

Baselines:
1. Persistence Baseline: uses T10 temperature as prediction for T15.
2. Mean Baseline: uses training mean T15 temperature.
3. Thermal TCN: spatial encoder + causal dilated TCN on [T0, T5, T10].
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ML_ROOT = Path(__file__).resolve().parent.parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from common.config import CHECKPOINTS_DIR, EXPERIMENTS_DIR, SEED, PROCESSED_THERMAL_DIR, SPLITS_THERMAL_DIR
from common.reproducibility import set_seed
from common.logging import get_logger
from common.trainer import TrainingConfig, EarlyStopping, ModelCheckpointManager, ExperimentTracker, build_optimizer_and_scheduler
from common.metrics import compute_regression_metrics
from thermal.tcn import ThermalTCN

logger = get_logger("experiment_c")


class IACPredictiveDataset(Dataset):
    """
    Dataset configuring earlier timepoints [T0, T5, T10] as input,
    and genuinely held-out T15 mean temperature as target.
    """
    def __init__(self, split: str = "train"):
        super().__init__()
        assert split in ["train", "val", "test"]
        self.split = split
        
        with open(SPLITS_THERMAL_DIR / "iac_tecmed_splits.json") as f:
            splits = json.load(f)
        self.allowed_subs = set(splits[split])
        
        manifest_path = PROCESSED_THERMAL_DIR / "iac_tecmed_manifest.csv"
        df = pd.read_csv(manifest_path)
        self.df = df[df["subject_id"].isin(self.allowed_subs)].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        file_path = PROCESSED_THERMAL_DIR.parent / row["file_path"]
        data = np.load(file_path)
        
        # full sequence is (4, 144, 192) for [T0, T5, T10, T15]
        seq = data["downsampled_sequence"]  # shape (4, 144, 192)
        
        # Input: first 3 timepoints [T0, T5, T10] -> shape (3, 1, 144, 192)
        input_seq = seq[:3]
        input_seq = np.expand_dims(input_seq, axis=1)  # (3, 1, 144, 192)
        
        # Normalize continuous Celsius temperature (mean 28.0, std 4.0)
        norm_seq = np.copy(input_seq)
        mask = norm_seq > 15.0
        norm_seq[mask] = (norm_seq[mask] - 28.0) / 4.0
        
        # Target: actual mean Celsius temperature at T15
        target_t15 = float(row["mean_temp_T15"])
        
        # Also return T10 mean for persistence baseline
        t10_mean = float(row["mean_temp_T10"])
        
        return (
            torch.from_numpy(norm_seq).float(),
            torch.tensor([target_t15], dtype=torch.float32),
            {
                "subject_id": str(data["subject_id"]),
                "t10_mean": t10_mean,
                "t15_target": target_t15
            }
        )


def compute_descriptive_and_simple_baselines(train_ds, val_ds, test_ds) -> dict:
    """Computes empirical rewarming stats and simple baselines."""
    manifest_path = PROCESSED_THERMAL_DIR / "iac_tecmed_manifest.csv"
    df_all = pd.read_csv(manifest_path)
    
    descriptive_stats = {
        "total_participants": len(df_all),
        "mean_temp_T0_celsius": float(df_all["mean_temp_T0"].mean()),
        "std_temp_T0_celsius": float(df_all["mean_temp_T0"].std()),
        "mean_temp_T5_celsius": float(df_all["mean_temp_T5"].mean()),
        "mean_temp_T10_celsius": float(df_all["mean_temp_T10"].mean()),
        "mean_temp_T15_celsius": float(df_all["mean_temp_T15"].mean()),
        "std_temp_T15_celsius": float(df_all["mean_temp_T15"].std()),
        "mean_rewarming_delta_T15_T5": float(df_all["rewarming_delta_T15_T5"].mean())
    }
    
    def eval_baseline(ds, baseline_type="persistence", train_mean=None):
        targets, preds = [], []
        for i in range(len(ds)):
            _, target, meta = ds[i]
            targets.append(target.item())
            if baseline_type == "persistence":
                preds.append(meta["t10_mean"])
            else:
                preds.append(train_mean)
        return compute_regression_metrics(targets, preds)
        
    train_mean_t15 = float(np.mean([train_ds[i][1].item() for i in range(len(train_ds))]))
    
    persist_val = eval_baseline(val_ds, "persistence")
    persist_test = eval_baseline(test_ds, "persistence")
    
    mean_val = eval_baseline(val_ds, "mean", train_mean=train_mean_t15)
    mean_test = eval_baseline(test_ds, "mean", train_mean=train_mean_t15)
    
    return {
        "descriptive_statistics": descriptive_stats,
        "persistence_baseline": {
            "validation": persist_val,
            "test": persist_test
        },
        "mean_baseline": {
            "train_mean_T15": train_mean_t15,
            "validation": mean_val,
            "test": mean_test
        }
    }


def run_experiment_c() -> dict:
    exp_name = "experiment_c_thermal_tcn"
    exp_dir = EXPERIMENTS_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    set_seed(SEED)
    logger.info("Initializing IAC-TecMed predictive datasets...")
    train_ds = IACPredictiveDataset(split="train")
    val_ds = IACPredictiveDataset(split="val")
    test_ds = IACPredictiveDataset(split="test")
    
    logger.info(f"Cohort splits: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    
    # 1. Descriptive & Simple Baselines
    baselines = compute_descriptive_and_simple_baselines(train_ds, val_ds, test_ds)
    
    # 2. Thermal TCN Model
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=3, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=3, shuffle=False)
    
    # Note: expected_seq_len=3 for [T0, T5, T10]
    model = ThermalTCN(
        in_channels=1,
        spatial_dim=32,
        tcn_channels=[32, 32],
        out_features=1,
        expected_seq_len=3
    ).to("cpu")
    
    config = TrainingConfig(
        experiment_name=exp_name,
        learning_rate=5e-4,
        weight_decay=1e-4,
        epochs=15,
        batch_size=4,
        patience=6,
        device="cpu"
    )
    
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)
    criterion = nn.MSELoss()
    ckpt_mgr = ModelCheckpointManager(CHECKPOINTS_DIR, exp_name)
    tracker = ExperimentTracker(EXPERIMENTS_DIR, exp_name)
    early_stopping = EarlyStopping(patience=config.patience, mode="min")
    
    best_val_mae = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_mae": []}
    
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        for seq, target, _ in train_loader:
            optimizer.zero_grad()
            pred = model(seq)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(target)
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for seq, target, _ in val_loader:
                pred = model(seq)
                loss = criterion(pred, target)
                val_loss += loss.item() * len(target)
                val_preds.extend(pred.flatten().tolist())
                val_targets.extend(target.flatten().tolist())
        val_loss /= len(val_loader.dataset)
        
        val_m = compute_regression_metrics(val_targets, val_preds)
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
            f"[Thermal TCN] Epoch {epoch:02d} | Train MSE: {train_loss:.4f} | "
            f"Val MSE: {val_loss:.4f} | Val MAE: {val_mae:.4f} C"
        )
        
        if early_stopping(val_loss):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break
            
    # Load best checkpoint and evaluate on held-out test set
    ckpt_mgr.load_checkpoint(model, load_best=True)
    model.eval()
    
    val_preds_best, val_targets_best = [], []
    with torch.no_grad():
        for seq, target, _ in val_loader:
            pred = model(seq)
            val_preds_best.extend(pred.flatten().tolist())
            val_targets_best.extend(target.flatten().tolist())
    val_final_m = compute_regression_metrics(val_targets_best, val_preds_best)
    
    test_preds, test_targets = [], []
    with torch.no_grad():
        for seq, target, _ in test_loader:
            pred = model(seq)
            test_preds.extend(pred.flatten().tolist())
            test_targets.extend(target.flatten().tolist())
    test_final_m = compute_regression_metrics(test_targets, test_preds)
    
    combined = {
        "experiment": exp_name,
        "sample_size_note": "N=19 total subjects (Train=13, Val=3, Test=3). Strictly exploratory vasomotor modeling.",
        "baselines": baselines,
        "thermal_tcn": {
            "validation_metrics": val_final_m,
            "test_metrics": test_final_m,
            "test_actual_vs_predicted": [
                {"actual_T15_celsius": float(a), "predicted_T15_celsius": float(p)}
                for a, p in zip(test_targets, test_preds)
            ],
            "history": history
        }
    }
    
    out_file = exp_dir / "experiment_c_results.json"
    with open(out_file, "w") as f:
        json.dump(combined, f, indent=2)
        
    logger.info(f"Experiment C complete. Results saved to {out_file}")
    return combined


if __name__ == "__main__":
    run_experiment_c()

