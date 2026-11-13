"""
FootGuard AI - Model Training Service
Orchestrates local model training across real research datasets and synthetic sessions.
Reuses existing PyTorch datasets, preprocessing contracts, models, EarlyStopping,
and ModelCheckpointManager without modifying existing Phase 4 checkpoints.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Tuple
import json
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import Ridge

from common.reproducibility import set_seed
from common.trainer import EarlyStopping, TrainingConfig, build_optimizer_and_scheduler
from common.metrics import compute_classification_metrics, compute_regression_metrics
from services.config_service import ConfigService
from services.dataset_service import DatasetService
from services.validation_service import ValidationService
from services.model_service import ModelService
from services.experiment_service import ExperimentService
from thermal.dataset import ThermoDataset, IACTemporalDataset
from pressure.dataset import CADWalkTemporalDataset


class SyntheticPressureSequenceDataset(Dataset):
    """
    Session-isolated synthetic pressure sequence dataset built from chunk-sampled
    sessions of footguard_synthetic_paired_readings.csv.
    """

    PRESSURE_COLS = [
        "heel_pressure_n",
        "mid_pressure_n",
        "mth1_pressure_n",
        "mth3_pressure_n",
        "mth5_pressure_n",
        "toe_hallux_pressure_n",
    ]

    def __init__(self, windows: np.ndarray, targets: np.ndarray):
        self.X = torch.from_numpy(windows).float()  # (N, 6, seq_len)
        self.y = torch.from_numpy(targets).float()  # (N,)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class TrainingService:
    """Executes reproducible local training runs and persists checkpoints + experiment JSON."""

    def __init__(
        self,
        config_service: Optional[ConfigService] = None,
        dataset_service: Optional[DatasetService] = None,
        validation_service: Optional[ValidationService] = None,
        model_service: Optional[ModelService] = None,
        experiment_service: Optional[ExperimentService] = None,
    ):
        self.cfg = config_service or ConfigService()
        self.dataset_service = dataset_service or DatasetService(self.cfg)
        self.validation_service = validation_service or ValidationService(self.dataset_service)
        self.model_service = model_service or ModelService(self.cfg)
        self.experiment_service = experiment_service or ExperimentService(self.cfg)
        self.checkpoints_dir = self.cfg.get_path("checkpoints_dir")
        self.logs_dir = self.cfg.get_path("experiments_logs_dir")

    def _build_synthetic_datasets(
        self,
        seed: int = 42,
        n_sample_rows: int = 1200,
        seq_len: int = 16,
    ) -> Tuple[Dataset, Dataset, Dataset, str]:
        """Samples synthetic sessions and builds session-isolated train/val/test window datasets."""
        df = self.dataset_service.sample_synthetic_rows(n_rows=n_sample_rows, seed=seed)
        sessions = sorted(df["session_id"].dropna().unique().tolist())
        if len(sessions) < 3:
            raise ValueError(f"Need at least 3 synthetic sessions for train/val/test split, got {len(sessions)}.")

        rng = np.random.default_rng(seed)
        rng.shuffle(sessions)
        n_train = max(1, int(0.70 * len(sessions)))
        n_val = max(1, int(0.15 * len(sessions)))
        train_sids = set(sessions[:n_train])
        val_sids = set(sessions[n_train : n_train + n_val])
        test_sids = set(sessions[n_train + n_val :]) or val_sids

        def make_split(target_sids: set) -> SyntheticPressureSequenceDataset:
            windows: List[np.ndarray] = []
            targets: List[float] = []
            sub = df[df["session_id"].isin(target_sids)]
            for _, grp in sub.groupby("session_id"):
                arr = grp[SyntheticPressureSequenceDataset.PRESSURE_COLS].to_numpy(dtype=np.float32) / 800.0
                tgt = grp["plantar_total_load_n"].to_numpy(dtype=np.float32)
                if len(arr) < seq_len:
                    # Pad short sampled session segment symmetrically to seq_len
                    pad_len = seq_len - len(arr)
                    arr = np.pad(arr, ((0, pad_len), (0, 0)), mode="edge")
                    tgt = np.pad(tgt, (0, pad_len), mode="edge")
                for start in range(0, len(arr) - seq_len + 1, max(1, seq_len // 2)):
                    win = arr[start : start + seq_len].T  # (6, seq_len)
                    windows.append(win)
                    targets.append(float(np.mean(tgt[start : start + seq_len])))
            if not windows:
                raise ValueError("No valid synthetic windows constructed for split.")
            return SyntheticPressureSequenceDataset(np.stack(windows), np.array(targets, dtype=np.float32))

        split_desc = f"Session-isolated split across {len(sessions)} sampled synthetic sessions (seed={seed})"
        return make_split(train_sids), make_split(val_sids), make_split(test_sids), split_desc

    PROTECTED_CHECKPOINT_DIRS = {
        "experiment_a_thermal_2d_cnn",
        "experiment_b_pressure_1d_cnn",
        "experiment_b_pressure_tcn",
        "experiment_c_thermal_tcn",
        "thermal_cnn_baseline",
        "pressure_cnn_baseline",
        "pressure_tcn_baseline",
    }

    def start_training(
        self,
        model_id: str,
        dataset_id: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        experiment_id: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Trains a model on a validated dataset, saves checkpoint + logs, and persists experiment JSON.
        """
        compat = self.validation_service.check_model_compatibility(dataset_id, model_id)
        if not compat.is_compatible:
            raise ValueError(f"Training blocked: {compat.reason}")

        defaults = self.cfg.get_training_defaults()
        hp = {**defaults, **(hyperparameters or {})}
        seed = int(hp.get("seed", 42))
        epochs = int(hp.get("epochs", 5))
        batch_size = int(hp.get("batch_size", 16))
        lr = float(hp.get("learning_rate", 1e-3))
        weight_decay = float(hp.get("weight_decay", 1e-4))
        patience = int(hp.get("patience", 5))
        device = str(hp.get("device", "cpu"))
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        set_seed(seed)

        ts_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        exp_id = experiment_id or f"exp_{model_id}_{dataset_id}_{ts_tag}"
        if exp_id in self.PROTECTED_CHECKPOINT_DIRS:
            raise ValueError(f"Cannot overwrite protected Phase 4/5 checkpoint directory '{exp_id}'.")
        ckpt_dir = self.checkpoints_dir / exp_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.logs_dir / f"{exp_id}.log"

        desc = self.dataset_service.get_descriptor(dataset_id)
        spec = self.model_service.get_model_spec(model_id)

        # Special case: Ridge regression baseline on CAD_WALK
        if model_id == "pressure_ridge_baseline":
            return self._train_ridge(
                exp_id=exp_id,
                ckpt_dir=ckpt_dir,
                log_path=log_path,
                hp=hp,
                desc=desc,
                spec=spec,
            )

        # Build datasets preserving exact preprocessing contracts
        in_channels = None
        if dataset_id == "thermodatabase":
            train_ds = ThermoDataset(split="train", normalize=False)
            val_ds = ThermoDataset(split="val", normalize=False)
            test_ds = ThermoDataset(split="test", normalize=False)
            split_desc = desc.split_methodology
        elif dataset_id == "iac_tecmed":
            train_ds = IACTemporalDataset(split="train")
            val_ds = IACTemporalDataset(split="val")
            test_ds = IACTemporalDataset(split="test")
            split_desc = desc.split_methodology
        elif dataset_id == "cad_walk":
            train_ds = CADWalkTemporalDataset(split="train", filter_valid_speed=True)
            val_ds = CADWalkTemporalDataset(split="val", filter_valid_speed=True)
            test_ds = CADWalkTemporalDataset(split="test", filter_valid_speed=True)
            split_desc = desc.split_methodology
        elif dataset_id == "synthetic_paired_readings":
            n_rows = int(hp.get("synthetic_sample_rows", 1200))
            train_ds, val_ds, test_ds, split_desc = self._build_synthetic_datasets(seed=seed, n_sample_rows=n_rows)
            in_channels = 6
        else:
            raise ValueError(f"Training not implemented for dataset '{dataset_id}'.")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

        model = self.model_service.build_model(model_id, in_channels=in_channels).to(device)
        train_cfg = TrainingConfig(
            experiment_name=exp_id,
            learning_rate=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
        )
        optimizer, scheduler = build_optimizer_and_scheduler(model, train_cfg)

        is_classification = spec.task_type == "classification"
        if is_classification:
            # Compute training-only class weights to avoid data leakage
            labels = [int(train_ds[i][1].item()) for i in range(len(train_ds))]
            counts = np.bincount(labels, minlength=2)
            weights = len(labels) / (2.0 * np.maximum(counts, 1))
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
            early_stopping = EarlyStopping(patience=patience, mode="min")
        else:
            criterion = nn.MSELoss()
            early_stopping = EarlyStopping(patience=patience, mode="min")

        best_val_loss = float("inf")
        best_ckpt_path = ckpt_dir / "best_model.pt"
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        log_lines: List[str] = [f"Starting {exp_id} | model={model_id} | dataset={dataset_id} ({desc.origin}) | device={device}"]

        for epoch in range(1, epochs + 1):
            model.train()
            train_losses: List[float] = []
            for batch in train_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()
                out = model(x)
                if is_classification:
                    loss = criterion(out, y.long())
                else:
                    loss = criterion(out.squeeze(-1), y.float())
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.item()))

            model.eval()
            val_losses: List[float] = []
            with torch.no_grad():
                for batch in val_loader:
                    x, y = batch[0].to(device), batch[1].to(device)
                    out = model(x)
                    if is_classification:
                        vloss = criterion(out, y.long())
                    else:
                        vloss = criterion(out.squeeze(-1), y.float())
                    val_losses.append(float(vloss.item()))

            avg_train = float(np.mean(train_losses)) if train_losses else 0.0
            avg_val = float(np.mean(val_losses)) if val_losses else avg_train
            history["train_loss"].append(round(avg_train, 5))
            history["val_loss"].append(round(avg_val, 5))
            scheduler.step(avg_val)

            log_lines.append(f"Epoch {epoch}/{epochs} | train_loss={avg_train:.4f} | val_loss={avg_val:.4f}")

            if avg_val <= best_val_loss:
                best_val_loss = avg_val
                torch.save(
                    {
                        "epoch": epoch,
                        "model_version": "1.0.0",
                        "experiment_name": exp_id,
                        "model_state_dict": model.state_dict(),
                        "metrics": {"best_val_loss": best_val_loss},
                        "config": {"model_name": model_id, "dataset_id": dataset_id, **hp},
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                    best_ckpt_path,
                )

            if progress_callback:
                progress_callback(epoch, epochs, {"train_loss": avg_train, "val_loss": avg_val})

            if early_stopping(avg_val):
                log_lines.append(f"Early stopping triggered at epoch {epoch}.")
                break

        # Evaluate best checkpoint on test split
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        y_true_all, y_pred_all, y_prob_all = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                out = model(x)
                y_true_all.extend(y.cpu().numpy().tolist())
                if is_classification:
                    probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                    preds = (probs >= 0.50).astype(int)
                    y_prob_all.extend(probs.tolist())
                    y_pred_all.extend(preds.tolist())
                else:
                    y_pred_all.extend(out.squeeze(-1).cpu().numpy().tolist())

        if is_classification:
            eval_metrics = compute_classification_metrics(
                np.array(y_true_all), np.array(y_pred_all), np.array(y_prob_all)
            )
        else:
            eval_metrics = compute_regression_metrics(np.array(y_true_all), np.array(y_pred_all))

        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n")

        record = {
            "experiment_id": exp_id,
            "dataset_name": dataset_id,
            "dataset_version": "1.0",
            "dataset_origin": desc.origin,
            "provenance_label": desc.provenance_label,
            "model_name": model_id,
            "model_version": "1.0.0",
            "hyperparameters": hp,
            "split_methodology": split_desc,
            "training_configuration": {
                "device": device,
                "epochs_completed": len(history["train_loss"]),
                "seed": seed,
            },
            "evaluation_metrics": eval_metrics,
            "history": history,
            "checkpoint_location": str(best_ckpt_path),
            "log_location": str(log_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "is_legacy": False,
        }
        self.experiment_service.save_experiment(record)
        return record

    def _train_ridge(
        self,
        exp_id: str,
        ckpt_dir: Path,
        log_path: Path,
        hp: Dict[str, Any],
        desc: Any,
        spec: Any,
    ) -> Dict[str, Any]:
        alpha = float(hp.get("alpha", 1.0))
        train_ds = CADWalkTemporalDataset(split="train", filter_valid_speed=True)
        test_ds = CADWalkTemporalDataset(split="test", filter_valid_speed=True)

        def extract_xy(ds):
            X, y = [], []
            for i in range(len(ds)):
                _, target, meta = ds[i]
                X.append([meta["duration_ms"], meta["peak_force_n"]])
                y.append(float(target.item()))
            return np.array(X), np.array(y)

        X_tr, y_tr = extract_xy(train_ds)
        X_te, y_te = extract_xy(test_ds)

        ridge = Ridge(alpha=alpha)
        ridge.fit(X_tr, y_tr)
        preds = ridge.predict(X_te)
        eval_metrics = compute_regression_metrics(y_te, preds)

        ckpt_path = ckpt_dir / "ridge_baseline.joblib"
        joblib.dump(ridge, ckpt_path)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"Ridge fitted on {len(X_tr)} CAD_WALK train trials | alpha={alpha} | test_mae={eval_metrics['mae']:.4f}\n")

        record = {
            "experiment_id": exp_id,
            "dataset_name": "cad_walk",
            "dataset_version": "1.0",
            "dataset_origin": desc.origin,
            "provenance_label": desc.provenance_label,
            "model_name": "pressure_ridge_baseline",
            "model_version": "1.0.0",
            "hyperparameters": {"alpha": alpha, "features": ["duration_ms", "peak_force_n"]},
            "split_methodology": desc.split_methodology,
            "training_configuration": {"device": "cpu", "seed": int(hp.get("seed", 42))},
            "evaluation_metrics": eval_metrics,
            "history": {},
            "checkpoint_location": str(ckpt_path),
            "log_location": str(log_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "is_legacy": False,
        }
        self.experiment_service.save_experiment(record)
        return record
