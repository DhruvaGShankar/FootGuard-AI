"""
FootGuard AI - Synthetic Experimental Training Service
Provides a dedicated, session-isolated training pathway for generated or ingested
synthetic datasets. Never mixes real and synthetic data, blocks derived pressure totals
as targets, and saves checkpoints exclusively under checkpoints/synthetic_experiments/.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Tuple
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.linear_model import Ridge

from common.reproducibility import set_seed
from common.metrics import compute_classification_metrics, compute_regression_metrics
from services.config_service import ConfigService
from services.synthetic_generator_service import SyntheticGeneratorService, SYNTHETIC_PROVENANCE_TAG
from services.experiment_service import ExperimentService
from thermal.cnn import Thermal2DCNN
from pressure.cnn import Pressure1DCNN
from pressure.tcn import PressureTCN


BANNED_DERIVED_TARGETS = {"plantar_total_load_n", "peak_site_pressure_n"}


class MultimodalSyntheticMLP(nn.Module):
    """Lightweight multimodal MLP for paired thermal + 6-site pressure + IMU features."""

    def __init__(self, in_features: int, num_outputs: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SyntheticTrainingService:
    """Executes controlled synthetic-only training runs with strict session isolation."""

    SUPPORTED_MODELS = {
        "thermal_2d_cnn": {"modality": "thermal", "task": "classification"},
        "pressure_1d_cnn": {"modality": "pressure", "task": "regression"},
        "pressure_tcn": {"modality": "pressure", "task": "regression"},
        "pressure_ridge_baseline": {"modality": "pressure", "task": "regression"},
        "multimodal_fusion_mlp": {"modality": "multimodal", "task": "auto"},
    }

    def __init__(
        self,
        config_service: Optional[ConfigService] = None,
        generator_service: Optional[SyntheticGeneratorService] = None,
        experiment_service: Optional[ExperimentService] = None,
    ):
        self.cfg = config_service or ConfigService()
        self.generator_service = generator_service or SyntheticGeneratorService(self.cfg)
        self.experiment_service = experiment_service or ExperimentService(self.cfg)
        self.checkpoints_dir = self.cfg.get_path("checkpoints_dir")
        self.synthetic_ckpt_root = self.checkpoints_dir / "synthetic_experiments"
        self.synthetic_ckpt_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def split_sessions_disjoint(
        session_ids: List[str],
        seed: int = 42,
    ) -> Tuple[List[str], List[str], List[str]]:
        """Splits unique session IDs into disjoint train / val / test lists."""
        unique = sorted(set(session_ids))
        if len(unique) < 3:
            raise ValueError(f"Need at least 3 unique sessions for train/val/test split, got {len(unique)}.")
        rng = np.random.default_rng(seed)
        perm = [str(x) for x in rng.permutation(unique)]
        n = len(perm)
        n_test = max(1, int(round(0.2 * n)))
        n_val = max(1, int(round(0.2 * n)))
        if n_train := n - n_test - n_val < 1:
            n_train = max(1, n - 2)
            n_val = 1
            n_test = n - n_train - n_val
        train_s = perm[:n - n_val - n_test]
        val_s = perm[n - n_val - n_test: n - n_test]
        test_s = perm[n - n_test:]

        assert not (set(train_s) & set(val_s)), "Leakage detected between train and val sessions!"
        assert not (set(train_s) & set(test_s)), "Leakage detected between train and test sessions!"
        assert not (set(val_s) & set(test_s)), "Leakage detected between val and test sessions!"
        return train_s, val_s, test_s

    def validate_training_request(
        self,
        df: pd.DataFrame,
        manifest: Dict[str, Any],
        model_id: str,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        session_col: str = "session_id",
    ) -> None:
        if model_id not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model '{model_id}' is not supported for synthetic training. Choose from {list(self.SUPPORTED_MODELS)}.")
        if manifest.get("data_status") != SYNTHETIC_PROVENANCE_TAG and manifest.get("origin") != "synthetic":
            raise ValueError("SyntheticTrainingService only accepts synthetic datasets tagged SYNTHETIC_NOT_CLINICAL_GROUND_TRUTH.")
        if df is None or df.empty:
            raise ValueError("Synthetic dataset is empty.")
        if session_col not in df.columns:
            raise ValueError(f"Session identifier column '{session_col}' not found in dataset.")
        if target_col in BANNED_DERIVED_TARGETS:
            raise ValueError(
                f"Target '{target_col}' is blocked: derived pressure totals/maxes cannot be used as independent ground-truth labels."
            )
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")
        if feature_cols:
            missing_f = [c for c in feature_cols if c not in df.columns]
            if missing_f:
                raise ValueError(f"Selected feature columns missing from dataset: {missing_f}")
            if target_col in feature_cols:
                raise ValueError(f"Target leakage blocked: '{target_col}' cannot be included in input features.")

    def _build_pressure_windows(
        self,
        df: pd.DataFrame,
        sessions: List[str],
        feature_cols: List[str],
        target_col: str,
        session_col: str,
        seq_len: int = 16,
    ) -> Tuple[np.ndarray, np.ndarray]:
        sub = df[df[session_col].isin(sessions)]
        X_list, y_list = [], []
        for _, grp in sub.groupby(session_col, sort=False):
            arr = grp[feature_cols].to_numpy(dtype=np.float32)  # (T, C)
            if len(arr) < seq_len:
                continue
            target_val = float(grp[target_col].iloc[0])
            for start in range(0, len(arr) - seq_len + 1, max(1, seq_len // 2)):
                X_list.append(arr[start:start + seq_len].T)  # (C, seq_len)
                y_list.append(target_val)
        if not X_list:
            raise ValueError(f"No pressure windows of length {seq_len} could be built for split.")
        return np.stack(X_list, axis=0), np.array(y_list, dtype=np.float32)

    def start_synthetic_training(
        self,
        dataset_id: str,
        model_id: str,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        session_col: str = "session_id",
        hyperparameters: Optional[Dict[str, Any]] = None,
        experiment_id: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
    ) -> Dict[str, Any]:
        hp = hyperparameters or {}
        seed = int(hp.get("seed", 42))
        epochs = int(hp.get("epochs", 5))
        batch_size = int(hp.get("batch_size", 8))
        lr = float(hp.get("learning_rate", 1e-3))
        alpha = float(hp.get("alpha", 1.0))
        set_seed(seed)

        df, manifest, tensors = self.generator_service.load_generated_dataset(dataset_id)
        self.validate_training_request(df, manifest, model_id, target_col, feature_cols, session_col)

        exp_id = experiment_id or f"syn_{model_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        ckpt_dir = self.synthetic_ckpt_root / exp_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        sessions = sorted(df[session_col].astype(str).unique().tolist())
        train_s, val_s, test_s = self.split_sessions_disjoint(sessions, seed=seed)

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        eval_metrics: Dict[str, Any] = {}
        used_features = list(feature_cols or [])

        if model_id == "thermal_2d_cnn":
            if "thermal_tensors" not in tensors:
                raise ValueError("Dataset lacks thermal_tensors (N, 3, 160, 80) required for Thermal2DCNN.")
            t_arr = tensors["thermal_tensors"]  # (N_sess, 3, 160, 80)
            s_ids = [str(x) for x in tensors["session_ids"].tolist()]
            y_map = df.groupby(session_col)[target_col].first().to_dict()
            y_all = np.array([int(y_map[s]) for s in s_ids], dtype=np.int64)

            idx_tr = [i for i, s in enumerate(s_ids) if s in train_s]
            idx_va = [i for i, s in enumerate(s_ids) if s in val_s]
            idx_te = [i for i, s in enumerate(s_ids) if s in test_s]

            model = Thermal2DCNN(in_channels=3, num_classes=2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            tr_loader = DataLoader(TensorDataset(torch.from_numpy(t_arr[idx_tr]), torch.from_numpy(y_all[idx_tr])), batch_size=batch_size, shuffle=True)
            va_x, va_y = torch.from_numpy(t_arr[idx_va]), torch.from_numpy(y_all[idx_va])
            te_x, te_y = torch.from_numpy(t_arr[idx_te]), y_all[idx_te]

            best_val = float("inf")
            ckpt_path = ckpt_dir / "best_model.pt"
            for ep in range(1, epochs + 1):
                model.train()
                tr_losses = []
                for bx, by in tr_loader:
                    optimizer.zero_grad()
                    loss = criterion(model(bx), by)
                    loss.backward()
                    optimizer.step()
                    tr_losses.append(float(loss.item()))
                model.eval()
                with torch.no_grad():
                    val_loss = float(criterion(model(va_x), va_y).item())
                tr_loss = float(np.mean(tr_losses))
                history["train_loss"].append(tr_loss)
                history["val_loss"].append(val_loss)
                if val_loss <= best_val:
                    best_val = val_loss
                    torch.save({"epoch": ep, "model_state_dict": model.state_dict(), "data_status": SYNTHETIC_PROVENANCE_TAG}, ckpt_path)
                if progress_callback:
                    progress_callback(ep, epochs, {"train_loss": tr_loss, "val_loss": val_loss})

            model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True)["model_state_dict"])
            model.eval()
            with torch.no_grad():
                probs = torch.softmax(model(te_x), dim=1)[:, 1].numpy()
            eval_metrics = compute_classification_metrics(te_y, (probs >= 0.5).astype(int), probs)
            used_features = ["thermal_tensors(3,160,80)"]

        elif model_id in ("pressure_1d_cnn", "pressure_tcn"):
            p_cols = used_features or SyntheticGeneratorService.PRESSURE_FEATURE_COLS
            X_tr, y_tr = self._build_pressure_windows(df, train_s, p_cols, target_col, session_col)
            X_va, y_va = self._build_pressure_windows(df, val_s, p_cols, target_col, session_col)
            X_te, y_te = self._build_pressure_windows(df, test_s, p_cols, target_col, session_col)

            in_ch = X_tr.shape[1]
            seq_len_actual = X_tr.shape[2]
            model = (
                Pressure1DCNN(in_channels=in_ch, expected_seq_len=seq_len_actual)
                if model_id == "pressure_1d_cnn"
                else PressureTCN(in_channels=in_ch)
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            criterion = nn.MSELoss()
            tr_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)), batch_size=batch_size, shuffle=True)
            va_x, va_y = torch.from_numpy(X_va), torch.from_numpy(y_va)
            te_x = torch.from_numpy(X_te)

            best_val = float("inf")
            ckpt_path = ckpt_dir / "best_model.pt"
            for ep in range(1, epochs + 1):
                model.train()
                tr_losses = []
                for bx, by in tr_loader:
                    optimizer.zero_grad()
                    loss = criterion(model(bx).squeeze(-1), by)
                    loss.backward()
                    optimizer.step()
                    tr_losses.append(float(loss.item()))
                model.eval()
                with torch.no_grad():
                    val_loss = float(criterion(model(va_x).squeeze(-1), va_y).item())
                tr_loss = float(np.mean(tr_losses))
                history["train_loss"].append(tr_loss)
                history["val_loss"].append(val_loss)
                if val_loss <= best_val:
                    best_val = val_loss
                    torch.save({"epoch": ep, "model_state_dict": model.state_dict(), "in_channels": in_ch, "data_status": SYNTHETIC_PROVENANCE_TAG}, ckpt_path)
                if progress_callback:
                    progress_callback(ep, epochs, {"train_loss": tr_loss, "val_loss": val_loss})

            model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True)["model_state_dict"])
            model.eval()
            with torch.no_grad():
                preds = model(te_x).squeeze(-1).numpy()
            eval_metrics = compute_regression_metrics(y_te, preds)
            used_features = p_cols

        elif model_id == "pressure_ridge_baseline":
            r_cols = used_features or SyntheticGeneratorService.PRESSURE_FEATURE_COLS
            tr_df = df[df[session_col].isin(train_s)]
            te_df = df[df[session_col].isin(test_s)]
            ridge = Ridge(alpha=alpha)
            ridge.fit(tr_df[r_cols].to_numpy(dtype=np.float64), tr_df[target_col].to_numpy(dtype=np.float64))
            preds = ridge.predict(te_df[r_cols].to_numpy(dtype=np.float64))
            eval_metrics = compute_regression_metrics(te_df[target_col].to_numpy(dtype=np.float64), preds)
            ckpt_path = ckpt_dir / "ridge_synthetic.joblib"
            joblib.dump({"model": ridge, "features": r_cols, "data_status": SYNTHETIC_PROVENANCE_TAG}, ckpt_path)
            used_features = r_cols
            if progress_callback:
                progress_callback(1, 1, {"train_loss": float(eval_metrics["rmse"]), "val_loss": float(eval_metrics["rmse"])})

        elif model_id == "multimodal_fusion_mlp":
            m_cols = used_features or (
                SyntheticGeneratorService.THERMAL_FEATURE_COLS
                + SyntheticGeneratorService.PRESSURE_FEATURE_COLS
                + SyntheticGeneratorService.IMU_FEATURE_COLS
            )
            tr_df = df[df[session_col].isin(train_s)]
            va_df = df[df[session_col].isin(val_s)]
            te_df = df[df[session_col].isin(test_s)]

            X_tr_raw = tr_df[m_cols].to_numpy(dtype=np.float32)
            mean_v = X_tr_raw.mean(axis=0, keepdims=True)
            std_v = np.where(X_tr_raw.std(axis=0, keepdims=True) < 1e-6, 1.0, X_tr_raw.std(axis=0, keepdims=True))
            X_tr = (X_tr_raw - mean_v) / std_v
            X_va = (va_df[m_cols].to_numpy(dtype=np.float32) - mean_v) / std_v
            X_te = (te_df[m_cols].to_numpy(dtype=np.float32) - mean_v) / std_v

            is_cls = target_col == "latent_thermal_regime" or df[target_col].nunique() <= 2
            num_out = 2 if is_cls else 1
            model = MultimodalSyntheticMLP(in_features=len(m_cols), num_outputs=num_out)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss() if is_cls else nn.MSELoss()

            y_tr_np = tr_df[target_col].to_numpy(dtype=np.int64 if is_cls else np.float32)
            y_va_np = va_df[target_col].to_numpy(dtype=np.int64 if is_cls else np.float32)
            y_te_np = te_df[target_col].to_numpy(dtype=np.int64 if is_cls else np.float32)

            tr_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr_np)), batch_size=batch_size, shuffle=True)
            va_x, va_y = torch.from_numpy(X_va), torch.from_numpy(y_va_np)
            te_x = torch.from_numpy(X_te)

            best_val = float("inf")
            ckpt_path = ckpt_dir / "best_model.pt"
            for ep in range(1, epochs + 1):
                model.train()
                tr_losses = []
                for bx, by in tr_loader:
                    optimizer.zero_grad()
                    out = model(bx)
                    loss = criterion(out, by) if is_cls else criterion(out.squeeze(-1), by)
                    loss.backward()
                    optimizer.step()
                    tr_losses.append(float(loss.item()))
                model.eval()
                with torch.no_grad():
                    v_out = model(va_x)
                    val_loss = float((criterion(v_out, va_y) if is_cls else criterion(v_out.squeeze(-1), va_y)).item())
                tr_loss = float(np.mean(tr_losses))
                history["train_loss"].append(tr_loss)
                history["val_loss"].append(val_loss)
                if val_loss <= best_val:
                    best_val = val_loss
                    torch.save(
                        {
                            "epoch": ep,
                            "model_state_dict": model.state_dict(),
                            "features": m_cols,
                            "mean": mean_v.tolist(),
                            "std": std_v.tolist(),
                            "data_status": SYNTHETIC_PROVENANCE_TAG,
                        },
                        ckpt_path,
                    )
                if progress_callback:
                    progress_callback(ep, epochs, {"train_loss": tr_loss, "val_loss": val_loss})

            model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True)["model_state_dict"])
            model.eval()
            with torch.no_grad():
                t_out = model(te_x)
                if is_cls:
                    probs = torch.softmax(t_out, dim=1)[:, 1].numpy()
                    eval_metrics = compute_classification_metrics(y_te_np, (probs >= 0.5).astype(int), probs)
                else:
                    preds = t_out.squeeze(-1).numpy()
                    eval_metrics = compute_regression_metrics(y_te_np, preds)
            used_features = m_cols

        record = {
            "experiment_id": exp_id,
            "dataset_name": dataset_id,
            "dataset_version": "synthetic-1.0",
            "dataset_origin": "synthetic",
            "data_status": SYNTHETIC_PROVENANCE_TAG,
            "model_name": model_id,
            "model_version": "1.0.0-synthetic",
            "hyperparameters": {"epochs": epochs, "batch_size": batch_size, "learning_rate": lr, "seed": seed, "alpha": alpha},
            "split_methodology": f"Session-isolated disjoint split ({len(train_s)} train / {len(val_s)} val / {len(test_s)} test sessions)",
            "training_configuration": {
                "seed": seed,
                "target_column": target_col,
                "feature_columns": used_features,
                "generator_config": manifest.get("generator_config", {}),
                "train_sessions": train_s,
                "val_sessions": val_s,
                "test_sessions": test_s,
            },
            "evaluation_metrics": eval_metrics,
            "history": history,
            "checkpoint_location": str(ckpt_path.relative_to(self.cfg.ml_root)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.experiment_service.save_experiment(record)
        return record
