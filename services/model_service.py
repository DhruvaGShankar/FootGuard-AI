"""
FootGuard AI - Model & Checkpoint Management Service
Manages model specifications, checkpoint discovery, metadata inspection,
and loading existing PyTorch / Ridge checkpoints without retraining.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import joblib
import torch

from services.config_service import ConfigService
from thermal.cnn import Thermal2DCNN
from thermal.tcn import ThermalTCN
from pressure.cnn import Pressure1DCNN
from pressure.tcn import PressureTCN


@dataclass
class ModelSpec:
    model_id: str
    name: str
    modality: str
    task_type: str  # "classification" or "regression"
    input_contract: str
    output_contract: str
    registry_status: str
    default_checkpoint: str
    limitations: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CheckpointInfo:
    checkpoint_id: str
    model_id: str
    experiment_dir: str
    file_path: str
    file_size_kb: float
    epoch: Optional[int]
    model_version: str
    timestamp: Optional[str]
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelService:
    """Provides model registry, checkpoint discovery, and safe model loading."""

    MODEL_SPECS: List[ModelSpec] = [
        ModelSpec(
            model_id="thermal_2d_cnn",
            name="Thermal 2D CNN",
            modality="thermal",
            task_type="classification",
            input_contract="(B, 3, 160, 80) [Left, Right_flipped, Delta_T]",
            output_contract="(B, 2) logits [Control, Diabetic Cohort]",
            registry_status="Research Candidate",
            default_checkpoint="experiment_a_thermal_2d_cnn/best_model.pt",
            limitations="Diabetes cohort stratification only (NOT ulcer detection). Uncalibrated softmax scores.",
        ),
        ModelSpec(
            model_id="thermal_tcn",
            name="Thermal TCN",
            modality="thermal",
            task_type="regression",
            input_contract="(B, 4, 1, H, W) cold-rewarming sequence",
            output_contract="(B, 1) predicted T15 mean temperature (°C)",
            registry_status="Exploratory Only",
            default_checkpoint="experiment_c_thermal_tcn/best_model.pt",
            limitations="Underfit on N=13 training subjects in IAC-TecMed; persistence baseline outperforms.",
        ),
        ModelSpec(
            model_id="pressure_1d_cnn",
            name="Pressure 1D CNN",
            modality="pressure",
            task_type="regression",
            input_contract="(B, C, L) temporal force curves (default C=1, L=100)",
            output_contract="(B, 1) walking speed (km/h) or synthetic load target",
            registry_status="Research Candidate",
            default_checkpoint="experiment_b_pressure_1d_cnn/best_model.pt",
            limitations="Trained on healthy controls (CAD_WALK). Outperforms Ridge at slow speeds (<4.5 km/h) only.",
        ),
        ModelSpec(
            model_id="pressure_tcn",
            name="Pressure TCN",
            modality="pressure",
            task_type="regression",
            input_contract="(B, C, L) causal dilated temporal force sequences",
            output_contract="(B, 1) continuous regression target",
            registry_status="Exploratory Only",
            default_checkpoint="experiment_b_pressure_tcn/best_model.pt",
            limitations="Underperformed on closed normalized 100-pt stance curves due to causal masking.",
        ),
        ModelSpec(
            model_id="pressure_ridge_baseline",
            name="Ridge Biomechanical Baseline",
            modality="pressure",
            task_type="regression",
            input_contract="[stance_duration_ms, peak_force_n]",
            output_contract="walking_speed (km/h)",
            registry_status="Primary Benchmark",
            default_checkpoint="experiment_b_pressure_1d_cnn/ridge_baseline.joblib",
            limitations="Trained on healthy controls (CAD_WALK); requires valid stance duration and peak force.",
        ),
        ModelSpec(
            model_id="multimodal_fusion_mlp",
            name="Multimodal Synthetic Fusion MLP",
            modality="multimodal",
            task_type="classification/regression",
            input_contract="(B, F) paired thermal + 6-site pressure + 6-axis IMU features",
            output_contract="(B, 2) regime logits or (B, 1) latent gait speed",
            registry_status="Experimental Synthetic Only",
            default_checkpoint="",
            limitations="Synthetic experimentation only; never trained or evaluated on real clinical patients.",
        ),
    ]

    def __init__(self, config_service: Optional[ConfigService] = None):
        self.cfg = config_service or ConfigService()
        self.checkpoints_dir = self.cfg.get_path("checkpoints_dir")

    def list_models(self) -> List[ModelSpec]:
        return list(self.MODEL_SPECS)

    def get_model_spec(self, model_id: str) -> ModelSpec:
        for spec in self.MODEL_SPECS:
            if spec.model_id == model_id:
                return spec
        raise ValueError(f"Unknown model_id: '{model_id}'")

    def _infer_model_id_from_dir(self, exp_dir_name: str, filename: str) -> str:
        name = exp_dir_name.lower()
        if filename.endswith(".joblib") or "ridge" in filename.lower():
            return "pressure_ridge_baseline"
        if "thermal_2d_cnn" in name or "thermal_cnn" in name:
            return "thermal_2d_cnn"
        if "thermal_tcn" in name:
            return "thermal_tcn"
        if "pressure_1d_cnn" in name or "pressure_cnn" in name:
            return "pressure_1d_cnn"
        if "pressure_tcn" in name:
            return "pressure_tcn"
        return "unknown"

    def list_checkpoints(self, model_id: Optional[str] = None) -> List[CheckpointInfo]:
        """Discovers all saved checkpoints and reads metadata without retraining."""
        if not self.checkpoints_dir.exists():
            return []

        results: List[CheckpointInfo] = []
        for path in sorted(self.checkpoints_dir.rglob("*")):
            if not path.is_file() or path.suffix not in [".pt", ".joblib"]:
                continue
            if path.name == "latest_model.pt":
                continue  # Prefer best_model.pt / named checkpoints in primary listing

            exp_dir = path.parent.name
            inferred_id = self._infer_model_id_from_dir(exp_dir, path.name)
            if model_id and inferred_id != model_id:
                continue

            epoch = None
            model_version = "1.0.0"
            timestamp = None
            metrics: Dict[str, Any] = {}

            if path.suffix == ".pt":
                try:
                    ckpt = torch.load(path, map_location="cpu", weights_only=False)
                    if isinstance(ckpt, dict):
                        epoch = ckpt.get("epoch")
                        model_version = str(ckpt.get("model_version", "1.0.0"))
                        timestamp = ckpt.get("timestamp")
                        raw_metrics = ckpt.get("metrics", {})
                        if isinstance(raw_metrics, dict):
                            metrics = raw_metrics
                        if "config" in ckpt and isinstance(ckpt["config"], dict):
                            cfg_model = ckpt["config"].get("model_name")
                            if cfg_model in [s.model_id for s in self.MODEL_SPECS]:
                                inferred_id = cfg_model
                except Exception:
                    metrics = {"load_warning": "Could not inspect checkpoint header"}

            results.append(
                CheckpointInfo(
                    checkpoint_id=f"{exp_dir}/{path.name}",
                    model_id=inferred_id,
                    experiment_dir=exp_dir,
                    file_path=str(path),
                    file_size_kb=round(path.stat().st_size / 1024.0, 2),
                    epoch=epoch,
                    model_version=model_version,
                    timestamp=timestamp,
                    metrics=metrics,
                )
            )

        return results

    def build_model(self, model_id: str, in_channels: Optional[int] = None) -> Any:
        """Instantiates a fresh model architecture."""
        if model_id == "thermal_2d_cnn":
            return Thermal2DCNN(in_channels=in_channels or 3, num_classes=2, dropout_rate=0.3)
        if model_id == "thermal_tcn":
            return ThermalTCN(in_channels=in_channels or 1, out_features=1)
        if model_id == "pressure_1d_cnn":
            return Pressure1DCNN(in_channels=in_channels or 1, out_features=1, expected_seq_len=100 if (in_channels or 1) == 1 else None)
        if model_id == "pressure_tcn":
            return PressureTCN(in_channels=in_channels or 1, out_features=1)
        raise ValueError(f"Cannot build PyTorch architecture for model_id '{model_id}'.")

    def load_checkpoint(
        self,
        model_id: str,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
    ) -> Tuple[Any, Dict[str, Any]]:
        """Loads an existing checkpoint (.pt or .joblib) without retraining."""
        spec = self.get_model_spec(model_id)
        ckpt_path = Path(checkpoint_path) if checkpoint_path else (self.checkpoints_dir / spec.default_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found for '{model_id}': {ckpt_path}")

        if model_id == "pressure_ridge_baseline" or ckpt_path.suffix == ".joblib":
            try:
                model = joblib.load(ckpt_path)
                return model, {"model_id": model_id, "checkpoint_path": str(ckpt_path), "model_version": "1.0.0"}
            except Exception as e:
                raise RuntimeError(f"Failed to load joblib checkpoint '{ckpt_path}': {e}")

        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            # Infer in_channels from checkpoint weights if custom synthetic channels were trained
            in_channels = None
            if model_id == "pressure_1d_cnn" and "encoder.block1.0.weight" in state_dict:
                in_channels = int(state_dict["encoder.block1.0.weight"].shape[1])
            elif model_id == "pressure_tcn" and "tcn.network.0.conv1.weight" in state_dict:
                in_channels = int(state_dict["tcn.network.0.conv1.weight"].shape[1])

            model = self.build_model(model_id, in_channels=in_channels)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            meta = {
                "model_id": model_id,
                "checkpoint_path": str(ckpt_path),
                "epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
                "model_version": ckpt.get("model_version", "1.0.0") if isinstance(ckpt, dict) else "1.0.0",
                "metrics": ckpt.get("metrics", {}) if isinstance(ckpt, dict) else {},
            }
            return model, meta
        except Exception as e:
            raise RuntimeError(f"Failed to load PyTorch checkpoint '{ckpt_path}': {e}")

