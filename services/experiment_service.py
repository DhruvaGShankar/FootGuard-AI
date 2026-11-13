"""
FootGuard AI - Experiment Persistence & Reporting Service
Stores and retrieves experiment metadata in JSON, configs in YAML, and logs locally.
Also surfaces legacy Phase 4 / Phase 5 benchmark results without altering them.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import yaml

from services.config_service import ConfigService


class ExperimentService:
    """Manages local JSON experiment persistence, history listing, and report export."""

    REQUIRED_FIELDS = [
        "experiment_id",
        "dataset_name",
        "dataset_version",
        "dataset_origin",
        "model_name",
        "model_version",
        "hyperparameters",
        "split_methodology",
        "training_configuration",
        "evaluation_metrics",
        "checkpoint_location",
        "timestamp",
    ]

    def __init__(self, config_service: Optional[ConfigService] = None):
        self.cfg = config_service or ConfigService()
        self.results_dir = self.cfg.get_path("experiments_results_dir")
        self.configs_dir = self.cfg.get_path("experiments_configs_dir")
        self.logs_dir = self.cfg.get_path("experiments_logs_dir")
        self.legacy_dir = self.cfg.get_path("legacy_experiments_dir")
        self.evaluation_dir = self.cfg.get_path("evaluation_dir")

    def save_experiment(self, record: Dict[str, Any]) -> Path:
        """Validates required experiment metadata fields and persists JSON + YAML config."""
        if "timestamp" not in record or not record["timestamp"]:
            record["timestamp"] = datetime.now(timezone.utc).isoformat()

        missing = [k for k in self.REQUIRED_FIELDS if k not in record]
        if missing:
            raise ValueError(f"Experiment record missing required fields: {missing}")

        if record["dataset_origin"] not in ["real", "synthetic"]:
            raise ValueError(f"dataset_origin must be 'real' or 'synthetic', got '{record['dataset_origin']}'")

        exp_id = str(record["experiment_id"])
        out_json = self.results_dir / f"{exp_id}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        cfg_yaml = self.configs_dir / f"{exp_id}.yaml"
        with open(cfg_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                {
                    "experiment_id": exp_id,
                    "dataset_name": record["dataset_name"],
                    "dataset_origin": record["dataset_origin"],
                    "model_name": record["model_name"],
                    "hyperparameters": record["hyperparameters"],
                    "training_configuration": record["training_configuration"],
                },
                f,
                sort_keys=False,
            )

        return out_json

    def load_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Loads an experiment record by ID from results/ or legacy Phase 4 artifacts."""
        candidate = self.results_dir / f"{experiment_id}.json"
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as f:
                return json.load(f)

        for exp in self.list_experiments():
            if exp.get("experiment_id") == experiment_id:
                return exp

        raise FileNotFoundError(f"Experiment '{experiment_id}' not found.")

    def _load_legacy_experiments(self) -> List[Dict[str, Any]]:
        """Reads Phase 4 benchmark JSONs into the standardized schema without altering files."""
        records: List[Dict[str, Any]] = []

        # Legacy Experiment A: Thermal 2D CNN
        exp_a_path = self.legacy_dir / "experiment_a_thermal_2d_cnn" / "experiment_a_results.json"
        hist_a_path = self.legacy_dir / "experiment_a_thermal_2d_cnn" / "training_history.json"
        if exp_a_path.exists():
            with open(exp_a_path, "r", encoding="utf-8") as f:
                data_a = json.load(f)
            history_a = {}
            if hist_a_path.exists():
                with open(hist_a_path, "r", encoding="utf-8") as f:
                    history_a = json.load(f)
            records.append(
                {
                    "experiment_id": "legacy_experiment_a_thermal_2d_cnn",
                    "dataset_name": "thermodatabase",
                    "dataset_version": "1.0",
                    "dataset_origin": "real",
                    "model_name": "thermal_2d_cnn",
                    "model_version": "1.0.0",
                    "hyperparameters": {"threshold": data_a.get("optimal_threshold", 0.5)},
                    "split_methodology": "Participant-level 116 train / 25 val / 26 test",
                    "training_configuration": {"source": "Phase 4 Controlled Baseline"},
                    "evaluation_metrics": data_a.get("test_metrics_default_0.5", {}),
                    "history": history_a,
                    "checkpoint_location": "checkpoints/experiment_a_thermal_2d_cnn/best_model.pt",
                    "timestamp": "Phase 4 Baseline",
                    "is_legacy": True,
                }
            )

        # Legacy Experiment B: CAD_WALK (Pressure 1D CNN, Ridge, Pressure TCN)
        exp_b_path = self.legacy_dir / "experiment_b_cad_walk" / "experiment_b_results.json"
        if exp_b_path.exists():
            with open(exp_b_path, "r", encoding="utf-8") as f:
                data_b = json.load(f)

            ridge_test = data_b.get("baselines", {}).get("ridge_regression_baseline", {}).get("test", {})
            records.append(
                {
                    "experiment_id": "legacy_experiment_b_ridge_baseline",
                    "dataset_name": "cad_walk",
                    "dataset_version": "1.0",
                    "dataset_origin": "real",
                    "model_name": "pressure_ridge_baseline",
                    "model_version": "1.0.0",
                    "hyperparameters": {"alpha": 1.0, "features": ["duration_ms", "peak_force_n"]},
                    "split_methodology": "Participant-level 36 train / 8 val / 7 test (288 valid trials)",
                    "training_configuration": {"source": "Phase 4 Controlled Baseline"},
                    "evaluation_metrics": ridge_test,
                    "checkpoint_location": "checkpoints/experiment_b_pressure_1d_cnn/ridge_baseline.joblib",
                    "timestamp": "Phase 4 Baseline",
                    "is_legacy": True,
                }
            )

            for key, model_id in [("pressure_1d_cnn", "pressure_1d_cnn"), ("pressure_tcn", "pressure_tcn")]:
                if key in data_b:
                    records.append(
                        {
                            "experiment_id": f"legacy_experiment_b_{model_id}",
                            "dataset_name": "cad_walk",
                            "dataset_version": "1.0",
                            "dataset_origin": "real",
                            "model_name": model_id,
                            "model_version": "1.0.0",
                            "hyperparameters": {},
                            "split_methodology": "Participant-level 36 train / 8 val / 7 test (288 valid trials)",
                            "training_configuration": {"source": "Phase 4 Controlled Baseline"},
                            "evaluation_metrics": data_b[key].get("test_metrics", {}),
                            "history": data_b[key].get("history", {}),
                            "checkpoint_location": f"checkpoints/experiment_b_{model_id}/best_model.pt",
                            "timestamp": "Phase 4 Baseline",
                            "is_legacy": True,
                        }
                    )

        # Legacy Experiment C: IAC-TecMed Thermal TCN
        exp_c_path = self.legacy_dir / "experiment_c_thermal_tcn" / "experiment_c_results.json"
        if exp_c_path.exists():
            with open(exp_c_path, "r", encoding="utf-8") as f:
                data_c = json.load(f)
            records.append(
                {
                    "experiment_id": "legacy_experiment_c_thermal_tcn",
                    "dataset_name": "iac_tecmed",
                    "dataset_version": "1.0",
                    "dataset_origin": "real",
                    "model_name": "thermal_tcn",
                    "model_version": "1.0.0",
                    "hyperparameters": {},
                    "split_methodology": "Participant-level 13 train / 3 val / 3 test",
                    "training_configuration": {"source": "Phase 4 Controlled Baseline"},
                    "evaluation_metrics": data_c.get("thermal_tcn", {}).get("test_metrics", {}),
                    "checkpoint_location": "checkpoints/experiment_c_thermal_tcn/best_model.pt",
                    "timestamp": "Phase 4 Baseline",
                    "is_legacy": True,
                }
            )

        return records

    def list_experiments(self, include_legacy: bool = True) -> List[Dict[str, Any]]:
        """Lists all saved experiments (newest local runs first, followed by legacy baselines)."""
        records: List[Dict[str, Any]] = []
        if self.results_dir.exists():
            for path in sorted(self.results_dir.glob("*.json"), reverse=True):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if not isinstance(data, dict):
                        continue
                    data.setdefault("experiment_id", path.stem)
                    data.setdefault("dataset_origin", "real")
                    records.append(data)
                except Exception:
                    continue
        if include_legacy:
            records.extend(self._load_legacy_experiments())
        return records

    def load_phase5_analysis(self) -> Optional[Dict[str, Any]]:
        p5 = self.evaluation_dir / "phase5_analysis_results.json"
        if not p5.exists():
            return None
        with open(p5, "r", encoding="utf-8") as f:
            return json.load(f)

    def export_report(self, experiment_id: str, fmt: str = "json") -> str:
        """Exports a single experiment record as formatted JSON or Markdown."""
        exp = self.load_experiment(experiment_id)
        if fmt == "json":
            return json.dumps(exp, indent=2)
        if fmt == "md":
            lines = [
                f"# Experiment Report: `{exp['experiment_id']}`",
                "",
                f"- **Model:** `{exp.get('model_name')}` (v{exp.get('model_version')})",
                f"- **Dataset:** `{exp.get('dataset_name')}` (Origin: **{exp.get('dataset_origin')}**)",
                f"- **Split Methodology:** {exp.get('split_methodology')}",
                f"- **Checkpoint:** `{exp.get('checkpoint_location')}`",
                f"- **Timestamp:** {exp.get('timestamp')}",
                "",
                "## Evaluation Metrics",
                "```json",
                json.dumps(exp.get("evaluation_metrics", {}), indent=2),
                "```",
                "",
                "## Hyperparameters",
                "```json",
                json.dumps(exp.get("hyperparameters", {}), indent=2),
                "```",
                "",
                "> **Research Disclaimer:** Experimental ML research output only. Not for clinical diagnosis or ulcer detection.",
            ]
            return "\n".join(lines)
        raise ValueError(f"Unsupported export format '{fmt}'. Use 'json' or 'md'.")

