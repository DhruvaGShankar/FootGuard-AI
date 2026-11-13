"""
FootGuard AI - YAML Configuration Service
Loads local configuration and initializes required filesystem directories.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml

ML_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ML_ROOT / "configs" / "default.yaml"


class ConfigService:
    """Loads YAML application configuration and manages local storage paths."""

    def __init__(self, config_path: Optional[Path] = None):
        self.ml_root = ML_ROOT
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration YAML not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f) or {}

        self._ensure_directories()

    def _resolve(self, rel_or_abs: str) -> Path:
        p = Path(rel_or_abs)
        return p if p.is_absolute() else (self.ml_root / p)

    def _ensure_directories(self) -> None:
        paths_cfg = self.config.get("paths", {})
        dir_keys = [
            "data_synthetic_dir",
            "data_real_dir",
            "data_processed_dir",
            "experiments_configs_dir",
            "experiments_results_dir",
            "experiments_logs_dir",
        ]
        for key in dir_keys:
            if key in paths_cfg:
                self._resolve(paths_cfg[key]).mkdir(parents=True, exist_ok=True)

    def get_path(self, key: str) -> Path:
        paths_cfg = self.config.get("paths", {})
        if key not in paths_cfg:
            raise KeyError(f"Unknown path key in configuration: {key}")
        return self._resolve(paths_cfg[key])

    def get_training_defaults(self) -> Dict[str, Any]:
        return dict(self.config.get("training_defaults", {}))

    def get_synthetic_loader_config(self) -> Dict[str, Any]:
        return dict(self.config.get("synthetic_loader", {}))

    def get_disclaimer(self) -> str:
        return str(
            self.config.get("app", {}).get(
                "research_disclaimer",
                "RESEARCH USE ONLY: Experimental ML prototype. Not a clinical diagnostic device.",
            )
        ).strip()

