"""
FootGuard AI - Controlled CSV Dataset Ingestion Service
Validates uploaded CSV datasets (features, target, session/participant ID, units, provenance),
enforces leakage and derived-target safeguards, and saves an isolated local copy without
modifying the original file.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import numpy as np
import pandas as pd

from services.config_service import ConfigService
from services.synthetic_generator_service import SYNTHETIC_PROVENANCE_TAG
from services.synthetic_training_service import BANNED_DERIVED_TARGETS


@dataclass
class IngestionSpec:
    dataset_name: str
    feature_columns: List[str]
    target_column: str
    session_id_column: str
    units: Dict[str, str]
    provenance: str  # "synthetic" or "real"
    is_calibrated_celsius: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetIngestionService:
    """Validates and persists user-uploaded CSV datasets into isolated local directories."""

    def __init__(self, config_service: Optional[ConfigService] = None):
        self.cfg = config_service or ConfigService()
        self.synthetic_dir = self.cfg.get_path("data_synthetic_dir")
        self.real_dir = self.cfg.get_path("data_real_dir")
        self.synthetic_dir.mkdir(parents=True, exist_ok=True)
        self.real_dir.mkdir(parents=True, exist_ok=True)

    def validate_ingestion(self, df: pd.DataFrame, spec: IngestionSpec) -> List[str]:
        """Validates CSV schema, target/feature separation, session count, and numeric types."""
        warnings: List[str] = []
        if df is None or df.empty:
            raise ValueError("Uploaded CSV is empty.")
        if spec.provenance not in ("synthetic", "real"):
            raise ValueError(f"Provenance must be 'synthetic' or 'real', got '{spec.provenance}'.")
        if not spec.feature_columns:
            raise ValueError("At least one input feature column must be selected.")
        if spec.session_id_column not in df.columns:
            raise ValueError(f"Session/participant ID column '{spec.session_id_column}' not found in CSV.")
        if spec.target_column not in df.columns:
            raise ValueError(f"Target column '{spec.target_column}' not found in CSV.")
        if spec.target_column in spec.feature_columns:
            raise ValueError(f"Target leakage blocked: '{spec.target_column}' cannot also be an input feature.")
        if spec.provenance == "synthetic" and spec.target_column in BANNED_DERIVED_TARGETS:
            raise ValueError(
                f"Target '{spec.target_column}' is blocked: derived pressure totals/maxes are not valid ground-truth labels."
            )

        missing_feats = [c for c in spec.feature_columns if c not in df.columns]
        if missing_feats:
            raise ValueError(f"Feature columns missing from CSV: {missing_feats}")

        for col in spec.feature_columns + [spec.target_column]:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column '{col}' must be numeric for ML ingestion (found {df[col].dtype}).")

        n_sessions = int(df[spec.session_id_column].dropna().nunique())
        if n_sessions < 3:
            raise ValueError(
                f"At least 3 unique sessions/participants are required for leakage-free splits (found {n_sessions})."
            )

        na_count = int(df[spec.feature_columns + [spec.target_column, spec.session_id_column]].isna().sum().sum())
        if na_count > 0:
            warnings.append(f"Dataset contains {na_count} missing values across selected columns; rows with NaNs will be dropped on save.")

        if spec.provenance == "real" and not spec.is_calibrated_celsius:
            warnings.append("Marked as uncalibrated real data: thermal Celsius calculations will remain blocked.")

        return warnings

    def ingest_dataframe(self, df: pd.DataFrame, spec: IngestionSpec) -> Dict[str, Any]:
        """Validates and saves a local copy under data/synthetic/ or data/real/."""
        warnings = self.validate_ingestion(df, spec)
        slug = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in spec.dataset_name).strip("_")
        if not slug:
            slug = f"csv_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        dir_name = f"ingested_{slug}"

        base_dir = self.synthetic_dir if spec.provenance == "synthetic" else self.real_dir
        out_dir = base_dir / dir_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Clean copy: drop rows with NaN in required columns, preserve original untouched
        req_cols = list(dict.fromkeys(spec.feature_columns + [spec.target_column, spec.session_id_column]))
        clean_df = df.dropna(subset=req_cols).copy()
        if spec.session_id_column != "session_id":
            clean_df["session_id"] = clean_df[spec.session_id_column].astype(str)
        else:
            clean_df["session_id"] = clean_df["session_id"].astype(str)

        if spec.provenance == "synthetic":
            clean_df["data_status"] = SYNTHETIC_PROVENANCE_TAG
            provenance_tag = SYNTHETIC_PROVENANCE_TAG
        else:
            clean_df["data_status"] = "INGESTED_REAL_RESEARCH_CSV"
            provenance_tag = "INGESTED_REAL_RESEARCH_CSV"

        csv_path = out_dir / "readings.csv"
        clean_df.to_csv(csv_path, index=False)

        manifest = {
            "dataset_id": dir_name,
            "data_status": provenance_tag,
            "origin": spec.provenance,
            "ingested": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "num_rows": int(len(clean_df)),
            "num_sessions": int(clean_df["session_id"].nunique()),
            "csv_path": str(csv_path),
            "feature_columns": spec.feature_columns,
            "target_column": spec.target_column,
            "session_id_column": "session_id",
            "units": spec.units,
            "is_calibrated_celsius": spec.is_calibrated_celsius,
            "warnings": warnings,
        }
        with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest

