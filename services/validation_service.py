"""
FootGuard AI - Dataset & Model Compatibility Validation Service
Validates dataset shapes, missing values, labels, provenance, calibration status,
and enforces model-dataset compatibility contracts.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from services.dataset_service import DatasetService


@dataclass
class ValidationReport:
    is_valid: bool
    dataset_id: str
    origin: str
    provenance_label: str
    shape: List[int]
    missing_values_count: int
    missing_by_column: Dict[str, int]
    label_summary: Dict[str, Any]
    warnings: List[str]
    errors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CompatibilityResult:
    is_compatible: bool
    dataset_id: str
    model_id: str
    reason: str
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ValidationService:
    """Validates data integrity and model-dataset compatibility."""

    SYNTHETIC_REQUIRED_COLS = [
        "reading_id",
        "session_id",
        "data_status",
        "thermal_mean_c",
        "heel_pressure_n",
        "mid_pressure_n",
        "mth1_pressure_n",
        "mth3_pressure_n",
        "mth5_pressure_n",
        "toe_hallux_pressure_n",
        "plantar_total_load_n",
    ]

    COMPATIBILITY_MATRIX = {
        ("thermodatabase", "thermal_2d_cnn"): (True, "Calibrated (160, 80) bilateral thermograms for cohort classification.", []),
        ("iac_tecmed", "thermal_tcn"): (True, "Calibrated 4-timepoint rewarming TIFF sequences.", ["Small sample size (N=19); exploratory trajectory modeling only."]),
        ("cad_walk", "pressure_1d_cnn"): (True, "100-pt normalized vGRF stance curves for walking speed regression.", ["Healthy controls only; no diabetic/ulcer labels."]),
        ("cad_walk", "pressure_tcn"): (True, "100-pt stance curves for experimental causal TCN regression.", ["Phase 4/5 showed Pressure TCN underperforms on normalized closed stance curves."]),
        ("cad_walk", "pressure_ridge_baseline"): (True, "Scalar stance duration + peak force for walking speed regression.", ["Healthy controls only."]),
        ("synthetic_paired_readings", "pressure_1d_cnn"): (
            False,
            "BLOCKED FOR TRAINING: Synthetic CSV contains no non-derived ground truth labels (plantar_total_load_n = sum of 6 pressure sites; peak_site_pressure_n = max of 6 sites). Inspection/preview only.",
            ["SYNTHETIC_NOT_CLINICAL_GROUND_TRUTH"],
        ),
        ("synthetic_paired_readings", "pressure_tcn"): (
            False,
            "BLOCKED FOR TRAINING: Synthetic CSV contains no non-derived ground truth labels (plantar_total_load_n = sum of 6 pressure sites; peak_site_pressure_n = max of 6 sites). Inspection/preview only.",
            ["SYNTHETIC_NOT_CLINICAL_GROUND_TRUTH"],
        ),
    }

    def __init__(self, dataset_service: Optional[DatasetService] = None):
        self.dataset_service = dataset_service or DatasetService()

    def validate_dataframe(self, dataset_id: str, df: pd.DataFrame) -> ValidationReport:
        """Validates a loaded DataFrame preview or sample batch."""
        desc = self.dataset_service.get_descriptor(dataset_id)
        warnings: List[str] = []
        errors: List[str] = []

        if df is None or df.empty:
            errors.append("Dataset DataFrame is empty.")
            return ValidationReport(
                is_valid=False,
                dataset_id=dataset_id,
                origin=desc.origin,
                provenance_label=desc.provenance_label,
                shape=[0, 0],
                missing_values_count=0,
                missing_by_column={},
                label_summary={},
                warnings=warnings,
                errors=errors,
            )

        missing_counts = {col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().sum() > 0}
        total_missing = int(sum(missing_counts.values()))
        if total_missing > 0:
            warnings.append(f"Detected {total_missing} missing values across {len(missing_counts)} columns.")

        # Provenance & schema checks
        if dataset_id == "synthetic_paired_readings":
            missing_req = [c for c in self.SYNTHETIC_REQUIRED_COLS if c not in df.columns]
            if missing_req:
                errors.append(f"Missing required synthetic columns: {missing_req}")
            if "data_status" in df.columns:
                statuses = set(df["data_status"].dropna().unique())
                if statuses != {"SYNTHETIC_NOT_CLINICAL_GROUND_TRUTH"}:
                    warnings.append(f"Unexpected data_status values: {statuses}")
            else:
                errors.append("Synthetic dataset missing 'data_status' provenance column.")

        # Calibration warning for STANDUP
        if dataset_id == "standup":
            warnings.append("Uncalibrated 8-bit PNG images; Celsius thermal inference is strictly blocked.")

        # Label inspection
        label_summary: Dict[str, Any] = {}
        for label_col in ["label", "class_name", "risk_group", "walking_speed", "peak_pressure_site"]:
            if label_col in df.columns:
                if pd.api.types.is_numeric_dtype(df[label_col]) and label_col == "walking_speed":
                    valid_vals = df[label_col][df[label_col] > 0.5]
                    zero_count = int((df[label_col] <= 0.5).sum())
                    label_summary[label_col] = {
                        "min": round(float(valid_vals.min()), 3) if len(valid_vals) else None,
                        "max": round(float(valid_vals.max()), 3) if len(valid_vals) else None,
                        "mean": round(float(valid_vals.mean()), 3) if len(valid_vals) else None,
                        "unlabeled_count": zero_count,
                    }
                    if zero_count > 0:
                        warnings.append(f"{zero_count} trials have missing/unrecorded walking_speed (excluded during training).")
                else:
                    vc = df[label_col].value_counts(dropna=False).to_dict()
                    label_summary[label_col] = {str(k): int(v) for k, v in vc.items()}

        return ValidationReport(
            is_valid=(len(errors) == 0),
            dataset_id=dataset_id,
            origin=desc.origin,
            provenance_label=desc.provenance_label,
            shape=[int(df.shape[0]), int(df.shape[1])],
            missing_values_count=total_missing,
            missing_by_column=missing_counts,
            label_summary=label_summary,
            warnings=warnings,
            errors=errors,
        )

    def check_model_compatibility(self, dataset_id: str, model_id: str) -> CompatibilityResult:
        """Checks whether a dataset can be legitimately paired with a model architecture."""
        # Ensure dataset exists
        desc = self.dataset_service.get_descriptor(dataset_id)

        if dataset_id == "standup":
            return CompatibilityResult(
                is_compatible=False,
                dataset_id=dataset_id,
                model_id=model_id,
                reason="BLOCKED: STANDUP images are uncalibrated pseudo-color PNGs and lack participant IDs.",
                warnings=[],
            )

        if dataset_id == "gait_lab":
            return CompatibilityResult(
                is_compatible=False,
                dataset_id=dataset_id,
                model_id=model_id,
                reason="BLOCKED: Gait Lab contains only N=2 participants; supervised training/evaluation is confounded.",
                warnings=[],
            )

        key = (dataset_id, model_id)
        if key in self.COMPATIBILITY_MATRIX:
            ok, reason, warns = self.COMPATIBILITY_MATRIX[key]
            return CompatibilityResult(
                is_compatible=ok,
                dataset_id=dataset_id,
                model_id=model_id,
                reason=reason,
                warnings=warns,
            )

        return CompatibilityResult(
            is_compatible=False,
            dataset_id=dataset_id,
            model_id=model_id,
            reason=(
                f"Incompatible pairing: dataset '{dataset_id}' ({desc.modality}, origin={desc.origin}) "
                f"does not match input contract of '{model_id}'."
            ),
            warnings=[],
        )

