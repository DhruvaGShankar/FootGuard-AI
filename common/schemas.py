"""
FootGuard AI - Common Pydantic Schemas for ML Inference API
Defines request and response data contracts for backend integration.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator


class ThermalInferenceRequest(BaseModel):
    """
    Thermal inference input payload.
    Supports either direct 2D temperature matrices (Celsius) or file paths to CSV matrices.
    NOTE: Uncalibrated RGB/PNG image formats (e.g. STANDUP Database) are strictly prohibited.
    """
    patient_id: Optional[str] = Field(None, description="Anonymized patient identifier")
    session_id: Optional[str] = Field(None, description="Recording session identifier")
    left_foot_array: Optional[List[List[float]]] = Field(
        None, description="2D temperature matrix (Celsius floats) for Left plantar surface"
    )
    right_foot_array: Optional[List[List[float]]] = Field(
        None, description="2D temperature matrix (Celsius floats) for Right plantar surface"
    )
    left_foot_csv_path: Optional[str] = Field(
        None, description="Path to CSV file containing Left plantar temperatures in Celsius"
    )
    right_foot_csv_path: Optional[str] = Field(
        None, description="Path to CSV file containing Right plantar temperatures in Celsius"
    )
    ambient_temp_c: Optional[float] = Field(
        None, ge=10.0, le=45.0, description="Ambient room temperature in Celsius during acquisition"
    )

    @field_validator("left_foot_csv_path", "right_foot_csv_path")
    @classmethod
    def validate_csv_path(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            lower_v = v.lower()
            if any(lower_v.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]):
                raise ValueError(
                    f"Invalid file format in path '{v}'. Thermal inference requires calibrated Celsius matrices (e.g. CSV). "
                    "Uncalibrated 8-bit image formats (PNG/JPEG) such as STANDUP are prohibited."
                )
            if "standup" in lower_v:
                raise ValueError(
                    f"STANDUP dataset file '{v}' rejected. STANDUP images contain uncalibrated colormap pixel values, "
                    "not physical Celsius temperatures. Celsius inference requires calibrated floating-point thermogram matrices."
                )
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "patient_id": "SUBJ_042",
                "session_id": "SES_01",
                "left_foot_csv_path": "raw/Thermal-Data/ThermoDataBase/Control Group/CG001_M/CG001_M_L.csv",
                "right_foot_csv_path": "raw/Thermal-Data/ThermoDataBase/Control Group/CG001_M/CG001_M_R.csv"
            }
        }
    }


class BilateralThermalMetrics(BaseModel):
    """Thermal asymmetry and regional distribution metrics."""
    mean_left_temp_c: float = Field(..., description="Mean temperature across Left foot surface (°C)")
    mean_right_temp_c: float = Field(..., description="Mean temperature across Right foot surface (°C)")
    mean_abs_asymmetry_c: float = Field(..., description="Mean bilateral absolute temperature difference |L - R| (°C)")
    max_abs_asymmetry_c: float = Field(..., description="Maximum bilateral absolute temperature difference (°C)")
    hyperthermic_asymmetry_fraction: float = Field(
        ..., description="Fraction of foot surface where bilateral asymmetry exceeds clinical threshold (>2.2°C)"
    )


class QualityFlags(BaseModel):
    """Data quality and validity indicators."""
    is_valid: bool = Field(..., description="True if data passed all clinical range and structural integrity checks")
    warnings: List[str] = Field(default_factory=list, description="Diagnostic and quality warnings")


class ThermalPrediction(BaseModel):
    """Model classification prediction details."""
    predicted_class_id: int = Field(..., description="0 for Control Cohort, 1 for Diabetic Cohort")
    predicted_label: str = Field(..., description="'Control Cohort' or 'Diabetic Cohort'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Raw softmax probability of predicted cohort (uncalibrated score)")
    probability_control: float = Field(..., ge=0.0, le=1.0, description="Softmax probability of Control cohort")
    probability_diabetic: float = Field(..., ge=0.0, le=1.0, description="Softmax probability of Diabetic cohort")
    decision_threshold: float = Field(..., description="Validation-tuned decision threshold applied (default 0.50)")


class ThermalInferenceResponse(BaseModel):
    """Thermal inference response payload."""
    model_id: str = Field("thermal_2d_cnn", description="Model architecture identifier")
    model_version: str = Field("1.0.0", description="Model checkpoint version")
    status: str = Field("success", description="'success', 'warning', or 'error'")
    input_hash: str = Field(..., description="SHA-256 hash of input arrays for audit and provenance")
    prediction: ThermalPrediction = Field(..., description="Cohort classification output")
    bilateral_metrics: BilateralThermalMetrics = Field(..., description="Quantitative thermal asymmetry parameters")
    quality_flags: QualityFlags = Field(..., description="Data validation and sensor quality assessment")
    disclaimer: str = Field(
        "RESEARCH USE ONLY: Diabetic cohort screening model. NOT an ulcer detector or clinical diagnostic device.",
        description="Mandatory medical and regulatory disclaimer"
    )
    latency_ms: float = Field(..., description="End-to-end inference processing latency in milliseconds")


class PressureInferenceRequest(BaseModel):
    """
    Pressure / Gait inference input payload.
    Accepts 1D vertical force curves, scalar biomechanical parameters, or file paths.
    """
    patient_id: Optional[str] = Field(None, description="Anonymized participant identifier")
    trial_id: Optional[str] = Field(None, description="Trial or step identifier")
    foot_side: Optional[str] = Field("unspecified", description="'left', 'right', or 'unspecified'")
    vertical_force_curve: Optional[List[float]] = Field(
        None, description="Stance phase vertical ground reaction force curve (N or normalized N)"
    )
    stance_duration_ms: Optional[float] = Field(
        None, gt=0.0, le=3000.0, description="Stance duration / ground contact time in milliseconds"
    )
    peak_force_n: Optional[float] = Field(
        None, gt=0.0, le=5000.0, description="Peak vertical ground reaction force in Newtons"
    )
    sample_rate_hz: Optional[float] = Field(
        100.0, gt=0.0, description="Sampling rate of input force signal in Hertz"
    )
    file_path: Optional[str] = Field(
        None, description="Optional path to CAD_WALK trial mat or npy file"
    )
    allow_synthetic_demo: bool = Field(
        False,
        description=(
            "If True, permits generating a synthetic canonical bimodal vGRF profile when only scalars are provided. "
            "RESTRICTED TO DEMO MODE ONLY. When False (production default), scalar-only input runs the Ridge baseline "
            "model exclusively and omits the 1D CNN."
        )
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "patient_id": "SUBJ_101",
                "trial_id": "T01",
                "foot_side": "left",
                "stance_duration_ms": 620.0,
                "peak_force_n": 820.5,
                "vertical_force_curve": [0.0, 120.0, 450.0, 820.5, 600.0, 750.0, 300.0, 0.0],
                "allow_synthetic_demo": False
            }
        }
    }


class StanceDynamics(BaseModel):
    """Biomechanical stance parameters."""
    peak_force_n: float = Field(..., description="Maximum vertical force during stance (N)")
    stance_duration_ms: float = Field(..., description="Ground contact time (ms)")
    impulse_n_s: Optional[float] = Field(None, description="Vertical force-time integral / impulse (N·s). None in scalar-only mode.")
    loading_rate_n_s: Optional[float] = Field(None, description="Initial vertical loading rate (N/s). None in scalar-only mode.")


class PressurePrediction(BaseModel):
    """Walking speed regression outputs."""
    predicted_walking_speed_kmh: Optional[float] = Field(
        None, description="Walking speed estimated by Pressure 1D CNN (km/h). None if no sensor curve provided and demo mode is disabled."
    )
    baseline_ridge_speed_kmh: Optional[float] = Field(
        None, description="Walking speed estimated by Ridge biomechanical baseline (km/h)"
    )
    primary_model_used: str = Field(
        "pressure_1d_cnn", description="Primary model used for speed estimation ('pressure_1d_cnn' or 'pressure_ridge_baseline')"
    )
    speed_bracket: str = Field(
        ..., description="'slow (<4.5 km/h)', 'normal (4.5-5.5 km/h)', or 'fast (>5.5 km/h)'"
    )


class PressureUncertainty(BaseModel):
    """Empirical confidence intervals and benchmark metrics."""
    estimated_mae_kmh: float = Field(0.431, description="Empirical test-set MAE on CAD_WALK (km/h)")
    confidence_interval_95: List[float] = Field(
        ..., description="Estimated 95% confidence interval for walking speed [lower, upper] (km/h)"
    )


class PressureInferenceResponse(BaseModel):
    """Pressure inference response payload."""
    model_id: str = Field("pressure_1d_cnn", description="Model architecture identifier")
    model_version: str = Field("1.0.0", description="Model checkpoint version")
    status: str = Field("success", description="'success', 'warning', or 'error'")
    is_demo: bool = Field(False, description="True if inference executed in DEMO mode using synthetic signals")
    input_hash: str = Field(..., description="SHA-256 hash of input data for audit and provenance")
    prediction: PressurePrediction = Field(..., description="Kinematic speed estimation")
    stance_dynamics: StanceDynamics = Field(..., description="Extracted biomechanical gait dynamics")
    uncertainty: PressureUncertainty = Field(..., description="Model error margins and confidence intervals")
    quality_flags: QualityFlags = Field(..., description="Sensor signal validity and filtering status")
    disclaimer: str = Field(
        "RESEARCH USE ONLY: Biomechanical walking speed estimation on healthy controls. "
        "NOT an ulcer, neuropathy, or clinical diagnostic tool.",
        description="Mandatory medical and regulatory disclaimer"
    )
    latency_ms: float = Field(..., description="End-to-end inference processing latency in milliseconds")
