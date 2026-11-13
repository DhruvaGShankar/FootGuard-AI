"""
FootGuard AI - Unit and Integration Tests for ML Inference Pipelines
Tests ThermalInferenceEngine and PressureInferenceEngine across input types,
edge cases, error conditions, uncalibrated inputs, missing checkpoints,
determinism, DEMO mode separation, and Pydantic schema validation.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import torch

ML_ROOT = Path(__file__).resolve().parent.parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from common.config import THERMAL_TDB_DIR, CHECKPOINTS_DIR
from common.schemas import (
    ThermalInferenceRequest,
    ThermalInferenceResponse,
    PressureInferenceRequest,
    PressureInferenceResponse
)
from thermal.inference import ThermalInferenceEngine
from pressure.inference import PressureInferenceEngine


# =====================================================================
# 1. THERMAL INFERENCE ENGINE TESTS
# =====================================================================

def test_thermal_inference_initialization():
    """Verifies ThermalInferenceEngine initializes cleanly with default and custom checkpoints."""
    engine = ThermalInferenceEngine(device="cpu")
    assert engine.model is not None
    assert engine.model_id == "thermal_2d_cnn"
    assert engine.model_version == "1.0.0"
    assert engine.decision_threshold == 0.50


def test_thermal_missing_checkpoint():
    """Verifies that pointing to a non-existent checkpoint raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Thermal CNN checkpoint not found"):
        ThermalInferenceEngine(checkpoint_path="non_existent_thermal_checkpoint.pt")


def test_thermal_inference_synthetic_arrays():
    """Tests thermal inference using synthetic Left and Right foot float32 arrays."""
    engine = ThermalInferenceEngine(device="cpu")
    np.random.seed(42)
    left_arr = np.random.uniform(24.0, 32.0, (140, 70)).astype(np.float32)
    right_arr = np.random.uniform(24.0, 32.0, (140, 70)).astype(np.float32)

    response = engine.predict((left_arr, right_arr))

    assert isinstance(response, ThermalInferenceResponse)
    assert response.status in ["success", "warning"]
    assert response.prediction.predicted_class_id in [0, 1]
    assert response.prediction.predicted_label in ["Control Cohort", "Diabetic Cohort"]
    assert 0.0 <= response.prediction.probability_control <= 1.0
    assert 0.0 <= response.prediction.probability_diabetic <= 1.0
    assert abs(response.prediction.probability_control + response.prediction.probability_diabetic - 1.0) < 1e-4
    assert response.bilateral_metrics.mean_left_temp_c > 0.0
    assert response.bilateral_metrics.mean_right_temp_c > 0.0
    assert response.latency_ms > 0.0
    assert "RESEARCH USE ONLY" in response.disclaimer


def test_thermal_inference_real_csv_files():
    """Tests thermal inference reading raw calibrated CSV files from ThermoDataBase."""
    ctrl_dir = THERMAL_TDB_DIR / "Control Group" / "CG001_M"
    left_csv = ctrl_dir / "CG001_M_L.csv"
    right_csv = ctrl_dir / "CG001_M_R.csv"

    if not left_csv.exists() or not right_csv.exists():
        pytest.skip(f"ThermoDataBase raw files not present at {ctrl_dir}")

    engine = ThermalInferenceEngine(device="cpu")
    response = engine.predict((left_csv, right_csv))

    assert isinstance(response, ThermalInferenceResponse)
    assert response.status == "success"
    assert response.bilateral_metrics.mean_abs_asymmetry_c >= 0.0
    assert len(response.input_hash) == 64


def test_thermal_inference_pydantic_request():
    """Tests thermal inference when passed a validated ThermalInferenceRequest payload."""
    engine = ThermalInferenceEngine(device="cpu")
    left_mat = [[28.0] * 50 for _ in range(80)]
    right_mat = [[28.2] * 50 for _ in range(80)]

    req = ThermalInferenceRequest(
        patient_id="ANON_TEST_01",
        left_foot_array=left_mat,
        right_foot_array=right_mat
    )

    response = engine.predict(req)
    assert isinstance(response, ThermalInferenceResponse)
    assert response.prediction is not None
    assert response.latency_ms > 0.0


def test_thermal_inference_invalid_inputs():
    """Tests robust error handling for missing files, invalid shapes, excessive NaNs, and low signal."""
    engine = ThermalInferenceEngine(device="cpu")

    # Missing CSV file
    with pytest.raises(FileNotFoundError):
        engine.predict(("non_existent_left.csv", "non_existent_right.csv"))

    # 1D array instead of 2D
    with pytest.raises(ValueError, match="must be 2D"):
        engine.predict((np.zeros(50), np.zeros(50)))

    # Array dimensions too small (< (30, 15))
    with pytest.raises(ValueError, match="dimensions too small"):
        engine.predict((np.full((20, 10), 28.0, dtype=np.float32), np.full((20, 10), 28.0, dtype=np.float32)))

    # Excessive NaNs (> 30%)
    left_heavy_nan = np.full((60, 40), 28.0, dtype=np.float32)
    left_heavy_nan[:30, :] = np.nan  # 50% NaNs
    right_clean = np.full((60, 40), 28.0, dtype=np.float32)
    with pytest.raises(ValueError, match="Excessive sensor dropout"):
        engine.predict((left_heavy_nan, right_clean))

    # Insufficient thermal signal (< 50 plantar pixels above 15.0°C)
    left_flat = np.full((60, 40), 10.0, dtype=np.float32)  # All below 15°C
    with pytest.raises(ValueError, match="Insufficient thermal signal"):
        engine.predict((left_flat, right_clean))

    # Acceptable small NaN count (<30%) should be handled gracefully with warning
    left_few_nan = np.full((60, 40), 28.0, dtype=np.float32)
    left_few_nan[5, 5] = np.nan
    resp = engine.predict((left_few_nan, right_clean))
    assert isinstance(resp, ThermalInferenceResponse)
    assert any("Sensor dropout" in w for w in resp.quality_flags.warnings)


def test_thermal_inference_uncalibrated_inputs():
    """Verifies that uncalibrated images, uint8 arrays, and STANDUP files are strictly rejected."""
    engine = ThermalInferenceEngine(device="cpu")

    # 1. 8-bit integer array (uint8)
    left_uint8 = np.full((60, 40), 128, dtype=np.uint8)
    right_uint8 = np.full((60, 40), 128, dtype=np.uint8)
    with pytest.raises(ValueError, match="Uncalibrated 8-bit integer matrix"):
        engine.predict((left_uint8, right_uint8))

    # 2. Values in uncalibrated intensity range (> 50.0°C)
    left_high_val = np.full((60, 40), 180.0, dtype=np.float32)
    right_normal = np.full((60, 40), 28.0, dtype=np.float32)
    with pytest.raises(ValueError, match="Uncalibrated image values detected"):
        engine.predict((left_high_val, right_normal))

    # 3. Path ending in .png / .jpg
    with pytest.raises(ValueError, match="Uncalibrated image format"):
        engine.predict(("sample_foot.png", "sample_foot.png"))

    # 4. Path containing 'standup'
    with pytest.raises(ValueError, match="STANDUP"):
        engine.predict(("raw/Thermal-Data/STANDUP_Database/R0_0001.csv", "raw/Thermal-Data/STANDUP_Database/R0_0001.csv"))


# =====================================================================
# 2. PRESSURE INFERENCE ENGINE TESTS
# =====================================================================

def test_pressure_inference_initialization():
    """Verifies PressureInferenceEngine initializes cleanly with default checkpoints."""
    engine = PressureInferenceEngine(device="cpu")
    assert engine.model is not None
    assert engine.model_id == "pressure_1d_cnn"
    assert engine.model_version == "1.0.0"
    assert engine.ridge_model is not None


def test_pressure_missing_checkpoint():
    """Verifies that pointing to a non-existent checkpoint raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Pressure CNN checkpoint not found"):
        PressureInferenceEngine(checkpoint_path="non_existent_pressure_checkpoint.pt")


def test_pressure_inference_100_pt_curve():
    """Tests pressure inference on canonical 100-point normalized vertical force curve."""
    engine = PressureInferenceEngine(device="cpu")
    curve = np.sin(np.linspace(0, np.pi, 100)) * 820.0

    response = engine.predict(curve)
    assert isinstance(response, PressureInferenceResponse)
    assert response.status == "success"
    assert response.is_demo is False
    assert response.prediction.predicted_walking_speed_kmh is not None
    assert response.prediction.predicted_walking_speed_kmh >= 0.0
    assert response.prediction.baseline_ridge_speed_kmh >= 0.0
    assert response.prediction.primary_model_used == "pressure_1d_cnn"
    assert response.prediction.speed_bracket in [
        "slow (<4.5 km/h)", "normal (4.5-5.5 km/h)", "fast (>5.5 km/h)"
    ]
    assert response.stance_dynamics.peak_force_n > 0.0
    assert response.stance_dynamics.impulse_n_s > 0.0
    assert response.stance_dynamics.loading_rate_n_s > 0.0
    assert response.uncertainty.estimated_mae_kmh == 0.431
    assert len(response.uncertainty.confidence_interval_95) == 2
    assert "RESEARCH USE ONLY" in response.disclaimer


def test_pressure_inference_arbitrary_length_curve():
    """Tests automatic 1D temporal resampling from arbitrary sample lengths (e.g. 250 pts)."""
    engine = PressureInferenceEngine(device="cpu")
    curve_long = np.sin(np.linspace(0, np.pi, 250)) * 780.0

    req = PressureInferenceRequest(
        vertical_force_curve=curve_long.tolist(),
        stance_duration_ms=600.0,
        sample_rate_hz=250.0
    )

    response = engine.predict(req)
    assert isinstance(response, PressureInferenceResponse)
    assert response.status == "success"
    assert response.is_demo is False
    assert response.stance_dynamics.stance_duration_ms == 600.0
    assert response.prediction.predicted_walking_speed_kmh is not None
    assert response.latency_ms > 0.0


def test_pressure_inference_scalar_real_vs_demo():
    """
    Tests rigorous separation between real production scalar inference
    and synthetic DEMO mode.
    """
    engine = PressureInferenceEngine(device="cpu")

    # A. PRODUCTION MODE (allow_synthetic_demo=False by default)
    # Must NOT run 1D CNN on synthetic curves; must run Ridge baseline legitimately
    req_prod = PressureInferenceRequest(
        patient_id="SUBJ_202",
        stance_duration_ms=615.0,
        peak_force_n=840.0,
        allow_synthetic_demo=False
    )
    resp_prod = engine.predict(req_prod)
    assert isinstance(resp_prod, PressureInferenceResponse)
    assert resp_prod.is_demo is False
    assert resp_prod.prediction.predicted_walking_speed_kmh is None  # CNN omitted
    assert resp_prod.prediction.baseline_ridge_speed_kmh is not None
    assert resp_prod.prediction.baseline_ridge_speed_kmh > 0.0
    assert resp_prod.prediction.primary_model_used == "pressure_ridge_baseline"
    assert resp_prod.stance_dynamics.impulse_n_s is None  # Cannot integrate curve without real signal
    assert resp_prod.uncertainty.estimated_mae_kmh == 0.259  # Ridge benchmark MAE
    assert any("Scalar-only" in w for w in resp_prod.quality_flags.warnings)

    # B. DEMO MODE (allow_synthetic_demo=True)
    # Permitted to generate synthetic bimodal curve, must be prominently flagged as DEMO
    req_demo = PressureInferenceRequest(
        patient_id="SUBJ_202",
        stance_duration_ms=615.0,
        peak_force_n=840.0,
        allow_synthetic_demo=True
    )
    resp_demo = engine.predict(req_demo)
    assert isinstance(resp_demo, PressureInferenceResponse)
    assert resp_demo.is_demo is True
    assert resp_demo.status == "warning"
    assert resp_demo.prediction.predicted_walking_speed_kmh is not None  # CNN evaluated on demo curve
    assert resp_demo.prediction.baseline_ridge_speed_kmh is not None
    assert any("[DEMO MODE]" in w for w in resp_demo.quality_flags.warnings)


def test_pressure_inference_invalid_inputs():
    """Tests robust error handling for missing inputs, zero signals, and invalid bounds."""
    engine = PressureInferenceEngine(device="cpu")

    # Missing both curve and scalars
    with pytest.raises(ValueError, match="Insufficient input"):
        engine.predict({})

    # Curve length too short (<10 points)
    short_curve = [0.0, 100.0, 500.0, 0.0]
    with pytest.raises(ValueError, match="too short"):
        engine.predict(short_curve)

    # Flatline / zero force curve (<50 N peak)
    flat_curve = np.full(100, 10.0, dtype=np.float32)
    with pytest.raises(ValueError, match="Insufficient vertical force signal"):
        engine.predict(flat_curve)

    # Zero or negative stance duration
    with pytest.raises(ValueError, match="strictly positive"):
        engine.predict({"stance_duration_ms": 0.0, "peak_force_n": 800.0})

    # Negative peak force
    with pytest.raises(ValueError, match="strictly positive"):
        engine.predict({"stance_duration_ms": 600.0, "peak_force_n": -100.0})

    # Stance duration exceeding limit (>3000 ms)
    with pytest.raises(ValueError, match="physiological limit"):
        engine.predict({"stance_duration_ms": 4000.0, "peak_force_n": 800.0})


def test_inference_determinism():
    """Verifies that identical inputs produce identical hash and numeric predictions."""
    t_engine = ThermalInferenceEngine(device="cpu")
    p_engine = PressureInferenceEngine(device="cpu")

    np.random.seed(123)
    left = np.random.uniform(25.0, 30.0, (100, 50)).astype(np.float32)
    right = np.random.uniform(25.0, 30.0, (100, 50)).astype(np.float32)

    t_res1 = t_engine.predict((left, right))
    t_res2 = t_engine.predict((left, right))
    assert t_res1.input_hash == t_res2.input_hash
    assert t_res1.prediction.probability_diabetic == t_res2.prediction.probability_diabetic
    assert t_res1.bilateral_metrics.mean_abs_asymmetry_c == t_res2.bilateral_metrics.mean_abs_asymmetry_c

    curve = np.sin(np.linspace(0, np.pi, 100)) * 750.0
    p_res1 = p_engine.predict(curve)
    p_res2 = p_engine.predict(curve)
    assert p_res1.input_hash == p_res2.input_hash
    assert p_res1.prediction.predicted_walking_speed_kmh == p_res2.prediction.predicted_walking_speed_kmh
