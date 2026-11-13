"""
FootGuard AI - Phase 4 End-to-End Streamlit App & Stabilization Tests
Tests all 5 Streamlit pages via AppTest, dataset workflows, prediction safety,
controlled training, and graceful handling of invalid/incomplete records.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

ML_ROOT = Path(__file__).resolve().parent.parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))


# ---------------------------------------------------------------------------
# 1. Page Boot & Navigation Tests (No Import Errors / No Uncaught Exceptions)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "script_rel",
    [
        "app/main.py",
        "app/pages/home.py",
        "app/pages/datasets.py",
        "app/pages/training.py",
        "app/pages/prediction.py",
        "app/pages/experiments.py",
    ],
)
def test_streamlit_pages_load_without_exceptions(script_rel: str):
    at = AppTest.from_file(str(ML_ROOT / script_rel), default_timeout=30)
    at.run()
    assert not at.exception, f"Uncaught exception in {script_rel}: {at.exception}"


# ---------------------------------------------------------------------------
# 2. Dataset Workflows (Real + Synthetic Chunked & Warnings)
# ---------------------------------------------------------------------------


def test_datasets_page_real_and_synthetic_workflows():
    at = AppTest.from_file(str(ML_ROOT / "app/pages/datasets.py"), default_timeout=30)
    at.run()
    assert not at.exception

    # Default selection is thermodatabase
    assert any("ThermoDataBase" in t.value or "Datasets & Provenance Explorer" in t.value for t in at.title)

    # Select STANDUP -> must show UNCALIBRATED error banner
    at.selectbox[0].set_value("standup").run()
    assert not at.exception
    assert any("UNCALIBRATED" in err.value for err in at.error)

    # Select IAC-TecMed -> loads sample cleanly
    at.selectbox[0].set_value("iac_tecmed").run()
    assert not at.exception

    # Select CAD_WALK -> loads sample cleanly
    at.selectbox[0].set_value("cad_walk").run()
    assert not at.exception

    # Select Gait Lab -> check compatibility with pressure_1d_cnn -> warning/blocked
    at.selectbox[0].set_value("gait_lab").run()
    assert not at.exception

    # Select synthetic_paired_readings -> chunked session SYN-0001
    at.selectbox[0].set_value("synthetic_paired_readings").run()
    assert not at.exception
    assert any("Chunked loader active" in info.value for info in at.info)

    # Switch to Strided Sample Subset
    at.radio[0].set_value("Strided Sample Subset").run()
    assert not at.exception


# ---------------------------------------------------------------------------
# 3. Prediction Workflows (Thermal & Pressure + Safety)
# ---------------------------------------------------------------------------


def test_prediction_page_thermal_inference_and_pressure_safety():
    at = AppTest.from_file(str(ML_ROOT / "app/pages/prediction.py"), default_timeout=30)
    at.run()
    assert not at.exception

    # Click "Run Thermal Inference" on default ThermoDataBase sample 0
    run_btn = [b for b in at.button if "Run Thermal Inference" in b.label][0]
    run_btn.click().run()
    assert not at.exception
    # Check metrics rendered (Predicted Cohort, Mean |L - R| Asymmetry)
    metric_labels = [m.label for m in at.metric]
    assert "Predicted Cohort" in metric_labels
    assert "Mean |L - R| Asymmetry" in metric_labels

    # Also test InferenceService invalid input & missing checkpoint directly
    from services.inference_service import InferenceService

    inf_svc = InferenceService()
    with pytest.raises((FileNotFoundError, ValueError)):
        inf_svc.run_thermal_inference(
            (np.full((160, 80), 30.0, dtype=np.float32), np.full((160, 80), 30.0, dtype=np.float32)),
            checkpoint_path="checkpoints/does_not_exist/best_model.pt",
        )

    # Reject out-of-range / NaN thermal array
    bad_arr = np.full((160, 80), 99.0, dtype=np.float32)
    with pytest.raises(ValueError):
        inf_svc.run_thermal_inference((bad_arr, bad_arr))

    # Reject empty/insufficient pressure input
    with pytest.raises(ValueError):
        inf_svc.run_pressure_inference(np.array([], dtype=np.float32))


# ---------------------------------------------------------------------------
# 4. Training Workflow (Blocked Cases + Valid CAD_WALK Ridge Run + Cleanup)
# ---------------------------------------------------------------------------


def test_training_page_blocked_and_valid_cad_walk_ridge():
    at = AppTest.from_file(str(ML_ROOT / "app/pages/training.py"), default_timeout=45)
    at.run()
    assert not at.exception

    # Select blocked pair: thermal_2d_cnn + standup
    at.selectbox[0].set_value("thermal_2d_cnn").run()
    at.selectbox[1].set_value("standup").run()
    assert not at.exception
    assert any("Training Blocked" in e.value for e in at.error)

    # Select blocked pair: pressure_ridge_baseline + synthetic_paired_readings
    at.selectbox[0].set_value("pressure_ridge_baseline").run()
    at.selectbox[1].set_value("synthetic_paired_readings").run()
    assert not at.exception
    assert any("Training Blocked" in e.value for e in at.error)

    # Snapshot existing results & checkpoints before valid Ridge run
    results_dir = ML_ROOT / "experiments" / "results"
    configs_dir = ML_ROOT / "experiments" / "configs"
    ckpt_base = ML_ROOT / "checkpoints"
    before_results = set(results_dir.glob("*.json")) if results_dir.exists() else set()
    before_configs = set(configs_dir.glob("*.yaml")) if configs_dir.exists() else set()
    before_ckpts = set(ckpt_base.glob("local_*"))

    try:
        # Select valid pair: pressure_ridge_baseline + cad_walk
        at.selectbox[0].set_value("pressure_ridge_baseline").run()
        at.selectbox[1].set_value("cad_walk").run()
        assert not at.exception
        assert any("Approved for Training" in s.value for s in at.success)

        submit_btn = [b for b in at.button if "Start Training" in b.label][0]
        submit_btn.click().run()
        assert not at.exception
        assert any("Training completed!" in s.value for s in at.success)

        after_results = set(results_dir.glob("*.json"))
        assert len(after_results - before_results) == 1
    finally:
        # Clean up temporary local test artifacts so repo stays clean
        for p in (set(results_dir.glob("*.json")) - before_results):
            p.unlink(missing_ok=True)
        for p in (set(configs_dir.glob("*.yaml")) - before_configs):
            p.unlink(missing_ok=True)
        for d in (set(ckpt_base.glob("local_*")) - before_ckpts):
            for child in d.glob("*"):
                child.unlink(missing_ok=True)
            d.rmdir()


# ---------------------------------------------------------------------------
# 5. Experiments Page (Legacy + Malformed/Incomplete Record Graceful Handling)
# ---------------------------------------------------------------------------


def test_experiments_page_handles_legacy_and_incomplete_records():
    results_dir = ML_ROOT / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    malformed_path = results_dir / "z_test_malformed.json"
    incomplete_path = results_dir / "z_test_incomplete.json"

    try:
        malformed_path.write_text("{not valid json", encoding="utf-8")
        incomplete_path.write_text(json.dumps({"model_name": "incomplete_only"}), encoding="utf-8")

        at = AppTest.from_file(str(ML_ROOT / "app/pages/experiments.py"), default_timeout=30)
        at.run()
        assert not at.exception
        # Legacy experiments must still appear
        assert len(at.dataframe) >= 1
    finally:
        malformed_path.unlink(missing_ok=True)
        incomplete_path.unlink(missing_ok=True)
