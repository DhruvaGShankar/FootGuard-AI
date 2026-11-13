"""
FootGuard AI - Unit & Integration Tests for Phase 2 Python Application Services
Verifies ConfigService, DatasetService, ValidationService, ModelService,
InferenceService, EvaluationService, ExperimentService, and TrainingService.
"""

import shutil
import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

ML_ROOT = Path(__file__).resolve().parent.parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from services.config_service import ConfigService
from services.dataset_service import DatasetService
from services.validation_service import ValidationService
from services.model_service import ModelService
from services.inference_service import InferenceService
from services.evaluation_service import EvaluationService
from services.experiment_service import ExperimentService
from services.training_service import TrainingService


def test_config_service_loads_yaml():
    cfg = ConfigService()
    assert cfg.get_path("checkpoints_dir").exists()
    assert cfg.get_path("experiments_results_dir").exists()
    assert cfg.get_path("data_synthetic_dir").exists()
    assert "RESEARCH USE ONLY" in cfg.get_disclaimer()


def test_dataset_service_metadata_without_full_load():
    ds = DatasetService()
    datasets = ds.list_datasets()
    ids = {d.dataset_id for d in datasets}
    assert {"thermodatabase", "iac_tecmed", "standup", "cad_walk", "gait_lab", "synthetic_paired_readings"}.issubset(ids)

    # Inspect 258.9 MB synthetic dataset metadata without loading all 1M rows
    meta = ds.inspect_metadata("synthetic_paired_readings")
    assert meta["descriptor"]["origin"] == "synthetic"
    assert meta["descriptor"]["provenance_label"] == "SYNTHETIC_NOT_CLINICAL_GROUND_TRUTH"
    assert len(meta["columns"]) == 35
    assert len(meta["sample_session_ids"]) >= 1


def test_dataset_service_load_single_synthetic_session():
    ds = DatasetService()
    df_session = ds.load_synthetic_session("SYN-0001")
    assert not df_session.empty
    assert (df_session["session_id"] == "SYN-0001").all()
    assert (df_session["data_status"] == "SYNTHETIC_NOT_CLINICAL_GROUND_TRUTH").all()


def test_dataset_service_sample_synthetic_rows():
    ds = DatasetService()
    df_sample = ds.sample_synthetic_rows(n_rows=150, seed=42)
    assert len(df_sample) == 150
    assert "plantar_total_load_n" in df_sample.columns
    assert "thermal_mean_c" in df_sample.columns


def test_dataset_service_real_samples():
    ds = DatasetService()
    t_sample = ds.get_real_sample("thermodatabase", index=0)
    assert t_sample["origin"] == "real"
    assert t_sample["is_calibrated_celsius"] is True
    assert "tensor" in t_sample["arrays"]

    p_sample = ds.get_real_sample("cad_walk", index=0)
    assert p_sample["origin"] == "real"
    assert "force_curve" in p_sample["arrays"]


def test_validation_service_and_compatibility():
    ds = DatasetService()
    vs = ValidationService(ds)

    df_syn = ds.sample_synthetic_rows(n_rows=50, seed=42)
    rep = vs.validate_dataframe("synthetic_paired_readings", df_syn)
    assert rep.is_valid is True
    assert rep.origin == "synthetic"
    assert rep.provenance_label == "SYNTHETIC_NOT_CLINICAL_GROUND_TRUTH"

    empty_rep = vs.validate_dataframe("synthetic_paired_readings", pd.DataFrame())
    assert empty_rep.is_valid is False

    # Valid pairings
    assert vs.check_model_compatibility("thermodatabase", "thermal_2d_cnn").is_compatible is True
    assert vs.check_model_compatibility("cad_walk", "pressure_1d_cnn").is_compatible is True
    assert vs.check_model_compatibility("cad_walk", "pressure_ridge_baseline").is_compatible is True

    # Blocked / incompatible pairings
    assert vs.check_model_compatibility("standup", "thermal_2d_cnn").is_compatible is False
    assert vs.check_model_compatibility("gait_lab", "pressure_1d_cnn").is_compatible is False
    assert vs.check_model_compatibility("synthetic_paired_readings", "thermal_2d_cnn").is_compatible is False
    assert vs.check_model_compatibility("synthetic_paired_readings", "pressure_1d_cnn").is_compatible is False


def test_model_service_loads_existing_checkpoints():
    ms = ModelService()
    specs = ms.list_models()
    assert len(specs) == 6

    ckpts = ms.list_checkpoints()
    assert len(ckpts) >= 4

    t_model, t_meta = ms.load_checkpoint("thermal_2d_cnn")
    assert t_model is not None
    assert t_meta["model_id"] == "thermal_2d_cnn"

    p_model, p_meta = ms.load_checkpoint("pressure_1d_cnn")
    assert p_model is not None
    assert p_meta["model_id"] == "pressure_1d_cnn"

    r_model, r_meta = ms.load_checkpoint("pressure_ridge_baseline")
    assert r_model is not None
    assert r_meta["model_id"] == "pressure_ridge_baseline"

    with pytest.raises(FileNotFoundError):
        ms.load_checkpoint("thermal_2d_cnn", checkpoint_path="missing_ckpt.pt")


def test_inference_service_executes_existing_engines():
    ds = DatasetService()
    inf = InferenceService()

    # Real ThermoDataBase sample -> ThermalInferenceEngine
    t_sample = ds.get_real_sample("thermodatabase", index=0)
    tensor = t_sample["arrays"]["tensor"]  # (3, 160, 80): [Left, Right_flipped, Delta_T]
    left = tensor[0]
    right = np.fliplr(tensor[1]).copy()
    t_res = inf.run_thermal_inference((left, right), data_origin="real")
    assert t_res["status"] in ["success", "warning"]
    assert t_res["data_origin"] == "real"
    assert "prediction" in t_res

    # Real CAD_WALK sample -> PressureInferenceEngine
    p_sample = ds.get_real_sample("cad_walk", index=0)
    force_curve = p_sample["arrays"]["force_curve"]
    p_res = inf.run_pressure_inference(force_curve, data_origin="real")
    assert p_res["status"] in ["success", "warning"]
    assert p_res["is_demo"] is False
    assert p_res["prediction"]["predicted_walking_speed_kmh"] is not None


def test_experiment_and_evaluation_services_roundtrip():
    cfg = ConfigService()
    es = ExperimentService(cfg)
    ev = EvaluationService()

    # Legacy experiments must be discoverable
    all_exps = es.list_experiments(include_legacy=True)
    assert any(e["experiment_id"] == "legacy_experiment_a_thermal_2d_cnn" for e in all_exps)

    # Save & reload a test experiment record, then clean up
    test_id = "pytest_temp_exp_roundtrip"
    metrics = ev.evaluate_regression(np.array([4.0, 5.0, 6.0]), np.array([4.1, 4.9, 6.2]))
    record = {
        "experiment_id": test_id,
        "dataset_name": "synthetic_paired_readings",
        "dataset_version": "1.0",
        "dataset_origin": "synthetic",
        "model_name": "pressure_1d_cnn",
        "model_version": "1.0.0",
        "hyperparameters": {"epochs": 1, "lr": 0.001},
        "split_methodology": "session-level",
        "training_configuration": {"device": "cpu"},
        "evaluation_metrics": metrics,
        "checkpoint_location": "checkpoints/test/best_model.pt",
        "timestamp": "",
    }
    try:
        saved_path = es.save_experiment(record)
        assert saved_path.exists()
        reloaded = es.load_experiment(test_id)
        assert reloaded["model_name"] == "pressure_1d_cnn"
        assert reloaded["dataset_origin"] == "synthetic"
        md_report = es.export_report(test_id, fmt="md")
        assert test_id in md_report
    finally:
        for p in [
            cfg.get_path("experiments_results_dir") / f"{test_id}.json",
            cfg.get_path("experiments_configs_dir") / f"{test_id}.yaml",
        ]:
            if p.exists():
                p.unlink()


def test_training_service_synthetic_and_blocked_validation():
    cfg = ConfigService()
    ts = TrainingService(cfg)

    # Blocked incompatible training (uncalibrated STANDUP & target-leaking synthetic CSV)
    with pytest.raises(ValueError, match="Training blocked"):
        ts.start_training(model_id="thermal_2d_cnn", dataset_id="standup")

    with pytest.raises(ValueError, match="Training blocked"):
        ts.start_training(model_id="pressure_1d_cnn", dataset_id="synthetic_paired_readings")

    # Protected Phase 4 checkpoint directory cannot be overwritten
    with pytest.raises(ValueError, match="Cannot overwrite protected"):
        ts.start_training(
            model_id="pressure_ridge_baseline",
            dataset_id="cad_walk",
            experiment_id="experiment_b_pressure_1d_cnn",
        )

    # Valid training on real CAD_WALK (Ridge baseline) -> save & reload checkpoint
    test_exp_id = "pytest_temp_cad_walk_ridge"
    try:
        result = ts.start_training(
            model_id="pressure_ridge_baseline",
            dataset_id="cad_walk",
            hyperparameters={"alpha": 1.0, "seed": 42},
            experiment_id=test_exp_id,
        )
        assert result["dataset_origin"] == "real"
        assert result["dataset_name"] == "cad_walk"
        assert Path(result["checkpoint_location"]).exists()
        assert "mae" in result["evaluation_metrics"]

        # Verify saved checkpoint reloads via ModelService
        loaded_model, meta = ts.model_service.load_checkpoint(
            "pressure_ridge_baseline", checkpoint_path=result["checkpoint_location"]
        )
        assert loaded_model is not None
        assert meta["model_id"] == "pressure_ridge_baseline"
    finally:
        ckpt_dir = cfg.get_path("checkpoints_dir") / test_exp_id
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        for p in [
            cfg.get_path("experiments_results_dir") / f"{test_exp_id}.json",
            cfg.get_path("experiments_configs_dir") / f"{test_exp_id}.yaml",
            cfg.get_path("experiments_logs_dir") / f"{test_exp_id}.log",
        ]:
            if p.exists():
                p.unlink()
