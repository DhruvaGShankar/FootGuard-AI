"""
FootGuard AI - Synthetic Generation, Training & CSV Ingestion Tests
Covers synthetic generation, latent target semantics, feature mapping, session-level
disjoint splitting, leakage prevention, checkpoint isolation, CSV ingestion, and UI rendering.
"""

import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ML_ROOT = Path(__file__).resolve().parent.parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from services.config_service import ConfigService
from services.validation_service import ValidationService
from services.synthetic_generator_service import (
    SyntheticGeneratorService,
    SyntheticGenerationConfig,
    SYNTHETIC_PROVENANCE_TAG,
)
from services.synthetic_training_service import SyntheticTrainingService
from services.ingestion_service import DatasetIngestionService, IngestionSpec
from services.experiment_service import ExperimentService


def test_existing_synthetic_paired_readings_still_blocked():
    val_svc = ValidationService()
    for mid in ("pressure_1d_cnn", "pressure_tcn", "pressure_ridge_baseline", "thermal_2d_cnn"):
        res = val_svc.check_model_compatibility("synthetic_paired_readings", mid)
        assert not res.is_compatible, f"Expected synthetic_paired_readings + {mid} to remain blocked."


def test_synthetic_generation_and_target_semantics():
    cfg = ConfigService()
    gen_svc = SyntheticGeneratorService(cfg)
    ds_name = "pytest_syn_gen_temp"
    out_dir = cfg.get_path("data_synthetic_dir") / ds_name

    try:
        manifest = gen_svc.generate(
            SyntheticGenerationConfig(
                dataset_name=ds_name,
                num_sessions=8,
                steps_per_session=30,
                sampling_hz=10.0,
                seed=123,
            )
        )
        assert manifest["data_status"] == SYNTHETIC_PROVENANCE_TAG
        assert manifest["num_sessions"] == 8
        assert manifest["num_rows"] == 240

        df, loaded_manifest, tensors = gen_svc.load_generated_dataset(ds_name)
        assert (df["data_status"] == SYNTHETIC_PROVENANCE_TAG).all()
        for col in gen_svc.THERMAL_FEATURE_COLS + gen_svc.PRESSURE_FEATURE_COLS + gen_svc.IMU_FEATURE_COLS:
            assert col in df.columns
        assert "latent_thermal_regime" in df.columns
        assert "latent_gait_speed_kmh" in df.columns
        # Latent gait speed must not be an algebraic sum/max of pressure sites
        assert not np.allclose(df["latent_gait_speed_kmh"], df["plantar_total_load_n"])
        assert not np.allclose(df["latent_gait_speed_kmh"], df["peak_site_pressure_n"])
        assert tensors["thermal_tensors"].shape == (8, 3, 160, 80)
        assert tensors["vgrf_curves"].shape == (8, 100)
        assert loaded_manifest["generator_config"]["seed"] == 123
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def test_synthetic_training_session_isolation_leakage_and_checkpoint_isolation():
    cfg = ConfigService()
    gen_svc = SyntheticGeneratorService(cfg)
    syn_train = SyntheticTrainingService(cfg, gen_svc)
    exp_svc = ExperimentService(cfg)
    ds_name = "pytest_syn_train_temp"
    out_dir = cfg.get_path("data_synthetic_dir") / ds_name
    created_exp_ids = []

    try:
        gen_svc.generate(
            SyntheticGenerationConfig(
                dataset_name=ds_name,
                num_sessions=8,
                steps_per_session=25,
                seed=77,
            )
        )
        df, manifest, _ = gen_svc.load_generated_dataset(ds_name)

        # 1. Leakage / derived target blocks
        with pytest.raises(ValueError, match="derived pressure totals"):
            syn_train.validate_training_request(df, manifest, "pressure_1d_cnn", "plantar_total_load_n")
        with pytest.raises(ValueError, match="Target leakage blocked"):
            syn_train.validate_training_request(
                df, manifest, "multimodal_fusion_mlp", "latent_gait_speed_kmh",
                feature_cols=["thermal_mean_c", "latent_gait_speed_kmh"],
            )

        # 2. Session-level disjoint split check
        tr, va, te = syn_train.split_sessions_disjoint(sorted(df["session_id"].unique().tolist()), seed=77)
        assert not (set(tr) & set(va))
        assert not (set(tr) & set(te))
        assert not (set(va) & set(te))

        # 3. Train Thermal 2D CNN, Pressure 1D CNN, and Multimodal Fusion MLP
        for model_id, target_col in [
            ("thermal_2d_cnn", "latent_thermal_regime"),
            ("pressure_1d_cnn", "latent_gait_speed_kmh"),
            ("multimodal_fusion_mlp", "latent_gait_speed_kmh"),
        ]:
            exp_id = f"pytest_syn_{model_id}"
            created_exp_ids.append(exp_id)
            rec = syn_train.start_synthetic_training(
                dataset_id=ds_name,
                model_id=model_id,
                target_col=target_col,
                hyperparameters={"epochs": 2, "batch_size": 4, "learning_rate": 0.001, "seed": 77},
                experiment_id=exp_id,
            )
            assert rec["dataset_origin"] == "synthetic"
            assert rec["data_status"] == SYNTHETIC_PROVENANCE_TAG
            assert "synthetic_experiments" in rec["checkpoint_location"].replace("\\", "/")
            assert (cfg.ml_root / rec["checkpoint_location"]).exists()
            loaded = exp_svc.load_experiment(exp_id)
            assert loaded["training_configuration"]["seed"] == 77
            assert loaded["training_configuration"]["generator_config"]["seed"] == 77
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
        for eid in created_exp_ids:
            shutil.rmtree(cfg.get_path("checkpoints_dir") / "synthetic_experiments" / eid, ignore_errors=True)
            (cfg.get_path("experiments_results_dir") / f"{eid}.json").unlink(missing_ok=True)
            (cfg.get_path("experiments_configs_dir") / f"{eid}.yaml").unlink(missing_ok=True)


def test_controlled_csv_ingestion_and_safeguards():
    cfg = ConfigService()
    ing_svc = DatasetIngestionService(cfg)
    ds_slug = "pytest_ingested_csv"
    out_dir = cfg.get_path("data_synthetic_dir") / f"ingested_{ds_slug}"

    df = pd.DataFrame({
        "participant": ["S1", "S1", "S2", "S2", "S3", "S3"],
        "f_temp": [30.1, 30.2, 31.0, 31.1, 29.8, 29.9],
        "f_press": [100.0, 110.0, 120.0, 130.0, 95.0, 98.0],
        "latent_speed": [3.5, 3.5, 4.8, 4.8, 3.1, 3.1],
        "plantar_total_load_n": [200.0, 210.0, 220.0, 230.0, 195.0, 198.0],
    })

    try:
        # Reject target in features
        with pytest.raises(ValueError, match="Target leakage"):
            ing_svc.ingest_dataframe(
                df,
                IngestionSpec(ds_slug, ["f_temp", "latent_speed"], "latent_speed", "participant", {"temp": "C"}, "synthetic"),
            )

        # Reject derived pressure total for synthetic
        with pytest.raises(ValueError, match="derived pressure totals"):
            ing_svc.ingest_dataframe(
                df,
                IngestionSpec(ds_slug, ["f_temp", "f_press"], "plantar_total_load_n", "participant", {"temp": "C"}, "synthetic"),
            )

        # Reject < 3 sessions
        with pytest.raises(ValueError, match="At least 3 unique sessions"):
            ing_svc.ingest_dataframe(
                df.iloc[:4],
                IngestionSpec(ds_slug, ["f_temp", "f_press"], "latent_speed", "participant", {"temp": "C"}, "synthetic"),
            )

        # Valid ingestion
        manifest = ing_svc.ingest_dataframe(
            df,
            IngestionSpec(ds_slug, ["f_temp", "f_press"], "latent_speed", "participant", {"temperature": "celsius"}, "synthetic", True),
        )
        assert manifest["data_status"] == SYNTHETIC_PROVENANCE_TAG
        assert manifest["num_sessions"] == 3
        assert Path(manifest["csv_path"]).exists()
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def test_streamlit_training_and_datasets_pages_render_new_sections():
    for rel in ("app/pages/training.py", "app/pages/datasets.py"):
        at = AppTest.from_file(str(ML_ROOT / rel), default_timeout=30)
        at.run()
        assert not at.exception

