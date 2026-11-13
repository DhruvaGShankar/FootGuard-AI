# FOOTGUARD AI — PHASE 1 AUDIT & MIGRATION ARCHITECTURE PLAN

**Date:** 2026-11-13  
**Scope:** Migration from multi-service web platform (`FE/` + `BE/` + `DB/` + `ML/`) to a **local-first Python ML research application using Streamlit** (`streamlit run app/main.py`).  
**Status:** Phase 1 Complete (Audit & Architecture Definition)

---

## 1. Repository Audit Summary

### A. Legacy Services (Marked Deprecated)
- `FE/` — React frontend (deprecated; preserved intact).
- `BE/` — FastAPI backend, auth middleware, storage routers (deprecated; preserved intact).
- `DB/` — PostgreSQL schemas and migrations (deprecated; preserved intact).
- Deprecation notice recorded in `../DEPRECATED_LEGACY_PLATFORM.md`.

### B. Existing ML Subsystem (`ML/`) — Verified Intact
All existing ML models, preprocessing, training, evaluation, checkpoints, and inference engines are verified and passing **42 / 42** pytest checks on CPU (`Python 3.14.3`, `torch 2.12.1+cpu`, `streamlit 1.58.0`, `plotly 6.9.0`, `PyYAML 6.0.3`):

| Component | Path | Status & Notes |
|:---|:---|:---|
| **Thermal 2D CNN** | `ML/thermal/cnn.py` | Reusable (`Thermal2DCNN`, 405,602 params, input `(B, 3, 160, 80)`). |
| **Thermal TCN** | `ML/thermal/tcn.py` | Reusable (`ThermalTCN`, 59,169 params, input `(B, 4, 1, H, W)`). Exploratory only ($N=19$). |
| **Pressure 1D CNN** | `ML/pressure/cnn.py` | Reusable (`Pressure1DCNN`, 43,873 params, configurable `in_channels`, `(B, C, L)`). |
| **Pressure TCN** | `ML/pressure/tcn.py` | Reusable (`PressureTCN`, 115,617 params, causal dilated TCN, configurable `in_channels`). |
| **Ridge Baseline** | `ML/checkpoints/experiment_b_pressure_1d_cnn/ridge_baseline.joblib` | Reusable biomechanical benchmark (`duration_ms`, `peak_force_n` $\rightarrow$ walking speed). |
| **Preprocessing** | `ML/thermal/preprocessing.py`, `ML/pressure/preprocessing.py` | Reusable. Preserves `(160, 80)` thermal standardization, `(48, 32)` CAD_WALK padding, 100-pt stance resampling, and `FixJumpv24` detrending. |
| **PyTorch Datasets & Splits** | `ML/thermal/dataset.py`, `ML/pressure/dataset.py`, `ML/splits/` | Reusable participant-isolated splits (`ThermoDataBase`, `IAC-TecMed`, `CAD_WALK`, `Gait Lab`, `STANDUP`). |
| **Training & Tracking** | `ML/common/trainer.py`, `ML/thermal/train.py`, `ML/pressure/train.py`, `ML/experiments/run_experiment_*.py` | Reusable (`TrainingConfig`, `EarlyStopping`, `ModelCheckpointManager`, `ExperimentTracker`, `set_seed`). |
| **Inference Engines** | `ML/thermal/inference.py`, `ML/pressure/inference.py` | Reusable (`ThermalInferenceEngine`, `PressureInferenceEngine`) with calibration, dropout, and DEMO-mode safety guards intact. |
| **Evaluation & Metrics** | `ML/common/metrics.py`, `ML/evaluation/phase5_analysis_results.json` | Reusable (`compute_classification_metrics`, `compute_regression_metrics`). |
| **Saved Checkpoints** | `ML/checkpoints/` | Intact: `experiment_a_thermal_2d_cnn/best_model.pt`, `experiment_b_pressure_1d_cnn/best_model.pt`, `ridge_baseline.joblib`, `experiment_b_pressure_tcn/best_model.pt`, `experiment_c_thermal_tcn/best_model.pt`. |

### C. Synthetic Dataset Discovered in Repository
- **File:** `ML/footguard_synthetic_paired_readings.csv` (258.9 MB)
- **Dimensions:** `1,000,000` rows $\times$ `35` columns
- **Sessions:** `3,334` synthetic sessions (`SYN-0001` .. `SYN-3334`)
- **Provenance Column:** `data_status == 'SYNTHETIC_NOT_CLINICAL_GROUND_TRUTH'`
- **Modalities Present:**
  - Thermal summary & zone metrics (`mlx90640_frame_width_px=32`, `height=24`, `thermal_mean_c`, `thermal_min_c`, `thermal_max_c`, `thermal_std_c`, 6 site means, 3 zone means)
  - Plantar pressure (`heel_pressure_n`, `mid_pressure_n`, `mth1_pressure_n`, `mth3_pressure_n`, `mth5_pressure_n`, `toe_hallux_pressure_n`, `plantar_total_load_n`, `peak_site_pressure_n`, `peak_pressure_site`)
  - IMU (`mpu6050_accel_x/y/z_g`, `mpu6050_gyro_x/y/z_dps`)
- **Engineering Implication:** At 258.9 MB / 1M rows, the dataset loader service must support chunked/sampled reads and session-level slicing (`session_id`) to keep the local Streamlit UI responsive while preserving `SYNTHETIC_NOT_CLINICAL_GROUND_TRUTH` provenance.

---

## 2. Target Local-First Architecture

Running from `FootGuard AI/ML/` (or repository root):
```text
streamlit run app/main.py
```

### Directory Layout (Additive & Non-Destructive)
```text
ML/
├── app/                              # Streamlit UI (5 sections, zero ML/training logic)
│   ├── main.py                       # Entry point & navigation
│   └── pages/                        # Home, Datasets, Models & Training, Prediction, Experiments
├── services/                         # Reusable Python application services
│   ├── __init__.py
│   ├── config_service.py             # YAML configuration loader
│   ├── dataset_service.py            # Real + Synthetic dataset loading, preview, session/participant splits
│   ├── validation_service.py         # Shape, NaN, calibration, label, & model-compatibility checks
│   ├── model_service.py              # Registry of PyTorch/Ridge models & checkpoints
│   ├── training_service.py           # Wraps existing trainers with progress callbacks
│   ├── inference_service.py          # Wraps ThermalInferenceEngine & PressureInferenceEngine
│   ├── evaluation_service.py         # Confusion matrices, ROC/PR, regression plots, comparisons
│   └── experiment_service.py         # JSON experiment configs, results, logs, & export
├── configs/
│   └── default.yaml                  # Central YAML configuration
├── data/
│   ├── synthetic/                    # Symlink/reference or synthetic samples (points to synthetic CSV)
│   ├── real/                         # Reference to ML/raw/ (read-only, immutable)
│   └── processed/                    # Reference to ML/processed/
├── experiments/
│   ├── configs/                      # YAML/JSON run configs
│   ├── results/                      # Standardized JSON experiment records
│   └── logs/                         # Training & run logs
├── checkpoints/                      # Existing & newly trained checkpoints (preserved intact)
├── common/, thermal/, pressure/      # Existing core ML modules (preserved intact)
├── raw/, processed/, splits/         # Existing datasets & manifests (preserved intact)
└── tests/                            # Existing 42 tests + new service & smoke tests
```

---

## 3. Python Service Interfaces (Phase 2 Contract)

1. **`DatasetService`**:
   - `list_datasets() -> List[DatasetDescriptor]` (distinguishes `real` vs `synthetic`, provenance, subject/session counts).
   - `load_preview(dataset_id, limit=500, session_id=None) -> pd.DataFrame` (chunked safe reads for 1M-row CSV).
   - `get_sample(dataset_id, index_or_id) -> Dict[str, Any]` (returns thermal matrices or pressure curves + metadata).
   - `get_split_summary(dataset_id) -> Dict[str, Any]` (participant/session-level split counts).
2. **`ValidationService`**:
   - `validate_dataset(dataset_id, df_or_sample) -> ValidationReport` (shape, NaNs, calibration, labels, provenance).
   - `check_model_compatibility(dataset_id, model_id) -> CompatibilityResult` (prevents mixing incompatible or uncalibrated inputs such as STANDUP PNGs with Celsius CNNs).
3. **`ModelService`**:
   - `list_models() -> List[ModelSpec]` (`thermal_2d_cnn`, `thermal_tcn`, `pressure_1d_cnn`, `pressure_tcn`, `pressure_ridge_baseline`).
   - `list_checkpoints(model_id=None) -> List[CheckpointInfo]`.
   - `load_model(model_id, checkpoint_path=None)`.
4. **`TrainingService`**:
   - `start_training(model_id, dataset_id, hyperparams, progress_callback=None) -> ExperimentResult` (delegates to existing PyTorch trainers/Ridge fitter, saves checkpoint & JSON experiment record).
5. **`InferenceService`**:
   - Delegates directly to `ThermalInferenceEngine` and `PressureInferenceEngine` (zero duplicated inference logic).
6. **`EvaluationService` & `ExperimentService`**:
   - `list_experiments() -> List[Dict]` (reads both legacy Phase 4/5 JSONs and new `experiments/results/*.json`).
   - `save_experiment(record: Dict) -> Path` (stores dataset name/version/origin, model name/version, hyperparams, split methodology, training config, metrics, checkpoint path, timestamp).
   - `export_report(experiment_id, fmt="json"|"md") -> str`.

---

## 4. Research Integrity Guardrails

- **No clinical claims:** Never claim DFU detection, future ulcer prediction, neuropathy diagnosis, healing-time prediction, or clinical validity.
- **Synthetic vs. Real separation:** Synthetic data (`footguard_synthetic_paired_readings.csv`) is always tagged `SYNTHETIC_NOT_CLINICAL_GROUND_TRUTH` and never mixed silently with real clinical cohorts (`ThermoDataBase`, `CAD_WALK`, `IAC-TecMed`).
- **Uncalibrated softmax:** Softmax outputs are reported strictly as uncalibrated cohort scores, never as clinical risk probabilities.
- **Raw data immutability:** `ML/raw/` and `ML/footguard_synthetic_paired_readings.csv` remain strictly read-only.

---

## 5. Phased Execution Roadmap

1. **Phase 1 (Current):** Repository audit, deprecation notice, architecture blueprint, verification of existing 42 ML tests.
2. **Phase 2:** YAML config, storage directories, Python application services (`services/`), and unit tests (`tests/test_services.py`).
3. **Phase 3:** 5-section Streamlit application (`app/main.py` + `app/pages/`), `requirements.txt`, and `README.md`.
4. **Phase 4:** End-to-end automated tests, local Streamlit smoke test (`streamlit run app/main.py`), and `MIGRATION_REPORT.md`.

