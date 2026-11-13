# FOOTGUARD AI — PHASE 2 SERVICE LAYER REPORT

**Date:** 2026-11-13  
**Scope:** Local Python Application Services & Storage Layout (`ML/` only; zero Streamlit UI code yet)  
**Test Status:** **52 / 52 Tests Passing** (`42` existing ML tests + `10` new service unit/integration tests)

---

## 1. Architecture Decisions

1. **Zero HTTP / Database Dependencies:**
   - All services in `ML/services/` are pure Python classes invoked directly in-process, with zero FastAPI, PostgreSQL, or MongoDB dependencies.
   - Configuration is YAML-based (`ML/configs/default.yaml`); experiment metadata is stored as local JSON (`ML/experiments/results/*.json`) with YAML run configs (`ML/experiments/configs/*.yaml`) and plain-text training logs (`ML/experiments/logs/*.log`).
2. **Memory-Safe Chunked Synthetic Loader (`258.9 MB` / `1,000,000` Rows):**
   - `DatasetService` inspects metadata (`inspect_metadata`) using `nrows=5` and `usecols=["session_id"]` chunked reads without loading the 1M-row `footguard_synthetic_paired_readings.csv` into RAM.
   - `load_synthetic_session(session_id)` streams 50,000-row chunks and exits early once the contiguous session block finishes.
   - `sample_synthetic_rows(n_rows, seed)` uses deterministic strided chunk sampling.
3. **Strict Real vs. Synthetic Provenance Separation:**
   - Every `DatasetDescriptor`, `ValidationReport`, inference response, and experiment JSON explicitly tags `origin` (`"real"` vs. `"synthetic"`) and `provenance_label` (`"CLINICAL_RESEARCH_DATASET"` vs. `"SYNTHETIC_NOT_CLINICAL_GROUND_TRUTH"`).
   - `ValidationService.check_model_compatibility()` blocks uncalibrated `STANDUP` PNGs from `thermal_2d_cnn`, blocks confounded `gait_lab` ($N=2$) from supervised classification, and prevents mixing synthetic paired readings with `thermal_2d_cnn`.
4. **Zero Duplication of ML Logic:**
   - `InferenceService` wraps and caches `ThermalInferenceEngine` (`thermal/inference.py`) and `PressureInferenceEngine` (`pressure/inference.py`) directly.
   - `ModelService` loads existing `.pt` and `.joblib` checkpoints directly without retraining.
   - `TrainingService` reuses `ThermoDataset`, `IACTemporalDataset`, `CADWalkTemporalDataset`, `TrainingConfig`, `EarlyStopping`, and `build_optimizer_and_scheduler`, while saving new runs to unique `checkpoints/<experiment_id>/` directories so Phase 4 baseline checkpoints remain untouched.

---

## 2. Files Created

| File | Purpose |
|:---|:---|
| `ML/configs/default.yaml` | Central YAML configuration for paths, training defaults, chunked synthetic loader, and research disclaimers. |
| `ML/services/__init__.py` | Services package marker. |
| `ML/services/config_service.py` | YAML config loader and local storage directory initializer (`data/`, `experiments/configs/`, `experiments/results/`, `experiments/logs/`). |
| `ML/services/dataset_service.py` | Real + synthetic dataset registry, chunked session/sample loading, metadata inspection, and split summaries. |
| `ML/services/validation_service.py` | Shape, NaN, label, provenance, calibration, and model-dataset compatibility validation. |
| `ML/services/model_service.py` | Model specification registry, checkpoint discovery, metadata inspection, and `.pt`/`.joblib` loading. |
| `ML/services/inference_service.py` | Cached facade over `ThermalInferenceEngine` and `PressureInferenceEngine`. |
| `ML/services/evaluation_service.py` | Classification/regression metrics wrapper and experiment comparison table builder. |
| `ML/services/experiment_service.py` | JSON/YAML experiment persistence, legacy Phase 4/5 loader, and JSON/Markdown report export. |
| `ML/services/training_service.py` | Reproducible training runner for real datasets and session-isolated synthetic pressure sequences with progress callbacks. |
| `ML/tests/test_services.py` | 10 unit/integration tests covering every service and Phase 2 acceptance criterion. |
| `ML/PHASE2_SERVICE_REPORT.md` | This Phase 2 report. |

---

## 3. Acceptance Criteria Verification

All Phase 2 acceptance criteria were verified programmatically in `tests/test_services.py`:

1. **Load a single synthetic session:** `test_dataset_service_load_single_synthetic_session` loaded `SYN-0001` via chunked reads with `SYNTHETIC_NOT_CLINICAL_GROUND_TRUTH` provenance.
2. **Sample a subset of the million-row dataset:** `test_dataset_service_sample_synthetic_rows` sampled 150 rows without loading the 258.9 MB file into memory.
3. **Inspect dataset metadata without loading everything:** `test_dataset_service_metadata_without_full_load` inspected all 6 datasets, 35 synthetic columns, and sample session IDs.
4. **Load existing thermal and pressure checkpoints:** `test_model_service_loads_existing_checkpoints` loaded `thermal_2d_cnn`, `pressure_1d_cnn`, and `pressure_ridge_baseline` checkpoints without retraining, and verified `FileNotFoundError` on missing checkpoints.
5. **Execute one inference through each existing engine:** `test_inference_service_executes_existing_engines` ran real `ThermoDataBase` inference via `ThermalInferenceEngine` and real `CAD_WALK` inference via `PressureInferenceEngine`.
6. **Save and reload experiment metadata:** `test_experiment_and_evaluation_services_roundtrip` saved JSON + YAML experiment records, reloaded them, and exported Markdown reports, while `test_training_service_synthetic_and_blocked_validation` trained a 2-epoch synthetic `pressure_1d_cnn` run, reloaded the saved checkpoint, and verified blocked training for `standup` + `thermal_2d_cnn`.

---

## 4. Automated Test Results

Command executed:
```bash
python -m pytest tests/ -v
```

Result: **52 passed in 25.16s**
- `tests/test_architectures.py`: 20 / 20 passed
- `tests/test_inference.py`: 14 / 14 passed
- `tests/test_pressure_preprocessing.py`: 4 / 4 passed
- `tests/test_thermal_preprocessing.py`: 4 / 4 passed
- `tests/test_services.py`: 10 / 10 passed

---

## 5. Known Limitations

1. `footguard_synthetic_paired_readings.csv` contains summary thermal metrics (`thermal_mean_c`, zone means) rather than raw `(160, 80)` bilateral plantar matrices; therefore it cannot be fed into `Thermal2DCNN` and is restricted to multi-channel pressure sequence experiments (`in_channels=6`) tagged `SYNTHETIC_NOT_CLINICAL_GROUND_TRUTH`.
2. `STANDUP` (uncalibrated PNGs, no participant IDs) and `Gait Lab` ($N=2$ participant confounding) remain blocked from supervised training/inference.
3. All training and inference currently target CPU (`torch 2.12.1+cpu`).

---

## Next Step: Phase 3
Upon approval, I will implement the 5-section Streamlit application (`ML/app/main.py` and `ML/app/pages/`), update `ML/requirements.txt`, and update `ML/README.md` with local launch instructions (`streamlit run app/main.py`).

