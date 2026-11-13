# FOOTGUARD AI — PHASE 2 FINAL VERIFICATION & ACCEPTANCE AUDIT

**Date:** 2026-11-13  
**Working Directory:** `c:\Users\Dhruva G Shankar\FootGuard AI\ML`  
**Scope:** Verification of Phase 2 services, test suite, model/dataset compatibility, synthetic safeguards, and training integrity prior to Phase 3 (Streamlit UI).  
**Status:** **AUDIT PASSED (52 / 52 Tests Passing, Exit Code 0)**

---

## 1. Test Verification & Exit Code Audit

### Final Verified Test Run
- **Command:** `python -m pytest tests/ -v`
- **Process Exit Code:** `0`
- **Duration:** `13.04s`
- **Total Collected / Passed:** **52 / 52** (`42` legacy ML tests + `10` Phase 2 service tests)

```text
tests/test_architectures.py (20 tests) .................... PASSED
tests/test_inference.py (14 tests) ........................ PASSED
tests/test_pressure_preprocessing.py (4 tests) ............ PASSED
tests/test_services.py (10 tests) ......................... PASSED
tests/test_thermal_preprocessing.py (4 tests) ............. PASSED
============================= 52 passed in 13.04s ==============================
EXIT_CODE=0
```

### Root-Cause Explanation of Earlier Phase 2 Exit Code `1` Runs
During Phase 2 development, two intermediate pytest runs exited with code `1` (each affecting only 3 newly added tests in `tests/test_services.py`, while all 42 legacy tests passed):
1. **First intermediate run:**
   - `DatasetService.get_real_sample()` resolved `file_path` relative to `ML/` instead of `ML/processed/` (where `thermodatabase_manifest.csv` stores `thermal/thermodatabase/CG001_M.npz`), causing `FileNotFoundError`.
   - `TrainingService._build_synthetic_datasets()` referenced `self.PRESSURE_COLS` instead of `SyntheticPressureSequenceDataset.PRESSURE_COLS`, causing `AttributeError`.
2. **Second intermediate run:**
   - `tests/test_services.py` looked up `arrays["thermal_tensor"]` instead of the actual `.npz` archive key `arrays["tensor"]`.
   - `TrainingService.start_training()` passed keyword arguments (`learning_rate=...`) to `build_optimizer_and_scheduler(model, config)` instead of a `TrainingConfig` instance, causing `TypeError`.

All four issues were resolved, and subsequent runs (including after adding the synthetic target-leakage block and protected checkpoint guard) exited cleanly with `EXIT_CODE=0`.

---

## 2. Model & Dataset Compatibility Audit

Inspected `services/validation_service.py`, `services/model_service.py`, and `services/inference_service.py`:

| Model ID | Expected Input Contract | Preprocessing Consistency | Compatible Dataset(s) | Blocked / Rejected Datasets |
|:---|:---|:---|:---|:---|
| `thermal_2d_cnn` | `(B, 3, 160, 80)` float32 `[Left, Right_flipped, Delta_T]` | `ThermalInferenceEngine` calls `resize_and_pad_thermal_matrix` from `thermal/preprocessing.py` (identical to training). | `thermodatabase` | `standup` (uncalibrated 8-bit PNGs, no participant IDs), `synthetic_paired_readings` (tabular summaries, not 2D matrices), `cad_walk`, `gait_lab` |
| `thermal_tcn` | `(B, 4, 1, H, W)` 4-timepoint Celsius TIFF sequence | Uses `IACTemporalDataset` directly. | `iac_tecmed` (exploratory $N=19$) | `standup`, `thermodatabase`, `synthetic_paired_readings` |
| `pressure_1d_cnn` | `(B, 1, 100)` normalized stance force curve (`/ 800.0 N`) | `PressureInferenceEngine` linearly resamples $\ge 10$-pt curves to 100 pts and scales by `800.0` (identical to `CADWalkTemporalDataset`). | `cad_walk` (healthy controls only) | `synthetic_paired_readings` (blocked: no non-derived target), `gait_lab` ($N=2$ confounded), `thermodatabase`, `standup` |
| `pressure_tcn` | `(B, 1, 100)` causal dilated sequence | Uses `CADWalkTemporalDataset`. | `cad_walk` (exploratory; underperformed in Phase 4/5) | `synthetic_paired_readings`, `gait_lab` |
| `pressure_ridge_baseline` | `[stance_duration_ms, peak_force_n]` | Matches `run_experiment_b.py` Ridge feature order. | `cad_walk` (healthy controls only) | All other datasets |

### Calibration & Invalid-Input Enforcement Verified
- **Thermal:** Rejects `.png`/`.jpg`/`.tiff`, `"standup"` paths, `uint8` arrays, values $>50.0^\circ\text{C}$, $>30\%$ NaN sensor dropout, `<50` plantar pixels above $15.0^\circ\text{C}$, and `<(30, 15)` matrices with explicit `ValueError`.
- **Pressure:** Rejects curves $<10$ points, flatline/zero signals ($<50.0\text{ N}$ peak or $\text{var} < 1.0$), non-positive duration/force, and blocks CNN execution on scalar-only inputs unless `allow_synthetic_demo=True` (DEMO mode).
- **Checkpoints:** Missing checkpoint paths raise `FileNotFoundError`; corrupted loads raise `RuntimeError`.

---

## 3. Synthetic Dataset Safeguards (`footguard_synthetic_paired_readings.csv`)

Audited `1,000,000` rows $\times$ `35` columns across `3,334` sessions (`SYN-0001`..`SYN-3334`, `258.9 MB`):

### Column & Target Verification
1. **Provenance:** Every row is tagged `data_status = 'SYNTHETIC_NOT_CLINICAL_GROUND_TRUTH'` (`sample_interval_s = 10`).
2. **Derived / Leaking Quantities Observed:**
   - `plantar_total_load_n` is an **exact algebraic sum** of the 6 site pressures (`heel + mid + mth1 + mth3 + mth5 + toe_hallux`, max rounding diff `0.02 N`).
   - `peak_site_pressure_n` is an **exact algebraic maximum** of those 6 site pressures (max diff `0.0 N`), and `peak_pressure_site` is the `argmax` site name.
   - `thermal_mean_c`, `thermal_min_c`, `thermal_max_c`, `thermal_std_c`, and the 3 zone means are synthetic summary statistics (mean diff between `thermal_mean_c` and mean of the 3 zone means is `0.117 °C`).
3. **Ground Truth Absence & Leakage Action Taken:**
   - The synthetic CSV contains **zero clinical or kinematic ground-truth labels** (no walking speed, no diabetes cohort label, no ulcer label).
   - **Action Applied:** Updated `ValidationService.COMPATIBILITY_MATRIX` to **block supervised training** on `synthetic_paired_readings` (`is_compatible=False`) because predicting `plantar_total_load_n` or `peak_site_pressure_n` from the 6 pressure sites constitutes 100% deterministic algebraic target leakage, and no non-derived target meaning is established.
   - `synthetic_paired_readings` remains enabled for **chunked dataset inspection, metadata auditing, session-level preview (`load_synthetic_session`), and strided sampling (`sample_synthetic_rows`)** with `SYNTHETIC_NOT_CLINICAL_GROUND_TRUTH` provenance preserved.
4. **Session-Level Isolation:**
   - `DatasetService.load_synthetic_session(session_id)` isolates single sessions via 50,000-row chunked streaming without loading the 258.9 MB file into memory.

---

## 4. Training Integrity Audit

Verified `services/training_service.py` and `services/experiment_service.py`:
- **Participant-Level Splits:** Real datasets (`ThermoDataset`, `IACTemporalDataset`, `CADWalkTemporalDataset`) load the immutable participant-isolated split JSONs in `ML/splits/` with zero train/val/test overlap.
- **Protected Phase 4/5 Checkpoints:** Added `PROTECTED_CHECKPOINT_DIRS` guard in `TrainingService.start_training()` so existing Phase 4/5 checkpoint directories (`experiment_a_thermal_2d_cnn`, `experiment_b_pressure_1d_cnn`, `experiment_b_pressure_tcn`, `experiment_c_thermal_tcn`, and `*_baseline`) cannot be overwritten even if explicitly passed as `experiment_id`.
- **Metadata Persistence:** Every run records `experiment_id`, `dataset_name`, `dataset_version`, `dataset_origin`, `model_name`, `model_version`, `hyperparameters`, `split_methodology`, `training_configuration` (including `seed`, `device`, `epochs_completed`), `evaluation_metrics`, `checkpoint_location`, and `timestamp` in JSON + YAML config + `.log`.
- **Blocked Configurations:** Incompatible pairings (`standup` + `thermal_2d_cnn`, `gait_lab` + `pressure_1d_cnn`, `synthetic_paired_readings` + `pressure_1d_cnn`) raise descriptive `ValueError("Training blocked: ...")` exceptions before any model or dataloader is built.

---

## 5. Remaining Limitations

1. `footguard_synthetic_paired_readings.csv` has no non-derived target labels; therefore it is restricted to dataset inspection/preview until a real or externally labeled synthetic target is defined.
2. `STANDUP` (uncalibrated PNGs, no participant IDs) and `Gait Lab` ($N=2$ participant confounding) remain permanently blocked from supervised training/inference.
3. Softmax outputs from `Thermal2DCNN` are uncalibrated raw probabilities (not clinical risk probabilities), and test cohorts remain small (`N=26` ThermoDataBase, `N=7` CAD_WALK, `N=19` IAC-TecMed).
4. Local execution targets CPU (`torch 2.12.1+cpu`).

---

## 6. Acceptance Criteria Checklist

| Acceptance Criterion | Status | Evidence |
|:---|:---|:---|
| Load a single synthetic session (`SYN-0001`) via chunked read | **PASS** | `test_dataset_service_load_single_synthetic_session` |
| Sample a subset of the 1M-row synthetic dataset without full load | **PASS** | `test_dataset_service_sample_synthetic_rows` |
| Inspect dataset metadata without loading full files | **PASS** | `test_dataset_service_metadata_without_full_load` |
| Load existing thermal and pressure checkpoints without retraining | **PASS** | `test_model_service_loads_existing_checkpoints` |
| Execute inference through both existing Thermal and Pressure engines | **PASS** | `test_inference_service_executes_existing_engines` |
| Save, reload, and export experiment metadata (JSON/YAML/MD) | **PASS** | `test_experiment_and_evaluation_services_roundtrip` |
| Block invalid/uncalibrated/target-leaking training & protect checkpoints | **PASS** | `test_validation_service_and_compatibility`, `test_training_service_synthetic_and_blocked_validation` |
| Preserve `FE/`, `BE/`, `DB/`, raw data, models, and Phase 4 checkpoints | **PASS** | Zero modifications outside `ML/services/`, `ML/configs/`, `ML/tests/`, `ML/PHASE2_*.md` |
| Complete 52-test suite passes with exit code `0` | **PASS** | `52 passed in 13.04s`, `EXIT_CODE=0` |

