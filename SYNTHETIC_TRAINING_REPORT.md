# FootGuard AI — Synthetic Dataset Training & CSV Ingestion Report

**Status:** **COMPLETE & VERIFIED**  
**Test Suite:** `python -m pytest tests/ -v` -> **67 passed in 11.76s** (`EXIT_CODE=0`)

---

## 1. Existing Dataset Audit (`footguard_synthetic_paired_readings.csv`)

Chunked audit across all **1,000,000 rows** (35 columns, **3,334 unique `session_id`** values, **0 missing values**, `data_status = 'SYNTHETIC_NOT_CLINICAL_GROUND_TRUTH'`):

- **Thermal features (13):** `thermal_mean_c` (28.726–32.983 °C, mean 30.928), `thermal_min_c`, `thermal_max_c`, `thermal_std_c`, `heel_temp_mean_c`, `mid_temp_mean_c`, `mth1_temp_mean_c`, `mth3_temp_mean_c`, `mth5_temp_mean_c`, `toe_hallux_temp_mean_c`, `thermal_heel_zone_mean_c`, `thermal_midfoot_zone_mean_c`, `thermal_forefoot_zone_mean_c`.
- **Six-site pressure features (6):** `heel_pressure_n` (mean 99.600 N), `mid_pressure_n` (47.776 N), `mth1_pressure_n` (71.646 N), `mth3_pressure_n` (67.739 N), `mth5_pressure_n` (47.878 N), `toe_hallux_pressure_n` (63.866 N).
- **IMU features (6):** `mpu6050_accel_x_g`, `mpu6050_accel_y_g`, `mpu6050_accel_z_g`, `mpu6050_gyro_x_dps`, `mpu6050_gyro_y_dps`, `mpu6050_gyro_z_dps`.
- **Identifiers & metadata (7):** `reading_id`, `timestamp_utc`, `data_status`, `session_id`, `sample_interval_s` (10 s), `mlx90640_frame_width_px` (32), `mlx90640_frame_height_px` (24).
- **Derived columns (3):** `plantar_total_load_n` (= sum of 6 sites), `peak_site_pressure_n` (= max of 6 sites), `peak_pressure_site`.
- **Safeguard Preserved:** `ValidationService.COMPATIBILITY_MATRIX` continues to block `synthetic_paired_readings` from supervised training.

---

## 2. Synthetic Data Generation (`services/synthetic_generator_service.py`)

`SyntheticGeneratorService` generates controlled multimodal datasets under `ML/data/synthetic/<dataset_name>/` (`readings.csv`, `tensors.npz`, `manifest.json`), with every row tagged `SYNTHETIC_NOT_CLINICAL_GROUND_TRUTH`.

- **Configurable parameters (`SyntheticGenerationConfig`):** `num_sessions`, `steps_per_session`, `sampling_hz`, `thermal_base_c`, `thermal_spread_c`, `pressure_load_scale_n`, `sensor_noise_std`, `measurement_variation`, `seed`.
- **Latent-State Supervised Targets (Non-Algebraic):**
  - `latent_thermal_regime` (`0` = symmetric baseline, `1` = simulated localized forefoot thermal asymmetry): drives spatial hotspot injection in `(3, 160, 80)` bilateral thermal tensors and subtle site temperature offsets.
  - `latent_gait_speed_kmh` (continuous, ~2.2–6.8 km/h): drives stance cadence, bimodal vGRF curve shape, and IMU accelerometer/gyroscope amplitudes with independent noise.

---

## 3. Synthetic Experimental Training Pathway (`services/synthetic_training_service.py`)

- **Strict Isolation:**
  - Accepts only datasets tagged `SYNTHETIC_NOT_CLINICAL_GROUND_TRUTH`; never mixes real and synthetic data.
  - Blocks `plantar_total_load_n` and `peak_site_pressure_n` from being used as supervised targets, and blocks target-in-features leakage.
  - Splits strictly by `session_id` (`split_sessions_disjoint`), asserting empty intersections across `train_sessions`, `val_sessions`, and `test_sessions`.
  - Saves checkpoints exclusively under `ML/checkpoints/synthetic_experiments/<experiment_id>/`, leaving Phase 4/5 checkpoints untouched.
  - Records `generator_config`, `seed`, `feature_columns`, `target_column`, and `split_methodology` via `ExperimentService`.

### Model Compatibility
| Model ID | Modality | Input Contract | Target |
| --- | --- | --- | --- |
| `thermal_2d_cnn` | Thermal-only | `(B, 3, 160, 80)` bilateral maps | `latent_thermal_regime` |
| `pressure_1d_cnn` | Pressure-only | `(B, 6, 16)` 6-site windows | `latent_gait_speed_kmh` |
| `pressure_tcn` | Pressure-only | `(B, 6, 16)` causal 6-site windows | `latent_gait_speed_kmh` |
| `pressure_ridge_baseline` | Pressure-only | Selected tabular pressure features | `latent_gait_speed_kmh` |
| `multimodal_fusion_mlp` | Multimodal (or subset) | `(B, F)` thermal + pressure + IMU | `latent_gait_speed_kmh` / `latent_thermal_regime` |

---

## 4. Controlled CSV Ingestion & Streamlit Integration

- **CSV Ingestion (`services/ingestion_service.py` + `app/pages/datasets.py`):** Users can upload a CSV, preview rows, select input features, target, session/participant ID column, units, and provenance (`synthetic` or `real`), validate against leakage and derived-target rules, and save an isolated local copy under `ML/data/synthetic/ingested_<slug>/` or `ML/data/real/ingested_<slug>/` without modifying the original file.
- **Training UI (`app/pages/training.py`):** Added the **Synthetic Data Experiments** tab alongside **Real Dataset Training**, enabling dataset generation, sample/schema inspection, model & feature selection, training execution, and loss/metric visualization.

---

## 5. Tests & Limitations

- **Tests Added (`tests/test_synthetic_training_and_ingestion.py`):**
  1. `test_existing_synthetic_paired_readings_still_blocked`
  2. `test_synthetic_generation_and_target_semantics`
  3. `test_synthetic_training_session_isolation_leakage_and_checkpoint_isolation`
  4. `test_controlled_csv_ingestion_and_safeguards`
  5. `test_streamlit_training_and_datasets_pages_render_new_sections`
- **Limitations:** Synthetic datasets and models are strictly for methodological and pipeline experimentation (`SYNTHETIC_NOT_CLINICAL_GROUND_TRUTH`); they carry zero clinical validity and must not be used for ulcer detection, neuropathy diagnosis, or clinical risk prediction.

