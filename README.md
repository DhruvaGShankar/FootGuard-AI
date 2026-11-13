# FootGuard AI — Local ML Research Studio (`ML/`)

**Subsystem:** FootGuard AI Local-First Python ML Research Studio  
**Version:** 2.0.0 (Local-First Streamlit Architecture)  
**Hardware Target:** CPU (x86_64 / ARM64) / CUDA optional  
**Stack:** Streamlit, PyTorch 2.x, NumPy, Pandas, Plotly, Scikit-Learn, Pydantic v2, PyYAML

---

## 1. Quick Start (Local Launch)

No external backend, React frontend, database, or Docker container is required. Everything runs locally in Python:

```bash
cd ML
pip install -r requirements.txt
python -m streamlit run app/main.py
```

> **Note:** On Windows, `python -m streamlit run app/main.py` works even if Python's user `Scripts/` directory is not on your `PATH`. If `streamlit` is on your `PATH`, `streamlit run app/main.py` also works.

---

## 2. Executive Summary & Scientific Safeguards

`FootGuard AI/ML` is a standalone, local-first ML research studio for **plantar thermography analysis**, **ground reaction force (vGRF) kinetic modeling**, and **controlled synthetic multimodal experimentation**.

All workflows enforce strict research-integrity guardrails:
- **Research-Only Output (Zero Clinical Claims):** No raw dataset contains diabetic foot ulcer (DFU) bounding boxes, clinical ulcer staging, neuropathy diagnoses, or wound healing timelines. Models must **not** be used for clinical diagnosis, ulcer detection, or risk prediction.
- **Strict Participant / Session Isolation:** Train, validation, and test splits are strictly disjoint at the participant or session level (`assert not (train & val)`, `assert not (train & test)`).
- **Real vs. Synthetic Provenance Separation:** Real datasets and synthetic datasets are never silently mixed. Synthetic records are explicitly tagged `SYNTHETIC_NOT_CLINICAL_GROUND_TRUTH`.
- **Derived-Target Leakage Block:** `footguard_synthetic_paired_readings.csv` (`plantar_total_load_n` = sum of 6 sites, `peak_site_pressure_n` = max of 6 sites) is permanently blocked from supervised training.
- **Checkpoint Isolation:** Phase 4/5 baseline checkpoints are protected from overwriting; new local runs save to `checkpoints/local_*` and synthetic runs save exclusively to `checkpoints/synthetic_experiments/`.
- **Calibration Enforcement:** Uncalibrated 8-bit PNGs (e.g., `STANDUP`) are blocked from Celsius thermal calculations and `Thermal2DCNN` inference.

---

## 3. Directory Structure

```text
ML/
├── app/                                # Local-first Streamlit Research Studio
│   ├── main.py                         # Navigation entrypoint (st.navigation)
│   ├── state.py                        # Cached read-only service registry
│   ├── components/                     # Reusable UI components
│   │   ├── metric_cards.py             # KPI cards, disclaimers, provenance badges
│   │   ├── pressure_viewer.py          # Interactive Plotly vGRF & synthetic signal plots
│   │   └── thermal_viewer.py           # Bilateral foot thermogram & ΔT heatmaps
│   └── pages/                          # 5 modular application pages
│       ├── home.py                     # Workspace overview, datasets, checkpoints, Phase 5 summary
│       ├── datasets.py                 # Real & synthetic explorer + Controlled CSV Ingestion
│       ├── training.py                 # Real Dataset Training + Synthetic Data Experiments
│       ├── prediction.py               # Local Thermal & Pressure inference UI
│       └── experiments.py              # Experiment history, comparison, JSON/MD exports
│
├── services/                           # Reusable Python business-logic services
│   ├── config_service.py               # YAML config & path resolution
│   ├── dataset_service.py              # Real dataset loaders + chunked 258.9 MB CSV reader
│   ├── validation_service.py           # Compatibility matrix, schema & calibration guards
│   ├── model_service.py                # Model specs, checkpoint discovery & loading
│   ├── inference_service.py            # Wrapper around Thermal & Pressure inference engines
│   ├── evaluation_service.py           # Comparison tables & Phase 5 metrics
│   ├── experiment_service.py           # Local JSON/YAML persistence & Markdown/JSON export
│   ├── training_service.py             # Real dataset PyTorch & Ridge training pipeline
│   ├── synthetic_generator_service.py  # Latent-state multimodal synthetic data generator
│   ├── synthetic_training_service.py   # Session-isolated synthetic training + Multimodal MLP
│   └── ingestion_service.py            # Controlled CSV validation & local ingestion
│
├── configs/
│   └── default.yaml                    # Paths, training defaults, chunk sizes, disclaimer
│
├── common/                             # Core ML utilities & Pydantic contracts
│   ├── config.py                       # Canonical shapes, thresholds, random seed
│   ├── logging.py                      # Structured logger
│   ├── metrics.py                      # Classification & regression metrics
│   ├── reproducibility.py              # Deterministic seed manager
│   ├── schemas.py                      # Pydantic v2 request/response contracts
│   └── trainer.py                      # EarlyStopping, CheckpointManager, Tracker
│
├── thermal/                            # Plantar thermography models & preprocessing
│   ├── cnn.py                          # Thermal2DCNN: (B, 3, 160, 80) -> (B, 2)
│   ├── tcn.py                          # ThermalTCN: (B, 4, 1, H, W) -> (B, 1)
│   ├── dataset.py                      # ThermoDataset, IACTemporalDataset, STANDUPDataset
│   ├── inference.py                    # ThermalInferenceEngine
│   └── preprocessing.py                # Canonical resize, flip, ΔT computation
│
├── pressure/                           # Plantar pressure & gait kinetics models
│   ├── cnn.py                          # Pressure1DCNN: (B, C, L) -> (B, 1)
│   ├── tcn.py                          # PressureTCN: (B, C, L) -> (B, 1)
│   ├── dataset.py                      # CADWalkSpatialDataset, CADWalkTemporalDataset, GaitLabDataset
│   ├── inference.py                    # PressureInferenceEngine (Curve, Ridge scalar, opt-in DEMO)
│   └── preprocessing.py                # FixJumpv24 detrending, stance segmentation, resampling
│
├── checkpoints/                        # Local model checkpoints (.pt, .joblib)
│   ├── experiment_a_thermal_2d_cnn/    # Protected Phase 4 Thermal 2D CNN
│   ├── experiment_b_pressure_1d_cnn/   # Protected Phase 4 Pressure 1D CNN & Ridge Baseline
│   ├── experiment_b_pressure_tcn/      # Protected Phase 4 Pressure TCN
│   ├── experiment_c_thermal_tcn/       # Protected Phase 4 Thermal TCN
│   └── synthetic_experiments/          # Isolated checkpoints from synthetic training runs
│
├── data/                               # Local generated & ingested datasets
│   ├── real/                           # Ingested real CSVs (isolated copies)
│   └── synthetic/                      # Generated & ingested synthetic datasets
│
├── experiments/                        # Phase 4 scripts, saved results & configs
│   ├── configs/                        # Per-run YAML snapshots
│   ├── results/                        # Per-run JSON experiment records
│   └── logs/                           # Training logs
│
├── raw/                                # READ-ONLY Raw Datasets (never modified)
├── processed/                          # Preprocessed cached tensors (.npz) & manifests
├── splits/                             # Participant-isolated JSON split manifests
├── evaluation/                         # Phase 5 error analysis & bootstrap results
├── tests/                              # Pytest suite (67 tests across 6 test files)
├── footguard_synthetic_paired_readings.csv  # 258.9 MB (1,000,000-row) paired synthetic CSV
└── requirements.txt                    # Python dependencies
```

---

## 4. Streamlit Research Studio Pages

1. **Home (`app/pages/home.py`)**: Displays workspace status, available real and synthetic datasets, discovered checkpoints, and Phase 4/5 benchmark highlights.
2. **Datasets & Provenance Explorer (`app/pages/datasets.py`)**:
   - Inspects metadata and split summaries without loading full datasets into memory.
   - Uses chunked reads (`chunksize=50000`) for `footguard_synthetic_paired_readings.csv` (single session `SYN-0001`..`SYN-3334` or strided sampling).
   - Displays calibration warnings (`STANDUP` uncalibrated) and compatibility checks.
   - **Controlled CSV Dataset Ingestion**: Upload a CSV, preview rows, map features/target/session ID, define units and provenance (`synthetic` or `real`), validate against leakage/derived-target rules, and save a clean local copy under `data/synthetic/` or `data/real/` without modifying the original.
3. **Models & Controlled Training (`app/pages/training.py`)**:
   - **Real Dataset Training** tab: Enforces `ValidationService` compatibility, runs reproducible CPU/CUDA training, logs live epoch loss curves, and saves new checkpoints (`checkpoints/local_*`) and experiment metadata.
   - **Synthetic Data Experiments** tab: Generates latent-state multimodal datasets via `SyntheticGeneratorService`, inspects generated samples/features/labels, trains compatible models (`thermal_2d_cnn`, `pressure_1d_cnn`, `pressure_tcn`, `pressure_ridge_baseline`, `multimodal_fusion_mlp`), and isolates checkpoints in `checkpoints/synthetic_experiments/`.
4. **Local Prediction & Inference (`app/pages/prediction.py`)**:
   - **Thermal Cohort Screening**: Runs `ThermalInferenceEngine` on `ThermoDataBase` subjects or uploaded calibrated Celsius `.csv` pairs; renders bilateral heatmaps, `|L - R|` asymmetry, `ΔT > 2.2°C` area fraction, SHA-256 input hash, and uncalibrated softmax warnings.
   - **Pressure & Gait Speed Inference**: Runs `PressureInferenceEngine` on `CAD_WALK` samples, uploaded `.csv`/`.npy` force curves, scalar Ridge-only inputs, or explicitly opt-in `DEMO` synthetic curves.
5. **Experiments & Evaluation Results (`app/pages/experiments.py`)**: Lists Phase 4/5 baselines and new local/synthetic runs, filters by origin (`All`, `real`, `synthetic`), renders comparison tables, confusion matrices, and loss curves, and exports JSON or Markdown reports.

---

## 5. Datasets & Provenance

| Dataset ID | Origin | Modality | Samples / Size | Ground Truth & Limitations |
| :--- | :--- | :--- | :--- | :--- |
| `thermodatabase` | Real | Plantar Thermography (CSV) | 167 subjects (45 Control, 122 Diabetic) | Type-2 diabetes cohort stratification only; no ulcer labels. Calibrated °C `(3, 160, 80)`. |
| `iac_tecmed` | Real | Dynamic Cold-Rewarming (TIFF) | 19 subjects (4 timepoints: T0–T15) | Cold-challenge rewarming curve; small sample size ($N=19$, exploratory only). |
| `standup` | Real | Plantar Screening (PNG) | 415 images | Uncalibrated 8-bit pseudo-color PNGs; lacks reliable participant IDs; Celsius & training blocked. |
| `cad_walk` | Real | Dynamic Insole Pressure (Mat/NIfTI) | 51 healthy controls | Treadmill walking speed (km/h); healthy controls only. |
| `gait_lab` | Real | 20-sensor Multi-Axial Insole | 2 subjects (1 Diabetic, 1 Control) | 18-channel `FixJumpv24` streams; confounded by $N=2$ (inspection only). |
| `synthetic_paired_readings` | Synthetic | Tabular Thermal + 6-site Pressure + IMU | 1,000,000 rows, 3,334 sessions (258.9 MB) | `SYNTHETIC_NOT_CLINICAL_GROUND_TRUTH`. Blocked from supervised training due to algebraic derived columns (`plantar_total_load_n`, `peak_site_pressure_n`). |
| Generated / Ingested (`data/synthetic/`) | Synthetic | Tabular + `(3, 160, 80)` Thermal + `(100,)` vGRF | Configurable | `SYNTHETIC_NOT_CLINICAL_GROUND_TRUTH`. Latent simulator targets (`latent_thermal_regime`, `latent_gait_speed_kmh`). |

---

## 6. Model Registry & Phase 4/5 Benchmarks

| Model ID | Modality & Task | Phase 4/5 Benchmark | Registry Status |
| :--- | :--- | :--- | :--- |
| `thermal_2d_cnn` | Plantar Thermograms (Classification) | Test AUROC `0.812`, F1 `0.829` | Research Candidate |
| `pressure_1d_cnn` | 100-pt vGRF Curves (Regression) | Test MAE `0.431 km/h`, $R^2$ `0.113` | Research Candidate (Slow Gait) |
| `pressure_ridge_baseline` | Stance Duration & Peak Force (Regression) | Test MAE `0.259 km/h`, $R^2$ `0.705` | Primary Benchmark |
| `thermal_tcn` | 4-Frame Rewarming (Regression) | Test MAE `0.741 °C` | Exploratory Only (Underfit on $N=13$) |
| `pressure_tcn` | Causal Stance Sequence (Regression) | Test MAE `1.216 km/h` | Exploratory Only (Causal Mask Mismatch) |
| `multimodal_fusion_mlp` | Paired Thermal + Pressure + IMU | Synthetic experiments only | Experimental Synthetic Only |

---

## 7. Python Inference Quick-Start

### Thermal Cohort Screening Engine
```python
from thermal.inference import ThermalInferenceEngine

engine = ThermalInferenceEngine(device="cpu", decision_threshold=0.50)
response = engine.predict(("path/to/left_foot.csv", "path/to/right_foot.csv"))

print("Predicted Cohort:", response.prediction.predicted_label)
print("Raw Softmax Score:", response.prediction.confidence)
print("Mean |L - R| Asymmetry:", response.bilateral_metrics.mean_abs_asymmetry_c, "°C")
print("Latency:", response.latency_ms, "ms")
```

### Pressure & Gait Dynamics Engine
```python
import numpy as np
from pressure.inference import PressureInferenceEngine

engine = PressureInferenceEngine(device="cpu")
force_curve = np.sin(np.linspace(0, np.pi, 120)) * 820.0
response = engine.predict(force_curve)

print("CNN Speed:", response.prediction.predicted_walking_speed_kmh, "km/h")
print("Ridge Baseline Speed:", response.prediction.baseline_ridge_speed_kmh, "km/h")
print("95% CI:", response.uncertainty.confidence_interval_95)
```

---

## 8. Running Tests & Documentation

### Full PyTest Suite (67 Tests)
```bash
cd ML
python -m pytest tests/ -v
```
Test coverage across `tests/`:
- `test_architectures.py` (20 tests): Forward/backward passes and gradient flow for all 4 neural architectures.
- `test_inference.py` (14 tests): Thermal & Pressure inference contracts, calibration guards, determinism, and missing checkpoints.
- `test_pressure_preprocessing.py` (4 tests) & `test_thermal_preprocessing.py` (4 tests): Preprocessing & dataset split integrity.
- `test_services.py` (10 tests): Config, chunked synthetic loader, validation, model loading, inference, training, and experiment persistence.
- `test_streamlit_app.py` (10 tests): Headless Streamlit `AppTest` verification for all 5 pages and end-to-end workflows.
- `test_synthetic_training_and_ingestion.py` (5 tests): Synthetic generation, latent target semantics, session isolation, checkpoint isolation, and CSV ingestion.

### Key Project Reports
- [DATASET_AUDIT.md](file:///c:/Users/Dhruva%20G%20Shankar/FootGuard%20AI/ML/DATASET_AUDIT.md) — Phase 1 dataset audit
- [PHASE4_REPORT.md](file:///c:/Users/Dhruva%20G%20Shankar/FootGuard%20AI/ML/PHASE4_REPORT.md) & [PHASE5_REPORT.md](file:///c:/Users/Dhruva%20G%20Shankar/FootGuard%20AI/ML/PHASE5_REPORT.md) — Controlled baseline benchmarks & error analysis
- [PHASE4_FINAL_REPORT.md](file:///c:/Users/Dhruva%20G%20Shankar/FootGuard%20AI/ML/PHASE4_FINAL_REPORT.md) — Streamlit migration verification & stabilization report
- [SYNTHETIC_TRAINING_REPORT.md](file:///c:/Users/Dhruva%20G%20Shankar/FootGuard%20AI/ML/SYNTHETIC_TRAINING_REPORT.md) — Synthetic training & CSV ingestion report
