# FOOTGUARD AI — PHASE 3 STREAMLIT UI REPORT

**Date:** 2026-11-13  
**Scope:** Local-First Streamlit Research Studio inside `ML/app/`  
**Status:** Phase 3 Implementation Complete (Syntax Checked; Full Automated & Smoke Tests Pending Phase 4)

---

## 1. Application Structure

```text
ML/
├── app/
│   ├── __init__.py
│   ├── main.py                      # Streamlit entrypoint using st.navigation
│   ├── state.py                     # @st.cache_resource service singletons (no cached mutable training state)
│   ├── components/
│   │   ├── __init__.py
│   │   ├── metric_cards.py          # Research disclaimer banner, provenance badges, KPI rows
│   │   ├── thermal_viewer.py        # Plotly bilateral Celsius heatmaps & uncalibrated image warnings
│   │   └── pressure_viewer.py       # Plotly vGRF stance curves & synthetic 6-site session plots
│   └── pages/
│       ├── __init__.py
│       ├── home.py                  # Overview, counts, quick navigation, recent experiments
│       ├── datasets.py              # Dataset discovery, metadata, chunked synthetic preview, compatibility
│       ├── training.py              # Model/checkpoint registry, gated training, live progress & curves
│       ├── prediction.py            # Thermal & Pressure inference with calibration & DEMO safeguards
│       └── experiments.py           # History, confusion matrices, metrics, loss curves, JSON/MD exports
├── requirements.txt                 # Updated with streamlit>=1.38.0, plotly>=5.20.0, pyyaml>=6.0.0
└── README.md                        # Updated with local launch instructions
```

---

## 2. Implemented Pages & Components

1. **Home (`app/pages/home.py`)**:
   - Displays dataset, architecture, checkpoint, and experiment counts.
   - Provides quick navigation buttons (`st.switch_page`) to Datasets, Prediction, Training, and Experiments.
   - Summarizes available datasets, models, and recent experiments.
2. **Datasets (`app/pages/datasets.py`)**:
   - Inspects metadata (`inspect_metadata`) without loading the 258.9 MB / 1M-row synthetic CSV into memory.
   - Supports single-session loading (`SYN-0001`..) and strided sampling for `synthetic_paired_readings` via `DatasetService`.
   - Displays `CLINICAL_RESEARCH_DATASET` vs. `SYNTHETIC_NOT_CLINICAL_GROUND_TRUTH` provenance badges, calibration status, validation reports (`ValidationService`), and model compatibility checks.
   - Renders Plotly bilateral heatmaps (`ThermoDataBase`, `IAC-TecMed`), uncalibrated RGB warning views (`STANDUP`), 100-pt vGRF curves (`CAD_WALK`), and synthetic 6-site pressure/thermal session plots.
3. **Models & Training (`app/pages/training.py`)**:
   - Lists discovered checkpoints and model contracts (`ModelService`).
   - Enforces `ValidationService.check_model_compatibility(dataset_id, model_id)` — **Start Training** is disabled whenever a pairing is blocked (`standup`, `gait_lab`, `synthetic_paired_readings`, or mismatched modalities).
   - Executes `TrainingService.start_training(...)` with a live progress bar, displays test evaluation metrics and loss curves, and saves new checkpoints without overwriting protected Phase 4/5 checkpoints.
4. **Prediction (`app/pages/prediction.py`)**:
   - **Thermal:** Supports real `ThermoDataBase` samples or calibrated `.csv` uploads, adjustable decision threshold $\tau$, displays uncalibrated softmax score caveats, bilateral asymmetry metrics ($\Delta T > 2.2^\circ\text{C}$), SHA-256 `input_hash`, and heatmaps.
   - **Pressure:** Supports real `CAD_WALK` trials, `.csv`/`.npy` 1D vGRF uploads, or scalar parameters (`stance_duration_ms`, `peak_force_n`). Scalar-only mode defaults to Ridge-only (`allow_synthetic_demo=False`); DEMO mode is explicitly gated behind an opt-in checkbox.
5. **Experiments & Results (`app/pages/experiments.py`)**:
   - Surfaces Phase 4/5 legacy baselines alongside new local runs, filterable by `dataset_origin` (`All`, `real`, `synthetic`).
   - Renders model comparison tables, Plotly confusion matrices, regression metrics, training curves, Phase 5 uncertainty JSON, and one-click JSON / Markdown report downloads (`st.download_button`).

---

## 3. Service Integration & Compatibility Handling

- All UI pages invoke `ML/services/` directly via `app/state.py` (`@st.cache_resource` for stateless/read-only services; `create_training_service()` creates a fresh instance per training run).
- Zero FastAPI, PostgreSQL, React, Docker, or duplicated ML/inference logic was introduced.
- Errors (missing checkpoints, uncalibrated `.png` uploads, invalid arrays, blocked training configs) are caught and rendered as clean `st.error(...)` messages without exposing raw Python stack traces.

---

## 4. Dependencies & Launch Instructions

Updated `ML/requirements.txt` (`streamlit>=1.38.0`, `plotly>=5.20.0`, `pyyaml>=6.0.0`) and `ML/README.md`:
```bash
cd ML
pip install -r requirements.txt
streamlit run app/main.py
```

---

## 5. Known Limitations

1. **Runtime Verification Pending Phase 4:** Syntax compilation (`python -m py_compile`) passed with `EXIT_CODE=0` across all 10 `app/` files, but live headless Streamlit startup and end-to-end UI smoke testing are reserved for Phase 4.
2. **CPU Execution:** All local training and inference run on CPU (`torch 2.12.1+cpu`).
3. **Research Restrictions Preserved:** No ulcer detection, healing-time prediction, or calibrated clinical probability claims; `synthetic_paired_readings` remains inspection-only.

