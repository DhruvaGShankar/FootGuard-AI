# FOOTGUARD AI — ML INTEGRATION READINESS & SAFETY HARDENING REPORT

**Subsystem:** `FootGuard AI/ML`  
**Document Version:** 1.0.0  
**Date:** November 2026  
**Audience:** Backend Engineering Team (`BE`), Clinical Governance, System Architects  
**Scope:** Integration Readiness Audit, Safety Hardening, Statistical Integrity Verification, and Contract Specification  

---

## 1. Executive Summary & Readiness Verdict

The FootGuard AI Machine Learning subsystem has undergone integration hardening to ensure sensor safety, prevent presentation of synthetic data as real measurements, enforce strict physical calibration boundaries, and protect statistical integrity.

### Immediate Production Readiness Verdict:

```
+------------------------------------+--------------------------+-----------------------+------------------------------------------+
| Pipeline / Capability              | Input Contract           | Target Endpoint       | Integration Status                       |
+------------------------------------+--------------------------+-----------------------+------------------------------------------+
| Thermal Plantar Cohort Screening   | Calibrated CSV / Arrays  | POST /thermal/predict | READY FOR IMMEDIATE INTEGRATION          |
| Pressure Curve Gait Regression     | Real 1D vGRF Curve       | POST /pressure/predict| READY FOR IMMEDIATE INTEGRATION          |
| Scalar Stance Biomechanical Speed  | Stance Time & Peak Force | POST /pressure/predict| READY FOR IMMEDIATE INTEGRATION (Ridge)  |
| Synthetic Demo Force Signal        | allow_synthetic_demo=True| POST /pressure/predict| RESTRICTED TO DEMO MODE ONLY             |
| STANDUP Image Processing (Celsius) | Uncalibrated PNG/JPEG    | N/A                   | STRICTLY BLOCKED (Non-Celsius Colormap)  |
| Diabetic Ulcer / DFU Detection     | Any                      | N/A                   | STRICTLY BLOCKED (Zero Clinical DFU Data)|
| Dynamic Rewarming TCN              | 4-timepoint thermograms  | N/A                   | REJECTED / BLOCKED (Underfit on N=13)    |
| Gait Lab Insole Multi-Channel      | Continuous 18-channel    | N/A                   | REJECTED / BLOCKED (Confounded N=2)      |
+------------------------------------+--------------------------+-----------------------+------------------------------------------+
```

---

## 2. Pressure Inference Safety Hardening (`ML/pressure/inference.py`)

### Problem Identified in Prior Baseline:
In the initial Phase 6 implementation, when clients provided only scalar gait parameters (`stance_duration_ms` and `peak_force_n`), a helper function (`generate_canonical_curve`) synthesized an artificial bimodal vertical ground reaction force (vGRF) curve and fed it into the Pressure 1D CNN. This posed an architectural safety hazard: artificial curves were evaluated by a deep neural network and presented as real sensor measurements.

### Hardening Actions Implemented:
1. **Separation of Production vs. DEMO Execution:**
   - Synthetic curve generation is now strictly gated behind the explicit boolean parameter `allow_synthetic_demo: bool = False` (default).
   - In production (`allow_synthetic_demo=False`), the inference engine **does not synthesize any curve**.
2. **Scalar-Only Inference Uses Validated Ridge Model Exclusively:**
   - When only scalar parameters (`stance_duration_ms`, `peak_force_n`) are supplied in production mode:
     - The 1D CNN forward pass is omitted (`predicted_walking_speed_kmh = None`).
     - The validated **Ridge Regression Baseline** is executed directly on `[stance_duration_ms, peak_force_n]` (its exact training features).
     - `primary_model_used` is set to `"pressure_ridge_baseline"`.
     - Stance dynamics requiring curve integration (`impulse_n_s`, `loading_rate_n_s`) are set to `None`.
     - The uncertainty benchmark reflects the validated Ridge test MAE ($0.259\text{ km/h}$).
     - `is_demo` is marked `False`.
3. **Explicit DEMO Mode Labeling:**
   - When `allow_synthetic_demo: true` is explicitly passed by a client in test/demo environments:
     - `is_demo` is set to `True`.
     - `status` is forced to `"warning"`.
     - A prominent diagnostic warning is appended: `"[DEMO MODE] Synthesized canonical bimodal vGRF profile from scalar parameters. NOT a real sensor measurement."`
4. **Rejection of Insufficient Real Sensor Input:**
   - Real force curves shorter than 10 points are rejected with `ValueError` (cannot capture stance dynamics).
   - Curves with flatline signals or peak forces $<50.0\text{ N}$ are rejected with `ValueError` ("Insufficient vertical force signal... Real sensor ground contact required").
   - Non-positive stance duration ($\le 0\text{ ms}$) or prolonged duration ($>3000\text{ ms}$) are rejected with `ValueError`.
   - Non-positive peak force ($\le 0\text{ N}$) or extreme force ($>5000\text{ N}$) are rejected with `ValueError`.

---

## 3. Thermal Inference Validation & Calibration Verification (`ML/thermal/inference.py`)

### Problem Identified in Prior Baseline:
The raw datasets include STANDUP Database images (`R0_0001.png`), which are 8-bit pseudo-color PNG images produced by camera colormaps, not calibrated floating-point Celsius thermograms. Applying Celsius calculations (such as calculating clinical $\Delta T > 2.2^\circ\text{C}$ or feeding raw RGB intensities into the Thermal 2D CNN) would produce invalid clinical artifacts.

### Hardening Actions Implemented:
1. **Strict Rejection of Uncalibrated Formats:**
   - Added validation in both `ThermalInferenceRequest` (Pydantic field validator) and `_validate_and_sanitize_matrix`:
     - File paths ending in `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.tif` are rejected with `ValueError`.
     - File paths containing `"standup"` are rejected with `ValueError`.
     - 8-bit integer arrays (`matrix.dtype == np.uint8`) are rejected with `ValueError`.
     - Pixel arrays with maximum intensity $>50.0^\circ\text{C}$ (typical of 0–255 RGB colormaps) are rejected with `ValueError`.
   - Clear error message returned:
     > *"Uncalibrated image format detected (...). STANDUP and standard RGB/PNG images contain colormap pixel values, not physical Celsius temperatures. Celsius inference requires calibrated floating-point thermogram matrices (e.g. ThermoDataBase CSV)."*
2. **Bilateral Foot Orientation & Anatomical Correspondence:**
   - Validates that Left and Right foot inputs are provided independently.
   - The Right foot is horizontally flipped (`np.fliplr`) to align anatomical axes (medial arch and hallux) with the Left foot before computing $\Delta T = |T_{\text{Left}} - T_{\text{Right\_flipped}}|$.
   - Validates vertical orientation (warns if matrix width $> 2.5 \times$ height).
3. **Sensor Dropout & NaN Handling:**
   - Enforces a strict dropout threshold: if NaNs exceed **30%** of total pixels, the thermogram is rejected with `ValueError` ("Excessive sensor dropout... limit 30%").
   - Dropouts $\le 30\%$ are safely replaced with $0.0^\circ\text{C}$ background and flagged in `quality_flags.warnings`.
4. **Physiological Temperature Range Screening:**
   - Rejects thermograms where fewer than 50 plantar pixels exceed the $15.0^\circ\text{C}$ background threshold.
   - Flags quality warnings if plantar tissue exhibits sub-physiological ($<18.0^\circ\text{C}$) or hyper-physiological ($>42.0^\circ\text{C}$) values.
5. **Data Provenance & Cryptographic Audit:**
   - Generates a SHA-256 hash (`input_hash`) of the sanitized Left and Right temperature arrays and embeds it in every `ThermalInferenceResponse`.

---

## 4. Statistical Integrity & Evaluation Review

In strict compliance with clinical integrity directives, **no model parameters or decision thresholds were retuned on held-out test data**. The following documented findings from Phase 4 and Phase 5 govern backend interpretation:

### A. Observation Aggregation (Participant-Level Evaluation):
- In ThermoDataBase, each participant has exactly **one bilateral thermogram pair** (Left foot and Right foot acquired during the same examination session).
- Therefore, all reported thermal test metrics (test AUROC 0.812, F1 0.829, Brier score 0.167) represent **one prediction per participant**, strictly evaluated on the $N=26$ held-out test cohort (7 controls, 19 diabetics).
- There is zero trial pooling or multi-observation correlation in the thermal evaluation.

### B. Calibration Status of Confidence Scores:
- The `confidence` score in `ThermalPrediction` is a raw softmax probability from the final linear layer:
  $$\hat{p} = \frac{e^{z_1}}{e^{z_0} + e^{z_1}}$$
- **These scores have NOT undergone post-hoc calibration** (e.g., Platt scaling, temperature scaling, or isotonic regression).
- While the model exhibits reasonable discriminatory ranking (AUROC 0.812) and low mean squared error (Brier score 0.167), raw confidence scores must be presented in UI dashboards as **relative model classification scores**, NOT as calibrated posterior clinical risk probabilities.

### C. Test Population Limitations & Uncertainty:
- The held-out test cohorts are small:
  - ThermoDataBase: $N=26$ participants (7 Control, 19 Diabetic).
  - CAD_WALK: $N=7$ participants ($288$ valid trials total; subject C51 lacked speed labels and was excluded).
- Because all 288 CAD_WALK test trials originate from only 7 individuals, treating trials as statistically independent would severely underestimate true clinical uncertainty.
- In Phase 5, **1,000 participant-clustered bootstrap resamplings** were conducted:
  - **Pressure 1D CNN MAE:** $0.431\text{ km/h}$ (95% Clustered CI: $[0.262, 0.595]\text{ km/h}$).
  - **Ridge Regression MAE:** $0.259\text{ km/h}$ (95% Clustered CI: $[0.180, 0.365]\text{ km/h}$).
  - **Thermal 2D CNN AUROC:** $0.812$ (95% Bootstrap CI: $[0.684, 0.936]$).

### D. Pressure 1D CNN vs. Ridge Biomechanical Baseline:
- **Biomechanical Mechanism:** The Ridge model uses stance duration ($t_{\text{contact}}$ in ms) and peak force ($F_{\text{peak}}$ in N). In treadmill walking, stance time is physically inverse to speed ($t_{\text{contact}} \propto 1/v$). This direct biomechanical relationship allows Ridge to dominate nominal speeds ($4.5\text{--}6.0\text{ km/h}$).
- **Morphological Waveform Mechanism:** The Pressure 1D CNN operates on 100-point normalized curves where contact duration was deliberately resampled out. At slower walking speeds ($<4.5\text{ km/h}$), the CNN outperforms Ridge ($0.176\text{ km/h}$ vs. $0.351\text{ km/h}$ MAE) because ground reaction force wave morphology changes sharply during slow gait.
- **Backend Routing:** Backend services can safely rely on Ridge for scalar-only inputs and 1D CNN for high-density sensor curves.

### E. Post-Mortem of Failed & Rejected Models:
- **Pressure TCN:** Underperformed ($1.216\text{ km/h}$ MAE). Causal dilated convolutions enforce directional streaming causality; applying causal masking to a closed, time-normalized 100-point stance curve introduced artificial receptive field degradation.
- **Thermal TCN:** Severely underfit ($0.741^\circ\text{C}$ MAE). With only $N=13$ training subjects in IAC-TecMed, training a deep spatial encoder + temporal convolutional network from scratch was data-starved; the simple physical persistence baseline ($T_{10} \rightarrow T_{15}$) easily outperformed it ($0.267^\circ\text{C}$ MAE).
- **Diabetic Foot Ulcer (DFU) Classifiers:** REJECTED. None of the raw clinical datasets contain DFU presence, ulcer bounding boxes, or wound healing timelines. Any claim of ulcer detection is barred.

---

## 5. Model Artifacts & Checkpoint Verification

Both inference engines have been hardened with robust checkpoint loading and version tracking:

1. **Configurable Checkpoint Paths:**
   - `ThermalInferenceEngine(checkpoint_path=..., device=...)`
   - `PressureInferenceEngine(checkpoint_path=..., ridge_path=..., device=...)`
   - Default paths point to `ML/checkpoints/experiment_a_thermal_2d_cnn/best_model.pt` and `ML/checkpoints/experiment_b_pressure_1d_cnn/best_model.pt`.
2. **Explicit Exception Handling:**
   - If a checkpoint path does not exist on disk, a descriptive `FileNotFoundError` is raised immediately during engine initialization.
   - If a checkpoint file is corrupted or contains invalid state keys, a descriptive `RuntimeError` is raised.
3. **Model Versioning:**
   - Model versions (`"1.0.0"`) are stored in checkpoints and exposed in every API response payload (`response.model_version`).
4. **Hardware Portability:**
   - Automatically detects CUDA availability (`torch.cuda.is_available()`) but defaults cleanly to CPU execution without warnings.

---

## 6. Backend Contract Coordination & Compatibility

The Pydantic data contracts in `ML/common/schemas.py` have been hardened without breaking changes:

### Key Additions to Schema:
1. `PressureInferenceRequest.allow_synthetic_demo`: `bool = False`
   - Backend can leave this omitted or set to `False` for standard clinical workflows.
2. `PressurePrediction.predicted_walking_speed_kmh`: `Optional[float] = None`
   - Populated when real curves (or demo mode) are supplied; `None` when running in scalar-only Ridge mode.
3. `PressurePrediction.primary_model_used`: `str = Field("pressure_1d_cnn", ...)`
   - Indicates whether `"pressure_1d_cnn"` or `"pressure_ridge_baseline"` generated the primary walking speed estimation.
4. `PressureInferenceResponse.is_demo`: `bool = False`
   - Flags whether the response was produced using synthetic signals.
5. `ThermalInferenceRequest` Validators:
   - Rejects uncalibrated file formats (`.png`, `.jpg`, STANDUP paths) at the Pydantic schema validation layer before reaching PyTorch.

---

## 7. Verification & PyTest Suite Status

A comprehensive suite of unit, integration, validation, and safety tests was executed:

```bash
python -m pytest tests/ -v
```

### PyTest Suite Breakdown (42 Tests Passing in 5.92s):
- **Architecture Validation (20 tests in `test_architectures.py`):**
  - Thermal 2D CNN forward, backward, gradient flow, real batch, CPU batch-size-1, invalid input handling.
  - Thermal TCN forward, backward, real batch, CPU execution.
  - Pressure 1D CNN forward, backward, multichannel, real batch, CPU execution.
  - Pressure TCN forward, backward, multichannel, real batch, sequence output.
- **Preprocessing & Splitting (8 tests in `test_*_preprocessing.py`):**
  - Sensor jump detrending (`FixJumpv24`), CAD_WALK spatial grid padding `(48, 32)`, temporal stance resampling (100 pts), Gait Lab insole loading, ThermoDataBase bilateral standardization, participant-level split isolation.
- **Inference Pipeline & Integration Hardening (14 tests in `test_inference.py`):**
  - `test_thermal_inference_initialization`: Configurable checkpoint loading & versioning (**PASSED**).
  - `test_thermal_missing_checkpoint`: Explicit `FileNotFoundError` on missing checkpoint (**PASSED**).
  - `test_thermal_inference_synthetic_arrays`: Valid float32 arrays (**PASSED**).
  - `test_thermal_inference_real_csv_files`: Raw ThermoDataBase CSV files (**PASSED**).
  - `test_thermal_inference_pydantic_request`: Valid Pydantic request payload (**PASSED**).
  - `test_thermal_inference_invalid_inputs`: Missing CSVs, 1D arrays, arrays too small, excessive NaNs (>30%), insufficient signal (<50 pixels) (**PASSED**).
  - `test_thermal_inference_uncalibrated_inputs`: uint8 arrays, values > 50°C, PNG/JPG paths, STANDUP dataset paths (**PASSED**).
  - `test_pressure_inference_initialization`: Default and custom initialization (**PASSED**).
  - `test_pressure_missing_checkpoint`: Explicit `FileNotFoundError` on missing checkpoint (**PASSED**).
  - `test_pressure_inference_100_pt_curve`: 100-pt real vGRF curve (**PASSED**).
  - `test_pressure_inference_arbitrary_length_curve`: Arbitrary length resampling (**PASSED**).
  - `test_pressure_inference_scalar_real_vs_demo`: Gated demo separation; scalar-only runs Ridge, omits CNN; demo mode runs CNN and flags `is_demo=True` (**PASSED**).
  - `test_pressure_inference_invalid_inputs`: Flatline signals, curves < 10 pts, negative duration, negative peak force, duration > 3000 ms (**PASSED**).
  - `test_inference_determinism`: Identical inputs produce identical `input_hash` and numeric predictions (**PASSED**).

---

## 8. Summary for Backend Engineering (`BE`)

### What BE Can Implement Immediately:
1. **Import Schemas:**
   ```python
   from common.schemas import (
       ThermalInferenceRequest,
       ThermalInferenceResponse,
       PressureInferenceRequest,
       PressureInferenceResponse
   )
   ```
2. **Lifespan Engine Initialization:**
   ```python
   @asynccontextmanager
   async def lifespan(app: FastAPI):
       app.state.thermal_engine = ThermalInferenceEngine(device="cpu", decision_threshold=0.50)
       app.state.pressure_engine = PressureInferenceEngine(device="cpu")
       yield
   ```
3. **Execute Endpoints:**
   - Thermal cohort screening: `app.state.thermal_engine.predict(payload)`
   - Pressure gait speed estimation: `app.state.pressure_engine.predict(payload)`

### What Remains Blocked:
- Do not build endpoints for ulcer risk detection or wound healing timelines (zero data available).
- Do not send STANDUP PNG images to the thermal engine (uncalibrated).
- Do not enable `allow_synthetic_demo=True` in production clinical workflows.

