# FootGuard AI — Backend Integration Guide & API Contracts

**Document Version:** 1.0.0  
**Target Audience:** Backend Engineering Team (`FootGuard AI/BE`)  
**Scope:** Machine Learning Service Contracts, Schemas, Latency Profiles, and Integration Blueprints  

---

## 1. Architectural Overview & Boundaries

The Machine Learning subsystem (`FootGuard AI/ML`) is strictly decoupled from the Backend web layer (`FootGuard AI/BE`). 

- The ML module **does not run HTTP servers or define FastAPI routes internally**.
- Instead, the ML module exposes lightweight, pure-Python, zero-network **Inference Engines** and **Pydantic Data Contracts** in `ML/common/schemas.py`.
- The Backend service imports these engines as local library modules, executes inference within asynchronous FastAPI endpoints or background task queues (e.g., Celery / Redis queue), and handles client transport, authentication, and database persistence.

```
+-----------------------------------------------------------------------------------+
|                            FootGuard AI Backend (BE)                             |
|                                                                                   |
|   +-----------------------+    HTTP/JSON     +--------------------------------+   |
|   |  Client / Clinical UI | <--------------> | FastAPI Application (main.py)  |   |
|   +-----------------------+                  +--------------------------------+   |
|                                                              |                    |
|                                               Pydantic Validation                 |
|                                               (ML/common/schemas.py)              |
|                                                              v                    |
|                                              +--------------------------------+   |
|                                              |  ML Inference Service Layer    |   |
|                                              +--------------------------------+   |
+--------------------------------------------------------------|--------------------+
                                                               v
+-----------------------------------------------------------------------------------+
|                        FootGuard AI Machine Learning (ML)                         |
|                                                                                   |
|      +--------------------------------+      +--------------------------------+   |
|      |     ThermalInferenceEngine     |      |    PressureInferenceEngine     |   |
|      |   (thermal/inference.py)       |      |    (pressure/inference.py)     |   |
|      +--------------------------------+      +--------------------------------+   |
|                      |                                       |                    |
|                      v                                       v                    |
|      +--------------------------------+      +--------------------------------+   |
|      |      Thermal 2D CNN Model      |      |   Pressure 1D CNN + Ridge      |   |
|      | (checkpoints/experiment_a/...) |      | (checkpoints/experiment_b/...) |   |
|      +--------------------------------+      +--------------------------------+   |
+-----------------------------------------------------------------------------------+
```

---

## 2. Environment & Dependency Setup

The Backend environment must have the ML package dependencies available. In production, the BE service can share the Python virtual environment or install `ML/requirements.txt`:

```bash
cd "FootGuard AI/ML"
pip install -r requirements.txt
```

### Python Path Configuration
Ensure that the `FootGuard AI/ML` root directory is on the Python path when launching the backend application:

```python
import sys
from pathlib import Path

# Add ML root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Points to FootGuard AI/
ML_ROOT = PROJECT_ROOT / "ML"
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))
```

---

## 3. Inference Engine Life Cycle & Singleton Pattern

Because loading PyTorch checkpoints and initializing neural networks involves disk I/O and weight allocation, **do not instantiate inference engines inside individual request handlers**.

Instantiate each engine once during the FastAPI application lifespan (startup event) and store it in `app.state`:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from thermal.inference import ThermalInferenceEngine
from pressure.inference import PressureInferenceEngine

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize ML Inference Engines once on startup
    app.state.thermal_engine = ThermalInferenceEngine(device="cpu", decision_threshold=0.50)
    app.state.pressure_engine = PressureInferenceEngine(device="cpu")
    yield
    # Clean up resources if necessary
    del app.state.thermal_engine
    del app.state.pressure_engine

app = FastAPI(title="FootGuard AI API", lifespan=lifespan)
```

---

## 4. API Endpoints & Request/Response Contracts

### A. Thermal Cohort Screening Endpoint

- **Route:** `POST /api/v1/ml/thermal/predict`
- **Purpose:** Plantar thermogram cohort classification (Control vs. Diabetic cohort) with quantitative bilateral temperature asymmetry analysis.
- **Latency Expectation:** 15–30 ms per pair (CPU).

#### Request Contract: `ThermalInferenceRequest`
```json
{
  "patient_id": "PT_10482",
  "session_id": "SES_2026_01",
  "left_foot_array": [
    [26.4, 27.1, 28.0, 27.5],
    [26.8, 28.2, 29.1, 28.0],
    [27.0, 28.5, 29.3, 27.9]
  ],
  "right_foot_array": [
    [26.2, 27.0, 27.8, 27.3],
    [26.6, 28.0, 28.9, 27.8],
    [26.9, 28.1, 29.0, 27.5]
  ],
  "left_foot_csv_path": null,
  "right_foot_csv_path": null,
  "ambient_temp_c": 22.5
}
```

> **IMPORTANT INTEGRITY RULES ON THERMAL CALIBRATION & STATISTICAL METRICS:**
> - **Celsius Temperature Matrices Required:** Inputs must contain true physical floating-point temperatures in Celsius (typically $18^\circ\text{C}$ to $38^\circ\text{C}$).
> - **Uncalibrated Images Prohibited:** 8-bit pseudo-color image formats (PNG, JPEG, TIFF from datasets like STANDUP) are strictly rejected with HTTP 422. Their pixel intensities represent colormap rendering, NOT physical Celsius temperatures.
> - **Statistical Confidence Interpretation:** The `confidence` score is a raw softmax probability from the 2D CNN. It has **NOT** undergone post-hoc calibration (such as Platt scaling or isotonic regression). BE and FE must display it as a relative model confidence score, NOT as an absolute posterior clinical risk probability.


#### Response Contract: `ThermalInferenceResponse`
```json
{
  "model_id": "thermal_2d_cnn",
  "model_version": "1.0.0",
  "status": "success",
  "input_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "prediction": {
    "predicted_class_id": 1,
    "predicted_label": "Diabetic Cohort",
    "confidence": 0.8842,
    "probability_control": 0.1158,
    "probability_diabetic": 0.8842,
    "decision_threshold": 0.50
  },
  "bilateral_metrics": {
    "mean_left_temp_c": 28.35,
    "mean_right_temp_c": 28.02,
    "mean_abs_asymmetry_c": 0.42,
    "max_abs_asymmetry_c": 1.15,
    "hyperthermic_asymmetry_fraction": 0.0000
  },
  "quality_flags": {
    "is_valid": true,
    "warnings": []
  },
  "disclaimer": "RESEARCH USE ONLY: Diabetic cohort screening model. NOT an ulcer detector or clinical diagnostic device.",
  "latency_ms": 14.85
}
```

---

### B. Pressure & Gait Dynamics Endpoint

- **Route:** `POST /api/v1/ml/pressure/predict`
- **Purpose:** Biomechanical walking speed regression and vertical ground reaction force (vGRF) dynamics extraction.
- **Latency Expectation:** 5–10 ms per step (CPU).

#### Request Contract: `PressureInferenceRequest`
```json
{
  "patient_id": "PT_10482",
  "trial_id": "STEP_04",
  "foot_side": "left",
  "vertical_force_curve": [
    0.0, 110.5, 320.0, 680.2, 850.4, 760.0, 690.1, 740.0, 830.5, 520.0, 150.0, 0.0
  ],
  "stance_duration_ms": 615.0,
  "peak_force_n": 850.4,
  "sample_rate_hz": 100.0,
  "file_path": null,
  "allow_synthetic_demo": false
}
```

> **IMPORTANT INTEGRITY RULE ON SYNTHETIC DATA & DEMO MODE:**
> - `allow_synthetic_demo` defaults to `false`.
> - When `vertical_force_curve` is omitted, the production pipeline runs **exclusively the validated Ridge regression baseline** (which is legitimately trained on `stance_duration_ms` and `peak_force_n`). The 1D CNN prediction will be `null`, `primary_model_used` will be `"pressure_ridge_baseline"`, and `is_demo` will be `false`.
> - Setting `allow_synthetic_demo: true` is strictly reserved for test/demo environments. It generates a synthetic bimodal curve for the 1D CNN, sets `is_demo: true`, and triggers a warning. **Never enable demo mode in clinical workflows.**

#### Response Contract: `PressureInferenceResponse`
```json
{
  "model_id": "pressure_1d_cnn",
  "model_version": "1.0.0",
  "status": "success",
  "is_demo": false,
  "input_hash": "4a53e50669e0f63b4f9810d79cfef6f8e77a16629910a3070d65b7588ef05b0d",
  "prediction": {
    "predicted_walking_speed_kmh": 4.15,
    "baseline_ridge_speed_kmh": 5.23,
    "primary_model_used": "pressure_1d_cnn",
    "speed_bracket": "slow (<4.5 km/h)"
  },
  "stance_dynamics": {
    "peak_force_n": 850.4,
    "stance_duration_ms": 615.0,
    "impulse_n_s": 325.40,
    "loading_rate_n_s": 4621.7
  },
  "uncertainty": {
    "estimated_mae_kmh": 0.431,
    "confidence_interval_95": [3.31, 4.99]
  },
  "quality_flags": {
    "is_valid": true,
    "warnings": []
  },
  "disclaimer": "RESEARCH USE ONLY: Biomechanical walking speed estimation on healthy controls. NOT an ulcer, neuropathy, or clinical diagnostic tool.",
  "latency_ms": 6.82
}
```

---

## 5. Production FastAPI Router Implementation Blueprint

Backend engineers can implement the following router in `BE/routers/ml_router.py`:

```python
"""
FootGuard AI - Backend ML Router (BE/routers/ml_router.py)
Standard production integration of ML inference engines.
"""

from fastapi import APIRouter, Request, HTTPException, status
from common.schemas import (
    ThermalInferenceRequest,
    ThermalInferenceResponse,
    PressureInferenceRequest,
    PressureInferenceResponse
)

router = APIRouter(prefix="/api/v1/ml", tags=["Machine Learning"])


@router.post(
    "/thermal/predict",
    response_model=ThermalInferenceResponse,
    summary="Plantar Thermal Cohort Screening",
    description="Evaluates bilateral plantar thermograms using Thermal 2D CNN."
)
async def predict_thermal(payload: ThermalInferenceRequest, request: Request):
    engine = getattr(request.app.state, "thermal_engine", None)
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Thermal ML Inference Engine is not initialized."
        )

    try:
        response = engine.predict(payload)
        return response
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(fnf))
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Thermal inference internal error: {str(exc)}"
        )


@router.post(
    "/pressure/predict",
    response_model=PressureInferenceResponse,
    summary="Plantar Kinetic & Stance Regression",
    description="Evaluates 1D ground reaction force curves using Pressure 1D CNN and Ridge benchmark."
)
async def predict_pressure(payload: PressureInferenceRequest, request: Request):
    engine = getattr(request.app.state, "pressure_engine", None)
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pressure ML Inference Engine is not initialized."
        )

    try:
        response = engine.predict(payload)
        return response
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pressure inference internal error: {str(exc)}"
        )
```

---

## 6. HTTP Error Taxonomy & Status Codes

| HTTP Status Code | Scenario | Recommended Client Action |
|:---|:---|:---|
| **`200 OK`** | Inference completed successfully. | Render prediction, asymmetry metrics, and disclaimer. |
| **`400 Bad Request`** | Malformed JSON or schema violation. | Ensure fields conform to `ML/common/schemas.py`. |
| **`404 Not Found`** | Provided CSV path does not exist on disk. | Verify file upload paths or transmit arrays directly. |
| **`422 Unprocessable Entity`** | Biological/structural impossibility (e.g., negative duration, 1D array for thermal, empty curve). | Inspect sensor data validity before resubmission. |
| **`503 Service Unavailable`** | Engine not initialized in app lifespan. | Check backend server initialization logs and checkpoint paths. |
| **`500 Internal Error`** | Unhandled runtime exception inside PyTorch. | Log incident with `input_hash` for ML team triage. |

---

## 7. Performance, Latency & Scaling Characteristics

- **Hardware Target:** Multi-core modern CPU (Intel/AMD x86_64 or Apple Silicon ARM). CUDA GPU acceleration is supported automatically if `torch.cuda.is_available()`.
- **Memory Footprint:** 
  - `ThermalInferenceEngine`: ~45 MB RAM (Model: 405K parameters).
  - `PressureInferenceEngine`: ~12 MB RAM (Model: 44K parameters).
- **Throughput Benchmark (Intel Core i7 CPU, single process):**
  - Thermal 2D CNN: ~65 requests / second.
  - Pressure 1D CNN: ~140 requests / second.
- **Worker Scaling:** Standard Uvicorn/Gunicorn multi-worker deployments (`workers=4`) achieve ~400 req/sec aggregate throughput without GPU dependencies.

---

## 8. Clinical Compliance & Regulatory Governance

Backend engineers and frontend developers must adhere to the following governance rules:

1. **Mandatory Disclaimer Display:** Every UI screen or API consumer displaying model outputs must include the verbatim disclaimer:
   > *"RESEARCH USE ONLY: Not a diagnostic medical device or ulcer detection system."*
2. **Ulcer Detection Claims Prohibited:** Never label model outputs as "Ulcer Risk", "Pre-Ulcer Stage", or "Wound Healing Velocity". The ThermoDataBase model stratifies general diabetic cohorts, and CAD_WALK represents healthy control kinematics.
3. **Data Provenance:** Always store the returned `input_hash` alongside the clinical session record in the database (`DB`) to maintain an unbroken audit trail for regulatory compliance (IEC 62304 / FDA SaMD).

