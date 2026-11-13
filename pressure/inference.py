"""
FootGuard AI - Pressure Inference Pipeline
Provides production-grade inference for the Pressure 1D CNN model and Ridge baseline.
Accepts vertical ground reaction force (vGRF) curves, raw temporal traces,
tabular gait parameters (stance duration, peak force), or Pydantic request objects.
Performs temporal resampling, normalization, biomechanical dynamics extraction,
walking speed regression, and uncertainty estimation.

SAFETY & INTEGRITY CONSTRAINTS:
- Real sensor measurements are strictly required for 1D CNN evaluation.
- Synthetic signal generation is prohibited in production and restricted exclusively
  to explicitly flagged DEMO mode (allow_synthetic_demo=True).
- In standard mode, scalar-only input executes the validated Ridge baseline model
  (which operates legitimately on stance duration and peak force) and omits the 1D CNN.
"""

import os
import sys
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Union, Tuple, Optional, List
import numpy as np
import torch
import joblib

ML_ROOT = Path(__file__).resolve().parent.parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from common.config import CHECKPOINTS_DIR, PRESSURE_STANCE_RESAMPLED_LEN
from common.logging import get_logger
from common.schemas import (
    PressureInferenceRequest,
    PressureInferenceResponse,
    PressurePrediction,
    StanceDynamics,
    PressureUncertainty,
    QualityFlags
)
from pressure.cnn import Pressure1DCNN

logger = get_logger("pressure_inference")


class PressureInferenceEngine:
    """
    Decoupled inference engine for temporal pressure and gait dynamics regression.
    Evaluates Pressure1DCNN alongside the biomechanical Ridge regression benchmark.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        ridge_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None
    ):
        """
        Initializes the pressure inference engine.
        Args:
            checkpoint_path: Path to best_model.pt for Pressure 1D CNN.
            ridge_path: Path to ridge_baseline.joblib.
            device: 'cuda', 'cpu', or None (auto-detected).
        """
        if checkpoint_path is None:
            self.checkpoint_path = CHECKPOINTS_DIR / "experiment_b_pressure_1d_cnn" / "best_model.pt"
        else:
            self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Pressure CNN checkpoint not found at: {self.checkpoint_path.resolve()}")

        # Device selection
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_version = "1.0.0"
        self.model_id = "pressure_1d_cnn"
        self.force_scale = 800.0  # Training normalization scale (N)
        self.target_len = PRESSURE_STANCE_RESAMPLED_LEN  # 100 points

        # Load 1D CNN
        self.model = Pressure1DCNN(
            in_channels=1,
            out_features=1,
            base_channels=32,
            feature_dim=128,
            dropout_rate=0.2,
            expected_seq_len=self.target_len
        )
        self._load_checkpoint()
        self.model.to(self.device)
        self.model.eval()

        # Load Ridge Baseline Model
        if ridge_path is None:
            self.ridge_path = CHECKPOINTS_DIR / "experiment_b_pressure_1d_cnn" / "ridge_baseline.joblib"
        else:
            self.ridge_path = Path(ridge_path)

        self.ridge_model = self._load_ridge_model()

        logger.info(
            f"PressureInferenceEngine initialized successfully on {self.device} "
            f"from {self.checkpoint_path.name}"
        )

    def _load_checkpoint(self):
        """Loads weights from checkpoint file with explicit validation."""
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                self.model_version = checkpoint.get("model_version", "1.0.0")
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint
            else:
                state_dict = checkpoint.state_dict()

            self.model.load_state_dict(state_dict)
        except Exception as e:
            raise RuntimeError(f"Failed to load Pressure CNN checkpoint from {self.checkpoint_path}: {e}")

    def _load_ridge_model(self):
        """Loads serialized Ridge model or instantiates with validated Phase 4 weights."""
        if self.ridge_path.exists():
            try:
                model = joblib.load(self.ridge_path)
                return model
            except Exception as e:
                logger.warning(f"Could not load Ridge model from {self.ridge_path}: {e}. Using calibrated fallback.")

        # Exact Phase 4 Ridge parameters fit on 36 training subjects:
        # intercept = 8.2479787, coef = [-0.00664975 (duration_ms), +0.00125725 (peak_force_n)]
        class CalibratedRidgeFallback:
            coef_ = np.array([-0.00664975, 0.00125725])
            intercept_ = 8.247978737665141
            def predict(self, X):
                return np.dot(X, self.coef_) + self.intercept_

        return CalibratedRidgeFallback()

    def _compute_input_hash(self, curve: Optional[np.ndarray], duration_ms: float, peak_force_n: float) -> str:
        """Generates SHA-256 hash for provenance and audit logging."""
        hasher = hashlib.sha256()
        if curve is not None:
            hasher.update(curve.tobytes())
        hasher.update(f"{duration_ms:.2f}:{peak_force_n:.2f}".encode("utf-8"))
        return hasher.hexdigest()

    def resample_curve(self, curve: np.ndarray, target_len: int = 100) -> np.ndarray:
        """
        Resamples a 1D force curve of arbitrary length L to canonical length (100).
        Uses continuous 1D linear interpolation across normalized stance percent [0, 1].
        """
        curr_len = len(curve)
        if curr_len == target_len:
            return curve.astype(np.float32)

        x_orig = np.linspace(0.0, 1.0, curr_len)
        x_target = np.linspace(0.0, 1.0, target_len)
        resampled = np.interp(x_target, x_orig, curve)
        return resampled.astype(np.float32)

    def generate_canonical_curve(self, peak_force_n: float) -> np.ndarray:
        """
        DEMO ONLY: Generates a canonical bimodal vertical ground reaction force (vGRF) curve
        characteristic of human walking stance (heel strike peak, mid-stance dip, push-off peak).
        RESTRICTED TO DEMO MODE. NEVER PRESENTED AS REAL SENSOR DATA.
        """
        t = np.linspace(0.0, 1.0, self.target_len)
        bell1 = np.exp(-((t - 0.25) ** 2) / (2 * 0.10 ** 2))
        bell2 = np.exp(-((t - 0.75) ** 2) / (2 * 0.10 ** 2))
        raw_curve = 0.95 * bell1 + 1.0 * bell2
        raw_curve[0] = 0.0
        raw_curve[-1] = 0.0
        scale = peak_force_n / np.max(raw_curve) if np.max(raw_curve) > 0 else 1.0
        return (raw_curve * scale).astype(np.float32)

    def extract_dynamics(
        self, curve: np.ndarray, duration_ms: float
    ) -> Tuple[float, float, float, float]:
        """
        Extracts biomechanical kinetic metrics from real sensor curves:
        Returns:
            peak_force_n, duration_ms, impulse_n_s, loading_rate_n_s
        """
        peak_force_n = float(np.max(curve))
        dt_seconds = (duration_ms / 1000.0) / max(1, len(curve) - 1)
        impulse_n_s = float(np.trapezoid(curve, dx=dt_seconds))

        # Initial loading rate: slope from contact to ~20% of stance
        idx_20 = max(1, int(len(curve) * 0.20))
        t_20_sec = (idx_20 / len(curve)) * (duration_ms / 1000.0)
        loading_rate_n_s = float(curve[idx_20] / t_20_sec) if t_20_sec > 0 else 0.0

        return peak_force_n, duration_ms, impulse_n_s, loading_rate_n_s

    def sanitize_inputs(
        self,
        curve_input: Optional[Union[np.ndarray, List[float]]],
        duration_input: Optional[float],
        peak_input: Optional[float],
        sample_rate_hz: float = 100.0,
        allow_synthetic_demo: bool = False
    ) -> Tuple[Optional[np.ndarray], float, float, QualityFlags, bool]:
        """
        Validates, cleans, and standardizes inputs.
        Supports real sensor curve or scalar biomechanical inputs.
        Returns:
            curve_100: (100,) array if real curve or demo mode, else None.
            duration_ms: float ground contact time.
            peak_force_n: float peak vertical force.
            quality: QualityFlags object.
            is_demo: bool flag indicating whether synthetic signals were generated.
        """
        warnings = []
        curve_100 = None
        duration_ms = duration_input
        peak_force_n = peak_input
        is_demo = False

        # 1. Process Real Force Curve
        if curve_input is not None:
            curve = np.array(curve_input, dtype=np.float32).ravel()

            if len(curve) < 10:
                raise ValueError(
                    f"Force curve length too short ({len(curve)} points). "
                    "Capturing stance phase dynamics requires at least 10 sensor samples."
                )

            if np.isnan(curve).any():
                warnings.append("NaN values detected in force curve; replaced with 0.0.")
                curve = np.nan_to_num(curve, nan=0.0)

            if np.isinf(curve).any():
                warnings.append("Infinite values detected in force curve; clipped.")
                curve = np.nan_to_num(curve, posinf=5000.0, neginf=0.0)

            # Ground reaction forces cannot be negative; clip noise
            if (curve < 0.0).any():
                min_val = float(np.min(curve))
                if min_val < -50.0:
                    warnings.append(f"Significant negative forces ({min_val:.1f} N) detected and zero-clipped.")
                curve = np.clip(curve, 0.0, None)

            measured_peak = float(np.max(curve))
            if measured_peak < 50.0:
                raise ValueError(
                    f"Insufficient vertical force signal: maximum force is {measured_peak:.1f} N (<50.0 N threshold). "
                    "Real sensor ground contact required."
                )

            if np.var(curve) < 1.0:
                raise ValueError("Flatline sensor signal detected. Real dynamic stance contact required.")

            if peak_force_n is None:
                peak_force_n = measured_peak
            elif abs(peak_force_n - measured_peak) > 50.0:
                warnings.append(
                    f"Provided peak_force ({peak_force_n:.1f} N) deviates from curve maximum ({measured_peak:.1f} N)."
                )

            # Reconcile duration
            if duration_ms is None:
                if len(curve) == self.target_len:
                    duration_ms = 620.0  # Standard nominal human stance duration
                else:
                    duration_ms = float((len(curve) / sample_rate_hz) * 1000.0)

            if duration_ms <= 0:
                raise ValueError(f"Stance duration must be strictly positive, got {duration_ms} ms.")

            # Resample real sensor curve to canonical 100 points
            curve_100 = self.resample_curve(curve, self.target_len)

        else:
            # 2. Scalar-Only Branch
            if peak_force_n is None or duration_ms is None:
                raise ValueError(
                    "Insufficient input: Either 'vertical_force_curve' or both 'peak_force_n' and 'stance_duration_ms' must be provided."
                )

            if duration_ms <= 0:
                raise ValueError(f"Stance duration must be strictly positive, got {duration_ms} ms.")
            if duration_ms > 3000.0:
                raise ValueError(f"Stance duration exceeds physiological limit ({duration_ms} ms > 3000 ms).")

            if peak_force_n <= 0:
                raise ValueError(f"Peak force must be strictly positive, got {peak_force_n} N.")
            if peak_force_n > 5000.0:
                raise ValueError(f"Peak force exceeds physiological limit ({peak_force_n} N > 5000 N).")

            if allow_synthetic_demo:
                curve_100 = self.generate_canonical_curve(peak_force_n)
                is_demo = True
                warnings.append(
                    "[DEMO MODE] Synthesized canonical bimodal vGRF profile from scalar parameters. "
                    "NOT a real sensor measurement."
                )
            else:
                # Standard mode: do NOT synthesize curve. CNN inference will be omitted.
                curve_100 = None
                warnings.append(
                    "Scalar-only input provided: 1D CNN waveform inference omitted. "
                    "Using validated Ridge biomechanical model fit directly on stance duration and peak force."
                )

        # Physiological sanity checks
        if peak_force_n < 100.0:
            warnings.append(f"Abnormally low peak vertical force: {peak_force_n:.1f} N.")
        elif peak_force_n > 3500.0:
            warnings.append(f"Supra-physiological peak vertical force: {peak_force_n:.1f} N (>3500 N).")

        if duration_ms < 150.0:
            warnings.append(f"Unrealistically short stance duration: {duration_ms:.1f} ms (<150 ms).")
        elif duration_ms > 2000.0:
            warnings.append(f"Abnormally prolonged stance duration: {duration_ms:.1f} ms (>2000 ms).")

        quality = QualityFlags(
            is_valid=(len(warnings) == 0 or (len(warnings) == 1 and "Scalar-only" in warnings[0])),
            warnings=warnings
        )

        return curve_100, float(duration_ms), float(peak_force_n), quality, is_demo

    @torch.no_grad()
    def predict(
        self,
        request: Union[PressureInferenceRequest, Dict[str, Any], np.ndarray, List[float]]
    ) -> PressureInferenceResponse:
        """
        Executes end-to-end pressure gait inference.
        Accepts:
            - PressureInferenceRequest Pydantic object
            - Dict matching the request schema
            - 1D numpy array or list of floats representing the force curve
        Returns:
            PressureInferenceResponse object
        """
        start_time = time.perf_counter()

        # Parse request inputs
        allow_demo = False
        if isinstance(request, (np.ndarray, list)):
            curve_input = request
            duration_input = None
            peak_input = None
            sample_rate = 100.0
        elif isinstance(request, PressureInferenceRequest):
            curve_input = request.vertical_force_curve
            duration_input = request.stance_duration_ms
            peak_input = request.peak_force_n
            sample_rate = request.sample_rate_hz or 100.0
            allow_demo = request.allow_synthetic_demo
        elif isinstance(request, dict):
            curve_input = request.get("vertical_force_curve")
            duration_input = request.get("stance_duration_ms")
            peak_input = request.get("peak_force_n")
            sample_rate = request.get("sample_rate_hz", 100.0)
            allow_demo = request.get("allow_synthetic_demo", False)
        else:
            raise TypeError(f"Invalid request type: {type(request)}")

        # Validate, sanitize, and enforce demo boundaries
        curve_100, duration_ms, peak_force_n, quality, is_demo = self.sanitize_inputs(
            curve_input, duration_input, peak_input, sample_rate, allow_synthetic_demo=allow_demo
        )

        input_hash = self._compute_input_hash(curve_100, duration_ms, peak_force_n)

        # 1. 1D CNN Prediction (Only if real curve is provided or in explicit demo mode)
        predicted_walking_speed_kmh = None
        speed_cnn = None
        if curve_100 is not None:
            curve_norm = (curve_100 / self.force_scale).astype(np.float32)
            tensor = torch.from_numpy(curve_norm).unsqueeze(0).unsqueeze(0).to(self.device)
            speed_cnn = float(self.model(tensor).item())
            predicted_walking_speed_kmh = round(speed_cnn, 2)
            peak_f, dur_m, impulse, loading_rate = self.extract_dynamics(curve_100, duration_ms)
        else:
            # Scalar-only mode: Stance dynamics without curve integration
            peak_f = peak_force_n
            dur_m = duration_ms
            impulse = None
            loading_rate = None

        # 2. Ridge Regression Baseline Prediction (Always evaluated from valid scalars)
        X_ridge = np.array([[duration_ms, peak_force_n]], dtype=np.float32)
        raw_speed_ridge = float(self.ridge_model.predict(X_ridge)[0])
        speed_ridge = max(0.0, raw_speed_ridge)
        baseline_ridge_speed_kmh = round(speed_ridge, 2)

        # Determine primary model, speed bracket, and uncertainty
        if curve_100 is not None and not is_demo:
            primary_model = "pressure_1d_cnn"
            eval_speed = speed_cnn
            benchmark_mae = 0.431  # CAD_WALK test MAE for 1D CNN
        else:
            primary_model = "pressure_ridge_baseline"
            eval_speed = speed_ridge
            benchmark_mae = 0.259  # CAD_WALK test MAE for Ridge baseline

        if eval_speed < 4.5:
            bracket = "slow (<4.5 km/h)"
        elif eval_speed <= 5.5:
            bracket = "normal (4.5-5.5 km/h)"
        else:
            bracket = "fast (>5.5 km/h)"

        ci_lower = max(0.0, round(eval_speed - 1.96 * benchmark_mae, 2))
        ci_upper = round(eval_speed + 1.96 * benchmark_mae, 2)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        status = "warning" if is_demo or not quality.is_valid else "success"

        response = PressureInferenceResponse(
            model_id=self.model_id,
            model_version=self.model_version,
            status=status,
            is_demo=is_demo,
            input_hash=input_hash,
            prediction=PressurePrediction(
                predicted_walking_speed_kmh=predicted_walking_speed_kmh,
                baseline_ridge_speed_kmh=baseline_ridge_speed_kmh,
                primary_model_used=primary_model,
                speed_bracket=bracket
            ),
            stance_dynamics=StanceDynamics(
                peak_force_n=round(peak_f, 1),
                stance_duration_ms=round(dur_m, 1),
                impulse_n_s=round(impulse, 2) if impulse is not None else None,
                loading_rate_n_s=round(loading_rate, 1) if loading_rate is not None else None
            ),
            uncertainty=PressureUncertainty(
                estimated_mae_kmh=benchmark_mae,
                confidence_interval_95=[ci_lower, ci_upper]
            ),
            quality_flags=quality,
            disclaimer=(
                "RESEARCH USE ONLY: Biomechanical walking speed estimation on healthy controls. "
                "NOT an ulcer, neuropathy, or clinical diagnostic tool."
            ),
            latency_ms=round(elapsed_ms, 2)
        )

        return response
