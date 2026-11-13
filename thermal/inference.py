"""
FootGuard AI - Thermal Inference Pipeline
Provides production-grade inference for the Thermal 2D CNN model.
Accepts calibrated CSV matrices, floating-point numpy arrays, or Pydantic request objects.
Performs bilateral standardization, anatomical flipping, asymmetry calculation,
quality screening, and cohort classification.

CALIBRATION & INTEGRITY CONSTRAINTS:
- Input thermograms must be calibrated physical Celsius floating-point temperatures.
- Uncalibrated 8-bit pseudo-color image formats (such as PNG/JPEG from STANDUP Database)
  are strictly prohibited because pixel values reflect RGB colormaps rather than true thermal energy.
- Requires both Left and Right feet to evaluate bilateral physiological asymmetry.
- Enforces strict sensor dropout thresholds (maximum 30% NaNs).
"""

import os
import sys
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Union, Tuple, Optional, List
import numpy as np
import pandas as pd
import cv2
import torch

ML_ROOT = Path(__file__).resolve().parent.parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from common.config import (
    CHECKPOINTS_DIR,
    THERMAL_CANONICAL_SHAPE,
    THERMAL_BACKGROUND_THRESHOLD_C,
    THERMAL_MIN_VALID_TEMP_C,
    THERMAL_MAX_VALID_TEMP_C
)
from common.logging import get_logger
from common.schemas import (
    ThermalInferenceRequest,
    ThermalInferenceResponse,
    ThermalPrediction,
    BilateralThermalMetrics,
    QualityFlags
)
from thermal.cnn import Thermal2DCNN
from thermal.preprocessing import resize_and_pad_thermal_matrix

logger = get_logger("thermal_inference")


class ThermalInferenceEngine:
    """
    Decoupled inference engine for thermal cohort classification.
    Loads trained Thermal2DCNN checkpoint and serves high-throughput, low-latency predictions.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        decision_threshold: float = 0.50
    ):
        """
        Initializes the inference engine.
        Args:
            checkpoint_path: Path to best_model.pt. Defaults to experiment_a_thermal_2d_cnn.
            device: 'cuda', 'cpu', or None (auto-detected).
            decision_threshold: Classification threshold (default 0.50 based on Phase 4 validation).
        """
        if checkpoint_path is None:
            self.checkpoint_path = CHECKPOINTS_DIR / "experiment_a_thermal_2d_cnn" / "best_model.pt"
        else:
            self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Thermal CNN checkpoint not found at: {self.checkpoint_path.resolve()}")

        # Device selection
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.decision_threshold = float(decision_threshold)
        self.model_version = "1.0.0"
        self.model_id = "thermal_2d_cnn"

        # Load model architecture and weights
        self.model = Thermal2DCNN(in_channels=3, num_classes=2, dropout_rate=0.3)
        self._load_checkpoint()
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            f"ThermalInferenceEngine initialized successfully on {self.device} "
            f"from {self.checkpoint_path.name} (threshold={self.decision_threshold})"
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
            raise RuntimeError(f"Failed to load Thermal CNN checkpoint from {self.checkpoint_path}: {e}")

    def _compute_input_hash(self, left_arr: np.ndarray, right_arr: np.ndarray) -> str:
        """Generates SHA-256 hash of input matrices for audit trails and data provenance."""
        hasher = hashlib.sha256()
        hasher.update(left_arr.tobytes())
        hasher.update(right_arr.tobytes())
        return hasher.hexdigest()

    def _validate_and_sanitize_matrix(
        self, matrix_input: Union[np.ndarray, List[List[float]], str, Path], side: str
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Validates, sanitizes, and converts input into a float32 2D numpy array.
        Enforces strict Celsius calibration and dimension checks.
        """
        warnings = []
        if isinstance(matrix_input, (str, Path)):
            path = Path(matrix_input)
            lower_name = str(path).lower()

            # Reject uncalibrated image formats
            if any(lower_name.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]):
                raise ValueError(
                    f"Uncalibrated image format '{path.suffix}' detected in {side} foot path '{path}'. "
                    "STANDUP and standard RGB/PNG images contain colormap pixel values, not physical Celsius temperatures. "
                    "Celsius inference requires calibrated floating-point thermogram matrices (e.g. ThermoDataBase CSV)."
                )

            # Reject STANDUP dataset paths
            if "standup" in lower_name:
                raise ValueError(
                    f"STANDUP dataset file rejected for {side} foot: STANDUP contains uncalibrated colormap images. "
                    "Celsius inference requires physical temperature matrices."
                )

            if not path.exists():
                raise FileNotFoundError(f"CSV file for {side} foot not found: {path.resolve()}")

            df = pd.read_csv(path, header=None)
            matrix = df.to_numpy(dtype=np.float32)

        elif isinstance(matrix_input, np.ndarray):
            # Reject uint8 matrices representing 8-bit images
            if matrix_input.dtype == np.uint8:
                raise ValueError(
                    f"Uncalibrated 8-bit integer matrix (uint8) detected for {side} foot. "
                    "Celsius thermogram inference requires physical temperature floating-point values."
                )
            matrix = matrix_input.astype(np.float32)

        elif isinstance(matrix_input, list):
            matrix = np.array(matrix_input, dtype=np.float32)
        else:
            raise TypeError(f"Unsupported matrix input type for {side}: {type(matrix_input)}")

        if matrix.ndim != 2:
            raise ValueError(f"{side} matrix must be 2D, but got shape {matrix.shape}.")

        if matrix.shape[0] < 30 or matrix.shape[1] < 15:
            raise ValueError(
                f"{side} matrix dimensions too small: {matrix.shape}. Minimum resolution is (30, 15)."
            )

        # Aspect ratio check: foot length is typically greater than foot width
        if matrix.shape[1] > 2.5 * matrix.shape[0]:
            warnings.append(
                f"Inverted orientation detected in {side} foot (width {matrix.shape[1]} > 2.5 * height {matrix.shape[0]}). "
                "Expected vertical plantar thermogram orientation."
            )

        # Reject uncalibrated intensity values (> 50.0°C)
        if np.max(matrix) > 50.0:
            raise ValueError(
                f"Uncalibrated image values detected in {side} foot (max={np.max(matrix):.1f} > 50.0°C). "
                "Human skin temperatures must be in Celsius float format (typically 18°C - 38°C)."
            )

        # Check for sensor dropouts / NaNs
        nan_count = int(np.isnan(matrix).sum())
        if nan_count / matrix.size > 0.30:
            raise ValueError(
                f"Excessive sensor dropout in {side} foot: {nan_count / matrix.size * 100:.1f}% NaNs detected (limit 30%)."
            )

        if nan_count > 0:
            warnings.append(f"Sensor dropout ({nan_count} NaNs) detected in {side} foot; replaced with 0.0 background.")
            matrix = np.nan_to_num(matrix, nan=0.0)

        if np.isinf(matrix).any():
            warnings.append(f"Infinite values detected in {side} foot; clipped to finite bounds.")
            matrix = np.nan_to_num(matrix, posinf=THERMAL_MAX_VALID_TEMP_C, neginf=0.0)

        # Plantar tissue presence check
        plantar_pixels = matrix[matrix >= THERMAL_BACKGROUND_THRESHOLD_C]
        if len(plantar_pixels) < 50:
            raise ValueError(
                f"Insufficient thermal signal in {side} foot: only {len(plantar_pixels)} plantar pixels detected "
                f"above {THERMAL_BACKGROUND_THRESHOLD_C}°C background threshold (minimum 50 required)."
            )

        min_p = float(np.min(plantar_pixels))
        max_p = float(np.max(plantar_pixels))
        if min_p < THERMAL_MIN_VALID_TEMP_C:
            warnings.append(f"Sub-physiological minimum temperature ({min_p:.1f}°C < {THERMAL_MIN_VALID_TEMP_C}°C) in {side} foot.")
        if max_p > THERMAL_MAX_VALID_TEMP_C:
            warnings.append(f"Hyper-physiological maximum temperature ({max_p:.1f}°C > {THERMAL_MAX_VALID_TEMP_C}°C) in {side} foot.")

        return matrix, warnings

    def preprocess_pair(
        self,
        left_input: Union[np.ndarray, List[List[float]], str, Path],
        right_input: Union[np.ndarray, List[List[float]], str, Path]
    ) -> Tuple[torch.Tensor, BilateralThermalMetrics, QualityFlags, str]:
        """
        Preprocesses Left and Right thermal matrices:
        1. Sanitizes inputs and verifies dimensions and Celsius calibration.
        2. Standardizes both to canonical (160, 80) resolution.
        3. Flips Right foot horizontally for anatomical parity.
        4. Calculates bilateral absolute difference: Delta_T = |T_Left - T_Right_flipped|.
        5. Computes asymmetry parameters (mean, max, % > 2.2°C).
        6. Stacks into (1, 3, 160, 80) PyTorch tensor.
        """
        left_raw, left_warns = self._validate_and_sanitize_matrix(left_input, "Left")
        right_raw, right_warns = self._validate_and_sanitize_matrix(right_input, "Right")
        all_warnings = left_warns + right_warns

        input_hash = self._compute_input_hash(left_raw, right_raw)

        # Standardize Left foot
        left_pad, left_mask = resize_and_pad_thermal_matrix(
            left_raw, target_shape=THERMAL_CANONICAL_SHAPE, mask_threshold=THERMAL_BACKGROUND_THRESHOLD_C
        )

        # Standardize Right foot
        right_pad, right_mask = resize_and_pad_thermal_matrix(
            right_raw, target_shape=THERMAL_CANONICAL_SHAPE, mask_threshold=THERMAL_BACKGROUND_THRESHOLD_C
        )

        # Anatomical horizontal flip of Right foot
        right_pad_flipped = np.fliplr(right_pad).copy()
        right_mask_flipped = np.fliplr(right_mask).copy()

        # Bilateral temperature difference
        delta_t = np.abs(left_pad - right_pad_flipped)

        # Intersected plantar mask for anatomical asymmetry metrics
        bilateral_mask = (left_mask > 0) & (right_mask_flipped > 0)

        if np.sum(bilateral_mask) > 0:
            mean_left = float(np.mean(left_pad[left_mask > 0])) if np.sum(left_mask) > 0 else 0.0
            mean_right = float(np.mean(right_pad[right_mask > 0])) if np.sum(right_mask) > 0 else 0.0
            mean_asym = float(np.mean(delta_t[bilateral_mask]))
            max_asym = float(np.max(delta_t[bilateral_mask]))
            # Clinical warning threshold: delta_T > 2.2°C (Armstrong / Frykberg criterion)
            hyperthermic_count = np.sum(delta_t[bilateral_mask] > 2.2)
            hyperthermic_fraction = float(hyperthermic_count / np.sum(bilateral_mask))
        else:
            all_warnings.append("No overlapping plantar mask between Left and Right feet.")
            mean_left = 0.0
            mean_right = 0.0
            mean_asym = 0.0
            max_asym = 0.0
            hyperthermic_fraction = 0.0

        if hyperthermic_fraction > 0.05:
            all_warnings.append(
                f"Significant hyperthermic asymmetry: {hyperthermic_fraction*100:.1f}% of plantar area > 2.2°C."
            )

        metrics = BilateralThermalMetrics(
            mean_left_temp_c=round(mean_left, 2),
            mean_right_temp_c=round(mean_right, 2),
            mean_abs_asymmetry_c=round(mean_asym, 2),
            max_abs_asymmetry_c=round(max_asym, 2),
            hyperthermic_asymmetry_fraction=round(hyperthermic_fraction, 4)
        )

        quality_flags = QualityFlags(
            is_valid=(len(left_warns) == 0 and len(right_warns) == 0),
            warnings=all_warnings
        )

        # Construct 3-channel tensor (1, 3, 160, 80)
        tensor_np = np.stack([left_pad, right_pad_flipped, delta_t], axis=0)
        tensor = torch.from_numpy(tensor_np).unsqueeze(0).float()

        return tensor, metrics, quality_flags, input_hash

    @torch.no_grad()
    def predict(
        self,
        request: Union[ThermalInferenceRequest, Dict[str, Any], Tuple[Any, Any]]
    ) -> ThermalInferenceResponse:
        """
        Executes end-to-end thermal inference.
        Accepts:
            - ThermalInferenceRequest Pydantic object
            - Dict matching the request schema
            - Tuple of (left_input, right_input)
        Returns:
            ThermalInferenceResponse object
        """
        start_time = time.perf_counter()

        # Parse request inputs
        if isinstance(request, tuple) and len(request) == 2:
            left_input, right_input = request
        elif isinstance(request, ThermalInferenceRequest):
            left_input = request.left_foot_array if request.left_foot_array is not None else request.left_foot_csv_path
            right_input = request.right_foot_array if request.right_foot_array is not None else request.right_foot_csv_path
        elif isinstance(request, dict):
            left_input = request.get("left_foot_array") or request.get("left_foot_csv_path")
            right_input = request.get("right_foot_array") or request.get("right_foot_csv_path")
        else:
            raise TypeError(f"Invalid request type: {type(request)}")

        if left_input is None or right_input is None:
            raise ValueError("Both left and right foot inputs must be provided (as arrays or CSV paths).")

        # Preprocess pair
        tensor, metrics, quality, input_hash = self.preprocess_pair(left_input, right_input)
        tensor = tensor.to(self.device)

        # Run forward pass
        logits = self.model(tensor)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

        prob_control = float(probabilities[0])
        prob_diabetic = float(probabilities[1])

        # Apply calibrated decision threshold
        predicted_class_id = 1 if prob_diabetic >= self.decision_threshold else 0
        predicted_label = "Diabetic Cohort" if predicted_class_id == 1 else "Control Cohort"
        confidence = prob_diabetic if predicted_class_id == 1 else prob_control

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        prediction = ThermalPrediction(
            predicted_class_id=predicted_class_id,
            predicted_label=predicted_label,
            confidence=round(confidence, 4),
            probability_control=round(prob_control, 4),
            probability_diabetic=round(prob_diabetic, 4),
            decision_threshold=self.decision_threshold
        )

        status = "success" if quality.is_valid else ("warning" if len(quality.warnings) > 0 else "success")

        response = ThermalInferenceResponse(
            model_id=self.model_id,
            model_version=self.model_version,
            status=status,
            input_hash=input_hash,
            prediction=prediction,
            bilateral_metrics=metrics,
            quality_flags=quality,
            disclaimer=(
                "RESEARCH USE ONLY: Diabetic cohort screening model. "
                "NOT an ulcer detector or clinical diagnostic device."
            ),
            latency_ms=round(elapsed_ms, 2)
        )

        return response
