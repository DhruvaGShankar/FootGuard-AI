"""
FootGuard AI - Inference Orchestration Service
Delegates directly to ThermalInferenceEngine and PressureInferenceEngine.
Zero duplicated preprocessing or model forward-pass logic.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List
import numpy as np

from services.config_service import ConfigService
from thermal.inference import ThermalInferenceEngine
from pressure.inference import PressureInferenceEngine
from common.schemas import (
    ThermalInferenceRequest,
    ThermalInferenceResponse,
    PressureInferenceRequest,
    PressureInferenceResponse,
)


class InferenceService:
    """Thin service facade over existing Thermal and Pressure inference engines."""

    def __init__(self, config_service: Optional[ConfigService] = None):
        self.cfg = config_service or ConfigService()
        self._thermal_cache: Dict[str, ThermalInferenceEngine] = {}
        self._pressure_cache: Dict[str, PressureInferenceEngine] = {}

    def get_thermal_engine(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        decision_threshold: float = 0.50,
        device: str = "cpu",
    ) -> ThermalInferenceEngine:
        key = f"{checkpoint_path or 'default'}::{decision_threshold:.4f}::{device}"
        if key not in self._thermal_cache:
            self._thermal_cache[key] = ThermalInferenceEngine(
                checkpoint_path=checkpoint_path,
                device=device,
                decision_threshold=decision_threshold,
            )
        return self._thermal_cache[key]

    def get_pressure_engine(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        ridge_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
    ) -> PressureInferenceEngine:
        key = f"{checkpoint_path or 'default'}::{ridge_path or 'default'}::{device}"
        if key not in self._pressure_cache:
            self._pressure_cache[key] = PressureInferenceEngine(
                checkpoint_path=checkpoint_path,
                ridge_path=ridge_path,
                device=device,
            )
        return self._pressure_cache[key]

    def run_thermal_inference(
        self,
        payload: Union[ThermalInferenceRequest, Dict[str, Any], Tuple[Any, Any]],
        checkpoint_path: Optional[Union[str, Path]] = None,
        decision_threshold: float = 0.50,
        data_origin: str = "real",
    ) -> Dict[str, Any]:
        engine = self.get_thermal_engine(checkpoint_path=checkpoint_path, decision_threshold=decision_threshold)
        response: ThermalInferenceResponse = engine.predict(payload)
        result = response.model_dump()
        result["data_origin"] = data_origin
        result["softmax_calibration_note"] = (
            "Raw softmax probability (uncalibrated). Represents relative cohort score, not clinical risk."
        )
        return result

    def run_pressure_inference(
        self,
        payload: Union[PressureInferenceRequest, Dict[str, Any], np.ndarray, List[float]],
        checkpoint_path: Optional[Union[str, Path]] = None,
        ridge_path: Optional[Union[str, Path]] = None,
        data_origin: str = "real",
    ) -> Dict[str, Any]:
        engine = self.get_pressure_engine(checkpoint_path=checkpoint_path, ridge_path=ridge_path)
        response: PressureInferenceResponse = engine.predict(payload)
        result = response.model_dump()
        result["data_origin"] = data_origin
        return result

