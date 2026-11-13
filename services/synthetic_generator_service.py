"""
FootGuard AI - Synthetic Multimodal Data Generator Service
Generates controlled, session-isolated synthetic thermal, 6-site plantar pressure,
and IMU datasets with latent-state supervised targets (never algebraic sums/maxes).
Every generated record is tagged SYNTHETIC_NOT_CLINICAL_GROUND_TRUTH.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import numpy as np
import pandas as pd

from services.config_service import ConfigService


SYNTHETIC_PROVENANCE_TAG = "SYNTHETIC_NOT_CLINICAL_GROUND_TRUTH"


@dataclass
class SyntheticGenerationConfig:
    dataset_name: str = "sim_multimodal_v1"
    num_sessions: int = 12
    steps_per_session: int = 60
    sampling_hz: float = 10.0
    thermal_base_c: float = 30.8
    thermal_spread_c: float = 1.2
    pressure_load_scale_n: float = 400.0
    sensor_noise_std: float = 0.05
    measurement_variation: float = 0.15
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SyntheticGeneratorService:
    """
    Simulates paired thermal, 6-site plantar pressure, and IMU sessions.
    Supervised labels are derived from hidden latent simulator states rather than
    algebraic transformations of the observed input columns.
    """

    THERMAL_FEATURE_COLS = [
        "thermal_mean_c",
        "thermal_min_c",
        "thermal_max_c",
        "thermal_std_c",
        "heel_temp_mean_c",
        "mid_temp_mean_c",
        "mth1_temp_mean_c",
        "mth3_temp_mean_c",
        "mth5_temp_mean_c",
        "toe_hallux_temp_mean_c",
        "thermal_heel_zone_mean_c",
        "thermal_midfoot_zone_mean_c",
        "thermal_forefoot_zone_mean_c",
        "bilateral_delta_t_c",
    ]

    PRESSURE_FEATURE_COLS = [
        "heel_pressure_n",
        "mid_pressure_n",
        "mth1_pressure_n",
        "mth3_pressure_n",
        "mth5_pressure_n",
        "toe_hallux_pressure_n",
    ]

    IMU_FEATURE_COLS = [
        "mpu6050_accel_x_g",
        "mpu6050_accel_y_g",
        "mpu6050_accel_z_g",
        "mpu6050_gyro_x_dps",
        "mpu6050_gyro_y_dps",
        "mpu6050_gyro_z_dps",
    ]

    LATENT_TARGET_COLS = [
        "latent_thermal_regime",
        "latent_gait_speed_kmh",
    ]

    def __init__(self, config_service: Optional[ConfigService] = None):
        self.cfg = config_service or ConfigService()
        self.synthetic_dir = self.cfg.get_path("data_synthetic_dir")
        self.synthetic_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _build_bilateral_thermal_map(
        rng: np.random.Generator,
        base_temp_c: float,
        asymmetry_c: float,
        noise_std: float,
    ) -> np.ndarray:
        """
        Generates a synthetic (3, 160, 80) tensor [Left, Right_flipped, Delta_T]
        matching Thermal2DCNN's calibrated input contract.
        """
        y_grid = np.linspace(-1.0, 1.0, 160, dtype=np.float32)[:, None]
        x_grid = np.linspace(-1.0, 1.0, 80, dtype=np.float32)[None, :]
        # Foot-shaped Gaussian envelope
        foot_mask = np.exp(-((x_grid / 0.65) ** 2 + (y_grid / 0.85) ** 2)).astype(np.float32)

        left = base_temp_c - 1.5 * (1.0 - foot_mask) + rng.normal(0.0, max(0.05, noise_std), (160, 80)).astype(np.float32)
        right = base_temp_c - 1.5 * (1.0 - foot_mask) + rng.normal(0.0, max(0.05, noise_std), (160, 80)).astype(np.float32)

        # Inject localized forefoot/heel thermal asymmetry when asymmetry_c > 0
        if abs(asymmetry_c) > 1e-4:
            hotspot = np.exp(-(((x_grid - 0.1) / 0.35) ** 2 + ((y_grid + 0.35) / 0.35) ** 2)).astype(np.float32)
            left = left + asymmetry_c * hotspot

        delta_t = left - right
        return np.stack([left, right, delta_t], axis=0).astype(np.float32)

    @staticmethod
    def _build_stance_vgrf_curve(
        rng: np.random.Generator,
        gait_speed_kmh: float,
        load_scale_n: float,
        noise_std: float,
    ) -> np.ndarray:
        """
        Generates a 100-point normalized bimodal stance vGRF curve (100,)
        driven by latent walking speed (higher speed -> sharper twin peaks & deeper mid-stance trough).
        """
        t = np.linspace(0.0, 1.0, 100, dtype=np.float32)
        speed_factor = np.clip((gait_speed_kmh - 2.5) / 4.0, 0.0, 1.0)
        peak1_amp = load_scale_n * (0.95 + 0.35 * speed_factor)
        peak2_amp = load_scale_n * (0.90 + 0.30 * speed_factor)
        trough_depth = 0.25 + 0.20 * speed_factor

        p1 = peak1_amp * np.exp(-((t - 0.24) / 0.11) ** 2)
        p2 = peak2_amp * np.exp(-((t - 0.76) / 0.11) ** 2)
        mid = -trough_depth * load_scale_n * np.exp(-((t - 0.50) / 0.12) ** 2)
        envelope = np.maximum(0.0, np.sin(np.pi * t)) ** 0.5
        curve = np.maximum(0.0, (p1 + p2 + mid) * envelope)
        curve += rng.normal(0.0, max(1.0, noise_std * load_scale_n * 0.05), size=100).astype(np.float32)
        return np.maximum(0.0, curve).astype(np.float32)

    def generate(self, config: SyntheticGenerationConfig) -> Dict[str, Any]:
        """
        Generates a complete synthetic multimodal dataset and persists:
        - readings.csv (tabular paired readings)
        - tensors.npz (pre-built thermal (N_sess, 3, 160, 80) and vGRF (N_sess, 100) arrays)
        - manifest.json (generator config, seed, provenance, target semantics)
        """
        if config.num_sessions < 6:
            raise ValueError("num_sessions must be >= 6 to support session-isolated train/val/test splits.")
        if config.steps_per_session < 20:
            raise ValueError("steps_per_session must be >= 20.")
        if config.sampling_hz <= 0:
            raise ValueError("sampling_hz must be > 0.")

        rng = np.random.default_rng(config.seed)
        safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in config.dataset_name).strip("_")
        if not safe_name:
            safe_name = f"sim_{config.seed}"

        out_dir = self.synthetic_dir / safe_name
        out_dir.mkdir(parents=True, exist_ok=True)

        rows: List[Dict[str, Any]] = []
        thermal_tensors: List[np.ndarray] = []
        vgrf_curves: List[np.ndarray] = []
        session_ids_list: List[str] = []
        session_regimes: List[int] = []
        session_speeds: List[float] = []
        session_durations: List[float] = []
        session_peak_forces: List[float] = []

        dt_s = 1.0 / float(config.sampling_hz)
        site_weights = np.array([0.25, 0.12, 0.18, 0.17, 0.12, 0.16], dtype=np.float64)

        for s_idx in range(config.num_sessions):
            sid = f"SIM-P{s_idx + 1:03d}"
            # Latent simulator states (independent of algebraic row sums)
            latent_regime = int(s_idx % 2)  # balanced 0/1 simulated regimes
            latent_speed_kmh = float(np.clip(rng.uniform(2.8, 6.2) + rng.normal(0.0, 0.15), 2.2, 6.8))
            session_temp_offset = float(rng.normal(0.0, config.thermal_spread_c * 0.5))
            asymmetry_c = float(rng.uniform(1.6, 2.8) if latent_regime == 1 else rng.uniform(0.05, 0.45))
            session_load_n = float(config.pressure_load_scale_n * rng.uniform(0.85, 1.15))

            # Session-level tensor artifacts
            t_map = self._build_bilateral_thermal_map(
                rng,
                base_temp_c=config.thermal_base_c + session_temp_offset,
                asymmetry_c=asymmetry_c,
                noise_std=config.sensor_noise_std,
            )
            vgrf = self._build_stance_vgrf_curve(
                rng,
                gait_speed_kmh=latent_speed_kmh,
                load_scale_n=session_load_n,
                noise_std=config.sensor_noise_std,
            )
            stance_duration_s = float(np.clip(0.85 - 0.06 * latent_speed_kmh + rng.normal(0.0, 0.02), 0.35, 0.95))

            thermal_tensors.append(t_map)
            vgrf_curves.append(vgrf)
            session_ids_list.append(sid)
            session_regimes.append(latent_regime)
            session_speeds.append(latent_speed_kmh)
            session_durations.append(stance_duration_s)
            session_peak_forces.append(float(np.max(vgrf)))

            # Time-series rows for this session
            cadence_hz = 0.6 + 0.18 * latent_speed_kmh
            for step in range(config.steps_per_session):
                t_sec = step * dt_s
                phase = 2.0 * np.pi * cadence_hz * t_sec

                # Simulated temperatures with subtle spatial gradient + noise
                base_t = config.thermal_base_c + session_temp_offset + 0.15 * np.sin(0.1 * t_sec)
                site_offsets = np.array([0.0, -0.15, 0.12, 0.22, -0.05, -0.02])
                if latent_regime == 1:
                    site_offsets = site_offsets + np.array([0.3, 0.1, 0.8, 0.7, 0.2, 0.4])
                site_temps = base_t + site_offsets + rng.normal(0.0, config.sensor_noise_std + 0.05 * config.measurement_variation, 6)

                # Simulated 6-site plantar pressures driven by gait phase and latent speed
                gait_wave = 0.55 + 0.45 * np.sin(phase)
                heel_phase = max(0.1, 0.5 + 0.5 * np.cos(phase))
                forefoot_phase = max(0.1, 0.5 - 0.5 * np.cos(phase))
                phase_mod = np.array([heel_phase, 0.7, forefoot_phase, forefoot_phase, 0.85 * forefoot_phase, 1.05 * forefoot_phase])
                raw_pressures = session_load_n * site_weights * gait_wave * phase_mod * (1.0 + 0.05 * (latent_speed_kmh - 4.0))
                site_pressures = np.maximum(
                    0.0,
                    raw_pressures + rng.normal(0.0, config.sensor_noise_std * 15.0 + config.measurement_variation * 8.0, 6),
                )

                total_load = float(np.sum(site_pressures))
                peak_idx = int(np.argmax(site_pressures))
                site_names = ["heel", "mid", "mth1", "mth3", "mth5", "toe_hallux"]

                # Simulated IMU signals driven by latent walking speed & cadence
                amp = 0.12 * latent_speed_kmh
                ax = float(amp * np.sin(phase) + rng.normal(0.0, config.sensor_noise_std))
                ay = float(0.7 * amp * np.cos(phase) + rng.normal(0.0, config.sensor_noise_std))
                az = float(1.0 + 0.18 * amp * np.sin(2.0 * phase) + rng.normal(0.0, config.sensor_noise_std))
                gx = float(12.0 * latent_speed_kmh * np.cos(phase) + rng.normal(0.0, config.sensor_noise_std * 10.0))
                gy = float(9.5 * latent_speed_kmh * np.sin(phase) + rng.normal(0.0, config.sensor_noise_std * 10.0))
                gz = float(6.0 * latent_speed_kmh * np.sin(2.0 * phase) + rng.normal(0.0, config.sensor_noise_std * 10.0))

                rows.append({
                    "reading_id": f"{sid}-R{step + 1:04d}",
                    "session_id": sid,
                    "data_status": SYNTHETIC_PROVENANCE_TAG,
                    "sample_interval_s": round(dt_s, 4),
                    "thermal_mean_c": round(float(np.mean(site_temps)), 4),
                    "thermal_min_c": round(float(np.min(site_temps) - 0.8), 4),
                    "thermal_max_c": round(float(np.max(site_temps) + 0.6), 4),
                    "thermal_std_c": round(float(np.std(site_temps) + 0.35), 4),
                    "heel_temp_mean_c": round(float(site_temps[0]), 4),
                    "mid_temp_mean_c": round(float(site_temps[1]), 4),
                    "mth1_temp_mean_c": round(float(site_temps[2]), 4),
                    "mth3_temp_mean_c": round(float(site_temps[3]), 4),
                    "mth5_temp_mean_c": round(float(site_temps[4]), 4),
                    "toe_hallux_temp_mean_c": round(float(site_temps[5]), 4),
                    "thermal_heel_zone_mean_c": round(float(site_temps[0]), 4),
                    "thermal_midfoot_zone_mean_c": round(float(site_temps[1]), 4),
                    "thermal_forefoot_zone_mean_c": round(float(np.mean(site_temps[2:])), 4),
                    "bilateral_delta_t_c": round(asymmetry_c + float(rng.normal(0.0, config.sensor_noise_std)), 4),
                    "heel_pressure_n": round(float(site_pressures[0]), 3),
                    "mid_pressure_n": round(float(site_pressures[1]), 3),
                    "mth1_pressure_n": round(float(site_pressures[2]), 3),
                    "mth3_pressure_n": round(float(site_pressures[3]), 3),
                    "mth5_pressure_n": round(float(site_pressures[4]), 3),
                    "toe_hallux_pressure_n": round(float(site_pressures[5]), 3),
                    "plantar_total_load_n": round(total_load, 3),
                    "peak_site_pressure_n": round(float(site_pressures[peak_idx]), 3),
                    "peak_pressure_site": site_names[peak_idx],
                    "mpu6050_accel_x_g": round(ax, 4),
                    "mpu6050_accel_y_g": round(ay, 4),
                    "mpu6050_accel_z_g": round(az, 4),
                    "mpu6050_gyro_x_dps": round(gx, 4),
                    "mpu6050_gyro_y_dps": round(gy, 4),
                    "mpu6050_gyro_z_dps": round(gz, 4),
                    # Latent-state supervised targets (NOT algebraic sums of observed inputs)
                    "latent_thermal_regime": latent_regime,
                    "latent_gait_speed_kmh": round(latent_speed_kmh, 4),
                })

        df = pd.DataFrame(rows)
        csv_path = out_dir / "readings.csv"
        df.to_csv(csv_path, index=False)

        npz_path = out_dir / "tensors.npz"
        np.savez_compressed(
            npz_path,
            thermal_tensors=np.stack(thermal_tensors, axis=0),  # (N_sess, 3, 160, 80)
            vgrf_curves=np.stack(vgrf_curves, axis=0),          # (N_sess, 100)
            session_ids=np.array(session_ids_list),
            latent_thermal_regime=np.array(session_regimes, dtype=np.int64),
            latent_gait_speed_kmh=np.array(session_speeds, dtype=np.float32),
            stance_duration_s=np.array(session_durations, dtype=np.float32),
            peak_force_n=np.array(session_peak_forces, dtype=np.float32),
        )

        manifest = {
            "dataset_id": safe_name,
            "data_status": SYNTHETIC_PROVENANCE_TAG,
            "origin": "synthetic",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "generator_config": config.to_dict(),
            "num_rows": int(len(df)),
            "num_sessions": int(config.num_sessions),
            "csv_path": str(csv_path),
            "tensors_path": str(npz_path),
            "feature_groups": {
                "thermal": self.THERMAL_FEATURE_COLS,
                "pressure": self.PRESSURE_FEATURE_COLS,
                "imu": self.IMU_FEATURE_COLS,
            },
            "target_semantics": {
                "latent_thermal_regime": "Simulated binary regime (0=symmetric baseline, 1=localized forefoot thermal asymmetry) from hidden simulator state; NOT clinical ulcer/neuropathy.",
                "latent_gait_speed_kmh": "Simulated continuous walking speed (km/h) driving gait cadence, vGRF stance curve shape, and IMU amplitude; NOT derived from algebraic sums.",
            },
        }
        manifest_path = out_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest

    def list_generated_datasets(self) -> List[Dict[str, Any]]:
        """Lists all locally generated synthetic datasets under data/synthetic/."""
        results: List[Dict[str, Any]] = []
        if not self.synthetic_dir.exists():
            return results
        for manifest_path in sorted(self.synthetic_dir.glob("*/manifest.json"), reverse=True):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and data.get("data_status") == SYNTHETIC_PROVENANCE_TAG:
                    results.append(data)
            except Exception:
                continue
        return results

    def load_generated_dataset(self, dataset_id: str) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, np.ndarray]]:
        """Loads readings.csv, manifest.json, and tensors.npz for a generated dataset."""
        ds_dir = self.synthetic_dir / dataset_id
        manifest_path = ds_dir / "manifest.json"
        csv_path = ds_dir / "readings.csv"
        npz_path = ds_dir / "tensors.npz"
        if not manifest_path.exists() or not csv_path.exists():
            raise FileNotFoundError(f"Generated synthetic dataset '{dataset_id}' not found in {self.synthetic_dir}.")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        df = pd.read_csv(csv_path)
        tensors = dict(np.load(npz_path, allow_pickle=False)) if npz_path.exists() else {}
        return df, manifest, tensors
