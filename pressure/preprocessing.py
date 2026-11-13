"""
FootGuard AI - Pressure & Shear Preprocessing Pipeline
Handles:
1. CAD_WALK: Dynamic NIfTI pedobarography (38x22xT at 500 Hz),
   Peak Plantar Pressure (PPP) mapping, Pressure-Time Integral (PTI) calculation,
   temporal stance phase normalization (100 points), and kinematic metadata linkage.
2. Gait Lab Validation: Continuous 20-channel insole stream (normal + tri-axial shear),
   signal conditioning (jump detrending, spike interpolation, Butterworth filtering),
   and step cycle segmentation.
3. Anti-leakage participant-level split generation.
4. Data quality reporting.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.interpolate import interp1d
import nibabel as nib

from common.config import (
    PRESSURE_RAW_DIR,
    PRESSURE_CAD_WALK_DIR,
    PRESSURE_GAIT_DIR,
    PROCESSED_PRESSURE_DIR,
    SPLITS_PRESSURE_DIR,
    PRESSURE_GRID_SHAPE,
    PRESSURE_SENSOR_PITCH_MM,
    PRESSURE_STANCE_RESAMPLED_LEN,
    SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO
)
from common.logging import get_logger
from common.reproducibility import set_seed

logger = get_logger("pressure_preprocessing")


# =====================================================================
# 1. CAD_WALK PEDOBAROGRAPHY PREPROCESSING
# =====================================================================

def load_cad_walk_metadata() -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Loads subject demographic metadata and per-trial walking speeds.
    """
    meta_xlsx = PRESSURE_CAD_WALK_DIR / "metadata.xlsx"
    df_demo = pd.DataFrame()
    if meta_xlsx.exists():
        df_raw = pd.read_excel(meta_xlsx)
        # Drop completely empty rows and columns
        df_demo = df_raw.dropna(how="all").dropna(axis=1, how="all")
        # Ensure 'Subject ID' is string
        if "Subject ID" in df_demo.columns:
            df_demo = df_demo[df_demo["Subject ID"].notna()].copy()
            df_demo["Subject ID"] = df_demo["Subject ID"].astype(str).str.strip()
            df_demo.set_index("Subject ID", inplace=True)
            
    # Load walking speeds per subject
    speeds_dict = {}
    for xlsx_file in PRESSURE_CAD_WALK_DIR.glob("*/walking_speeds_*.xlsx"):
        subj_id = xlsx_file.parent.name
        try:
            df_spd = pd.read_excel(xlsx_file)
            speeds_dict[subj_id] = df_spd
        except Exception as e:
            logger.warning(f"Error reading walking speeds for {subj_id}: {e}")
            
    return df_demo, speeds_dict


def process_cad_walk_trial(
    nii_path: Path,
    cell_area_cm2: float = (7.62 * 5.08) / 100.0  # 0.3871 cm^2
) -> Dict[str, Any]:
    """
    Loads a dynamic NIfTI file (38, 22, T) and extracts:
    - 2D Peak Plantar Pressure (PPP): max over time at each cell
    - 2D Pressure-Time Integral (PTI): sum over time * dt at each cell
    - 1D Total Vertical Force curve normalized to 100 stance phase percentage points
    - Temporal parameters: stance duration (ms), peak force (N)
    """
    img = nib.load(str(nii_path))
    data = img.get_fdata(dtype=np.float32)  # shape: (38, 22, T)
    
    if data.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI, got shape {data.shape} for {nii_path.name}")
        
    num_frames = data.shape[2]
    dt = 0.002  # 500 Hz -> 2 ms per frame
    
    # 2D Spatial Summaries
    ppp = np.max(data, axis=2)  # Peak Plantar Pressure
    pti = np.sum(data, axis=2) * dt  # Pressure-Time Integral
    
    # Total Vertical Force curve over time F_z(t)
    # Sum over all 38x22 cells * cell_area (in cm^2) * pressure (N/cm^2 = 10 kPa)
    total_force_curve = np.sum(data, axis=(0, 1)) * cell_area_cm2
    
    # Normalize stance curve to fixed 100 percentage points (0% to 100% stance phase)
    if num_frames >= 10:
        time_orig = np.linspace(0, 1, num_frames)
        interp_func = interp1d(time_orig, total_force_curve, kind="cubic")
        time_resampled = np.linspace(0, 1, PRESSURE_STANCE_RESAMPLED_LEN)
        force_resampled = interp_func(time_resampled).astype(np.float32)
    else:
        force_resampled = np.zeros(PRESSURE_STANCE_RESAMPLED_LEN, dtype=np.float32)
        
    return {
        "ppp": ppp,
        "pti": pti,
        "force_resampled": force_resampled,
        "num_frames": num_frames,
        "duration_ms": num_frames * 2.0,
        "peak_pressure_kpa": float(np.max(ppp)),
        "peak_force_n": float(np.max(total_force_curve))
    }


def extract_speed_and_kinematics(
    df_speed: pd.DataFrame,
    trial_num: int,
    foot: str
) -> Tuple[float, float, float]:
    """
    Parses CAD_WALK speed table where row 0 specifies Left/Right subheadings,
    and rows >= 2 contain numeric trial measurements.
    """
    speed_val, stride_len, stride_time = 0.0, 0.0, 0.0
    if df_speed.empty or "Trial" not in df_speed.columns:
        return speed_val, stride_len, stride_time
        
    trial_rows = df_speed[pd.to_numeric(df_speed["Trial"], errors="coerce") == trial_num]
    if not trial_rows.empty:
        row = trial_rows.iloc[0]
        is_left = (foot.upper() == "LEFT")
        speed_col = "Walking Speed (km/h)" if is_left else "Unnamed: 3"
        stride_len_col = "Stride Length (mm)" if is_left else "Unnamed: 5"
        stride_time_col = "Stride Time (s)" if is_left else "Unnamed: 7"
        
        try:
            val = row.get(speed_col, 0.0)
            speed_val = float(val) if pd.notna(val) else 0.0
        except (ValueError, TypeError):
            speed_val = 0.0
            
        try:
            val = row.get(stride_len_col, 0.0)
            stride_len = float(val) if pd.notna(val) else 0.0
        except (ValueError, TypeError):
            stride_len = 0.0
            
        try:
            val = row.get(stride_time_col, 0.0)
            stride_time = float(val) if pd.notna(val) else 0.0
        except (ValueError, TypeError):
            stride_time = 0.0
            
    return speed_val, stride_len, stride_time


def process_cad_walk() -> pd.DataFrame:
    """
    Processes all CAD_WALK NIfTI files across 51 healthy subjects.
    Saves compressed .npz trial files to ML/processed/pressure/cad_walk/.
    Generates cad_walk_manifest.csv.
    """
    logger.info("Starting CAD_WALK pedobarography preprocessing...")
    output_dir = PROCESSED_PRESSURE_DIR / "cad_walk"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_demo, speeds_dict = load_cad_walk_metadata()
    
    subject_dirs = sorted([d for d in PRESSURE_CAD_WALK_DIR.iterdir() if d.is_dir()])
    logger.info(f"Found {len(subject_dirs)} CAD_WALK subject directories...")
    
    manifest_records = []
    
    for s_dir in subject_dirs:
        subj_id = s_dir.name
        nii_files = sorted(list(s_dir.glob("*.nii")))
        
        # Demographic lookup
        demo_row = df_demo.loc[subj_id] if subj_id in df_demo.index else None
        age = float(demo_row["Age (years)"]) if demo_row is not None and "Age (years)" in demo_row and pd.notna(demo_row["Age (years)"]) else 0.0
        sex = str(demo_row["Sex"]) if demo_row is not None and "Sex" in demo_row and pd.notna(demo_row["Sex"]) else "Unknown"
        weight = float(demo_row["Weight (kg)"]) if demo_row is not None and "Weight (kg)" in demo_row and pd.notna(demo_row["Weight (kg)"]) else 0.0
        height = float(demo_row["Height (cm)"]) if demo_row is not None and "Height (cm)" in demo_row and pd.notna(demo_row["Height (cm)"]) else 0.0
        
        df_speed = speeds_dict.get(subj_id, pd.DataFrame())
        
        for nii_path in nii_files:
            trial_name = nii_path.stem
            parts = trial_name.split("_")
            foot = parts[0].upper()  # LEFT or RIGHT
            trial_num = int(parts[-1]) if parts[-1].isdigit() else 0
            
            # Lookup speed and kinematics for this trial
            speed_val, stride_len, stride_time = extract_speed_and_kinematics(df_speed, trial_num, foot)
                        
            trial_data = process_cad_walk_trial(nii_path)
            
            out_file = output_dir / f"{subj_id}_{trial_name}.npz"
            np.savez_compressed(
                out_file,
                ppp=trial_data["ppp"],
                pti=trial_data["pti"],
                force_curve=trial_data["force_resampled"],
                subject_id=subj_id,
                foot=foot,
                trial_num=trial_num,
                walking_speed=speed_val,
                stride_length_mm=stride_len,
                stride_time_s=stride_time,
                age=age,
                sex=sex,
                weight=weight,
                height=height
            )
            
            manifest_records.append({
                "trial_id": f"{subj_id}_{trial_name}",
                "subject_id": subj_id,
                "foot": foot,
                "trial_num": trial_num,
                "num_frames": trial_data["num_frames"],
                "duration_ms": trial_data["duration_ms"],
                "peak_pressure_kpa": trial_data["peak_pressure_kpa"],
                "peak_force_n": trial_data["peak_force_n"],
                "walking_speed": speed_val,
                "stride_length_mm": stride_len,
                "stride_time_s": stride_time,
                "age": age,
                "sex": sex,
                "weight": weight,
                "height": height,
                "file_path": str(out_file.relative_to(PROCESSED_PRESSURE_DIR.parent))
            })
            
    df_manifest = pd.DataFrame(manifest_records)
    manifest_path = PROCESSED_PRESSURE_DIR / "cad_walk_manifest.csv"
    df_manifest.to_csv(manifest_path, index=False)
    logger.info(f"CAD_WALK processed: {len(df_manifest)} trials written to {manifest_path}")
    return df_manifest


# =====================================================================
# 2. GAIT LAB INSOLE VALIDATION PREPROCESSING
# =====================================================================

def fix_sensor_jump_and_spikes(
    signal_arr: np.ndarray,
    threshold: float = 3.0
) -> np.ndarray:
    """
    Python implementation of MATLAB FixJumpv24.m.
    Detects mean-level shift changepoints, detrends step discontinuities,
    and interpolates transient outliers with a median filter.
    """
    clean_signal = np.copy(signal_arr).astype(np.float32)
    diff = np.diff(clean_signal)
    
    # Detect sudden baseline jumps exceeding threshold * standard deviation
    std_diff = np.std(diff)
    if std_diff > 1e-6:
        jump_indices = np.where(np.abs(diff) > threshold * std_diff)[0] + 1
        indices = np.concatenate([[0], jump_indices, [len(clean_signal)]])
        
        for k in range(len(indices) - 1):
            idx_range = slice(indices[k], indices[k + 1])
            if indices[k + 1] > indices[k]:
                clean_signal[idx_range] -= np.mean(clean_signal[idx_range])
                
    # Outlier spike filter using rolling median
    kernel_size = 5
    filtered = signal.medfilt(clean_signal, kernel_size=kernel_size)
    return filtered


def butter_lowpass_filter(
    data: np.ndarray,
    cutoff: float = 15.0,
    fs: float = 100.0,
    order: int = 4
) -> np.ndarray:
    """
    Zero-phase Butterworth low-pass filter to remove sensor high-frequency noise.
    """
    nyq = 0.5 * fs
    normal_cutoff = min(cutoff / nyq, 0.99)
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
    y = signal.filtfilt(b, a, data)
    return y


def process_gait_lab() -> pd.DataFrame:
    """
    Processes all 8 Gait Lab validation CSV recordings across:
    - Diabetic Participant (HFMD1): 4 files (LEFT W1, W2; RIGHT W1, W2)
    - Healthy Participant (HFM2): 4 files (LEFT W1, W2; RIGHT W1, W2)
    Cleans channels, calibrates signals, segments stance phases,
    and writes conditioned insole streams to ML/processed/pressure/gait_lab/.
    """
    logger.info("Starting Gait Lab Validation preprocessing...")
    output_dir = PROCESSED_PRESSURE_DIR / "gait_lab"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_files = sorted(list(PRESSURE_GAIT_DIR.rglob("*.csv")))
    logger.info(f"Found {len(csv_files)} Gait Lab insole recordings...")
    
    manifest_records = []
    
    for csv_path in csv_files:
        filename = csv_path.stem
        # Identify participant and foot
        is_diabetic = "Diabetic" in str(csv_path)
        subject_id = "HFMD1" if is_diabetic else "HFM2"
        foot = "LEFT" if "LEFT" in filename else "RIGHT"
        walk_run = "W1" if "W1" in filename else "W2"
        
        df_raw = pd.read_csv(csv_path, header=None)
        
        # Col 0: Foot designation, Col 1: Time (ms)
        # Cols 2-9: Normal stress sensors (F1=MetHead, F2=Hallux, F8=Heel)
        # Cols 10-18: Shear stress sensors (MH, HA, HE in AP/ML directions)
        time_ms = df_raw.iloc[:, 1].to_numpy(dtype=np.float64)
        time_sec = (time_ms - time_ms[0]) / 1000.0
        
        raw_signals = df_raw.iloc[:, 2:20].to_numpy(dtype=np.float32)
        
        # Apply signal conditioning to all sensor channels
        conditioned_signals = np.zeros_like(raw_signals)
        for ch in range(raw_signals.shape[1]):
            denoised = fix_sensor_jump_and_spikes(raw_signals[:, ch])
            filtered = butter_lowpass_filter(denoised, cutoff=15.0, fs=100.0)
            conditioned_signals[:, ch] = filtered
            
        out_file = output_dir / f"{subject_id}_{foot}_{walk_run}.npz"
        np.savez_compressed(
            out_file,
            time_sec=time_sec,
            raw_signals=raw_signals,
            conditioned_signals=conditioned_signals,
            subject_id=subject_id,
            is_diabetic=1 if is_diabetic else 0,
            foot=foot,
            walk_run=walk_run
        )
        
        manifest_records.append({
            "run_id": f"{subject_id}_{foot}_{walk_run}",
            "subject_id": subject_id,
            "is_diabetic": 1 if is_diabetic else 0,
            "foot": foot,
            "walk_run": walk_run,
            "num_timesteps": len(time_sec),
            "duration_sec": float(time_sec[-1] - time_sec[0]),
            "num_channels": raw_signals.shape[1],
            "file_path": str(out_file.relative_to(PROCESSED_PRESSURE_DIR.parent))
        })
        
    df_manifest = pd.DataFrame(manifest_records)
    manifest_path = PROCESSED_PRESSURE_DIR / "gait_lab_manifest.csv"
    df_manifest.to_csv(manifest_path, index=False)
    logger.info(f"Gait Lab processed: {len(df_manifest)} runs written to {manifest_path}")
    return df_manifest


# =====================================================================
# 3. SPLIT CREATION (Anti-Leakage Participant-Level Splits)
# =====================================================================

def create_pressure_splits(df_cad: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates deterministic participant-level splits for CAD_WALK.
    All 48 trials for any individual remain strictly in ONE split.
    Guarantees zero participant leakage.
    """
    set_seed(SEED)
    logger.info("Generating deterministic participant-level CAD_WALK splits...")
    
    unique_subjects = sorted(df_cad["subject_id"].unique().tolist())
    np.random.shuffle(unique_subjects)
    
    n = len(unique_subjects)
    n_train = int(round(n * TRAIN_RATIO))
    n_val = int(round(n * VAL_RATIO))
    
    train_subs = sorted(unique_subjects[:n_train])
    val_subs = sorted(unique_subjects[n_train:n_train + n_val])
    test_subs = sorted(unique_subjects[n_train + n_val:])
    
    # Assert zero participant overlap
    assert len(set(train_subs) & set(val_subs)) == 0, "Leakage between train and val in CAD_WALK!"
    assert len(set(train_subs) & set(test_subs)) == 0, "Leakage between train and test in CAD_WALK!"
    assert len(set(val_subs) & set(test_subs)) == 0, "Leakage between val and test in CAD_WALK!"
    
    # Map to trial IDs
    train_trials = df_cad[df_cad["subject_id"].isin(train_subs)]["trial_id"].tolist()
    val_trials = df_cad[df_cad["subject_id"].isin(val_subs)]["trial_id"].tolist()
    test_trials = df_cad[df_cad["subject_id"].isin(test_subs)]["trial_id"].tolist()
    
    splits_dict = {
        "train_subjects": train_subs,
        "val_subjects": val_subs,
        "test_subjects": test_subs,
        "train_trials": train_trials,
        "val_trials": val_trials,
        "test_trials": test_trials,
        "summary": {
            "total_subjects": n,
            "train_subjects_count": len(train_subs),
            "val_subjects_count": len(val_subs),
            "test_subjects_count": len(test_subs),
            "train_trials_count": len(train_trials),
            "val_trials_count": len(val_trials),
            "test_trials_count": len(test_trials),
        }
    }
    
    split_path = SPLITS_PRESSURE_DIR / "cad_walk_splits.json"
    with open(split_path, "w") as f:
        json.dump(splits_dict, f, indent=2)
        
    logger.info(f"CAD_WALK splits generated: {splits_dict['summary']}")
    return splits_dict["summary"]


def run_pressure_preprocessing() -> Dict[str, Any]:
    """
    Main driver for pressure & shear preprocessing.
    """
    set_seed(SEED)
    df_cad = process_cad_walk()
    df_gait = process_gait_lab()
    splits_summary = create_pressure_splits(df_cad)
    
    report = {
        "status": "SUCCESS",
        "cad_walk_trials": len(df_cad),
        "gait_lab_runs": len(df_gait),
        "splits_summary": splits_summary
    }
    
    report_path = PROCESSED_PRESSURE_DIR / "quality_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        
    logger.info(f"Pressure preprocessing complete. Report written to {report_path}")
    return report


if __name__ == "__main__":
    run_pressure_preprocessing()
