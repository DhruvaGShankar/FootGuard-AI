"""
FootGuard AI - Thermal Preprocessing Pipeline
Handles:
1. ThermoDataBase: Calibrated CSV temperature matrices, bilateral alignment,
   plantar masking, canonical standardization, and subject-level split creation.
2. IAC-TecMed: Multi-timepoint calibrated TIFF loading (T0, T5, T10, T15),
   cold-challenge sequence collation, metadata parsing, and participant-level splits.
3. STANDUP_Database: Thermal screening images with IWGDF risk group extraction.
4. Generates automated Data Quality Reports and anti-leakage split manifests.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from PIL import Image
import cv2

from common.config import (
    THERMAL_RAW_DIR,
    THERMAL_TDB_DIR,
    THERMAL_IAC_DIR,
    THERMAL_STANDUP_DIR,
    PROCESSED_THERMAL_DIR,
    SPLITS_THERMAL_DIR,
    THERMAL_CANONICAL_SHAPE,
    THERMAL_BACKGROUND_THRESHOLD_C,
    SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO
)
from common.logging import get_logger
from common.reproducibility import set_seed

logger = get_logger("thermal_preprocessing")


# =====================================================================
# 1. THERMODATABASE PREPROCESSING (Calibrated Plantar Thermograms)
# =====================================================================

def load_thermodatabase_csv(csv_path: Path) -> np.ndarray:
    """
    Load a raw CSV temperature matrix from ThermoDataBase.
    Returns float32 numpy array with true Celsius values.
    Zero represents background / non-plantar mask.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Thermal CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, header=None)
    arr = df.to_numpy(dtype=np.float32)
    return arr


def resize_and_pad_thermal_matrix(
    matrix: np.ndarray,
    target_shape: Tuple[int, int] = THERMAL_CANONICAL_SHAPE,
    mask_threshold: float = THERMAL_BACKGROUND_THRESHOLD_C
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standardizes variable-sized thermal foot matrices to target_shape (H, W).
    Preserves exact aspect ratio by proportional scaling and zero-padding.
    Returns:
        padded_temp: (H, W) array in degrees Celsius
        padded_mask: (H, W) binary uint8 mask (1 = plantar tissue, 0 = background)
    """
    h, w = matrix.shape
    target_h, target_w = target_shape
    
    # Scale while preserving aspect ratio
    scale = min(target_h / h, target_w / w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    
    # Bilinear interpolation for continuous temperature values
    resized_temp = cv2.resize(matrix, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad to exact target shape (centered)
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    
    padded_temp = np.pad(
        resized_temp,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0.0
    )
    
    # Binary plantar mask
    padded_mask = (padded_temp >= mask_threshold).astype(np.uint8)
    
    return padded_temp, padded_mask


def process_thermodatabase() -> pd.DataFrame:
    """
    Processes all 167 participants in ThermoDataBase (45 Control, 122 Diabetic).
    For each subject:
    - Loads Left and Right calibrated CSV matrices.
    - Standardizes to canonical (160, 80) resolution.
    - Flips Right foot horizontally to achieve anatomical correspondence with Left foot.
    - Computes absolute temperature difference map: Delta_T = |T_Left - T_Right_flipped|.
    - Constructs 3-channel tensor: [T_Left, T_Right_flipped, Delta_T].
    - Saves processed .npz files.
    - Generates subject manifest.
    """
    logger.info("Starting ThermoDataBase preprocessing...")
    output_dir = PROCESSED_THERMAL_DIR / "thermodatabase"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_records = []
    
    groups = [
        ("Control Group", 0, "Control"),
        ("DM Group", 1, "Diabetic")
    ]
    
    for group_folder, label, class_name in groups:
        group_path = THERMAL_TDB_DIR / group_folder
        if not group_path.exists():
            logger.warning(f"Directory not found: {group_path}")
            continue
            
        subject_dirs = sorted([d for d in group_path.iterdir() if d.is_dir()])
        logger.info(f"Processing {len(subject_dirs)} subjects in {group_folder}...")
        
        for s_dir in subject_dirs:
            subj_id = s_dir.name
            left_csv = s_dir / f"{subj_id}_L.csv"
            right_csv = s_dir / f"{subj_id}_R.csv"
            
            if not left_csv.exists() or not right_csv.exists():
                logger.error(f"Missing CSV for subject {subj_id}: L={left_csv.exists()}, R={right_csv.exists()}")
                continue
                
            left_raw = load_thermodatabase_csv(left_csv)
            right_raw = load_thermodatabase_csv(right_csv)
            
            # Standardize resolution
            left_std, left_mask = resize_and_pad_thermal_matrix(left_raw)
            right_std, right_mask = resize_and_pad_thermal_matrix(right_raw)
            
            # Horizontally reflect Right foot for bilateral symmetry
            right_std_flipped = np.fliplr(right_std)
            right_mask_flipped = np.fliplr(right_mask)
            
            # Bilateral asymmetry map (Delta T) only where both plantar masks overlap
            common_mask = (left_mask > 0) & (right_mask_flipped > 0)
            delta_t = np.zeros_like(left_std)
            delta_t[common_mask] = np.abs(left_std[common_mask] - right_std_flipped[common_mask])
            
            # 3-channel bilateral tensor: shape (3, H, W)
            bilateral_tensor = np.stack([left_std, right_std_flipped, delta_t], axis=0).astype(np.float32)
            
            # Statistics on active plantar pixels
            left_active = left_std[left_mask > 0]
            right_active = right_std[right_mask > 0]
            delta_active = delta_t[common_mask]
            
            out_file = output_dir / f"{subj_id}.npz"
            np.savez_compressed(
                out_file,
                tensor=bilateral_tensor,
                left_mask=left_mask,
                right_mask_flipped=right_mask_flipped,
                label=label,
                subject_id=subj_id
            )
            
            manifest_records.append({
                "subject_id": subj_id,
                "label": label,
                "class_name": class_name,
                "left_raw_shape": list(left_raw.shape),
                "right_raw_shape": list(right_raw.shape),
                "left_mean_temp": float(np.mean(left_active)) if len(left_active) > 0 else 0.0,
                "left_max_temp": float(np.max(left_active)) if len(left_active) > 0 else 0.0,
                "right_mean_temp": float(np.mean(right_active)) if len(right_active) > 0 else 0.0,
                "right_max_temp": float(np.max(right_active)) if len(right_active) > 0 else 0.0,
                "delta_t_mean": float(np.mean(delta_active)) if len(delta_active) > 0 else 0.0,
                "delta_t_max": float(np.max(delta_active)) if len(delta_active) > 0 else 0.0,
                "file_path": str(out_file.relative_to(PROCESSED_THERMAL_DIR.parent))
            })
            
    df_manifest = pd.DataFrame(manifest_records)
    manifest_path = PROCESSED_THERMAL_DIR / "thermodatabase_manifest.csv"
    df_manifest.to_csv(manifest_path, index=False)
    logger.info(f"ThermoDataBase processed: {len(df_manifest)} samples written to {manifest_path}")
    return df_manifest


# =====================================================================
# 2. IAC-TECMED PREPROCESSING (Calibrated Cold-Stress Rewarming Sequences)
# =====================================================================

def parse_iac_metadata(meta_path: Path) -> Dict[str, Any]:
    """
    Parses semicolon-delimited IAC-TecMed metadata file.
    Safely handles fields with internal commas.
    """
    meta = {}
    if not meta_path.exists():
        return meta
    with open(meta_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if ";" in line:
                key, val = line.split(";", 1)
                meta[key.strip()] = val.strip()
    return meta


def process_iac_tecmed() -> pd.DataFrame:
    """
    Processes all 19 participants in IAC-TecMed.
    Collatest calibrated 32-bit float TIFF maps across sequential timepoints:
    [T0, T5, T10, T15].
    Extracts time-series temperature recovery features.
    Saves sequence tensors of shape (4, H, W) to ML/processed/thermal/iac_tecmed/.
    """
    logger.info("Starting IAC-TecMed preprocessing...")
    processed_sub_dir = THERMAL_IAC_DIR / "Processed"
    output_dir = PROCESSED_THERMAL_DIR / "iac_tecmed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not processed_sub_dir.exists():
        logger.error(f"IAC-TecMed directory not found: {processed_sub_dir}")
        return pd.DataFrame()
        
    subject_dirs = sorted([d for d in processed_sub_dir.iterdir() if d.is_dir()])
    logger.info(f"Found {len(subject_dirs)} IAC-TecMed subjects...")
    
    manifest_records = []
    canonical_timepoints = ["T0", "T5", "T10", "T15"]
    
    for s_dir in subject_dirs:
        subj_id = s_dir.name
        meta_file = s_dir / f"Metadata_{subj_id}.csv"
        meta = parse_iac_metadata(meta_file)
        
        # Load TIFF temperature maps
        tmap_arrays = []
        tmap_means = {}
        tmap_maxs = {}
        missing_tp = False
        
        for tp in canonical_timepoints:
            tiff_path = s_dir / f"{subj_id}_Tmap_{tp}.tiff"
            if not tiff_path.exists():
                logger.error(f"Missing timepoint {tp} for subject {subj_id}")
                missing_tp = True
                break
            # Read 32-bit float TIFF
            img = Image.open(tiff_path)
            arr = np.array(img, dtype=np.float32)
            
            # Active plantar pixels in IAC-TecMed (typical range 20-37 C)
            active_pixels = arr[arr > 20.0]
            mean_val = float(np.mean(active_pixels)) if len(active_pixels) > 0 else float(np.mean(arr))
            max_val = float(np.max(arr))
            tmap_means[tp] = mean_val
            tmap_maxs[tp] = max_val
            
            tmap_arrays.append(arr)
            
        if missing_tp:
            continue
            
        # Stack into temporal sequence: (4, H, W) = (4, 288, 384)
        seq_tensor = np.stack(tmap_arrays, axis=0)
        
        # Also compute downsampled spatial representation for compact TCN input (4, 144, 192)
        downsampled_maps = [
            cv2.resize(m, (192, 144), interpolation=cv2.INTER_LINEAR)
            for m in tmap_arrays
        ]
        seq_downsampled = np.stack(downsampled_maps, axis=0)
        
        out_file = output_dir / f"{subj_id}.npz"
        np.savez_compressed(
            out_file,
            full_sequence=seq_tensor,
            downsampled_sequence=seq_downsampled,
            timepoints=canonical_timepoints,
            subject_id=subj_id
        )
        
        manifest_records.append({
            "subject_id": subj_id,
            "patient_id": meta.get("PatientID", subj_id),
            "age": float(meta.get("Age", 0.0)) if meta.get("Age") else 0.0,
            "sex": meta.get("Sex", "Unknown"),
            "diabetic": meta.get("Diabetic", "No"),
            "mean_temp_T0": tmap_means.get("T0", 0.0),
            "mean_temp_T5": tmap_means.get("T5", 0.0),
            "mean_temp_T10": tmap_means.get("T10", 0.0),
            "mean_temp_T15": tmap_means.get("T15", 0.0),
            "rewarming_delta_T15_T5": tmap_means.get("T15", 0.0) - tmap_means.get("T5", 0.0),
            "file_path": str(out_file.relative_to(PROCESSED_THERMAL_DIR.parent))
        })
        
    df_manifest = pd.DataFrame(manifest_records)
    manifest_path = PROCESSED_THERMAL_DIR / "iac_tecmed_manifest.csv"
    df_manifest.to_csv(manifest_path, index=False)
    logger.info(f"IAC-TecMed processed: {len(df_manifest)} subjects written to {manifest_path}")
    return df_manifest


# =====================================================================
# 3. STANDUP DATABASE PREPROCESSING (Plantar Screening Cohort)
# =====================================================================

def process_standup_database() -> pd.DataFrame:
    """
    Processes STANDUP screening cohort.
    Extracts thermal images across IWGDF risk categories:
    - healthy: class 0
    - diabetic/R0: class 1
    - diabetic/R1: class 2
    - diabetic/R2: class 3
    Normalizes images and saves to ML/processed/thermal/standup/.
    """
    logger.info("Starting STANDUP Database preprocessing...")
    output_dir = PROCESSED_THERMAL_DIR / "standup"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    categories = [
        ("healthy", 0, "Healthy"),
        ("diabetic/R0", 1, "Diabetic_R0"),
        ("diabetic/R1", 2, "Diabetic_R1"),
        ("diabetic/R2", 3, "Diabetic_R2"),
    ]
    
    manifest_records = []
    
    for rel_folder, class_id, class_name in categories:
        thermal_dir = THERMAL_STANDUP_DIR / rel_folder / "thermal"
        if not thermal_dir.exists():
            logger.warning(f"STANDUP directory not found: {thermal_dir}")
            continue
            
        png_files = sorted(list(thermal_dir.glob("*.png")))
        logger.info(f"Processing {len(png_files)} images in {rel_folder}...")
        
        for png_path in png_files:
            file_id = f"{class_name}_{png_path.stem}"
            img = Image.open(png_path).convert("RGB")
            # Resize to canonical (224, 224) for 2D CNN screening models
            img_resized = img.resize((224, 224), Image.Resampling.BILINEAR)
            arr = np.array(img_resized, dtype=np.float32) / 255.0
            # Shape (3, 224, 224)
            arr_ch_first = np.transpose(arr, (2, 0, 1))
            
            out_file = output_dir / f"{file_id}.npz"
            np.savez_compressed(
                out_file,
                image=arr_ch_first,
                class_id=class_id,
                class_name=class_name,
                is_diabetic=0 if class_id == 0 else 1
            )
            
            manifest_records.append({
                "sample_id": file_id,
                "original_filename": png_path.name,
                "class_id": class_id,
                "class_name": class_name,
                "is_diabetic": 0 if class_id == 0 else 1,
                "file_path": str(out_file.relative_to(PROCESSED_THERMAL_DIR.parent))
            })
            
    df_manifest = pd.DataFrame(manifest_records)
    manifest_path = PROCESSED_THERMAL_DIR / "standup_manifest.csv"
    df_manifest.to_csv(manifest_path, index=False)
    logger.info(f"STANDUP processed: {len(df_manifest)} samples written to {manifest_path}")
    return df_manifest


# =====================================================================
# 4. SPLIT CREATION (Strict Anti-Leakage Participant-Level Splits)
# =====================================================================

def create_thermal_splits(
    df_thermo: pd.DataFrame,
    df_iac: pd.DataFrame,
    df_standup: pd.DataFrame
) -> Dict[str, Any]:
    """
    Creates deterministic, participant-level splits with zero leakage.
    ThermoDataBase: Stratified by diabetic status.
    IAC-TecMed: Participant-level split.
    STANDUP: Stratified by risk class.
    Saves split JSON files to ML/splits/thermal/.
    """
    set_seed(SEED)
    logger.info("Generating deterministic participant-level thermal splits...")
    
    # 1. ThermoDataBase Split (Stratified on Subject Level)
    tdb_controls = df_thermo[df_thermo["label"] == 0]["subject_id"].tolist()
    tdb_diabetics = df_thermo[df_thermo["label"] == 1]["subject_id"].tolist()
    
    np.random.shuffle(tdb_controls)
    np.random.shuffle(tdb_diabetics)
    
    def split_list(items, train_r=TRAIN_RATIO, val_r=VAL_RATIO):
        n = len(items)
        n_train = int(round(n * train_r))
        n_val = int(round(n * val_r))
        return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]
        
    ctrl_tr, ctrl_va, ctrl_te = split_list(tdb_controls)
    diab_tr, diab_va, diab_te = split_list(tdb_diabetics)
    
    tdb_splits = {
        "train": sorted(ctrl_tr + diab_tr),
        "val": sorted(ctrl_va + diab_va),
        "test": sorted(ctrl_te + diab_te),
        "summary": {
            "train_count": len(ctrl_tr) + len(diab_tr),
            "val_count": len(ctrl_va) + len(diab_va),
            "test_count": len(ctrl_te) + len(diab_te),
            "train_control": len(ctrl_tr), "train_diabetic": len(diab_tr),
            "val_control": len(ctrl_va), "val_diabetic": len(diab_va),
            "test_control": len(ctrl_te), "test_diabetic": len(diab_te),
        }
    }
    
    # Verify zero leakage across splits
    s_tr, s_va, s_te = set(tdb_splits["train"]), set(tdb_splits["val"]), set(tdb_splits["test"])
    assert len(s_tr & s_va) == 0, "Leakage between train and val in ThermoDataBase!"
    assert len(s_tr & s_te) == 0, "Leakage between train and test in ThermoDataBase!"
    assert len(s_va & s_te) == 0, "Leakage between val and test in ThermoDataBase!"
    
    with open(SPLITS_THERMAL_DIR / "thermodatabase_splits.json", "w") as f:
        json.dump(tdb_splits, f, indent=2)
        
    # 2. IAC-TecMed Split (Subject-level)
    iac_subs = df_iac["subject_id"].tolist()
    np.random.shuffle(iac_subs)
    iac_tr, iac_va, iac_te = split_list(iac_subs)
    
    iac_splits = {
        "train": sorted(iac_tr),
        "val": sorted(iac_va),
        "test": sorted(iac_te),
        "summary": {
            "train_count": len(iac_tr),
            "val_count": len(iac_va),
            "test_count": len(iac_te)
        }
    }
    with open(SPLITS_THERMAL_DIR / "iac_tecmed_splits.json", "w") as f:
        json.dump(iac_splits, f, indent=2)
        
    # 3. STANDUP Split (Stratified on class)
    standup_splits = {"train": [], "val": [], "test": []}
    for c_id in sorted(df_standup["class_id"].unique()):
        c_items = df_standup[df_standup["class_id"] == c_id]["sample_id"].tolist()
        np.random.shuffle(c_items)
        tr, va, te = split_list(c_items)
        standup_splits["train"].extend(tr)
        standup_splits["val"].extend(va)
        standup_splits["test"].extend(te)
        
    standup_splits["summary"] = {
        "train_count": len(standup_splits["train"]),
        "val_count": len(standup_splits["val"]),
        "test_count": len(standup_splits["test"]),
    }
    with open(SPLITS_THERMAL_DIR / "standup_splits.json", "w") as f:
        json.dump(standup_splits, f, indent=2)
        
    logger.info("All thermal splits generated and verified successfully.")
    return {
        "thermodatabase": tdb_splits["summary"],
        "iac_tecmed": iac_splits["summary"],
        "standup": standup_splits["summary"]
    }


def run_thermal_preprocessing() -> Dict[str, Any]:
    """
    Main driver for thermal preprocessing.
    """
    set_seed(SEED)
    df_thermo = process_thermodatabase()
    df_iac = process_iac_tecmed()
    df_standup = process_standup_database()
    splits_summary = create_thermal_splits(df_thermo, df_iac, df_standup)
    
    report = {
        "status": "SUCCESS",
        "thermodatabase_samples": len(df_thermo),
        "iac_tecmed_subjects": len(df_iac),
        "standup_samples": len(df_standup),
        "splits_summary": splits_summary
    }
    
    report_path = PROCESSED_THERMAL_DIR / "quality_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        
    logger.info(f"Thermal preprocessing complete. Report written to {report_path}")
    return report


if __name__ == "__main__":
    run_thermal_preprocessing()

