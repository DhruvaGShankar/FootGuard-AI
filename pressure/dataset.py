"""
FootGuard AI - Pressure PyTorch Datasets
Standardized PyTorch Dataset classes for:
1. CADWalkSpatialDataset: 2D spatial maps (Peak Plantar Pressure & Pressure-Time Integral)
   with shape (2, 38, 22).
2. CADWalkTemporalDataset: 1D normalized vertical force stance curves (1, 100).
3. GaitLabInsoleDataset: Multi-channel insole streams (normal + tri-axial shear).
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from common.config import (
    PROCESSED_PRESSURE_DIR,
    SPLITS_PRESSURE_DIR,
    PRESSURE_CANONICAL_GRID,
    PRESSURE_STANCE_RESAMPLED_LEN
)


def pad_to_canonical_grid(
    arr: np.ndarray,
    target_shape: Tuple[int, int] = PRESSURE_CANONICAL_GRID
) -> np.ndarray:
    """
    Pads 2D pressure matrix to canonical grid (48, 32) centered with zeros.
    Preserves exact physical pressure values without interpolation distortion.
    """
    h, w = arr.shape
    th, tw = target_shape
    pad_top = max(0, (th - h) // 2)
    pad_bottom = max(0, th - h - pad_top)
    pad_left = max(0, (tw - w) // 2)
    pad_right = max(0, tw - w - pad_left)
    
    padded = np.pad(
        arr[:th, :tw],
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0.0
    )
    return padded[:th, :tw]


class CADWalkSpatialDataset(Dataset):
    """
    PyTorch Dataset for 2D Plantar Pressure Maps (CAD_WALK).
    Returns:
        spatial_map: torch.FloatTensor of shape (2, 48, 32) -> [PPP, PTI]
        target_speed: torch.FloatTensor scalar (walking speed in km/h)
        meta: dict with subject_id, foot, trial_num, duration_ms
    """
    def __init__(
        self,
        split: str = "train",
        normalize: bool = True,
        target_shape: Tuple[int, int] = PRESSURE_CANONICAL_GRID,
        filter_valid_speed: bool = True
    ):
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.normalize = normalize
        self.target_shape = target_shape
        self.filter_valid_speed = filter_valid_speed
        
        split_path = SPLITS_PRESSURE_DIR / "cad_walk_splits.json"
        with open(split_path, "r") as f:
            splits = json.load(f)
            
        allowed_trials = set(splits[f"{split}_trials"])
        
        manifest_path = PROCESSED_PRESSURE_DIR / "cad_walk_manifest.csv"
        df_manifest = pd.read_csv(manifest_path)
        df_filtered = df_manifest[df_manifest["trial_id"].isin(allowed_trials)]
        
        if self.filter_valid_speed:
            df_filtered = df_filtered[df_filtered["walking_speed"] > 0.5]
            
        self.df = df_filtered.reset_index(drop=True)
        
        # Mean and std for PPP and PTI scaling
        self.ppp_scale = 300.0  # kPa
        self.pti_scale = 100.0  # kPa*s

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        row = self.df.iloc[idx]
        file_path = PROCESSED_PRESSURE_DIR.parent / row["file_path"]
        
        data = np.load(file_path)
        ppp = data["ppp"]  # variable shape e.g. (38, 22)
        pti = data["pti"]
        
        if self.normalize:
            ppp_norm = ppp / self.ppp_scale
            pti_norm = pti / self.pti_scale
        else:
            ppp_norm = ppp
            pti_norm = pti
            
        # Pad to canonical uniform grid (48, 32)
        ppp_pad = pad_to_canonical_grid(ppp_norm, self.target_shape)
        pti_pad = pad_to_canonical_grid(pti_norm, self.target_shape)
            
        # Shape: (2, 48, 32)
        spatial_map = np.stack([ppp_pad, pti_pad], axis=0).astype(np.float32)
        tensor_t = torch.from_numpy(spatial_map).float()
        
        speed = float(row["walking_speed"])
        target_t = torch.tensor(speed, dtype=torch.float32)
        
        meta = {
            "trial_id": row["trial_id"],
            "subject_id": row["subject_id"],
            "foot": row["foot"],
            "trial_num": int(row["trial_num"]),
            "duration_ms": float(row["duration_ms"]),
            "peak_pressure_kpa": float(row["peak_pressure_kpa"])
        }
        
        return tensor_t, target_t, meta


class CADWalkTemporalDataset(Dataset):
    """
    PyTorch Dataset for 1D Plantar Force Curves (CAD_WALK).
    Returns:
        force_curve: torch.FloatTensor of shape (1, 100) (normalized stance curve)
        target_speed: torch.FloatTensor scalar (walking speed in km/h)
        meta: dict with subject_id, foot, trial_num, duration_ms
    """
    def __init__(
        self,
        split: str = "train",
        normalize: bool = True,
        filter_valid_speed: bool = True
    ):
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.normalize = normalize
        self.filter_valid_speed = filter_valid_speed
        
        split_path = SPLITS_PRESSURE_DIR / "cad_walk_splits.json"
        with open(split_path, "r") as f:
            splits = json.load(f)
            
        allowed_trials = set(splits[f"{split}_trials"])
        
        manifest_path = PROCESSED_PRESSURE_DIR / "cad_walk_manifest.csv"
        df_manifest = pd.read_csv(manifest_path)
        df_filtered = df_manifest[df_manifest["trial_id"].isin(allowed_trials)]
        
        # Rule #10: Do not train or evaluate using unlabeled data
        # Exclude trials where walking speed is unrecorded / missing (e.g. subject C51)
        if self.filter_valid_speed:
            df_filtered = df_filtered[df_filtered["walking_speed"] > 0.5]
            
        self.df = df_filtered.reset_index(drop=True)
        self.force_scale = 800.0  # Newtons

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        row = self.df.iloc[idx]
        file_path = PROCESSED_PRESSURE_DIR.parent / row["file_path"]
        
        data = np.load(file_path)
        force = data["force_curve"]  # shape (100,)
        
        if self.normalize:
            force_norm = force / self.force_scale
        else:
            force_norm = force
            
        # Shape: (1, 100)
        force_t = torch.from_numpy(force_norm).unsqueeze(0).float()
        speed = float(row["walking_speed"])
        target_t = torch.tensor(speed, dtype=torch.float32)
        
        meta = {
            "trial_id": row["trial_id"],
            "subject_id": row["subject_id"],
            "foot": row["foot"],
            "trial_num": int(row["trial_num"]),
            "duration_ms": float(row["duration_ms"]),
            "peak_force_n": float(row["peak_force_n"])
        }
        
        return force_t, target_t, meta


class GaitLabInsoleDataset(Dataset):
    """
    PyTorch Dataset for Continuous Insole Sensor Streams (Gait Lab).
    Segments continuous multi-channel signals into sliding temporal windows.
    Returns:
        window: torch.FloatTensor of shape (C, window_len) where C=18 channels
        meta: dict with subject_id, foot, walk_run
    """
    def __init__(
        self,
        window_len: int = 200,   # 2 seconds at 100 Hz
        stride: int = 100,       # 1 second step
        normalize: bool = True
    ):
        super().__init__()
        self.window_len = window_len
        self.stride = stride
        self.normalize = normalize
        
        manifest_path = PROCESSED_PRESSURE_DIR / "gait_lab_manifest.csv"
        self.df_manifest = pd.read_csv(manifest_path)
        
        self.windows = []
        for idx, row in self.df_manifest.iterrows():
            file_path = PROCESSED_PRESSURE_DIR.parent / row["file_path"]
            data = np.load(file_path)
            sig = data["conditioned_signals"]  # shape (T, 18)
            T_len = sig.shape[0]
            
            for start in range(0, T_len - window_len + 1, stride):
                end = start + window_len
                self.windows.append({
                    "file_path": file_path,
                    "start": start,
                    "end": end,
                    "subject_id": row["subject_id"],
                    "foot": row["foot"],
                    "walk_run": row["walk_run"]
                })

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        win_info = self.windows[idx]
        data = np.load(win_info["file_path"])
        sig_window = data["conditioned_signals"][win_info["start"]:win_info["end"]]  # (L, C)
        
        # Channels-first: (C, L)
        sig_ch_first = np.transpose(sig_window, (1, 0)).astype(np.float32)
        
        if self.normalize:
            # Per-channel zero-mean unit-variance within window
            means = np.mean(sig_ch_first, axis=1, keepdims=True)
            stds = np.std(sig_ch_first, axis=1, keepdims=True) + 1e-6
            sig_ch_first = (sig_ch_first - means) / stds
            
        tensor_t = torch.from_numpy(sig_ch_first).float()
        
        meta = {
            "subject_id": win_info["subject_id"],
            "foot": win_info["foot"],
            "walk_run": win_info["walk_run"]
        }
        
        return tensor_t, meta
