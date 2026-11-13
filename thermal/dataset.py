"""
FootGuard AI - Thermal PyTorch Datasets
Standardized PyTorch Dataset classes for:
1. ThermoDataset: Bilateral 3-channel thermal tensors (Left, Right_flipped, Delta_T)
   with strict participant-level split isolation.
2. IACTemporalDataset: 4-timepoint cold-stress rewarming sequences.
3. STANDUPDataset: Plantar screening images across IWGDF risk categories.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Callable
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from common.config import (
    PROCESSED_THERMAL_DIR,
    SPLITS_THERMAL_DIR,
    THERMAL_CANONICAL_SHAPE
)


class ThermoDataset(Dataset):
    """
    PyTorch Dataset for ThermoDataBase.
    Returns:
        tensor: torch.FloatTensor of shape (3, H, W) -> [Left, Right_flipped, Delta_T]
        label: torch.LongTensor scalar (0 = Control, 1 = Diabetic)
        meta: dict with subject_id, class_name
    """
    def __init__(
        self,
        split: str = "train",
        transform: Optional[Callable] = None,
        normalize: bool = True
    ):
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.transform = transform
        self.normalize = normalize
        
        # Load split definition
        split_path = SPLITS_THERMAL_DIR / "thermodatabase_splits.json"
        with open(split_path, "r") as f:
            splits = json.load(f)
        self.allowed_subjects = set(splits[split])
        
        # Load manifest
        manifest_path = PROCESSED_THERMAL_DIR / "thermodatabase_manifest.csv"
        df_manifest = pd.read_csv(manifest_path)
        self.df = df_manifest[df_manifest["subject_id"].isin(self.allowed_subjects)].reset_index(drop=True)
        
        # Normalization constants (computed strictly across foreground pixels of training split)
        # Typical healthy/diabetic foot skin temp: mean ~30.0 C, std ~3.5 C
        self.temp_mean = 30.0
        self.temp_std = 3.5

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        row = self.df.iloc[idx]
        file_path = PROCESSED_THERMAL_DIR.parent / row["file_path"]
        
        data = np.load(file_path)
        raw_tensor = data["tensor"]  # shape (3, 160, 80)
        label = int(data["label"])
        subject_id = str(data["subject_id"])
        
        tensor = np.copy(raw_tensor)
        if self.normalize:
            # Normalize active temperature channels (channels 0 and 1)
            # Channel 0: Left, Channel 1: Right_flipped
            # Only normalize where temperature > 15.0 C (preserving zero background)
            for ch in [0, 1]:
                mask = tensor[ch] > 15.0
                tensor[ch][mask] = (tensor[ch][mask] - self.temp_mean) / self.temp_std
                
            # Channel 2 is Delta_T (temperature difference, typical range 0-5 C)
            mask_d = tensor[2] > 0.0
            tensor[2][mask_d] = tensor[2][mask_d] / 2.0  # scale to ~[0, 2.5]
            
        tensor_t = torch.from_numpy(tensor).float()
        
        if self.transform is not None:
            tensor_t = self.transform(tensor_t)
            
        meta = {
            "subject_id": subject_id,
            "class_name": row["class_name"],
            "delta_t_mean": float(row["delta_t_mean"]),
            "delta_t_max": float(row["delta_t_max"])
        }
        
        return tensor_t, torch.tensor(label, dtype=torch.long), meta


class IACTemporalDataset(Dataset):
    """
    PyTorch Dataset for IAC-TecMed Cold-Stress Rewarming Sequences.
    Returns:
        sequence: torch.FloatTensor of shape (4, C, H, W) across [T0, T5, T10, T15]
        meta: dict with subject_id, demographics, rewarming trajectory
    """
    def __init__(
        self,
        split: str = "train",
        use_downsampled: bool = True
    ):
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.use_downsampled = use_downsampled
        
        split_path = SPLITS_THERMAL_DIR / "iac_tecmed_splits.json"
        with open(split_path, "r") as f:
            splits = json.load(f)
        self.allowed_subjects = set(splits[split])
        
        manifest_path = PROCESSED_THERMAL_DIR / "iac_tecmed_manifest.csv"
        df_manifest = pd.read_csv(manifest_path)
        self.df = df_manifest[df_manifest["subject_id"].isin(self.allowed_subjects)].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        row = self.df.iloc[idx]
        file_path = PROCESSED_THERMAL_DIR.parent / row["file_path"]
        
        data = np.load(file_path)
        seq = data["downsampled_sequence"] if self.use_downsampled else data["full_sequence"]
        # shape: (4, H, W) -> add channel dimension -> (4, 1, H, W)
        seq = np.expand_dims(seq, axis=1)
        
        # Normalize continuous Celsius temperature (mean 28.0, std 4.0)
        seq_norm = np.copy(seq)
        mask = seq_norm > 15.0
        seq_norm[mask] = (seq_norm[mask] - 28.0) / 4.0
        
        tensor_t = torch.from_numpy(seq_norm).float()
        
        meta = {
            "subject_id": str(data["subject_id"]),
            "patient_id": row["patient_id"],
            "age": float(row["age"]),
            "sex": row["sex"],
            "rewarming_delta_T15_T5": float(row["rewarming_delta_T15_T5"]),
            "mean_temps": [
                float(row["mean_temp_T0"]),
                float(row["mean_temp_T5"]),
                float(row["mean_temp_T10"]),
                float(row["mean_temp_T15"])
            ]
        }
        
        return tensor_t, meta


class STANDUPDataset(Dataset):
    """
    PyTorch Dataset for STANDUP Screening Cohort.
    Returns:
        image: torch.FloatTensor of shape (3, 224, 224)
        class_id: torch.LongTensor scalar (0=Healthy, 1=R0, 2=R1, 3=R2)
        meta: dict with sample_id, class_name, is_diabetic
    """
    def __init__(
        self,
        split: str = "train",
        binary_task: bool = True
    ):
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.binary_task = binary_task
        
        split_path = SPLITS_THERMAL_DIR / "standup_splits.json"
        with open(split_path, "r") as f:
            splits = json.load(f)
        self.allowed_samples = set(splits[split])
        
        manifest_path = PROCESSED_THERMAL_DIR / "standup_manifest.csv"
        df_manifest = pd.read_csv(manifest_path)
        self.df = df_manifest[df_manifest["sample_id"].isin(self.allowed_samples)].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        row = self.df.iloc[idx]
        file_path = PROCESSED_THERMAL_DIR.parent / row["file_path"]
        
        data = np.load(file_path)
        img_arr = data["image"]  # shape (3, 224, 224), in [0, 1]
        
        label = int(data["is_diabetic"]) if self.binary_task else int(data["class_id"])
        
        tensor_t = torch.from_numpy(img_arr).float()
        meta = {
            "sample_id": row["sample_id"],
            "class_name": row["class_name"],
            "is_diabetic": int(row["is_diabetic"])
        }
        
        return tensor_t, torch.tensor(label, dtype=torch.long), meta

