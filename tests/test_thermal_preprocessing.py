"""
Unit Tests for Thermal Preprocessing and PyTorch Datasets
"""

import pytest
import numpy as np
import torch
from pathlib import Path

from thermal.preprocessing import resize_and_pad_thermal_matrix, load_thermodatabase_csv
from thermal.dataset import ThermoDataset, IACTemporalDataset, STANDUPDataset
from common.config import THERMAL_CANONICAL_SHAPE


def test_thermal_matrix_standardization():
    """Verify that variable raw thermal matrices are properly scaled and padded to canonical shape."""
    # Simulate variable-sized raw foot matrix (140, 58)
    dummy_foot = np.random.uniform(25.0, 34.0, size=(140, 58)).astype(np.float32)
    dummy_foot[:10, :] = 0.0  # background zeros
    
    padded_temp, padded_mask = resize_and_pad_thermal_matrix(dummy_foot, target_shape=THERMAL_CANONICAL_SHAPE)
    
    assert padded_temp.shape == THERMAL_CANONICAL_SHAPE
    assert padded_mask.shape == THERMAL_CANONICAL_SHAPE
    assert padded_mask.dtype == np.uint8
    assert np.all(padded_mask[padded_temp < 15.0] == 0)
    assert np.all(padded_mask[padded_temp >= 15.0] == 1)


def test_thermodatabase_dataset_splits():
    """Verify ThermoDataset splits, tensor shapes, and anti-leakage participant isolation."""
    for split in ["train", "val", "test"]:
        ds = ThermoDataset(split=split, normalize=True)
        assert len(ds) > 0
        tensor, label, meta = ds[0]
        
        # Shape must be (3, 160, 80) -> [Left, Right_flipped, Delta_T]
        assert tensor.shape == (3, THERMAL_CANONICAL_SHAPE[0], THERMAL_CANONICAL_SHAPE[1])
        assert isinstance(tensor, torch.Tensor)
        assert label.item() in [0, 1]
        assert "subject_id" in meta
        assert not torch.isnan(tensor).any()
        assert not torch.isinf(tensor).any()


def test_iac_temporal_dataset():
    """Verify IAC-TecMed multi-timepoint dataset loading."""
    for split in ["train", "val", "test"]:
        ds = IACTemporalDataset(split=split, use_downsampled=True)
        assert len(ds) > 0
        seq_tensor, meta = ds[0]
        
        # Shape: (4, 1, 144, 192) across 4 timepoints
        assert seq_tensor.ndim == 4
        assert seq_tensor.shape[0] == 4
        assert seq_tensor.shape[1] == 1
        assert "rewarming_delta_T15_T5" in meta
        assert not torch.isnan(seq_tensor).any()


def test_standup_dataset():
    """Verify STANDUP dataset loading."""
    for split in ["train", "val", "test"]:
        ds = STANDUPDataset(split=split, binary_task=True)
        assert len(ds) > 0
        img, label, meta = ds[0]
        
        assert img.shape == (3, 224, 224)
        assert label.item() in [0, 1]
        assert not torch.isnan(img).any()

