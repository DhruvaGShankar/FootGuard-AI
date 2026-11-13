"""
Unit Tests for Pressure Preprocessing and PyTorch Datasets
"""

import pytest
import numpy as np
import torch
from pathlib import Path

from pressure.preprocessing import fix_sensor_jump_and_spikes, butter_lowpass_filter
from pressure.dataset import CADWalkSpatialDataset, CADWalkTemporalDataset, GaitLabInsoleDataset
from common.config import PRESSURE_CANONICAL_GRID, PRESSURE_STANCE_RESAMPLED_LEN


def test_sensor_jump_and_spike_filtering():
    """Verify changepoint baseline detrending and outlier filtering."""
    # Synthetic signal with baseline step jump and spike
    t = np.linspace(0, 10, 1000)
    sig = np.sin(2 * np.pi * 1.0 * t)
    sig[500:] += 5.0  # baseline step jump
    sig[250] += 20.0  # impulse spike
    
    filtered = fix_sensor_jump_and_spikes(sig)
    assert len(filtered) == len(sig)
    assert not np.isnan(filtered).any()
    # The jump discontinuity and spike should be mitigated
    assert np.max(np.abs(np.diff(filtered))) < 10.0


def test_cad_walk_spatial_dataset():
    """Verify CAD_WALK spatial dataset (2D PPP + PTI maps)."""
    for split in ["train", "val", "test"]:
        ds = CADWalkSpatialDataset(split=split, normalize=True)
        assert len(ds) > 0
        spatial_map, speed, meta = ds[0]
        
        # Shape: (2, 48, 32) -> [PPP, PTI]
        assert spatial_map.shape == (2, PRESSURE_CANONICAL_GRID[0], PRESSURE_CANONICAL_GRID[1])
        assert isinstance(spatial_map, torch.Tensor)
        assert isinstance(speed, torch.Tensor)
        assert not torch.isnan(spatial_map).any()
        assert not torch.isinf(spatial_map).any()
        assert "trial_id" in meta


def test_cad_walk_temporal_dataset():
    """Verify CAD_WALK 1D normalized vertical force curve dataset."""
    for split in ["train", "val", "test"]:
        ds = CADWalkTemporalDataset(split=split, normalize=True)
        assert len(ds) > 0
        force_curve, speed, meta = ds[0]
        
        # Shape: (1, 100)
        assert force_curve.shape == (1, PRESSURE_STANCE_RESAMPLED_LEN)
        assert isinstance(force_curve, torch.Tensor)
        assert not torch.isnan(force_curve).any()
        assert "duration_ms" in meta


def test_gait_lab_insole_dataset():
    """Verify Gait Lab multi-channel insole dataset windowing."""
    ds = GaitLabInsoleDataset(window_len=200, stride=500, normalize=True)
    assert len(ds) > 0
    window, meta = ds[0]
    
    # Shape: (18, 200) -> 18 sensor channels, 200 timesteps
    assert window.ndim == 2
    assert window.shape[0] == 18
    assert window.shape[1] == 200
    assert not torch.isnan(window).any()
    assert "subject_id" in meta
