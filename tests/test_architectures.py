"""
FootGuard AI - Comprehensive Architecture Test Suite
Verifies all 4 baseline architectures:
1. Thermal 2D CNN (Thermal2DCNN)
2. Thermal TCN (ThermalTCN)
3. Pressure 1D CNN (Pressure1DCNN)
4. Pressure TCN (PressureTCN)

Checks:
- Initialization
- Output dimensions
- Forward pass with synthetic batches
- Forward pass with real processed batches
- Loss calculation
- Backpropagation and gradient existence/finiteness
- Batch size of one
- Invalid input shape handling
- CPU execution
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from thermal.cnn import Thermal2DCNN
from thermal.tcn import ThermalTCN
from thermal.dataset import ThermoDataset, IACTemporalDataset
from pressure.cnn import Pressure1DCNN
from pressure.tcn import PressureTCN
from pressure.dataset import CADWalkTemporalDataset, GaitLabInsoleDataset


# =====================================================================
# 1. THERMAL 2D CNN ARCHITECTURE TESTS
# =====================================================================

def test_thermal_2d_cnn_initialization():
    """Verify initialization and parameter existence."""
    model = Thermal2DCNN(in_channels=3, num_classes=2, base_channels=16, feature_dim=128)
    assert isinstance(model, nn.Module)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert num_params > 0


def test_thermal_2d_cnn_synthetic_forward_and_backward():
    """Verify forward pass, loss calculation, and finite gradients on synthetic batch."""
    model = Thermal2DCNN(in_channels=3, num_classes=2, base_channels=16, feature_dim=128)
    model.train()
    
    batch_size = 4
    x = torch.randn(batch_size, 3, 160, 80)
    targets = torch.randint(0, 2, (batch_size,))
    
    # Forward pass
    logits = model(x)
    assert logits.shape == (batch_size, 2)
    
    # Loss calculation
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, targets)
    assert torch.isfinite(loss).item()
    
    # Backpropagation
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Missing gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"


def test_thermal_2d_cnn_real_batch():
    """Verify forward pass using real processed batch from ThermoDataset."""
    dataset = ThermoDataset(split="train")
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch_tensor, batch_label, meta = next(iter(loader))
    
    model = Thermal2DCNN(in_channels=3, num_classes=2)
    model.eval()
    with torch.no_grad():
        logits = model(batch_tensor)
        probs = model.predict_proba(batch_tensor)
        
    assert logits.shape == (4, 2)
    assert probs.shape == (4, 2)
    assert torch.allclose(torch.sum(probs, dim=1), torch.ones(4), atol=1e-5)


def test_thermal_2d_cnn_batch_size_one_and_cpu():
    """Verify model functions with batch size of 1 in eval mode on CPU."""
    model = Thermal2DCNN(in_channels=3, num_classes=2).to("cpu")
    model.eval()
    x = torch.randn(1, 3, 160, 80, device="cpu")
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 2)


def test_thermal_2d_cnn_invalid_inputs():
    """Verify explicit ValueError on incorrect dimensions or channels."""
    model = Thermal2DCNN(in_channels=3, num_classes=2)
    # Wrong dimension (3D instead of 4D)
    with pytest.raises(ValueError):
        model(torch.randn(3, 160, 80))
    # Wrong channels (1 instead of 3)
    with pytest.raises(ValueError):
        model(torch.randn(2, 1, 160, 80))
    # Wrong resolution
    with pytest.raises(ValueError):
        model(torch.randn(2, 3, 120, 60))


# =====================================================================
# 2. THERMAL TCN ARCHITECTURE TESTS
# =====================================================================

def test_thermal_tcn_initialization():
    """Verify ThermalTCN initialization."""
    model = ThermalTCN(in_channels=1, spatial_dim=32, tcn_channels=[32, 32], out_features=1)
    assert isinstance(model, nn.Module)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert num_params > 0


def test_thermal_tcn_synthetic_forward_and_backward():
    """Verify forward pass, loss calculation, and backprop on synthetic sequence."""
    model = ThermalTCN(in_channels=1, spatial_dim=32, tcn_channels=[32, 32], out_features=1)
    model.train()
    
    batch_size = 2
    # 4 timepoints, 1 channel, 144x192 spatial resolution
    x = torch.randn(batch_size, 4, 1, 144, 192)
    target = torch.randn(batch_size, 1)
    
    out = model(x)
    assert out.shape == (batch_size, 1)
    
    criterion = nn.MSELoss()
    loss = criterion(out, target)
    assert torch.isfinite(loss).item()
    
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Missing gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"


def test_thermal_tcn_real_batch():
    """Verify forward pass on real processed batch from IACTemporalDataset."""
    dataset = IACTemporalDataset(split="train", use_downsampled=True)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    seq_tensor, meta = next(iter(loader))
    
    model = ThermalTCN(in_channels=1, spatial_dim=32, tcn_channels=[32, 32], out_features=1)
    model.eval()
    with torch.no_grad():
        out = model(seq_tensor)
    assert out.shape == (2, 1)


def test_thermal_tcn_batch_size_one_and_cpu():
    """Verify ThermalTCN batch size of 1 on CPU."""
    model = ThermalTCN(in_channels=1, spatial_dim=32, tcn_channels=[32, 32], out_features=1).to("cpu")
    model.eval()
    x = torch.randn(1, 4, 1, 144, 192, device="cpu")
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1)


def test_thermal_tcn_invalid_inputs():
    """Verify error on invalid sequence length or dimensions."""
    model = ThermalTCN(in_channels=1, expected_seq_len=4)
    # Wrong sequence length (3 instead of 4)
    with pytest.raises(ValueError):
        model(torch.randn(2, 3, 1, 144, 192))
    # Wrong tensor dimensions
    with pytest.raises(ValueError):
        model(torch.randn(2, 4, 144, 192))


# =====================================================================
# 3. PRESSURE 1D CNN ARCHITECTURE TESTS
# =====================================================================

def test_pressure_1d_cnn_initialization():
    """Verify Pressure1DCNN initialization."""
    model = Pressure1DCNN(in_channels=1, out_features=1, expected_seq_len=100)
    assert isinstance(model, nn.Module)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert num_params > 0


def test_pressure_1d_cnn_synthetic_forward_and_backward():
    """Verify 1D CNN forward and backward pass on synthetic batch."""
    model = Pressure1DCNN(in_channels=1, out_features=1, expected_seq_len=100)
    model.train()
    
    batch_size = 4
    x = torch.randn(batch_size, 1, 100)
    targets = torch.randn(batch_size, 1)
    
    out = model(x)
    assert out.shape == (batch_size, 1)
    
    criterion = nn.MSELoss()
    loss = criterion(out, targets)
    assert torch.isfinite(loss).item()
    
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Missing gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"


def test_pressure_1d_cnn_real_batch():
    """Verify 1D CNN forward pass on real CAD_WALK stance curves."""
    dataset = CADWalkTemporalDataset(split="train")
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    force_curve, speed, meta = next(iter(loader))
    
    model = Pressure1DCNN(in_channels=1, out_features=1, expected_seq_len=100)
    model.eval()
    with torch.no_grad():
        pred_speed = model(force_curve)
    assert pred_speed.shape == (4, 1)


def test_pressure_1d_cnn_multichannel():
    """Verify 1D CNN support for multi-channel insole inputs (e.g. 18 channels, 200 points)."""
    model = Pressure1DCNN(in_channels=18, out_features=1, expected_seq_len=200)
    model.eval()
    x = torch.randn(2, 18, 200)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1)


def test_pressure_1d_cnn_invalid_inputs():
    """Verify exception on incorrect input shape."""
    model = Pressure1DCNN(in_channels=1, expected_seq_len=100)
    with pytest.raises(ValueError):
        model(torch.randn(2, 1, 50))  # wrong length
    with pytest.raises(ValueError):
        model(torch.randn(2, 2, 100))  # wrong channels


# =====================================================================
# 4. PRESSURE TCN ARCHITECTURE TESTS
# =====================================================================

def test_pressure_tcn_initialization():
    """Verify PressureTCN initialization."""
    model = PressureTCN(in_channels=1, tcn_channels=[32, 64], out_features=1)
    assert isinstance(model, nn.Module)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert num_params > 0


def test_pressure_tcn_synthetic_forward_and_backward():
    """Verify PressureTCN forward pass and causal gradient propagation."""
    model = PressureTCN(in_channels=1, tcn_channels=[32, 64], out_features=1, pooling="last")
    model.train()
    
    batch_size = 4
    x = torch.randn(batch_size, 1, 100)
    target = torch.randn(batch_size, 1)
    
    out = model(x)
    assert out.shape == (batch_size, 1)
    
    criterion = nn.MSELoss()
    loss = criterion(out, target)
    assert torch.isfinite(loss).item()
    
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Missing gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"


def test_pressure_tcn_real_batch():
    """Verify PressureTCN forward pass on real CAD_WALK stance curves."""
    dataset = CADWalkTemporalDataset(split="train")
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    force_curve, speed, meta = next(iter(loader))
    
    model = PressureTCN(in_channels=1, tcn_channels=[32, 64], out_features=1)
    model.eval()
    with torch.no_grad():
        out = model(force_curve)
    assert out.shape == (4, 1)


def test_pressure_tcn_multichannel_and_sequence_output():
    """Verify PressureTCN multi-channel insole stream with sequence-to-sequence output."""
    model = PressureTCN(
        in_channels=18,
        tcn_channels=[32, 64],
        out_features=4,
        pooling="none"
    )
    model.eval()
    x = torch.randn(2, 18, 200)
    with torch.no_grad():
        seq_out = model(x)
    # Shape: (B, out_features, seq_len)
    assert seq_out.shape == (2, 4, 200)


def test_pressure_tcn_invalid_inputs():
    """Verify exception on wrong channel count."""
    model = PressureTCN(in_channels=1)
    with pytest.raises(ValueError):
        model(torch.randn(2, 3, 100))
    with pytest.raises(ValueError):
        model(torch.randn(100))  # 1D instead of 3D

