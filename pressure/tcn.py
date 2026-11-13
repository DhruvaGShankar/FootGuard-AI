"""
FootGuard AI - Pressure Temporal Convolutional Network (TCN)
Designed for multi-channel temporal sensor streams (Gait Lab: 18 channels)
and normalized dynamic stance force curves (CAD_WALK: 1 channel).
Task: Causal temporal feature modeling and kinematic regression.
"""

from typing import Dict, Any, Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """
    Causal 1D Convolution with explicit left-padding to prevent future information leakage.
    Output at time t depends strictly on inputs at <= t.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=self.padding,
            dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class PressureTemporalBlock(nn.Module):
    """
    Residual Causal Dilated Temporal Block for 1D pressure/shear time-series.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        return self.relu(out + residual)


class PressureTCN(nn.Module):
    """
    Causal Dilated Temporal Convolutional Network for Pressure & Shear Signals.
    Input: (B, in_channels, seq_len)
    Output:
        - If pooling='last': (B, out_features) using final causal timepoint representation
        - If pooling='mean': (B, out_features) using global temporal average
        - If pooling='none': (B, out_features, seq_len) sequence-to-sequence
    """
    def __init__(
        self,
        in_channels: int = 1,
        tcn_channels: List[int] = [32, 64, 128],
        kernel_size: int = 3,
        out_features: int = 1,
        dropout: float = 0.2,
        pooling: str = "last",
        expected_seq_len: Optional[int] = None
    ):
        super().__init__()
        assert pooling in ["last", "mean", "none"], f"Invalid pooling: {pooling}"
        self.in_channels = in_channels
        self.out_features = out_features
        self.pooling = pooling
        self.expected_seq_len = expected_seq_len
        
        layers = []
        num_levels = len(tcn_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = in_channels if i == 0 else tcn_channels[i - 1]
            out_ch = tcn_channels[i]
            layers.append(
                PressureTemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation_size,
                    dropout=dropout
                )
            )
        self.network = nn.Sequential(*layers)
        
        last_ch = tcn_channels[-1]
        self.head = nn.Sequential(
            nn.Linear(last_ch, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, out_features)
        ) if pooling != "none" else nn.Conv1d(last_ch, out_features, 1)

    def validate_input(self, x: torch.Tensor) -> None:
        """Validate input tensor dimensions and channel count."""
        if x.ndim != 3:
            raise ValueError(f"Expected 3D tensor (B, C, L), got {x.ndim}D tensor of shape {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {x.shape[1]}")
        if self.expected_seq_len is not None and x.shape[2] != self.expected_seq_len:
            raise ValueError(f"Expected sequence length {self.expected_seq_len}, got {x.shape[2]}")

    def extract_temporal_features(self, x: torch.Tensor) -> torch.Tensor:
        """Passes through TCN returning feature representations."""
        self.validate_input(x)
        features = self.network(x)  # (B, last_ch, L)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.extract_temporal_features(x)
        
        if self.pooling == "last":
            # Causal last timepoint
            pooled = features[:, :, -1]  # (B, last_ch)
            out = self.head(pooled)
        elif self.pooling == "mean":
            # Global temporal average
            pooled = torch.mean(features, dim=2)  # (B, last_ch)
            out = self.head(pooled)
        else:
            # Sequence-to-sequence
            out = self.head(features)  # (B, out_features, L)
            
        return out

