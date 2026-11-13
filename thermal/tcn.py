"""
FootGuard AI - Thermal Temporal Convolutional Network (TCN)
Designed for sequential thermal cold-challenge rewarming observations (IAC-TecMed: T0, T5, T10, T15).
Task: Experimental thermal rewarming trajectory modeling (autonomic vasomotor tone).
NOTE: N=19 non-diabetic participants. Strictly experimental architecture; NOT wound healing prediction.
"""

from typing import Dict, Any, Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameSpatialEncoder(nn.Module):
    """
    Lightweight 2D CNN encoder applied per temporal frame.
    Reduces spatial thermal matrix (C, H, W) to compact spatial embedding (D_spatial).
    """
    def __init__(
        self,
        in_channels: int = 1,
        spatial_dim: int = 64
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1)  # H/2, W/2
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)            # H/4, W/4
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, spatial_dim, kernel_size=3, stride=2, padding=1)   # H/8, W/8
        self.bn3 = nn.BatchNorm2d(spatial_dim)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B * T, C, H, W)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)  # (B * T, spatial_dim, 1, 1)
        x = torch.flatten(x, 1)
        return x


class CausalConv1d(nn.Module):
    """
    Causal 1D Convolution with explicit left-padding to prevent future information leakage.
    Output at time t depends strictly on inputs at <= t.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
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
        # x: (B, C, T)
        out = self.conv(x)
        # Remove right-side excess padding to maintain exact sequence length
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalBlock(nn.Module):
    """
    Residual Causal Dilated Temporal Convolutional Block.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
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


class ThermalTCN(nn.Module):
    """
    Thermal Temporal Convolutional Network.
    Combines 2D frame spatial encoder with 1D causal dilated TCN.
    Input: (B, T, C, H, W) -> e.g. (B, 4, 1, 144, 192) for IAC-TecMed.
    Output: Continuous rewarming prediction (B, out_features) e.g. rewarming delta T15 - T5.
    """
    def __init__(
        self,
        in_channels: int = 1,
        spatial_dim: int = 64,
        tcn_channels: List[int] = [64, 64],
        kernel_size: int = 2,
        out_features: int = 1,
        dropout: float = 0.2,
        expected_seq_len: int = 4
    ):
        super().__init__()
        self.in_channels = in_channels
        self.expected_seq_len = expected_seq_len
        self.spatial_dim = spatial_dim
        
        # 1. Spatial frame encoder
        self.spatial_encoder = FrameSpatialEncoder(
            in_channels=in_channels,
            spatial_dim=spatial_dim
        )
        
        # 2. Causal Dilated Temporal Convolutional Network
        layers = []
        num_levels = len(tcn_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = spatial_dim if i == 0 else tcn_channels[i - 1]
            out_ch = tcn_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation_size,
                    dropout=dropout
                )
            )
        self.tcn = nn.Sequential(*layers)
        
        # 3. Trajectory / Rewarming Regression Head
        last_ch = tcn_channels[-1]
        self.head = nn.Sequential(
            nn.Linear(last_ch, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, out_features)
        )

    def validate_input(self, x: torch.Tensor) -> None:
        """Validate input tensor dimensions."""
        if x.ndim != 5:
            raise ValueError(f"Expected 5D tensor (B, T, C, H, W), got {x.ndim}D tensor of shape {tuple(x.shape)}")
        if x.shape[1] != self.expected_seq_len:
            raise ValueError(f"Expected sequence length {self.expected_seq_len}, got {x.shape[1]}")
        if x.shape[2] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {x.shape[2]}")

    def extract_temporal_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts TCN output embedding at the final timepoint (B, last_ch)."""
        self.validate_input(x)
        b, t, c, h, w = x.shape
        
        # Fold batch and time into spatial encoder: (B * T, C, H, W)
        x_flat = x.view(b * t, c, h, w)
        spatial_feats = self.spatial_encoder(x_flat)  # (B * T, spatial_dim)
        
        # Reshape to sequence for TCN: (B, spatial_dim, T)
        tcn_in = spatial_feats.view(b, t, self.spatial_dim).transpose(1, 2)
        
        # Pass through causal TCN: (B, last_ch, T)
        tcn_out = self.tcn(tcn_in)
        
        # Take the final timepoint representation (causal accumulated history)
        final_embedding = tcn_out[:, :, -1]  # (B, last_ch)
        return final_embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass predicting rewarming parameter (B, out_features)."""
        embedding = self.extract_temporal_features(x)
        pred = self.head(embedding)
        return pred

