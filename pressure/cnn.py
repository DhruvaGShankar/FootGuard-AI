"""
FootGuard AI - Pressure 1D CNN Architecture
Designed for temporal stance curves (CAD_WALK: 1 channel, 100 points)
and multi-channel continuous insole streams (Gait Lab: 18 channels, 200 points).
Task: Temporal feature extraction and kinematic parameter regression (e.g. walking speed).
NOTE: No diabetic or ulcer classification is performed on healthy-only CAD_WALK data.
"""

from typing import Dict, Any, Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class Pressure1DEncoder(nn.Module):
    """
    1D Convolutional Temporal Feature Extractor.
    Extracts multi-scale temporal patterns from 1D kinetic pressure/shear curves.
    """
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        feature_dim: int = 128,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv1d(base_channels * 2, feature_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)  # Global Average Pooling -> (B, feature_dim, 1)
        )
        
        self.dropout = nn.Dropout(p=dropout_rate)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = torch.flatten(x, 1)  # (B, feature_dim)
        x = self.dropout(x)
        return x


class Pressure1DCNN(nn.Module):
    """
    Pressure 1D CNN with decoupled encoder and regression/feature head.
    Input: (B, in_channels, seq_len)
    Output: Regression target (B, out_features) e.g., walking speed in km/h.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_features: int = 1,
        base_channels: int = 32,
        feature_dim: int = 128,
        dropout_rate: float = 0.2,
        expected_seq_len: Optional[int] = 100
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.expected_seq_len = expected_seq_len
        
        self.encoder = Pressure1DEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            feature_dim=feature_dim,
            dropout_rate=dropout_rate
        )
        
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.7),
            nn.Linear(64, out_features)
        )
        
        self._init_head_weights()

    def _init_head_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def validate_input(self, x: torch.Tensor) -> None:
        """Validate input tensor dimensionality and channels."""
        if x.ndim != 3:
            raise ValueError(f"Expected 3D tensor (B, C, L), got {x.ndim}D tensor of shape {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {x.shape[1]}")
        if self.expected_seq_len is not None and x.shape[2] != self.expected_seq_len:
            raise ValueError(f"Expected sequence length {self.expected_seq_len}, got {x.shape[2]}")

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts dense feature representation (B, feature_dim)."""
        self.validate_input(x)
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass predicting continuous target (B, out_features)."""
        self.validate_input(x)
        features = self.encoder(x)
        out = self.head(features)
        return out

