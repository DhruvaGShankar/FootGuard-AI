"""
FootGuard AI - Thermal 2D CNN Architecture
Designed for bilateral plantar thermograms from ThermoDataBase (3 channels: Left, Right_flipped, Delta_T).
Task: Control vs Diabetic classification (cohort stratification, NOT ulcer detection).
"""

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.config import THERMAL_CANONICAL_SHAPE


class ThermalSpatialEncoder(nn.Module):
    """
    Compact 2D Convolutional Spatial Feature Extractor.
    Extracts deep spatial temperature distribution and bilateral asymmetry features.
    """
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        feature_dim: int = 256,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        
        # Block 1: (B, 3, 160, 80) -> (B, 32, 80, 40)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 2: (B, 32, 80, 40) -> (B, 64, 40, 20)
        self.block2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 3: (B, 64, 40, 20) -> (B, 128, 20, 10)
        self.block3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 4: (B, 128, 20, 10) -> (B, 256, 10, 5)
        self.block4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, feature_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling -> (B, 256, 1, 1)
        )
        
        self.dropout = nn.Dropout(p=dropout_rate)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = torch.flatten(x, 1)  # (B, feature_dim)
        x = self.dropout(x)
        return x


class Thermal2DCNN(nn.Module):
    """
    Thermal 2D CNN with decoupled encoder and classification head.
    Input: (B, 3, 160, 80)
    Output: Logits of shape (B, num_classes) where num_classes=2 (0=Control, 1=Diabetic).
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        base_channels: int = 32,
        feature_dim: int = 256,
        dropout_rate: float = 0.3,
        expected_shape: Tuple[int, int] = THERMAL_CANONICAL_SHAPE
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.expected_shape = expected_shape
        
        self.encoder = ThermalSpatialEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            feature_dim=feature_dim,
            dropout_rate=dropout_rate
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.7),
            nn.Linear(64, num_classes)
        )
        
        self._init_classifier_weights()

    def _init_classifier_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def validate_input(self, x: torch.Tensor) -> None:
        """Explicit input dimension and shape check."""
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor (B, C, H, W), got {x.ndim}D tensor of shape {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {x.shape[1]}")
        if self.expected_shape is not None:
            if (x.shape[2], x.shape[3]) != self.expected_shape:
                raise ValueError(
                    f"Expected spatial resolution {self.expected_shape}, got ({x.shape[2]}, {x.shape[3]})"
                )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts dense feature representation (B, feature_dim)."""
        self.validate_input(x)
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning class logits (B, num_classes)."""
        self.validate_input(x)
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax class probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

