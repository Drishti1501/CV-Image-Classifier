"""
CNN Model Architecture for CIFAR-10 Image Classification.
Defines a custom CNN with residual connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Reusable Conv -> BatchNorm -> ReLU block."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """Basic residual block with skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        return self.relu(out + residual)


class CIFAR10CNN(nn.Module):
    """
    Custom CNN for CIFAR-10 classification.

    Architecture:
        Stage 1: Conv(3->64)  + Conv(64->64)  + ResBlock + MaxPool2d -> 16x16
        Stage 2: Conv(64->128)+ Conv(128->128)+ ResBlock + MaxPool2d ->  8x8
        Stage 3: Conv(128->256)+Conv(256->256)+ ResBlock + MaxPool2d ->  4x4
        Global Average Pooling -> 256-d vector
        FC(256->256) -> ReLU -> Dropout -> FC(256->num_classes)
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.4):
        super().__init__()

        self.stage1 = nn.Sequential(
            ConvBlock(3, 64), ConvBlock(64, 64),
            ResidualBlock(64), nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),
        )
        self.stage2 = nn.Sequential(
            ConvBlock(64, 128), ConvBlock(128, 128),
            ResidualBlock(128), nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),
        )
        self.stage3 = nn.Sequential(
            ConvBlock(128, 256), ConvBlock(256, 256),
            ResidualBlock(256), nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)
        return self.classifier(x)

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Return last feature maps (for Grad-CAM)."""
        x = self.stage1(x)
        x = self.stage2(x)
        return self.stage3(x)


def build_model(num_classes: int = 10, dropout: float = 0.4) -> CIFAR10CNN:
    """Factory to instantiate the CNN."""
    return CIFAR10CNN(num_classes=num_classes, dropout=dropout)


if __name__ == "__main__":
    model = build_model()
    dummy = torch.randn(4, 3, 32, 32)
    out = model(dummy)
    print(f"Output shape : {out.shape}")
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n:,}")
