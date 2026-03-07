from __future__ import annotations

import warnings

import torch
from torch import nn

try:
    from torchvision.models import ResNet50_Weights, resnet50
    _HAS_TORCHVISION = True
except Exception:
    ResNet50_Weights = None
    resnet50 = None
    _HAS_TORCHVISION = False


class TongueImageEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,
        dropout: float = 0.2,
        pretrained_backbone: bool = True,
        input_channels: int = 3,
    ) -> None:
        super().__init__()
        backbone, in_features = self._build_backbone(
            pretrained_backbone=pretrained_backbone,
            input_channels=input_channels,
        )

        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, latent_dim),
        )

    @staticmethod
    def _build_backbone(pretrained_backbone: bool, input_channels: int) -> tuple[nn.Module, int]:
        if not _HAS_TORCHVISION:
            warnings.warn(
                "torchvision is not available. Falling back to a small CNN backbone. "
                "Install torchvision for ResNet-50."
            )
            return _fallback_backbone(input_channels=input_channels)

        if not pretrained_backbone:
            backbone = resnet50(weights=None)
            _adapt_first_conv(backbone, input_channels=input_channels)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            return backbone, in_features

        try:
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        except Exception as exc:  # pragma: no cover
            warnings.warn(
                f"Could not load pretrained ResNet-50 weights ({exc}). Falling back to random init."
            )
            backbone = resnet50(weights=None)

        _adapt_first_conv(backbone, input_channels=input_channels)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, in_features

    def forward(self, image):
        feat = self.backbone(image)
        latent = self.head(feat)
        return latent


def _adapt_first_conv(backbone: nn.Module, input_channels: int) -> None:
    if input_channels == 3:
        return

    old_conv = backbone.conv1
    new_conv = nn.Conv2d(
        input_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )

    with torch.no_grad():
        if input_channels > old_conv.in_channels:
            new_conv.weight[:, : old_conv.in_channels] = old_conv.weight
            extra = input_channels - old_conv.in_channels
            mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight[:, old_conv.in_channels :] = mean_weight.repeat(1, extra, 1, 1)
        else:
            new_conv.weight.copy_(old_conv.weight[:, :input_channels])

        if old_conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    backbone.conv1 = new_conv


def _fallback_backbone(input_channels: int) -> tuple[nn.Module, int]:
    backbone = nn.Sequential(
        nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(start_dim=1),
    )
    return backbone, 512


def has_torchvision() -> bool:
    return _HAS_TORCHVISION
