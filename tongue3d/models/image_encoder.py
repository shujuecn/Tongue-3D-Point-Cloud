from __future__ import annotations

import warnings

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
    ) -> None:
        super().__init__()
        backbone, in_features = self._build_backbone(pretrained_backbone=pretrained_backbone)

        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, latent_dim),
        )

    @staticmethod
    def _build_backbone(pretrained_backbone: bool) -> tuple[nn.Module, int]:
        if not _HAS_TORCHVISION:
            warnings.warn(
                "torchvision is not available. Falling back to a small CNN backbone. "
                "Install torchvision for ResNet-50."
            )
            return _fallback_backbone()

        if not pretrained_backbone:
            backbone = resnet50(weights=None)
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

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, in_features

    def forward(self, image):
        feat = self.backbone(image)
        latent = self.head(feat)
        return latent


def _fallback_backbone() -> tuple[nn.Module, int]:
    backbone = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
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
