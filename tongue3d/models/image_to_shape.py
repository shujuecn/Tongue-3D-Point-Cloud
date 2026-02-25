from __future__ import annotations

from torch import nn

from tongue3d.models.image_encoder import TongueImageEncoder
from tongue3d.models.point_autoencoder import PointDecoder


class LatentMapper(nn.Module):
    def __init__(self, latent_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class TongueImageToShape(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_points: int,
        decoder_hidden_dim: int,
        dropout: float,
        pretrained_backbone: bool = True,
        decoder: PointDecoder | None = None,
    ) -> None:
        super().__init__()
        self.image_encoder = TongueImageEncoder(
            latent_dim=latent_dim,
            dropout=dropout,
            pretrained_backbone=pretrained_backbone,
        )
        self.mapper = LatentMapper(latent_dim=latent_dim, dropout=dropout)

        if decoder is None:
            decoder = PointDecoder(
                latent_dim=latent_dim,
                num_points=num_points,
                hidden_dim=decoder_hidden_dim,
                dropout=dropout,
            )
        self.decoder = decoder

    def forward(self, image):
        latent_raw = self.image_encoder(image)
        latent = self.mapper(latent_raw)
        points, normals = self.decoder(latent)
        return latent, points, normals
