from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class PointNetEncoder(nn.Module):
    def __init__(self, latent_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        x = points.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = torch.max(x, dim=2).values
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class PointDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,
        num_points: int = 2048,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_points = num_points

        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_points * 6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(latent))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = x.view(x.shape[0], self.num_points, 6)

        points = torch.tanh(x[:, :, :3])
        normals = F.normalize(x[:, :, 3:], p=2, dim=-1)
        return points, normals


class TonguePointAutoEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,
        num_points: int = 2048,
        decoder_hidden_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = PointNetEncoder(latent_dim=latent_dim, dropout=dropout)
        self.decoder = PointDecoder(
            latent_dim=latent_dim,
            num_points=num_points,
            hidden_dim=decoder_hidden_dim,
            dropout=dropout,
        )

    def encode(self, points: torch.Tensor) -> torch.Tensor:
        return self.encoder(points)

    def decode(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.decoder(latent)

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encode(points)
        recon_points, recon_normals = self.decode(latent)
        return latent, recon_points, recon_normals
