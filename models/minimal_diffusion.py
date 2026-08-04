from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_embedding(timesteps: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device) / max(half - 1, 1))
    angles = timesteps.float()[:, None] * frequencies[None, :]
    embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
    return embedding if dimension % 2 == 0 else torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=1)


class ResidualBlock1D(nn.Module):
    def __init__(self, hidden: int, time_dimension: int, dilation: int = 1):
        super().__init__()
        self.time_projection = nn.Linear(time_dimension, hidden)
        self.network = nn.Sequential(nn.GroupNorm(8, hidden), nn.SiLU(), nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation),
                                     nn.GroupNorm(8, hidden), nn.SiLU(), nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation))

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        return x + self.network(x + self.time_projection(time_embedding)[:, :, None])


class MinimalConditionalDiffusion1D(nn.Module):
    """Small epsilon predictor used only for fixed-subset MCAR restoration screening."""

    def __init__(self, channels: int, hidden: int = 64, time_dimension: int = 64, blocks: int = 3):
        super().__init__()
        self.time_dimension = time_dimension
        self.input = nn.Conv1d(channels * 3, hidden, 3, padding=1)
        self.blocks = nn.ModuleList([ResidualBlock1D(hidden, time_dimension, 2 ** index) for index in range(blocks)])
        self.output = nn.Sequential(nn.GroupNorm(8, hidden), nn.SiLU(), nn.Conv1d(hidden, channels, 3, padding=1))

    def forward(self, noisy: torch.Tensor, degraded: torch.Tensor, observation_mask: torch.Tensor,
                timesteps: torch.Tensor) -> torch.Tensor:
        time_embedding = sinusoidal_embedding(timesteps, self.time_dimension)
        hidden = self.input(torch.cat([noisy, degraded, observation_mask.float()], dim=1))
        for block in self.blocks: hidden = block(hidden, time_embedding)
        return self.output(hidden)
