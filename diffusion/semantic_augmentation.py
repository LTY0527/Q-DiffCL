from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from diffusion import DiffusionSchedule
from models.minimal_diffusion import sinusoidal_embedding


class SemanticResidualBlock1D(nn.Module):
    def __init__(self, hidden: int, time_dimension: int, semantic_dimension: int, dilation: int):
        super().__init__()
        self.time_projection = nn.Linear(time_dimension, hidden)
        self.semantic_projection = nn.Linear(semantic_dimension, hidden)
        self.network = nn.Sequential(
            nn.GroupNorm(8, hidden), nn.SiLU(),
            nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, hidden), nn.SiLU(),
            nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation),
        )

    def forward(self, value: torch.Tensor, time_embedding: torch.Tensor,
                semantic_embedding: torch.Tensor) -> torch.Tensor:
        condition = self.time_projection(time_embedding) + self.semantic_projection(semantic_embedding)
        return value + self.network(value + condition[:, :, None])


class SemanticPartialDiffusion1D(nn.Module):
    """Epsilon predictor with semantic FiLM-like additive injection in every block."""

    def __init__(self, channels: int, semantic_dimension: int, hidden: int = 64,
                 time_dimension: int = 64, blocks: int = 3):
        super().__init__(); self.time_dimension = time_dimension
        self.input = nn.Conv1d(channels * 2, hidden, 3, padding=1)
        self.blocks = nn.ModuleList([
            SemanticResidualBlock1D(hidden, time_dimension, semantic_dimension, 2 ** index)
            for index in range(blocks)
        ])
        self.output = nn.Sequential(nn.GroupNorm(8, hidden), nn.SiLU(), nn.Conv1d(hidden, channels, 3, padding=1))

    def forward(self, noisy: torch.Tensor, observation_mask: torch.Tensor,
                timesteps: torch.Tensor, semantic_embedding: torch.Tensor) -> torch.Tensor:
        if noisy.shape != observation_mask.shape: raise ValueError("noisy and observation mask must match")
        time_embedding = sinusoidal_embedding(timesteps, self.time_dimension)
        hidden = self.input(torch.cat([noisy, observation_mask.float()], dim=1))
        for block in self.blocks: hidden = block(hidden, time_embedding, semantic_embedding)
        return self.output(hidden)


def partial_diffusion_objective(
    model: nn.Module, schedule: DiffusionSchedule, base: torch.Tensor,
    observation: torch.Tensor, semantic_embedding: torch.Tensor,
    timesteps: torch.Tensor, noise: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    noisy = schedule.q_sample(base, timesteps, noise)
    predicted_noise = model(noisy, observation, timesteps, semantic_embedding)
    diffusion_loss = F.mse_loss(predicted_noise, noise)
    predicted_base = schedule.predict_x0(noisy, timesteps, predicted_noise)
    return diffusion_loss, predicted_base


def partial_q_sample(base: torch.Tensor, schedule: DiffusionSchedule,
                     t_aug: int, noise: torch.Tensor) -> torch.Tensor:
    if not 0 <= t_aug < len(schedule.betas): raise ValueError("t_aug outside schedule")
    timesteps = torch.full((len(base),), int(t_aug), device=base.device, dtype=torch.long)
    return schedule.q_sample(base, timesteps, noise)


@torch.no_grad()
def partial_reverse_sample(
    model: nn.Module, schedule: DiffusionSchedule, base: torch.Tensor,
    observation: torch.Tensor, semantic_embedding: torch.Tensor, t_aug: int,
    generator: torch.Generator, clip_min: torch.Tensor | None = None,
    clip_max: torch.Tensor | None = None,
) -> torch.Tensor:
    current = partial_q_sample(base, schedule, t_aug, torch.randn(base.shape, device=base.device, generator=generator))
    for step in reversed(range(t_aug + 1)):
        timesteps = torch.full((len(base),), step, device=base.device, dtype=torch.long)
        predicted_noise = model(current, observation, timesteps, semantic_embedding)
        predicted_base = schedule.predict_x0(current, timesteps, predicted_noise)
        if clip_min is not None and clip_max is not None:
            predicted_base = torch.maximum(torch.minimum(predicted_base, clip_max), clip_min)
        reverse_noise = torch.zeros_like(current) if step == 0 else torch.randn(current.shape, device=current.device, generator=generator)
        current = schedule.posterior_step_from_x0(current, timesteps, predicted_base, reverse_noise)
    return current
