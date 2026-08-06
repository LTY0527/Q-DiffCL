from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from diffusion import DiffusionSchedule
from models.minimal_diffusion import sinusoidal_embedding


class SemanticResidualBlock1D(nn.Module):
    """Residual block with post-normalization timestep+semantic FiLM."""

    def __init__(self, hidden: int, time_dimension: int, semantic_dimension: int, dilation: int):
        super().__init__()
        self.hidden = hidden
        self.condition_projection = nn.Linear(time_dimension + semantic_dimension, hidden * 4)
        self.norm1 = nn.GroupNorm(8, hidden)
        self.conv1 = nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(8, hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation)
        self.activation = nn.SiLU()
        self._initialize_film()

    def _initialize_film(self) -> None:
        nn.init.zeros_(self.condition_projection.weight)
        nn.init.zeros_(self.condition_projection.bias)
        # Gamma starts at zero. Small beta paths retain semantic dependence without amplification.
        nn.init.normal_(self.condition_projection.weight[self.hidden:2 * self.hidden], std=0.02)
        nn.init.normal_(self.condition_projection.weight[3 * self.hidden:], std=0.02)

    @staticmethod
    def _film(value: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        return value * (1 + gamma[:, :, None]) + beta[:, :, None]

    def forward(self, value: torch.Tensor, time_embedding: torch.Tensor,
                semantic_embedding: torch.Tensor) -> torch.Tensor:
        condition = torch.cat([time_embedding, semantic_embedding], dim=1)
        gamma1, beta1, gamma2, beta2 = self.condition_projection(condition).chunk(4, dim=1)
        hidden = self._film(self.norm1(value), gamma1, beta1)
        hidden = self.conv1(self.activation(hidden))
        hidden = self._film(self.norm2(hidden), gamma2, beta2)
        hidden = self.conv2(self.activation(hidden))
        return value + hidden


class SemanticPartialDiffusion1D(nn.Module):
    """Epsilon predictor conditioned on noisy/base/mask and per-block semantic FiLM."""

    def __init__(self, channels: int, semantic_dimension: int, hidden: int = 64,
                 time_dimension: int = 64, blocks: int = 3):
        super().__init__()
        self.channels = channels
        self.time_dimension = time_dimension
        self.input = nn.Conv1d(channels * 3, hidden, 3, padding=1)
        self.blocks = nn.ModuleList([
            SemanticResidualBlock1D(hidden, time_dimension, semantic_dimension, 2 ** index)
            for index in range(blocks)
        ])
        self.output = nn.Sequential(nn.GroupNorm(8, hidden), nn.SiLU(), nn.Conv1d(hidden, channels, 3, padding=1))

    def forward(self, noisy: torch.Tensor, base: torch.Tensor, observation_mask: torch.Tensor,
                timesteps: torch.Tensor, semantic_embedding: torch.Tensor) -> torch.Tensor:
        if noisy.shape != base.shape or noisy.shape != observation_mask.shape:
            raise ValueError("noisy, base and observation mask must match")
        if noisy.shape[1] != self.channels:
            raise ValueError("input channel count does not match model")
        time_embedding = sinusoidal_embedding(timesteps, self.time_dimension)
        hidden = self.input(torch.cat([noisy, base, observation_mask.float()], dim=1))
        for block in self.blocks:
            hidden = block(hidden, time_embedding, semantic_embedding)
        output = self.output(hidden)
        if not torch.isfinite(output).all():
            raise FloatingPointError("non-finite semantic diffusion output")
        return output


def sample_training_timesteps(allowed: list[int] | tuple[int, ...], count: int,
                              generator: torch.Generator, device: str | torch.device) -> torch.Tensor:
    choices = torch.as_tensor(allowed, dtype=torch.long, device=device)
    if choices.ndim != 1 or len(choices) == 0 or torch.any(choices < 0):
        raise ValueError("allowed timesteps must be a non-empty non-negative sequence")
    indices = torch.randint(0, len(choices), (count,), device=device, generator=generator)
    return choices[indices]


def partial_diffusion_objective(
    model: nn.Module, schedule: DiffusionSchedule, base: torch.Tensor,
    observation: torch.Tensor, semantic_embedding: torch.Tensor,
    timesteps: torch.Tensor, noise: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    noisy = schedule.q_sample(base, timesteps, noise)
    predicted_noise = model(noisy, base, observation, timesteps, semantic_embedding)
    diffusion_loss = F.mse_loss(predicted_noise, noise)
    predicted_base = schedule.predict_x0(noisy, timesteps, predicted_noise)
    return diffusion_loss, predicted_base


def partial_q_sample(base: torch.Tensor, schedule: DiffusionSchedule,
                     t_aug: int, noise: torch.Tensor) -> torch.Tensor:
    if not 0 <= t_aug < len(schedule.betas):
        raise ValueError("t_aug outside schedule")
    timesteps = torch.full((len(base),), int(t_aug), device=base.device, dtype=torch.long)
    return schedule.q_sample(base, timesteps, noise)


def residual_augment(base: torch.Tensor, sampled: torch.Tensor, alpha: float) -> torch.Tensor:
    if base.shape != sampled.shape:
        raise ValueError("base and sampled shapes must match")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    if alpha == 0:
        return base
    if alpha == 1:
        return sampled
    return base + float(alpha) * (sampled - base)


@torch.no_grad()
def partial_reverse_sample(
    model: nn.Module, schedule: DiffusionSchedule, base: torch.Tensor,
    observation: torch.Tensor, semantic_embedding: torch.Tensor, t_aug: int,
    generator: torch.Generator, clip_min: torch.Tensor | None = None,
    clip_max: torch.Tensor | None = None, alpha: float = 1.0,
) -> torch.Tensor:
    current = partial_q_sample(base, schedule, t_aug,
                               torch.randn(base.shape, device=base.device, generator=generator))
    for step in reversed(range(t_aug + 1)):
        timesteps = torch.full((len(base),), step, device=base.device, dtype=torch.long)
        predicted_noise = model(current, base, observation, timesteps, semantic_embedding)
        predicted_base = schedule.predict_x0(current, timesteps, predicted_noise)
        if clip_min is not None and clip_max is not None:
            predicted_base = torch.maximum(torch.minimum(predicted_base, clip_max), clip_min)
        reverse_noise = (torch.zeros_like(current) if step == 0 else
                         torch.randn(current.shape, device=current.device, generator=generator))
        current = schedule.posterior_step_from_x0(current, timesteps, predicted_base, reverse_noise)
    return residual_augment(base, current, alpha)
