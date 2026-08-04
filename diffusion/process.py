from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor

    @classmethod
    def cosine(cls, steps: int, device: str | torch.device, offset: float = 0.008) -> "DiffusionSchedule":
        if steps < 2: raise ValueError("steps must be at least 2")
        positions = torch.linspace(0, steps, steps + 1, device=device, dtype=torch.float32)
        cumulative = torch.cos(((positions / steps + offset) / (1 + offset)) * math.pi / 2).square()
        cumulative = cumulative / cumulative[0]
        betas = (1 - cumulative[1:] / cumulative[:-1]).clamp(1e-5, 0.999)
        alphas = 1 - betas
        return cls(betas, alphas, torch.cumprod(alphas, dim=0))

    def q_sample(self, clean: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alpha_bars[timesteps][:, None, None]
        return alpha_bar.sqrt() * clean + (1 - alpha_bar).sqrt() * noise

    def predict_x0(self, noisy: torch.Tensor, timesteps: torch.Tensor,
                   predicted_noise: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alpha_bars[timesteps][:, None, None]
        return (noisy - (1 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt().clamp_min(1e-8)

    def posterior_step(self, noisy: torch.Tensor, timesteps: torch.Tensor,
                       predicted_noise: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        if not torch.all(timesteps == timesteps[0]):
            raise ValueError("posterior_step expects one shared timestep per batch")
        step = int(timesteps[0])
        beta = self.betas[step]; alpha = self.alphas[step]; alpha_bar = self.alpha_bars[step]
        mean = (noisy - beta / (1 - alpha_bar).sqrt() * predicted_noise) / alpha.sqrt()
        if step == 0: return mean
        previous_alpha_bar = self.alpha_bars[step - 1]
        posterior_variance = beta * (1 - previous_alpha_bar) / (1 - alpha_bar)
        return mean + posterior_variance.clamp_min(0).sqrt() * noise

    def posterior_step_from_x0(self, noisy: torch.Tensor, timesteps: torch.Tensor,
                               predicted_clean: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        if not torch.all(timesteps == timesteps[0]): raise ValueError("one shared timestep is required")
        step = int(timesteps[0]); beta = self.betas[step]; alpha = self.alphas[step]; alpha_bar = self.alpha_bars[step]
        previous_alpha_bar = torch.tensor(1.0, device=noisy.device) if step == 0 else self.alpha_bars[step - 1]
        clean_coefficient = beta * previous_alpha_bar.sqrt() / (1 - alpha_bar)
        noisy_coefficient = (1 - previous_alpha_bar) * alpha.sqrt() / (1 - alpha_bar)
        mean = clean_coefficient * predicted_clean + noisy_coefficient * noisy
        if step == 0: return mean
        posterior_variance = beta * (1 - previous_alpha_bar) / (1 - alpha_bar)
        return mean + posterior_variance.clamp_min(0).sqrt() * noise


@torch.no_grad()
def ddpm_restore(
    model: torch.nn.Module, degraded: torch.Tensor, observation_mask: torch.Tensor,
    schedule: DiffusionSchedule, generator: torch.Generator,
    noise_factory: Callable[[torch.Size], torch.Tensor] | None = None,
    clip_min: torch.Tensor | None = None, clip_max: torch.Tensor | None = None,
) -> torch.Tensor:
    """Full-step DDPM sampling; True mask means observed and is clamped every step."""
    if degraded.shape != observation_mask.shape or observation_mask.dtype != torch.bool:
        raise ValueError("degraded and bool observation_mask must share [B,C,L] shape")
    create_noise = noise_factory or (lambda shape: torch.randn(shape, device=degraded.device, generator=generator))
    current = create_noise(degraded.shape)
    current = torch.where(observation_mask, degraded, current)
    for step in reversed(range(len(schedule.betas))):
        timesteps = torch.full((len(current),), step, device=degraded.device, dtype=torch.long)
        predicted_noise = model(current, degraded, observation_mask, timesteps)
        predicted_clean = schedule.predict_x0(current, timesteps, predicted_noise)
        if clip_min is not None and clip_max is not None:
            predicted_clean = torch.maximum(torch.minimum(predicted_clean, clip_max), clip_min)
        reverse_noise = torch.zeros_like(current) if step == 0 else create_noise(current.shape)
        generated = schedule.posterior_step_from_x0(current, timesteps, predicted_clean, reverse_noise)
        current = torch.where(observation_mask, degraded, generated)
    return current
