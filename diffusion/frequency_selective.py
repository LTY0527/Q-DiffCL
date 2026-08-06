from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch


Mode = Literal["uniform", "selective"]


def continuous_alpha_bar(alpha_bars: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    maximum = len(alpha_bars) - 1
    clipped = timesteps.clamp(0, maximum)
    lower = torch.floor(clipped).long(); upper = torch.ceil(clipped).long()
    weight = clipped - lower
    return alpha_bars[lower] * (1 - weight) + alpha_bars[upper] * weight


def match_noise_budget(variance: torch.Tensor, target_mean: float, preserve_dc: bool) -> torch.Tensor:
    value = variance.clone().clamp(0, .999)
    if preserve_dc: value[:, 0] = 0
    adjustable = torch.ones_like(value, dtype=torch.bool)
    if preserve_dc: adjustable[:, 0] = False
    if not adjustable.any(): raise ValueError("noise budget has no adjustable frequency bins")
    desired_adjustable = float(target_mean) * value.numel() / int(adjustable.sum())
    if not 0 <= desired_adjustable < .999:
        raise ValueError("target noise budget is outside feasible range")
    raw = value[adjustable]
    if float(raw.mean()) <= 0: raise ValueError("selective noise variance must be positive")
    low, high = 0.0, max(1.0, desired_adjustable / float(raw.mean()) * 2)
    for _ in range(60):
        middle = .5 * (low + high)
        current = torch.clamp(raw * middle, max=.999).mean()
        if float(current) < desired_adjustable: low = middle
        else: high = middle
    value[adjustable] = torch.clamp(raw * (.5 * (low + high)), max=.999)
    if abs(float(value.mean()) - float(target_mean)) > 1e-6:
        raise RuntimeError("failed to match spectral noise budget")
    return value


def spectral_noise_variance(
    alpha_bars: torch.Tensor, channels: int, frequencies: int, mode: Mode,
    t_uniform: int, preserve_dc: bool, soft_mask: torch.Tensor | None = None,
    t_critical: int | None = None, t_noncritical: int | None = None,
) -> torch.Tensor:
    uniform = 1 - alpha_bars[int(t_uniform)]
    target = torch.full((channels, frequencies), uniform, device=alpha_bars.device)
    if preserve_dc: target[:, 0] = 0
    if mode == "uniform": return target
    if soft_mask is None or t_critical is None or t_noncritical is None:
        raise ValueError("selective diffusion requires mask and critical/noncritical timesteps")
    if soft_mask.shape != (channels, frequencies): raise ValueError("soft mask shape mismatch")
    timestep = float(t_critical) + (1 - soft_mask.clamp(0, 1)) * (float(t_noncritical) - float(t_critical))
    variance = 1 - continuous_alpha_bar(alpha_bars, timestep)
    return match_noise_budget(variance, float(target.mean()), preserve_dc)


@dataclass(frozen=True)
class SpectralStatistics:
    mean: np.ndarray
    scale: np.ndarray
    maximum_log_amplitude: np.ndarray
    fit_split: str = "train"


def fit_spectral_statistics(train_values: np.ndarray, quantile: float = .999,
                            split: str = "train") -> SpectralStatistics:
    if split != "train": raise ValueError("spectral statistics may only be fitted on train")
    spectrum = torch.fft.rfft(torch.as_tensor(train_values, dtype=torch.float32), dim=-1)
    log_amplitude = torch.log1p(torch.abs(spectrum)).cpu().numpy().astype(np.float64)
    scale = log_amplitude.std(0)
    maximum = np.quantile(log_amplitude, float(quantile), axis=0)
    return SpectralStatistics(log_amplitude.mean(0), np.where(scale > 1e-8, scale, 1.0),
                              np.maximum(maximum, 1e-6), split)


def normalized_difference(base: np.ndarray, changed: np.ndarray) -> float:
    scale = np.maximum(np.asarray(base).std(axis=(1, 2)), 1e-6)
    return float(np.mean(np.abs(np.asarray(changed) - base).mean(axis=(1, 2)) / scale))


class FrequencyForwardDiffusion:
    def __init__(
        self, statistics: SpectralStatistics, alpha_bars: torch.Tensor, soft_mask: np.ndarray,
        t_uniform: int, t_critical: int, preserve_phase: bool = True, preserve_dc: bool = True,
        device: str = "cpu",
    ) -> None:
        if statistics.fit_split != "train": raise ValueError("spectral statistics must be train-only")
        self.device = device; self.statistics = statistics
        self.alpha_bars = alpha_bars.to(device); self.soft_mask = torch.as_tensor(soft_mask, dtype=torch.float32, device=device)
        self.t_uniform = int(t_uniform); self.t_critical = int(t_critical)
        self.preserve_phase = bool(preserve_phase); self.preserve_dc = bool(preserve_dc)
        if not self.preserve_phase: raise ValueError("MVP requires original phase preservation")

    def variance(self, mode: Mode, t_noncritical: int | None = None) -> torch.Tensor:
        channels, frequencies = self.soft_mask.shape
        return spectral_noise_variance(
            self.alpha_bars, channels, frequencies, mode, self.t_uniform, self.preserve_dc,
            self.soft_mask, self.t_critical, t_noncritical)

    @torch.no_grad()
    def augment(self, values: np.ndarray, mode: Mode, seed: int, t_noncritical: int | None = None,
                batch_size: int = 256) -> tuple[np.ndarray, dict[str, Any]]:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 3 or values.shape[1:] != (self.soft_mask.shape[0], (self.soft_mask.shape[1] - 1) * 2):
            raise ValueError("frequency diffusion input shape mismatch")
        variance = self.variance(mode, t_noncritical); alpha = 1 - variance
        mean = torch.as_tensor(self.statistics.mean, dtype=torch.float32, device=self.device)
        scale = torch.as_tensor(self.statistics.scale, dtype=torch.float32, device=self.device)
        maximum = torch.as_tensor(self.statistics.maximum_log_amplitude, dtype=torch.float32, device=self.device)
        generator = torch.Generator(device=self.device).manual_seed(int(seed)); output = np.empty_like(values)
        phase_error = []; reconstruction_error = []
        for start in range(0, len(values), batch_size):
            stop = min(start + batch_size, len(values)); base = torch.from_numpy(values[start:stop]).to(self.device)
            spectrum = torch.fft.rfft(base, dim=-1); log_amplitude = torch.log1p(torch.abs(spectrum)); phase = torch.angle(spectrum)
            standardized = (log_amplitude - mean) / scale
            noise = torch.randn(standardized.shape, device=self.device, generator=generator)
            changed_standardized = alpha.sqrt() * standardized + variance.sqrt() * noise
            changed_log_amplitude = (changed_standardized * scale + mean).clamp_min(0)
            changed_log_amplitude = torch.minimum(changed_log_amplitude, maximum)
            if self.preserve_dc: changed_log_amplitude[:, :, 0] = log_amplitude[:, :, 0]
            amplitude = torch.expm1(changed_log_amplitude).clamp_min(0)
            changed_spectrum = torch.polar(amplitude, phase)
            changed = torch.fft.irfft(changed_spectrum, n=values.shape[-1], dim=-1)
            if not torch.isfinite(changed).all(): raise FloatingPointError("non-finite spectral augmentation")
            output[start:stop] = changed.cpu().numpy()
            roundtrip = torch.fft.rfft(changed, dim=-1)
            valid = amplitude > 1e-6
            angular = torch.atan2(torch.sin(torch.angle(roundtrip) - phase), torch.cos(torch.angle(roundtrip) - phase)).abs()
            phase_error.append(float(angular[valid].mean()) if valid.any() else 0.0)
            reconstructed = torch.fft.irfft(spectrum, n=values.shape[-1], dim=-1)
            reconstruction_error.append(float(torch.max(torch.abs(reconstructed - base))))
        critical = self.soft_mask >= .5; noncritical = ~critical
        diagnostics = {
            "mode": mode, "t_uniform": self.t_uniform, "t_critical": self.t_critical,
            "t_noncritical": t_noncritical, "expected_total_noise_budget": float(variance.mean()),
            "critical_noise_budget": float(variance[critical].mean()) if critical.any() else None,
            "noncritical_noise_budget": float(variance[noncritical].mean()) if noncritical.any() else None,
            "phase_preserved": self.preserve_phase, "mean_phase_error": float(np.mean(phase_error)),
            "dc_preserved": self.preserve_dc, "inverse_fft_reconstruction_max_error": float(np.max(reconstruction_error)),
            "finite": bool(np.isfinite(output).all()), "time_normalized_l1": normalized_difference(values, output),
            "amplitude_min": float(output.min()), "amplitude_max": float(output.max()),
        }
        return output, diagnostics

