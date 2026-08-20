from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch

from frequency.cross_channel_structure import CrossChannelSpectralStructure


Mode = Literal["uniform", "selective", "uncertainty_gated", "domain_reliable_safe",
               "budget_scaled_selective"]
NoiseStructure = Literal["iid", "correlated"]


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


def scale_spectral_budget(variance: torch.Tensor, rho: float, preserve_dc: bool) -> torch.Tensor:
    """Scale total variance while preserving the frozen relative allocation."""
    if not 0.0 <= float(rho) <= 1.0:
        raise ValueError("budget shrinkage rho must lie in [0,1]")
    if variance.ndim != 2 or not torch.isfinite(variance).all():
        raise ValueError("spectral variance must be a finite channel x frequency matrix")
    if torch.any((variance < 0) | (variance >= 1)):
        raise ValueError("spectral variance must lie in [0,1)")
    if preserve_dc and torch.any(variance[:, 0] != 0):
        raise ValueError("spectral variance must preserve DC before shrinkage")
    scaled = variance * float(rho)
    if preserve_dc:
        scaled[:, 0] = 0
    expected = float((variance * float(rho)).mean())
    if abs(float(scaled.mean()) - expected) > 1e-8:
        raise RuntimeError("budget shrinkage changed the requested total budget")
    return scaled


def spectral_noise_variance(
    alpha_bars: torch.Tensor, channels: int, frequencies: int, mode: Mode,
    t_uniform: int, preserve_dc: bool, soft_mask: torch.Tensor | None = None,
    t_critical: int | None = None, t_noncritical: int | None = None,
    assignment_confidence: torch.Tensor | None = None,
) -> torch.Tensor:
    uniform = 1 - alpha_bars[int(t_uniform)]
    target = torch.full((channels, frequencies), uniform, device=alpha_bars.device)
    if preserve_dc: target[:, 0] = 0
    if mode == "uniform": return target
    if soft_mask is None or t_critical is None or t_noncritical is None:
        raise ValueError("selective diffusion requires mask and critical/noncritical timesteps")
    if soft_mask.shape != (channels, frequencies): raise ValueError("soft mask shape mismatch")
    r1_timestep = float(t_critical) + (1 - soft_mask.clamp(0, 1)) * (float(t_noncritical) - float(t_critical))
    if mode == "uncertainty_gated":
        if assignment_confidence is None or assignment_confidence.shape != soft_mask.shape:
            raise ValueError("uncertainty-gated diffusion requires aligned assignment confidence")
        if torch.any((assignment_confidence < 0) | (assignment_confidence > 1)):
            raise ValueError("assignment confidence must lie in [0,1]")
        timestep = float(t_uniform) + assignment_confidence * (r1_timestep - float(t_uniform))
    elif mode == "selective": timestep = r1_timestep
    else: raise ValueError(f"unknown spectral diffusion mode: {mode}")
    variance = 1 - continuous_alpha_bar(alpha_bars, timestep)
    return match_noise_budget(variance, float(target.mean()), preserve_dc)


def unmatched_selective_noise_variance(
    alpha_bars: torch.Tensor, soft_mask: torch.Tensor, t_critical: int,
    t_noncritical: int, preserve_dc: bool,
) -> torch.Tensor:
    """Return the raw selective allocation before total-budget matching.

    This is exposed only for the preregistered paper ablation.  It keeps the
    frozen timestep map and DC rule, while intentionally omitting the global
    rescaling performed by :func:`spectral_noise_variance`.
    """
    if soft_mask.ndim != 2 or not torch.isfinite(soft_mask).all():
        raise ValueError("unmatched selective mask must be a finite 2-D matrix")
    if torch.any((soft_mask < 0) | (soft_mask > 1)):
        raise ValueError("unmatched selective mask must lie in [0,1]")
    timestep = float(t_critical) + (1 - soft_mask) * (float(t_noncritical) - float(t_critical))
    variance = (1 - continuous_alpha_bar(alpha_bars, timestep)).clamp(0, .999)
    if preserve_dc:
        variance[:, 0] = 0
    return variance


def constrain_channel_budget(variance: torch.Tensor, c1_variance: torch.Tensor,
                             maximum_ratio: float) -> torch.Tensor:
    if variance.shape != c1_variance.shape:
        raise ValueError("channel budget reference shape mismatch")
    if maximum_ratio < 1:
        raise ValueError("maximum channel budget ratio must be at least one")
    raw_energy = variance.sum(1)
    caps = c1_variance.sum(1) * float(maximum_ratio)
    target_total = variance.sum()
    desired = torch.minimum(raw_energy, caps)
    deficit = target_total - desired.sum()
    for _ in range(variance.shape[0] + 1):
        if float(deficit) <= 1e-8:
            break
        room = (caps - desired).clamp_min(0)
        if float(room.sum()) <= 0:
            raise ValueError("channel budget cannot preserve the total noise budget")
        addition = torch.minimum(room, deficit * room / room.sum())
        desired = desired + addition
        deficit = target_total - desired.sum()
    if float(deficit) > 1e-6:
        raise RuntimeError("failed to redistribute channel noise budget")
    scale = desired / raw_energy.clamp_min(1e-12)
    constrained = variance * scale[:, None]
    if torch.any(constrained > .999):
        raise ValueError("channel budget redistribution exceeds feasible per-bin variance")
    if torch.any(constrained.sum(1) > caps + 1e-6):
        raise RuntimeError("channel noise budget constraint failed")
    if abs(float(constrained.mean() - variance.mean())) > 1e-8:
        raise RuntimeError("channel budget changed the total noise budget")
    return constrained


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
        device: str = "cpu", cross_channel_structure: CrossChannelSpectralStructure | None = None,
        maximum_channel_budget_ratio: float | None = None,
    ) -> None:
        if statistics.fit_split != "train": raise ValueError("spectral statistics must be train-only")
        self.device = device; self.statistics = statistics
        self.alpha_bars = alpha_bars.to(device); self.soft_mask = torch.as_tensor(soft_mask, dtype=torch.float32, device=device)
        self.t_uniform = int(t_uniform); self.t_critical = int(t_critical)
        self.preserve_phase = bool(preserve_phase); self.preserve_dc = bool(preserve_dc)
        self.cross_channel_structure = cross_channel_structure
        self.maximum_channel_budget_ratio = maximum_channel_budget_ratio
        if not self.preserve_phase: raise ValueError("MVP requires original phase preservation")
        if cross_channel_structure is not None:
            if cross_channel_structure.fit_split != "train":
                raise ValueError("cross-channel structure must be train-only")
            expected = (self.soft_mask.shape[1], self.soft_mask.shape[0], self.soft_mask.shape[0])
            if cross_channel_structure.covariance.shape != expected:
                raise ValueError("cross-channel structure shape mismatch")

    def variance(self, mode: Mode, t_noncritical: int | None = None,
                 apply_channel_budget: bool = False,
                 assignment_confidence: np.ndarray | torch.Tensor | None = None,
                 variance_override: np.ndarray | torch.Tensor | None = None) -> torch.Tensor:
        channels, frequencies = self.soft_mask.shape
        if mode in {"domain_reliable_safe", "budget_scaled_selective"}:
            if variance_override is None:
                raise ValueError(f"{mode} requires an audited variance override")
            variance = torch.as_tensor(variance_override, dtype=torch.float32, device=self.device)
            if variance.shape != self.soft_mask.shape or not torch.isfinite(variance).all():
                raise ValueError("variance override must be finite and aligned")
            if torch.any((variance < 0) | (variance >= 1)):
                raise ValueError("variance override must lie in [0,1)")
            if self.preserve_dc and torch.any(variance[:, 0] != 0):
                raise ValueError("variance override must preserve DC")
        else:
            variance = spectral_noise_variance(
                self.alpha_bars, channels, frequencies, mode, self.t_uniform, self.preserve_dc,
                self.soft_mask, self.t_critical, t_noncritical,
                None if assignment_confidence is None else torch.as_tensor(assignment_confidence, dtype=torch.float32, device=self.device))
        if apply_channel_budget:
            if self.maximum_channel_budget_ratio is None:
                raise ValueError("channel budget ratio is not configured")
            reference = spectral_noise_variance(
                self.alpha_bars, channels, frequencies, "uniform", self.t_uniform, self.preserve_dc)
            variance = constrain_channel_budget(variance, reference, self.maximum_channel_budget_ratio)
        return variance

    @torch.no_grad()
    def augment(self, values: np.ndarray, mode: Mode, seed: int, t_noncritical: int | None = None,
                batch_size: int = 256, noise_structure: NoiseStructure = "iid",
                apply_channel_budget: bool = False,
                assignment_confidence: np.ndarray | None = None,
                variance_override: np.ndarray | torch.Tensor | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 3 or values.shape[1:] != (self.soft_mask.shape[0], (self.soft_mask.shape[1] - 1) * 2):
            raise ValueError("frequency diffusion input shape mismatch")
        if noise_structure not in {"iid", "correlated"}:
            raise ValueError("unknown spectral noise structure")
        if noise_structure == "correlated" and self.cross_channel_structure is None:
            raise ValueError("correlated noise requires fitted train-only structure")
        variance = self.variance(mode, t_noncritical, apply_channel_budget, assignment_confidence,
                                 variance_override); alpha = 1 - variance
        mean = torch.as_tensor(self.statistics.mean, dtype=torch.float32, device=self.device)
        scale = torch.as_tensor(self.statistics.scale, dtype=torch.float32, device=self.device)
        maximum = torch.as_tensor(self.statistics.maximum_log_amplitude, dtype=torch.float32, device=self.device)
        factors = None if self.cross_channel_structure is None else torch.as_tensor(
            self.cross_channel_structure.factors, dtype=torch.float32, device=self.device)
        generator = torch.Generator(device=self.device).manual_seed(int(seed)); output = np.empty_like(values)
        phase_error = []; reconstruction_error = []
        for start in range(0, len(values), batch_size):
            stop = min(start + batch_size, len(values)); base = torch.from_numpy(values[start:stop]).to(self.device)
            spectrum = torch.fft.rfft(base, dim=-1); log_amplitude = torch.log1p(torch.abs(spectrum)); phase = torch.angle(spectrum)
            standardized = (log_amplitude - mean) / scale
            iid_noise = torch.randn(standardized.shape, device=self.device, generator=generator)
            if noise_structure == "iid":
                noise = iid_noise
            else:
                noise = torch.einsum("bfc,fkc->bfk", iid_noise.permute(0, 2, 1), factors).permute(0, 2, 1)
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
            "noise_structure": noise_structure, "channel_budget_applied": bool(apply_channel_budget),
            "maximum_channel_budget_ratio": self.maximum_channel_budget_ratio,
            "maximum_observed_channel_budget_ratio": float((variance.sum(1) / self.variance("uniform").sum(1).clamp_min(1e-12)).max()),
            "critical_noise_budget": float(variance[critical].mean()) if critical.any() else None,
            "noncritical_noise_budget": float(variance[noncritical].mean()) if noncritical.any() else None,
            "phase_preserved": self.preserve_phase, "mean_phase_error": float(np.mean(phase_error)),
            "dc_preserved": self.preserve_dc, "inverse_fft_reconstruction_max_error": float(np.max(reconstruction_error)),
            "finite": bool(np.isfinite(output).all()), "time_normalized_l1": normalized_difference(values, output),
            "amplitude_min": float(output.min()), "amplitude_max": float(output.max()),
        }
        if mode == "uncertainty_gated":
            confidence = torch.as_tensor(assignment_confidence, dtype=torch.float32, device=self.device)
            r1_timestep = self.t_critical + (1 - self.soft_mask.clamp(0, 1)) * (int(t_noncritical) - self.t_critical)
            timestep = self.t_uniform + confidence * (r1_timestep - self.t_uniform)
            uncertain = confidence < .25; stable_critical = (confidence >= .75) & (self.soft_mask >= .5)
            stable_noncritical = (confidence >= .75) & (self.soft_mask < .5)
            diagnostics.update({"assignment_confidence_min": float(confidence.min()),
                "assignment_confidence_max": float(confidence.max()),
                "timestep_mean": float(timestep.mean()), "timestep_min": float(timestep.min()),
                "timestep_max": float(timestep.max()),
                "timestep_mean_absolute_change_from_r1": float(torch.mean(torch.abs(timestep - r1_timestep))),
                "uncertain_noise_budget": float(variance[uncertain].mean()) if uncertain.any() else None,
                "stable_critical_noise_budget": float(variance[stable_critical].mean()) if stable_critical.any() else None,
                "stable_noncritical_noise_budget": float(variance[stable_noncritical].mean()) if stable_noncritical.any() else None})
        return output, diagnostics

    @torch.no_grad()
    def augment_hierarchical(self, values: np.ndarray, class_ids: np.ndarray,
                             class_soft_masks: dict[int, np.ndarray], seed: int,
                             t_noncritical: int, batch_size: int = 256,
                             noise_structure: NoiseStructure = "iid") -> tuple[np.ndarray, dict[str, Any]]:
        """Apply a train-fitted class-specific mask while preserving the global RNG order."""
        values = np.asarray(values, dtype=np.float32); class_ids = np.asarray(class_ids, dtype=np.int64)
        if len(values) != len(class_ids): raise ValueError("hierarchical class ids must align with samples")
        if values.ndim != 3 or values.shape[1:] != (self.soft_mask.shape[0], (self.soft_mask.shape[1] - 1) * 2):
            raise ValueError("frequency diffusion input shape mismatch")
        if noise_structure != "iid": raise ValueError("HFSC first version freezes iid noise")
        present = set(map(int, np.unique(class_ids)))
        if not present.issubset(set(map(int, class_soft_masks))):
            raise ValueError("hierarchical soft masks do not cover all sample classes")
        variances = {}
        for kind in sorted(present):
            mask = torch.as_tensor(class_soft_masks[kind], dtype=torch.float32, device=self.device)
            variances[kind] = spectral_noise_variance(
                self.alpha_bars, self.soft_mask.shape[0], self.soft_mask.shape[1], "selective",
                self.t_uniform, self.preserve_dc, mask, self.t_critical, int(t_noncritical))
        mean = torch.as_tensor(self.statistics.mean, dtype=torch.float32, device=self.device)
        scale = torch.as_tensor(self.statistics.scale, dtype=torch.float32, device=self.device)
        maximum = torch.as_tensor(self.statistics.maximum_log_amplitude, dtype=torch.float32, device=self.device)
        generator = torch.Generator(device=self.device).manual_seed(int(seed)); output = np.empty_like(values)
        phase_error = []; reconstruction_error = []
        for start in range(0, len(values), batch_size):
            stop = min(start + batch_size, len(values)); base = torch.from_numpy(values[start:stop]).to(self.device)
            spectrum = torch.fft.rfft(base, dim=-1); log_amplitude = torch.log1p(torch.abs(spectrum)); phase = torch.angle(spectrum)
            standardized = (log_amplitude - mean) / scale
            variance = torch.stack([variances[int(kind)] for kind in class_ids[start:stop]])
            noise = torch.randn(standardized.shape, device=self.device, generator=generator)
            changed_standardized = (1 - variance).sqrt() * standardized + variance.sqrt() * noise
            changed_log_amplitude = torch.minimum((changed_standardized * scale + mean).clamp_min(0), maximum)
            if self.preserve_dc: changed_log_amplitude[:, :, 0] = log_amplitude[:, :, 0]
            amplitude = torch.expm1(changed_log_amplitude).clamp_min(0)
            changed = torch.fft.irfft(torch.polar(amplitude, phase), n=values.shape[-1], dim=-1)
            if not torch.isfinite(changed).all(): raise FloatingPointError("non-finite hierarchical spectral augmentation")
            output[start:stop] = changed.cpu().numpy(); roundtrip = torch.fft.rfft(changed, dim=-1)
            valid = amplitude > 1e-6
            angular = torch.atan2(torch.sin(torch.angle(roundtrip) - phase), torch.cos(torch.angle(roundtrip) - phase)).abs()
            phase_error.append(float(angular[valid].mean()) if valid.any() else 0.0)
            reconstructed = torch.fft.irfft(spectrum, n=values.shape[-1], dim=-1)
            reconstruction_error.append(float(torch.max(torch.abs(reconstructed - base))))
        class_budget = {kind: float(value.mean()) for kind, value in variances.items()}
        target_budget = float(self.variance("uniform").mean())
        if any(abs(value - target_budget) > 1e-6 for value in class_budget.values()):
            raise RuntimeError("HFSC class noise budget differs from Uniform")
        diagnostics = {"mode": "hierarchical_selective", "t_uniform": self.t_uniform,
                       "t_critical": self.t_critical, "t_noncritical": int(t_noncritical),
                       "expected_total_noise_budget": float(np.mean([class_budget[int(kind)] for kind in class_ids])),
                       "per_class_noise_budget": class_budget, "noise_structure": "iid",
                       "phase_preserved": self.preserve_phase, "mean_phase_error": float(np.mean(phase_error)),
                       "dc_preserved": self.preserve_dc,
                       "inverse_fft_reconstruction_max_error": float(np.max(reconstruction_error)),
                       "finite": bool(np.isfinite(output).all()),
                       "time_normalized_l1": normalized_difference(values, output),
                       "amplitude_min": float(output.min()), "amplitude_max": float(output.max())}
        return output, diagnostics
