import numpy as np
import pytest
import torch

from diffusion import (DiffusionSchedule, FrequencyForwardDiffusion,
                       fit_spectral_statistics, spectral_noise_variance)


def _augmenter(values, mask=None):
    mask = np.linspace(0, 1, values.shape[1] * 33).reshape(values.shape[1], 33) if mask is None else mask
    stats = fit_spectral_statistics(values, .999, "train")
    schedule = DiffusionSchedule.cosine(50, "cpu")
    return FrequencyForwardDiffusion(stats, schedule.alpha_bars, mask, 3, 1, True, True, "cpu")


def test_rfft_irfft_reconstruction_is_finite_and_accurate():
    values = np.random.default_rng(1).normal(size=(4, 3, 64)).astype(np.float32)
    reconstructed = torch.fft.irfft(torch.fft.rfft(torch.from_numpy(values), dim=-1), n=64, dim=-1)
    assert reconstructed.shape == (4, 3, 64)
    assert torch.isfinite(reconstructed).all()
    assert float(torch.max(torch.abs(reconstructed - torch.from_numpy(values)))) < 1e-5


def test_spectral_statistics_are_train_only():
    values = np.random.default_rng(1).normal(size=(4, 3, 64)).astype(np.float32)
    assert fit_spectral_statistics(values).fit_split == "train"
    with pytest.raises(ValueError, match="only be fitted on train"):
        fit_spectral_statistics(values, split="validation")


def test_c1_uniform_strength_and_c2_ordered_budget_matching():
    schedule = DiffusionSchedule.cosine(50, "cpu")
    mask = torch.linspace(0, 1, 3 * 33).reshape(3, 33)
    uniform = spectral_noise_variance(schedule.alpha_bars, 3, 33, "uniform", 3, True)
    selective = spectral_noise_variance(schedule.alpha_bars, 3, 33, "selective", 3, True, mask, 1, 8)
    assert torch.all(uniform[:, 1:] == uniform[0, 1]) and torch.all(uniform[:, 0] == 0)
    assert abs(float(uniform.mean() - selective.mean())) < 1e-6
    non_dc = torch.ones_like(mask, dtype=torch.bool); non_dc[:, 0] = False
    assert float(selective[(mask >= .8) & non_dc].mean()) < float(selective[(mask <= .2) & non_dc].mean())


def test_preserve_phase_dc_determinism_and_finite_output():
    values = np.random.default_rng(2).normal(size=(6, 3, 64)).astype(np.float32)
    augmenter = _augmenter(values)
    first, diagnostic = augmenter.augment(values, "selective", 17, 5, batch_size=3)
    second, _ = augmenter.augment(values, "selective", 17, 5, batch_size=3)
    different, _ = augmenter.augment(values, "selective", 18, 5, batch_size=3)
    assert np.array_equal(first, second) and not np.array_equal(first, different)
    assert np.isfinite(first).all() and diagnostic["finite"]
    assert diagnostic["mean_phase_error"] < 1e-4
    base_dc = np.abs(np.fft.rfft(values, axis=-1)[:, :, 0])
    changed_dc = np.abs(np.fft.rfft(first, axis=-1)[:, :, 0])
    assert np.allclose(base_dc, changed_dc, atol=1e-4)


def test_c1_c2_use_same_total_budget_for_all_candidates():
    values = np.random.default_rng(3).normal(size=(5, 2, 64)).astype(np.float32)
    augmenter = _augmenter(values)
    target = float(augmenter.variance("uniform").mean())
    for timestep in (3, 5, 8):
        assert abs(float(augmenter.variance("selective", timestep).mean()) - target) < 1e-6
