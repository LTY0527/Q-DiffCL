import numpy as np
import pytest
import torch

from augmentations import domain_budget_route
from baselines.external_augmentations import traditional_view
from diffusion import (DiffusionSchedule, FrequencyForwardDiffusion,
                       fit_spectral_statistics, spectral_noise_variance)


def _inputs():
    clean = np.arange(48, dtype=np.float32).reshape(3, 2, 8) / 10
    diffused = clean + .125
    ids = np.asarray(["a", "b", "c"])
    return clean, diffused, ids


def test_dcbr_rho_zero_exactly_matches_frozen_scaling():
    clean, diffused, ids = _inputs()
    actual, audit = domain_budget_route(clean, diffused, ids, 0, .05, 42)
    expected = traditional_view(clean, ids, "SCALING", 42, 0, .05)
    assert np.array_equal(actual, expected)
    assert audit["rho_zero_exact_scaling_protocol"]


def test_dcbr_rho_one_exactly_matches_final_view():
    clean, diffused, ids = _inputs()
    actual, audit = domain_budget_route(clean, diffused, ids, 1, .05, 42)
    assert np.array_equal(actual, diffused)
    assert audit["rho_one_exact_final_view"]


def test_dcbr_intermediate_is_finite_reproducible_and_bounded():
    clean, diffused, ids = _inputs()
    left, audit = domain_budget_route(clean, diffused, ids, .5, .05, 9)
    right, _ = domain_budget_route(clean, diffused, ids, .5, .05, 9)
    assert np.array_equal(left, right)
    assert np.isfinite(left).all() and audit["effective_scaling_std"] == .025
    with pytest.raises(ValueError):
        domain_budget_route(clean, diffused, ids, 1.1, .05, 9)


def test_dcbr_rho_one_regresses_to_selective_diffusion_numerically():
    rng = np.random.default_rng(5); values = rng.normal(size=(6, 2, 8)).astype(np.float32)
    soft = np.asarray([[.2, .4, .6, .8, .3], [.7, .1, .5, .9, .2]], np.float32)
    stats = fit_spectral_statistics(values, split="train")
    schedule = DiffusionSchedule.cosine(50, "cpu")
    augmenter = FrequencyForwardDiffusion(stats, schedule.alpha_bars, soft, 3, 1, True, True, "cpu")
    final = spectral_noise_variance(
        schedule.alpha_bars, 2, 5, "selective", 3, True, torch.as_tensor(soft), 1, 5)
    expected, expected_audit = augmenter.augment(values, "selective", 77, 5, 4)
    overridden, override_audit = augmenter.augment(
        values, "budget_scaled_selective", 77, 5, 4, variance_override=final)
    actual, _ = domain_budget_route(values, overridden, np.arange(len(values)), 1, .05, 8)
    assert np.array_equal(actual, expected)
    assert expected_audit["phase_preserved"] and override_audit["dc_preserved"]
    assert torch.all(final[:, 0] == 0)
