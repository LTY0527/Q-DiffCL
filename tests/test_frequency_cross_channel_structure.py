import numpy as np
import pytest
import torch

from diffusion import (DiffusionSchedule, FrequencyForwardDiffusion,
                       constrain_channel_budget, fit_spectral_statistics)
from frequency import fit_cross_channel_spectral_structure


def _correlated_augmenter(values):
    mask = np.linspace(0, 1, values.shape[1] * 33).reshape(values.shape[1], 33)
    structure = fit_cross_channel_spectral_structure(values, .25, 1e-5, True, "train")
    return FrequencyForwardDiffusion(
        fit_spectral_statistics(values), DiffusionSchedule.cosine(50, "cpu").alpha_bars,
        mask, 3, 1, True, True, "cpu", structure, 1.20,
    )


def test_structure_is_train_only_symmetric_stable_and_frequency_shaped():
    values = np.random.default_rng(1).normal(size=(24, 4, 64)).astype(np.float32)
    structure = fit_cross_channel_spectral_structure(values, .25, 1e-5, True)
    assert structure.fit_split == "train"
    assert structure.covariance.shape == (33, 4, 4)
    assert np.allclose(structure.covariance, structure.covariance.transpose(0, 2, 1), atol=1e-6)
    assert np.linalg.eigvalsh(structure.covariance).min() >= 1e-5 - 1e-6
    assert np.allclose(np.diagonal(structure.covariance, axis1=1, axis2=2), 1, atol=1e-6)
    with pytest.raises(ValueError, match="only be fitted on train"):
        fit_cross_channel_spectral_structure(values, split="validation")


def test_channel_budget_caps_each_channel_and_preserves_total():
    variance = torch.tensor([[0., .8, .8], [0., .1, .1], [0., .1, .1]])
    reference = torch.tensor([[0., 1 / 3, 1 / 3]]).repeat(3, 1)
    constrained = constrain_channel_budget(variance, reference, 1.2)
    assert torch.all(constrained.sum(1) <= reference.sum(1) * 1.2 + 1e-6)
    assert torch.isclose(constrained.sum(), variance.sum(), atol=1e-6)


def test_correlated_noise_is_reproducible_seed_sensitive_finite_and_budgeted():
    rng = np.random.default_rng(2)
    latent = rng.normal(size=(32, 1, 64)).astype(np.float32)
    values = np.concatenate([latent, .8 * latent + .2 * rng.normal(size=latent.shape),
                             rng.normal(size=latent.shape)], axis=1).astype(np.float32)
    augmenter = _correlated_augmenter(values)
    first, diagnostic = augmenter.augment(values, "selective", 17, 8, 8, "correlated", True)
    second, _ = augmenter.augment(values, "selective", 17, 8, 8, "correlated", True)
    different, _ = augmenter.augment(values, "selective", 18, 8, 8, "correlated", True)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)
    assert np.isfinite(first).all() and diagnostic["finite"]
    assert diagnostic["noise_structure"] == "correlated"
    assert diagnostic["maximum_observed_channel_budget_ratio"] <= 1.2 + 1e-6
    assert diagnostic["dc_preserved"] and diagnostic["phase_preserved"]


def test_iid_path_remains_available_for_r0_r1():
    values = np.random.default_rng(3).normal(size=(8, 3, 64)).astype(np.float32)
    augmenter = _correlated_augmenter(values)
    _, diagnostic = augmenter.augment(values, "selective", 9, 5, noise_structure="iid")
    assert diagnostic["noise_structure"] == "iid"
    assert not diagnostic["channel_budget_applied"]
