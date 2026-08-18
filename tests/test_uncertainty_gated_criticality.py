import numpy as np
import pytest
import torch

from diffusion import DiffusionSchedule, FrequencyForwardDiffusion, fit_spectral_statistics
from frequency import assignment_confidence, build_uncertainty_gated_criticality


SETTINGS = {"critical_ratio": .3, "weight_discriminative": .5, "weight_early": .3,
            "weight_run_stability": .2, "bootstrap_repeats": 64, "bootstrap_seed": 19}


def _fixture():
    rng = np.random.default_rng(12); features = []; labels = []; stages = []; units = []; strata = []; run_uid = []
    for kind in (0, 1, 2):
        for run in range(3):
            uid = f"training:fault_{kind:02d}:r{run}"
            for window in range(5):
                value = rng.normal(0, .3, (2, 5)); label = int(kind != 0 and window >= 2)
                if label: value[kind - 1, kind] += 3
                features.append(value); labels.append(label); stages.append("early" if label and window < 4 else "stable" if label else "prefault")
                units.append(uid); strata.append(kind); run_uid.append(uid)
    return np.asarray(features, np.float32), {"labels": np.asarray(labels), "run_uid": np.asarray(run_uid)}, np.asarray(stages), np.asarray(units), np.asarray(strata)


def test_assignment_confidence_is_bidirectional():
    probability = np.array([0, .25, .5, .75, 1])
    assert np.allclose(assignment_confidence(probability), [1, .5, 0, .5, 1])
    with pytest.raises(ValueError): assignment_confidence(np.array([1.1]))


def test_complete_r1_bootstrap_is_train_unit_stratified_and_reproducible():
    features, bundle, stages, units, strata = _fixture()
    first = build_uncertainty_gated_criticality(features, bundle, stages, units, strata, SETTINGS)
    second = build_uncertainty_gated_criticality(features, bundle, stages, units, strata, SETTINGS)
    assert first["fit_split"] == "train" and first["bootstrap_repeats"] == 64
    assert first["bootstrap_unit_count"] == 9 and first["stratified_unit_counts"] == {"0": 3, "1": 3, "2": 3}
    assert np.array_equal(first["selection_probability"], second["selection_probability"])
    assert np.all((first["assignment_confidence"] >= 0) & (first["assignment_confidence"] <= 1))
    assert "complete D/E/S" in first["bootstrap_scope"]


def test_uncertainty_gated_timestep_endpoints_and_budget_matching():
    values = np.random.default_rng(4).normal(size=(6, 2, 64)).astype(np.float32)
    mask = np.linspace(0, 1, 66).reshape(2, 33).astype(np.float32)
    schedule = DiffusionSchedule.cosine(50, "cpu"); augmenter = FrequencyForwardDiffusion(
        fit_spectral_statistics(values), schedule.alpha_bars, mask, 3, 1)
    r1 = augmenter.variance("selective", 5)
    uniform = augmenter.variance("uniform")
    confident = augmenter.variance("uncertainty_gated", 5, assignment_confidence=np.ones_like(mask))
    uncertain = augmenter.variance("uncertainty_gated", 5, assignment_confidence=np.zeros_like(mask))
    assert torch.allclose(r1, confident) and torch.allclose(uniform, uncertain)
    mixed = np.full_like(mask, .5)
    variance = augmenter.variance("uncertainty_gated", 5, assignment_confidence=mixed)
    assert abs(float(variance.mean() - uniform.mean())) < 1e-6
    changed, audit = augmenter.augment(values, "uncertainty_gated", 8, 5, 3, assignment_confidence=mixed)
    assert np.isfinite(changed).all() and audit["timestep_mean_absolute_change_from_r1"] > 0


def test_uncertainty_rejects_window_level_or_unfrozen_configuration():
    features, bundle, stages, units, strata = _fixture()
    bad_units = np.asarray([f"window-{index}" for index in range(len(features))])
    # A caller must provide declared aggregate units; unique-per-window identifiers are rejected by audit policy.
    with pytest.raises(ValueError):
        build_uncertainty_gated_criticality(features, bundle, stages, bad_units, strata, SETTINGS)
    settings = dict(SETTINGS); settings["bootstrap_repeats"] = 32
    with pytest.raises(ValueError):
        build_uncertainty_gated_criticality(features, bundle, stages, units, strata, settings)
