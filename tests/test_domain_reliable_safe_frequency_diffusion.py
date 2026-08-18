import numpy as np
import pytest
import torch

from diffusion import (DiffusionSchedule, FrequencyForwardDiffusion,
                       asymmetric_safe_timestep, constrained_safe_variance,
                       fit_spectral_statistics)
from frequency import (build_tep_stratified_run_bootstrap,
                       build_three_w_leave_one_well_out, percentile_ranks,
                       summarize_rank_distribution)


SETTINGS = {"critical_ratio": .3, "weight_discriminative": .5, "weight_early": .3,
            "weight_run_stability": .2, "bootstrap_repeats": 64, "bootstrap_seed": 19}


def _fixture(unit_prefix="WELL"):
    rng = np.random.default_rng(12)
    features = []; labels = []; stages = []; units = []; run_uids = []
    for kind in (0, 1, 2):
        for run in range(3):
            unit = f"{unit_prefix}-{kind}-{run}"
            uid = f"training:fault_{kind:02d}:{unit}"
            for window in range(5):
                value = rng.normal(0, .3, (2, 5)); label = int(kind != 0 and window >= 2)
                if label: value[kind - 1, kind] += 3
                features.append(value); labels.append(label)
                stages.append("early" if label and window < 4 else "stable" if label else "prefault")
                units.append(unit); run_uids.append(uid)
    bundle = {"labels": np.asarray(labels), "run_uid": np.asarray(run_uids)}
    return np.asarray(features, np.float32), bundle, np.asarray(stages), np.asarray(units)


def test_percentile_rank_and_q25_q75_partition():
    values = np.asarray([[[0., 1., 2., 3.]], [[0., 2., 1., 3.]]])
    ranks = percentile_ranks(values)
    assert ranks.min() == 0 and ranks.max() == 1
    summary = summarize_rank_distribution(values)
    assert np.all(summary["reliable_critical"] | summary["ambiguous"] | summary["reliable_noncritical"])
    assert not np.any(summary["reliable_critical"] & summary["reliable_noncritical"])


def test_three_w_jackknife_uses_wells_and_train_only_arrays():
    features, bundle, stages, wells = _fixture()
    result = build_three_w_leave_one_well_out(features, bundle, stages, wells, SETTINGS)
    assert result["fit_split"] == "train" and result["replicate_count"] == 9
    assert result["resampling"] == "leave-one-WELL-out"
    assert {row["omitted_well"] for row in result["profiles"]} == set(wells)
    with pytest.raises(ValueError):
        build_three_w_leave_one_well_out(features, bundle, stages,
                                         np.asarray([f"window-{i}" for i in range(len(features))]), SETTINGS)


def test_tep_bootstrap_is_run_stratified_and_frozen_to_64():
    features, bundle, stages, _ = _fixture("run")
    result = build_tep_stratified_run_bootstrap(features, bundle, stages, SETTINGS)
    assert result["replicate_count"] == 64
    assert result["stratified_unit_counts"] == {"0": 3, "1": 3, "2": 3}
    changed = dict(SETTINGS); changed["bootstrap_repeats"] = 8
    with pytest.raises(ValueError):
        build_tep_stratified_run_bootstrap(features, bundle, stages, changed)


def test_safe_timestep_and_constrained_budget_invariants():
    soft = np.asarray([[1., .8, .5, .2, 0.]])
    q25 = np.asarray([[.9, .6, .2, .1, .0]])
    q75 = np.asarray([[.95, .8, .7, .4, .0]])
    reliable_noncritical = q75 < .7
    ambiguous = (q25 < .7) & (q75 >= .7)
    r1, safe = asymmetric_safe_timestep(soft, q75)
    assert np.allclose(safe[r1 <= 3], r1[r1 <= 3])
    schedule = DiffusionSchedule.cosine(50, "cpu")
    variance, audit = constrained_safe_variance(schedule.alpha_bars, soft, q25, q75,
                                                 reliable_noncritical, ambiguous, preserve_dc=False)
    assert audit["protected_timestep_not_increased"]
    assert audit["protected_variance_not_increased"]
    assert audit["ambiguous_variance_not_increased"]
    assert audit["extra_only_reliable_noncritical"]
    assert audit["budget_adjustment_only_reliable_noncritical"]
    assert audit["maximum_variance_respected"] and audit["finite"]
    assert torch.isfinite(variance).all()
    assert audit["budget_error_fraction"] < 1e-5


def test_budget_error_reports_infeasible_safe_allocation():
    schedule = DiffusionSchedule.cosine(50, "cpu")
    soft = np.ones((2, 4)); q25 = np.ones((2, 4)); q75 = np.ones((2, 4))
    variance, audit = constrained_safe_variance(schedule.alpha_bars, soft, q25, q75,
                                                 np.zeros_like(soft, bool), np.zeros_like(soft, bool), False)
    assert audit["budget_error_fraction"] > .02
    assert audit["eligible_bin_count"] == 0 and torch.isfinite(variance).all()


def test_audited_variance_override_preserves_default_paths():
    values = np.random.default_rng(8).normal(size=(4, 2, 64)).astype(np.float32)
    soft = np.linspace(0, 1, 66).reshape(2, 33).astype(np.float32)
    schedule = DiffusionSchedule.cosine(50, "cpu")
    augmenter = FrequencyForwardDiffusion(fit_spectral_statistics(values), schedule.alpha_bars, soft, 3, 1)
    before = augmenter.variance("selective", 5)
    override = augmenter.variance("uniform")
    changed, audit = augmenter.augment(values, "domain_reliable_safe", 4, 5, 2,
                                       variance_override=override)
    assert torch.equal(before, augmenter.variance("selective", 5))
    assert np.isfinite(changed).all() and audit["mode"] == "domain_reliable_safe"
    with pytest.raises(ValueError): augmenter.variance("domain_reliable_safe")
