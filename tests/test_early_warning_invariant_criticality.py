import numpy as np
import pytest

from frequency import build_early_warning_criticality, onset_horizons
from metrics.fixed_far import calibrate_fixed_far, fixed_far_metrics


SETTINGS = {"critical_ratio": .3, "weight_discriminative": .5, "weight_early": .3,
            "weight_run_stability": .2, "bootstrap_repeats": 16, "bootstrap_seed": 5,
            "horizon_count": 8, "lead_decay": .35}


def _fixture():
    rng = np.random.default_rng(21); rows = []; uids = []; labels = []; stages = []; horizons = []
    for index, kind in enumerate((0, 1, 1, 1)):
        uid = f"training:fault_{kind:02d}:run{index}"
        if kind == 0:
            for _ in range(8):
                rows.append(rng.normal(0, .2, (2, 5))); uids.append(uid); labels.append(0)
                stages.append("prefault"); horizons.append(0)
        else:
            for horizon in range(1, 9):
                value = rng.normal(0, .2, (2, 5)); value[0, 1] += 5 * np.exp(-.4 * (horizon - 1))
                value[1, 4] += .2 * horizon
                rows.append(value); uids.append(uid); labels.append(1)
                stages.append("early" if horizon <= 4 else "stable"); horizons.append(horizon)
    features = np.asarray(rows, np.float32)
    return features, {"run_uid": np.asarray(uids), "labels": np.asarray(labels)}, np.asarray(stages), np.asarray(horizons)


def test_onset_horizons_are_relative_and_exclude_transition_or_late_windows():
    bundle = {"run_uid": np.array(["training:fault_01:a"] * 5),
              "labels": np.array([0, 1, 1, 1, 1]),
              "start_sample": np.array([0, 20, 21, 37, 149])}
    assert onset_horizons(bundle, {"training": 21}, 16).tolist() == [0, 0, 1, 2, 0]


def test_ewic_builds_weighted_reliable_trajectory_and_preserves_ds():
    features, bundle, stages, horizons = _fixture()
    first = build_early_warning_criticality(features, bundle, stages, horizons, SETTINGS)
    second = build_early_warning_criticality(features, bundle, stages, horizons, SETTINGS)
    assert len(first["horizon_fisher"]) == 8 and np.isclose(first["lead_weights"].sum(), 1)
    assert np.all(np.diff(first["lead_weights"]) < 0)
    assert first["horizon_coverage"]["1"]["runs_or_wells"] == 3
    assert first["horizon_normalized"][0][0, 1] > first["horizon_normalized"][7][0, 1]
    assert np.array_equal(first["r1"]["discriminative"], second["r1"]["discriminative"])
    assert np.array_equal(first["r1"]["stability"], second["r1"]["stability"])
    assert np.array_equal(first["early_reliability"], second["early_reliability"])
    assert first["hard_mask"].sum() == 3 and first["bootstrap_unit_count"] == 3


def test_ewic_rejects_searchable_method_parameters():
    features, bundle, stages, horizons = _fixture()
    for key, value in (("horizon_count", 7), ("lead_decay", .2), ("critical_ratio", .4), ("weight_early", .4)):
        settings = dict(SETTINGS); settings[key] = value
        with pytest.raises(ValueError):
            build_early_warning_criticality(features, bundle, stages, horizons, settings)


def test_fixed_far_uses_validation_normal_only_and_frozen_points():
    validation_y = np.array([0, 0, 0, 0, 1]); validation_scores = np.array([.1, .2, .3, .4, .99])
    test_y = np.array([0, 0, 1, 1]); test_scores = np.array([.2, .5, .35, .8])
    result = fixed_far_metrics(validation_y, validation_scores, test_y, test_scores)
    assert result["far_5pct"]["threshold"] == .4
    assert result["far_5pct"]["validation_observed_far"] == .25
    assert result["far_5pct"]["observed_far"] == .5
    assert result["far_5pct"]["fault_recall"] == .5
    with pytest.raises(ValueError): calibrate_fixed_far(validation_scores[:4], .02)
