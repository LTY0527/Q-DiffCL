import numpy as np

from frequency.budget_demand import (cross_group_shift_demand,
                                     group_bootstrap,
                                     normalized_log_spectrum,
                                     separability_difficulty_demand)


def _features():
    rng = np.random.default_rng(4)
    values = rng.normal(size=(24, 2, 8)).astype(np.float32)
    groups = np.repeat(np.asarray(["a", "b", "c", "d"], dtype=object), 6)
    stages = np.tile(np.asarray(["normal", "normal", "early", "early", "mature", "mature"], dtype=object), 4)
    return values, groups, stages


def test_proxies_are_finite_and_train_normalization_is_audited():
    values, groups, stages = _features(); z, audit = normalized_log_spectrum(values)
    shift = cross_group_shift_demand(z, groups, stages)
    difficulty = separability_difficulty_demand(z, stages)
    assert audit["fit_split"] == "train"
    assert np.isfinite(shift["score"])
    assert 0 <= difficulty["score"] <= 1
    assert set(shift["stage"]) == {"normal", "early", "mature"}


def test_group_bootstrap_is_reproducible():
    values, groups, stages = _features(); z, _ = normalized_log_spectrum(values)
    left = group_bootstrap(z, groups, stages, 8, 12)
    right = group_bootstrap(z, groups, stages, 8, 12)
    assert all(np.array_equal(left[key], right[key]) for key in left)
