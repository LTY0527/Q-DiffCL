import numpy as np
import pytest

from trainers.balanced import PositiveSafeBatchSampler, sqrt_inverse_frequency_weights


def test_positive_safe_sampler_has_pairs_and_is_reproducible():
    labels = np.repeat(np.arange(4), 20)
    first = PositiveSafeBatchSampler(labels, 4, 3, 5, seed=42)
    second = PositiveSafeBatchSampler(labels, 4, 3, 5, seed=42)
    batches = list(first)
    assert batches == list(second)
    for batch in batches:
        assert np.all(np.bincount(labels[batch], minlength=4) == 3)


def test_sampler_rejects_unbounded_oversampling():
    labels = np.r_[np.zeros(2, dtype=int), np.ones(100, dtype=int)]
    with pytest.raises(ValueError, match="oversampling"):
        PositiveSafeBatchSampler(labels, 2, 4, 10, seed=7, max_oversampling=3)


def test_sqrt_inverse_weights_are_finite_and_train_label_derived():
    weights = sqrt_inverse_frequency_weights(np.asarray([0] * 100 + [1] * 25))
    assert np.isfinite(weights).all()
    assert weights[1] / weights[0] == pytest.approx(2.0)
