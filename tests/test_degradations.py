import numpy as np

from degradations import apply_degradation


X = np.ones((8, 200), dtype=np.float32)


def test_index_based_determinism_and_seed_difference():
    a = apply_degradation(X, "mcar_missing", 0.3, 7, "window-1")
    b = apply_degradation(X, "mcar_missing", 0.3, 7, "window-1")
    c = apply_degradation(X, "mcar_missing", 0.3, 8, "window-1")
    assert np.array_equal(a.data, b.data)
    assert np.array_equal(a.observation_mask, b.observation_mask)
    assert not np.array_equal(a.observation_mask, c.observation_mask)
    assert abs((~a.observation_mask).mean() - 0.3) < 0.05


def test_block_is_contiguous_and_channel_dropout_count():
    block = apply_degradation(X, "block_missing", 0.25, 1, "a")
    for row in ~block.observation_mask:
        positions = np.flatnonzero(row); assert len(positions) == 50; assert np.all(np.diff(positions) == 1)
    dropout = apply_degradation(X, "channel_dropout", 3, 1, "a")
    assert int((~dropout.observation_mask).all(axis=1).sum()) == 3


def test_gaussian_snr_and_drift():
    noisy = apply_degradation(X, "gaussian_noise", 10, 3, "a")
    noise_power = np.mean((noisy.data - X) ** 2)
    assert abs(10 * np.log10(np.mean(X ** 2) / noise_power) - 10) < 0.5
    drift = apply_degradation(X, "linear_drift", 0.5, 3, "a", channel_std=np.full(8, 2.0))
    assert np.allclose(drift.data[:, -1] - X[:, -1], 1.0)


def test_missing_values_are_zero_and_masks_match():
    result = apply_degradation(X, "mcar_missing", 0.5, 4, "a")
    assert result.observation_mask.dtype == bool
    assert np.all(result.data[~result.observation_mask] == 0)
    assert np.array_equal(result.corruption_mask, ~result.observation_mask)

