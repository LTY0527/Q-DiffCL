import numpy as np
import pytest

from frequency.safe_capacity import safe_capacity


def test_safe_capacity_is_bounded_and_sample_adaptive():
    x = np.zeros((2, 1, 8), dtype=np.float32)
    x[0] = np.sin(2 * np.pi * np.arange(8) / 8)
    x[1] = np.sin(2 * np.pi * 3 * np.arange(8) / 8)
    mask = np.asarray([[0, 1, 0, 0, 0]], dtype=np.float32)
    result = safe_capacity(x, mask)
    assert np.all((result["rho"] >= 0) & (result["rho"] <= 1))
    assert result["rho"][0] < result["rho"][1]


def test_safe_capacity_rejects_unregistered_gamma():
    with pytest.raises(ValueError):
        safe_capacity(np.ones((1, 1, 8), np.float32), np.zeros((1, 5), np.float32), .75)
