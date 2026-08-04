import numpy as np

from metrics import (classification_metrics, detection_delay, drop_rate,
                     performance_retention, representation_diagnostics,
                     select_binary_threshold, supcon_gain)


def test_classification_and_relative_metrics():
    y = np.array([0, 0, 1, 1]); prediction = np.array([0, 1, 1, 0]); probability = np.array([0.1, 0.7, 0.8, 0.2])
    result = classification_metrics(y, prediction, probability)
    assert result["accuracy"] == 0.5 and result["macro_f1"] == 0.5
    assert result["far"] == 0.5 and result["mdr"] == 0.5
    assert performance_retention(0.6, 0.8) == pytest.approx(0.75)
    assert drop_rate(0.8, 0.6) == pytest.approx(0.25)
    assert performance_retention(1, 0) is None
    assert supcon_gain(0.7, 0.6) == pytest.approx(0.1)
    assert 0 <= select_binary_threshold(y, probability) <= 1
    assert detection_delay(np.arange(8), np.array([0, 0, 0, 0, 0, 0, 1, 1]), 5) == 1
    assert detection_delay(np.arange(8), np.zeros(8), 5) is None


def test_representation_diagnostics_are_finite():
    clean = np.array([[0., 0.], [0., 1.], [3., 3.], [3., 4.]])
    degraded = clean + 0.1
    result = representation_diagnostics(clean, degraded, np.array([0, 0, 1, 1]))
    assert result["effective_rank"] > 0
    assert result["fisher_ratio"] > 0


import pytest
