import numpy as np

from losses import (fit_robust_gain_calibration, relative_gain,
                    relative_quality, relative_semantic_quality,
                    semantic_score)


def test_relative_gain_sign_matches_improvement():
    gain = relative_gain(np.array([1.0, 1.0]), np.array([0.5, 1.5]))
    assert gain[0] > 0
    assert gain[1] < 0


def test_relative_calibration_is_fit_once_on_train_and_bounded():
    train = np.array([-1.0, -0.2, 0.0, 0.3, 1.0])
    calibration = fit_robust_gain_calibration(train, 0.1, 0.9)
    validation = relative_quality(np.array([-100.0, 0.0, 100.0]), calibration)
    assert calibration.median == 0.0
    assert np.all((validation >= 0.1) & (validation <= 0.9))


def test_validation_values_do_not_change_train_calibration():
    train = np.array([-0.5, -0.1, 0.2, 0.8])
    first = fit_robust_gain_calibration(train, 0.1, 0.9)
    _ = relative_quality(np.array([-1000.0, 1000.0]), first)
    second = fit_robust_gain_calibration(train, 0.1, 0.9)
    assert first == second


def test_semantic_score_uses_prediction_distance_without_labels():
    clean = np.array([[0.9, 0.1], [0.5, 0.5]])
    restored = np.array([[0.9, 0.1], [0.9, 0.1]])
    score = semantic_score(clean, restored, floor=0.1)
    assert score[0] == 1.0
    assert 0.1 <= score[1] < 1.0


def test_relative_semantic_quality_is_stable_and_bounded():
    combined = relative_semantic_quality(np.array([0.2, 0.8]), np.array([0.1, 1.0]), 0.1)
    assert np.allclose(combined, [0.11, 0.8])
    assert np.isfinite(combined).all()
