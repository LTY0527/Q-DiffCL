import numpy as np

from analysis.drfd_failure_audit import first_alarm


def test_first_alarm_distinguishes_transient_and_sustained_three_window_alarm():
    horizons = np.asarray([-2, -1, 0, 1, 2, 3, 4])
    prediction = np.asarray([0, 0, 1, 0, 1, 1, 1])
    assert first_alarm(prediction, horizons, 1) == 0
    assert first_alarm(prediction, horizons, 3) == 2
    assert first_alarm(np.zeros_like(prediction), horizons, 3) is None
