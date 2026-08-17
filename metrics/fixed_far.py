from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import recall_score


OPERATING_POINTS = (.01, .05)


def calibrate_fixed_far(normal_validation_scores: np.ndarray, target_far: float) -> float:
    scores = np.asarray(normal_validation_scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("fixed-FAR calibration requires finite validation-normal scores")
    if float(target_far) not in OPERATING_POINTS:
        raise ValueError("EWIC fixed-FAR operating points are frozen at 1% and 5%")
    try:
        return float(np.quantile(scores, 1 - target_far, method="higher"))
    except TypeError:
        return float(np.quantile(scores, 1 - target_far, interpolation="higher"))


def fixed_far_metrics(validation_y: np.ndarray, validation_scores: np.ndarray,
                      test_y: np.ndarray, test_scores: np.ndarray) -> dict[str, Any]:
    validation_y = np.asarray(validation_y); test_y = np.asarray(test_y)
    result = {}
    for target in OPERATING_POINTS:
        threshold = calibrate_fixed_far(np.asarray(validation_scores)[validation_y == 0], target)
        validation_normal_prediction = np.asarray(validation_scores)[validation_y == 0] >= threshold
        prediction = (np.asarray(test_scores) >= threshold).astype(np.int64)
        result[f"far_{int(target * 100)}pct"] = {
            "target_far": target, "threshold": threshold,
            "validation_observed_far": float(validation_normal_prediction.mean()),
            "observed_far": float(prediction[test_y == 0].mean()),
            "fault_recall": float(recall_score(test_y, prediction, pos_label=1, zero_division=0)),
            "prediction": prediction,
            "calibration_split": "validation_normal_only",
        }
    return result
