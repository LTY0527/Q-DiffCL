import numpy as np
import torch
from pathlib import Path

from scripts.diagnose_frequency_selective_far import (
    _fit_replay,
    classify_cause, correlation_drift, score_profile,
)


def test_fit_replay_resumes_from_self_contained_checkpoint(monkeypatch):
    checkpoint = Path("unused-replay.pt")
    expected = {"name": "C1", "validation": {}, "test": {}, "checkpoint": checkpoint.as_posix()}
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {
        "model_state_dict": {}, "replay_result": expected,
    })

    actual = _fit_replay(
        "C1", {}, None, None, None, None, None, None, None, "cpu", checkpoint, True,
    )

    assert actual == expected


def test_correlation_drift_is_zero_for_identical_and_positive_for_broken_structure():
    rng = np.random.default_rng(2); base = rng.normal(size=(20, 3, 8)).astype(np.float32)
    labels = np.array([0] * 10 + [1] * 10)
    same = correlation_drift(base, base.copy(), labels)
    changed = base.copy(); changed[:, 1] = rng.normal(size=(20, 8))
    broken = correlation_drift(base, changed, labels)
    assert same["all"]["corr_drift"] == 0
    assert broken["all"]["corr_drift"] > 0


def test_score_profile_reports_both_flip_directions_and_threshold_band():
    labels = np.array([0, 0, 1, 1]); scores = np.array([.2, .8, .3, .9])
    result = score_profile(labels, scores, .5, .1)
    assert result["normal_to_fault"] == .5 and result["fault_to_normal"] == .5
    assert result["threshold_near_count"] == 0


def test_cause_classification_uses_validation_intensity_and_structure_only():
    def record(far, drift, normal_mean):
        return {"metrics": {"far": far}, "time_structure": {"normal": {"corr_drift": drift}},
                "score_profile": {"normal": {"mean": normal_mean}}}
    validation = {"C1": record(.04, .01, .2),
                  "candidates": {"3": record(.03, .015, .22), "5": record(.035, .018, .23),
                                 "8": record(.06, .02, .25)}}
    config = {"diagnosis": {"minimum_far_reduction_signal": .005, "minimum_structure_drift_ratio": 1.1}}
    category, checks = classify_cause(validation, config)
    assert category == "C. BOTH"
    assert all(checks.values())
