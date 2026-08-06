from __future__ import annotations

import numpy as np
import pytest

from scripts.run_stage_frequency_diffusion_mvp import (
    METHODS, detection_delays, early_fault_recall, select_t_noncritical,
)


CONFIG = {"protocol": {"fault_onset": {"testing": 161}},
          "detection": {"sustained_alarm_windows": 3}}


def test_timestep_selection_is_validation_only_and_deterministic():
    records = [{"selection_split": "validation", "t_noncritical": value, "uniform_budget": .1,
                "metrics": {"finite": True, "critical_fisher_retention": score,
                            "early_frequency_retention": score, "expected_total_noise_budget": .1}}
               for value, score in ((3, .8), (5, 1.1), (8, .9))]
    selected, returned = select_t_noncritical(records)
    assert selected == 5 and returned is records
    records[0]["selection_split"] = "test"
    with pytest.raises(ValueError, match="validation"):
        select_t_noncritical(records)


def test_early_recall_uses_stage_labels_not_absolute_samples():
    result = early_fault_recall(np.array([1, 0, 1, 0]), np.array(["early", "early", "middle", "prefault"]))
    assert result == {"count": 2, "recall": .5}


def test_detection_delay_requires_sustained_alarm_and_reports_miss():
    run1, run2 = "testing:fault_01:0001", "testing:fault_02:0001"
    bundle = {"run_uid": np.array([run1] * 5 + [run2] * 5),
              "end_sample": np.array([160, 224, 240, 256, 272] * 2)}
    prediction = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])
    result = detection_delays(bundle, prediction, CONFIG)
    assert result["detected_runs"] == 1 and result["missed_runs"] == 1
    assert result["per_run"][run1]["delay_samples"] == 111
    assert result["per_run"][run2] == {"detected": False, "delay_samples": None}


def test_method_names_define_only_positive_view_variable():
    assert METHODS == ("C0 传统增强", "C1 统一频谱扩散", "C2 频率选择性扩散")

