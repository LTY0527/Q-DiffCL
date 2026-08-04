import numpy as np

from scripts.audit_quality_definition import (fault_type, safe_spearman,
                                              standardized_mean_difference)


def test_fault_type_is_derived_from_run_uid_without_label_input():
    assert fault_type("training:fault_07:0003") == 7
    assert fault_type("testing:normal:0001") == 0


def test_standardized_mean_difference_direction():
    values = np.array([0.0, 0.2, 1.0, 1.2])
    labels = np.array([0, 0, 1, 1])
    assert standardized_mean_difference(values, labels) > 0


def test_safe_spearman_handles_missing_onset_positions():
    value = safe_spearman(np.array([1.0, 2.0, 3.0, 4.0]), np.array([np.nan, 2.0, 3.0, 4.0]))
    assert value == 1.0
