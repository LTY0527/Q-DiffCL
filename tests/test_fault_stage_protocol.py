import numpy as np

from frequency import classify_stage, fault_stages


CONFIG = {"protocol": {"fault_onset": {"training": 21, "testing": 161},
                        "window_length": 64, "stride": 16},
          "stage": {"early_horizon_windows": 4, "middle_horizon_windows": 12}}


def test_stage_uses_true_split_onset_and_excludes_transition():
    assert classify_stage("training:normal:0001", 1, 64, CONFIG) == "prefault"
    assert classify_stage("testing:fault_01:0001", 81, 144, CONFIG) == "prefault"
    assert classify_stage("testing:fault_01:0001", 113, 176, CONFIG) == "transition"


def test_first_four_complete_fault_windows_are_early_without_off_by_one():
    stages = [classify_stage("testing:fault_01:0001", start, start + 63, CONFIG)
              for start in (161, 177, 193, 209, 225)]
    assert stages == ["early", "early", "early", "early", "middle"]
    training = [classify_stage("training:fault_01:0001", start, start + 63, CONFIG)
                for start in (33, 49, 65, 81, 97)]
    assert training == ["early", "early", "early", "early", "middle"]


def test_middle_stable_boundary_and_split_rules_are_identical_in_progress_space():
    assert classify_stage("testing:fault_01:0001", 337, 400, CONFIG) == "middle"
    assert classify_stage("testing:fault_01:0001", 353, 416, CONFIG) == "stable"
    assert classify_stage("training:fault_01:0001", 225, 288, CONFIG) == "stable"


def test_fixed_bundle_rejects_transition_and_stage_is_not_a_threshold_input():
    bundle = {"run_uid": np.array(["testing:fault_01:0001"]),
              "start_sample": np.array([113]), "end_sample": np.array([176]),
              "labels": np.array([1])}
    try:
        fault_stages(bundle, CONFIG)
    except RuntimeError as error:
        assert "transition" in str(error)
    else:
        raise AssertionError("transition window must be rejected")

