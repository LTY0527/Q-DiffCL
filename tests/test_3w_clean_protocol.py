import numpy as np

from scripts.run_3w_clean_baseline import PRIMARY_CLASSES, stage_and_target


def test_transient_offset_maps_to_fault_not_new_class():
    stages, targets = stage_and_target(np.asarray([np.nan, 0, 105, 5]), 5, 100)
    assert stages.tolist() == ["unlabeled", "normal", "early", "established"]
    assert targets.tolist() == [-1, 0, PRIMARY_CLASSES.index(5), PRIMARY_CLASSES.index(5)]


def test_normal_instance_maps_only_raw_zero():
    stages, targets = stage_and_target(np.asarray([np.nan, 0, 100]), 0, 100)
    assert stages.tolist() == ["unlabeled", "normal", "unlabeled"]
    assert targets.tolist() == [-1, 0, -1]
