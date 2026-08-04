import numpy as np
import pytest

from datasets.protocol import (Run, Standardizer, label_run, make_run_uid,
                               split_runs, split_training_runs_stratified,
                               window_runs)


def faulty_run() -> Run:
    return Run("faulty", np.arange(20, dtype=float).reshape(10, 2), np.arange(10), 2, 5)


def test_split_manifest_is_disjoint_and_reproducible():
    first = split_runs([str(i) for i in range(10)], (0.6, 0.2, 0.2), 4)
    second = split_runs([str(i) for i in range(10)], (0.6, 0.2, 0.2), 4)
    assert first == second
    assert not set(first.train) & set(first.validation)
    assert not set(first.train) & set(first.test)


def test_fault_boundary_labels_and_normal_run():
    assert label_run(faulty_run()).tolist() == [0] * 5 + [1] * 5
    normal = Run("normal", np.zeros((10, 2)), np.arange(10), 0, None)
    assert not label_run(normal).any()


@pytest.mark.parametrize("policy,expected", [("exclude_transition", 2), ("label_by_last_step", 4), ("transition_class", 4)])
def test_transition_policies(policy, expected):
    x, y, ids, stats = window_runs([faulty_run()], 4, 2, policy)
    assert len(x) == expected == len(y) == len(ids)
    assert stats["transition_windows"] == 2
    assert stats["excluded_transition_windows"] == (2 if policy == "exclude_transition" else 0)
    assert sum(stats["class_distribution_before_exclusion"].values()) == 4
    assert sum(stats["class_distribution_after_exclusion"].values()) == expected
    if policy == "transition_class": assert (y >= 0).all()


def test_fault_ratio_policy_requires_threshold_and_labels():
    with pytest.raises(ValueError): window_runs([faulty_run()], 4, 2, "label_by_fault_ratio")
    _, y, _, _ = window_runs([faulty_run()], 4, 2, "label_by_fault_ratio", 0.5)
    assert y.tolist() == [0, 0, 1, 1]


def test_standardizer_uses_explicit_fit_data_only():
    train = np.array([[0.0, 2.0], [2.0, 4.0]])
    scaler = Standardizer().fit(train)
    before = scaler.mean_.copy()
    transformed = scaler.transform(np.array([[100.0, 200.0]]))
    assert np.array_equal(before, scaler.mean_)
    assert transformed.shape == (1, 2)


def test_first_faulty_sample_has_no_off_by_one():
    training = Run("training:fault_01:0001", np.zeros((4, 1)), np.array([19, 20, 21, 22]), 1, 21)
    testing = Run("testing:fault_01:0001", np.zeros((4, 1)), np.array([159, 160, 161, 162]), 1, 161)
    assert label_run(training).tolist() == [0, 0, 1, 1]
    assert label_run(testing).tolist() == [0, 0, 1, 1]


def test_run_uid_is_global_and_traceable_from_windows():
    normal = make_run_uid("training", 0, 1)
    fault_1 = make_run_uid("training", 1, 1)
    fault_2 = make_run_uid("training", 2, 1)
    test_fault = make_run_uid("testing", 1, 1)
    assert len({normal, fault_1, fault_2, test_fault}) == 4
    run = Run(fault_1, np.zeros((6, 2)), np.arange(1, 7), 1, 4)
    _, _, ids, stats = window_runs([run], 2, 2, "label_by_last_step")
    assert all(identifier.startswith(fault_1) for identifier in ids)
    assert all(record["run_uid"] == fault_1 for record in stats["window_metadata"])
    assert stats["window_metadata"][0]["start_sample"] == 1


def test_stratified_training_split_keeps_official_testing_out():
    runs = [Run(make_run_uid("training", fault, index), np.zeros((2, 1)), np.arange(1, 3), fault, None if fault == 0 else 2)
            for fault in range(3) for index in range(1, 501)]
    testing = [make_run_uid("testing", fault, 1) for fault in range(3)]
    first = split_training_runs_stratified(runs, 0.2, 7, testing)
    second = split_training_runs_stratified(runs, 0.2, 7, testing)
    assert first == second
    assert len(first.train) == 1200 and len(first.validation) == 300
    for fault in range(3):
        marker = ":normal:" if fault == 0 else f":fault_{fault:02d}:"
        assert sum(marker in uid for uid in first.train) == 400
        assert sum(marker in uid for uid in first.validation) == 100
    assert set(first.test) == set(testing)
    assert not set(first.test) & (set(first.train) | set(first.validation))
