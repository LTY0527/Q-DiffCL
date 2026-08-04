import numpy as np

from scripts.run_candidate_selection_retest import (
    audit_allows_retest, deterministic_random_indices, no_reference_scores,
    select_candidate,
)


def test_deterministic_random_candidate_selection():
    ids = np.array(["w1", "w2", "w3"])
    first = deterministic_random_indices(7, "test", ids, 3)
    second = deterministic_random_indices(7, "test", ids, 3)
    assert np.array_equal(first, second)
    assert np.all((first >= 0) & (first < 3))


def test_select_candidate_preserves_one_view_per_sample():
    candidates = np.arange(2 * 3 * 1 * 2).reshape(2, 3, 1, 2)
    chosen = select_candidate(candidates, np.array([2, 0]))
    assert np.array_equal(chosen[0], candidates[0, 2])
    assert np.array_equal(chosen[1], candidates[1, 0])


def test_no_reference_h1_does_not_require_teacher_or_target():
    candidates = np.array([[[[0.0, 1.0]], [[0.0, 2.0]]]], dtype=np.float32)
    observation = np.array([[[True, False]]])
    score = no_reference_scores("h1_center", candidates, observation, None)
    assert score.shape == (1, 2)


def test_retest_gate_is_pure_and_defaults_to_skip():
    assert audit_allows_retest({"downstream_retest_allowed": True}) is True
    assert audit_allows_retest({"downstream_retest_allowed": False}) is False
    assert audit_allows_retest({}) is False
