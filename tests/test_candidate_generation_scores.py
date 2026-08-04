import inspect

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from diffusion import DiffusionSchedule
from diffusion.candidate_generation import (candidate_seed,
                                            candidate_seed_matrix,
                                            restore_candidates,
                                            select_balanced_audit_indices,
                                            validate_shared_context)
from models import MinimalConditionalDiffusion1D
from quality import (center_scores, combined_scores, semantic_scores,
                     soft_candidate_weights)
from scripts.audit_candidate_rankability import oracle_candidate_errors
from scripts.generate_diffusion_candidates import validate_candidate_trace


def test_candidate_seed_is_reproducible_and_candidate_specific():
    first = candidate_seed(7, "train", "window:1", 0)
    assert first == candidate_seed(7, "train", "window:1", 0)
    assert first != candidate_seed(7, "train", "window:1", 1)
    matrix = candidate_seed_matrix(7, "train", ["a", "b"], 3)
    assert matrix.shape == (2, 3)
    assert len(set(matrix.ravel().tolist())) == 6


def test_balanced_audit_subset_is_reproducible():
    labels = np.r_[np.zeros(20, dtype=int), np.ones(20, dtype=int)]
    first = select_balanced_audit_indices(labels, 16, 7, "train")
    second = select_balanced_audit_indices(labels, 16, 7, "train")
    assert np.array_equal(first, second)
    assert np.bincount(labels[first], minlength=2).tolist() == [8, 8]


def test_different_candidate_ids_produce_different_sampling_with_shared_mask():
    torch.manual_seed(3)
    model = MinimalConditionalDiffusion1D(2, hidden=8, time_dimension=8, blocks=1)
    degraded = np.zeros((1, 2, 6), dtype=np.float32)
    observation = np.ones_like(degraded, dtype=bool); observation[:, :, 2:4] = False
    seeds = candidate_seed_matrix(7, "train", ["window"], 2)
    candidates = restore_candidates(model, degraded, observation, seeds, DiffusionSchedule.cosine(2, "cpu"),
                                    1, "cpu", np.full(2, -3, np.float32), np.full(2, 3, np.float32))
    validate_shared_context(candidates, degraded, observation)
    assert not np.array_equal(candidates[0, 0, :, 2:4], candidates[0, 1, :, 2:4])


def test_candidate_manifest_is_traceable():
    records = [{"split": "train", "run_uid": "training:normal:0001", "window_id": "w1",
                "mask_id": "m1", "candidate_seeds": [1, 2], "candidate_ids": [0, 1]}]
    validate_candidate_trace(records, 2)
    with pytest.raises(ValueError, match="incomplete"):
        validate_candidate_trace([{"split": "train"}], 2)


def test_h1_h2_interfaces_cannot_read_clean_or_label():
    assert list(inspect.signature(center_scores).parameters) == ["candidates", "observation"]
    assert list(inspect.signature(semantic_scores).parameters) == ["candidate_probabilities"]
    assert "clean" in inspect.signature(oracle_candidate_errors).parameters


def test_candidate_scores_soft_weights_and_k_one_are_finite():
    candidates = np.array([[[[0.0, 1.0]], [[0.0, 2.0]], [[0.0, 3.0]]]], dtype=np.float32)
    observation = np.array([[[True, False]]])
    probabilities = np.array([[[0.8, 0.2], [0.7, 0.3], [0.6, 0.4]]])
    h1 = center_scores(candidates, observation); h2 = semantic_scores(probabilities)
    h3 = combined_scores(h1, h2); weights = soft_candidate_weights(h3, 0.5)
    assert np.isfinite(h1).all() and np.isfinite(h2).all() and np.isfinite(h3).all()
    assert np.allclose(weights.sum(1), 1.0)
    assert np.allclose(soft_candidate_weights(np.zeros((4, 1)), 1.0), 1.0)
    assert np.allclose(center_scores(candidates[:, :1], observation), 0.0)
    assert np.allclose(semantic_scores(probabilities[:, :1]), 0.0)


def test_soft_candidate_total_weight_is_equal_for_normal_and_fault():
    scores = np.array([[0.0, 1.0, 2.0], [4.0, 0.0, -1.0]])
    labels = np.array([0, 1])
    weights = soft_candidate_weights(scores, 0.3)
    totals = {int(label): float(weights[labels == label].sum(1).mean()) for label in labels}
    assert totals[0] == pytest.approx(1.0)
    assert totals[1] == pytest.approx(1.0)
