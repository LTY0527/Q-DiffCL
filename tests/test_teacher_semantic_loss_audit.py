from __future__ import annotations

import numpy as np
import torch
from torch import nn

from losses import freeze_teacher
from scripts.audit_teacher_reliability import (
    effective_rank, make_perturbations, perturbation_metrics, teacher_gate,
)


class TinyTeacher(nn.Module):
    def __init__(self):
        super().__init__(); self.linear = nn.Linear(2, 2)

    def forward(self, values):
        embedding = values.mean(-1)
        return {"logits": self.linear(embedding), "embedding": embedding}


def test_teacher_is_frozen_for_audit():
    teacher = freeze_teacher(TinyTeacher())
    assert not teacher.training
    assert all(not parameter.requires_grad for parameter in teacher.parameters())


def test_light_perturbations_are_deterministic_finite_and_shape_preserving():
    values = np.random.default_rng(2).normal(size=(4, 3, 12)).astype(np.float32)
    settings = {"jitter_std": .01, "scaling_std": .01, "masking_ratio": .02,
                "interpolation_mask_ratio": .05, "interpolation_blend": .2}
    first = make_perturbations(values, settings, 9)
    second = make_perturbations(values, settings, 9)
    assert set(first) == {"jitter", "scaling", "light_masking", "interpolation_perturbation"}
    for name in first:
        assert first[name].shape == values.shape
        assert np.isfinite(first[name]).all()
        assert np.array_equal(first[name], second[name])


def test_effective_rank_detects_collapsed_and_two_dimensional_embeddings():
    line = np.column_stack([np.arange(20), np.zeros(20)])
    plane = np.vstack([np.eye(2), -np.eye(2)])
    assert effective_rank(line) == 1.0
    assert effective_rank(plane) > 1.99


def test_directional_flip_metrics_use_base_predictions_and_group_fault_types():
    bundle = {"labels": np.array([0, 0, 1, 1]),
              "run_uid": np.array(["training:normal:1", "training:normal:2",
                                   "training:fault_01:1", "training:fault_01:2"])}
    probability = np.array([[.9, .1], [.8, .2], [.2, .8], [.1, .9]])
    changed_probability = np.array([[.1, .9], [.8, .2], [.9, .1], [.1, .9]])
    base = {"prediction": probability.argmax(1), "probability": probability,
            "logits": np.log(probability), "embedding": probability}
    changed = {"prediction": changed_probability.argmax(1), "probability": changed_probability,
               "logits": np.log(changed_probability), "embedding": changed_probability}
    result = perturbation_metrics(bundle, base, changed)
    assert result["prediction_consistency"] == .5
    assert result["normal_to_fault"] == .25
    assert result["fault_to_normal"] == .25
    assert result["groups"]["fault_type_01"]["consistency"] == .5


def test_teacher_gate_requires_embedding_and_never_depends_on_test():
    base = {"macro_f1": .9, "auprc": .9, "auroc": .9, "embedding_effective_rank": 1.1,
            "fault_type_recall": {str(i): {"recall": .8} for i in range(1, 21)}}
    perturbation = {"jitter": {"prediction_consistency": .99, "normal_to_fault": .01,
                                "fault_to_normal": .01,
                                "groups": {"fault_type_01": {"count": 20, "consistency": .99}}}}
    gate = {"minimum_consistency": .95, "maximum_directional_flip": .05,
            "maximum_direction_gap": .03, "minimum_major_fault_type_count": 16,
            "minimum_major_fault_type_consistency": .9, "minimum_macro_f1": .8,
            "minimum_auprc": .85, "minimum_auroc": .85, "minimum_effective_rank": 2,
            "minimum_fault_type_recall": .5, "maximum_low_fault_type_fraction": .25}
    checks, passed = teacher_gate(base, perturbation, gate)
    assert not checks["embedding_effective_rank_acceptable"]
    assert not passed

