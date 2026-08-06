from __future__ import annotations

import torch
import numpy as np
from torch import nn

from diffusion.semantic_augmentation import (
    SemanticPartialDiffusion1D,
    residual_augment,
    sample_training_timesteps,
)
from losses import balanced_semantic_consistency_loss, freeze_teacher
from scripts.run_diffusion_quality_retest import best_probe_record
from scripts.run_semantic_generator_stability_fix import (
    diversity_penalty,
    generator_score,
    select_best_candidate,
    update_ema,
)
from scripts.run_semantic_generator_downstream_retest import generator_allows_downstream


class BoundaryTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, values):
        score = values.mean((1, 2)) * self.scale
        logits = torch.stack([-score, score], 1)
        return {"logits": logits, "embedding": torch.stack([score, -score], 1)}


def _model():
    return SemanticPartialDiffusion1D(2, 8, hidden=16, time_dimension=16, blocks=2)


def test_true_film_is_post_norm_and_semantic_survives_normalization():
    torch.manual_seed(3); model = _model().eval()
    block = model.blocks[0]
    assert hasattr(block, "norm1") and hasattr(block, "condition_projection")
    noisy = torch.randn(3, 2, 10); base = torch.randn_like(noisy)
    mask = torch.ones_like(noisy, dtype=torch.bool); timestep = torch.full((3,), 5, dtype=torch.long)
    first = model(noisy, base, mask, timestep, torch.zeros(3, 8))
    second = model(noisy, base, mask, timestep, torch.ones(3, 8))
    assert not torch.allclose(first, second)
    gamma_rows = torch.cat([block.condition_projection.weight[:16], block.condition_projection.weight[32:48]])
    assert torch.count_nonzero(gamma_rows) == 0
    assert torch.isfinite(first).all() and torch.isfinite(second).all()


def test_base_is_explicit_three_channel_group_input_and_changes_output():
    model = _model().eval()
    assert model.input.in_channels == 3 * model.channels
    noisy = torch.randn(2, 2, 9); mask = torch.ones_like(noisy, dtype=torch.bool)
    timestep = torch.tensor([4, 5]); semantic = torch.randn(2, 8)
    first = model(noisy, torch.zeros_like(noisy), mask, timestep, semantic)
    second = model(noisy, torch.ones_like(noisy), mask, timestep, semantic)
    assert first.shape == noisy.shape and not torch.allclose(first, second)


def test_residual_alpha_boundaries_are_exact():
    base = torch.randn(2, 3, 5); sampled = torch.randn_like(base)
    assert torch.equal(residual_augment(base, sampled, 0), base)
    assert torch.equal(residual_augment(base, sampled, 1), sampled)
    assert torch.allclose(residual_augment(base, sampled, .4), base + .4 * (sampled - base))


def test_training_timesteps_only_use_narrow_allowed_set_and_are_reproducible():
    allowed = [3, 4, 5, 6, 7]
    first = sample_training_timesteps(allowed, 100, torch.Generator().manual_seed(8), "cpu")
    second = sample_training_timesteps(allowed, 100, torch.Generator().manual_seed(8), "cpu")
    assert torch.equal(first, second)
    assert set(first.tolist()) <= set(allowed) and 5 in allowed


def test_balanced_semantic_loss_penalizes_normal_and_fault_flips():
    teacher = freeze_teacher(BoundaryTeacher())
    base = torch.cat([-torch.ones(2, 1, 8), torch.ones(2, 1, 8)])
    labels = torch.tensor([0, 0, 1, 1])
    same = balanced_semantic_consistency_loss(teacher, base, base.clone(), labels)
    normal_flip = base.clone(); normal_flip[:2] *= -1
    fault_flip = base.clone(); fault_flip[2:] *= -1
    normal = balanced_semantic_consistency_loss(teacher, base, normal_flip, labels)
    fault = balanced_semantic_consistency_loss(teacher, base, fault_flip, labels)
    assert normal["total"] > same["total"] and fault["total"] > same["total"]
    assert normal["margin"] > 0 and fault["margin"] > 0


def test_single_class_batch_is_finite_teacher_frozen_and_generator_gets_gradient():
    teacher = freeze_teacher(BoundaryTeacher())
    base = -torch.ones(3, 1, 8); generated = base.clone().requires_grad_(True)
    result = balanced_semantic_consistency_loss(teacher, base, generated, torch.zeros(3, dtype=torch.long))
    result["total"].backward()
    assert result["fault_count"] == 0 and result["normal_count"] == 3
    assert generated.grad is not None and torch.isfinite(generated.grad).all()
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in teacher.parameters())


def test_generator_score_and_best_checkpoint_use_validation_only_formula():
    config = {"validation": {"normalized_l1_minimum": .1, "normalized_l1_maximum": .2,
                             "score_weights": {"balanced_flip_rate": 1., "probability_distance": 1., "diversity_penalty": 1.}}}
    assert diversity_penalty(.05, .1, .2) == .05
    good = {"normal_to_fault_flip": .1, "fault_to_normal_flip": .1, "teacher_probability_kl": .02, "normalized_l1": .15}
    bad = dict(good, normal_to_fault_flip=.3, normalized_l1=.3)
    records = [{"epoch": 0, "variant": "raw", "score": generator_score(bad, config)},
               {"epoch": 1, "variant": "ema", "score": generator_score(good, config)}]
    assert select_best_candidate(records)["epoch"] == 1


def test_ema_update_matches_formula():
    ema = {"weight": torch.tensor([1.0]), "count": torch.tensor(1)}
    state = {"weight": torch.tensor([3.0]), "count": torch.tensor(2)}
    update_ema(ema, state, .5)
    assert torch.equal(ema["weight"], torch.tensor([2.0])) and int(ema["count"]) == 2


def test_probe_selection_uses_thresholded_validation_tie_breakers():
    history = [
        {"epoch": 0, "validation_macro_f1": .8, "validation_far": .2, "validation_fault_recall": .9, "validation_auprc": .9, "validation_threshold": .4},
        {"epoch": 1, "validation_macro_f1": .8, "validation_far": .1, "validation_fault_recall": .8, "validation_auprc": .8, "validation_threshold": .6},
    ]
    assert best_probe_record(history)["epoch"] == 1
    assert best_probe_record(history)["validation_threshold"] == .6


def test_downstream_gate_defaults_to_skip_and_has_no_test_selection_input():
    assert not generator_allows_downstream({})
    assert not generator_allows_downstream({"status": "SEMANTIC_GENERATOR_FIX_NO_GO"})
    assert generator_allows_downstream({"status": "SEMANTIC_GENERATOR_FIX_READY_FOR_DOWNSTREAM_RETEST"})
