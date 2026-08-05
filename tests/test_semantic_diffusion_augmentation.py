from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from torch import nn

from diffusion import DiffusionSchedule
from diffusion.semantic_augmentation import (
    SemanticPartialDiffusion1D,
    partial_q_sample,
    partial_reverse_sample,
)
from losses import freeze_teacher, semantic_consistency_losses
from scripts.audit_semantic_diffusion_augmentation import epoch_orders, generate_repeats


class TinyTeacher(nn.Module):
    def __init__(self, channels: int = 2, embedding_dimension: int = 8):
        super().__init__()
        self.encoder = nn.Conv1d(channels, embedding_dimension, 1, bias=False)
        self.classifier = nn.Linear(embedding_dimension, 2, bias=False)

    def forward(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.encoder(values).mean(-1)
        return {"embedding": embedding, "logits": self.classifier(embedding)}


class SignTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        score = values.mean((1, 2)) * self.scale
        embedding = torch.stack([score, -score], dim=1)
        return {"embedding": embedding, "logits": torch.stack([-score, score], dim=1)}


class RecordingZeroModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.timesteps: list[int] = []

    def forward(self, noisy, observation, timesteps, semantic):
        self.timesteps.append(int(timesteps[0]))
        return torch.zeros_like(noisy)


def _model() -> SemanticPartialDiffusion1D:
    return SemanticPartialDiffusion1D(2, 8, hidden=16, time_dimension=16, blocks=3)


def test_teacher_is_frozen_but_semantic_loss_keeps_input_gradient():
    teacher = freeze_teacher(TinyTeacher())
    base = torch.randn(4, 2, 12)
    generated = torch.randn_like(base, requires_grad=True)
    probability_loss, feature_loss = semantic_consistency_losses(teacher, base, generated)
    (probability_loss + feature_loss).backward()
    assert generated.grad is not None and torch.isfinite(generated.grad).all()
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in teacher.parameters())
    assert torch.isfinite(probability_loss) and torch.isfinite(feature_loss)


def test_semantic_embedding_changes_output_and_enters_every_residual_block():
    torch.manual_seed(4)
    model = _model().eval()
    noisy = torch.randn(3, 2, 10)
    observation = torch.ones_like(noisy, dtype=torch.bool)
    timesteps = torch.full((3,), 5, dtype=torch.long)
    calls = [0] * len(model.blocks)
    hooks = []
    for index, block in enumerate(model.blocks):
        hooks.append(block.semantic_projection.register_forward_hook(
            lambda _module, _inputs, _output, index=index: calls.__setitem__(index, calls[index] + 1)
        ))
    first = model(noisy, observation, timesteps, torch.zeros(3, 8))
    second = model(noisy, observation, timesteps, torch.ones(3, 8))
    for hook in hooks:
        hook.remove()
    assert calls == [2, 2, 2]
    assert not torch.allclose(first, second)
    assert torch.isfinite(first).all() and torch.isfinite(second).all()


def test_partial_q_sample_uses_requested_timestep():
    schedule = DiffusionSchedule.cosine(20, "cpu")
    base = torch.randn(2, 2, 6)
    noise = torch.randn_like(base)
    actual = partial_q_sample(base, schedule, 7, noise)
    expected = schedule.q_sample(base, torch.full((2,), 7, dtype=torch.long), noise)
    assert torch.equal(actual, expected)
    with pytest.raises(ValueError, match="outside"):
        partial_q_sample(base, schedule, 20, noise)


def test_partial_reverse_only_runs_from_t_aug_to_zero():
    model = RecordingZeroModel()
    base = torch.randn(2, 2, 6)
    result = partial_reverse_sample(
        model, DiffusionSchedule.cosine(20, "cpu"), base, torch.ones_like(base, dtype=torch.bool),
        torch.zeros(2, 8), 5, torch.Generator().manual_seed(3),
    )
    assert model.timesteps == [5, 4, 3, 2, 1, 0]
    assert result.shape == base.shape and torch.isfinite(result).all()


def test_g0_g1_fair_inputs_and_sampling_are_reproducible():
    torch.manual_seed(11)
    template = _model()
    g0 = _model(); g1 = _model()
    state = copy.deepcopy(template.state_dict())
    g0.load_state_dict(state); g1.load_state_dict(state)
    assert all(torch.equal(g0.state_dict()[key], g1.state_dict()[key]) for key in state)
    assert all(np.array_equal(a, b) for a, b in zip(epoch_orders(17, 3, 19), epoch_orders(17, 3, 19)))

    base = np.random.default_rng(2).normal(size=(4, 2, 8)).astype(np.float32)
    observation = np.ones_like(base, dtype=bool)
    semantic = np.random.default_rng(3).normal(size=(4, 8)).astype(np.float32)
    schedule = DiffusionSchedule.cosine(12, "cpu")
    limits = (base.min((0, 2)), base.max((0, 2)))
    first = generate_repeats(g0, base, observation, semantic, schedule, 4, 2, 4, "cpu", 23, *limits)
    second = generate_repeats(g1, base, observation, semantic, schedule, 4, 2, 4, "cpu", 23, *limits)
    repeated = generate_repeats(g0, base, observation, semantic, schedule, 4, 2, 4, "cpu", 23, *limits)
    assert np.array_equal(first, second)
    assert np.array_equal(first, repeated)
    assert np.isfinite(first).all()


def test_consistent_input_has_lower_semantic_loss_than_flipped_input():
    teacher = freeze_teacher(SignTeacher())
    base = torch.ones(4, 2, 8)
    same = semantic_consistency_losses(teacher, base, base.clone())
    flipped = semantic_consistency_losses(teacher, base, -base)
    assert sum(float(value) for value in same) < sum(float(value) for value in flipped)
