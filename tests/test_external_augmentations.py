import numpy as np
import pytest
import torch

from baselines.external_augmentations import FreRAAdapter, traditional_view
from scripts.run_external_baselines import _effect_audit, validate_config


@pytest.mark.parametrize("method", ["NO_AUG", "JITTER", "SCALING", "JITTER_SCALING"])
def test_traditional_views_are_deterministic_finite_and_shaped(method):
    values = np.arange(48, dtype=np.float32).reshape(2, 3, 8) / 10
    ids = np.array(["a", "b"])
    first = traditional_view(values, ids, method, 42, .02, .05)
    second = traditional_view(values, ids, method, 42, .02, .05)
    assert first.shape == values.shape and first.dtype == np.float32
    assert np.array_equal(first, second) and np.isfinite(first).all()
    assert np.array_equal(first, values) == (method == "NO_AUG")


def test_traditional_methods_and_seeds_change_views():
    values = np.ones((2, 3, 8), np.float32); ids = np.array(["a", "b"])
    jitter = traditional_view(values, ids, "JITTER", 42, .02, .05)
    scaling = traditional_view(values, ids, "SCALING", 42, .02, .05)
    other_seed = traditional_view(values, ids, "JITTER", 43, .02, .05)
    assert not np.array_equal(jitter, scaling)
    assert not np.array_equal(jitter, other_seed)


def test_frera_adapter_runs_on_cpu_and_backpropagates():
    torch.manual_seed(7)
    adapter = FreRAAdapter(16)
    values = torch.randn(4, 3, 16, requires_grad=True)
    changed = adapter(values, temperature=.1)
    loss = changed.square().mean() + .003 * adapter.l1_regularizer() / 16
    loss.backward()
    assert changed.shape == values.shape and torch.isfinite(changed).all()
    assert adapter.weight.grad is not None and torch.isfinite(adapter.weight.grad).all()


def test_frera_eval_is_deterministic():
    adapter = FreRAAdapter(16).eval(); values = torch.randn(2, 3, 16)
    assert torch.equal(adapter(values), adapter(values))


def test_effect_audit_distinguishes_identity_from_changed_view():
    clean = np.ones((2, 3, 8), np.float32)
    identity = _effect_audit(clean, clean.copy())
    changed = _effect_audit(clean, clean + .1)
    assert not identity["augmentation_effective"] and identity["changed_fraction"] == 0
    assert changed["augmentation_effective"] and changed["changed_fraction"] == 1


def test_external_config_keeps_final_and_tier_sets_frozen():
    import yaml
    from pathlib import Path
    config = yaml.safe_load(Path("configs/external_baselines.yaml").read_text(encoding="utf-8"))
    validate_config(config)
    config["methods"]["tier1"][0] = "CHANGED"
    with pytest.raises(ValueError, match="Tier 1"):
        validate_config(config)
