from pathlib import Path

import numpy as np
import torch
import yaml

from baselines.posthoc_recent import (AutoTCLGate, SoftCLTAdaptation, TFCRepresentation, bcl_to_btc,
                                      canonical_hash, deterministic_grain_split)
from scripts.run_posthoc_recent_baselines import EVIDENCE_CLASS, stratified_subset, validate_protocol
from scripts.summarize_posthoc_recent_baselines import _macro_f1_from_confusions, split_first_summary


def test_layout_conversion_is_exact_and_contiguous():
    x = np.arange(48, dtype=np.float32).reshape(2, 3, 8)
    converted = bcl_to_btc(x)
    assert converted.shape == (2, 8, 3) and converted.flags.c_contiguous
    assert np.array_equal(converted.transpose(0, 2, 1), x)


def test_stratified_sanity_subset_is_deterministic_and_has_all_classes():
    labels = np.repeat(np.arange(4), 20)
    first = stratified_subset(labels, 24, 42); second = stratified_subset(labels, 24, 42)
    assert np.array_equal(first, second)
    assert set(labels[first]) == {0, 1, 2, 3}


def test_autotcl_gate_is_shaped_finite_and_differentiable():
    torch.manual_seed(7); gate = AutoTCLGate(3); values = torch.randn(4, 3, 16, requires_grad=True)
    changed, probability = gate(values); loss = changed.square().mean() + probability.mean(); loss.backward()
    assert changed.shape == values.shape and probability.shape == values.shape
    assert torch.isfinite(changed).all() and gate.features[0].weight.grad is not None


def test_autotcl_gate_stays_finite_when_probability_saturates():
    gate = AutoTCLGate(3)
    with torch.no_grad():
        gate.features[-1].weight.zero_()
        gate.features[-1].bias.fill_(1000.0)
    values = torch.randn(4, 3, 16, requires_grad=True)
    changed, probability = gate(values)
    changed.sum().backward()
    assert torch.isfinite(changed).all()
    assert torch.isfinite(probability).all()
    assert torch.isfinite(values.grad).all()


def test_posthoc_split_first_summary_and_confusion_macro_f1():
    records = []
    for outer, values in ((1, (0.2, 0.4)), (2, (0.6, 0.8)), (3, (0.4, 0.6))):
        for seed, value in enumerate(values):
            records.append({"dataset": "3W", "method": "M", "track": "A", "outer_seed": outer,
                            "model_seed": seed, "metrics": {name: value for name in
                            ("macro_f1", "auprc", "far", "fault_recall", "early_recall", "detection_delay")}})
    overall = next(row for row in split_first_summary(records) if row["level"] == "overall")
    assert np.isclose(overall["macro_f1_mean"], 0.5)
    confusion = np.asarray([[[2, 0], [0, 2]], [[1, 1], [1, 1]]])
    assert np.allclose(_macro_f1_from_confusions(confusion), [1.0, 0.5])


def test_grain_split_is_deterministic_and_covers_channels():
    assert deterministic_grain_split(22) == [11, 22]
    assert deterministic_grain_split(52) == [26, 52]


def test_fallback_representations_are_multivariate_and_finite():
    values = np.random.default_rng(7).normal(size=(4, 3, 64)).astype(np.float32)
    tfc = TFCRepresentation(64, 8)
    assert tfc.encode(values, 4, "cpu").shape == (4, 16)
    soft = SoftCLTAdaptation(3, 2, {"hidden_channels": 8, "projection_dim": 8, "levels": 2})
    encoded = soft.encode(values, 4, "cpu")
    assert encoded.shape == (4, 8) and np.isfinite(encoded).all()


def test_posthoc_config_preserves_selection_boundary():
    config = yaml.safe_load(Path("configs/posthoc_recent_baselines.yaml").read_text(encoding="utf-8"))
    assert config["evidence_class"] == EVIDENCE_CLASS
    assert config["selected_methods"] == ["TimesURL", "MF-CLR", "REBAR", "AutoTCL"]
    assert config["active_methods"] == ["TF-C", "SoftCLT", "TS2Vec", "AutoTCL"]
    assert canonical_hash(config["selected_methods"]) == canonical_hash(["TimesURL", "MF-CLR", "REBAR", "AutoTCL"])
