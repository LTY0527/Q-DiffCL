from __future__ import annotations

import numpy as np

import scripts.run_semantic_diffusion_retest as retest


def test_retest_gate_defaults_to_skip_before_loading_views(monkeypatch):
    class FakePath:
        def __init__(self, value): self.value = value
        def read_text(self, encoding=None): return '{"downstream_retest_allowed": false}'
        def mkdir(self, parents=False, exist_ok=False): return None
        def __truediv__(self, other): return FakePath(f"{self.value}/{other}")

    monkeypatch.setattr(retest, "Path", FakePath)
    monkeypatch.setattr(retest, "write_json", lambda _path, _value: None)
    monkeypatch.setattr(retest, "load_fixed_views", lambda _config: (_ for _ in ()).throw(AssertionError("must not load")))
    result = retest.run({
        "audit_result": "audit.json", "output_dir": "result",
        "markers": ["SEMANTIC_DIFFUSION_AUGMENTATION", "SINGLE_SEED", "SUBSET_DATA", "NOT_FOR_PAPER_CLAIMS"],
    })
    assert result["training_skipped"] is True
    assert result["status"] == "SEMANTIC_DIFFUSION_AUGMENTATION_NO_GO"


def test_b2_positive_signal_requires_maintained_recall_and_auprc():
    b1 = {"metrics": {"macro_f1": .70, "far": .20, "fault_recall": .80, "auprc": .90}}
    b2 = {"metrics": {"macro_f1": .72, "far": .19, "fault_recall": .795, "auprc": .899}}
    assert retest.b2_has_positive_signal(b1, b2)
    b2["metrics"]["fault_recall"] = .70
    assert not retest.b2_has_positive_signal(b1, b2)


def test_semantic_gate_and_fallback_keep_one_equal_weight_view_per_sample():
    base = np.tile(np.arange(4, dtype=np.float32), (3, 1, 1))
    first = base.copy(); second = base.copy(); fallback = np.zeros_like(base)
    base_probability = np.array([[.9, .1]] * 3)
    probability = np.array([[.9, .1], [.1, .9], [.9, .1]])
    base_feature = np.array([[1., 0.]] * 3)
    feature = np.array([[1., 0.], [-1., 0.], [1., 0.]])
    first[0] += .1; first[1] += .1; first[2] += 3
    thresholds = {"maximum_probability_kl": .1, "minimum_feature_cosine": .9,
                  "minimum_normalized_l1": .01, "maximum_normalized_l1": 1.0}
    valid = retest.semantic_validity_mask(base, first, base_probability, base_feature,
                                         probability, feature, thresholds)
    assert valid.tolist() == [True, False, False]
    selected = fallback.copy(); selected[valid] = first[valid]
    assert selected.shape == base.shape
    assert len(selected) == len(base)
