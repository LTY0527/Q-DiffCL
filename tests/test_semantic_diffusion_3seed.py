from __future__ import annotations

import copy
import subprocess

import numpy as np
import pytest
import yaml

from scripts.audit_semantic_diffusion_augmentation import traditional_augmentation
from scripts.run_diffusion_quality_retest import epoch_orders
from scripts.run_semantic_diffusion_3seed import (
    METHODS,
    completed_result_is_reusable,
    configure_determinism,
    initialization_hash,
    validate_frozen_config,
)
from scripts.summarize_semantic_diffusion_3seed import mean_std, paired_deltas, stability_status


def _config():
    return yaml.safe_load(open("configs/semantic_diffusion_3seed.yaml", encoding="utf-8"))


def test_three_seed_configs_are_identical_except_runtime_seed():
    config = _config(); validate_frozen_config(config)
    variants = []
    for seed in config["seeds"]:
        value = copy.deepcopy(config); value["random_seed"] = seed; variants.append(value)
    normalized = []
    for value in variants:
        value.pop("random_seed"); normalized.append(value)
    assert normalized[0] == normalized[1] == normalized[2]


def test_same_seed_initialization_and_orders_are_fair_and_reproducible():
    config = _config()
    settings = configure_determinism(True, config["cublas_workspace_config"])
    assert settings["cublas_workspace_config"] == ":4096:8"
    assert initialization_hash(config, 52, 7) == initialization_hash(config, 52, 7)
    first, second = epoch_orders(31, 4, 10007), epoch_orders(31, 4, 10007)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_different_seeds_change_initialization_batch_order_and_augmentation_noise():
    config = _config()
    assert initialization_hash(config, 52, 7) != initialization_hash(config, 52, 42)
    assert any(not np.array_equal(a, b) for a, b in zip(epoch_orders(31, 4, 10007), epoch_orders(31, 4, 10042)))
    values = np.ones((3, 2, 8), np.float32); ids = np.array(["a", "b", "c"])
    first = traditional_augmentation(values, ids, config["traditional_augmentation"], 7)
    second = traditional_augmentation(values, ids, config["traditional_augmentation"], 42)
    assert not np.array_equal(first, second)


def test_seed7_reuse_requires_all_fingerprints_and_b3_is_disabled():
    config = _config(); assert METHODS == ("B0", "B1", "B2") and config["run_b3"] is False
    fingerprints = {"config_sha256": "a", "split_manifest_sha256": "b", "teacher_checkpoint_sha256": "c"}
    result = {"status": "COMPLETE", "markers": config["markers"], "seed": 7, "method": "B2",
              "fingerprints": fingerprints}
    assert completed_result_is_reusable(result, 7, "B2", fingerprints)
    changed = dict(fingerprints, config_sha256="different")
    assert not completed_result_is_reusable(result, 7, "B2", changed)


def test_mean_std_and_delta_directions_are_correct():
    assert mean_std([1., 2., 3.]) == {"mean": 2.0, "std": 1.0}
    results = {}
    for seed, offset in zip((7, 42, 2026), (0., .1, .2)):
        for method, value in (("B0", .5), ("B1", .6), ("B2", .7)):
            results[f"{seed}:{method}"] = {"metrics": {key: value + offset for key in ("macro_f1", "auprc", "fault_recall", "far", "auroc")}}
    deltas = paired_deltas(results, [7, 42, 2026])
    assert deltas["B2-B1"]["7"]["macro_f1"] == pytest.approx(.1)
    assert deltas["B2-B1"]["7"]["far"] == pytest.approx(.1)


def test_stability_gate_reports_go_for_consistent_improvement():
    summary = {method: {metric: {"mean": value, "std": .001} for metric in ("macro_f1", "auprc", "fault_recall", "far", "auroc")}
               for method, value in (("B0", .7), ("B1", .6), ("B2", .71))}
    summary["B1"]["far"]["mean"] = .08; summary["B2"]["far"]["mean"] = .04; summary["B0"]["far"]["mean"] = .03
    deltas = {"B2-B1": {str(seed): {"macro_f1": .01, "auprc": .001, "fault_recall": -.005, "far": -.04, "auroc": .001} for seed in (7, 42, 2026)}}
    gate = {"maximum_mean_recall_drop": .01, "maximum_mean_auprc_drop": .005,
            "catastrophic_far_increase": .05, "catastrophic_recall_drop": .05}
    status, checks, counts = stability_status(summary, deltas, [7, 42, 2026], gate)
    assert status == "SEMANTIC_DIFFUSION_3SEED_GO" and all(checks.values())
    assert counts["far_below_b1"] == 3


def test_nonfinite_summary_is_rejected_and_outputs_are_gitignored():
    with pytest.raises(ValueError, match="finite"):
        mean_std([1., np.nan, 3.])
    ignored = subprocess.run(["git", "check-ignore", "outputs/semantic_diffusion_3seed/example.pt"], capture_output=True)
    assert ignored.returncode == 0
