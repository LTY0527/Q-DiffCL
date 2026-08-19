import copy
from pathlib import Path

import pytest
import yaml

from scripts.run_qdiffcl_final_5seed import _fairness_subset, validate_config


def _config():
    return yaml.safe_load(Path("configs/qdiffcl_final_5seed.yaml").read_text(encoding="utf-8"))


def test_final_weights_and_five_seed_lists_are_frozen():
    config = _config(); validate_config(config)
    assert config["variants"]["FINAL_DE"] == {
        "weight_discriminative": .5, "weight_early": .5, "weight_run_stability": 0.0}
    assert config["three_w"]["seeds"] == [42, 43, 44, 45, 46]
    assert config["tep"]["seeds"] == [7, 42, 43, 44, 2026]


def test_reliability_cannot_reopen_weight_or_timestep_search():
    config = _config(); config["variants"]["FINAL_DE"]["weight_early"] = .4
    with pytest.raises(ValueError, match="FINAL_DE"):
        validate_config(config)
    config = _config(); config["spectral_diffusion"]["t_noncritical"] = 4
    with pytest.raises(ValueError, match="frozen diffusion"):
        validate_config(config)


def test_fairness_subset_uses_protocol_hashes_only():
    record = {"manifest_sha256": "m", "initialization_sha256": "i",
              "pretrain_order_sha256": "p", "probe_order_sha256": "q", "metric": 1}
    assert _fairness_subset(record, "TEP") == {
        "manifest_sha256": "m", "initialization_sha256": "i",
        "pretrain_order_sha256": "p", "probe_order_sha256": "q"}
