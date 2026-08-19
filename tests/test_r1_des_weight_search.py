import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.run_r1_des_weight_search import _validate_config, _weights, validation_metrics


def _config():
    return yaml.safe_load(Path("configs/r1_des_weight_search.yaml").read_text(encoding="utf-8"))


def test_all_weight_candidates_are_nonnegative_normalized_and_bounded():
    config = _config()
    _validate_config(config)
    assert len(config["variants"]) == 12
    assert _weights(config, "CURRENT") == (.5, .3, .2)
    for name in config["variants"]:
        weights = _weights(config, name)
        assert all(weight >= 0 for weight in weights)
        assert np.isclose(sum(weights), 1)


def test_only_weights_change_across_candidates():
    config = _config()
    frozen = copy.deepcopy(config)
    frozen.pop("variants")
    for name in config["variants"]:
        assert set(config["variants"][name]) == {
            "weight_discriminative", "weight_early", "weight_run_stability"
        }
        assert frozen == {key: value for key, value in config.items() if key != "variants"}


def test_validation_parser_rejects_test_or_nonvalidation_records():
    record = {"evaluation_split": "validation", "test_metrics_read": False,
              "method": {"validation": {"metrics": {"macro_f1": .8, "far": .1, "auprc": .9},
                                          "early_fault": {"recall": .7}}}}
    assert validation_metrics("TEP", record)["macro_f1"] == .8
    bad = copy.deepcopy(record); bad["method"]["test"] = {}
    with pytest.raises(RuntimeError, match="test metrics"):
        validation_metrics("TEP", bad)
    bad = copy.deepcopy(record); bad["evaluation_split"] = "test"
    with pytest.raises(RuntimeError, match="non-validation"):
        validation_metrics("TEP", bad)


def test_frozen_diffusion_protocol_rejects_joint_tuning():
    config = _config(); config["spectral_diffusion"]["t_noncritical"] = 4
    with pytest.raises(ValueError, match="frozen diffusion"):
        _validate_config(config)
