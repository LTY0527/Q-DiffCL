from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _config(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_paper_final_protocol_is_frozen_and_dry_run_only():
    config = _config("paper_final_protocol.yaml")
    algorithm = config["algorithm"]
    audit = config["audit"]

    assert config["frozen"] is True
    assert algorithm["criticality_weights"] == {"D": 0.5, "E": 0.5, "S": 0.0}
    assert algorithm["critical_ratio"] == 0.30
    assert (algorithm["t_critical"], algorithm["t_noncritical"]) == (1, 5)
    assert config["selection"]["forbidden_outer_test_selection"] is True
    assert audit == {
        "dry_run_only": True,
        "outer_test_metrics_read": False,
        "outer_training_run": False,
    }


def test_paper_final_protocol_uses_grouped_outer_splits():
    config = _config("paper_final_protocol.yaml")
    three_w = config["datasets"]["three_w"]
    tep = config["datasets"]["tep"]

    assert three_w["grouping_unit"] == "WELL"
    assert three_w["wells"] == {"train": 20, "inner_validation": 8, "outer_test": 8}
    assert len(three_w["outer_split_seeds"]) == 3
    assert tep["grouping_unit"] == "Run"
    assert len(tep["outer_split_seeds"]) == 3


def test_mechanism_ablation_is_validation_only_and_complete():
    config = _config("paper_mechanism_ablation.yaml")

    assert config["audit"] == {"evaluation_split": "validation", "test_read": False}
    assert config["variants"] == [
        "UNIFORM_DIFFUSION",
        "HARD_MASK_SELECTIVE",
        "SOFT_MASK_SELECTIVE",
        "SOFT_MASK_WO_BUDGET_MATCH",
    ]
    assert len(config["three_w"]["seeds"]) == 3
    assert len(config["tep"]["seeds"]) == 3
    assert config["spectral_diffusion"]["preserve_dc"] is True
