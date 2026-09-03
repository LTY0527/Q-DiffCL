from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import pytest
import yaml

from scripts.audit_qdiffcl_data_regime import canonical_hash, nested_subsets, SourceUnit
from scripts.run_qdiffcl_data_regime import (
    FORMAL_METHODS, accounting, choose_rho, legal_dataset_fractions, load_config,
    load_fraction_manifest, reuse_compatible,
)
from scripts.summarize_qdiffcl_data_regime import cluster_bootstrap_ci, split_first_summary


CONFIG_PATH = Path("configs/qdiffcl_data_regime_v1.yaml")


def config():
    return load_config(CONFIG_PATH)


def manifests():
    return sorted(Path("configs/data_regime_manifests").glob("*.json"))


def test_fraction_nested():
    for path in manifests():
        value = json.loads(path.read_text(encoding="utf-8"))
        full = set(value["fractions"]["1.0"]["source_ids"])
        quarter = set(value["fractions"]["0.25"]["source_ids"])
        tenth = set(value["fractions"]["0.1"]["source_ids"])
        assert tenth < quarter < full


def test_fraction_seed_independent():
    for path in manifests():
        value = json.loads(path.read_text(encoding="utf-8"))
        for record in value["fractions"].values():
            assert "model_seed" not in record
            assert record["protocol_seed"] == config()["fraction_protocol_seed"]


def test_no_validation_or_test_in_train_fraction():
    for path in manifests():
        value = json.loads(path.read_text(encoding="utf-8"))
        held_out = set(value["outer_protocol"]["groups"]["validation"]) | set(value["outer_protocol"]["groups"]["test"])
        for record in value["fractions"].values():
            if value["dataset"] == "TEP":
                assert set(record["source_ids"]).isdisjoint(held_out)
            else:
                assert all(source_id.split("_", 1)[0] not in held_out for source_id in record["source_ids"])


def test_group_integrity():
    for path in manifests():
        groups = json.loads(path.read_text(encoding="utf-8"))["outer_protocol"]["groups"]
        assert set(groups["train"]).isdisjoint(groups["validation"])
        assert set(groups["train"]).isdisjoint(groups["test"])
        assert set(groups["validation"]).isdisjoint(groups["test"])


def test_no_window_random_subsampling():
    cfg = config()
    assert cfg["sampling_policy"]["window_level_sampling"] is False
    assert cfg["three_w"]["sampling_unit"] == "instance_id"
    assert cfg["tep"]["sampling_unit"] == "run_uid"


def test_fraction_local_normalization():
    source = inspect.getsource(__import__("scripts.run_qdiffcl_data_regime", fromlist=["prepare_three_w"]))
    assert 'fit_fraction_preprocessor(by_split["train"]' in source
    assert 'fit_many([run.values for run in groups["train"]])' in source


def test_empty_fraction_channel_does_not_borrow_full_data_statistics():
    module = __import__("scripts.run_qdiffcl_data_regime", fromlist=["fit_fraction_preprocessor"])
    source = inspect.getsource(module.fit_fraction_preprocessor)
    assert "neutral_zero_channel" in source
    assert "unused train units" in source


def test_fraction_local_criticality_fit():
    source = inspect.getsource(__import__("scripts.run_qdiffcl_data_regime", fromlist=["prepare_three_w"]))
    assert "phase_g._criticality(train_x" in source
    assert 'bundles["train"]["clean"]' in source


@pytest.mark.parametrize("dataset", ["three_w", "tep"])
def test_training_budget_fixed_across_fractions(dataset):
    budget = config()[dataset]["training_budget"]
    assert budget["pretrain_epochs"] > 0 and budget["probe_epochs"] > 0
    assert all(str(fraction) not in budget for fraction in (1.0, .25, .1))


def test_batch_size_fixed_across_fractions():
    assert config()["three_w"]["training_budget"]["batch_size"] == 256
    assert config()["tep"]["training_budget"]["batch_size"] == 128


def test_patience_fixed_across_fractions():
    assert config()["three_w"]["training_budget"]["early_stopping_patience"] == 20
    assert config()["tep"]["training_budget"]["early_stopping_patience"] == 8


def test_optimizer_step_accounting():
    source = inspect.getsource(__import__("scripts.run_qdiffcl_data_regime", fromlist=["_training_accounting"]))
    assert "math.ceil(samples / batch) * pre_epochs" in source
    assert "examples_seen" in source


def test_validation_size_fixed_by_design():
    assert config()["sampling_policy"]["validation_size_fixed"] is True


def test_e_identifiability_threshold():
    assert config()["sampling_policy"]["minimum_onset_bearing_units_per_e_class"] == 2


def test_e_identifiability_hold_fallback():
    pairs = legal_dataset_fractions(config())
    assert ("TEP", .10) not in pairs
    assert ("TEP", .25) in pairs and ("3W", .10) in pairs


def test_final_de_weights_locked():
    assert config()["algorithm"]["criticality_weights"] == {
        "weight_discriminative": .5, "weight_early": .5, "weight_run_stability": 0.,
    }


def test_s_weight_zero():
    assert config()["algorithm"]["criticality_weights"]["weight_run_stability"] == 0


def test_rho_grid_locked():
    assert config()["rho_grid"] == [0, .25, .5, .75, 1]
    assert config()["algorithm"]["rho_candidates"] == config()["rho_grid"]


def test_rho_validation_only_and_tie_break():
    rows = [
        {"rho": .5, "macro_f1": .8, "auprc": .9, "far": .1},
        {"rho": .25, "macro_f1": .8, "auprc": .9, "far": .1},
    ]
    assert choose_rho(rows)["rho"] == .25
    source = inspect.getsource(__import__("scripts.run_qdiffcl_data_regime", fromlist=["select_rho"]))
    assert "outer_test_read" in source and "validation_only" in source


def test_rho_selection_smoke_namespace():
    cfg = config()
    assert cfg["output"]["smoke_namespace"] == "SMOKE_ONLY"
    assert cfg["output"]["namespace"] == "DATA_REGIME_GENERALIZATION_V1"


def test_historical_dcbr_vs_data_regime_rho_lineage():
    cfg = config()
    assert cfg["historical_dcbr_global_rho"] == {"3W": 1, "TEP": .75}
    assert cfg["rho_selection_order"] == ["macro_f1", "auprc", "negative_far", "negative_rho"]


def test_outer_once():
    source = inspect.getsource(__import__("scripts.run_qdiffcl_data_regime", fromlist=["evaluate_once"]))
    assert "outer-test once guard" in source
    assert "outer_test_started.json" in source


def test_resume_skip_completed():
    source = inspect.getsource(__import__("scripts.run_qdiffcl_data_regime", fromlist=["evaluate_once"]))
    assert "result_path.exists() and prediction_path.exists()" in source
    assert "completed result hash mismatch" in source


def test_manifest_hash_validation():
    for path in manifests():
        value = json.loads(path.read_text(encoding="utf-8")); claimed = value.pop("sha256")
        assert canonical_hash(value) == claimed
        dataset = "3W" if path.name.startswith("3w") else "TEP"
        load_fraction_manifest(dataset, int(path.stem.rsplit("_", 1)[1]), .25)


def test_100pct_reuse_compatibility():
    expected = {key: f"value-{key}" for key in (
        "dataset", "outer_id", "method_semantics", "weights", "model_seed", "train_hash",
        "validation_hash", "test_hash", "training_budget_hash", "preprocessing_hash",
        "evaluation_hash", "protocol_hash", "checkpoint_sha256", "prediction_sha256",
    )}
    assert reuse_compatible(expected, copy.deepcopy(expected))
    changed = copy.deepcopy(expected); changed["training_budget_hash"] = "different"
    assert not reuse_compatible(expected, changed)


def test_cell_accounting_after_hold():
    counts = accounting(config())
    assert counts["formal_cells_expected"] == 375
    assert counts["rho_candidate_cells_expected"] == 225
    assert counts["duplicate_count"] == 0


def test_core_methods_locked():
    assert tuple(config()["methods"]) == FORMAL_METHODS


def test_split_first_summary_averages_seeds_before_outers():
    rows = []
    for outer, values in ((1, [0., 2.]), (2, [10., 10.])):
        for seed, value in enumerate(values):
            rows.append({"dataset": "3W", "fraction": 1., "method": "NO_AUG", "outer_id": outer,
                         "model_seed": seed, **{metric: value for metric in ("macro_f1", "auprc", "far", "early_recall", "detection_delay")}})
    assert split_first_summary(rows)[0]["macro_f1_mean"] == pytest.approx(5.5)


def test_group_bootstrap_uses_outer_and_group_units():
    rows = [{"outer_id": outer, "group_id": group, "delta": 1.0}
            for outer in (1, 2, 3) for group in ("a", "b")]
    assert cluster_bootstrap_ci(rows, 50, 7) == (1.0, 1.0)


def test_rho_training_writes_heartbeat_before_candidate():
    module = __import__("scripts.run_qdiffcl_data_regime", fromlist=["select_rho"])
    source = inspect.getsource(module.select_rho)
    assert "rho_validation_training" in source
    assert source.index("heartbeat(") < source.index("train_validation(")
