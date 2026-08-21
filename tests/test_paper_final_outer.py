from __future__ import annotations

import json
from pathlib import Path

import yaml

import numpy as np
from scripts.run_paper_final_outer import (METHODS, _compatible_metadata, augmentation_views, rho_name,
                                           run_id, select_dcbr_rho, split_record, validate_frozen)


def _config():
    return yaml.safe_load(Path("configs/paper_final_outer.yaml").read_text(encoding="utf-8"))


def test_outer_matrix_and_ids_are_frozen():
    config = _config()
    assert METHODS == ("NO_AUG", "JITTER", "SCALING", "JITTER_SCALING", "UNIFORM_DIFFUSION",
                       "FRERA", "FINAL_QDIFFCL", "DCBR")
    assert 3 * 5 * 7 + 3 * 5 * 8 == 225
    assert 3 * 5 * len(METHODS) * 2 == 240
    assert run_id("3W", 31001, 42, "FINAL_QDIFFCL") == "3w-outer31001-seed42-final_qdiffcl"
    assert [rho_name(value) for value in (0, .25, .5, .75, 1)] == ["rho_000", "rho_025", "rho_050", "rho_075", "rho_100"]
    assert config["algorithm"]["rho_candidates"] == [0, .25, .5, .75, 1]


def test_frozen_split_records_are_disjoint_and_exact_size():
    config = _config()
    for dataset, seeds, sizes in (("3W", [31001, 31002, 31003], [20, 8, 8]),
                                  ("TEP", [32001, 32002, 32003], [248, 72, 80])):
        for seed in seeds:
            row = split_record(config, dataset, seed)
            groups = [set(row["groups"][name]) for name in ("train", "validation", "test")]
            assert [len(group) for group in groups] == sizes
            assert not groups[0] & groups[1]
            assert not groups[0] & groups[2]
            assert not groups[1] & groups[2]


def test_dcbr_selection_uses_domain_mean_and_frozen_tie_break():
    config = _config(); context = {"dataset": "TEP", "outer_seed": 32001}
    records = {}
    for seed in (7, 42):
        records[seed] = {}
        for rho in (0, .25, .5, .75, 1):
            records[seed][rho] = {"validation": {"macro_f1": .8 + (.1 if rho in (.5, .75) else 0),
                                                   "auprc": .9, "far": .05, "threshold": .5}}
    selected = select_dcbr_rho(config, context, records)
    assert selected["selected_rho"] == .5
    assert selected["selection_split"] == "inner-validation"
    assert selected["outer_test_read"] is False


def test_pre_outer_freeze_still_valid_and_has_no_outer_metric():
    config = _config()
    freeze = json.loads(Path(config["freeze_manifest"]).read_text(encoding="utf-8"))
    dry = json.loads(Path(config["dry_run_manifest"]).read_text(encoding="utf-8"))
    leakage = json.loads(Path(config["leakage_audit"]).read_text(encoding="utf-8"))
    assert freeze["status"] == "PAPER_FINAL_FREEZE_READY"
    assert dry["outer_metrics"] is None
    assert leakage["outer_test_metrics_read"] is False
    assert freeze["outer_training_run"] is False


def test_frera_dispatcher_does_not_require_dcbr_rho():
    config = _config(); clean = np.zeros((2, 3, 4), dtype=np.float32)
    context = {"dataset": "3W", "train": clean, "validation": clean.copy(),
               "ids": {"train": np.asarray(["a", "b"]), "validation": np.asarray(["c", "d"])},
               "training": {"batch_size": 2}}
    views, audit = augmentation_views(context, config, "FRERA", 42, "cpu")
    assert np.array_equal(views["train"], clean)
    assert audit["train"]["mode"] == "FRERA_internal_learned_view"


def test_resume_allows_only_declared_provenance_hash_change():
    config = _config(); expected = {"dataset": "3W", "phase_g_config_sha256": "new"}
    existing = {"dataset": "3W", "phase_g_config_sha256": config["git_freeze"]["provenance_compatible_config_hashes"][0]}
    config["git_freeze"]["provenance_compatible_config_hashes"].append("new")
    assert _compatible_metadata(existing, expected, config)
    existing["dataset"] = "TEP"
    assert not _compatible_metadata(existing, expected, config)
