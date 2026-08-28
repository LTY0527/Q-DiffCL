from pathlib import Path

import yaml

from scripts.run_posthoc_baseline_5seed_extension import (
    EVIDENCE_CLASS,
    build_cells,
    canonical_hash,
    locked_protocol,
    runtime_config,
)
from scripts.summarize_posthoc_baseline_5seed_extension import (
    COMPLETE,
    combined_raw_rows,
    extension_cells_complete,
)
from scripts.summarize_posthoc_recent_baselines import METRICS


def _config():
    return yaml.safe_load(Path("configs/posthoc_baseline_5seed_extension.yaml").read_text(encoding="utf-8"))


def test_extension_lock_is_exactly_four_methods_and_48_missing_cells():
    config = _config()
    cells = build_cells(config)
    assert config["evidence_class"] == EVIDENCE_CLASS
    assert config["active_methods"] == ["TF-C", "SoftCLT", "TS2Vec", "AutoTCL"]
    assert len(cells) == len({cell["run_id"] for cell in cells}) == 48
    assert {cell["model_seed"] for cell in cells if cell["dataset"] == "3W"} == {45, 46}
    assert {cell["model_seed"] for cell in cells if cell["dataset"] == "TEP"} == {43, 44}


def test_extension_seed_sets_match_frozen_paper_final_five_seeds():
    config = _config()
    for dataset in ("3W", "TEP"):
        assert not (set(config["h1_completed_seeds"][dataset]) & set(config["missing_seeds"][dataset]))
        assert set(config["h1_completed_seeds"][dataset]) | set(config["missing_seeds"][dataset]) == set(config["full_seeds"][dataset])


def test_extension_runtime_reuses_h1_numerical_configuration():
    config = _config()
    base = yaml.safe_load(Path(config["h1_config"]).read_text(encoding="utf-8"))
    runtime = runtime_config(config)
    for key in ("representation_dim", "learning_rate", "temperature", "native"):
        assert runtime[key] == base[key]
    for key in ("epochs", "probe_epochs", "batch_size"):
        assert runtime["benchmark"][key] == base["benchmark"][key]
    assert runtime["output"]["root"] != base["output"]["root"]
    assert runtime["benchmark"]["model_seeds"] == config["missing_seeds"]


def test_extension_protocol_hash_is_canonical_and_result_independent():
    config = _config()
    first = canonical_hash(locked_protocol(config))
    second = canonical_hash(locked_protocol(_config()))
    assert first == second and len(first) == 64
    assert "output" not in locked_protocol(config)


def test_combined_raw_rows_accepts_frozen_paper_final_evidence_source():
    reference = {
        "run_id": "paper-final-reference-not-in-h1-source-map",
        "dataset": "3W",
        "outer_seed": 31001,
        "model_seed": 42,
        "method": "FINAL_QDIFFCL",
        "track": "TRACK_A_FROZEN_PAPER_FINAL_REFERENCE",
        "metrics": {metric: None for metric in METRICS},
        "prediction_path": "prediction.npz",
        "prediction_sha256": "prediction-sha256",
        "checkpoint_sha256": "checkpoint-sha256",
        "evidence_source": "FROZEN_PAPER_FINAL_5SEED_REUSE",
    }

    rows = combined_raw_rows([], [], [reference])

    assert len(rows) == 1
    assert rows[0]["evidence_source"] == "FROZEN_PAPER_FINAL_5SEED_REUSE"


def test_finalized_extension_status_remains_a_complete_state():
    assert extension_cells_complete("POSTHOC_BASELINE_5SEED_EXTENSION_CELLS_COMPLETE")
    assert extension_cells_complete(COMPLETE)
    assert not extension_cells_complete("POSTHOC_BASELINE_5SEED_EXTENSION_PREPARED")
