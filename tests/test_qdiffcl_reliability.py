from __future__ import annotations

import numpy as np
import pytest

from analysis.qdiffcl_reliability import (
    combine_reliability,
    compute_full_rho_regret,
    mask_reliability,
    spearman_rank_reliability,
    summarize_bootstrap_reliability,
    _top_mask,
)


def test_identical_maps_rank_reliability_one():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    assert abs(spearman_rank_reliability(x, x) - 1.0) < 1e-9


def test_identical_masks_jaccard_one():
    m = np.array([True, True, False, False, True, False])
    assert mask_reliability(m, m) == pytest.approx(1.0)


def test_R_bounds_zero_one():
    for rr in [-0.5, 0.0, 0.3, 0.7, 1.0, 1.5]:
        for rm in [-0.1, 0.0, 0.2, 0.8, 1.0, 2.0]:
            r = combine_reliability(rr, rm)
            assert 0.0 <= r <= 1.0


def test_reversed_ranking_handled():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert spearman_rank_reliability(x, y) == pytest.approx(-1.0)


def test_ties_deterministic():
    x = np.array([1.0, 1.0, 2.0, 2.0, 3.0])
    y = np.array([1.0, 1.0, 2.0, 2.0, 3.0])
    a = spearman_rank_reliability(x, y)
    b = spearman_rank_reliability(x, y)
    assert np.isfinite(a)
    assert a == pytest.approx(b)


def test_fixed_bootstrap_seed_deterministic():
    rng = np.random.default_rng(42)
    n = 300
    features = rng.normal(size=(n, 50))
    run_uids = np.array([f"r{i % 12}" for i in range(n)])
    labels = np.array([0 if i % 3 == 0 else 1 for i in range(n)])
    stages = np.array(["early" if i % 6 == 1 else ("normal" if labels[i] == 0 else "fault") for i in range(n)])
    a = summarize_bootstrap_reliability(features, run_uids, labels, stages, bootstrap_seed=22042, bootstrap_repeats=8)
    b = summarize_bootstrap_reliability(features, run_uids, labels, stages, bootstrap_seed=22042, bootstrap_repeats=8)
    assert a["R"] == pytest.approx(b["R"])
    assert a["reference_map_sha256"] == b["reference_map_sha256"]


def test_no_window_iid_bootstrap_path():
    rng = np.random.default_rng(1)
    n = 60
    features = rng.normal(size=(n, 20))
    run_uids = np.array([f"r{i % 6}" for i in range(n)])
    labels = np.array([0 if i % 3 == 0 else 1 for i in range(n)])
    stages = np.array(["fault" if labels[i] else "normal" for i in range(n)])
    with pytest.raises(ValueError, match="window_iid.*forbidden"):
        summarize_bootstrap_reliability(features, run_uids, labels, stages, bootstrap_strategy="window_iid")


def test_tep10_hold_cannot_produce_full_R():
    from scripts.audit_qdiffcl_reliability import FRACTIONS, fraction_numeric
    assert "f010" not in FRACTIONS["TEP"]


def test_test_rows_rejected_filtered():
    from scripts.audit_qdiffcl_reliability import attach_regret
    import tempfile, json
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        rho_dir = tmp_path / "fake_outer"
        rho_dir.mkdir(parents=True, exist_ok=True)
        rho_file = rho_dir / "rho_selection.json"
        with open(rho_file, "w", encoding="utf-8") as f:
            json.dump({"outer_test_read": True, "candidate_rows": []}, f)
        cells = [{"_rho_dir": str(rho_dir), "dataset": "3W", "outer_id": 31001, "fraction": "f100", "fraction_numeric": 1.0}]
        with pytest.raises(RuntimeError, match="rho selection used test"):
            attach_regret(cells)
    from scripts.audit_qdiffcl_reliability import build_cells_from_context, load_yaml_config, CONFIG_PATH
    config = load_yaml_config(CONFIG_PATH)
    algorithm = config["algorithm"]
    cells = build_cells_from_context(config, algorithm, force_rebuild=False)
    assert len(cells) > 0
    for c in cells:
        hashes = json.loads(c["source_artifact_hashes"])
        assert len(hashes.get("audit_sha256", "")) == 64
        assert len(hashes.get("criticality_sha256", "")) == 64


def test_frozen_historical_criticality_top_mask_unchanged():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    m = _top_mask(values, 0.30)
    assert m.sum() == 3
    assert m[-1] and m[-2] and m[-3]
    v2 = np.array([5.0, 1.0, 3.0, 4.0, 2.0, 6.0, 10.0, 8.0, 7.0, 9.0])
    m2 = _top_mask(v2, 0.30)
    assert m2.sum() == 3
    assert set(np.where(v2 >= v2[np.where(m2)[0]].min())[0].tolist()) == set(np.where(m2)[0].tolist())
