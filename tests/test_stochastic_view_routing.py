import numpy as np
import pytest
import yaml

from augmentations import stochastic_view_route
from baselines.external_augmentations import traditional_view
from scripts.run_stochastic_view_routing import _validate, variant_name


def _inputs(n: int = 128):
    clean = np.arange(n * 16, dtype=np.float32).reshape(n, 2, 8) / 100
    diffused = clean + np.float32(0.125)
    ids = np.asarray([f"sample-{index}" for index in range(n)])
    return clean, diffused, ids


def test_svr_p_zero_exactly_matches_frozen_scaling():
    clean, diffused, ids = _inputs(7)
    actual, audit = stochastic_view_route(clean, diffused, ids, 0, 11, 42, .05)
    expected = traditional_view(clean, ids, "SCALING", 42, 0, .05)
    assert np.array_equal(actual, expected)
    assert audit["p_zero_exact_scaling"] and audit["qdiffcl_route_count"] == 0


def test_svr_p_one_exactly_matches_final_qdiffcl():
    clean, diffused, ids = _inputs(7)
    actual, audit = stochastic_view_route(clean, diffused, ids, 1, 11, 42, .05)
    assert np.array_equal(actual, diffused)
    assert audit["p_one_exact_final"] and audit["scaling_route_count"] == 0


def test_svr_is_reproducible_order_independent_and_mutually_exclusive():
    clean, diffused, ids = _inputs()
    first, audit = stochastic_view_route(clean, diffused, ids, .5, 123, 456)
    second, repeated = stochastic_view_route(clean, diffused, ids, .5, 123, 456)
    order = np.arange(len(ids))[::-1]
    shuffled, shuffled_audit = stochastic_view_route(clean[order], diffused[order], ids[order], .5, 123, 456)
    assert np.array_equal(first, second)
    assert np.array_equal(first[order], shuffled)
    assert audit["route_mask_sha256"] == repeated["route_mask_sha256"]
    assert audit["qdiffcl_route_count"] + audit["scaling_route_count"] == len(ids)
    assert audit["exactly_one_branch_per_sample"] and not audit["simultaneous_augmentation"]
    assert .35 <= audit["realized_route_ratio"] <= .65
    assert shuffled_audit["realized_route_ratio"] == audit["realized_route_ratio"]


def test_svr_candidates_use_nested_fair_routes_and_validate_inputs():
    clean, diffused, ids = _inputs(64)
    _, low = stochastic_view_route(clean, diffused, ids, .25, 8, 9)
    _, high = stochastic_view_route(clean, diffused, ids, .75, 8, 9)
    assert low["fairness_sha256"] == high["fairness_sha256"]
    assert low["qdiffcl_route_count"] <= high["qdiffcl_route_count"]
    with pytest.raises(ValueError):
        stochastic_view_route(clean, diffused, ids, 1.01, 8, 9)


def test_svr_frozen_config_and_variant_grid_are_auditable():
    config = yaml.safe_load(open("configs/stochastic_view_routing.yaml", encoding="utf-8"))
    _validate(config)
    assert [variant_name(p) for p in config["candidates"]] == [
        "SVR_000", "SVR_025", "SVR_050", "SVR_075", "SVR_100"]
    assert config["audit"] == {"validation_only": True, "test_read": False}
