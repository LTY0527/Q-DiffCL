import numpy as np
import pytest

from frequency import build_rival_aware_criticality
from scripts.summarize_rival_aware_reliable_diagnostic_criticality import _mask_audit


SETTINGS = {"critical_ratio": .3, "weight_discriminative": .5, "weight_early": .3,
            "weight_run_stability": .2, "bootstrap_repeats": 12, "bootstrap_seed": 17,
            "hard_rival_quantile": .25, "diagnostic_classes": [1, 2, 3]}


def _fixture():
    rng = np.random.default_rng(9)
    features = rng.normal(0, .15, size=(40, 2, 5)).astype(np.float32)
    run_uid = np.array(["training:fault_00:n1"] * 4 + ["training:fault_00:n2"] * 4
                       + ["training:fault_01:a"] * 4 + ["training:fault_01:b"] * 4
                       + ["training:fault_02:a"] * 4 + ["training:fault_02:b"] * 4
                       + ["training:fault_03:a"] * 4 + ["training:fault_03:b"] * 4
                       + ["training:fault_03:c"] * 4 + ["training:fault_02:c"] * 4)
    labels = np.array([0] * 8 + [1] * 32)
    stages = np.array(["prefault"] * 8 + ["early"] * 12 + ["stable"] * 20)
    features[8:16, 0, 1] += 4.0
    features[16:28, 1, 3] += 4.0
    features[28:36, 0, 4] -= 4.0
    return features, {"run_uid": run_uid, "labels": labels}, stages


def test_rrdc_is_fault_only_reliable_and_deterministic():
    features, bundle, stages = _fixture()
    first = build_rival_aware_criticality(features, bundle, stages, SETTINGS)
    second = build_rival_aware_criticality(features, bundle, stages, SETTINGS)
    assert first["fit_split"] == "train" and set(first["soft_masks"]) == {0, 1, 2, 3}
    assert first["fault_run_counts"] == {1: 2, 2: 3, 3: 3}
    assert 0 not in first["diagnostic_classes"]
    for kind, item in first["diagnostic"].items():
        assert set(item["pairwise"]) == ({1, 2, 3} - {kind})
        assert np.all((item["reliability"] >= 0) & (item["reliability"] <= 1))
        assert np.array_equal(item["reliability"], second["diagnostic"][kind]["reliability"])
        assert item["hardest_rival"] in item["pairwise"]
        assert int(item["hard_mask"].sum()) == 3
    assert first["combination"] == "C_shared + C_diag * R"


def test_rrdc_rejects_unfrozen_method_choices():
    features, bundle, stages = _fixture()
    for key, value in (("hard_rival_quantile", .5), ("critical_ratio", .4), ("weight_early", .2)):
        settings = dict(SETTINGS); settings[key] = value
        with pytest.raises(ValueError):
            build_rival_aware_criticality(features, bundle, stages, settings)


def test_rrdc_final_score_is_exact_frozen_sum():
    features, bundle, stages = _fixture()
    result = build_rival_aware_criticality(features, bundle, stages, SETTINGS)
    for kind in result["diagnostic_classes"]:
        expected = result["shared"]["composite"] + result["diagnostic"][kind]["reliable_score"]
        assert np.array_equal(result["final"][kind]["score"], expected)


def test_rrdc_mask_audit_records_reliability_hashes_and_rivals():
    features, bundle, stages = _fixture()
    result = build_rival_aware_criticality(features, bundle, stages, SETTINGS)
    ready = {"shared_hard_mask": result["shared"]["masks"]["composite"].astype(int).tolist(),
             "diagnostic": {}, "final": {}}
    for kind, item in result["diagnostic"].items():
        ready["diagnostic"][str(kind)] = {
            "hard_mask": item["hard_mask"].astype(int).tolist(),
            "reliability": item["reliability"].tolist(),
            "hardest_rival": item["hardest_rival"], "hardest_rival_score": item["hardest_rival_score"]}
        ready["final"][str(kind)] = {"hard_mask": result["final"][kind]["hard_mask"].astype(int).tolist()}
    audit = _mask_audit(ready)
    assert set(audit["reliability"]) == {"1", "2", "3"}
    assert len(audit["shared_mask_sha256"]) == 64
    assert audit["hardest_rivals"]["1"]["rival"] in {2, 3}
