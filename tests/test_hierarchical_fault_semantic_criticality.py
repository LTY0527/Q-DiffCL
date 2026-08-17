import numpy as np

from diffusion import DiffusionSchedule, FrequencyForwardDiffusion, fit_spectral_statistics
from frequency import build_hierarchical_criticality
from scripts.run_hierarchical_fault_semantic_criticality import multiclass_labels
from scripts.summarize_hierarchical_fault_semantic_criticality import mask_audit


SETTINGS = {"critical_ratio": .3, "weight_discriminative": .5, "weight_early": .3,
            "weight_run_stability": .2, "bootstrap_repeats": 4, "bootstrap_seed": 3,
            "diagnostic_classes": [1, 2, 3], "hierarchical_shared_weight": .5,
            "hierarchical_diagnostic_weight": .5}


def _fixture():
    rng = np.random.default_rng(4); features = rng.normal(size=(32, 2, 5)).astype(np.float32)
    run_uid = np.array(["training:fault_00:n1"] * 4 + ["training:fault_00:n2"] * 4
                       + ["training:fault_01:a"] * 4 + ["training:fault_01:b"] * 4
                       + ["training:fault_02:a"] * 4 + ["training:fault_02:b"] * 4
                       + ["training:fault_03:a"] * 4 + ["training:fault_03:b"] * 4)
    labels = np.array([0] * 8 + [1] * 24); stages = np.array(["prefault"] * 8 + ["early"] * 8 + ["stable"] * 16)
    features[8:16, 0, 1] += 5; features[16:24, 1, 3] += 5; features[24:32, 0, 4] -= 5
    return features, {"run_uid": run_uid, "labels": labels}, stages


def test_hfsc_builds_distinct_train_only_class_maps_with_frozen_weights():
    features, bundle, stages = _fixture(); result = build_hierarchical_criticality(features, bundle, stages, SETTINGS)
    assert result["fit_split"] == "train" and result["diagnostic_classes"] == [1, 2, 3]
    assert result["fault_run_counts"] == {1: 2, 2: 2, 3: 2}
    assert set(result["soft_masks"]) == {0, 1, 2, 3}
    assert result["diagnostic"][1]["score"][0, 1] > np.median(result["diagnostic"][1]["score"])
    assert result["diagnostic"][2]["score"][1, 3] > np.median(result["diagnostic"][2]["score"])
    assert not np.array_equal(result["diagnostic"][1]["hard_mask"], result["diagnostic"][2]["hard_mask"])


def test_hierarchical_diffusion_matches_global_path_when_all_masks_match():
    rng = np.random.default_rng(8); values = rng.normal(size=(6, 2, 8)).astype(np.float32)
    statistics = fit_spectral_statistics(values); alpha = DiffusionSchedule.cosine(10, "cpu").alpha_bars
    mask = rng.uniform(size=(2, 5)).astype(np.float32)
    augmenter = FrequencyForwardDiffusion(statistics, alpha, mask, 3, 1)
    global_values, global_diag = augmenter.augment(values, "selective", 19, 5, 3)
    hierarchical, diag = augmenter.augment_hierarchical(values, np.array([0, 1, 0, 1, 0, 1]), {0: mask, 1: mask}, 19, 5, 3)
    assert np.array_equal(global_values, hierarchical)
    assert abs(diag["expected_total_noise_budget"] - global_diag["expected_total_noise_budget"]) < 1e-7


def test_tep_multiclass_mapping_keeps_prefault_windows_normal():
    bundle = {"labels": np.array([0, 1, 0, 1]),
              "run_uid": np.array(["training:fault_01:a", "training:fault_01:a",
                                   "training:fault_20:b", "training:fault_20:b"])}
    assert multiclass_labels(bundle).tolist() == [0, 1, 0, 20]


def test_mask_audit_detects_class_specific_patterns():
    shared = np.array([[1, 1, 0, 0]], bool)
    diagnostic = {1: np.array([[1, 0, 1, 0]], bool), 2: np.array([[0, 1, 0, 1]], bool)}
    audit = mask_audit(shared, diagnostic, diagnostic)
    assert audit["class_specific_patterns_confirmed"]
    assert audit["shared_vs_diagnostic"]["1"]["changed_bins"] == 2
