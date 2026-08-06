import numpy as np
import pytest

from frequency import (build_criticality, fit_frequency_scaler,
                       log_amplitude_phase, mask_jaccard)


SETTINGS = {"critical_ratio": .3, "weight_discriminative": .5, "weight_early": .3,
            "weight_run_stability": .2, "bootstrap_repeats": 8, "bootstrap_seed": 9}


def _fixture():
    rng = np.random.default_rng(3); features = rng.normal(size=(24, 3, 5)).astype(np.float32)
    run_uid = np.array(["training:normal:0001"] * 4 + ["training:normal:0002"] * 4
                       + ["training:fault_01:0001"] * 4 + ["training:fault_01:0002"] * 4
                       + ["training:fault_02:0001"] * 4 + ["training:fault_02:0002"] * 4)
    labels = np.array([0] * 8 + [1] * 16); stages = np.array(["prefault"] * 8 + ["early"] * 8 + ["stable"] * 8)
    features[labels == 1, 1, 3] += 3; features[stages == "early", 2, 2] += 2
    return features, {"run_uid": run_uid, "labels": labels}, stages


def test_rfft_shape_bins_and_finite_representation():
    values = np.random.default_rng(1).normal(size=(7, 52, 64)).astype(np.float32)
    amplitude, phase = log_amplitude_phase(values)
    assert amplitude.shape == phase.shape == (7, 52, 33)
    assert np.isfinite(amplitude).all() and np.isfinite(phase).all()


def test_frequency_scaler_is_train_only():
    train = np.random.default_rng(1).normal(size=(5, 2, 4))
    scaler = fit_frequency_scaler(train, "train")
    assert scaler.fit_split == "train" and scaler.transform(train).shape == train.shape
    with pytest.raises(ValueError, match="only be fitted on train"):
        fit_frequency_scaler(train, "validation")


def test_composite_mask_is_soft_non_degenerate_and_bootstrap_reproducible():
    features, bundle, stages = _fixture()
    first = build_criticality(features, bundle, stages, SETTINGS)
    second = build_criticality(features, bundle, stages, SETTINGS)
    assert first["fit_split"] == "train"
    assert np.all((first["soft_mask"] >= 0) & (first["soft_mask"] <= 1))
    assert not np.all(first["masks"]["composite"]) and np.any(first["masks"]["composite"])
    assert np.array_equal(first["bootstrap_overlap"], second["bootstrap_overlap"])
    assert mask_jaccard(first["masks"]["composite"], first["masks"]["composite"]) == 1


def test_discriminative_early_and_stability_use_run_aggregates():
    features, bundle, stages = _fixture(); result = build_criticality(features, bundle, stages, SETTINGS)
    assert result["discriminative"][1, 3] > np.median(result["discriminative"])
    assert result["early"][2, 2] > np.median(result["early"])
    assert result["run_counts"] == {"normal": 2, "fault": 4, "early_fault": 2}

