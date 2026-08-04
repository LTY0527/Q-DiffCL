import numpy as np
import pytest

from diffusion.fixed_views import (fit_quality_scale, mask_id,
                                   per_sample_masked_mae, quality_scores,
                                   split_window_id, validate_view_splits)
from scripts.run_diffusion_quality_retest import epoch_orders, fit_train_only_quality


def _view(run_uid: str, errors=(1.0, 2.0)):
    clean = np.asarray(errors, dtype=np.float32)[:, None, None]
    restored = np.zeros_like(clean)
    observation = np.zeros_like(clean, dtype=bool)
    count = len(clean)
    ids = np.asarray([f"{run_uid}:samples_{i}_{i}" for i in range(count)])
    masks = np.asarray([mask_id(value) for value in observation])
    return {"clean": clean, "degraded": restored.copy(), "restored": restored,
            "observation": observation, "labels": np.arange(count) % 2,
            "window_id": ids, "run_uid": np.asarray([run_uid] * count), "mask_id": masks}


def test_quality_scale_and_scores_are_finite():
    errors = np.array([0.1, 0.2, 0.4])
    scale = fit_quality_scale(errors, "median")
    scores = quality_scores(errors, scale, 0.1)
    assert scale == pytest.approx(0.2)
    assert np.isfinite(scores).all()
    assert np.all((scores >= 0.1) & (scores <= 1.0))


def test_quality_scale_uses_train_only():
    config = {"quality": {"formula": "exp(-masked_mae/scale)", "scale_estimator": "median", "q_min": 0.1}}
    views = {"train": _view("training:normal:0001", (0.1, 0.3)),
             "validation": _view("training:normal:0002", (10.0, 20.0)),
             "test": _view("testing:normal:0001", (30.0, 40.0))}
    _, first = fit_train_only_quality(views, config)
    views["validation"] = _view("training:normal:0002", (1000.0, 2000.0))
    views["test"] = _view("testing:normal:0001", (3000.0, 4000.0))
    _, second = fit_train_only_quality(views, config)
    assert first["scale_fit_split"] == "train"
    assert first["scale"] == second["scale"] == pytest.approx(0.2)


def test_fixed_view_manifest_has_no_run_leakage():
    views = {"train": _view("training:normal:0001"),
             "validation": _view("training:normal:0002"),
             "test": _view("testing:normal:0001")}
    expected = {key: [str(value["run_uid"][0])] for key, value in views.items()}
    validate_view_splits(views, expected)
    views["test"] = _view("training:normal:0001")
    expected["test"] = ["training:normal:0001"]
    with pytest.raises(ValueError, match="leakage"):
        validate_view_splits(views, expected)


def test_epoch_orders_are_reproducible():
    first = epoch_orders(17, 4, 99)
    second = epoch_orders(17, 4, 99)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_window_metadata_and_masked_mae():
    assert split_window_id("training:fault_01:0001:samples_21_84") == ("training:fault_01:0001", 21, 84)
    view = _view("training:normal:0001", (0.5, 1.5))
    assert np.allclose(per_sample_masked_mae(view["clean"], view["restored"], view["observation"]), [0.5, 1.5])
