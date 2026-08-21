import numpy as np
import torch

from scripts.run_diffusion_quality_retest import _fit_ce_rep
from trainers import build_model


def test_ce_rep_trains_on_clean_and_augmented_pairs():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(8, 2, 16)).astype(np.float32)
    bundle = {"clean": x, "restored": x + .01, "labels": np.asarray([0, 1] * 4)}
    model = build_model({"name":"tcn","hidden_channels":4,"projection_dim":4,"levels":1}, 2, 2)
    history = _fit_ce_rep(model, bundle, bundle, [np.arange(8)],
                          {"batch_size":4,"learning_rate":.001,"early_stopping_patience":1}, "cpu")
    assert len(history) == 1
    assert np.isfinite(history[0]["validation_ce_loss"])
