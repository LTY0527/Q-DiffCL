import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from diffusion import scale_spectral_budget
from scripts.run_budget_shrinkage_diagnostic import _validate, rho_name, validation_metrics


@pytest.mark.parametrize("rho", [0.0, .25, .5, .75, 1.0])
def test_budget_shrinkage_is_exact_and_preserves_allocation(rho):
    final = torch.tensor([[0.0, .1, .2], [0.0, .3, .4]])
    scaled = scale_spectral_budget(final, rho, preserve_dc=True)
    assert scaled.mean().item() == pytest.approx((final * rho).mean().item(), abs=1e-8)
    assert torch.all(scaled[:, 0] == 0)
    positive = final > 0
    assert torch.allclose(scaled[positive] / final[positive], torch.full_like(final[positive], rho))


def test_budget_shrinkage_rejects_invalid_rho_and_dc():
    with pytest.raises(ValueError, match="rho"):
        scale_spectral_budget(torch.ones(2, 3) * .1, 1.1, False)
    with pytest.raises(ValueError, match="DC"):
        scale_spectral_budget(torch.ones(2, 3) * .1, .5, True)


def test_budget_config_freezes_grid_final_and_masks():
    config = yaml.safe_load(Path("configs/budget_shrinkage_diagnostic.yaml").read_text(encoding="utf-8"))
    _validate(config)
    changed = copy.deepcopy(config); changed["rhos"][2] = .55
    with pytest.raises(ValueError, match="five-point"):
        _validate(changed)


def test_validation_metrics_refuses_test_bearing_record():
    record = {"evaluation_split": "validation", "test_metrics_read": False,
              "method": {"test": {}, "validation": {}}}
    with pytest.raises(RuntimeError, match="contains test"):
        validation_metrics("TEP", record)


def test_rho_names_are_stable():
    assert [rho_name(value) for value in (0, .25, .5, .75, 1)] == [
        "rho_000", "rho_025", "rho_050", "rho_075", "rho_100"]
