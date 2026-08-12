from copy import deepcopy

import numpy as np
import pytest
import yaml

from scripts.run_frequency_selective_r1_3seed import array_sha256, validate_frozen_config
from scripts.summarize_frequency_selective_r1_3seed import evaluate_gate, mean_sample_std


def _config():
    config = yaml.safe_load(open("configs/frequency_selective_r1_3seed.yaml", encoding="utf-8"))
    base = yaml.safe_load(open(config["base_config"], encoding="utf-8"))
    far = yaml.safe_load(open(config["far_fix_config"], encoding="utf-8"))
    return config, base, far


def test_frozen_seed_method_spectral_stage_and_criticality_protocol():
    config, base, far = _config(); validate_frozen_config(config, base, far)
    assert config["seeds"] == [7, 42, 2026] and config["methods"] == ["C0", "C1", "R1"]
    spectral = config["frozen"]["spectral_diffusion"]
    assert spectral == {"diffusion_steps": 50, "t_uniform": 3, "t_critical": 1, "t_noncritical": 5,
                        "preserve_phase": True, "preserve_dc": True,
                        "noise_budget_matching": "alpha_bar_mean", "noise_structure": "iid"}
    assert config["frozen"]["criticality"] == {"critical_ratio": .3, "weight_discriminative": .5,
                                                 "weight_early": .3, "weight_run_stability": .2}


def test_frozen_protocol_rejects_parameter_search_and_correlated_noise():
    config, base, far = _config()
    for field, value in (("t_noncritical", 4), ("t_critical", 2), ("noise_structure", "correlated")):
        changed = deepcopy(config); changed["frozen"]["spectral_diffusion"][field] = value
        with pytest.raises(ValueError, match="not frozen"):
            validate_frozen_config(changed, base, far)


def test_mask_hash_is_deterministic_and_content_sensitive():
    mask = np.arange(12, dtype=np.int8).reshape(3, 4)
    assert array_sha256(mask) == array_sha256(mask.copy())
    changed = mask.copy(); changed[0, 0] += 1
    assert array_sha256(mask) != array_sha256(changed)


def _seed(macro_delta=.01, far_delta=-.01, recall_delta=0, auprc_delta=0,
          early_delta=0, delay_delta=0):
    c1 = {"macro_f1": .8, "far": .05, "fault_recall": .8, "auprc": .9,
          "early_recall": .7, "mean_delay": 100.}
    r1 = dict(c1); r1.update({"macro_f1": .8 + macro_delta, "far": .05 + far_delta,
                              "fault_recall": .8 + recall_delta, "auprc": .9 + auprc_delta,
                              "early_recall": .7 + early_delta, "mean_delay": 100 + delay_delta})
    return {"C1": c1, "R1": r1}


def _gate(): return _config()[0]["gate"]


def test_gate_go_and_far_direction_reversal():
    seeds = {str(seed): _seed() for seed in (7, 42, 2026)}
    assert evaluate_gate(seeds, _gate())[0] == "FREQUENCY_SELECTIVE_R1_3SEED_GO"
    for value in seeds.values(): value["R1"]["far"] = .06
    assert evaluate_gate(seeds, _gate())[0] == "FREQUENCY_SELECTIVE_R1_3SEED_NO_GO"


def test_gate_distinguishes_unstable_from_no_go():
    unstable = {"7": _seed(), "42": _seed(macro_delta=-.001), "2026": _seed(macro_delta=-.001)}
    assert evaluate_gate(unstable, _gate())[0] == "FREQUENCY_SELECTIVE_R1_3SEED_UNSTABLE"
    failed = {str(seed): _seed(macro_delta=-.001) for seed in (7, 42, 2026)}
    assert evaluate_gate(failed, _gate())[0] == "FREQUENCY_SELECTIVE_R1_3SEED_NO_GO"


def test_recall_early_delay_and_catastrophic_gates():
    seeds = {"7": _seed(), "42": _seed(), "2026": _seed(recall_delta=-.031)}
    status, audit = evaluate_gate(seeds, _gate())
    assert status == "FREQUENCY_SELECTIVE_R1_3SEED_NO_GO" and audit["catastrophic_by_seed"]["2026"]
    early = {str(seed): _seed(early_delta=-.011) for seed in (7, 42, 2026)}
    assert evaluate_gate(early, _gate())[0] == "FREQUENCY_SELECTIVE_R1_3SEED_NO_GO"
    delay = {str(seed): _seed(delay_delta=17) for seed in (7, 42, 2026)}
    assert evaluate_gate(delay, _gate())[0] == "FREQUENCY_SELECTIVE_R1_3SEED_UNSTABLE"


def test_sample_std_uses_ddof_one():
    result = mean_sample_std([1, 2, 3])
    assert result == {"mean": 2., "std": 1.}


def test_seed7_reuse_is_explicitly_disabled_without_complete_hash_proof():
    config, _, _ = _config()
    assert config["seed7_reuse"] is False
    assert "缺少" in config["seed7_reuse_reason"]
