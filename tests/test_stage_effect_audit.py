import pytest
from scripts.audit_stage_timestep_effect import STAGES, classify_effect, effect_ratio


def records(time=1.05,freq=1.05,representation=1.05):
    return {stage:{"ratios":{"time_t5_t3":time,"noncritical_frequency_t5_t3":freq,"representation_t5_t3":representation},
                   "monotonic":{"time":True,"noncritical_frequency":True,"representation":True}} for stage in STAGES}


def test_effect_ratio_and_only_allowed_timesteps_config():
    assert effect_ratio(2,1)==2
    import yaml
    config=yaml.safe_load(open("configs/stage_effect_audit.yaml",encoding="utf8"))
    assert config["timesteps"] == [3,4,5]


def test_weak_effect_allows_budget_and_two_strong_layers_stop_it():
    assert classify_effect(records())[0] == "STAGE_TIMESTEP_EFFECT_WEAK"
    assert classify_effect(records(1.2,1.2,1.05))[0] == "STAGE_TIMESTEP_EFFECT_PRESENT_BUT_TASK_NO_GAIN"


def test_exact_threshold_counts_as_strong():
    status,audit=classify_effect(records(1.1,1.1,1.0))
    assert status == "STAGE_TIMESTEP_EFFECT_PRESENT_BUT_TASK_NO_GAIN" and audit["strong_layer_count"]==2
