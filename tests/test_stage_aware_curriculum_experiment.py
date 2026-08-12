import yaml
import numpy as np
from scripts.run_stage_aware_diffusion_curriculum import training_stage_names, validate_config
from scripts.summarize_stage_aware_diffusion_curriculum import mean_std, seed7_gate, three_seed_gate


def config():
    c=yaml.safe_load(open("configs/stage_aware_diffusion_curriculum.yaml",encoding="utf8")); b=yaml.safe_load(open(c["base_config"],encoding="utf8")); r=yaml.safe_load(open(c["r1_config"],encoding="utf8")); return c,b,r


def test_config_freezes_r1_mask_weights_targets_and_methods():
    c,b,r=config(); validate_config(c,b,r)
    assert c["methods"] == ["R1","C3-E","C3-S"] and c["frozen"]["noise_structure"] == "iid"


def test_training_stage_names_force_prefault_normal_labels_to_normal():
    assert training_stage_names(np.array([0,1,1]),np.array(["prefault","early","stable"])).tolist() == ["normal","early","stable"]


def metric(m=.9,far=.04,rec=.8,a=.92,e=.7,d=100): return {"macro_f1":m,"far":far,"fault_recall":rec,"auprc":a,"early_recall":e,"mean_delay":d}


def test_seed7_gate_requires_stage_gain_and_industrial_gain():
    c,_,_=config(); status,audit=seed7_gate({"R1":metric(),"C3-E":metric(e=.705),"C3-S":metric(e=.715)},c["seed7_gate"])
    assert status == "STAGE_AWARE_CURRICULUM_SEED7_GO" and audit["stage_gain_over_c3e"]


def test_three_seed_gate_go_curriculum_only_and_catastrophic():
    c,_,_=config(); seeds={str(s):{"R1":metric(),"C3-E":metric(e=.705),"C3-S":metric(e=.715)} for s in (7,42,2026)}
    assert three_seed_gate(seeds,c["three_seed_gate"])[0] == "STAGE_AWARE_DIFFUSION_CURRICULUM_3SEED_GO"
    for x in seeds.values(): x["C3-E"]["early_recall"]=.72
    assert three_seed_gate(seeds,c["three_seed_gate"])[0] == "EPOCH_CURRICULUM_3SEED_GO_STAGE_AWARE_NO_GAIN"
    seeds["42"]["C3-S"]["macro_f1"]=.87
    assert three_seed_gate(seeds,c["three_seed_gate"])[0] == "STAGE_AWARE_DIFFUSION_CURRICULUM_3SEED_NO_GO"


def test_sample_std_ddof_one(): assert mean_std([1,2,3]) == {"mean":2.,"std":1.}
