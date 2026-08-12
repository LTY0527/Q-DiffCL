import numpy as np
import yaml
import pytest
from diffusion import FIXED_STAGE_BETAS,apply_stage_perturbation_budget
from scripts.run_stage_perturbation_budget import build_budget_views,validate_config
from scripts.summarize_stage_perturbation_budget import mean_std,seed7_gate,three_gate

def metric(m=.9,far=.04,rec=.8,a=.92,e=.7,d=100): return {"macro_f1":m,"far":far,"fault_recall":rec,"auprc":a,"early_recall":e,"mean_delay":d}
def test_formula_beta_endpoints_and_fixed_values():
    b=np.zeros((4,1,2),np.float32); c=np.ones_like(b); stages=np.array(["normal","early","middle","stable"]); out=apply_stage_perturbation_budget(b,c,stages,FIXED_STAGE_BETAS)
    assert np.allclose(out[:,0,0],[1,.6,.8,1]) and FIXED_STAGE_BETAS=={"normal":1.,"early":.6,"middle":.8,"stable":1.}
    with pytest.raises(ValueError): apply_stage_perturbation_budget(b,c,stages,{**FIXED_STAGE_BETAS,"early":.5})
def test_beta_zero_and_one_math_general_identity():
    base=np.array([1.]); candidate=np.array([3.]); assert np.allclose(base+0*(candidate-base),base); assert np.allclose(base+1*(candidate-base),candidate)
def test_config_requires_weak_audit():
    c=yaml.safe_load(open("configs/stage_perturbation_budget.yaml",encoding="utf8")); validate_config(c,{"status":"STAGE_TIMESTEP_EFFECT_WEAK","budget_mvp_allowed":True})
    with pytest.raises(RuntimeError): validate_config(c,{"status":"STAGE_TIMESTEP_EFFECT_PRESENT_BUT_TASK_NO_GAIN","budget_mvp_allowed":False})
def test_seed7_and_three_seed_gates_and_catastrophe():
    c=yaml.safe_load(open("configs/stage_perturbation_budget.yaml",encoding="utf8")); m={"R1":metric(),"B3":metric(e=.72)}; assert seed7_gate(m,c["seed7_gate"],True)[0]=="STAGE_PERTURBATION_BUDGET_SEED7_GO"
    sm={str(s):m for s in (7,42,2026)}; assert three_gate(sm,c["three_seed_gate"])[0]=="STAGE_PERTURBATION_BUDGET_3SEED_GO"; sm["42"]={"R1":metric(),"B3":metric(m=.87)}; assert three_gate(sm,c["three_seed_gate"])[0]=="STAGE_PERTURBATION_BUDGET_3SEED_NO_GO"
def test_sample_std_ddof_one(): assert mean_std([1,2,3])=={"mean":2.,"std":1.}
