from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from diffusion import FIXED_STAGE_BETAS, apply_stage_perturbation_budget
from frequency import fault_stages
from scripts.audit_semantic_diffusion_augmentation import bases
from scripts.run_diffusion_quality_retest import epoch_orders, load_fixed_views
from scripts.run_frequency_selective_r1_3seed import array_sha256, file_sha256, sha256_strings
from scripts.run_stage_aware_diffusion_curriculum import (
    _fit_method, _strength_audit, training_stage_names,
)
from scripts.run_stage_frequency_diffusion_mvp import _build_frequency_components, _configure, _runtime
from trainers import build_model
from utils import environment_metadata, seed_everything, write_json


METHODS = ("R1", "B3")


def validate_config(config: dict[str, Any], audit: dict[str, Any]) -> None:
    if config["seeds"] != [7, 42, 2026] or config["methods"] != list(METHODS): raise ValueError("Budget seeds/methods changed")
    if config["stage_budget"] != FIXED_STAGE_BETAS: raise ValueError("Budget betas changed")
    if audit["status"] != "STAGE_TIMESTEP_EFFECT_WEAK" or not audit["budget_mvp_allowed"]:
        raise RuntimeError("Budget MVP requires weak timestep effect")


def build_budget_views(base_train, candidate, stage_names, betas, epochs, critical_mask):
    budget = apply_stage_perturbation_budget(base_train, candidate, stage_names, betas)
    timestep = {stage: 5 for stage in FIXED_STAGE_BETAS}
    r1_audit = _strength_audit(base_train, candidate, stage_names, timestep, critical_mask)
    budget_audit = _strength_audit(base_train, budget, stage_names, timestep, critical_mask)
    l1 = {stage: budget_audit["stages"][stage]["normalized_l1"] for stage in ("early", "middle", "stable")}
    valid = bool(l1["early"] < l1["middle"] < l1["stable"])
    return {"R1": [candidate] * epochs, "B3": [budget] * epochs}, {"R1": [r1_audit] * epochs, "B3": [budget_audit] * epochs}, valid


def run_seed(config, seed, views, base, stages, critical, augmenter, fingerprints):
    output=Path(config["output_dir"])/f"seed_{seed}"; result_path=output/"result.json"
    if result_path.exists(): return json.loads(result_path.read_text(encoding="utf-8"))
    base_config=yaml.safe_load(Path(config["base_config"]).read_text(encoding="utf-8")); runtime=_runtime(base_config,seed)
    stage_names=training_stage_names(views["train"]["labels"],stages["train"]); sampling_seed=seed+int(base_config["spectral_diffusion"]["sampling_seed_offset"])
    candidate,_=augmenter.augment(base["train"],"selective",sampling_seed,5,int(runtime["batch_size"]),noise_structure="iid")
    epoch_views,epoch_audits,budget_valid=build_budget_views(base["train"],candidate,stage_names,config["stage_budget"],int(runtime["epochs"]),critical["masks"]["composite"])
    val_aug,_=augmenter.augment(base["validation"],"selective",sampling_seed+100,5,int(runtime["batch_size"]),noise_structure="iid")
    test_aug,_=augmenter.augment(base["test"],"selective",sampling_seed+200,5,int(runtime["batch_size"]),noise_structure="iid")
    pretrain_orders=epoch_orders(len(base["train"]),int(runtime["epochs"]),seed+10_000); probe_orders=epoch_orders(len(base["train"]),int(runtime["probe_epochs"]),seed+20_000)
    seed_everything(seed); template=build_model(runtime["model"],base["train"].shape[1],2); initial_state=copy.deepcopy(template.state_dict())
    common={**fingerprints,"seed":seed,"initialization_sha256":__import__('scripts.run_diffusion_quality_retest',fromlist=['_state_hash'])._state_hash(initial_state),
            "pretrain_order_sha256":sha256_strings([','.join(map(str,x)) for x in pretrain_orders]),"probe_order_sha256":sha256_strings([','.join(map(str,x)) for x in probe_orders]),
            "same_r1_candidate_noise":True,"beta_only_training_augmentation":True}
    methods={name:_fit_method(name,epoch_views[name],epoch_audits[name],val_aug,test_aug,views,base,stages,initial_state,pretrain_orders,probe_orders,runtime,str(config["device"]),output/name/"model.pt",{**common,"method":name}) for name in METHODS}
    result={"seed":seed,"methods":methods,"budget_order_valid":budget_valid,"fairness":common,"stage_budget":config["stage_budget"],
            "stage_not_used_by_encoder_probe_validation_or_test":True}; write_json(result_path,result); return result


def run(config):
    final=Path(config["output_dir"])/"result.json"
    if final.exists(): return json.loads(final.read_text(encoding="utf-8"))
    audit=json.loads(Path(config["audit_result"]).read_text(encoding="utf-8")); validate_config(config,audit)
    base_config=yaml.safe_load(Path(config["base_config"]).read_text(encoding="utf-8")); _configure(base_config)
    views,_=load_fixed_views(base_config); base=bases(views); stages={s:fault_stages(views[s],base_config) for s in views}; critical,augmenter=_build_frequency_components(base_config,views,base,stages,str(config["device"]))
    fingerprints={"manifest_sha256":file_sha256(config["fixed_views_manifest"]),"mask_sha256":array_sha256(critical["masks"]["composite"]),"audit_result_sha256":file_sha256(config["audit_result"]),
                  "training_code_sha256":sha256_strings([file_sha256(p) for p in ("diffusion/stage_budget.py","scripts/run_stage_perturbation_budget.py","scripts/run_stage_aware_diffusion_curriculum.py")])}
    results={"7":run_seed(config,7,views,base,stages,critical,augmenter,fingerprints)}
    from scripts.summarize_stage_perturbation_budget import summarize
    seed7=summarize(config,results,fingerprints)
    if seed7["seed7_status"]=="STAGE_PERTURBATION_BUDGET_SEED7_GO":
        for seed in (42,2026): results[str(seed)]=run_seed(config,seed,views,base,stages,critical,augmenter,fingerprints)
    result=summarize(config,results,fingerprints); result.update(environment_metadata()); final.parent.mkdir(parents=True,exist_ok=True); write_json(final,result); summarize(config,results,fingerprints,result=result,report_path=config["report"]); return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/stage_perturbation_budget.yaml"); args=parser.parse_args(); c=yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); r=run(c); print(json.dumps({"status":r["status"],"seed7":r["seed7_status"],"seeds":list(r["seed_results"])},ensure_ascii=False))
if __name__=="__main__": main()
