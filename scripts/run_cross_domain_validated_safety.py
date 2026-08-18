from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import yaml
from scipy.stats import spearmanr

from diffusion import (DiffusionSchedule, FrequencyForwardDiffusion,
                       cross_domain_safe_variance, fit_spectral_statistics)
from frequency import (build_criticality, build_tep_cross_domain_safety,
                       build_three_w_cross_domain_safety, fault_stages,
                       fit_frequency_scaler, log_amplitude_phase)
from scripts.audit_semantic_diffusion_augmentation import bases
from scripts.run_diffusion_quality_retest import load_fixed_views
from scripts.run_diffusion_quality_retest import _state_hash, epoch_orders
from scripts.run_domain_reliable_safe_frequency_diffusion import _load_three_w_train
from scripts.run_3w_diffusion_1seed import (CDVS_METHOD, run as run_three_w)
from scripts.run_frequency_selective_r1_3seed import (_fit_method as fit_tep_method,
                                                       file_sha256, sha256_strings)
from scripts.run_stage_frequency_diffusion_mvp import (_configure, _runtime,
                                                        augmentation_mechanism_metrics)
from trainers import build_model
from utils import seed_everything, select_device, write_json


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): _jsonable(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [_jsonable(v) for v in value]
    return value


def _three_w(config: dict[str, Any], data_root: Path):
    stage=config["three_w"]; values,bundle,stages,wells,_=_load_three_w_train(stage,data_root)
    log=log_amplitude_phase(values)[0]; features=fit_frequency_scaler(log,"train").transform(log)
    safety=build_three_w_cross_domain_safety(features,bundle,stages,wells,stage["criticality"])
    r1=build_criticality(features,bundle,stages,stage["criticality"],log)
    return safety,r1,{"train_windows":len(values),"train_wells":sorted(set(map(str,wells))),"test_used":False}


def _tep(config: dict[str, Any]):
    stage=config["tep"]; base_config=yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8"))
    _configure(base_config); views,_=load_fixed_views(base_config); clean=bases(views); stages=fault_stages(views["train"],base_config)
    log=log_amplitude_phase(clean["train"])[0]; features=fit_frequency_scaler(log,"train").transform(log)
    safety=build_tep_cross_domain_safety(features,views["train"],stages,stage["criticality"],8,7)
    r1=build_criticality(features,views["train"],stages,stage["criticality"],log)
    return safety,r1,{"train_windows":len(clean["train"]),"train_runs":len(np.unique(views["train"]["run_uid"])),"test_used":False}


def _audit(name: str,safety: dict[str,Any],r1:dict[str,Any],stage:dict[str,Any],drfd:dict[str,Any],device:str):
    spectral=stage["spectral_diffusion"]; schedule=DiffusionSchedule.cosine(int(spectral["diffusion_steps"]),device)
    _,allocation=cross_domain_safe_variance(schedule.alpha_bars,r1["soft_mask"],safety["safe_prob"],
        bool(spectral["preserve_dc"]),int(spectral["t_critical"]),int(spectral["t_uniform"]),int(spectral["t_noncritical"]))
    safe=np.asarray(safety["safe_prob"]); unsafe=np.asarray(safety["unsafe_rate"]); r1t=np.asarray(allocation["r1_timestep"])
    cdvs=np.asarray(allocation["cdvs_timestep"]); protected=r1t<=3; unsafe_bins=(r1t>3)&(safe==0); safe_non=(r1t>3)&(safe>0)
    drfd_t=np.asarray(drfd["allocation"]["safe_timestep"]); rank_iqr=np.asarray(drfd["reliability"]["rank_iqr"])
    invariants={key:allocation[key] for key in ("protected_timestep_not_increased","protected_variance_not_increased",
        "unsafe_variance_not_increased","budget_adjustment_only_safe_noncritical","maximum_variance_respected","finite")}
    interior=float(np.mean((safe>0)&(safe<1))); budget_ok=allocation["budget_error_fraction"]<=float(spectral["maximum_budget_error"])
    nondegenerate=interior>=.05 and float(np.mean(np.abs(cdvs-drfd_t)))>1e-3
    passed=all(invariants.values()) and budget_ok and nondegenerate
    return {"dataset":name,"passed":passed,"safe_prob_distribution":{q:float(v) for q,v in zip(
        ("min","p25","median","p75","max"),np.quantile(safe,[0,.25,.5,.75,1]))},
        "unsafe_rate_distribution":{q:float(v) for q,v in zip(("min","p25","median","p75","max"),np.quantile(unsafe,[0,.25,.5,.75,1]))},
        "interior_safe_prob_fraction":interior,"valid_support":int(np.asarray(safety["valid_support"]).max()),
        "unsafe_r1_noncritical_bin_count":int(np.sum((r1t>3)&(unsafe>0))),
        "rank_iqr_safe_prob_spearman":float(spearmanr(rank_iqr.reshape(-1),safe.reshape(-1)).statistic),
        "mean_absolute_timestep_difference_from_drfd":float(np.mean(np.abs(cdvs-drfd_t))),
        "category_timestep":{
            "protected":float(cdvs[protected].mean()) if protected.any() else None,
            "unsafe":float(cdvs[unsafe_bins].mean()) if unsafe_bins.any() else None,
            "safe_noncritical":float(cdvs[safe_non].mean()) if safe_non.any() else None},
        "typical_stable_rank_high_risk_bins":[{"channel":int(i//safe.shape[1]),"frequency_bin":int(i%safe.shape[1]),
            "safe_prob":float(safe.reshape(-1)[i]),"rank_iqr":float(rank_iqr.reshape(-1)[i])}
            for i in np.argsort((safe+rank_iqr).reshape(-1))[:20]],
        "invariants":invariants,"budget_ok":budget_ok,"nondegenerate":nondegenerate,
        "allocation":_jsonable(allocation),"safety":_jsonable(safety),"r1_soft_mask":r1["soft_mask"].tolist()}


def run_stage_0(config:dict[str,Any],data_root:Path):
    phase=json.loads(Path(config["phase_a"]["output"]).read_text(encoding="utf-8"))
    if phase["status"]!="DRFD_RANK_RELIABILITY_INSUFFICIENT": raise RuntimeError("Phase A does not permit CDVS")
    drfd=json.loads(Path(config["drfd_mechanism_audit"]).read_text(encoding="utf-8")); device=select_device(str(config["device"]))
    sw,rw,scopew=_three_w(config,data_root); st,rt,scopet=_tep(config)
    aw=_audit("3W",sw,rw,config["three_w"],drfd["three_w"],device); at=_audit("TEP",st,rt,config["tep"],drfd["tep"],device)
    status="CDVS_MECHANISM_GO" if aw["passed"] and at["passed"] else "CDVS_MECHANISM_NO_GO"
    result={"status":status,"stage":0,"new_training_runs":0,"three_w":{**aw,"scope":scopew},"tep":{**at,"scope":scopet},
            "test_or_validation_used_for_cdvs":False}
    write_json(Path(config["stage_0"]["output"]),result)
    write_json(Path(config["stage_0"]["fold_audit"]),{"three_w":_jsonable(sw),"tep":_jsonable(st)})
    path=Path(config["stage_0"]["profile_csv"]); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8-sig",newline="") as h:
        writer=csv.DictWriter(h,fieldnames=("dataset","channel","frequency_bin","unsafe_rate","safe_prob","valid_support"));writer.writeheader()
        for label,safety in (("3W",sw),("TEP",st)):
            for c in range(safety["safe_prob"].shape[0]):
                for f in range(safety["safe_prob"].shape[1]): writer.writerow({"dataset":label,"channel":c,"frequency_bin":f,
                    "unsafe_rate":safety["unsafe_rate"][c,f],"safe_prob":safety["safe_prob"][c,f],"valid_support":safety["valid_support"][c,f]})
    _report(Path(config["stage_0"]["report"]),result); return result


def _require(path:Path,status:str):
    if not path.exists(): raise RuntimeError(f"required Gate missing: {path}")
    value=json.loads(path.read_text(encoding="utf-8"))
    if value["status"]!=status: raise RuntimeError(f"required Gate is not {status}")
    return value


def _run_three_w_seed(config:dict[str,Any],data_root:Path,seed:int):
    stage=config["three_w"]; current=yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8"))
    current["seed"]=int(seed);current["protocol_seed"]=42;current.pop("criticality_source",None)
    current["cross_domain_validated_safety"]=True;current["methods"]=[CDVS_METHOD]
    current["training"]["supcon_batching"]="original";current["criticality"]=copy.deepcopy(stage["criticality"])
    current["output_dir"]=str(Path(config["output"]["three_w"])/f"seed_{seed}")
    result=run_three_w(current,data_root)
    return {"result_path":str(Path(current["output_dir"])/"result.json"),"method":result["methods"][CDVS_METHOD],"fairness":result["fairness"]}


def _run_tep_seed(config:dict[str,Any],seed:int):
    stage=config["tep"];base_config=yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8"))
    _configure(base_config);views,_=load_fixed_views(base_config);clean=bases(views);stages={s:fault_stages(views[s],base_config) for s in views}
    log=log_amplitude_phase(clean["train"])[0];features=fit_frequency_scaler(log,"train").transform(log)
    safety=build_tep_cross_domain_safety(features,views["train"],stages["train"],stage["criticality"],8,7)
    r1=build_criticality(features,views["train"],stages["train"],stage["criticality"],log)
    statistics=fit_spectral_statistics(clean["train"],float(stage["spectral_diffusion"]["clip_quantile"]),"train")
    device=select_device(str(config["device"]));schedule=DiffusionSchedule.cosine(int(stage["spectral_diffusion"]["diffusion_steps"]),device)
    augmenter=FrequencyForwardDiffusion(statistics,schedule.alpha_bars,r1["soft_mask"],3,1,True,True,device)
    variance,allocation=cross_domain_safe_variance(schedule.alpha_bars,r1["soft_mask"],safety["safe_prob"],True,1,3,5)
    if allocation["budget_error_fraction"]>.02: raise RuntimeError("TEP CDVS mechanism Gate is not GO")
    baseline=json.loads(Path(stage["r1_result"]).read_text(encoding="utf-8"));augmented={};audits={}
    for split,offset in (("train",0),("validation",100),("test",200)):
        sampling=int(seed)+int(stage["spectral_diffusion"]["sampling_seed_offset"])+offset
        augmented[split],diag=augmenter.augment(clean[split],"domain_reliable_safe",sampling,5,
            int(base_config["training"]["batch_size"]),noise_structure="iid",variance_override=variance)
        audits[split]=augmentation_mechanism_metrics(clean[split],augmented[split],views[split]["labels"],stages[split],r1["masks"]["composite"],diag)
        old=baseline["seed_results"][str(seed)]["methods"]["R1"]["augmentation_audit"][split]["expected_total_noise_budget"]
        if abs(audits[split]["expected_total_noise_budget"]-old)>1e-6: raise RuntimeError("TEP CDVS/R1 budgets differ")
    runtime=_runtime(base_config,int(seed));runtime["diagnosis"]={"threshold_band_width":.05,"high_correlation_quantile":.90}
    pretrain=epoch_orders(len(clean["train"]),int(runtime["epochs"]),int(seed)+10000);probe=epoch_orders(len(clean["train"]),int(runtime["probe_epochs"]),int(seed)+20000)
    seed_everything(int(seed));template=build_model(runtime["model"],clean["train"].shape[1],2);initial=copy.deepcopy(template.state_dict())
    old=baseline["seed_results"][str(seed)]["fairness"];fairness={"manifest_sha256":file_sha256(base_config["fixed_views"]["manifest"]),
        "initialization_sha256":_state_hash(initial),"pretrain_order_sha256":sha256_strings([','.join(map(str,o)) for o in pretrain]),
        "probe_order_sha256":sha256_strings([','.join(map(str,o)) for o in probe])}
    for key,value in fairness.items():
        if value!=old[key]:raise RuntimeError(f"TEP CDVS fairness differs: {key}")
    output=Path(config["output"]["tep"])/f"seed_{seed}"/"CDVS";metadata={**old,"method":"CDVS","seed":int(seed),"augmentation":"cross_domain_validated_safety_iid_t5"}
    record=fit_tep_method("CDVS",augmented,audits,views,clean,stages,initial,pretrain,probe,runtime,device,output/"model.pt",metadata)
    payload={"seed":int(seed),"method":record,"fairness":fairness,"allocation":_jsonable(allocation),"test_used_for_fit":False}
    write_json(Path(config["output"]["tep"])/f"seed_{seed}"/"result.json",payload);return payload


def run_stage_1(config:dict[str,Any],data_root:Path):
    _require(Path(config["stage_0"]["output"]),"CDVS_MECHANISM_GO")
    w=_run_three_w_seed(config,data_root,42);t=_run_tep_seed(config,7)
    wr=json.loads(Path("outputs/3w_diffusion_1seed_seed42/result.json").read_text(encoding="utf-8"))["methods"]["FREQUENCY_SELECTIVE_R1"]
    tr=json.loads(Path(config["tep"]["r1_result"]).read_text(encoding="utf-8"))["seed_results"]["7"]["methods"]["R1"]
    comparison={"3W":{"macro_f1_delta":w["method"]["metrics"]["macro_f1"]-wr["metrics"]["macro_f1"],"far_delta":w["method"]["metrics"]["far"]-wr["metrics"]["far"]},
        "TEP":{"macro_f1_delta":t["method"]["test"]["metrics"]["macro_f1"]-tr["test"]["metrics"]["macro_f1"],"far_delta":t["method"]["test"]["metrics"]["far"]-tr["test"]["metrics"]["far"]}}
    passed=all(x["macro_f1_delta"]>=-.03 and x["far_delta"]<=.05 for x in comparison.values())
    result={"status":"CDVS_KILL_TEST_GO" if passed else "CDVS_KILL_TEST_NO_GO","new_training_runs":2,"comparisons":comparison,"three_w":w,"tep":t}
    write_json(Path(config["stage_1"]["output"]),result);return result


def run_stage_2(config:dict[str,Any],data_root:Path):
    kill=_require(Path(config["stage_1"]["output"]),"CDVS_KILL_TEST_GO");w={"42":kill["three_w"]};t={"7":kill["tep"]}
    for seed in (43,44):w[str(seed)]=_run_three_w_seed(config,data_root,seed)
    for seed in (42,2026):t[str(seed)]=_run_tep_seed(config,seed)
    result={"status":"CDVS_STAGE_2_COMPLETE","new_training_runs":4,"total_training_runs":6,"three_w":w,"tep":t}
    write_json(Path(config["stage_2"]["output"]),result);return result


def _report(path:Path,result:dict[str,Any]):
    lines=["# CDVS 机制审计","",f"结论：`{result['status']}`。未训练。",""]
    for key in ("three_w","tep"):
        x=result[key];lines += [f"## {x['dataset']}","",f"- safe_prob：{x['safe_prob_distribution']}",
            f"- unsafe R1 non-critical bins：{x['unsafe_r1_noncritical_bin_count']}",f"- interior safe_prob fraction：{x['interior_safe_prob_fraction']:.2%}",
            f"- 与 DRFD mean |Δt|：{x['mean_absolute_timestep_difference_from_drfd']:.6f}",f"- budget error：{x['allocation']['budget_error_fraction']:.6%}",
            f"- invariants：{x['invariants']}",""]
    path.write_text("\n".join(lines),encoding="utf-8")


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/cross_domain_validated_safety.yaml");p.add_argument("--data-root",type=Path,required=True);p.add_argument("--stage",choices=("0","1","2"),default="0")
    a=p.parse_args();c=yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    r=run_stage_0(c,a.data_root) if a.stage=="0" else run_stage_1(c,a.data_root) if a.stage=="1" else run_stage_2(c,a.data_root)
    print(json.dumps({"status":r["status"],"new_training_runs":r["new_training_runs"]},ensure_ascii=False))

if __name__=="__main__":main()
