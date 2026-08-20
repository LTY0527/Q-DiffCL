from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from diffusion import (DiffusionSchedule, FrequencyForwardDiffusion,
                       fit_spectral_statistics, spectral_noise_variance,
                       unmatched_selective_noise_variance)
from scripts.run_3w_diffusion_1seed import METHODS as THREE_W_METHODS, run as run_three_w_model
from scripts.run_diffusion_quality_retest import _state_hash, epoch_orders
from scripts.run_domain_budget_routing import _load_tep_context, _read
from scripts.run_frequency_selective_r1_3seed import _fit_method, file_sha256
from scripts.run_stage_frequency_diffusion_mvp import _runtime, augmentation_mechanism_metrics
from trainers import build_model
from utils import seed_everything, select_device, write_json


VARIANTS=("UNIFORM_DIFFUSION","HARD_MASK_SELECTIVE","SOFT_MASK_SELECTIVE","SOFT_MASK_WO_BUDGET_MATCH")


def _manifest(path: Path) -> dict[str,Any]:
    return _read(path).get("results",{}) if path.exists() else {}


def _store(path: Path,records: dict[str,Any],key: str,record: dict[str,Any]) -> None:
    records[key]=record; write_json(path,{"results":records,"evaluation_split":"validation","test_read":False})


def _variances(config: dict[str,Any],mask: dict[str,Any],device: str) -> dict[str,np.ndarray]:
    schedule=DiffusionSchedule.cosine(int(config["spectral_diffusion"]["diffusion_steps"]),device)
    soft=torch.as_tensor(mask["soft_mask"],dtype=torch.float32,device=device); hard=torch.as_tensor(mask["hard_mask"],dtype=torch.float32,device=device)
    common=(schedule.alpha_bars,*soft.shape)
    return {
      "HARD_MASK_SELECTIVE":spectral_noise_variance(*common,"selective",3,True,hard,1,5).cpu().numpy(),
      "SOFT_MASK_WO_BUDGET_MATCH":unmatched_selective_noise_variance(schedule.alpha_bars,soft,1,5,True).cpu().numpy(),
    }


def _validate(config: dict[str,Any]) -> None:
    if tuple(config["variants"])!=VARIANTS: raise RuntimeError("paper mechanism variant grid changed")
    if config["audit"]!={"evaluation_split":"validation","test_read":False}: raise RuntimeError("mechanism ablation must remain validation-only")
    final=yaml.safe_load(Path("configs/qdiffcl_final.yaml").read_text(encoding="utf-8"))
    if final["weights"]!={"weight_discriminative":.5,"weight_early":.5,"weight_run_stability":0.0}: raise RuntimeError("FINAL weights changed")
    for dataset,key in (("3W","three_w"),("TEP","tep")):
        mask=_read(config[key]["final_mask"])["criticality"]
        if mask["fit_split"]!="train" or mask["mask_sha256"]!=final["mask_sha256"][dataset]: raise RuntimeError(f"{dataset} train-only FINAL mask changed")


def run_three_w(config: dict[str,Any],data_root: Path,device: str) -> dict[str,Any]:
    stage=config["three_w"]; output=Path(stage["output_dir"]); path=output/"manifest.json"; records=_manifest(path)
    mask=_read(stage["final_mask"])["criticality"]; variances=_variances(config,mask,device); variance_dir=output/"variances"; variance_dir.mkdir(parents=True,exist_ok=True)
    variance_paths={}
    for name,value in variances.items():
        current=variance_dir/f"{name}.npy"; np.save(current,value,allow_pickle=False); variance_paths[name]=current
    for variant in VARIANTS:
        for seed in map(int,stage["seeds"]):
            key=f"{variant}|{seed}"
            if key in records: continue
            base=yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8")); current=copy.deepcopy(base)
            method=THREE_W_METHODS[1] if variant=="UNIFORM_DIFFUSION" else THREE_W_METHODS[2]
            current.update({"seed":seed,"protocol_seed":int(stage["protocol_seed"]),"criticality_source":str(stage["final_mask"]),
                            "methods":[method],"evaluation_split":"validation","device":device,
                            "output_dir":str(output/variant/f"seed_{seed}")})
            if variant in variance_paths: current["selective_variance_override"]=str(variance_paths[variant])
            current["training"]["supcon_batching"]="original"; result=run_three_w_model(current,data_root)
            if result["evaluation_split"]!="validation": raise RuntimeError("3W mechanism ablation read test")
            diagnostic=result["augmentation_diagnostics"][method]
            record={"dataset":"3W","variant":variant,"seed":seed,"evaluation_split":"validation","method":result["methods"][method],
                    "fairness":result["fairness"],"augmentation_audit":diagnostic,"mask_sha256":mask["mask_sha256"],
                    "test_metrics_read":False,"training":"new_training"}
            _store(path,records,key,record)
    return records


def run_tep(config: dict[str,Any],device: str) -> dict[str,Any]:
    stage=config["tep"]; base,views,clean,stages=_load_tep_context({"tep":stage}); mask=_read(stage["final_mask"])["criticality"]
    output=Path(stage["output_dir"]); path=output/"manifest.json"; records=_manifest(path)
    statistics=fit_spectral_statistics(clean["train"],float(config["spectral_diffusion"]["clip_quantile"]),"train")
    schedule=DiffusionSchedule.cosine(int(config["spectral_diffusion"]["diffusion_steps"]),device)
    augmenter=FrequencyForwardDiffusion(statistics,schedule.alpha_bars,np.asarray(mask["soft_mask"],np.float32),3,1,True,True,device)
    variances=_variances(config,mask,device)
    for seed in map(int,stage["seeds"]):
        runtime=_runtime(base,seed); runtime["diagnosis"]={"threshold_band_width":.05,"high_correlation_quantile":.90}
        pretrain=epoch_orders(len(clean["train"]),int(runtime["epochs"]),seed+10000); probe=epoch_orders(len(clean["train"]),int(runtime["probe_epochs"]),seed+20000)
        seed_everything(seed); template=build_model(runtime["model"],clean["train"].shape[1],2); initial=copy.deepcopy(template.state_dict())
        fairness={"manifest_sha256":file_sha256(base["fixed_views"]["manifest"]),"initialization_sha256":_state_hash(initial),
                  "pretrain_order_sha256":hashlib.sha256("\n".join(','.join(map(str,row)) for row in pretrain).encode()).hexdigest(),
                  "probe_order_sha256":hashlib.sha256("\n".join(','.join(map(str,row)) for row in probe).encode()).hexdigest()}
        for variant in VARIANTS:
            key=f"{variant}|{seed}"
            if key in records: continue
            augmented={}; audits={}
            for split,offset in (("train",0),("validation",100)):
                sampling_seed=seed+int(config["spectral_diffusion"]["sampling_seed_offset"])+offset
                if variant=="UNIFORM_DIFFUSION": changed,diag=augmenter.augment(clean[split],"uniform",sampling_seed,batch_size=int(runtime["batch_size"]))
                elif variant=="SOFT_MASK_SELECTIVE": changed,diag=augmenter.augment(clean[split],"selective",sampling_seed,5,int(runtime["batch_size"]))
                else: changed,diag=augmenter.augment(clean[split],"budget_scaled_selective",sampling_seed,5,int(runtime["batch_size"]),variance_override=variances[variant])
                augmented[split]=changed; audits[split]=augmentation_mechanism_metrics(clean[split],changed,views[split]["labels"],stages[split],np.asarray(mask["hard_mask"],bool),diag)
            checkpoint=output/variant/f"seed_{seed}"/"model.pt"; metadata={**fairness,"variant":variant,"evaluation_splits":["validation"],"test_metrics_read":False}
            method=_fit_method(variant,augmented,audits,views,clean,stages,initial,pretrain,probe,runtime,device,checkpoint,metadata,evaluation_splits=("validation",))
            record={"dataset":"TEP","variant":variant,"seed":seed,"evaluation_split":"validation","method":method,"fairness":fairness,
                    "augmentation_audit":audits,"mask_sha256":mask["mask_sha256"],"test_metrics_read":False,"training":"new_training"}
            _store(path,records,key,record)
    return records


def run(config: dict[str,Any],data_root: Path,dataset: str) -> dict[str,Any]:
    _validate(config); device=select_device(str(config["device"])); os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    if dataset in ("3w","both"): run_three_w(config,data_root,device)
    if dataset in ("tep","both"): run_tep(config,device)
    result={"status":"PAPER_MECHANISM_ABLATION_COMPLETE","datasets":dataset,"evaluation_split":"validation","test_read":False}
    write_json(Path(config["output"]["manifest"]),result); return result


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/paper_mechanism_ablation.yaml"); parser.add_argument("--data-root",type=Path,required=True); parser.add_argument("--dataset",choices=("3w","tep","both"),default="both")
    args=parser.parse_args(); config=yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); print(json.dumps(run(config,args.data_root,args.dataset),ensure_ascii=False))


if __name__=="__main__": main()
