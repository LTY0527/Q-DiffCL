from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import numpy as np
import torch
import yaml

from diffusion import DiffusionSchedule,FrequencyForwardDiffusion,fit_spectral_statistics
from frequency import build_criticality,fault_stages,fit_frequency_scaler,log_amplitude_phase,mask_jaccard
from scripts.audit_semantic_diffusion_augmentation import bases
from scripts.run_3w_diffusion_1seed import METHODS as THREE_W_METHODS,run as run_three_w_model
from scripts.run_diffusion_quality_retest import _state_hash,epoch_orders,load_fixed_views
from scripts.run_domain_reliable_safe_frequency_diffusion import _load_three_w_train
from scripts.run_frequency_selective_r1_3seed import (_fit_method as fit_tep_method,file_sha256,sha256_strings)
from scripts.run_stage_frequency_diffusion_mvp import _configure,_runtime,augmentation_mechanism_metrics
from trainers import build_model
from utils import seed_everything,select_device,write_json


TRAIN_VARIANTS=("W/O_D","W/O_E","W/O_S","D_ONLY","E_ONLY","S_ONLY")
STAGE_A_VARIANTS=("W/O_D","W/O_E","W/O_S")
STAGE_B_VARIANTS=("D_ONLY","E_ONLY","S_ONLY")


def variant_settings(config:dict[str,Any],name:str)->dict[str,Any]:
    value={**config["criticality_base"],**config["variants"][name]}
    if abs(sum(value[key] for key in ("weight_discriminative","weight_early","weight_run_stability"))-1)>1e-12:
        raise ValueError(f"DES weights do not sum to one: {name}")
    return value


def _hash(value:np.ndarray)->str:return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _ready(record:dict[str,Any],name:str)->dict[str,Any]:
    return {"variant":name,"fit_split":"train","weights":record["component_weights"],
        "discriminative":record["discriminative"].tolist(),"early":record["early"].tolist(),"stability":record["stability"].tolist(),
        "composite":record["composite"].tolist(),"hard_mask":record["masks"]["composite"].astype(int).tolist(),"soft_mask":record["soft_mask"].tolist(),
        "mask_sha256":_hash(record["masks"]["composite"]),"soft_mask_sha256":_hash(record["soft_mask"]),"test_or_validation_used":False}


def _build_three_w_masks(config:dict[str,Any],data_root:Path):
    stage=config["three_w"];values,bundle,stages,_,_=_load_three_w_train(stage,data_root);log=log_amplitude_phase(values)[0];features=fit_frequency_scaler(log,"train").transform(log)
    return {name:_ready(build_criticality(features,bundle,stages,variant_settings(config,name),log),name) for name in config["variants"]}


def _tep_context(config:dict[str,Any]):
    stage=config["tep"];base_config=yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8"));_configure(base_config)
    views,_=load_fixed_views(base_config);clean=bases(views);stages={s:fault_stages(views[s],base_config) for s in views}
    log=log_amplitude_phase(clean["train"])[0];features=fit_frequency_scaler(log,"train").transform(log)
    masks={name:_ready(build_criticality(features,views["train"],stages["train"],variant_settings(config,name),log),name) for name in config["variants"]}
    return base_config,views,clean,stages,masks


def build_masks(config:dict[str,Any],data_root:Path):
    path=Path(config["output"]["mask_dir"]);path.mkdir(parents=True,exist_ok=True)
    three=_build_three_w_masks(config,data_root);base,views,clean,stages,tep=_tep_context(config)
    audits={}
    for dataset,masks,shape in (("3W",three,None),("TEP",tep,None)):
        full=np.asarray(masks["FULL_DES"]["hard_mask"],bool);audits[dataset]={}
        for name,item in masks.items():
            hard=np.asarray(item["hard_mask"],bool);soft=np.asarray(item["soft_mask"],np.float32)
            schedule=DiffusionSchedule.cosine(50,"cpu")
            # Mechanism variance depends only on schedule and mask; use the public helper through a minimal augmenter-free calculation.
            from diffusion.frequency_selective import spectral_noise_variance
            variance=spectral_noise_variance(schedule.alpha_bars,*soft.shape,"selective",3,True,torch.as_tensor(soft),1,5)
            uniform_variance=spectral_noise_variance(schedule.alpha_bars,*soft.shape,"uniform",3,True)
            timestep=1+(1-soft)*4;critical=soft>=.5
            audits[dataset][name]={"weights":item["weights"],"mask_sha256":item["mask_sha256"],"hard_mask_jaccard_full":mask_jaccard(hard,full),
                "changed_bins_from_full":int(np.sum(hard!=full)),"timestep":timestep.tolist(),"critical_noise_budget":float(variance[torch.as_tensor(critical)].mean()),
                "noncritical_noise_budget":float(variance[torch.as_tensor(~critical)].mean()),"total_budget_error":float(abs(variance.mean()-uniform_variance.mean())),
                "largest_composite_changes":[{"channel":int(i//hard.shape[1]),"frequency_bin":int(i%hard.shape[1]),"absolute_change":float(v)}
                    for i,v in sorted(enumerate(np.abs(np.asarray(item["composite"])-np.asarray(masks["FULL_DES"]["composite"])).reshape(-1)),key=lambda x:x[1],reverse=True)[:20]]}
            write_json(path/f"{dataset.lower()}_{name.replace('/','_')}.json",{"criticality":item})
    write_json(Path(config["output"]["mask_audit"]),audits)
    return three,(base,views,clean,stages,tep),audits


def _selected(mode:str):return STAGE_A_VARIANTS if mode=="stage_a" else STAGE_B_VARIANTS if mode=="stage_b" else TRAIN_VARIANTS


def run_three_w(config:dict[str,Any],data_root:Path,masks:dict[str,Any],mode:str,seeds:list[int]|None=None):
    stage=config["three_w"];base=yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8"));completed={}
    manifest_path=Path(stage["output_dir"])/"manifest.json"
    if manifest_path.exists():completed=json.loads(manifest_path.read_text(encoding="utf-8")).get("results",{})
    for variant in _selected(mode):
        source=Path(config["output"]["mask_dir"])/f"3w_{variant.replace('/','_')}.json"
        for seed in map(int,seeds or stage["seeds"]):
            key=f"{variant}|{seed}";current=copy.deepcopy(base);current["seed"]=seed;current["protocol_seed"]=42
            if key in completed and Path(completed[key]["result_path"]).exists():
                continue
            current["criticality_source"]=str(source);current["methods"]=[THREE_W_METHODS[2]];current["training"]["supcon_batching"]="original"
            current["output_dir"]=str(Path(stage["output_dir"])/variant.replace('/','_')/f"seed_{seed}")
            result=run_three_w_model(current,data_root);completed[key]={"variant":variant,"seed":seed,"result_path":str(Path(current["output_dir"])/"result.json"),
                "method":result["methods"][THREE_W_METHODS[2]],"fairness":result["fairness"],"mask_sha256":masks[variant]["mask_sha256"]}
            write_json(manifest_path,{"results":completed})
    return completed


def run_tep(config:dict[str,Any],context,masks:dict[str,Any],mode:str,seeds:list[int]|None=None):
    stage=config["tep"];base_config,views,clean,stages,_=context;baseline=json.loads(Path(stage["baseline_result"]).read_text(encoding="utf-8"));device=select_device(str(config["device"]))
    output=Path(stage["output_dir"]);manifest_path=output/"manifest.json";completed=json.loads(manifest_path.read_text(encoding="utf-8")).get("results",{}) if manifest_path.exists() else {}
    statistics=fit_spectral_statistics(clean["train"],float(stage["spectral_diffusion"]["clip_quantile"]),"train");schedule=DiffusionSchedule.cosine(50,device)
    for variant in _selected(mode):
        mask=np.asarray(masks[variant]["soft_mask"],np.float32);augmenter=FrequencyForwardDiffusion(statistics,schedule.alpha_bars,mask,3,1,True,True,device)
        for seed in map(int,seeds or stage["seeds"]):
            if f"{variant}|{seed}" in completed:
                continue
            augmented={};audits={}
            for split,offset in (("train",0),("validation",100),("test",200)):
                sampling=seed+int(stage["spectral_diffusion"]["sampling_seed_offset"])+offset
                augmented[split],diag=augmenter.augment(clean[split],"selective",sampling,5,int(base_config["training"]["batch_size"]),noise_structure="iid")
                audits[split]=augmentation_mechanism_metrics(clean[split],augmented[split],views[split]["labels"],stages[split],np.asarray(masks[variant]["hard_mask"],bool),diag)
                old=baseline["seed_results"][str(seed)]["methods"]["R1"]["augmentation_audit"][split]["expected_total_noise_budget"]
                if abs(audits[split]["expected_total_noise_budget"]-old)>1e-6:raise RuntimeError("DES/R1 budget mismatch")
            runtime=_runtime(base_config,seed);runtime["diagnosis"]={"threshold_band_width":.05,"high_correlation_quantile":.90}
            pre=epoch_orders(len(clean["train"]),int(runtime["epochs"]),seed+10000);probe=epoch_orders(len(clean["train"]),int(runtime["probe_epochs"]),seed+20000)
            seed_everything(seed);template=build_model(runtime["model"],clean["train"].shape[1],2);initial=copy.deepcopy(template.state_dict());old=baseline["seed_results"][str(seed)]["fairness"]
            fairness={"manifest_sha256":file_sha256(base_config["fixed_views"]["manifest"]),"initialization_sha256":_state_hash(initial),
                "pretrain_order_sha256":sha256_strings([','.join(map(str,o)) for o in pre]),"probe_order_sha256":sha256_strings([','.join(map(str,o)) for o in probe])}
            for key,value in fairness.items():
                if value!=old[key]:raise RuntimeError(f"TEP DES fairness mismatch {key}")
            ckpt=output/variant.replace('/','_')/f"seed_{seed}"/"model.pt";metadata={**old,"method":variant,"seed":seed,"mask_sha256":masks[variant]["mask_sha256"]}
            record=fit_tep_method(variant,augmented,audits,views,clean,stages,initial,pre,probe,runtime,device,ckpt,metadata)
            completed[f"{variant}|{seed}"]={"variant":variant,"seed":seed,"method":record,"fairness":fairness,"mask_sha256":masks[variant]["mask_sha256"]}
            write_json(manifest_path,{"results":completed})
    return completed


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/r1_des_ablation.yaml");p.add_argument("--data-root",type=Path,required=True);p.add_argument("--dataset",choices=("3w","tep","both"),default="both");p.add_argument("--stage",choices=("stage_a","stage_b","all"),default="all");p.add_argument("--seeds",type=int,nargs="+")
    a=p.parse_args();config=yaml.safe_load(Path(a.config).read_text(encoding="utf-8"));three,tep_context,audit=build_masks(config,a.data_root);results={}
    if a.dataset in ("3w","both"):results["three_w"]=run_three_w(config,a.data_root,three,a.stage,a.seeds)
    if a.dataset in ("tep","both"):results["tep"]=run_tep(config,tep_context,tep_context[-1],a.stage,a.seeds)
    write_json(Path(config["output"]["manifest"]),{"stage":a.stage,"datasets":list(results),"new_training_runs":sum(len(v) for v in results.values()),"results":results})
    print(json.dumps({"stage":a.stage,"datasets":list(results),"completed_records":{k:len(v) for k,v in results.items()}},ensure_ascii=False))

if __name__=="__main__":main()
