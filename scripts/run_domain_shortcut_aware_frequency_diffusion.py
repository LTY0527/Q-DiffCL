from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import numpy as np
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score,f1_score,recall_score

from diffusion import (DiffusionSchedule,FrequencyForwardDiffusion,
                       fit_spectral_statistics,matched_domain_shortcut_variance)
from frequency import (build_criticality,build_domain_shortcut_score,fault_stages,
                       fit_frequency_scaler,log_amplitude_phase)
from frequency.criticality import fault_type
from scripts.audit_semantic_diffusion_augmentation import bases
from scripts.run_diffusion_quality_retest import load_fixed_views
from scripts.run_domain_reliable_safe_frequency_diffusion import _load_three_w_train
from scripts.run_stage_frequency_diffusion_mvp import _configure
from utils import select_device,write_json


def _jsonable(v:Any)->Any:
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {str(k):_jsonable(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [_jsonable(x) for x in v]
    return v


def _split(labels:np.ndarray,seed:int)->tuple[np.ndarray,np.ndarray]:
    rng=np.random.default_rng(seed);train=[];test=[]
    for label in np.unique(labels):
        group=np.flatnonzero(labels==label);rng.shuffle(group);cut=max(1,int(round(len(group)*.7)))
        train.extend(group[:cut]);test.extend(group[cut:])
    return np.asarray(train),np.asarray(test)


def _probe(features:np.ndarray,labels:np.ndarray,train:np.ndarray,test:np.ndarray)->dict[str,float]:
    x=np.asarray(features,dtype=np.float64).reshape(len(features),-1);mean=x[train].mean(0);scale=np.where(x[train].std(0)>1e-8,x[train].std(0),1)
    x=(x-mean)/scale;classes=np.unique(labels);centroids=np.stack([x[train][labels[train]==c].mean(0) for c in classes])
    prediction=classes[np.square(x[test,None]-centroids[None]).mean(2).argmin(1)]
    return {"accuracy":float(accuracy_score(labels[test],prediction)),"macro_f1":float(f1_score(labels[test],prediction,average="macro",zero_division=0)),
            "macro_recall":float(recall_score(labels[test],prediction,average="macro",zero_division=0)),"classes":int(len(classes)),"train":int(len(train)),"test":int(len(test))}


def _dataset(name:str,values:np.ndarray,bundle:dict[str,np.ndarray],stages:np.ndarray,domains:np.ndarray,channel_names:list[str],stage:dict[str,Any],device:str):
    log=log_amplitude_phase(values)[0];scaler=fit_frequency_scaler(log,"train");features=scaler.transform(log)
    domain=build_domain_shortcut_score(features,bundle["labels"],stages,domains);r1=build_criticality(features,bundle,stages,stage["criticality"],log)
    spectral=stage["spectral_diffusion"];schedule=DiffusionSchedule.cosine(int(spectral["diffusion_steps"]),device)
    variance,allocation=matched_domain_shortcut_variance(schedule.alpha_bars,r1["soft_mask"],domain["domain_mask"],bool(spectral["preserve_dc"]),3,1,5)
    statistics=fit_spectral_statistics(values,float(spectral["clip_quantile"]),"train")
    augmenter=FrequencyForwardDiffusion(statistics,schedule.alpha_bars,r1["soft_mask"],3,1,True,True,device)
    seed=int(stage["seed"])+int(spectral["sampling_seed_offset"])
    r1_view,_=augmenter.augment(values,"selective",seed,5,256,noise_structure="iid")
    dsfd_view,_=augmenter.augment(values,"domain_reliable_safe",seed,5,256,noise_structure="iid",variance_override=variance)
    r1_features=log_amplitude_phase(r1_view)[0];dsfd_features=log_amplitude_phase(dsfd_view)[0]
    domain_labels=np.asarray(domains);fault_labels=np.asarray(bundle["labels"])
    identity_train,identity_test=_split(domain_labels,int(stage["seed"]));fault_train,fault_test=_split(fault_labels,int(stage["seed"])+1)
    probes={"identity":{"R1":_probe(r1_features,domain_labels,identity_train,identity_test),"DSFD":_probe(dsfd_features,domain_labels,identity_train,identity_test)},
            "fault":{"R1":_probe(r1_features,fault_labels,fault_train,fault_test),"DSFD":_probe(dsfd_features,fault_labels,fault_train,fault_test)}}
    score=np.asarray(domain["domain_score"]);mask=np.asarray(domain["domain_mask"]);fault=np.asarray(r1["soft_mask"]);high_fault=fault>=.5;high_domain=mask>=.5
    quadrants={"fault_high_domain_low":int(np.sum(high_fault&~high_domain)),"fault_high_domain_high":int(np.sum(high_fault&high_domain)),
               "fault_low_domain_high":int(np.sum(~high_fault&high_domain)),"fault_low_domain_low":int(np.sum(~high_fault&~high_domain))}
    flat=np.argsort(score.reshape(-1))[::-1][:20];frequencies=score.shape[1]
    top=[{"channel":int(i//frequencies),"channel_name":channel_names[int(i//frequencies)],"frequency_bin":int(i%frequencies),
          "domain_score":float(score.reshape(-1)[i]),"domain_mask":float(mask.reshape(-1)[i]),"fault_mask":float(fault.reshape(-1)[i])} for i in flat]
    invariants={key:allocation[key] for key in ("protected_variance_not_increased","budget_adjustment_only_low_fault","maximum_variance_respected","minimum_timestep_respected","maximum_timestep_respected","finite")}
    audit={"dataset":name,"domain_score_distribution":{q:float(v) for q,v in zip(("min","p25","median","p75","max"),np.quantile(score,[0,.25,.5,.75,1]))},
           "top_domain_bins":top,"fault_domain_spearman":float(spearmanr(fault.reshape(-1),score.reshape(-1)).statistic),"quadrants":quadrants,
           "changed_bins":allocation["changed_bin_count"],"mean_absolute_timestep_change_from_r1":float(np.mean(np.abs(np.asarray(allocation["dsfd_timestep"])-(1+(1-fault)*4)))),
           "budget_error":allocation["budget_error_fraction"],"invariants":invariants,"domain":_jsonable(domain),"r1_soft_mask":fault.tolist(),"allocation":_jsonable(allocation)}
    return audit,probes


def run_stage_a(config:dict[str,Any],data_root:Path):
    device=select_device(str(config["device"]));sw=config["three_w"];values,bundle,stages,wells,_=_load_three_w_train(sw,data_root)
    base=yaml.safe_load(Path(sw["base_config"]).read_text(encoding="utf-8"));grouped=Path(base["grouped_output"]);idx=int(base["canonical_split_index"])
    pre=json.loads((grouped/f"split_{idx:02d}"/"preprocessor.json").read_text(encoding="utf-8"));names=list(pre["retained_features"])
    aw,pw=_dataset("3W",values,bundle,stages,wells,names,sw,device)
    st=config["tep"];bc=yaml.safe_load(Path(st["base_config"]).read_text(encoding="utf-8"));_configure(bc);views,_=load_fixed_views(bc);clean=bases(views)
    tep_stages=fault_stages(views["train"],bc);runs=np.asarray(views["train"]["run_uid"],dtype=object)
    at,pt=_dataset("TEP",clean["train"],views["train"],tep_stages,runs,[f"xmeas_{i+1}" for i in range(clean["train"].shape[1])],st,device)
    mechanism=all(all(x["invariants"].values()) and x["budget_error"]<=.02 for x in (aw,at))
    id_r1=pw["identity"]["R1"]["accuracy"];id_dsfd=pw["identity"]["DSFD"]["accuracy"];relative=(id_r1-id_dsfd)/max(id_r1,1e-12)
    fault_delta=pw["fault"]["DSFD"]["macro_f1"]-pw["fault"]["R1"]["macro_f1"]
    if not mechanism:status="DSFD_MECHANISM_NO_GO"
    elif relative<.05:status="DSFD_SHORTCUT_GATE_NO_GO"
    elif fault_delta<-.01:status="DSFD_SEMANTIC_PRESERVATION_NO_GO"
    else:status="DSFD_STAGE_A_GO"
    result={"status":status,"new_training_runs":0,"three_w":{"audit":aw,"probes":pw},"tep":{"audit":at,"probes":pt},
            "shortcut_gate":{"well_id_relative_accuracy_drop":relative,"fault_macro_f1_delta":fault_delta,"passed":status=="DSFD_STAGE_A_GO"},
            "test_or_validation_used":False}
    write_json(Path(config["stage_a"]["output"]),result);_outputs(config,result);return result


def _outputs(config,result):
    rows=[];quadrants=[]
    for key in ("three_w","tep"):
        x=result[key]["audit"];score=np.asarray(x["domain"]["domain_score"]);mask=np.asarray(x["domain"]["domain_mask"]);fault=np.asarray(x["r1_soft_mask"])
        for c in range(score.shape[0]):
            for f in range(score.shape[1]):
                rows.append({"dataset":x["dataset"],"channel":c,"frequency_bin":f,"domain_score":score[c,f],"domain_mask":mask[c,f],"fault_mask":fault[c,f]})
                quadrants.append({**rows[-1],"quadrant":("fault_high_" if fault[c,f]>=.5 else "fault_low_")+("domain_high" if mask[c,f]>=.5 else "domain_low")})
    for path,items in ((config["stage_a"]["domain_csv"],rows),(config["stage_a"]["quadrant_csv"],quadrants)):
        with Path(path).open("w",encoding="utf-8-sig",newline="") as h:writer=csv.DictWriter(h,fieldnames=list(items[0]));writer.writeheader();writer.writerows(items)
    probe_rows=[]
    for key in ("three_w","tep"):
        for task,methods in result[key]["probes"].items():
            for method,metrics in methods.items():probe_rows.append({"dataset":result[key]["audit"]["dataset"],"task":task,"method":method,**metrics})
    with Path(config["stage_a"]["probe_csv"]).open("w",encoding="utf-8-sig",newline="") as h:writer=csv.DictWriter(h,fieldnames=list(probe_rows[0]));writer.writeheader();writer.writerows(probe_rows)
    a=result["three_w"]["audit"];Path(config["stage_a"]["audit_report"]).write_text("# DSFD Domain Audit\n\n"+f"结论：`{result['status']}`。\n\n"
        f"3W DomainScore={a['domain_score_distribution']}，Fault/Domain Spearman={a['fault_domain_spearman']:.6f}，四象限={a['quadrants']}，budget error={a['budget_error']:.6%}。\n\n"
        f"TEP DomainScore={result['tep']['audit']['domain_score_distribution']}，与 R1 mean |Δt|={result['tep']['audit']['mean_absolute_timestep_change_from_r1']:.6f}。\n",encoding="utf-8")
    g=result["shortcut_gate"];Path(config["stage_a"]["probe_report"]).write_text("# DSFD Shortcut Suppression Probe\n\n"+f"Gate：`{result['status']}`。\n\n"
        f"3W WELL-ID relative accuracy drop={g['well_id_relative_accuracy_drop']:.2%}；Fault Macro-F1 delta={g['fault_macro_f1_delta']:+.6f}。\n\n"
        f"3W probes={result['three_w']['probes']}\n\nTEP auxiliary probes={result['tep']['probes']}\n",encoding="utf-8")


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/domain_shortcut_aware_frequency_diffusion.yaml");p.add_argument("--data-root",type=Path,required=True)
    a=p.parse_args();r=run_stage_a(yaml.safe_load(Path(a.config).read_text(encoding="utf-8")),a.data_root);print(json.dumps({"status":r["status"],"shortcut_gate":r["shortcut_gate"],"new_training_runs":0},ensure_ascii=False))

if __name__=="__main__":main()
