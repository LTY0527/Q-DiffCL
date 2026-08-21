from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from frequency.criticality import fault_type
from scripts.run_diffusion_quality_retest import _probabilities, load_fixed_views
from scripts.run_stage_frequency_diffusion_mvp import _runtime
from trainers import build_model
from utils import select_device, write_json


def relative_window(start:int,end:int,onset:int,stride:int)->int:
    if start>=onset:return (start-onset)//stride
    if end<onset:return -int(math.ceil((onset-end)/stride))
    raise ValueError("transition window cannot be aligned")


def load_state(path:str|Path,device:str)->dict[str,torch.Tensor]:
    payload=torch.load(path,map_location=device,weights_only=True)
    return payload.get("model_state_dict",payload)


def bootstrap_interval(values:np.ndarray,repeats:int,seed:int)->tuple[float,float]:
    rng=np.random.default_rng(seed);means=np.asarray([rng.choice(values,len(values),replace=True).mean() for _ in range(repeats)])
    return tuple(map(float,np.quantile(means,[.025,.975])))


def run(config:dict[str,Any])->dict[str,Any]:
    if config["audit"]!={"checkpoint_replay_only":True,"all_fault_runs":True,"outer_test_run":False}:raise RuntimeError("trajectory audit boundary changed")
    base=yaml.safe_load(Path(config["base_config"]).read_text(encoding="utf-8"));views,_=load_fixed_views(base);test=views["test"]
    device=select_device(str(config["device"]));runtime=_runtime(base,int(config["seed"]));all_scores={}
    for method,checkpoint in config["methods"].items():
        model=build_model(runtime["model"],test["clean"].shape[1],2).to(device);model.load_state_dict(load_state(checkpoint,device))
        probability,_=_probabilities(model,test["clean"].astype(np.float32),int(runtime["batch_size"]),device);all_scores[method]=probability[:,1]
    onset=int(base["protocol"]["fault_onset"]["testing"]);stride=int(base["protocol"]["stride"])
    run_uid=np.asarray(test["run_uid"]);fault_selector=np.asarray([fault_type(uid)>0 for uid in run_uid]);fault_runs=sorted(set(run_uid[fault_selector].tolist()))
    relative=np.asarray([relative_window(int(s),int(e),onset,stride) if fault_type(uid)>0 else -999
                         for uid,s,e in zip(run_uid,test["start_sample"],test["end_sample"])])
    pre=int(config["horizon"]["pre_windows"]);post=int(config["horizon"]["post_windows"]);rows=[];raw={"run_uid":run_uid,"relative_window":relative}
    repeats=int(config["bootstrap"]["repeats"]);bootstrap_seed=int(config["bootstrap"]["seed"])
    for method,scores in all_scores.items():
        pre_values=scores[fault_selector&(relative<0)&(relative>=-pre)];mean=float(pre_values.mean());std=max(float(pre_values.std()),1e-8)
        normalized=(scores-mean)/std;raw[f"score_{method}"]=scores;raw[f"normalized_{method}"]=normalized
        for rel in range(-pre,post+1):
            values=[];raw_values=[]
            for uid in fault_runs:
                selected=(run_uid==uid)&(relative==rel)
                if selected.any():values.append(float(normalized[selected].mean()));raw_values.append(float(scores[selected].mean()))
            array=np.asarray(values);low,high=bootstrap_interval(array,repeats,bootstrap_seed+rel+1000*list(all_scores).index(method))
            rows.append({"dataset":"TEP","method":method,"seed":int(config["seed"]),"relative_window":rel,
                "phase":"pre_onset" if rel<0 else "early_fault" if rel<4 else "post_onset",
                "run_count":len(array),"normalized_score_mean":float(array.mean()),"normalized_score_median":float(np.median(array)),
                "normalized_score_ci_low":low,"normalized_score_ci_high":high,"raw_probability_mean":float(np.mean(raw_values)),
                "pre_onset_mean":mean,"pre_onset_std":std})
    raw_path=Path(config["output"]["raw_scores"]);raw_path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(raw_path,**raw)
    output=Path(config["output"]["results_csv"]);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w",newline="",encoding="utf-8-sig") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axis=plt.subplots(figsize=(8.5,4.8))
    for method in config["methods"]:
        selected=[r for r in rows if r["method"]==method];x=np.asarray([r["relative_window"] for r in selected]);y=np.asarray([r["normalized_score_mean"] for r in selected]);lo=np.asarray([r["normalized_score_ci_low"] for r in selected]);hi=np.asarray([r["normalized_score_ci_high"] for r in selected])
        axis.plot(x,y,marker='o',ms=3,label=method);axis.fill_between(x,lo,hi,alpha=.12)
    axis.axvline(-.5,color='black',ls='--',lw=1);axis.axvspan(-pre,-1,color='gray',alpha=.07,label='pre-onset');axis.axvspan(0,3,color='orange',alpha=.07,label='early fault')
    axis.set_xlabel('Window relative to first fully faulty window');axis.set_ylabel('Fault score z-normalized by pre-onset');axis.grid(alpha=.25);axis.legend(ncol=2,fontsize=8);fig.tight_layout()
    figure=Path(config["output"]["figure"]);figure.parent.mkdir(parents=True,exist_ok=True);fig.savefig(figure,dpi=180);plt.close(fig)
    by={(r["method"],r["relative_window"]):r for r in rows};lines=["# Early Fault Score Trajectory","",
      "使用既有 development checkpoint 对 TEP 全部 40 个 fault runs 只读重放。每个方法按其 onset 前窗口分数做 z-normalization；阴影为 run-level bootstrap 95% CI。该证据不是 Paper-final outer evaluation。","",
      "| Method | Last pre-onset z | First early z | Early windows 0–3 mean z | Raw probability 0–3 |","|---|---:|---:|---:|---:|"]
    for method in config["methods"]:
        early=[by[method,i] for i in range(4)];lines.append(f"| {method} | {by[method,-1]['normalized_score_mean']:.3f} | {by[method,0]['normalized_score_mean']:.3f} | {np.mean([r['normalized_score_mean'] for r in early]):.3f} | {np.mean([r['raw_probability_mean'] for r in early]):.3f} |")
    lines += ["","- 聚合覆盖所有可用 fault runs，没有按结果选择 representative cases。","- 对齐点 0 是第一个 fully post-fault window；transition windows 按冻结协议排除。"]
    Path(config["output"]["report"]).write_text("\n".join(lines)+"\n",encoding="utf-8")
    manifest={"status":"PAPER_FAULT_TRAJECTORY_COMPLETE","methods":list(config["methods"]),"fault_runs":len(fault_runs),"rows":len(rows),"outer_test_run":False,"raw_scores":str(raw_path)};write_json(Path(config["output"]["manifest"]),manifest);return manifest


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--config",default="configs/paper_fault_trajectory.yaml");args=parser.parse_args();config=yaml.safe_load(Path(args.config).read_text(encoding="utf-8"));print(json.dumps(run(config),ensure_ascii=False))


if __name__=="__main__":main()
