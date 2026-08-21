from __future__ import annotations

import argparse
import copy
import csv
import json
import platform
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml

from augmentations import domain_budget_route
from diffusion import DiffusionSchedule, FrequencyForwardDiffusion, fit_spectral_statistics
from baselines.external_augmentations import traditional_view
from scripts.run_3w_diffusion_1seed import METHODS as THREE_W_METHODS, run as run_three_w
from scripts.run_external_baselines import _prepare_three_w, _prepare_tep
from scripts.run_3w_final_primary_grouped import build_model as build_three_w_model
from trainers import build_model
from utils import environment_metadata, select_device, write_json


METHODS=("NO_AUG","UNIFORM_DIFFUSION","STRONGEST_TRADITIONAL","FRERA","FINAL_QDIFFCL","DCBR")


def read(path: str|Path)->dict[str,Any]: return json.loads(Path(path).read_text(encoding="utf-8"))


def _strongest(manifest:dict[str,Any],dataset:str,seeds:list[int])->str:
    means={}
    for method in ("JITTER","SCALING","JITTER_SCALING"):
        values=[]
        for seed in seeds:
            item=manifest["results"][f"{dataset}|{method}|{seed}"]["record"]
            metrics=item["metrics"] if dataset=="3W" else item["test"]["metrics"]
            values.append(float(metrics["macro_f1"]))
        means[method]=float(np.mean(values))
    return max(means,key=means.get)


def _sync(device:str)->None:
    if device.startswith("cuda"): torch.cuda.synchronize()


def benchmark(function:Callable[[],Any],repeats:int,device:str)->tuple[float,float]:
    function();_sync(device);values=[]
    for _ in range(repeats):
        started=time.perf_counter();function();_sync(device);values.append((time.perf_counter()-started)*1000)
    return float(np.mean(values)),float(np.std(values,ddof=1)) if repeats>1 else 0.0


def _train_missing_three_w(config:dict[str,Any],data_root:Path)->dict[str,dict[str,Any]]:
    stage=config["three_w"];seed=int(stage["canonical_seed"]);result={}
    for label,method in (("NO_AUG",THREE_W_METHODS[0]),("UNIFORM_DIFFUSION",THREE_W_METHODS[1]),("FINAL_QDIFFCL",THREE_W_METHODS[2])):
        base=yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8"));current=copy.deepcopy(base)
        current.update({"seed":seed,"protocol_seed":int(stage["protocol_seed"]),"criticality_source":stage["final_mask"],
            "methods":[method],"evaluation_split":"validation","output_dir":str(Path(stage["output_dir"])/label)})
        current["training"]["supcon_batching"]="original";payload=run_three_w(current,data_root);result[label]=payload["methods"][method]
    return result


def _training_rows(config:dict[str,Any],three_w_new:dict[str,dict[str,Any]])->list[dict[str,Any]]:
    rows=[];external=read(config["three_w"]["external_manifest"]); three_best=_strongest(external,"3W",[42,43,44])
    seed=int(config["three_w"]["canonical_seed"])
    for method in METHODS:
        source_method=three_best if method=="STRONGEST_TRADITIONAL" else method
        if method in three_w_new: record=three_w_new[method]; source="new canonical timing"
        elif method=="DCBR": record=three_w_new["FINAL_QDIFFCL"];source="rho=1 exact FINAL equivalent"
        else: record=external["results"][f"3W|{source_method}|{seed}"]["record"];source=config["three_w"]["external_manifest"]
        rows.append({"dataset":"3W","method":method,"resolved_method":source_method,"canonical_seed":seed,
            "training_seconds":record.get("training_seconds"),"peak_gpu_mib":record.get("peak_gpu_mib"),"timing_source":source})
    external=read(config["tep"]["external_manifest"]);tep_best=_strongest(external,"TEP",[7,42,2026]);seed=int(config["tep"]["canonical_seed"])
    for method in METHODS:
        source_method=tep_best if method=="STRONGEST_TRADITIONAL" else method
        if method=="FINAL_QDIFFCL": record=read(config["tep"]["final_metrics"]);source=config["tep"]["final_metrics"]
        elif method=="DCBR": record=read(config["tep"]["dcbr_metrics"]);source=config["tep"]["dcbr_metrics"]
        else: record=external["results"][f"TEP|{source_method}|{seed}"]["record"];source=config["tep"]["external_manifest"]
        rows.append({"dataset":"TEP","method":method,"resolved_method":source_method,"canonical_seed":seed,
            "training_seconds":record.get("training_seconds"),"peak_gpu_mib":record.get("peak_gpu_mib"),"timing_source":source})
    return rows


def _bench_dataset(config:dict[str,Any],dataset:str,data_root:Path,device:str,rows:list[dict[str,Any]])->None:
    count=int(config["benchmark_windows"]);repeats=int(config["benchmark_repeats"]);spectral=config["spectral_diffusion"]
    if dataset=="3W":
        context=_prepare_three_w(config["three_w"],data_root,int(config["three_w"]["canonical_seed"]),device)
        clean=context["train_x"][:count];channels=clean.shape[1];model=build_three_w_model(context["base"]["training"]["model"],channels,device)
        checkpoint=Path(config["three_w"]["output_dir"])/"FINAL_QDIFFCL"/f"{THREE_W_METHODS[2]}_model.pt";model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
        ids=np.asarray([f"bench:{i}" for i in range(len(clean))]);traditional_settings={"jitter_std":.03,"scaling_std":.05};rho=1.0
    else:
        context=_prepare_tep(config["tep"],int(config["tep"]["canonical_seed"]),device);clean=context["clean"]["train"][:count];channels=clean.shape[1]
        model=build_model(context["runtime"]["model"],channels,2).to(device);checkpoint=torch.load("outputs/r1_des_weight_search/tep/DE_50_50/seed_7/model.pt",map_location=device,weights_only=True)
        model.load_state_dict(checkpoint.get("model_state_dict",checkpoint))
        ids=context["views"]["train"]["window_id"][:len(clean)];traditional_settings={"jitter_std":.03,"scaling_std":.05};rho=.75
    model.eval();batch=torch.from_numpy(clean).float().to(device)
    infer_mean,infer_std=benchmark(lambda:model(batch),repeats,device)
    mask=read(config["three_w" if dataset=="3W" else "tep"]["final_mask"])["criticality"]
    statistics=fit_spectral_statistics(clean,float(spectral["clip_quantile"]),"train")
    schedule=DiffusionSchedule.cosine(int(spectral["diffusion_steps"]),device);augmenter=FrequencyForwardDiffusion(statistics,schedule.alpha_bars,np.asarray(mask["soft_mask"],np.float32),3,1,True,True,device)
    seed=int(config["three_w" if dataset=="3W" else "tep"]["canonical_seed"])+int(spectral["sampling_seed_offset"])
    functions={
      "NO_AUG":lambda:clean.copy(),
      "UNIFORM_DIFFUSION":lambda:augmenter.augment(clean,"uniform",seed,batch_size=min(256,len(clean)))[0],
      "STRONGEST_TRADITIONAL":lambda:traditional_view(clean,ids,config["three_w" if dataset=="3W" else "tep"].get("strongest_traditional","SCALING"),seed,traditional_settings["jitter_std"],traditional_settings["scaling_std"]),
      "FINAL_QDIFFCL":lambda:augmenter.augment(clean,"selective",seed,5,min(256,len(clean)))[0],
    }
    if dataset=="3W":
        functions["DCBR"]=functions["FINAL_QDIFFCL"]
    else:
        def dcbr_benchmark():
            diffused=functions["FINAL_QDIFFCL"]()
            return domain_budget_route(clean,diffused,ids,rho,.05,seed)[0]
        functions["DCBR"]=dcbr_benchmark
    for row in (r for r in rows if r["dataset"]==dataset):
        model_params=sum(p.numel() for p in model.parameters());row.update({"total_parameters":model_params,
            "trainable_parameters":model_params,"augmentation_parameters":66 if row["method"]=="FRERA" else 0,
            "inference_additional_parameters":0,"inference_ms_per_1024_mean":infer_mean,
            "inference_ms_per_1024_std":infer_std,"benchmark_windows":len(clean),"benchmark_repeats":repeats,
            "macs_flops":"N/A (no stable counter in frozen environment)","peak_host_memory_mib":"N/A"})
        if row["method"]=="FRERA": row.update({"augmentation_ms_per_1024_mean":None,"augmentation_ms_per_1024_std":None})
        else:
            mean,std=benchmark(functions[row["method"]],repeats,device);row.update({"augmentation_ms_per_1024_mean":mean,"augmentation_ms_per_1024_std":std})


def report(config:dict[str,Any],rows:list[dict[str,Any]],metadata:dict[str,Any])->None:
    output=Path(config["output"]["results_csv"]);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w",newline="",encoding="utf-8-sig") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=["# Efficiency / Practicality","",f"环境：{metadata['gpu']}；PyTorch {metadata['pytorch']}；CUDA {metadata['cuda']}；Python {platform.python_version()}。所有 benchmark 在同一进程、同一硬件上重复 `{config['benchmark_repeats']}` 次。","",
      "| Dataset | Method | Training s | Peak GPU MiB | Aug. ms / 1024 | Inference ms / 1024 | Total params | Aug. params | Inference add. params |","|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(v:Any,digits:int=2)->str:return "N/A" if v is None else f"{float(v):.{digits}f}"
    for row in rows:
        aug="N/A" if row["augmentation_ms_per_1024_mean"] is None else f"{row['augmentation_ms_per_1024_mean']:.2f} ± {row['augmentation_ms_per_1024_std']:.2f}"
        inf=f"{row['inference_ms_per_1024_mean']:.2f} ± {row['inference_ms_per_1024_std']:.2f}"
        lines.append(f"| {row['dataset']} | {row['method']} | {fmt(row['training_seconds'],1)} | {fmt(row['peak_gpu_mib'],1)} | {aug} | {inf} | {row['total_parameters']} | {row['augmentation_parameters']} | 0 |")
    lines += ["","- 3W DCBR `rho=1` 与 FINAL 逐元素等价，训练与 augmentation timing 直接复用 FINAL。",
      "- FRERA 的训练时间/显存来自已有 shared-backbone canonical run；其 augmenter checkpoint 未单独保存，因此不事后伪造 augmentation-only timing。",
      "- 推理阶段所有方法只保留同一 frozen TCN + Linear Probe；DCBR/FINAL/传统增强均新增 0 个推理参数。",
      "- MACs/FLOPs 未报告：冻结环境没有稳定计数器，避免引入新依赖后得到不可比数字。"]
    Path(config["output"]["report"]).write_text("\n".join(lines)+"\n",encoding="utf-8")


def run(config:dict[str,Any],data_root:Path)->dict[str,Any]:
    if config["audit"]["outer_test_run"]:raise RuntimeError("efficiency runner must not run outer evaluation")
    device=select_device(str(config["device"]));three_w_new=_train_missing_three_w(config,data_root);rows=_training_rows(config,three_w_new)
    external=read(config["three_w"]["external_manifest"]);config["three_w"]["strongest_traditional"]=_strongest(external,"3W",[42,43,44]);config["tep"]["strongest_traditional"]=_strongest(external,"TEP",[7,42,2026])
    _bench_dataset(config,"3W",data_root,device,rows);_bench_dataset(config,"TEP",data_root,device,rows)
    metadata=environment_metadata();write_json(Path(config["output"]["manifest"]),{"rows":rows,"environment":metadata,"outer_test_run":False});report(config,rows,metadata)
    return {"status":"PAPER_EFFICIENCY_COMPLETE","rows":len(rows),"outer_test_run":False}


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--config",default="configs/paper_efficiency.yaml");parser.add_argument("--data-root",type=Path,required=True)
    args=parser.parse_args();config=yaml.safe_load(Path(args.config).read_text(encoding="utf-8"));print(json.dumps(run(config,args.data_root),ensure_ascii=False))


if __name__=="__main__":main()
