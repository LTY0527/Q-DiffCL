from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from diffusion import DiffusionSchedule, FrequencyForwardDiffusion, fit_spectral_statistics, spectral_noise_variance
from frequency import fault_stages
from scripts.run_3w_diffusion_1seed import METHODS as THREE_W_METHODS, run as run_three_w
from scripts.run_diffusion_quality_retest import _state_hash, epoch_orders, load_fixed_views
from scripts.run_frequency_selective_r1_3seed import _fit_method, file_sha256
from scripts.run_paper_contrastive_ablation import _metric_record
from scripts.run_stage_frequency_diffusion_mvp import _runtime, augmentation_mechanism_metrics
from trainers import build_model
from utils import seed_everything, select_device, write_json


def read(path: str | Path) -> dict[str, Any]: return json.loads(Path(path).read_text(encoding="utf-8"))


def top_mask(score: np.ndarray, ratio: float) -> np.ndarray:
    count = max(1, int(round(score.size * ratio))); flat = score.ravel(); selected = np.argpartition(flat, -count)[-count:]
    result = np.zeros(score.size, bool); result[selected] = True; return result.reshape(score.shape)


def ratio_mask(source: dict[str, Any], ratio: float) -> dict[str, Any]:
    payload = copy.deepcopy(source); current = payload["criticality"]; composite = np.asarray(current["composite"], float)
    hard = top_mask(composite, ratio); threshold = float(np.min(composite[hard]))
    scale = max(float(np.quantile(composite, .75) - np.quantile(composite, .25)), 1e-8)
    soft = (1 / (1 + np.exp(np.clip(-(composite - threshold) / scale, -30, 30)))).astype(np.float32)
    current["hard_mask"] = hard.astype(int).tolist(); current["soft_mask"] = soft.tolist()
    current["critical_ratio"] = ratio; current["sensitivity_only"] = True
    current["mask_sha256"] = hashlib.sha256(np.ascontiguousarray(hard).tobytes()).hexdigest()
    current["soft_mask_sha256"] = hashlib.sha256(np.ascontiguousarray(soft).tobytes()).hexdigest()
    return payload


def budget_audit(mask: dict[str, Any]) -> dict[str, Any]:
    soft = torch.as_tensor(mask["soft_mask"], dtype=torch.float32); schedule = DiffusionSchedule.cosine(50, "cpu")
    selective = spectral_noise_variance(schedule.alpha_bars, *soft.shape, "selective", 3, True, soft, 1, 5)
    uniform = spectral_noise_variance(schedule.alpha_bars, *soft.shape, "uniform", 3, True, soft, 1, 5)
    hard = np.asarray(mask["hard_mask"], bool)
    return {"actual_spectral_noise_budget": float(selective.mean()),
            "uniform_spectral_noise_budget": float(uniform.mean()),
            "matched_budget_error": abs(float(selective.mean()-uniform.mean())),
            "selected_bins": int(hard.sum()), "total_bins": int(hard.size),
            "soft_weight_mean": float(soft.mean()), "soft_weight_std": float(soft.std()),
            "soft_weight_min": float(soft.min()), "soft_weight_max": float(soft.max())}


def store(path: Path, records: dict[str, Any], key: str, item: dict[str, Any]) -> None:
    records[key] = item; write_json(path, {"results": records, "outer_test_run": False,
        "ratio_selected_from_test": False})


def ensure_masks(config: dict[str, Any]) -> dict[tuple[str, float], Path]:
    result = {}
    for dataset, section in (("3W", "three_w"), ("TEP", "tep")):
        source = read(config[section]["final_mask"])
        for ratio in map(float, config["ratios"]):
            path = Path(config[section]["output_dir"]) / "masks" / f"ratio_{int(ratio*100):02d}.json"
            expected = source if ratio == .3 else ratio_mask(source, ratio)
            if ratio == .3: expected = copy.deepcopy(source); expected["criticality"]["critical_ratio"] = .3
            if path.exists() and read(path) != expected: raise RuntimeError(f"ratio mask changed: {path}")
            if not path.exists(): write_json(path, expected)
            result[dataset, ratio] = path
    return result


def reuse_frozen(config: dict[str, Any], records: dict[str, Any], path: Path, masks: dict[tuple[str,float],Path]) -> None:
    for dataset, section in (("3W","three_w"),("TEP","tep")):
        manifest = read(config[section]["final_manifest"])["results"]; audit = budget_audit(read(masks[dataset,.3])["criticality"])
        for seed in map(int, config[section]["seeds"]):
            key = f"{dataset}|0.30|{seed}"
            if key in records: continue
            item = manifest[f"FINAL_QDIFFCL|{seed}"]
            store(path, records, key, {"dataset":dataset,"critical_ratio":.3,"seed":seed,
                "metrics":_metric_record(dataset,item["metrics"]),"budget":audit,
                "action":"REUSE_EXISTING","source":config[section]["final_manifest"]})


def run_three_w_ratios(config: dict[str, Any], data_root: Path, records: dict[str, Any], path: Path,
                       masks: dict[tuple[str,float],Path]) -> None:
    stage=config["three_w"]
    for ratio in (.2,.4):
        for seed in map(int,stage["seeds"]):
            key=f"3W|{ratio:.2f}|{seed}"
            if key in records: continue
            base=yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8")); current=copy.deepcopy(base)
            current.update({"seed":seed,"protocol_seed":int(stage["protocol_seed"]),"criticality_source":str(masks["3W",ratio]),
                "methods":[THREE_W_METHODS[2]],"evaluation_split":"test","output_dir":str(Path(stage["output_dir"])/f"ratio_{int(ratio*100):02d}"/f"seed_{seed}")})
            current["training"]["supcon_batching"]="original"; result=run_three_w(current,data_root)
            store(path,records,key,{"dataset":"3W","critical_ratio":ratio,"seed":seed,
                "metrics":_metric_record("3W",result["methods"][THREE_W_METHODS[2]]),
                "budget":budget_audit(read(masks["3W",ratio])["criticality"]),"action":"NEW_TRAINING_REQUIRED",
                "source":str(Path(current["output_dir"])/"result.json"),"fairness":result["fairness"]})


def run_tep_ratios(config: dict[str, Any], device: str, records: dict[str, Any], path: Path,
                   masks: dict[tuple[str,float],Path]) -> None:
    stage=config["tep"]; base=yaml.safe_load(Path(stage["base_config"]).read_text(encoding="utf-8")); views,_=load_fixed_views(base)
    clean={split:views[split]["clean"].astype(np.float32) for split in views}; stages={split:fault_stages(views[split],base) for split in views}
    spectral=config["spectral_diffusion"]
    statistics=fit_spectral_statistics(clean["train"],float(spectral["clip_quantile"]),"train")
    schedule=DiffusionSchedule.cosine(int(spectral["diffusion_steps"]),device)
    for ratio in (.2,.4):
        mask=read(masks["TEP",ratio])["criticality"]; augmenter=FrequencyForwardDiffusion(statistics,schedule.alpha_bars,
            np.asarray(mask["soft_mask"],np.float32),3,1,True,True,device)
        for seed in map(int,stage["seeds"]):
            key=f"TEP|{ratio:.2f}|{seed}"
            if key in records: continue
            runtime=_runtime(base,seed); runtime["diagnosis"]={"threshold_band_width":.05,"high_correlation_quantile":.90}
            pretrain=epoch_orders(len(clean["train"]),int(runtime["epochs"]),seed+10000); probe=epoch_orders(len(clean["train"]),int(runtime["probe_epochs"]),seed+20000)
            seed_everything(seed); template=build_model(runtime["model"],clean["train"].shape[1],2); initial=copy.deepcopy(template.state_dict())
            fairness={"manifest_sha256":file_sha256(base["fixed_views"]["manifest"]),"initialization_sha256":_state_hash(initial),
                "pretrain_order_sha256":hashlib.sha256("\n".join(','.join(map(str,row)) for row in pretrain).encode()).hexdigest(),
                "probe_order_sha256":hashlib.sha256("\n".join(','.join(map(str,row)) for row in probe).encode()).hexdigest()}
            augmented={"test":clean["test"]}; audits={}
            for split,offset in (("train",0),("validation",100)):
                changed,diag=augmenter.augment(clean[split],"selective",seed+int(spectral["sampling_seed_offset"])+offset,5,int(runtime["batch_size"])); augmented[split]=changed
                audits[split]=augmentation_mechanism_metrics(clean[split],changed,views[split]["labels"],stages[split],np.asarray(mask["hard_mask"],bool),diag)
            output=Path(stage["output_dir"])/f"ratio_{int(ratio*100):02d}"/f"seed_{seed}"; metadata={**fairness,"critical_ratio":ratio,"outer_test_run":False}
            method=_fit_method(f"RATIO_{ratio:.2f}",augmented,audits,views,clean,stages,initial,pretrain,probe,runtime,device,output/"model.pt",metadata,evaluation_splits=("test",))
            store(path,records,key,{"dataset":"TEP","critical_ratio":ratio,"seed":seed,"metrics":_metric_record("TEP",method),
                "budget":budget_audit(mask),"action":"NEW_TRAINING_REQUIRED","source":str(output/"metrics.json"),"fairness":fairness})


def summarize(config: dict[str, Any], records: dict[str, Any]) -> None:
    rows=[]
    for item in records.values(): rows.append({k:item[k] for k in ("dataset","critical_ratio","seed","action","source")}|item["metrics"]|item["budget"])
    output=Path(config["output"]["results_csv"]); output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w",newline="",encoding="utf-8-sig") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(sorted(rows,key=lambda x:(x["dataset"],x["critical_ratio"],x["seed"])))
    lines=["# Critical Ratio Sensitivity","","冻结方法的 D/E、S、timesteps、soft allocation、matched budget 与训练协议保持不变；本结果不重新选择 0.30。",""]
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),sharey=False)
    for axis,dataset in zip(axes,("3W","TEP")):
        lines += [f"## {dataset}","","| Ratio | Macro-F1 | AUPRC | FAR | Early Recall | Delay | Budget error |","|---:|---:|---:|---:|---:|---:|---:|"]
        xs=[];ys=[];errs=[]
        for ratio in (.2,.3,.4):
            selected=[r for r in rows if r["dataset"]==dataset and float(r["critical_ratio"])==ratio]; values=np.asarray([r["macro_f1"] for r in selected])
            mean=lambda field:float(np.mean([r[field] for r in selected])); xs.append(ratio);ys.append(values.mean());errs.append(values.std(ddof=1))
            lines.append(f"| {ratio:.2f} | {values.mean():.4f} ± {values.std(ddof=1):.4f} | {mean('auprc'):.4f} | {mean('far'):.4f} | {mean('early_recall'):.4f} | {mean('detection_delay'):.2f} | {max(r['matched_budget_error'] for r in selected):.2e} |")
        span=max(ys)-min(ys)
        interpretation=("0.30 与 0.40 形成近似平台，而 0.20 更低且方差更大；0.30 对 3W 位于合理稳定区间。"
                        if dataset=="3W" else
                        "0.30 是邻域局部低点，0.20/0.40 均更高；TEP 不支持 0.30 为下游性能稳定最优点，但冻结参数不重开。")
        lines += ["",f"0.20–0.40 Macro-F1 range: `{span:.4f}`。{interpretation}",""]
        axis.errorbar(xs,ys,yerr=errs,marker='o',capsize=4);axis.axvline(.3,color='gray',ls='--',lw=1);axis.set_title(dataset);axis.set_xlabel('critical ratio');axis.set_ylabel('Macro-F1');axis.grid(alpha=.25)
    fig.tight_layout(); figure=Path(config["output"]["figure"]);figure.parent.mkdir(parents=True,exist_ok=True);fig.savefig(figure,dpi=180);plt.close(fig)
    Path(config["output"]["report"]).write_text("\n".join(lines)+"\n",encoding="utf-8")


def run(config: dict[str, Any], data_root: Path, dataset: str) -> dict[str, Any]:
    if config["audit"]["test_used_for_ratio_selection"] or config["audit"]["outer_test_run"]: raise RuntimeError("ratio audit boundary changed")
    path=Path(config["output"]["manifest"]);records=read(path).get("results",{}) if path.exists() else {};masks=ensure_masks(config);reuse_frozen(config,records,path,masks)
    device=select_device(str(config["device"]));os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    if dataset in ("3w","both"):run_three_w_ratios(config,data_root,records,path,masks)
    if dataset in ("tep","both"):run_tep_ratios(config,device,records,path,masks)
    if len(records)==18:summarize(config,records)
    return {"status":"PAPER_RATIO_SENSITIVITY_COMPLETE","records":len(records),"outer_test_run":False}


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--config",default="configs/paper_ratio_sensitivity.yaml");parser.add_argument("--data-root",type=Path,required=True);parser.add_argument("--dataset",choices=("3w","tep","both"),default="both")
    args=parser.parse_args();config=yaml.safe_load(Path(args.config).read_text(encoding="utf-8"));print(json.dumps(run(config,args.data_root,args.dataset),ensure_ascii=False))


if __name__=="__main__":main()
