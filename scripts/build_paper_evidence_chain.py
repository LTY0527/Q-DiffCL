from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, precision_recall_fscore_support

import scripts.run_3w_clean_baseline as base3w
from baselines.external_augmentations import FreRAAdapter
from diffusion import DiffusionSchedule, spectral_noise_variance
from scripts.run_3w_final_primary_grouped import FINAL_PRIMARY_CLASSES, build_model as build_3w_model
from trainers import build_model as build_tep_model
from utils import select_device, write_json


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with target.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def stats(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def _external_metrics(dataset: str, record: dict[str, Any]) -> dict[str, float]:
    if dataset == "3W":
        item = record["metrics"]
        return {"macro_f1": float(item["macro_f1"]), "auprc": float(item["auprc_multiclass_macro"]),
                "far": float(item["far"]), "early_recall": float(item["early_recall"]),
                "delay": float(item["mean_detection_delay_seconds"])}
    item = record["test"]
    return {"macro_f1": float(item["metrics"]["macro_f1"]), "auprc": float(item["metrics"]["auprc"]),
            "far": float(item["metrics"]["far"]), "early_recall": float(item["early_fault"]["recall"]),
            "delay": float(item["detection_delay"]["mean_delay_samples"])}


def inventory(config: dict[str, Any]) -> dict[str, Any]:
    sources = config["sources"]
    required = {name: str(path) for name, path in sources.items() if isinstance(path, str)}
    files = {name: {"path": path, "exists": Path(path).exists(),
                    "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).is_file() else None}
             for name, path in required.items()}
    external = read_json(sources["external_manifest"])
    coverage = {dataset: sorted({key.split("|")[1] for key in external["results"] if key.startswith(dataset + "|")})
                for dataset in ("3W", "TEP")}
    frozen = yaml.safe_load(Path(sources["final_config"]).read_text(encoding="utf-8"))
    mechanism_summary_path = Path("outputs/paper_mechanism_ablation/summary.json")
    mechanism_complete = (mechanism_summary_path.exists() and
                          read_json(mechanism_summary_path).get("status") == "PAPER_MECHANISM_ABLATION_AUDIT_GO")
    result = {
        "frozen_method": frozen,
        "files": files,
        "external_coverage": coverage,
        "existing_evidence": {
            "A1_uniform_and_soft_final": True,
            "A1_hard_mask_selective": False,
            "A1_soft_without_budget_match": False,
            "A2_semantic_components": True,
            "A3_dcbr_domain_calibration": True,
            "B1_3w_checkpoints_for_group_replay": True,
            "B2_tep_checkpoints_for_fault_replay": True,
            "B3_five_seed_metrics": True,
            "D1_limited_data": False,
            "D2_missingness_sensitivity": False,
            "D3_critical_ratio_sensitivity": False,
            "D4_efficiency": True,
        },
        "supplemented_evidence": {
            "A1_hard_mask_selective": mechanism_complete,
            "A1_soft_without_budget_match": mechanism_complete,
        },
        "selection_policy": {"old_locked_test": "development evidence only",
                             "new_method_selection": False, "paper_final_outer_test_run": False},
    }
    write_json(Path(config["output"]["inventory_json"]), result)
    lines = ["# Repository / Result Inventory", "", "本清单先于任何补充实验生成。旧 locked test 仅作为 development evidence，不用于新方法或参数选择。", "",
             "| Evidence | Available | Action |", "|---|---|---|"]
    actions = {True: "复用并核验公平性", False: "初始缺失"}
    for name, available in result["existing_evidence"].items():
        action=actions[available]
        if not available and result["supplemented_evidence"].get(name):
            action="初始缺失；本阶段已按 validation-only 公平协议补齐"
        elif not available:
            action+="；不得声称已支持"
        lines.append(f"| `{name}` | `{available}` | {action} |")
    lines += ["", "## External baseline coverage", "",
              f"- 3W: `{coverage['3W']}`", f"- TEP: `{coverage['TEP']}`", "",
              "当前主表已覆盖 NoAug、Jitter、Scaling、Jitter+Scaling、Uniform Diffusion、FreRA、FINAL_QDIFFCL/DCBR。自动增强与工业 diffusion-native baseline 尚无公平 shared-backbone 适配，标记为 supplementary coverage gap，不为刷榜强行加入。"]
    target = Path(config["output"]["inventory"]); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def mechanism_tables(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    external = read_json(config["sources"]["external_manifest"])["results"]
    rows: list[dict[str, Any]] = []
    seeds = {"3W": config["seeds"]["three_w"], "TEP": config["seeds"]["tep"]}
    for dataset in ("3W", "TEP"):
        for method in ("UNIFORM_DIFFUSION", "FINAL_QDIFFCL"):
            values = [_external_metrics(dataset, external[f"{dataset}|{method}|{seed}"]["record"])
                      for seed in seeds[dataset]]
            row = {"dataset": dataset, "method": method, "support": "SUPPORTED", "seeds": "/".join(map(str, seeds[dataset]))}
            for metric in ("macro_f1", "auprc", "far", "early_recall", "delay"):
                row[f"{metric}_mean"], row[f"{metric}_std"] = stats([value[metric] for value in values])
            rows.append(row)
        for method in ("HARD_MASK_SELECTIVE", "SOFT_MASK_WO_BUDGET_MATCH"):
            rows.append({"dataset": dataset, "method": method, "support": "UNSUPPORTED / DO NOT CLAIM",
                         "seeds": "", "reason": "no protocol-aligned 3-seed result in inventory"})
    component = []
    with Path(config["sources"]["component_table"]).open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["method"] in {"UNIFORM", "D_ONLY", "E_ONLY", "FINAL_DE"}:
                component.append({"dataset": row["dataset"], "method": row["method"], "support": "SUPPORTED",
                                  "macro_f1_mean": row["macro_f1_mean"], "macro_f1_std": row["macro_f1_std"],
                                  "far_mean": row["far_mean"], "far_std": row["far_std"],
                                  "early_recall_mean": row["early_recall_mean"], "early_recall_std": row["early_recall_std"]})
    write_csv(config["output"]["mechanism_csv"], rows + component)

    dcbr_rows: list[dict[str, Any]] = []
    locked = read_json(config["sources"]["dcbr_locked"])["results"]
    for dataset, ds_key in (("3W", "3W"), ("TEP", "TEP")):
        for method in (("DCBR",),):
            values = []
            for seed in seeds[dataset]:
                item = locked[ds_key][str(seed)]
                if dataset == "3W":
                    metric = item["metrics"]
                    values.append({"macro_f1": metric["macro_f1"], "auprc": metric["auprc_multiclass_macro"],
                                   "far": metric["far"], "early_recall": metric["early_recall"]})
                else:
                    values.append({"macro_f1": item["metrics"]["macro_f1"], "auprc": item["metrics"]["auprc"],
                                   "far": item["metrics"]["far"], "early_recall": item["early_fault"]["recall"]})
            row = {"dataset": dataset, "method": "DCBR", "rho": 1.0 if dataset == "3W" else .75,
                   "scope": "existing development locked test"}
            for metric in ("macro_f1", "auprc", "far", "early_recall"):
                row[f"{metric}_mean"], row[f"{metric}_std"] = stats([float(v[metric]) for v in values])
            dcbr_rows.append(row)
        for method in ("SCALING", "FINAL_QDIFFCL"):
            values = [_external_metrics(dataset, external[f"{dataset}|{method}|{seed}"]["record"])
                      for seed in seeds[dataset]]
            row = {"dataset": dataset, "method": method, "rho": 0.0 if method == "SCALING" else 1.0,
                   "scope": "existing development locked test"}
            for metric in ("macro_f1", "auprc", "far", "early_recall"):
                row[f"{metric}_mean"], row[f"{metric}_std"] = stats([v[metric] for v in values])
            dcbr_rows.append(row)
        validation = read_json(config["sources"]["dcbr_validation_3w" if dataset == "3W" else "dcbr_validation_tep"])["results"]
        values = []
        stage_seeds = [42, 43, 44] if dataset == "3W" else [7, 42, 2026]
        for seed in stage_seeds:
            record = validation[f"DCBR_075|{seed}"]
            method = record["method"]
            if dataset == "3W":
                metric = method["metrics"]; values.append({"macro_f1": metric["macro_f1"], "auprc": metric["auprc_multiclass_macro"], "far": metric["far"], "early_recall": metric["early_recall"]})
            else:
                item = method["validation"]; values.append({"macro_f1": item["metrics"]["macro_f1"], "auprc": item["metrics"]["auprc"], "far": item["metrics"]["far"], "early_recall": item["early_fault"]["recall"]})
        row = {"dataset": dataset, "method": "GLOBAL_RHO_075", "rho": .75, "scope": "validation mechanism reference"}
        for metric in ("macro_f1", "auprc", "far", "early_recall"):
            row[f"{metric}_mean"], row[f"{metric}_std"] = stats([float(v[metric]) for v in values])
        dcbr_rows.append(row)
    write_csv(config["output"]["dcbr_csv"], dcbr_rows)
    return rows + component, dcbr_rows


def _checkpoint_3w(method: str, seed: int) -> Path:
    if method == "FINAL_QDIFFCL":
        return Path(f"outputs/qdiffcl_final_5seed/3w/FINAL_QDIFFCL/seed_{seed}/FREQUENCY_SELECTIVE_R1_model.pt")
    return Path(f"outputs/external_baselines/3w/seed_{seed}/{method}/model.pt")


def _model_state(path: Path, device: str) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location=device, weights_only=True)
    return payload["model_state_dict"] if "model_state_dict" in payload else payload


def _load_3w_context(config: dict[str, Any], data_root: Path):
    from datasets.three_w import discover_instances
    base3w.PRIMARY_CLASSES = FINAL_PRIMARY_CLASSES
    base3w.CLASS_TO_TARGET = {value: index for index, value in enumerate(FINAL_PRIMARY_CLASSES)}
    diffusion_config = yaml.safe_load(Path("configs/3w_diffusion_1seed.yaml").read_text(encoding="utf-8"))
    grouped_config = yaml.safe_load(Path(diffusion_config["base_config"]).read_text(encoding="utf-8"))
    base = yaml.safe_load(Path(grouped_config["base_config"]).read_text(encoding="utf-8"))
    grouped = Path(config["sources"]["grouped_3w"]); manifest = read_json(grouped / "grouped_split_manifest.json")
    wells = set(manifest["splits"][0]["wells"]["test"])
    instances = [item for item in discover_instances(data_root) if item.source == "WELL" and item.event_class in FINAL_PRIMARY_CLASSES and item.well_id in wells]
    preprocessor = read_json(grouped / "split_00" / "preprocessor.json")
    refs = {item.instance_id: base3w.instance_refs(item, int(base["protocol"]["window_length"]),
            int(base["protocol"]["stride"]), int(base["protocol"]["transient_offset"])) for item in instances}
    return base, instances, preprocessor, refs


def three_w_group_analysis(config: dict[str, Any], data_root: Path, device: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base, instances, preprocessor, refs_by_instance = _load_3w_context(config, data_root)
    seeds = list(map(int, config["seeds"]["three_w"])); methods = ("FINAL_QDIFFCL", "FRERA", "JITTER_SCALING")
    channels = len(preprocessor["retained_features"]); models = {}
    for seed in seeds:
        for method in methods:
            path = _checkpoint_3w(method, seed)
            model = build_3w_model(base["training"]["model"], channels, device)
            model.load_state_dict(_model_state(path, device)); model.eval()
            models[(method, seed)] = model
    grouped_predictions: dict[tuple[str, int, str], list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    instance_rows: list[dict[str, Any]] = []
    length = int(base["protocol"]["window_length"]); batch = 512
    for instance in instances:
        refs = refs_by_instance[instance.instance_id]
        if not refs: continue
        x, y = base3w.materialize(refs, {instance.instance_id: instance}, preprocessor, length, False)
        predictions = {}
        for (method, seed), model in models.items():
            pred = base3w.probabilities(model, x, y, batch, device).argmax(1); predictions[(method, seed)] = pred
            grouped_predictions[(method, seed, instance.well_id)].append((y, pred))
        for seed in seeds:
            final = predictions[("FINAL_QDIFFCL", seed)]
            for baseline in ("FRERA", "JITTER_SCALING"):
                other = predictions[(baseline, seed)]
                instance_rows.append({"seed": seed, "well_id": instance.well_id, "instance_id": instance.instance_id,
                    "class": instance.event_class, "baseline": baseline, "windows": len(y),
                    "rescued_windows": int(np.sum((final == y) & (other != y))),
                    "lost_windows": int(np.sum((final != y) & (other == y))),
                    "delta_accuracy": float(np.mean(final == y) - np.mean(other == y))})
    rows = []
    for (method, seed, well), bundles in grouped_predictions.items():
        y = np.concatenate([item[0] for item in bundles]); pred = np.concatenate([item[1] for item in bundles])
        present = np.unique(y)
        rows.append({"method": method, "seed": seed, "well_id": well, "windows": len(y),
                     "macro_f1": float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
                     "macro_f1_all_primary_classes": float(f1_score(y, pred, labels=np.arange(4), average="macro", zero_division=0)),
                     "observed_targets": "/".join(map(str, present.tolist())),
                     "far": float(np.mean(pred[y == 0] != 0)) if np.any(y == 0) else None,
                     **{f"class_{original}_recall": float(np.mean(pred[y == target] == target)) if np.any(y == target) else None
                        for target, original in enumerate(FINAL_PRIMARY_CLASSES) if target}})
    write_csv(config["output"]["well_csv"], rows); write_csv(config["output"]["instance_csv"], instance_rows)
    return rows, instance_rows


def _tep_checkpoint(method: str, seed: int) -> Path:
    if method == "FINAL_QDIFFCL":
        primary = Path(f"outputs/qdiffcl_final_5seed/tep/FINAL_QDIFFCL/seed_{seed}/model.pt")
        return primary if primary.exists() else Path(f"outputs/r1_des_weight_search/tep/DE_50_50/seed_{seed}/model.pt")
    if method == "DCBR": return Path(f"outputs/domain_budget_routing/tep/DCBR_075/seed_{seed}/model.pt")
    return Path(f"outputs/external_baselines/tep/seed_{seed}/SCALING/model.pt")


def _tep_threshold(method: str, seed: int, external: dict[str, Any], locked: dict[str, Any]) -> float:
    if method == "DCBR": return float(locked[str(seed)]["validation_threshold"])
    return float(external[f"TEP|{method}|{seed}"]["record"]["validation_threshold"])


def _predict_binary(model, values: np.ndarray, batch: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval(); scores=[]; embeddings=[]
    with torch.no_grad():
        for start in range(0, len(values), batch):
            output=model(torch.from_numpy(values[start:start+batch]).to(device)); scores.append(torch.softmax(output["logits"],1)[:,1].cpu().numpy()); embeddings.append(output["projection"].cpu().numpy())
    return np.concatenate(scores), np.concatenate(embeddings)


def tep_fault_analysis(config: dict[str, Any], device: str) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    manifest=read_json(config["sources"]["fixed_views"]); record=manifest["splits"]["test"]
    with np.load(record["path"],allow_pickle=False) as z: values=z["clean"].astype(np.float32); labels=z["labels"].astype(int); runs=z["run_uid"].astype(str); ends=z["end_sample"].astype(int)
    base=yaml.safe_load(Path("configs/stage_frequency_diffusion_mvp.yaml").read_text(encoding="utf-8")); spec=base["training"]["model"]
    external=read_json(config["sources"]["external_manifest"])["results"]; locked=read_json(config["sources"]["dcbr_locked"])["results"]["TEP"]
    rows=[]; embedding_sample={}; rng=np.random.default_rng(2026); chosen=np.sort(rng.choice(len(values),min(1600,len(values)),replace=False))
    for seed in map(int,config["seeds"]["tep"]):
        for method in ("FINAL_QDIFFCL","DCBR","SCALING"):
            model=build_tep_model(spec,values.shape[1],2).to(device); model.load_state_dict(_model_state(_tep_checkpoint(method,seed),device));
            score,embedding=_predict_binary(model,values,512,device); threshold=_tep_threshold(method,seed,external,locked); pred=score>=threshold
            if seed==7: embedding_sample[method]=embedding[chosen]
            for fault in range(1,21):
                selected=np.char.find(runs,f"fault_{fault:02d}:")>=0; truth=labels[selected]; current=pred[selected]
                recall=float(np.mean(current[truth==1])) if np.any(truth==1) else None
                precision=float(np.mean(truth[current]==1)) if np.any(current) else 0.0
                f1=2*precision*recall/max(precision+recall,1e-12) if recall is not None else None
                fault_runs=np.unique(runs[selected]); delays=[]; missed=0
                for run in fault_runs:
                    idx=np.flatnonzero(runs==run); detected=idx[(labels[idx]==1)&pred[idx]]
                    if len(detected): delays.append(int(ends[detected[0]]-161))
                    else: missed+=1
                rows.append({"method":method,"seed":seed,"fault":fault,"windows":int(selected.sum()),"recall":recall,"f1":f1,
                             "mean_delay_samples":float(np.mean(delays)) if delays else None,"missed_runs":missed,
                             "far_global":float(np.mean(pred[labels==0])),"early_recall":float(np.mean(pred[(labels==1)&(ends<161+64+4*16)]))})
    embedding_sample["labels"]=labels[chosen]
    write_csv(config["output"]["fault_csv"],rows); return rows,embedding_sample


def bootstrap_groups(config: dict[str, Any], well_rows: list[dict[str, Any]], fault_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng=np.random.default_rng(int(config["bootstrap"]["seed"])); repeats=int(config["bootstrap"]["repeats"]); rows=[]
    for baseline in ("FRERA","JITTER_SCALING"):
        for metric in ("macro_f1","far"):
            deltas=[]
            for seed in config["seeds"]["three_w"]:
                final={r["well_id"]:r[metric] for r in well_rows if r["method"]=="FINAL_QDIFFCL" and r["seed"]==seed and r[metric] is not None}
                other={r["well_id"]:r[metric] for r in well_rows if r["method"]==baseline and r["seed"]==seed and r[metric] is not None}
                common=sorted(final.keys()&other.keys()); deltas.extend([final[w]-other[w] for w in common])
            boots=[float(np.mean(rng.choice(deltas,len(deltas),replace=True))) for _ in range(repeats)]
            rows.append({"dataset":"3W","comparison":f"FINAL_QDIFFCL-{baseline}","metric":metric,"groups":len(deltas),
                         "effect":float(np.mean(deltas)),"ci_low":float(np.quantile(boots,.025)),"ci_high":float(np.quantile(boots,.975))})
    for comparison in (("DCBR","FINAL_QDIFFCL"),("DCBR","SCALING")):
        for metric in ("recall","f1","mean_delay_samples"):
            a={(r["seed"],r["fault"]):r[metric] for r in fault_rows if r["method"]==comparison[0] and r[metric] is not None}
            b={(r["seed"],r["fault"]):r[metric] for r in fault_rows if r["method"]==comparison[1] and r[metric] is not None}
            delta=[a[k]-b[k] for k in sorted(a.keys()&b.keys())]; boots=[float(np.mean(rng.choice(delta,len(delta),replace=True))) for _ in range(repeats)]
            rows.append({"dataset":"TEP","comparison":f"{comparison[0]}-{comparison[1]}","metric":metric,"groups":len(delta),
                         "effect":float(np.mean(delta)),"ci_low":float(np.quantile(boots,.025)),"ci_high":float(np.quantile(boots,.975))})
    write_csv(config["output"]["stability_csv"],rows); return rows


def efficiency(config: dict[str, Any]) -> list[dict[str, Any]]:
    external=read_json(config["sources"]["external_manifest"])["results"]; rows=[]
    for dataset,seeds in (("3W",config["seeds"]["three_w"]),("TEP",config["seeds"]["tep"])):
        for method in ("SCALING","UNIFORM_DIFFUSION","FINAL_QDIFFCL","FRERA"):
            times=[]; peaks=[]
            for seed in seeds:
                record=external[f"{dataset}|{method}|{seed}"]["record"]
                if record.get("training_seconds") is not None: times.append(float(record["training_seconds"]))
                if record.get("peak_gpu_mib") is not None: peaks.append(float(record["peak_gpu_mib"]))
            rows.append({"dataset":dataset,"method":method,"training_seconds_mean":mean(times) if times else None,
                         "training_seconds_std":stdev(times) if len(times)>1 else 0.0 if times else None,
                         "peak_gpu_mib_mean":mean(peaks) if peaks else None,"trainable_augmentation_parameters":66 if method=="FRERA" else 0,
                         "inference_additional_parameters":0})
        dcbr=next(row.copy() for row in rows if row["dataset"]==dataset and row["method"]=="FINAL_QDIFFCL")
        dcbr.update({"method":"DCBR","trainable_augmentation_parameters":0,"inference_additional_parameters":0,
                     "note":"same inference model; domain-level rho only"}); rows.append(dcbr)
    write_csv(config["output"]["efficiency_csv"],rows); return rows


def figures(config: dict[str, Any], well_rows: list[dict[str, Any]], embedding: dict[str,np.ndarray]) -> list[str]:
    directory=Path(config["output"]["figure_dir"]); directory.mkdir(parents=True,exist_ok=True); paths=[]
    raw_directory=Path(config["output"]["docs_root"])/"visualization_data"; raw_directory.mkdir(parents=True,exist_ok=True)
    provenance=[]
    masks={name:read_json(path)["criticality"] for name,path in config["sources"]["final_masks"].items()}
    criticality_rows=[]
    for dataset,item in masks.items():
        for field in ("discriminative","early","composite","soft_mask"):
            for channel,row in enumerate(np.asarray(item[field],float)):
                for frequency,value in enumerate(row):
                    criticality_rows.append({"dataset":dataset,"field":field,"channel":channel,"frequency_bin":frequency,"value":float(value)})
    write_csv(raw_directory/"criticality_maps.csv",criticality_rows)
    for field,title in (("discriminative","D criticality"),("early","E criticality"),("composite","D+E composite"),("soft_mask","Soft mask")):
        fig,axes=plt.subplots(1,2,figsize=(10,3.8))
        for ax,(dataset,item) in zip(axes,masks.items()):
            image=np.asarray(item[field],float); im=ax.imshow(image,aspect="auto",cmap="viridis"); ax.set(title=dataset.upper(),xlabel="frequency bin",ylabel="channel"); fig.colorbar(im,ax=ax,fraction=.046)
        fig.suptitle(title); fig.tight_layout(); path=directory/f"{field}_heatmap.png"; fig.savefig(path,dpi=180); plt.close(fig); paths.append(str(path)); provenance.append({"figure":path.name,"raw_data":"criticality_maps.csv","filter":f"field={field}"})
    schedule=DiffusionSchedule.cosine(50,"cpu")
    allocation_rows=[]
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for ax,(dataset,item) in zip(axes,masks.items()):
        soft=np.asarray(item["soft_mask"],np.float32); timestep=1+(1-soft)*4
        uniform=spectral_noise_variance(schedule.alpha_bars,*soft.shape,"uniform",3,True).numpy()
        selective=spectral_noise_variance(schedule.alpha_bars,*soft.shape,"selective",3,True,torch.as_tensor(soft),1,5).numpy()
        for channel in range(soft.shape[0]):
            for frequency in range(soft.shape[1]):
                allocation_rows.append({"dataset":dataset,"channel":channel,"frequency_bin":frequency,"soft_mask":float(soft[channel,frequency]),"effective_timestep":float(timestep[channel,frequency]),"uniform_variance":float(uniform[channel,frequency]),"selective_variance":float(selective[channel,frequency])})
        im=ax.imshow(timestep,aspect="auto",cmap="magma"); ax.set(title=dataset.upper(),xlabel="frequency bin",ylabel="channel"); fig.colorbar(im,ax=ax,fraction=.046,label="effective timestep")
    write_csv(raw_directory/"frequency_allocation.csv",allocation_rows)
    fig.tight_layout(); path=directory/"timestep_map.png"; fig.savefig(path,dpi=180); plt.close(fig); paths.append(str(path)); provenance.append({"figure":path.name,"raw_data":"frequency_allocation.csv","filter":"effective_timestep"})
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for ax,(dataset,item) in zip(axes,masks.items()):
        soft=torch.as_tensor(item["soft_mask"],dtype=torch.float32); uniform=spectral_noise_variance(schedule.alpha_bars,*soft.shape,"uniform",3,True).numpy(); selective=spectral_noise_variance(schedule.alpha_bars,*soft.shape,"selective",3,True,soft,1,5).numpy()
        ax.plot(uniform.mean(0),label="Uniform"); ax.plot(selective.mean(0),label="Q-DiffCL"); ax.set(title=dataset.upper(),xlabel="frequency bin",ylabel="mean variance"); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); path=directory/"uniform_vs_qdiff_noise.png"; fig.savefig(path,dpi=180); plt.close(fig); paths.append(str(path)); provenance.append({"figure":path.name,"raw_data":"frequency_allocation.csv","filter":"uniform_variance/selective_variance"})
    fig,ax=plt.subplots(figsize=(6.4,4)); rho=np.linspace(0,1,101); ax.plot(rho,rho,label="selective diffusion share"); ax.plot(rho,1-rho,label="scaling share"); ax.scatter([1,.75],[1,.75],label="frozen 3W / TEP",zorder=3); ax.set(xlabel=r"domain $\rho$",ylabel="relative route strength",title="DCBR domain-level routing"); ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); path=directory/"dcbr_routing.png"; fig.savefig(path,dpi=180); plt.close(fig); paths.append(str(path))
    write_csv(raw_directory/"dcbr_routing.csv",[{"rho":float(value),"selective_share":float(value),"scaling_share":float(1-value)} for value in rho]); provenance.append({"figure":path.name,"raw_data":"dcbr_routing.csv","filter":"all"})
    budget_rows=[]
    with Path("docs/budget_shrinkage_results.csv").open(encoding="utf-8-sig") as handle:
        budget_rows=list(csv.DictReader(handle))
    write_csv(raw_directory/"budget_response.csv",budget_rows)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for ax,dataset in zip(axes,("3W","TEP")):
        grouped=defaultdict(list)
        for row in budget_rows:
            if row["dataset"]==dataset: grouped[float(row["rho"])].append(float(row["macro_f1"]))
        x=sorted(grouped); y=[mean(grouped[value]) for value in x]; error=[stdev(grouped[value]) if len(grouped[value])>1 else 0 for value in x]
        ax.errorbar(x,y,yerr=error,marker="o",capsize=3); ax.set(title=dataset,xlabel=r"diffusion budget $\rho$",ylabel="validation Macro-F1",xticks=x); ax.grid(alpha=.2)
    fig.suptitle("Frozen budget-response curves"); fig.tight_layout(); path=directory/"budget_response_3w_tep.png"; fig.savefig(path,dpi=180); plt.close(fig); paths.append(str(path)); provenance.append({"figure":path.name,"raw_data":"budget_response.csv","filter":"all"})
    final=[r for r in well_rows if r["method"]=="FINAL_QDIFFCL"]; wells=sorted({r["well_id"] for r in final}); seeds=sorted({r["seed"] for r in final}); matrix=np.full((len(seeds),len(wells)),np.nan)
    for row in final: matrix[seeds.index(row["seed"]),wells.index(row["well_id"])]=row["macro_f1"]
    fig,ax=plt.subplots(figsize=(max(7,len(wells)*.65),3.5)); im=ax.imshow(matrix,aspect="auto",vmin=0,vmax=1,cmap="viridis"); ax.set(xticks=range(len(wells)),xticklabels=wells,yticks=range(len(seeds)),yticklabels=seeds,xlabel="WELL",ylabel="seed",title="3W FINAL per-WELL Macro-F1"); plt.setp(ax.get_xticklabels(),rotation=45,ha="right"); fig.colorbar(im,ax=ax); fig.tight_layout(); path=directory/"3w_seed_well_heatmap.png"; fig.savefig(path,dpi=180); plt.close(fig); paths.append(str(path)); provenance.append({"figure":path.name,"raw_data":"../3w_per_well.csv","filter":"method=FINAL_QDIFFCL"})
    values=np.concatenate([embedding[m] for m in ("FINAL_QDIFFCL","DCBR","SCALING")]); labels_method=np.concatenate([[m]*len(embedding[m]) for m in ("FINAL_QDIFFCL","DCBR","SCALING")]); labels_class=np.tile(embedding["labels"],3); coords=TSNE(n_components=2,perplexity=35,init="pca",learning_rate="auto",random_state=2026).fit_transform(values)
    write_csv(raw_directory/"tep_embedding_tsne.csv",[{"method":str(method),"label":int(label),"x":float(point[0]),"y":float(point[1])} for method,label,point in zip(labels_method,labels_class,coords)])
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),sharex=True,sharey=True)
    for ax,method in zip(axes,("FINAL_QDIFFCL","DCBR","SCALING")):
        selected=labels_method==method; ax.scatter(coords[selected& (labels_class==0),0],coords[selected&(labels_class==0),1],s=5,alpha=.35,label="normal"); ax.scatter(coords[selected&(labels_class==1),0],coords[selected&(labels_class==1),1],s=5,alpha=.35,label="fault"); ax.set_title(method); ax.legend(markerscale=2)
    fig.suptitle("TEP shared-protocol t-SNE (seed 7)"); fig.tight_layout(); path=directory/"tep_embedding_tsne.png"; fig.savefig(path,dpi=180); plt.close(fig); paths.append(str(path)); provenance.append({"figure":path.name,"raw_data":"tep_embedding_tsne.csv","filter":"all"})
    write_csv(raw_directory/"figure_provenance.csv",provenance)
    return paths


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/paper_evidence_chain.yaml"); parser.add_argument("--data-root",type=Path,required=True)
    args=parser.parse_args(); config=yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); config["data_root"]=str(args.data_root)
    device=select_device(str(config["device"])); inv=inventory(config); mechanism,dcbr=mechanism_tables(config)
    well,instances=three_w_group_analysis(config,args.data_root,device); faults,embedding=tep_fault_analysis(config,device)
    bootstrap=bootstrap_groups(config,well,faults); eff=efficiency(config); plots=figures(config,well,embedding)
    result={"inventory":inv,"mechanism_rows":len(mechanism),"dcbr_rows":len(dcbr),"well_rows":len(well),"instance_rows":len(instances),"fault_rows":len(faults),"bootstrap":bootstrap,"efficiency":eff,"figures":plots,"test_used_for_new_selection":False,"paper_final_outer_test_run":False}
    write_json(Path(config["output"]["analysis_json"]),result); print(json.dumps({"status":"PAPER_EVIDENCE_REPLAY_COMPLETE","counts":{k:result[k] for k in ("well_rows","instance_rows","fault_rows")}},ensure_ascii=False))


if __name__=="__main__": main()
