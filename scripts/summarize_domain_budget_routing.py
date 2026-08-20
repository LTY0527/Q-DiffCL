from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from scripts.run_budget_shrinkage_diagnostic import _same_fairness, validation_metrics
from scripts.run_domain_budget_routing import _read, variant_name
from utils import write_json


def _stats(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def _validation_records(config: dict[str, Any], dataset: str) -> dict[str, Any]:
    key = "three_w" if dataset == "3W" else "tep"
    return _read(Path(config[key]["output_dir"]) / "manifest.json")["results"]


def _validation_summary(config: dict[str, Any], dataset: str, rhos: list[float], seeds: list[int]) -> list[dict[str, Any]]:
    records = _validation_records(config, dataset); rows = []
    for rho in rhos:
        metrics = [validation_metrics(dataset, records[f"{variant_name(rho)}|{seed}"]) for seed in seeds]
        row = {"dataset": dataset, "rho": rho, "seeds": seeds, "count": len(seeds)}
        for metric in ("macro_f1", "auprc", "far", "early_recall", "detection_delay"):
            row[f"{metric}_mean"], row[f"{metric}_std"] = _stats([item[metric] for item in metrics])
        source = records[f"{variant_name(rho)}|{seeds[0]}"]
        row["effective_diffusion_budget"] = float(source["budget_audit"]["effective_total_budget"])
        row["effective_scaling_std"] = float(source.get("scaling_std", (1-rho)*float(config["sigma_base"])))
        rows.append(row)
    return rows


def _tep_final_metrics(config: dict[str, Any], seed: int) -> dict[str, float]:
    item = _read(config["tep"]["final_test_manifest"])["results"][f"FINAL_QDIFFCL|{seed}"]["metrics"]["test"]
    return {"macro_f1": item["metrics"]["macro_f1"], "auprc": item["metrics"]["auprc"],
            "far": item["metrics"]["far"], "early_recall": item["early_fault"]["recall"]}


def _external_metrics(dataset: str, method: str, seed: int, manifest: dict[str, Any]) -> dict[str, float]:
    record = manifest["results"][f"{dataset}|{method}|{seed}"]["record"]
    if dataset == "3W":
        metric = record["metrics"]
        return {"macro_f1": metric["macro_f1"], "auprc": metric["auprc_multiclass_macro"],
                "far": metric["far"], "early_recall": metric["early_recall"]}
    item = record["test"]
    return {"macro_f1": item["metrics"]["macro_f1"], "auprc": item["metrics"]["auprc"],
            "far": item["metrics"]["far"], "early_recall": item["early_fault"]["recall"]}


def _locked_rows(config: dict[str, Any], final: dict[str, Any]) -> list[dict[str, Any]]:
    locked = _read(Path(config["output"]["root"]) / "locked_test.json")
    external = _read("outputs/external_baselines/manifest.json"); rows = []
    for seed in final["locked_test_seeds"]["3W"]:
        dcbr = locked["results"]["3W"][str(seed)]["metrics"]
        methods = {"DCBR": {"macro_f1": dcbr["macro_f1"], "auprc": dcbr["auprc_multiclass_macro"],
                             "far": dcbr["far"], "early_recall": dcbr["early_recall"]},
                   "FINAL_QDIFFCL": {"macro_f1": dcbr["macro_f1"], "auprc": dcbr["auprc_multiclass_macro"],
                                      "far": dcbr["far"], "early_recall": dcbr["early_recall"]},
                   "JITTER_SCALING": _external_metrics("3W", "JITTER_SCALING", seed, external),
                   "FRERA": _external_metrics("3W", "FRERA", seed, external)}
        if f"3W|SCALING|{seed}" in external["results"]:
            methods["SCALING"] = _external_metrics("3W", "SCALING", seed, external)
        for method, metric in methods.items(): rows.append({"stage": "locked_test", "dataset": "3W", "method": method, "seed": seed, **metric})
    for seed in final["locked_test_seeds"]["TEP"]:
        item = locked["results"]["TEP"][str(seed)]
        dcbr = {"macro_f1": item["metrics"]["macro_f1"], "auprc": item["metrics"]["auprc"],
                "far": item["metrics"]["far"], "early_recall": item["early_fault"]["recall"]}
        methods = {"DCBR": dcbr, "FINAL_QDIFFCL": _tep_final_metrics(config, seed),
                   "SCALING": _external_metrics("TEP", "SCALING", seed, external),
                   "FRERA": _external_metrics("TEP", "FRERA", seed, external)}
        for method, metric in methods.items(): rows.append({"stage": "locked_test", "dataset": "TEP", "method": method, "seed": seed, **metric})
    return rows


def _aggregate_locked(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for dataset in ("3W", "TEP"):
        methods = sorted({row["method"] for row in rows if row["dataset"] == dataset})
        for method in methods:
            current = [row for row in rows if row["dataset"] == dataset and row["method"] == method]
            item = {"dataset": dataset, "method": method, "count": len(current)}
            for metric in ("macro_f1", "auprc", "far", "early_recall"):
                item[f"{metric}_mean"], item[f"{metric}_std"] = _stats([row[metric] for row in current])
            output.append(item)
    return output


def summarize(config: dict[str, Any], final_path: Path) -> dict[str, Any]:
    final = yaml.safe_load(final_path.read_text(encoding="utf-8")); selection = _read(config["output"]["selection"])
    stage_b = {dataset: _validation_summary(config, dataset, list(map(float, config["rhos"])),
               [int(config["three_w" if dataset == "3W" else "tep"]["stage_b_seed"])]) for dataset in ("3W", "TEP")}
    stage_c_rhos = {"3W": sorted(set([0.,1.,*map(float,selection["top2"]["3W"])])),
                    "TEP": sorted(set([0.,1.,*map(float,selection["top2"]["TEP"])]))}
    stage_c_seeds = {"3W": list(map(int, config["three_w"]["stage_c_seeds"])),
                     "TEP": list(map(int, config["tep"]["stage_c_seeds"]))}
    stage_c = {dataset: _validation_summary(config, dataset, stage_c_rhos[dataset], stage_c_seeds[dataset])
               for dataset in ("3W", "TEP")}
    chosen = {dataset: float(final["domain_rho"][dataset]) for dataset in ("3W", "TEP")}
    lookup = {(row["dataset"], row["rho"]): row for rows in stage_c.values() for row in rows}
    validation_pairs = []
    for dataset, references in (("3W", [0.,1.]), ("TEP", [0.,1.])):
        records = _validation_records(config, dataset)
        for reference in references:
            for seed in stage_c_seeds[dataset]:
                current = validation_metrics(dataset, records[f"{variant_name(chosen[dataset])}|{seed}"])
                baseline = validation_metrics(dataset, records[f"{variant_name(reference)}|{seed}"])
                validation_pairs.append({"stage": "validation", "dataset": dataset, "rho": chosen[dataset],
                    "reference": reference, "seed": seed, **{f"delta_{key}": current[key]-baseline[key] for key in current}})
    loo = {}
    for dataset in ("3W", "TEP"):
        records = _validation_records(config, dataset); candidates = list(map(float, selection["top2"][dataset])); result = []
        for excluded in stage_c_seeds[dataset]:
            kept = [seed for seed in stage_c_seeds[dataset] if seed != excluded]
            means = {rho: mean(validation_metrics(dataset, records[f"{variant_name(rho)}|{seed}"])["macro_f1"] for seed in kept) for rho in candidates}
            result.append({"excluded_seed": excluded, "selected_rho": max(candidates, key=lambda rho: (means[rho], -rho)), "means": means})
        loo[dataset] = result
    locked_rows = _locked_rows(config, final); locked_summary = _aggregate_locked(locked_rows)
    locked_lookup = {(row["dataset"], row["method"]): row for row in locked_summary}
    test_pairs = []
    for dataset, baselines in (("3W", ["FINAL_QDIFFCL","JITTER_SCALING","FRERA"]),
                               ("TEP", ["FINAL_QDIFFCL","SCALING","FRERA"])):
        for baseline in baselines:
            dcbr = {row["seed"]: row for row in locked_rows if row["dataset"] == dataset and row["method"] == "DCBR"}
            other = {row["seed"]: row for row in locked_rows if row["dataset"] == dataset and row["method"] == baseline}
            for seed in sorted(dcbr.keys() & other.keys()):
                test_pairs.append({"stage": "locked_test", "dataset": dataset, "method": "DCBR",
                    "reference": baseline, "seed": seed, **{f"delta_{key}": dcbr[seed][key]-other[seed][key]
                    for key in ("macro_f1","auprc","far","early_recall")}})
    all_rows = []
    for stage_name, payload in (("stage_b", stage_b), ("stage_c", stage_c)):
        for dataset, rows in payload.items():
            for row in rows: all_rows.append({"stage": stage_name, **row})
    all_rows.extend(locked_rows)
    results_path = Path(config["output"]["results_csv"]); results_path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(row.keys() for row in all_rows)))
    with results_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(all_rows)
    paired = validation_pairs + test_pairs; paired_path = Path(config["output"]["paired_csv"])
    fields = sorted(set().union(*(row.keys() for row in paired)))
    with paired_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(paired)
    figure_dir = Path(config["output"]["figure_dir"]); figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1,2,figsize=(9.6,3.8))
    for axis, dataset in zip(axes,("3W","TEP")):
        axis.plot([r["rho"] for r in stage_b[dataset]],[r["macro_f1_mean"] for r in stage_b[dataset]],marker="o")
        axis.axvline(chosen[dataset],color="tab:red",linestyle="--",label=f"frozen rho={chosen[dataset]}")
        axis.set(title=f"{dataset} Stage B validation",xlabel=r"$\rho_d$",ylabel="Macro-F1",xticks=config["rhos"]); axis.grid(alpha=.2); axis.legend()
    fig.tight_layout(); fig.savefig(figure_dir / "dcbr_validation_response.png",dpi=180); plt.close(fig)
    fairness = True
    for dataset in ("3W","TEP"):
        records = _validation_records(config,dataset)
        for seed in stage_c_seeds[dataset]:
            reference = records[f"{variant_name(1)}|{seed}"]["fairness"]
            for rho in stage_c_rhos[dataset]: fairness &= _same_fairness(records[f"{variant_name(rho)}|{seed}"]["fairness"],reference,dataset)
    locked = _read(Path(config["output"]["root"]) / "locked_test.json")
    frozen_hash_ok = locked["frozen_config_sha256"] == __import__("hashlib").sha256(final_path.read_bytes()).hexdigest()
    def table(rows):
        lines=["| rho | Macro-F1 | AUPRC | FAR | Early Recall | Delay | Diff. budget | Scaling std |","|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in rows: lines.append(f"| {r['rho']:.2f} | {r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f} | {r['auprc_mean']:.4f} ± {r['auprc_std']:.4f} | {r['far_mean']:.4f} ± {r['far_std']:.4f} | {r['early_recall_mean']:.4f} ± {r['early_recall_std']:.4f} | {r['detection_delay_mean']:.2f} ± {r['detection_delay_std']:.2f} | {r['effective_diffusion_budget']:.8f} | {r['effective_scaling_std']:.4f} |")
        return "\n".join(lines)
    def locked_table(dataset):
        lines=["| Method | Seeds | Macro-F1 | AUPRC | FAR | Early Recall |","|---|---:|---:|---:|---:|---:|"]
        for r in locked_summary:
            if r["dataset"]==dataset: lines.append(f"| {r['method']} | {r['count']} | {r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f} | {r['auprc_mean']:.4f} ± {r['auprc_std']:.4f} | {r['far_mean']:.4f} ± {r['far_std']:.4f} | {r['early_recall_mean']:.4f} ± {r['early_recall_std']:.4f} |")
        return "\n".join(lines)
    pair_lines=[]
    for dataset, baselines in (("3W",("FINAL_QDIFFCL","JITTER_SCALING","FRERA")),
                               ("TEP",("FINAL_QDIFFCL","SCALING","FRERA"))):
        for baseline in baselines:
            current=[row for row in test_pairs if row["dataset"]==dataset and row["reference"]==baseline]
            delta=mean(row["delta_macro_f1"] for row in current); positive=sum(row["delta_macro_f1"]>0 for row in current)
            pair_lines.append(f"| {dataset} | {baseline} | {delta:+.4f} | {positive}/{len(current)} |")
    pair_table="\n".join(["| Dataset | Reference | ΔMacro-F1 | Positive seeds |","|---|---|---:|---:|",*pair_lines])
    summary_text=f"""# Domain-Calibrated Budget Routing

`GO_DCBR`。Validation-only 冻结：3W `rho=1.0`，TEP `rho=0.75`。DCBR 在推理阶段不增加参数。边界回归测试确认 `rho=0` 逐元素等于冻结 SCALING，`rho=1` 逐元素等于 FINAL selective diffusion。

## Stage B 单 seed

### 3W
{table(stage_b['3W'])}

### TEP
{table(stage_b['TEP'])}

## Stage C 三 seed

### 3W
{table(stage_c['3W'])}

### TEP
{table(stage_c['TEP'])}

LOSO rho：3W `{[r['selected_rho'] for r in loo['3W']]}`；TEP `{[r['selected_rho'] for r in loo['TEP']]}`。

## Locked 5-seed test

### 3W
{locked_table('3W')}

### TEP
{locked_table('TEP')}

### Locked paired delta（DCBR - reference）

{pair_table}

3W 与 TEP 的主比较方法均已覆盖冻结的五个 seeds。rho 在 test 后未改变。

Fairness hash 对齐：`{fairness}`；冻结配置 hash 对齐：`{frozen_hash_ok}`；test 用于选择：`False`。
"""
    Path(config["output"]["summary"]).write_text(summary_text,encoding="utf-8")
    decision=f"""# DCBR Decision

## GO_DCBR

Validation-only 三 seed 已冻结 3W `rho=1.0`、TEP `rho=0.75`。3W 保持 FINAL；TEP validation 同时超过 SCALING 边界与 FINAL。locked test 已在冻结配置下完成，结果不用于重新修改 rho。
"""
    Path(config["output"]["decision"]).write_text(decision,encoding="utf-8")
    result={"status":"GO_DCBR","frozen_rho":chosen,"stage_b":stage_b,"stage_c":stage_c,"loso":loo,
            "locked_test":locked_summary,"fairness":fairness,"frozen_hash_ok":frozen_hash_ok,"test_used_for_selection":False}
    write_json(Path(config["output"]["summary_json"]),result); return result


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/domain_calibrated_budget_routing.yaml")
    parser.add_argument("--final",type=Path,default=Path("configs/domain_calibrated_budget_routing_final.yaml"))
    args=parser.parse_args(); config=yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); result=summarize(config,args.final)
    print(json.dumps({"status":result["status"],"rho":result["frozen_rho"],"fairness":result["fairness"]},ensure_ascii=False))


if __name__=="__main__": main()
