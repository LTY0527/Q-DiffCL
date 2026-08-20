from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml

from scripts.run_budget_shrinkage_diagnostic import validation_metrics
from utils import write_json


VARIANTS=("UNIFORM_DIFFUSION","HARD_MASK_SELECTIVE","SOFT_MASK_SELECTIVE","SOFT_MASK_WO_BUDGET_MATCH")


def read(path: str|Path) -> dict[str,Any]: return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: str|Path,rows: list[dict[str,Any]]) -> None:
    target=Path(path); target.parent.mkdir(parents=True,exist_ok=True); fields=sorted(set().union(*(row.keys() for row in rows)))
    with target.open("w",newline="",encoding="utf-8-sig") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _stats(values:list[float])->tuple[float,float]: return mean(values),stdev(values) if len(values)>1 else 0.0


def summarize(config: dict[str,Any]) -> dict[str,Any]:
    rows=[]; paired=[]; audits={}; fairness=True; matched=True
    for dataset,key in (("3W","three_w"),("TEP","tep")):
        records=read(Path(config[key]["output_dir"])/"manifest.json")["results"]; seeds=list(map(int,config[key]["seeds"])); audits[dataset]={}
        if set(records)!={f"{variant}|{seed}" for variant in VARIANTS for seed in seeds}: raise RuntimeError(f"{dataset} mechanism record grid incomplete")
        for seed in seeds:
            reference=records[f"SOFT_MASK_SELECTIVE|{seed}"]["fairness"]
            varying={"uniform_r1_total_budget_matched","budget_scaled_selective_override"}
            normalized_reference={name:value for name,value in reference.items() if name not in varying}
            fairness &= all({name:value for name,value in records[f"{variant}|{seed}"]["fairness"].items() if name not in varying}==normalized_reference for variant in VARIANTS)
        for variant in VARIANTS:
            metrics=[validation_metrics(dataset,records[f"{variant}|{seed}"]) for seed in seeds]
            budget=[]; actual=[]; measurement_domain=None
            for seed in seeds:
                audit=records[f"{variant}|{seed}"]["augmentation_audit"]["validation"]
                budget.append(float(audit["expected_total_noise_budget"])); measurement_domain="frequency" if "actual_total_frequency_l1" in audit else "time"
                actual.append(float(audit.get("actual_total_frequency_l1",audit["time_normalized_l1"])))
            row={"dataset":dataset,"variant":variant,"seeds":"/".join(map(str,seeds)),"count":len(seeds),"evaluation_split":"validation","test_read":False}
            for metric in ("macro_f1","auprc","far","early_recall","detection_delay"):
                row[f"{metric}_mean"],row[f"{metric}_std"]=_stats([item[metric] for item in metrics])
            row["expected_spectral_budget_mean"],row["expected_spectral_budget_std"]=_stats(budget)
            row["measured_perturbation_l1_mean"],row["measured_perturbation_l1_std"]=_stats(actual); row["measurement_domain"]=measurement_domain; rows.append(row)
            audits[dataset][variant]={"expected_budgets":budget,"actual_frequency_l1":actual}
        uniform=audits[dataset]["UNIFORM_DIFFUSION"]["expected_budgets"]
        for variant in ("HARD_MASK_SELECTIVE","SOFT_MASK_SELECTIVE"):
            matched &= all(abs(a-b)<=1e-6 for a,b in zip(uniform,audits[dataset][variant]["expected_budgets"]))
        for reference_name in ("UNIFORM_DIFFUSION","HARD_MASK_SELECTIVE","SOFT_MASK_WO_BUDGET_MATCH"):
            for seed in seeds:
                current=validation_metrics(dataset,records[f"SOFT_MASK_SELECTIVE|{seed}"]); reference=validation_metrics(dataset,records[f"{reference_name}|{seed}"])
                paired.append({"dataset":dataset,"method":"SOFT_MASK_SELECTIVE","reference":reference_name,"seed":seed,
                               **{f"delta_{metric}":current[metric]-reference[metric] for metric in current}})
    if not fairness: raise RuntimeError("mechanism ablation fairness hashes differ")
    if not matched: raise RuntimeError("matched mechanism variants changed total budget")
    write_csv(config["output"]["results_csv"],rows); write_csv(config["output"]["paired_csv"],paired)
    def table(dataset):
        lines=["| Variant | Macro-F1 | AUPRC | FAR | Early Recall | Expected budget | Measured L1 (domain) |","|---|---:|---:|---:|---:|---:|---:|"]
        for row in rows:
            if row["dataset"]==dataset: lines.append(f"| {row['variant']} | {row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f} | {row['auprc_mean']:.4f} ± {row['auprc_std']:.4f} | {row['far_mean']:.4f} ± {row['far_std']:.4f} | {row['early_recall_mean']:.4f} ± {row['early_recall_std']:.4f} | {row['expected_spectral_budget_mean']:.8f} | {row['measured_perturbation_l1_mean']:.4f} ({row['measurement_domain']}) |")
        return "\n".join(lines)
    pair_lines=["| Dataset | Reference | ΔMacro-F1 | Positive seeds | ΔAUPRC | ΔFAR |","|---|---|---:|---:|---:|---:|"]
    for dataset in ("3W","TEP"):
        for reference in ("UNIFORM_DIFFUSION","HARD_MASK_SELECTIVE","SOFT_MASK_WO_BUDGET_MATCH"):
            current=[row for row in paired if row["dataset"]==dataset and row["reference"]==reference]
            pair_lines.append(f"| {dataset} | {reference} | {mean(r['delta_macro_f1'] for r in current):+.4f} | {sum(r['delta_macro_f1']>0 for r in current)}/{len(current)} | {mean(r['delta_auprc'] for r in current):+.4f} | {mean(r['delta_far'] for r in current):+.4f} |")
    text=f"""# Paper Mechanism Ablation — Validation Only

所有方法使用相同 split、初始化、batch order、TCN、Hard SupCon 和 Probe。3W seeds `42/43/44`，TEP seeds `7/42/2026`；未读取 test。

## 3W

{table('3W')}

## TEP

{table('TEP')}

## Paired Soft matched delta

{chr(10).join(pair_lines)}

Uniform、Hard 和 Soft matched 的 expected total spectral budget 数值相等：`{matched}`。Unmatched 仅移除全局预算匹配，保留相同 soft allocation 和 timestep map。Fairness hash 对齐：`{fairness}`。
"""
    Path(config["output"]["report"]).write_text(text,encoding="utf-8")
    result={"status":"PAPER_MECHANISM_ABLATION_AUDIT_GO","rows":rows,"paired":paired,"fairness":fairness,"matched_budget":matched,"test_read":False}
    write_json(Path("outputs/paper_mechanism_ablation/summary.json"),result); return result


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/paper_mechanism_ablation.yaml"); args=parser.parse_args(); config=yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); result=summarize(config); print(json.dumps({"status":result["status"],"fairness":result["fairness"],"matched_budget":result["matched_budget"]},ensure_ascii=False))


if __name__=="__main__": main()
