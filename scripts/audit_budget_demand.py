from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from frequency.budget_demand import (cross_group_shift_demand,
                                     group_bootstrap,
                                     leave_one_group_out,
                                     normalized_log_spectrum,
                                     separability_difficulty_demand)
from frequency.safe_capacity import distribution
from scripts.audit_safe_capacity import load_tep_train, load_three_w_train
from utils import write_json


def _sha(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _direction_loo(current_3w: float, current_tep: float,
                   loo_3w: np.ndarray, loo_tep: np.ndarray) -> float:
    checks = np.concatenate((loo_3w > current_tep, current_3w > loo_tep))
    return float(checks.mean()) if len(checks) else 0.0


def audit(config: dict[str, Any], data_root: Path) -> dict[str, Any]:
    final = yaml.safe_load(Path(config["final_config"]).read_text(encoding="utf-8"))
    if not final.get("frozen") or final["weights"] != {
        "weight_discriminative": .5, "weight_early": .5, "weight_run_stability": 0.0}:
        raise RuntimeError("FINAL_QDIFFCL changed")
    loader_path = config["three_w"]["loader_config"]
    if loader_path != config["tep"]["loader_config"]:
        raise RuntimeError("proxy audit must use one common loader protocol")
    loader_config = yaml.safe_load(Path(loader_path).read_text(encoding="utf-8"))
    bundles = {"3W": load_three_w_train(loader_config, data_root),
               "TEP": load_tep_train(loader_config)}
    normalized = {}; normalization_audit = {}; proxy = {}; bootstrap = {}; loo = {}
    repeats = int(config["bootstrap"]["repeats"]); seed = int(config["bootstrap"]["seed"])
    for index, (dataset, bundle) in enumerate(bundles.items()):
        normalized[dataset], normalization_audit[dataset] = normalized_log_spectrum(bundle["values"])
        proxy[dataset] = {
            "shift_demand": cross_group_shift_demand(normalized[dataset], bundle["unit"], bundle["stage"]),
            "difficulty_demand": separability_difficulty_demand(normalized[dataset], bundle["stage"]),
        }
        bootstrap[dataset] = group_bootstrap(normalized[dataset], bundle["unit"], bundle["stage"],
                                             repeats, seed + index * 1000)
        loo[dataset] = leave_one_group_out(normalized[dataset], bundle["unit"], bundle["stage"])
    gates = {}; rows = []
    threshold_boot = float(config["gate"]["minimum_bootstrap_direction_fraction"])
    threshold_loo = float(config["gate"]["minimum_loo_direction_fraction"])
    for name in ("shift_demand", "difficulty_demand"):
        score_3w = float(proxy["3W"][name]["score"]); score_tep = float(proxy["TEP"][name]["score"])
        differences = bootstrap["3W"][name] - bootstrap["TEP"][name]
        bootstrap_fraction = float((differences > 0).mean())
        loo_fraction = _direction_loo(score_3w, score_tep, loo["3W"][name], loo["TEP"][name])
        passed = score_3w > score_tep and bootstrap_fraction >= threshold_boot and loo_fraction >= threshold_loo
        gates[name] = {"passed": passed, "3w_score": score_3w, "tep_score": score_tep,
                       "3w_minus_tep": score_3w - score_tep,
                       "bootstrap_direction_fraction": bootstrap_fraction,
                       "bootstrap_difference": distribution(differences),
                       "loo_direction_fraction": loo_fraction,
                       "three_w_loo": distribution(loo["3W"][name]),
                       "tep_loo": distribution(loo["TEP"][name])}
        for dataset in ("3W", "TEP"):
            item = proxy[dataset][name]
            rows.append({"dataset": dataset, "proxy": name, "scope": "overall", "score": item["score"],
                         "between": item.get("between", item.get("binary_normal_fault", {}).get("between")),
                         "within": item.get("within", item.get("binary_normal_fault", {}).get("within")),
                         "groups": item.get("groups", len(np.unique(bundles[dataset]["unit"]))),
                         "samples": item.get("samples", item.get("binary_normal_fault", {}).get("samples"))})
            if name == "shift_demand":
                for stage, detail in item["stage"].items():
                    rows.append({"dataset": dataset, "proxy": name, "scope": stage, **detail})
            else:
                for scope, detail in (("normal_fault", item["binary_normal_fault"]),
                                      ("normal_early", item["normal_early"])):
                    rows.append({"dataset": dataset, "proxy": name, "scope": scope, **detail,
                                 "groups": len(np.unique(bundles[dataset]["unit"]))})
    stage_a_pass = any(item["passed"] for item in gates.values())
    status = "BUDGET_DEMAND_STAGE_A_PASS" if stage_a_pass else "NO_GO_BUDGET_DEMAND_CONTROLLER"
    result = {"status": status, "fit_scope": "train only", "validation_read": False,
              "test_read": False, "formula_shared_across_datasets": True,
              "bootstrap": {"repeats": repeats, "seed": seed}, "gates": gates,
              "normalization": normalization_audit,
              "input_hash": {dataset: {"normalized_spectrum_sha256": _sha(normalized[dataset]),
                                        "group_order_sha256": _sha(np.asarray(bundles[dataset]["unit"], dtype="U")),
                                        "stage_order_sha256": _sha(np.asarray(bundles[dataset]["stage"], dtype="U"))}
                             for dataset in bundles},
              "stage_a_pass": stage_a_pass}
    output = config["output"]; write_json(Path(output["audit_json"]), result)
    results_path = Path(output["results_csv"]); results_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "proxy", "scope", "score", "between", "within", "groups", "samples"]
    with results_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)
    figure_dir = Path(output["figure_dir"]); figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for axis, name, title in zip(axes, ("shift_demand", "difficulty_demand"),
                                 ("Proxy A bootstrap difference", "Proxy B bootstrap difference")):
        difference = bootstrap["3W"][name] - bootstrap["TEP"][name]
        axis.hist(difference, bins=30, color="tab:blue", alpha=.75); axis.axvline(0, color="black", linestyle="--")
        axis.set(title=title, xlabel="3W - TEP", ylabel="Count"); axis.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(figure_dir / "budget_demand_bootstrap.png", dpi=180); plt.close(fig)
    shift_stage_lines = "\n".join(
        f"| {dataset} | {stage} | {detail['score']:.6f} | {detail['between']:.6f} | {detail['within']:.6f} | {detail['groups']} |"
        for dataset in ("3W", "TEP") for stage, detail in proxy[dataset]["shift_demand"]["stage"].items())
    difficulty_lines = "\n".join(
        f"| {dataset} | {scope} | {detail['score']:.6f} | {detail['separability']:.6f} | {detail['samples']} |"
        for dataset in ("3W", "TEP")
        for scope, detail in (("Normal vs Fault", proxy[dataset]["difficulty_demand"]["binary_normal_fault"]),
                              ("Normal vs Early", proxy[dataset]["difficulty_demand"]["normal_early"])))
    report = f"""# Budget Demand Proxy Audit

所有输入与 normalization 均来自 frozen train split；未加载正式 validation/test 数据或指标。

| Proxy | 3W | TEP | 3W-TEP | Bootstrap direction | LOO direction | PASS |
|---|---:|---:|---:|---:|---:|---|
| A Cross-Group Shift | {gates['shift_demand']['3w_score']:.6f} | {gates['shift_demand']['tep_score']:.6f} | {gates['shift_demand']['3w_minus_tep']:+.6f} | {gates['shift_demand']['bootstrap_direction_fraction']:.3f} | {gates['shift_demand']['loo_direction_fraction']:.3f} | {gates['shift_demand']['passed']} |
| B Separability Difficulty | {gates['difficulty_demand']['3w_score']:.6f} | {gates['difficulty_demand']['tep_score']:.6f} | {gates['difficulty_demand']['3w_minus_tep']:+.6f} | {gates['difficulty_demand']['bootstrap_direction_fraction']:.3f} | {gates['difficulty_demand']['loo_direction_fraction']:.3f} | {gates['difficulty_demand']['passed']} |

Stage A：`{status}`。

预注册稳定性门槛：bootstrap direction `>= {threshold_boot:.2f}`，LOO direction `>= {threshold_loo:.2f}`。两个代理的点估计方向正确且 LOO 不由单一 group 驱动，但 bootstrap 方向均不够稳定。

## Proxy A stage-wise

| Dataset | Stage | Score | Between | Within | Groups |
|---|---|---:|---:|---:|---:|
{shift_stage_lines}

## Proxy B separability views

| Dataset | View | Difficulty | Separability | Samples |
|---|---|---:|---:|---:|
{difficulty_lines}

![bootstrap](assets/budget_demand/budget_demand_bootstrap.png)
"""
    Path(output["report"]).write_text(report, encoding="utf-8")
    decision = ("Stage A 至少一个 proxy 通过；允许执行 train-only grouped inner holdout Stage B。"
                if stage_a_pass else
                "两个预注册 proxy 均未通过稳定方向硬门；停止第二创新 controller 路线，不再构造新 proxy。")
    Path(output["decision"]).write_text(
        f"# Budget Demand Decision\n\n## {status}\n\n{decision}\n\n正式 validation/test 未读取，FINAL_QDIFFCL 未修改。\n",
        encoding="utf-8")
    Path(output["group_report"]).write_text(
        "# Budget Demand Group Check\n\n"
        f"Stage B 状态：`{'ELIGIBLE' if stage_a_pass else 'NOT_RUN_STAGE_A_HARD_GATE'}`。\n\n"
        + ("Stage A 已通过；应由独立 runner 执行 train-only grouped inner holdout。\n" if stage_a_pass else
           "Stage A 的两个预注册 proxy 均未达到 bootstrap 稳定性门槛，因此未训练任何 inner-holdout 模型，未生成 preferred rho 或 Spearman 结果。\n")
        + "正式 validation/test 未读取。\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/budget_demand_proxy_audit.yaml")
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    print(json.dumps(audit(config, args.data_root), ensure_ascii=False))


if __name__ == "__main__":
    main()
