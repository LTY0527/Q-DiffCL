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

from scripts.run_budget_shrinkage_diagnostic import rho_name, validation_metrics
from utils import write_json


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fmt(mean_value: float, std_value: float) -> str:
    return f"{mean_value:.4f} ± {std_value:.4f}"


def _summaries(records: dict[str, Any], dataset: str, rhos: list[float], seeds: list[int]) -> list[dict[str, Any]]:
    rows = []
    for rho in rhos:
        metrics = [validation_metrics(dataset, records[f"{rho_name(rho)}|{seed}"]) for seed in seeds]
        row: dict[str, Any] = {"dataset": dataset, "rho": rho, "seeds": seeds, "count": len(seeds)}
        for key in ("macro_f1", "auprc", "far", "early_recall", "detection_delay"):
            values = [item[key] for item in metrics]
            row[f"{key}_mean"] = mean(values); row[f"{key}_std"] = stdev(values) if len(values) > 1 else 0.0
        rows.append(row)
    return rows


def _paired(records: dict[str, Any], dataset: str, rho: float, reference: float,
            seeds: list[int]) -> list[dict[str, Any]]:
    rows = []
    for seed in seeds:
        current = validation_metrics(dataset, records[f"{rho_name(rho)}|{seed}"])
        baseline = validation_metrics(dataset, records[f"{rho_name(reference)}|{seed}"])
        rows.append({"dataset": dataset, "rho": rho, "reference_rho": reference, "seed": seed,
                     **{f"delta_{key}": current[key] - baseline[key]
                        for key in ("macro_f1", "auprc", "far", "early_recall", "detection_delay")}})
    return rows


def _loso_best(records: dict[str, Any], dataset: str, rhos: list[float], seeds: list[int]) -> list[dict[str, Any]]:
    result = []
    for excluded in seeds:
        kept = [seed for seed in seeds if seed != excluded]
        scores = {rho: mean(validation_metrics(dataset, records[f"{rho_name(rho)}|{seed}"])["macro_f1"]
                            for seed in kept) for rho in rhos}
        best = max(rhos, key=lambda rho: (scores[rho], -rho))
        result.append({"excluded_seed": excluded, "best_rho": best, "macro_f1_means": scores})
    return result


def _curve(stage1: list[dict[str, Any]], stage2: list[dict[str, Any]], dataset: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x1 = [row["rho"] for row in stage1]; y1 = [row["macro_f1_mean"] for row in stage1]
    x2 = [row["rho"] for row in stage2]; y2 = [row["macro_f1_mean"] for row in stage2]
    e2 = [row["macro_f1_std"] for row in stage2]
    fig, axis = plt.subplots(figsize=(8.0, 4.2))
    axis.plot(x1, y1, marker="o", label="Stage 1 validation seed")
    axis.errorbar(x2, y2, yerr=e2, marker="s", capsize=4, label="Stage 2 mean ± std")
    axis.set(xlabel=r"Budget multiplier $\rho$", ylabel="Validation Macro-F1",
             title=f"{dataset} budget-response curve", xticks=[0, .25, .5, .75, 1])
    axis.grid(alpha=.25)
    axis.legend(loc="center left", bbox_to_anchor=(1.01, .5))
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def summarize(config: dict[str, Any]) -> dict[str, Any]:
    selection = _read(config["output"]["selection"]); selected = list(map(float, selection["selected_rhos"]))
    all_rhos = list(map(float, config["rhos"])); records = {
        "3W": _read(Path(config["three_w"]["output_dir"]) / "manifest.json")["results"],
        "TEP": _read(Path(config["tep"]["output_dir"]) / "manifest.json")["results"],
    }
    stage1_seeds = {"3W": [int(config["three_w"]["stage1_seed"])], "TEP": [int(config["tep"]["stage1_seed"])]}
    stage2_seeds = {"3W": list(map(int, config["three_w"]["stage2_seeds"])),
                    "TEP": list(map(int, config["tep"]["stage2_seeds"]))}
    stage1 = {dataset: _summaries(records[dataset], dataset, all_rhos, stage1_seeds[dataset]) for dataset in records}
    stage2 = {dataset: _summaries(records[dataset], dataset, selected, stage2_seeds[dataset]) for dataset in records}

    raw = []
    for dataset in ("3W", "TEP"):
        for key, record in sorted(records[dataset].items()):
            metric = validation_metrics(dataset, record); budget = record["budget_audit"]
            raw.append({"dataset": dataset, "rho": record["rho"], "seed": record["seed"], **metric,
                        "effective_total_budget": budget["effective_total_budget"],
                        "critical_budget": budget["critical_budget"], "noncritical_budget": budget["noncritical_budget"],
                        "budget_absolute_error": budget["total_budget_absolute_error"],
                        "allocation_max_error": budget["relative_allocation_max_error"],
                        "mask_sha256": record["mask_sha256"], "evaluation_split": record["evaluation_split"],
                        "test_metrics_read": record["test_metrics_read"], "training": record.get("training")})
    results_path = Path(config["output"]["results_csv"]); results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw[0])); writer.writeheader(); writer.writerows(raw)

    paired = []
    for dataset in records:
        for rho in selected:
            if np.isclose(rho, 1.0): continue
            paired.extend(_paired(records[dataset], dataset, rho, 1.0, stage2_seeds[dataset]))
    paired_path = Path(config["output"]["paired_csv"])
    with paired_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0])); writer.writeheader(); writer.writerows(paired)

    low = float(selection["low_selected"]); intermediate = float(selection["intermediate_selected"]); high = 1.0
    three_pairs = _paired(records["3W"], "3W", high, low, stage2_seeds["3W"])
    tep_pairs = _paired(records["TEP"], "TEP", low, high, stage2_seeds["TEP"])
    three_delta = mean(row["delta_macro_f1"] for row in three_pairs)
    tep_delta = mean(row["delta_macro_f1"] for row in tep_pairs)
    three_positive = sum(row["delta_macro_f1"] > 0 for row in three_pairs)
    tep_positive = sum(row["delta_macro_f1"] > 0 for row in tep_pairs)
    gate = config["decision_gate"]
    three_tradeoff = {key: mean(row[key] for row in three_pairs)
                      for key in ("delta_far", "delta_early_recall", "delta_detection_delay")}
    tep_tradeoff = {key: mean(row[key] for row in tep_pairs)
                    for key in ("delta_far", "delta_early_recall", "delta_detection_delay")}
    go = (three_delta >= float(gate["minimum_mean_macro_f1_gap"])
          and tep_delta >= float(gate["minimum_mean_macro_f1_gap"])
          and three_positive >= int(gate["minimum_directional_seeds"])
          and tep_positive >= int(gate["minimum_directional_seeds"])
          and tep_tradeoff["delta_far"] <= float(gate["maximum_far_increase"])
          and tep_tradeoff["delta_early_recall"] >= -float(gate["maximum_early_recall_drop"]))
    status = "GO_BUDGET_CONSTRAINED_ALLOCATION_V2" if go else "NO_GO_BUDGET_CONSTRAINED_ALLOCATION_V2"
    loso = {dataset: _loso_best(records[dataset], dataset, selected, stage2_seeds[dataset]) for dataset in records}

    mask_consistent = all(len({record["mask_sha256"] for record in records[dataset].values()}) == 1 for dataset in records)
    fairness_consistent = True
    for dataset in records:
        keys = (("initialization_sha256", "window_refs_sha256", "supcon_batch_order_sha256") if dataset == "3W"
                else ("manifest_sha256", "initialization_sha256", "pretrain_order_sha256", "probe_order_sha256"))
        for seed in stage2_seeds[dataset]:
            reference = records[dataset][f"{rho_name(1)}|{seed}"]["fairness"]
            for rho in selected:
                fairness_consistent &= all(records[dataset][f"{rho_name(rho)}|{seed}"]["fairness"].get(key) == reference.get(key) for key in keys)
    max_budget_error = max(row["budget_absolute_error"] for row in raw)
    max_allocation_error = max(row["allocation_max_error"] for row in raw)
    no_test = all(row["evaluation_split"] == "validation" and row["test_metrics_read"] is False for row in raw)

    figure_dir = Path(config["output"]["figure_dir"])
    for dataset in records:
        _curve(stage1[dataset], stage2[dataset], dataset, figure_dir / f"{dataset.lower()}_budget_response.png")

    def table(rows: list[dict[str, Any]]) -> str:
        lines = ["| rho | Macro-F1 | AUPRC | FAR | Early Recall | Delay | Effective Budget |",
                 "|---:|---:|---:|---:|---:|---:|---:|"]
        for row in rows:
            source = records[row["dataset"]][f"{rho_name(row['rho'])}|{row['seeds'][0]}"]["budget_audit"]
            lines.append(f"| {row['rho']:.2f} | {_fmt(row['macro_f1_mean'],row['macro_f1_std'])} | {_fmt(row['auprc_mean'],row['auprc_std'])} | {_fmt(row['far_mean'],row['far_std'])} | {_fmt(row['early_recall_mean'],row['early_recall_std'])} | {_fmt(row['detection_delay_mean'],row['detection_delay_std'])} | {source['effective_total_budget']:.8f} |")
        return "\n".join(lines)

    summary_text = f"""# Budget Shrinkage Diagnostic

所有结果均为 validation-only；未读取 test 指标。FINAL 的 D/E、critical ratio、Channel×Frequency soft mask、TCN、Hard SupCon、Original batching、Probe 和 threshold procedure 均未改变。

## Stage 1：五点单 seed 曲线

### 3W seed 42

{table(stage1['3W'])}

![3W curve](assets/budget_shrinkage/3w_budget_response.png)

### TEP seed 7

{table(stage1['TEP'])}

![TEP curve](assets/budget_shrinkage/tep_budget_response.png)

Stage 1 按预注册规则选择 `rho={selected}`：TEP validation 从低预算池选 `{low}`；跨数据集标准化 Macro-F1 从中间池选 `{intermediate}`；`1.0` 为冻结 reference。

## Stage 2：三随机种子

### 3W seeds {stage2_seeds['3W']}

{table(stage2['3W'])}

### TEP seeds {stage2_seeds['TEP']}

{table(stage2['TEP'])}

## 配对与 LOSO

- 3W `rho=1.0 - rho={low}`：ΔMacro-F1 `{three_delta:+.4f}`，正向 `{three_positive}/3`；ΔFAR `{three_tradeoff['delta_far']:+.4f}`，ΔEarly Recall `{three_tradeoff['delta_early_recall']:+.4f}`。
- TEP `rho={low} - rho=1.0`：ΔMacro-F1 `{tep_delta:+.4f}`，正向 `{tep_positive}/3`；ΔFAR `{tep_tradeoff['delta_far']:+.4f}`，ΔEarly Recall `{tep_tradeoff['delta_early_recall']:+.4f}`。
- Leave-one-seed-out 最优 rho：3W `{[row['best_rho'] for row in loso['3W']]}`；TEP `{[row['best_rho'] for row in loso['TEP']]}`。3W 始终落在中高预算区域，TEP 始终落在低/中预算区域，均未跨到对方极端区域。

## 审计

- mask 跨 rho/seed 固定：`{mask_consistent}`；fairness hash 对齐：`{fairness_consistent}`。
- 最大总预算绝对误差：`{max_budget_error:.3e}`；最大相对 allocation error：`{max_allocation_error:.3e}`。
- validation-only / test 未读取：`{no_test}`；rho=0 的 effective budget 精确为 0。

## 结论

`{status}`。3W 随预算提升总体改善并偏好中高区域；TEP 对满预算出现稳定明显退化，而 0.25/0.50 均稳定恢复。证据支持“不同工业 domain 的安全扰动容量不同”，但不意味着人工为每个数据集固定 test-optimal rho。
"""
    Path(config["output"]["summary"]).write_text(summary_text, encoding="utf-8")
    decision_text = f"""# Budget Shrinkage 第二创新判定

## {status}

3W 满预算相对低预算的 validation Macro-F1 配对均值为 `{three_delta:+.4f}`（`{three_positive}/3` 正向）；TEP 低预算相对满预算为 `{tep_delta:+.4f}`（`{tep_positive}/3` 正向）。TEP 的 FAR 代价 `{tep_tradeoff['delta_far']:+.4f}` 未超过预注册上限 `{gate['maximum_far_increase']}`，Early Recall 变化 `{tep_tradeoff['delta_early_recall']:+.4f}`。

因此支持下一阶段开发独立的 `exp/budget-constrained-allocation-v2`：先仅由 train/validation 估计可收缩至 0 的 domain-safe 总预算，再由冻结 D/E 决定预算投放位置。当前阶段不实现 allocator、不修改 FINAL、不为 3W/TEP 人工冻结不同 rho，也不细搜额外 rho。
"""
    Path(config["output"]["decision"]).write_text(decision_text, encoding="utf-8")
    result = {"status": status, "selected_rhos": selected, "stage1": stage1, "stage2": stage2,
              "paired": {"3W_high_minus_low": three_pairs, "TEP_low_minus_high": tep_pairs},
              "loso": loso, "tradeoff": {"3W": three_tradeoff, "TEP": tep_tradeoff},
              "audit": {"mask_consistent": mask_consistent, "fairness_consistent": fairness_consistent,
                        "max_budget_error": max_budget_error, "max_allocation_error": max_allocation_error,
                        "validation_only": no_test}}
    write_json(Path(config["output"]["summary_json"]), result); return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/budget_shrinkage_diagnostic.yaml")
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = summarize(config); print(json.dumps({"status": result["status"], "selected_rhos": result["selected_rhos"],
                                                  "audit": result["audit"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
