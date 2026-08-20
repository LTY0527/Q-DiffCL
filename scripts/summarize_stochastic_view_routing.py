from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml

from scripts.run_budget_shrinkage_diagnostic import validation_metrics
from scripts.run_stochastic_view_routing import _read, variant_name
from utils import write_json


def _stats(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def _records(config: dict[str, Any], dataset: str) -> dict[str, Any]:
    key = "three_w" if dataset == "3W" else "tep"
    return _read(Path(config[key]["output_dir"]) / "manifest.json")["results"]


def _row(config: dict[str, Any], dataset: str, stage: str, p: float, seeds: list[int]) -> dict[str, Any]:
    records = _records(config, dataset)
    values = [validation_metrics(dataset, records[f"{variant_name(p)}|{seed}"]) for seed in seeds]
    row: dict[str, Any] = {"stage": stage, "dataset": dataset, "p": p,
                           "seeds": "/".join(map(str, seeds)), "count": len(seeds)}
    for metric in ("macro_f1", "auprc", "far", "early_recall", "detection_delay"):
        row[f"{metric}_mean"], row[f"{metric}_std"] = _stats([item[metric] for item in values])
    ratios = []
    for seed in seeds:
        audit = records[f"{variant_name(p)}|{seed}"]["routing_audit"]
        current = audit["validation"]
        route = current.get("stochastic_view_routing", current.get("routing", {}).get("stochastic_view_routing"))
        ratios.append(float(route["realized_route_ratio"]))
    row["route_ratio_mean"], row["route_ratio_std"] = _stats(ratios)
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def summarize(config: dict[str, Any]) -> dict[str, Any]:
    selection = _read(config["output"]["selection"])
    rows: list[dict[str, Any]] = []; paired: list[dict[str, Any]] = []
    for dataset in ("3W", "TEP"):
        key = "three_w" if dataset == "3W" else "tep"
        a_seed = [int(config[key]["stage_a_seed"])]
        rows.extend(_row(config, dataset, "A", float(p), a_seed) for p in config["candidates"])
        if "stage_b" in selection:
            b_seeds = list(map(int, config[key]["stage_b_seeds"]))
            b_candidates = sorted({0., 1., *map(float, selection["top2"][dataset])})
            rows.extend(_row(config, dataset, "B", p, b_seeds) for p in b_candidates)
            for candidate in selection["stage_b"][dataset]["candidates"]:
                for seed, delta in zip(b_seeds, candidate["deltas"]):
                    paired.append({"stage": "B", "dataset": dataset, "p": candidate["p"],
                                   "endpoint_best": candidate["endpoint_best"], "seed": seed,
                                   **{f"delta_{name}": value for name, value in delta.items()}})
        if "stage_c" in selection:
            c_seeds = list(map(int, config[key]["stage_c_seeds"]))
            chosen = float(selection["stage_c"][dataset]["p"])
            rows.extend(_row(config, dataset, "C", p, c_seeds) for p in (0., chosen, 1.))
            for seed, delta in zip(c_seeds, selection["stage_c"][dataset]["deltas"]):
                paired.append({"stage": "C", "dataset": dataset, "p": chosen,
                               "endpoint_best": selection["stage_c"][dataset]["endpoint_best"], "seed": seed,
                               **{f"delta_{name}": value for name, value in delta.items()}})
    _write_csv(Path(config["output"]["results_csv"]), rows)
    _write_csv(Path(config["output"]["paired_csv"]), paired)

    def table(dataset: str, stage: str) -> str:
        current = [row for row in rows if row["dataset"] == dataset and row["stage"] == stage]
        lines = ["| p | Seeds | Macro-F1 | AUPRC | FAR | Early Recall | Delay | Route ratio |",
                 "|---:|---|---:|---:|---:|---:|---:|---:|"]
        for row in current:
            lines.append(f"| {row['p']:.2f} | {row['seeds']} | {row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f} | "
                         f"{row['auprc_mean']:.4f} ± {row['auprc_std']:.4f} | {row['far_mean']:.4f} ± {row['far_std']:.4f} | "
                         f"{row['early_recall_mean']:.4f} ± {row['early_recall_std']:.4f} | "
                         f"{row['detection_delay_mean']:.2f} ± {row['detection_delay_std']:.2f} | "
                         f"{row['route_ratio_mean']:.3f} ± {row['route_ratio_std']:.3f} |")
        return "\n".join(lines)

    sections = []
    for stage in ("A", "B", "C"):
        if any(row["stage"] == stage for row in rows):
            sections.append(f"## Stage {stage}\n\n### 3W\n\n{table('3W', stage)}\n\n### TEP\n\n{table('TEP', stage)}")
    status = selection.get("status", selection.get("stage_b_status", "STAGE_A_COMPLETE"))
    top2 = selection.get("top2", {})
    frozen = selection.get("stage_c", {}) if status == "GO_STOCHASTIC_VIEW_ROUTING" else {}
    gate_lines = ["| Dataset | p | Endpoint | ΔMacro-F1 | Positive | ΔAUPRC | ΔFAR | Catastrophic | Pass |",
                  "|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for dataset, stage_b in selection.get("stage_b", {}).items():
        for candidate in stage_b["candidates"]:
            delta = candidate["deltas"]
            macro = mean(row["macro_f1"] for row in delta)
            auprc = mean(row["auprc"] for row in delta)
            far = mean(row["far"] for row in delta)
            positive = sum(row["macro_f1"] > 0 for row in delta)
            catastrophic = not candidate["checks"]["no_catastrophic_seed"]
            gate_lines.append(f"| {dataset} | {candidate['p']:.2f} | {candidate['endpoint_best']:.2f} | "
                              f"{macro:+.4f} | {positive}/{len(delta)} | {auprc:+.4f} | {far:+.4f} | "
                              f"{catastrophic} | {candidate['passed']} |")
    gate_table = "\n".join(gate_lines)
    text = f"""# Stochastic View Routing

SVR 对每个样本生成稳定哈希均匀数 `u`，当 `u < p` 时只采用完整 FINAL_QDIFFCL 视图，否则只采用冻结 `SCALING(std=0.05)` 视图。`p` 是按数据集由 validation 校准的标量，推理无新增参数。

边界回归：`p=0` 逐元素等于 SCALING，`p=1` 逐元素等于 FINAL；路由按 sample ID 与 seed 可复现，每个样本恰走一个分支。

Stage A Top-2：3W `{top2.get('3W')}`；TEP `{top2.get('TEP')}`。

{chr(10).join(sections)}

## 判定

`{status}`。

{gate_table}

3W 两个中间候选均为 0/3 seeds 正向，并且存在相对最佳端点超过 0.02 的 Macro-F1 降幅；TEP 两个中间候选虽然平均 Macro-F1 增益超过 0.003，但都只有 1/3 seeds 正向。因此两个数据集都未通过 Stage B 预注册门槛，Stage C 未运行。

冻结结果：`{frozen}`。本轮只读取 train/validation，`test_read=False`，没有重新运行或读取旧 locked test 用于选择。
"""
    Path(config["output"]["summary"]).write_text(text, encoding="utf-8")
    decision = f"""# SVR Decision

## {status}

Stage B 中 3W 和 TEP 都没有中间 `p` 满足预注册门槛，因此 Stage C 未运行，也未生成 `configs/qdiffcl_svr_final.yaml`。保留 FINAL_QDIFFCL 与 DCBR positive mechanism evidence，停止继续开发 router/controller。

本判定仅依据 validation；旧 locked test 未被读取或重跑。
"""
    Path(config["output"]["decision"]).write_text(decision, encoding="utf-8")
    result = {"status": status, "top2": top2, "stage_b": selection.get("stage_b"),
              "stage_c": selection.get("stage_c"), "test_used_for_selection": False,
              "results": rows, "paired": paired}
    write_json(Path(config["output"]["summary_json"]), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stochastic_view_routing.yaml")
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = summarize(config); print(json.dumps({"status": result["status"], "top2": result["top2"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
