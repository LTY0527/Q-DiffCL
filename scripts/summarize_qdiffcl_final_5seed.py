from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import yaml

from scripts.run_qdiffcl_final_5seed import METHODS, _assert_fairness


THREE_W_METRICS = ("macro_f1", "recall_macro", "auprc_fault_vs_normal", "auprc_multiclass_macro",
                   "far", "early_recall", "mean_detection_delay_seconds", "class_2_recall",
                   "class_8_recall", "class_9_recall", "class_2_f1", "class_8_f1", "class_9_f1")
TEP_METRICS = ("macro_f1", "auprc", "fault_recall", "far", "early_recall",
               "mean_detection_delay_samples", "detected_rate", "missed_runs")
LOWER_IS_BETTER = {"far", "mean_detection_delay_seconds", "mean_detection_delay_samples", "missed_runs"}


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _three_w(method: dict[str, Any]) -> dict[str, float]:
    metrics = method["metrics"]
    result = {key: float(metrics[key]) for key in THREE_W_METRICS[:7]}
    by_class = {int(row["original_class"]): row for row in metrics["per_class"]}
    for class_id in (2, 8, 9):
        result[f"class_{class_id}_recall"] = float(by_class[class_id]["recall"])
        result[f"class_{class_id}_f1"] = float(by_class[class_id]["f1"])
    return result


def _tep(method: dict[str, Any]) -> dict[str, float]:
    test = method["test"]; metrics = test["metrics"]; delay = test["detection_delay"]
    return {"macro_f1": float(metrics["macro_f1"]), "auprc": float(metrics["auprc"]),
            "fault_recall": float(metrics["fault_recall"]), "far": float(metrics["far"]),
            "early_recall": float(test["early_fault"]["recall"]),
            "mean_detection_delay_samples": float(delay["mean_delay_samples"]),
            "detected_rate": float(delay["detection_rate"]), "missed_runs": float(delay["missed_runs"])}


def metric_values(dataset: str, method: dict[str, Any]) -> dict[str, float]:
    return _three_w(method) if dataset == "3W" else _tep(method)


def load_reliability(config: dict[str, Any]) -> dict[str, dict[str, dict[int, dict[str, Any]]]]:
    result = {dataset: {method: {} for method in METHODS} for dataset in ("3W", "TEP")}
    for dataset, key in (("3W", "three_w"), ("TEP", "tep")):
        records = _read(Path(config[key]["output_dir"]) / "manifest.json")["results"]
        for record in records.values():
            result[dataset][record["method"]][int(record["seed"])] = {
                **record, "raw_method": record["metrics"], "metrics": metric_values(dataset, record["metrics"])}
    return result


def audit_reliability(config: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    expected = {"3W": list(map(int, config["three_w"]["seeds"])),
                "TEP": list(map(int, config["tep"]["seeds"]))}
    for dataset in expected:
        for method in METHODS:
            if sorted(result[dataset][method]) != sorted(expected[dataset]):
                raise RuntimeError(f"incomplete {dataset} {method}")
        for seed in expected[dataset]:
            base = result[dataset]["UNIFORM"][seed]["fairness"]
            for method in METHODS[1:]:
                _assert_fairness(base, result[dataset][method][seed]["fairness"], dataset, f"seed={seed} {method}")
    final = yaml.safe_load(Path(config["final_config"]).read_text(encoding="utf-8"))
    mask_audit = _read(config["output"]["mask_audit"])
    if final["weights"] != {"weight_discriminative": .5, "weight_early": .5, "weight_run_stability": 0.0}:
        raise RuntimeError("final weights reopened")
    for dataset in ("3W", "TEP"):
        if mask_audit[dataset]["FINAL_DE"]["mask_sha256"] != final["mask_sha256"][dataset]:
            raise RuntimeError(f"final mask mismatch {dataset}")
    max_budget = max(float(row["total_budget_error"]) for dataset in ("3W", "TEP")
                     for row in mask_audit[dataset].values())
    return {"weights_frozen": True, "fairness_hashes_consistent": True, "fixed_mask_across_seeds": True,
            "test_used_for_weight_selection": False, "max_budget_error": max_budget}


def reliability_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for dataset in ("3W", "TEP"):
        for method in METHODS:
            for seed, record in sorted(result[dataset][method].items()):
                rows.append({"dataset": dataset, "method": method, "seed": seed,
                             "training": record["training"], "source": record["source"], **record["metrics"]})
    return rows


def paired_rows(config: dict[str, Any], result: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tolerance = float(config["reliability_audit"]["nonworse_tolerance"]); rows = []; summary = {}
    for dataset in ("3W", "TEP"):
        metrics = THREE_W_METRICS if dataset == "3W" else TEP_METRICS
        summary[dataset] = {}
        for baseline in ("UNIFORM", "CURRENT_R1"):
            comparison = f"FINAL_QDIFFCL-{baseline}"; summary[dataset][comparison] = {}
            seeds = sorted(result[dataset]["FINAL_QDIFFCL"])
            for metric in metrics:
                raw = [result[dataset]["FINAL_QDIFFCL"][seed]["metrics"][metric] -
                       result[dataset][baseline][seed]["metrics"][metric] for seed in seeds]
                benefit = [-value if metric in LOWER_IS_BETTER else value for value in raw]
                loso = [mean(raw[:index] + raw[index + 1:]) for index in range(len(raw))]
                item = {"mean_delta": mean(raw), "std_delta": stdev(raw),
                        "positive_seed_count": sum(value > tolerance for value in benefit),
                        "nonworse_seed_count": sum(value >= -tolerance for value in benefit),
                        "loso_mean_min": min(loso), "loso_mean_max": max(loso),
                        "loso_mean_range": max(loso) - min(loso)}
                summary[dataset][comparison][metric] = item
                for seed, delta, directional in zip(seeds, raw, benefit):
                    rows.append({"row_type": "seed", "dataset": dataset, "comparison": comparison,
                                 "metric": metric, "seed": seed, "delta": delta,
                                 "directional_benefit": directional, **{key: "" for key in item}})
                rows.append({"row_type": "summary", "dataset": dataset, "comparison": comparison,
                             "metric": metric, "seed": "", "delta": "", "directional_benefit": "", **item})
    return rows, summary


def load_components(config: dict[str, Any], reliability: dict[str, Any]) -> dict[str, Any]:
    result = {dataset: {method: {} for method in ("UNIFORM", "D_ONLY", "E_ONLY", "FINAL_DE", "CURRENT_DES")}
              for dataset in ("3W", "TEP")}
    for dataset, key in (("3W", "three_w"), ("TEP", "tep")):
        seeds = list(map(int, config[key]["component_seeds"])); ablation = _read(config[key]["component_manifest"])["results"]
        for seed in seeds:
            result[dataset]["UNIFORM"][seed] = reliability[dataset]["UNIFORM"][seed]
            result[dataset]["FINAL_DE"][seed] = reliability[dataset]["FINAL_QDIFFCL"][seed]
            result[dataset]["CURRENT_DES"][seed] = reliability[dataset]["CURRENT_R1"][seed]
            for method in ("D_ONLY", "E_ONLY"):
                source = ablation[f"{method}|{seed}"]
                _assert_fairness(source["fairness"], reliability[dataset]["UNIFORM"][seed]["fairness"],
                                 dataset, f"component {method} seed={seed}")
                result[dataset][method][seed] = {"metrics": metric_values(dataset, source["method"]),
                                                 "training": "reused_des_ablation", "source": config[key]["component_manifest"],
                                                 "fairness": source["fairness"]}
    return result


def component_rows(components: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for dataset in ("3W", "TEP"):
        metrics = THREE_W_METRICS if dataset == "3W" else TEP_METRICS
        for method, seeds in components[dataset].items():
            row = {"dataset": dataset, "method": method,
                   "table_section": "main" if method != "CURRENT_DES" else "supplementary"}
            for metric in metrics:
                values = [record["metrics"][metric] for record in seeds.values()]
                row[f"{metric}_mean"] = mean(values); row[f"{metric}_std"] = stdev(values)
            rows.append(row)
    return rows


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


def _fmt(result: dict[str, Any], dataset: str, method: str, metric: str) -> str:
    values = [row["metrics"][metric] for row in result[dataset][method].values()]
    return f"{mean(values):.4f} ± {stdev(values):.4f}"


def _main_table(result: dict[str, Any], dataset: str) -> str:
    metrics = ("macro_f1", "far", "early_recall",
               "auprc_multiclass_macro" if dataset == "3W" else "auprc")
    lines = ["| Method | Macro-F1 | FAR | Early Recall | AUPRC | Worst seed |",
             "|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        worst = min(result[dataset][method], key=lambda seed: result[dataset][method][seed]["metrics"]["macro_f1"])
        lines.append(f"| {method} | {_fmt(result,dataset,method,metrics[0])} | {_fmt(result,dataset,method,metrics[1])} | "
                     f"{_fmt(result,dataset,method,metrics[2])} | {_fmt(result,dataset,method,metrics[3])} | {worst} |")
    return "\n".join(lines)


def _component_table(rows: list[dict[str, Any]], dataset: str) -> str:
    selected = [row for row in rows if row["dataset"] == dataset]
    lines = ["| Method | Section | Macro-F1 | FAR | Early Recall | AUPRC |", "|---|---|---:|---:|---:|---:|"]
    auprc = "auprc_multiclass_macro" if dataset == "3W" else "auprc"
    for row in selected:
        fmt = lambda metric: f"{row[f'{metric}_mean']:.4f} ± {row[f'{metric}_std']:.4f}"
        lines.append(f"| {row['method']} | {row['table_section']} | {fmt('macro_f1')} | {fmt('far')} | "
                     f"{fmt('early_recall')} | {fmt(auprc)} |")
    return "\n".join(lines)


def write_reports(config: dict[str, Any], reliability: dict[str, Any], paired: dict[str, Any],
                  components: list[dict[str, Any]], audit: dict[str, Any]) -> None:
    threshold = float(config["reliability_audit"]["catastrophic_macro_f1_drop"]); catastrophic = []
    for dataset in ("3W", "TEP"):
        for seed, final in reliability[dataset]["FINAL_QDIFFCL"].items():
            value = final["metrics"]["macro_f1"]
            if any(value < reliability[dataset][baseline][seed]["metrics"]["macro_f1"] - threshold
                   for baseline in ("UNIFORM", "CURRENT_R1")):
                catastrophic.append(f"{dataset}:{seed}")
    mask = _read(config["output"]["mask_audit"])
    source_counts: dict[str, int] = {}
    for dataset in ("3W", "TEP"):
        for method in METHODS:
            for record in reliability[dataset][method].values():
                source_counts[record["training"]] = source_counts.get(record["training"], 0) + 1
    summary = ["# FINAL_QDIFFCL 5-Seed Reliability", "",
               "冻结方法严格保持 `0.5D+0.5E`；本轮结果不用于重新调权重。", "",
               "## 3W", "", _main_table(reliability, "3W"), "",
               "## TEP", "", _main_table(reliability, "TEP"), "",
               "## Paired reliability", ""]
    for dataset in ("3W", "TEP"):
        summary.append(f"### {dataset}"); summary.append("")
        summary.extend(["| Comparison | Metric | Mean Δ | Positive seeds | Non-worse seeds | LOSO range |",
                        "|---|---|---:|---:|---:|---:|"])
        for comparison in ("FINAL_QDIFFCL-UNIFORM", "FINAL_QDIFFCL-CURRENT_R1"):
            for metric in ("macro_f1", "far", "early_recall"):
                row = paired[dataset][comparison][metric]
                summary.append(f"| {comparison} | {metric} | {row['mean_delta']:+.4f} | "
                               f"{row['positive_seed_count']}/5 | {row['nonworse_seed_count']}/5 | {row['loso_mean_range']:.4f} |")
        summary.append("")
    summary.extend(["## Catastrophic seed audit", "",
                    f"按 FINAL Macro-F1 比任一基线低至少 {threshold:.2f} 的预注册定义："
                    + ("、".join(catastrophic) if catastrophic else "无 catastrophic seed") + "。", "",
                    "## Mechanism / fairness", "",
                    f"- 3W FINAL vs CURRENT hard-mask Jaccard `{mask['3W']['FINAL_DE']['hard_mask_jaccard_full']:.4f}`，changed bins `{mask['3W']['FINAL_DE']['changed_bins_from_full']}`。",
                    f"- TEP FINAL vs CURRENT hard-mask Jaccard `{mask['TEP']['FINAL_DE']['hard_mask_jaccard_full']:.4f}`，changed bins `{mask['TEP']['FINAL_DE']['changed_bins_from_full']}`。",
                    f"- 最大 matched-budget error `{audit['max_budget_error']:.3e}`；mask 跨 seed 固定且公平性哈希一致。",
                    "- test 仅用于冻结后的可靠性评估，未用于权重选择。", "",
                    "## Result reuse", "",
                    f"- 直接复用既有 Uniform/CURRENT 测试结果：{source_counts.get('reused_existing', 0)} 条。",
                    f"- 复用 validation 搜索 checkpoint、仅新增 test 评估：{source_counts.get('reused_validation_checkpoint_test_evaluation_only', 0)} 条。",
                    f"- 新训练：{source_counts.get('new_training', 0)} 条（3W FINAL seeds 45/46；TEP seeds 43/44 的三种方法）。",
                    "- 最终组件表额外公平复用既有 D_ONLY/E_ONLY 结果 12 条，未重跑完整 8-variant 消融。", "",
                    "无论本轮结果好坏，FINAL 权重均不重新打开。下一阶段进入 external baseline / SOTA comparison。", ""])
    Path(config["docs"]["summary"]).write_text("\n".join(summary), encoding="utf-8")
    component = ["# FINAL_QDIFFCL Paper-Final Component Ablation", "",
                 "主表仅包含 Uniform、D_ONLY、E_ONLY、FINAL_DE；CURRENT_DES 作为历史 +S reference 放入 supplementary。", "",
                 "## 3W（seeds 42/43/44）", "", _component_table(components, "3W"), "",
                 "## TEP（seeds 7/42/2026）", "", _component_table(components, "TEP"), "",
                 "D_ONLY/E_ONLY 来自协议与公平性哈希一致的既有 DES 消融；未重新运行上一轮完整 8-variant 消融。", ""]
    Path(config["docs"]["component_report"]).write_text("\n".join(component), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/qdiffcl_final_5seed.yaml")
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    reliability = load_reliability(config); audit = audit_reliability(config, reliability)
    result_rows = reliability_rows(reliability); paired_rows_value, paired = paired_rows(config, reliability)
    components = component_rows(load_components(config, reliability))
    _write_csv(config["docs"]["results_csv"], result_rows); _write_csv(config["docs"]["paired_csv"], paired_rows_value)
    _write_csv(config["docs"]["component_csv"], components); write_reports(config, reliability, paired, components, audit)
    sources = {}
    for row in result_rows: sources[row["training"]] = sources.get(row["training"], 0) + 1
    print(json.dumps({"records": len(result_rows), "sources": sources, "audit": audit}, ensure_ascii=False))


if __name__ == "__main__": main()
