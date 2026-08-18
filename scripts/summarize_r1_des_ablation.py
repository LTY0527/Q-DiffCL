from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


VARIANT_ORDER = ("UNIFORM", "W/O_D", "W/O_E", "W/O_S", "FULL_DES", "D_ONLY", "E_ONLY", "S_ONLY")
MAIN_VARIANTS = VARIANT_ORDER[:5]
SUPPLEMENTARY_VARIANTS = ("D_ONLY", "E_ONLY", "S_ONLY", "FULL_DES")
THREE_W_METRICS = (
    "macro_f1", "recall_macro", "auprc_fault_vs_normal", "auprc_multiclass_macro",
    "far", "early_recall", "mean_detection_delay_seconds", "class_2_recall",
    "class_8_recall", "class_9_recall", "class_2_f1", "class_8_f1", "class_9_f1",
)
TEP_METRICS = (
    "macro_f1", "auprc", "fault_recall", "far", "early_recall",
    "mean_detection_delay_samples", "detected_rate", "missed_runs",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _three_w_metrics(method: dict[str, Any]) -> dict[str, float]:
    metrics = method["metrics"]
    result = {key: float(metrics[key]) for key in THREE_W_METRICS[:7]}
    by_class = {int(item["original_class"]): item for item in metrics["per_class"]}
    for class_id in (2, 8, 9):
        result[f"class_{class_id}_recall"] = float(by_class[class_id]["recall"])
        result[f"class_{class_id}_f1"] = float(by_class[class_id]["f1"])
    return result


def _tep_metrics(method: dict[str, Any]) -> dict[str, float]:
    test = method["test"]
    metrics, delay = test["metrics"], test["detection_delay"]
    return {
        "macro_f1": float(metrics["macro_f1"]),
        "auprc": float(metrics["auprc"]),
        "fault_recall": float(metrics["fault_recall"]),
        "far": float(metrics["far"]),
        "early_recall": float(test["early_fault"]["recall"]),
        "mean_detection_delay_samples": float(delay["mean_delay_samples"]),
        "detected_rate": float(delay["detection_rate"]),
        "missed_runs": float(delay["missed_runs"]),
    }


def load_results(root: Path) -> dict[str, dict[str, dict[int, dict[str, float]]]]:
    results: dict[str, dict[str, dict[int, dict[str, float]]]] = {
        "3W": defaultdict(dict), "TEP": defaultdict(dict)
    }
    three_baseline = _read(root / "outputs/3w_diffusion_3seed/result_manifest.json")
    for seed_text, seed_record in three_baseline["seed_results"].items():
        run = _read(root / Path(seed_record["result_path"]))
        results["3W"]["UNIFORM"][int(seed_text)] = _three_w_metrics(run["methods"]["UNIFORM_DIFFUSION"])
        results["3W"]["FULL_DES"][int(seed_text)] = _three_w_metrics(run["methods"]["FREQUENCY_SELECTIVE_R1"])
    tep_baseline = _read(root / "outputs/frequency_selective_r1_3seed/result.json")
    for seed_text, seed_record in tep_baseline["seed_results"].items():
        results["TEP"]["UNIFORM"][int(seed_text)] = _tep_metrics(seed_record["methods"]["C1"])
        results["TEP"]["FULL_DES"][int(seed_text)] = _tep_metrics(seed_record["methods"]["R1"])
    for dataset, parser, path in (
        ("3W", _three_w_metrics, root / "outputs/r1_des_ablation/3w/manifest.json"),
        ("TEP", _tep_metrics, root / "outputs/r1_des_ablation/tep/manifest.json"),
    ):
        manifest = _read(path)
        for record in manifest["results"].values():
            results[dataset][record["variant"]][int(record["seed"])] = parser(record["method"])
    return results


def validate_complete(results: dict[str, Any]) -> None:
    expected = {"3W": {42, 43, 44}, "TEP": {7, 42, 2026}}
    for dataset in ("3W", "TEP"):
        for variant in VARIANT_ORDER:
            actual = set(results[dataset].get(variant, {}))
            if actual != expected[dataset]:
                raise RuntimeError(f"Incomplete {dataset} {variant}: expected {sorted(expected[dataset])}, got {sorted(actual)}")


def _summary(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def _fmt(values: list[float]) -> str:
    avg, std = _summary(values)
    return f"{avg:.4f} ± {std:.4f}"


def build_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for dataset in ("3W", "TEP"):
        metrics = THREE_W_METRICS if dataset == "3W" else TEP_METRICS
        full, uniform = results[dataset]["FULL_DES"], results[dataset]["UNIFORM"]
        for variant in VARIANT_ORDER:
            for seed, values in sorted(results[dataset][variant].items()):
                for metric in metrics:
                    value = values[metric]
                    rows.append({
                        "dataset": dataset, "variant": variant, "seed": seed, "metric": metric,
                        "value": value, "delta_vs_full": value - full[seed][metric],
                        "delta_vs_uniform": value - uniform[seed][metric],
                    })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _table_rows(results: dict[str, Any], variants: tuple[str, ...]) -> list[dict[str, str]]:
    columns = {
        "3W": ("macro_f1", "far", "early_recall"),
        "TEP": ("macro_f1", "far", "early_recall"),
    }
    rows = []
    for variant in variants:
        row = {"variant": variant}
        for dataset, metrics in columns.items():
            for metric in metrics:
                row[f"{dataset.lower()}_{metric}"] = _fmt([v[metric] for v in results[dataset][variant].values()])
        rows.append(row)
    return rows


def _markdown_table(rows: list[dict[str, str]]) -> str:
    fields = list(rows[0])
    labels = [field.replace("_", " ") for field in fields]
    lines = ["| " + " | ".join(labels) + " |", "|" + "|".join("---" for _ in fields) + "|"]
    lines.extend("| " + " | ".join(str(row[field]) for field in fields) + " |" for row in rows)
    return "\n".join(lines)


def _dataset_report(dataset: str, results: dict[str, Any]) -> str:
    metrics = THREE_W_METRICS if dataset == "3W" else TEP_METRICS
    lines = [f"# R1 DES {dataset} 正式消融", "", "所有值均来自冻结协议下的 3 个模型种子；std 为样本标准差。", ""]
    for metric in metrics:
        lines.extend([f"## {metric}", "", "| Variant | mean ± std | paired Δ vs Full | paired Δ vs Uniform |", "|---|---:|---:|---:|"])
        full, uniform = results[dataset]["FULL_DES"], results[dataset]["UNIFORM"]
        for variant in VARIANT_ORDER:
            values = results[dataset][variant]
            delta_full = mean(values[s][metric] - full[s][metric] for s in values)
            delta_uniform = mean(values[s][metric] - uniform[s][metric] for s in values)
            lines.append(f"| {variant} | {_fmt([v[metric] for v in values.values()])} | {delta_full:+.4f} | {delta_uniform:+.4f} |")
        lines.append("")
    lines.extend(["## 每 seed 明细", "", "完整的每 seed 数值及 paired delta 见 `r1_des_ablation_results.csv`。", ""])
    return "\n".join(lines)


def _effect_sentence(results: dict[str, Any], variant: str, label: str) -> str:
    parts = []
    for dataset in ("3W", "TEP"):
        full = results[dataset]["FULL_DES"]
        current = results[dataset][variant]
        for metric in ("macro_f1", "far", "early_recall"):
            delta = mean(current[s][metric] - full[s][metric] for s in current)
            parts.append(f"{dataset} {metric} {delta:+.4f}")
    return f"- 删除 {label}（{variant}）相对 Full 的 paired mean：" + "；".join(parts) + "。"


def write_outputs(root: Path, results: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    docs = root / "docs"
    docs.mkdir(exist_ok=True)
    fields = ["dataset", "variant", "seed", "metric", "value", "delta_vs_full", "delta_vs_uniform"]
    _write_csv(docs / "r1_des_ablation_results.csv", rows, fields)
    main_rows = _table_rows(results, MAIN_VARIANTS)
    supp_rows = _table_rows(results, SUPPLEMENTARY_VARIANTS)
    table_fields = list(main_rows[0])
    _write_csv(docs / "r1_des_ablation_main_table.csv", main_rows, table_fields)
    _write_csv(docs / "r1_des_ablation_supplementary.csv", supp_rows, table_fields)
    (docs / "r1_des_ablation_3w.md").write_text(_dataset_report("3W", results), encoding="utf-8")
    (docs / "r1_des_ablation_tep.md").write_text(_dataset_report("TEP", results), encoding="utf-8")
    audit = _read(docs / "r1_des_mask_audit.json")
    audit_lines = []
    for dataset in ("3W", "TEP"):
        audit_lines.extend([f"### {dataset}", "", "| Variant | Jaccard vs Full | Changed bins | Budget error |", "|---|---:|---:|---:|"])
        for variant in VARIANT_ORDER[1:]:
            item = audit[dataset][variant]
            audit_lines.append(f"| {variant} | {item['hard_mask_jaccard_full']:.4f} | {item['changed_bins_from_full']} | {item['total_budget_error']:.3e} |")
        audit_lines.append("")
    summary = [
        "# R1 DES 双数据集正式消融总结", "",
        "## 主消融表", "", _markdown_table(main_rows), "",
        "## 单分量补充表", "", _markdown_table(supp_rows), "",
        "## 删除分量的观测影响", "",
        _effect_sentence(results, "W/O_D", "D"),
        _effect_sentence(results, "W/O_E", "E"),
        _effect_sentence(results, "W/O_S", "S"), "",
        "## 冻结实验判读", "",
        "- **D：未得到‘删除即退化’的必要性证据。** W/O_D 在 3W 的 Macro-F1、FAR、Early Recall 均值优于 Full，但 FAR std 从 0.0984 增至 0.1726；TEP 变化接近零。D 明显改变了频率选择，却不能由本轮结果宣称为不可缺少。",
        "- **E：支持 3W 整体判别与 FAR，但没有验证预期的早期优势。** W/O_E 相对 Full 的 3W Macro-F1 降低 0.0375、FAR 增加 0.0776；然而 Early Recall 基本不变，平均检测延迟反而由 503.63 秒降至 151.89 秒。TEP 同样没有出现清晰的 Early 退化。",
        "- **S：表现为指标权衡，稳定性假设未获直接支持。** W/O_S 的 3W Macro-F1 与 Early Recall 更高，但 FAR 更差且检测延迟升至 1143.96 秒；其 Macro-F1 std 反而小于 Full。TEP 三项主指标近似持平。",
        "- **单分量结果：D_ONLY 最强，S_ONLY 最弱。** D_ONLY 在两数据集的主检测指标上具有竞争力；S_ONLY 的 Macro-F1/FAR 明显较差，说明单独依靠稳定性统计不足以支撑选择性扩散。",
        "- **Full D+E+S 并非本轮双数据集上最均衡的唯一或一致最优方案。** Full 在 TEP 上保持稳定且有竞争力，但 3W 被 W/O_D 或 D_ONLY 在多项均值上超过。本轮因此不能声称三个分量均为必要，也不能声称 Full 稳定优于所有消融。",
        "",
        "以上是冻结权重、冻结协议的描述性结论，不进行显著性外推，不据此修改权重或提出 R1-v2。", "",
        "## Mask 与预算审计", "", *audit_lines,
        "删除 D/E/S 后，3W hard mask 分别改变 28/22/82 个 bins，TEP 分别改变 140/90/62 个 bins。3W 删除 D 的最大 composite 变化集中于 `(channel, frequency_bin)=(3,19),(6,3),(6,9),(6,15),(6,4)`；TEP 对应为 `(2,7),(7,18),(21,22),(26,4),(40,2)`。", "",
        "各变体均在 canonical train split 上独立重建一次 mask，并跨模型 seed 冻结；timestep 保持 `t_critical=1`、`t_noncritical=5`，总频谱噪声预算与 Uniform `t=3` 匹配，最大误差仅 3.725e-09。完整 component/composite/soft mask、timestep map 和最大变化 bins 见 `r1_des_mask_audit.json` 及 `outputs/r1_des_ablation/masks/`。", "",
        "## 完整性", "",
        "Stage A 与 Stage B 均完整覆盖 3W 和 TEP 的 3 seeds：新增训练 36 runs；Uniform 与 Full 基线复用 12 runs。", "",
    ]
    (docs / "r1_des_ablation_summary.md").write_text("\n".join(summary), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    results = load_results(root)
    validate_complete(results)
    rows = build_rows(results)
    write_outputs(root, results, rows)
    print(json.dumps({"status": "complete", "per_seed_rows": len(rows), "new_training_runs": 36, "reused_baseline_runs": 12}, ensure_ascii=False))


if __name__ == "__main__":
    main()
