from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


CORE_METRICS = ("macro_f1", "auprc", "fault_recall", "far", "early_recall", "mean_delay")


def mean_sample_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1))}


def _flat(record: dict[str, Any]) -> dict[str, float]:
    test = record["test"]
    return {**{key: float(test["metrics"][key]) for key in
               ("accuracy", "macro_f1", "auprc", "auroc", "fault_recall", "far")},
            "early_recall": float(test["stages"]["early"]["recall"]),
            "middle_recall": float(test["stages"]["middle"]["recall"]),
            "stable_recall": float(test["stages"]["stable"]["recall"]),
            "mean_delay": float(test["detection_delay"]["mean_delay_samples"]),
            "median_delay": float(test["detection_delay"]["median_delay_samples"]),
            "detected_run_rate": float(test["detection_delay"]["detection_rate"]),
            "missed_fault_runs": int(test["detection_delay"]["missed_runs"])}


def evaluate_gate(seed_metrics: dict[str, dict[str, dict[str, float]]], gate: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    deltas = {seed: {key: seed_metrics[seed]["R1"][key] - seed_metrics[seed]["C1"][key]
                     for key in CORE_METRICS} for seed in seed_metrics}
    c1_mean = {key: np.mean([seed_metrics[seed]["C1"][key] for seed in seed_metrics]) for key in CORE_METRICS}
    r1_mean = {key: np.mean([seed_metrics[seed]["R1"][key] for seed in seed_metrics]) for key in CORE_METRICS}
    macro_wins = sum(delta["macro_f1"] > 0 for delta in deltas.values())
    far_wins = sum(delta["far"] < 0 for delta in deltas.values())
    early_preserved = sum(delta["early_recall"] >= -float(gate["maximum_early_seed_drop"])
                          for delta in deltas.values())
    catastrophic = {seed: (delta["macro_f1"] < -float(gate["catastrophic_macro_f1_drop"])
                           or delta["far"] > float(gate["catastrophic_far_increase"])
                           or delta["fault_recall"] < -float(gate["catastrophic_recall_drop"])
                           or delta["early_recall"] < -float(gate["catastrophic_early_recall_drop"]))
                    for seed, delta in deltas.items()}
    checks = {
        "mean_macro_f1_above_c1": r1_mean["macro_f1"] > c1_mean["macro_f1"],
        "macro_f1_wins_at_least_2": macro_wins >= int(gate["minimum_macro_f1_wins"]),
        "mean_far_below_c1": r1_mean["far"] < c1_mean["far"],
        "far_wins_at_least_2": far_wins >= int(gate["minimum_far_wins"]),
        "mean_recall_preserved": r1_mean["fault_recall"] >= c1_mean["fault_recall"] - float(gate["maximum_mean_recall_drop"]),
        "no_single_recall_drop_over_limit": all(delta["fault_recall"] >= -float(gate["maximum_single_recall_drop"])
                                                 for delta in deltas.values()),
        "mean_auprc_preserved": r1_mean["auprc"] >= c1_mean["auprc"] - float(gate["maximum_mean_auprc_drop"]),
        "mean_early_recall_preserved": r1_mean["early_recall"] >= c1_mean["early_recall"] - float(gate["maximum_mean_early_recall_drop"]),
        "early_preserved_at_least_2": early_preserved >= int(gate["minimum_early_preserved_seeds"]),
        "mean_delay_within_one_stride": r1_mean["mean_delay"] <= c1_mean["mean_delay"] + float(gate["maximum_mean_delay_increase_samples"]),
        "no_catastrophic_seed": not any(catastrophic.values()),
    }
    all_core = all(checks.values())
    outright_failure = (not checks["mean_macro_f1_above_c1"] or not checks["mean_far_below_c1"]
                        or not checks["mean_recall_preserved"] or not checks["mean_early_recall_preserved"]
                        or not checks["no_catastrophic_seed"])
    status = ("FREQUENCY_SELECTIVE_R1_3SEED_GO" if all_core else
              "FREQUENCY_SELECTIVE_R1_3SEED_NO_GO" if outright_failure else
              "FREQUENCY_SELECTIVE_R1_3SEED_UNSTABLE")
    return status, {"checks": checks, "deltas": deltas, "macro_f1_win_count": macro_wins,
                    "far_win_count": far_wins, "early_preserved_count": early_preserved,
                    "catastrophic_by_seed": catastrophic}


def summarize(config: dict[str, Any], seed_results: dict[str, Any], fingerprints: dict[str, str],
              result: dict[str, Any] | None = None, report_path: str | None = None) -> dict[str, Any]:
    seed_metrics = {seed: {method: _flat(value["methods"][method]) for method in ("C0", "C1", "R1")}
                    for seed, value in seed_results.items()}
    summary = {method: {key: mean_sample_std([seed_metrics[seed][method][key] for seed in seed_metrics])
                        for key in seed_metrics[next(iter(seed_metrics))][method] if key != "missed_fault_runs"}
               for method in ("C0", "C1", "R1")}
    summary["R1-C1"] = {key: mean_sample_std([seed_metrics[seed]["R1"][key] - seed_metrics[seed]["C1"][key]
                                              for seed in seed_metrics]) for key in CORE_METRICS}
    summary["R1-C0"] = {key: mean_sample_std([seed_metrics[seed]["R1"][key] - seed_metrics[seed]["C0"][key]
                                              for seed in seed_metrics]) for key in CORE_METRICS}
    status, gate_audit = evaluate_gate(seed_metrics, config["gate"])
    c0_warning = (summary["R1"]["macro_f1"]["mean"] < summary["C0"]["macro_f1"]["mean"] - float(config["gate"]["c0_macro_f1_margin"])
                  or summary["R1"]["far"]["mean"] > summary["C0"]["far"]["mean"] + float(config["gate"]["c0_far_margin"]))
    value = result or {"markers": config["markers"], "status": status, "seeds": list(map(int, config["seeds"])),
                       "seed7_reused": bool(config["seed7_reuse"]), "seed7_reuse_reason": config["seed7_reuse_reason"],
                       "fingerprints": fingerprints, "seed_results": seed_results, "seed_metrics": seed_metrics,
                       "summary": summary, "gate": gate_audit, "c0_auxiliary_warning": c0_warning,
                       "c0_auxiliary_status": ("R1_BEATS_UNIFORM_DIFFUSION_BUT_NOT_TRADITIONAL_AUGMENTATION"
                                               if status == "FREQUENCY_SELECTIVE_R1_3SEED_GO" and c0_warning else None),
                       "three_seeds_completed": len(seed_results) == 3, "test_used_for_tuning_or_selection": False,
                       "statistical_significance_claimed": False,
                       "c3_allowed": status == "FREQUENCY_SELECTIVE_R1_3SEED_GO"}
    if report_path is not None:
        render_report(value, config, report_path)
    return value


def _pm(record: dict[str, float]) -> str:
    return f"{record['mean']:.4f} ± {record['std']:.4f}"


def render_report(result: dict[str, Any], config: dict[str, Any], path: str) -> None:
    seed_rows = []
    for seed in map(str, config["seeds"]):
        for method in ("C0", "C1", "R1"):
            m = result["seed_metrics"][seed][method]
            seed_rows.append(f"| {seed} | {method} | {m['macro_f1']:.4f} | {m['auprc']:.4f} | {m['fault_recall']:.4f} | "
                             f"{m['far']:.4f} | {m['early_recall']:.4f} | {m['middle_recall']:.4f} | "
                             f"{m['stable_recall']:.4f} | {m['mean_delay']:.2f} | {m['detected_run_rate']:.4f} | {m['missed_fault_runs']} |")
    mean_rows = []
    for method in ("C0", "C1", "R1"):
        s = result["summary"][method]
        mean_rows.append(f"| {method} | {_pm(s['macro_f1'])} | {_pm(s['auprc'])} | {_pm(s['fault_recall'])} | "
                         f"{_pm(s['far'])} | {_pm(s['early_recall'])} | {_pm(s['mean_delay'])} |")
    delta_rows = []
    for seed in map(str, config["seeds"]):
        for comparison, baseline in (("R1-C1", "C1"), ("R1-C0", "C0")):
            r1, other = result["seed_metrics"][seed]["R1"], result["seed_metrics"][seed][baseline]
            delta_rows.append(f"| {seed} | {comparison} | {r1['macro_f1']-other['macro_f1']:+.4f} | "
                              f"{r1['auprc']-other['auprc']:+.4f} | {r1['fault_recall']-other['fault_recall']:+.4f} | "
                              f"{r1['far']-other['far']:+.4f} | {r1['early_recall']-other['early_recall']:+.4f} | "
                              f"{r1['mean_delay']-other['mean_delay']:+.2f} |")
    checks = "\n".join(f"- {key}: `{value}`" for key, value in result["gate"]["checks"].items())
    fingerprints = "\n".join(f"- {key}: `{value}`" for key, value in result["fingerprints"].items())
    report = f"""# R1 频率选择性扩散 3-Seed 稳定性复核

> **FREQUENCY_SELECTIVE_R1_3SEED_VALIDATION / FIXED_R1_CONFIG / EXPLORATORY_TEP_SUBSET / NOT_FOR_PAPER_FINAL_CLAIMS**

## 结论

最终状态：`{result['status']}`。三个 Seed 全部完成：`{result['three_seeds_completed']}`。允许进入 C3：`{result['c3_allowed']}`。

上一阶段 NO-GO 混合了 Early Recall/Delay、correlation drift 和相对原始 C2 的 critical retention 硬条件，但 FAR 诊断已证明主要原因为 `A. INTENSITY_DOMINANT`。冻结 R1（iid，`t_noncritical=5`）在 Seed 7 相对 C1 呈现 Macro-F1 与 FAR 弱正向信号，因此本轮只验证该方向能否跨 Seed 稳定复现，不继续优化参数。

Seed 7 reused=`{str(result['seed7_reused']).lower()}`；reason={result['seed7_reuse_reason']}。

## 冻结配置与指纹

- Seeds：`[7, 42, 2026]`
- 方法：C0 传统 jitter+scaling、C1 uniform iid `t=3`、R1 selective iid `t_critical=1/t_noncritical=5`
- phase/DC：保持；C1/R1 总频谱噪声预算一致
- D/E/S=`0.5/0.3/0.2`，critical ratio=`0.30`，三个 Seed 共享同一 composite mask
- window/stride=`64/16`，MCAR=`0.30`，固定 manifest，threshold 仅由 validation 选择

{fingerprints}

## 逐 Seed 结果

| Seed | 方法 | Macro-F1 | AUPRC | Recall | FAR | Early | Middle | Stable | Mean Delay | Detected Rate | Missed Runs |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(seed_rows)}

每个方法的 Accuracy/AUROC、median delay、validation threshold、normal→fault/fault→normal、normal score mean/P95、fault score mean、representation Fisher/class-center/effective-rank、correlation drift、critical retention、normalized L1、预算、训练历史与 Probe 历史保存在对应 `metrics.json`。correlation drift 与 critical retention 只作机制诊断，不参与本轮 Gate。

## Mean ± sample std（ddof=1）

| 方法 | Macro-F1 | AUPRC | Recall | FAR | Early Recall | Mean Delay |
|---|---|---|---|---|---|---|
{chr(10).join(mean_rows)}

## 配对差值

`ΔFAR<0`、`ΔDelay<0` 表示改善。

| Seed | 比较 | ΔMacro-F1 | ΔAUPRC | ΔRecall | ΔFAR | ΔEarly | ΔDelay |
|---:|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(delta_rows)}

R1 相对 C1：Macro-F1 胜 `{result['gate']['macro_f1_win_count']}/3`，FAR 胜 `{result['gate']['far_win_count']}/3`，Early Recall 不下降超过 1pp 为 `{result['gate']['early_preserved_count']}/3`。灾难性 Seed：`{result['gate']['catastrophic_by_seed']}`。

## Gate

{checks}

C0 辅助警告：`{result['c0_auxiliary_warning']}`；辅助状态：`{result['c0_auxiliary_status']}`。C0 不是本轮 R1 vs C1 的主要因果对照，但若 R1 明显逊于 C0，不得据此进入论文主张。

## 统计与下一步边界

本轮只有 3 个 Seed，只报告 mean、sample std、配对方向与胜率，不计算 p-value，也不声称统计显著。test 未用于参数、epoch、threshold、方法或 Gate 选择；但当前 TEP test 已在之前多轮工程探索中被查看，本轮仍是探索性验证，不是论文最终无偏评测。论文级实验必须使用第二数据集或新的未触碰评测协议。

若状态为 GO，唯一下一步是冻结 R1 为新频率选择性基线，另行增量验证 C3；本轮未实现 C3。若为 UNSTABLE/NO-GO，则停止，不搜索 t=4/6，不添加 C3/C4/C5。
"""
    Path(path).write_text(report, encoding="utf-8")
