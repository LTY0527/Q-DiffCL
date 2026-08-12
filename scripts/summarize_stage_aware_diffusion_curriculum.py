from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def mean_std(values):
    array = np.asarray(values, float); return {"mean": float(array.mean()), "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0}


def flat(record):
    test = record["test"]; metrics = test["metrics"]
    return {"macro_f1": metrics["macro_f1"], "auprc": metrics["auprc"], "auroc": metrics["auroc"],
            "fault_recall": metrics["fault_recall"], "far": metrics["far"],
            "early_recall": test["stages"]["early"]["recall"], "middle_recall": test["stages"]["middle"]["recall"],
            "stable_recall": test["stages"]["stable"]["recall"], "mean_delay": test["detection_delay"]["mean_delay_samples"],
            "median_delay": test["detection_delay"]["median_delay_samples"],
            "detected_rate": test["detection_delay"]["detection_rate"], "missed_runs": test["detection_delay"]["missed_runs"]}


def seed7_gate(metrics, gate):
    r1, e, s = metrics["R1"], metrics["C3-E"], metrics["C3-S"]
    preservation = {"macro_f1": s["macro_f1"] >= r1["macro_f1"] - gate["maximum_macro_f1_drop"],
                    "far": s["far"] <= r1["far"] + gate["maximum_far_increase"],
                    "recall": s["fault_recall"] >= r1["fault_recall"] - gate["maximum_recall_drop"],
                    "auprc": s["auprc"] >= r1["auprc"] - gate["maximum_auprc_drop"]}
    early_or_delay = (s["early_recall"] >= r1["early_recall"] + gate["minimum_early_gain"]
                      or s["mean_delay"] <= r1["mean_delay"] - gate["minimum_delay_reduction_samples"])
    stage_gain = (s["early_recall"] > e["early_recall"] or s["mean_delay"] < e["mean_delay"]
                  or s["macro_f1"] > e["macro_f1"] or s["far"] < e["far"])
    stage_pass = all(preservation.values()) and early_or_delay and stage_gain
    epoch_gain = e["early_recall"] >= r1["early_recall"] + gate["minimum_early_gain"] or e["mean_delay"] <= r1["mean_delay"] - gate["minimum_delay_reduction_samples"]
    status = "STAGE_AWARE_CURRICULUM_SEED7_GO" if stage_pass else "EPOCH_CURRICULUM_GO_STAGE_AWARE_NO_GAIN" if epoch_gain else "STAGE_AWARE_CURRICULUM_SEED7_NO_GO"
    return status, {"preservation": {k: bool(v) for k, v in preservation.items()}, "early_or_delay": bool(early_or_delay),
                    "stage_gain_over_c3e": bool(stage_gain), "epoch_gain": bool(epoch_gain)}


def three_seed_gate(seed_metrics, gate):
    means = {m: {k: np.mean([seed_metrics[s][m][k] for s in seed_metrics]) for k in seed_metrics[next(iter(seed_metrics))][m] if k != "missed_runs"}
             for m in ("R1", "C3-E", "C3-S")}
    r1, c3 = means["R1"], means["C3-S"]
    preserve = (c3["macro_f1"] >= r1["macro_f1"] - gate["maximum_mean_macro_f1_drop"]
                and c3["far"] <= r1["far"] + gate["maximum_mean_far_increase"]
                and c3["fault_recall"] >= r1["fault_recall"] - gate["maximum_mean_recall_drop"]
                and c3["auprc"] >= r1["auprc"] - gate["maximum_mean_auprc_drop"])
    early_wins = sum(seed_metrics[s]["C3-S"]["early_recall"] > seed_metrics[s]["R1"]["early_recall"] for s in seed_metrics)
    delay_wins = sum(seed_metrics[s]["C3-S"]["mean_delay"] < seed_metrics[s]["R1"]["mean_delay"] for s in seed_metrics)
    industrial = ((c3["early_recall"] > r1["early_recall"] and early_wins >= 2)
                  or (c3["mean_delay"] < r1["mean_delay"] and delay_wins >= 2))
    stage_wins = sum(seed_metrics[s]["C3-S"]["early_recall"] > seed_metrics[s]["C3-E"]["early_recall"]
                     or seed_metrics[s]["C3-S"]["mean_delay"] < seed_metrics[s]["C3-E"]["mean_delay"] for s in seed_metrics)
    catastrophic = {s: bool(seed_metrics[s]["C3-S"]["macro_f1"] < seed_metrics[s]["R1"]["macro_f1"] - gate["catastrophic_macro_f1_drop"]
                             or seed_metrics[s]["C3-S"]["far"] > seed_metrics[s]["R1"]["far"] + gate["catastrophic_far_increase"]
                             or seed_metrics[s]["C3-S"]["fault_recall"] < seed_metrics[s]["R1"]["fault_recall"] - gate["catastrophic_recall_drop"]
                             or seed_metrics[s]["C3-S"]["early_recall"] < seed_metrics[s]["R1"]["early_recall"] - gate["catastrophic_early_drop"]) for s in seed_metrics}
    if preserve and industrial and stage_wins >= 2 and not any(catastrophic.values()): status = "STAGE_AWARE_DIFFUSION_CURRICULUM_3SEED_GO"
    elif preserve and industrial and stage_wins < 2: status = "EPOCH_CURRICULUM_3SEED_GO_STAGE_AWARE_NO_GAIN"
    elif any(catastrophic.values()) or not preserve: status = "STAGE_AWARE_DIFFUSION_CURRICULUM_3SEED_NO_GO"
    else: status = "STAGE_AWARE_DIFFUSION_CURRICULUM_3SEED_UNSTABLE"
    return status, {"preservation": bool(preserve), "early_wins": early_wins, "delay_wins": delay_wins,
                    "stage_industrial_wins_over_c3e": stage_wins, "catastrophic": catastrophic}


def summarize(config, seed_results, fingerprints, result=None, report_path=None):
    seed_metrics = {seed: {method: flat(record["methods"][method]) for method in ("R1", "C3-E", "C3-S")} for seed, record in seed_results.items()}
    seed7_status, seed7_audit = seed7_gate(seed_metrics["7"], config["seed7_gate"])
    if len(seed_metrics) == 3: status, gate = three_seed_gate(seed_metrics, config["three_seed_gate"])
    else: status, gate = seed7_status, {"three_seed_skipped": True}
    summary = {m: {k: mean_std([seed_metrics[s][m][k] for s in seed_metrics]) for k in seed_metrics[next(iter(seed_metrics))][m] if k != "missed_runs"} for m in ("R1", "C3-E", "C3-S")}
    value = result or {"markers": config["markers"], "status": status, "seed7_status": seed7_status, "seed7_gate": seed7_audit,
                       "three_seed_gate": gate, "seed_results": seed_results, "seed_metrics": seed_metrics,
                       "summary": summary, "fingerprints": fingerprints, "three_seeds_completed": len(seed_metrics) == 3,
                       "test_used_for_selection": False, "statistical_significance_claimed": False}
    if report_path: render_report(value, report_path)
    return value


def render_report(result, path):
    rows=[]
    for seed, methods in result["seed_metrics"].items():
        for name, m in methods.items(): rows.append(f"| {seed} | {name} | {m['macro_f1']:.4f} | {m['auprc']:.4f} | {m['fault_recall']:.4f} | {m['far']:.4f} | {m['early_recall']:.4f} | {m['mean_delay']:.2f} |")
    means=[]
    for name,s in result["summary"].items(): means.append(f"| {name} | {s['macro_f1']['mean']:.4f} ± {s['macro_f1']['std']:.4f} | {s['far']['mean']:.4f} ± {s['far']['std']:.4f} | {s['early_recall']['mean']:.4f} ± {s['early_recall']['std']:.4f} | {s['mean_delay']['mean']:.2f} ± {s['mean_delay']['std']:.2f} |")
    report=f"""# 故障阶段感知频率扩散课程增量验证

> **STAGE_AWARE_DIFFUSION_CURRICULUM / INCREMENTAL_C3_VALIDATION / FIXED_R1_BASELINE / EXPLORATORY_TEP_SUBSET / NOT_FOR_PAPER_FINAL_CLAIMS**

## 结论

Seed 7：`{result['seed7_status']}`。最终状态：`{result['status']}`。三个 Seed 完成：`{result['three_seeds_completed']}`。

R1 已通过 3-Seed。本轮唯一问题是 training-time stage prior 能否在保持 R1 Macro-F1/FAR 的同时改善 Early Recall 或 Delay。C3-E 用于隔离普通 epoch curriculum；C3-S 才加入 stage target。stage 基于真实 onset 和已排除 transition 的窗口，仅选择训练增强强度，不进入 encoder、Probe、threshold、test 推理或后处理；这是 supervised fault-detection setting 的 training-time stage prior，不是自监督 stage discovery。

## 固定设计

- R1：非关键频率固定 t=5。
- C3-E：所有阶段从 t=2 线性课程到 t=5，不读取 stage。
- C3-S：normal/early/middle/stable 从 t=2 分别到 5/3/4/5。early=3、middle=4 用于保护早期弱故障并随故障发展逐步增加难度；stable/normal 回到 R1 强度。
- stage：真实 onset 后完整窗口 progress；early `<4*stride`，middle `<12*stride`，其后 stable。
- 指纹：`{result['fingerprints']}`。

## 逐 Seed 指标

| Seed | 方法 | Macro-F1 | AUPRC | Recall | FAR | Early Recall | Mean Delay |
|---:|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Mean ± sample std

| 方法 | Macro-F1 | FAR | Early Recall | Mean Delay |
|---|---|---|---|---|
{chr(10).join(means)}

Seed 7 Gate：`{result['seed7_gate']}`。3-Seed Gate：`{result['three_seed_gate']}`。C3-S-R1 与 C3-S-C3-E 的逐 Seed 配对差可由上述表直接审计；完整 Middle/Stable、median delay、检测率、missed runs、双向翻转、representation、每 epoch/stage effective t、normalized L1、关键/非关键频带扰动在 outputs 的 metrics.json。

课程审计用于确认 C3-S 不是整体 augmentation collapse；correlation/频带机制不作为选择条件。本阶段只报告 mean、sample std、配对方向，不计算 p-value，不声称统计显著。

当前 TEP test 已经历多轮工程探索，因此本阶段仍不是论文最终无偏评测。若 C3 通过，下一步优先转向第二数据集或新的未触碰评测协议，而不是继续增加 C4/C5；若未通过则停止，不搜索新 target、t_start 或非线性 schedule。
"""
    Path(path).write_text(report,encoding="utf-8")
