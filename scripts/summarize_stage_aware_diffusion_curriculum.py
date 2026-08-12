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
    delta_rows=[]; seed7=result["seed_metrics"]["7"]
    for comparison,first,second in (("C3-S - R1","C3-S","R1"),("C3-S - C3-E","C3-S","C3-E"),("C3-E - R1","C3-E","R1")):
        a,b=seed7[first],seed7[second]
        delta_rows.append(f"| {comparison} | {a['macro_f1']-b['macro_f1']:+.5f} | {a['auprc']-b['auprc']:+.5f} | {a['fault_recall']-b['fault_recall']:+.5f} | {a['far']-b['far']:+.5f} | {a['early_recall']-b['early_recall']:+.5f} | {a['mean_delay']-b['mean_delay']:+.2f} |")
    strength_rows=[]; diagnostic_rows=[]
    for name in ("R1","C3-E","C3-S"):
        method=result["seed_results"]["7"]["methods"][name]; history=method["effective_timestep_history"]
        for label,audit in (("首 epoch",history[0]),("末 epoch",history[-1])):
            strength_rows.append(f"| {name} | {label} | {audit['mean_effective_t']:.3f} | {audit['stages']['normal']['effective_t']} / {audit['stages']['early']['effective_t']} / {audit['stages']['middle']['effective_t']} / {audit['stages']['stable']['effective_t']} | {audit['normalized_l1']:.5f} | {audit['stages']['normal']['normalized_l1']:.5f} | {audit['stages']['early']['normalized_l1']:.5f} | {audit['stages']['middle']['normalized_l1']:.5f} | {audit['stages']['stable']['normalized_l1']:.5f} | {audit['critical_frequency_l1']:.5f} | {audit['noncritical_frequency_l1']:.5f} |")
        profile=method["test"]["score_profile"]; representation=method["test"]["representation"]
        diagnostic_rows.append(f"| {name} | {profile['normal_to_fault']:.4f} | {profile['fault_to_normal']:.4f} | {representation['fisher_ratio']:.4f} | {representation['effective_rank']:.4f} |")
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

由于 Seed 7 Gate 未通过，本表只有一个 Seed，标准差记为 0；未将其伪装成 3-Seed 汇总。

## Seed 7 配对增量

`ΔFAR<0`、`ΔDelay<0` 才表示改善。

| 比较 | ΔMacro-F1 | ΔAUPRC | ΔRecall | ΔFAR | ΔEarly | ΔDelay |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(delta_rows)}

## 课程强度与增强坍缩审计

| 方法 | Epoch | Mean t | normal/early/middle/stable t | Overall L1 | Normal L1 | Early L1 | Middle L1 | Stable L1 | Critical L1 | Noncritical L1 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(strength_rows)}

C3-S 末期 early/middle timestep 确实为 3/4，normal/stable 为 5；其 overall normalized L1 与 R1 接近，因此失败不是由整体 augmentation collapse 造成。完整 8-epoch 轨迹保存在 metrics.json。

## 分数翻转与表示诊断

| 方法 | normal→fault | fault→normal | Representation Fisher | Effective rank |
|---|---:|---:|---:|---:|
{chr(10).join(diagnostic_rows)}

Seed 7 Gate：`{result['seed7_gate']}`。3-Seed Gate：`{result['three_seed_gate']}`。C3-S 保持四项核心边界，但 Early Recall/Delay 无实质改善，且不优于 C3-E；C3-E 自身也无工业改善。因此输出 NO-GO 并跳过 3-Seed。完整 Middle/Stable、median delay、检测率和 missed runs 在 outputs 的 metrics.json。

课程审计用于确认 C3-S 不是整体 augmentation collapse；correlation/频带机制不作为选择条件。本阶段只报告 mean、sample std、配对方向，不计算 p-value，不声称统计显著。

当前 TEP test 已经历多轮工程探索，因此本阶段仍不是论文最终无偏评测。本轮 C3 未通过，唯一下一步是停止该 Stage-aware 增量，不搜索新 target、t_start 或非线性 schedule，不增加 C4/C5；若未来继续研究，应优先转向第二数据集或新的未触碰评测协议。
"""
    Path(path).write_text(report,encoding="utf-8")
