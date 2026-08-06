from __future__ import annotations

from pathlib import Path
from typing import Any


def summarize(result: dict[str, Any], path: str) -> None:
    validation_rows = []
    for name in ("R0", "R1", "R2", "R3"):
        value = result["validation_variants"][name]
        metrics = value["validation"]["metrics"]
        validation_rows.append(
            f"| {name} | {value['specification']['noise_structure']} | {value['specification']['t_noncritical']} | "
            f"{metrics['macro_f1']:.4f} | {metrics['auprc']:.4f} | {metrics['fault_recall']:.4f} | "
            f"{metrics['far']:.4f} | {value['validation']['early_fault']['recall']:.4f} | "
            f"{value['validation_structure']['normal']['corr_drift']:.5f} |"
        )
    seed_rows = []
    for name in ("C0", "C1", "C2", "C2-S"):
        value = result["seed7"][name]; metrics = value["metrics"]
        seed_rows.append(f"| {name} | {metrics['macro_f1']:.4f} | {metrics['auprc']:.4f} | {metrics['auroc']:.4f} | "
                         f"{metrics['fault_recall']:.4f} | {metrics['far']:.4f} | {value['early_fault']['recall']:.4f} | "
                         f"{value['detection_delay']['mean_delay_samples']} |")
    checks = "\n".join(f"- {key}: `{value}`" for key, value in result["seed7_gate_checks"].items())
    report = f"""# 频率选择性扩散 FAR 结构保持修复报告

> **FREQUENCY_SELECTIVE_FAR_FIX / STRUCTURE_PRESERVING_SPECTRAL_NOISE / EXPLORATORY_TEP_SUBSET / NOT_FOR_PAPER_CLAIMS**

## 结论

诊断分类：`{result['cause_category']}`。validation 选择：`{result['selected_variant']}`，配置为 `{result['selected_specification']}`。Seed 7 状态：`{result['status']}`。

## R0–R3 validation 选择

| 版本 | 噪声 | t | Macro-F1 | AUPRC | Recall | FAR | Early Recall | Normal corr drift |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(validation_rows)}

选择严格按 Macro-F1（0.001 容差）、FAR、Early Recall、correlation drift 的词典序进行，并先执行相对 C1 的 FAR/AUPRC/Recall 硬约束。test 未参与选择。

## Seed 7 外部复测

| 方法 | Macro-F1 | AUPRC | AUROC | Recall | FAR | Early Recall | Mean Delay |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(seed_rows)}

C2 normal correlation drift={result['c2_validation_normal_corr_drift']:.6f}，C2-S={result['c2s_validation_normal_corr_drift']:.6f}。

Seed 7 门控：

{checks}

## 决策与边界

若状态为 `FREQUENCY_SELECTIVE_STRUCTURE_FIX_SEED7_NO_GO`，按协议停止，不运行 3 Seed，不增加 C3/C4/C5。若为 GO，才允许冻结同一版本进入 3 Seed。当前 test 已被多轮探索查看，结果仅是工程筛选信号，不是论文无偏结论；论文阶段必须使用额外数据集、重新冻结协议或新的未触碰评测设置。
"""
    Path(path).write_text(report, encoding="utf-8")
