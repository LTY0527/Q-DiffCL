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
            f"{value['validation_structure']['normal']['corr_drift']:.5f} | "
            f"{value['validation_audit']['critical_fisher_retention']:.4f} | "
            f"{value['validation_audit']['time_normalized_l1']:.4f} | {value['training_seconds']:.1f} |"
        )
    seed_rows = []
    for name in ("C0", "C1", "C2", "C2-S"):
        value = result["seed7"][name]; metrics = value["metrics"]
        seed_rows.append(f"| {name} | {metrics['macro_f1']:.4f} | {metrics['auprc']:.4f} | {metrics['auroc']:.4f} | "
                         f"{metrics['fault_recall']:.4f} | {metrics['far']:.4f} | {value['early_fault']['recall']:.4f} | "
                         f"{value['detection_delay']['mean_delay_samples']} |")
    checks = "\n".join(f"- {key}: `{value}`" for key, value in result["seed7_gate_checks"].items())
    eligibility = "\n".join(
        f"- {name}: eligible=`{record['eligible']}`，checks={record['checks']}"
        for name, record in result["selection"]["decisions"].items()
    )
    score_rows = []
    for name in ("C1", "C2", "C2-S"):
        profile = result["seed7"][name].get("score_profile")
        if profile is not None:
            score_rows.append(f"| {name} | {profile['normal']['mean']:.4f} | {profile['normal']['p95']:.4f} | "
                              f"{profile['fault']['mean']:.4f} | {profile['normal_to_fault']:.4f} | "
                              f"{profile['fault_to_normal']:.4f} | {profile['threshold']:.4f} |")
    c2_audit = result.get("c2_test_audit", {})
    c2s_audit = result["c2s_test_audit"]
    report = f"""# 频率选择性扩散 FAR 结构保持修复报告

> **FREQUENCY_SELECTIVE_FAR_FIX / STRUCTURE_PRESERVING_SPECTRAL_NOISE / EXPLORATORY_TEP_SUBSET / NOT_FOR_PAPER_CLAIMS**

## 结论

诊断分类：`{result['cause_category']}`。validation 选择：`{result['selected_variant']}`，配置为 `{result['selected_specification']}`。Seed 7 状态：`{result['status']}`。

## R0–R3 validation 选择

| 版本 | 噪声 | t | Macro-F1 | AUPRC | Recall | FAR | Early Recall | Normal corr drift | Critical retention | Norm. L1 | Train s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(validation_rows)}

选择严格按 Macro-F1（0.001 容差）、FAR、Early Recall、correlation drift 的词典序进行，并先执行相对 C1 的 FAR/AUPRC/Recall 硬约束。test 未参与选择。

{eligibility}

R2 的 validation Macro-F1 最好，但 FAR 超出 C1+0.005 的硬上限；R3 的 Recall 低于 C1-0.01。故二者均不可选择。R0/R1 均合格且 Macro-F1 差小于 0.001，按 FAR 选择 R1。

## Seed 7 外部复测

| 方法 | Macro-F1 | AUPRC | AUROC | Recall | FAR | Early Recall | Mean Delay |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(seed_rows)}

C2 normal correlation drift={result['c2_validation_normal_corr_drift']:.6f}，C2-S={result['c2s_validation_normal_corr_drift']:.6f}。

## Normal/Fault score 与翻转方向

| 方法 | Normal mean | Normal P95 | Fault mean | N→F | F→N | Threshold |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(score_rows)}

完整 prefault/early/middle/stable、Fault 1–20（含 3/9/15）和逐 Run delay/miss 在诊断结果 JSON 与 `frequency_selective_far_diagnosis.md` 中。R1 降低了 normal→fault，但 fault→normal 的权衡反映在 Recall 略低于 C1。

## 结构、频带保持与增强幅度

C2-S test critical Fisher retention={c2s_audit['critical_fisher_retention']:.6f}，normalized L1={c2s_audit['time_normalized_l1']:.6f}，finite=`{c2s_audit['finite']}`。原始 C2 critical Fisher retention={c2_audit.get('critical_fisher_retention')}，normalized L1={c2_audit.get('time_normalized_l1')}。C2-S 没有退化为近似复制，但关键频带保持不弱于 C2 的门控未通过；validation normal correlation drift 也未低于 C2。

各 replay 的训练秒数列于 validation 表。旧诊断 replay 未单独持久化峰值显存，本轮没有用不可靠的事后估值补写；模型、batch size 与旧 MVP 保持一致。该缺失已在报告中显式披露，不影响门控判定。

Seed 7 门控：

{checks}

## 决策与边界

本轮为 `FREQUENCY_SELECTIVE_STRUCTURE_FIX_SEED7_NO_GO`：按协议停止，未运行 3 Seed，不增加 C3/C4/C5，也不再扩大 rho、t 或候选集合。唯一建议是停止该频率选择性主线，将本次结果保留为负结果；若未来要形成论文结论，只能在额外数据集、重新冻结协议或新的未触碰评测设置上重新验证。当前 test 已被多轮探索查看，结果仅是工程筛选信号，不是论文无偏结论。
"""
    Path(path).write_text(report, encoding="utf-8")
