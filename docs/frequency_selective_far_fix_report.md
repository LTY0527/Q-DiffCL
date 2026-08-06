# 频率选择性扩散 FAR 结构保持修复报告

> **FREQUENCY_SELECTIVE_FAR_FIX / STRUCTURE_PRESERVING_SPECTRAL_NOISE / EXPLORATORY_TEP_SUBSET / NOT_FOR_PAPER_CLAIMS**

## 结论

诊断分类：`A. INTENSITY_DOMINANT`。validation 选择：`R1`，配置为 `{'noise_structure': 'iid', 't_noncritical': 5}`。Seed 7 状态：`FREQUENCY_SELECTIVE_STRUCTURE_FIX_SEED7_NO_GO`。

## R0–R3 validation 选择

| 版本 | 噪声 | t | Macro-F1 | AUPRC | Recall | FAR | Early Recall | Normal corr drift | Critical retention | Norm. L1 | Train s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 | iid | 8 | 0.9223 | 0.9818 | 0.9067 | 0.0592 | 0.8688 | 0.00608 | 0.9946 | 0.0517 | 287.9 |
| R1 | iid | 5 | 0.9225 | 0.9820 | 0.8837 | 0.0324 | 0.8375 | 0.00651 | 0.9933 | 0.0519 | 288.4 |
| R2 | correlated | 8 | 0.9465 | 0.9915 | 0.9462 | 0.0525 | 0.9313 | 0.00464 | 0.9937 | 0.0517 | 225.9 |
| R3 | correlated | 5 | 0.9085 | 0.9782 | 0.8788 | 0.0569 | 0.8375 | 0.00480 | 0.9927 | 0.0520 | 247.2 |

选择严格按 Macro-F1（0.001 容差）、FAR、Early Recall、correlation drift 的词典序进行，并先执行相对 C1 的 FAR/AUPRC/Recall 硬约束。test 未参与选择。

- R0: eligible=`False`，checks={'far_within_c1_limit': False, 'auprc_within_c1_limit': True, 'recall_within_c1_limit': True}
- R1: eligible=`True`，checks={'far_within_c1_limit': True, 'auprc_within_c1_limit': True, 'recall_within_c1_limit': True}
- R2: eligible=`False`，checks={'far_within_c1_limit': False, 'auprc_within_c1_limit': True, 'recall_within_c1_limit': True}
- R3: eligible=`False`，checks={'far_within_c1_limit': False, 'auprc_within_c1_limit': True, 'recall_within_c1_limit': False}

R2 的 validation Macro-F1 最好，但 FAR 超出 C1+0.005 的硬上限；R3 的 Recall 低于 C1-0.01。故二者均不可选择。R0/R1 均合格且 Macro-F1 差小于 0.001，按 FAR 选择 R1。

## Seed 7 外部复测

| 方法 | Macro-F1 | AUPRC | AUROC | Recall | FAR | Early Recall | Mean Delay |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.8936 | 0.9284 | 0.9118 | 0.7878 | 0.0187 | 0.7500 | 111.0 |
| C1 | 0.8895 | 0.9314 | 0.9176 | 0.8021 | 0.0375 | 0.7937 | 103.0 |
| C2 | 0.8858 | 0.9312 | 0.9174 | 0.8112 | 0.0512 | 0.8000 | 101.58823529411765 |
| C2-S | 0.8920 | 0.9316 | 0.9179 | 0.7979 | 0.0297 | 0.7937 | 103.0 |

C2 normal correlation drift=0.006078，C2-S=0.006510。

## Normal/Fault score 与翻转方向

| 方法 | Normal mean | Normal P95 | Fault mean | N→F | F→N | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 0.3818 | 0.4146 | 0.6692 | 0.0375 | 0.1979 | 0.4178 |
| C2 | 0.4133 | 0.4381 | 0.6597 | 0.0512 | 0.1888 | 0.4380 |
| C2-S | 0.4535 | 0.4678 | 0.6366 | 0.0297 | 0.2021 | 0.4707 |

完整 prefault/early/middle/stable、Fault 1–20（含 3/9/15）和逐 Run delay/miss 在诊断结果 JSON 与 `frequency_selective_far_diagnosis.md` 中。R1 降低了 normal→fault，但 fault→normal 的权衡反映在 Recall 略低于 C1。

## 结构、频带保持与增强幅度

C2-S test critical Fisher retention=1.001322，normalized L1=0.051832，finite=`True`。原始 C2 critical Fisher retention=1.002056360244751，normalized L1=0.05178430303931236。C2-S 没有退化为近似复制，但关键频带保持不弱于 C2 的门控未通过；validation normal correlation drift 也未低于 C2。

各 replay 的训练秒数列于 validation 表。旧诊断 replay 未单独持久化峰值显存，本轮没有用不可靠的事后估值补写；模型、batch size 与旧 MVP 保持一致。该缺失已在报告中显式披露，不影响门控判定。

Seed 7 门控：

- macro_f1_above_c1: `True`
- far_within_c1_limit: `True`
- auprc_within_c1_limit: `True`
- recall_within_c1_limit: `True`
- early_or_delay_improved: `False`
- correlation_drift_below_c2: `False`
- critical_retention_not_below_c2: `False`
- augmentation_not_collapsed: `True`
- finite: `True`

## 决策与边界

本轮为 `FREQUENCY_SELECTIVE_STRUCTURE_FIX_SEED7_NO_GO`：按协议停止，未运行 3 Seed，不增加 C3/C4/C5，也不再扩大 rho、t 或候选集合。唯一建议是停止该频率选择性主线，将本次结果保留为负结果；若未来要形成论文结论，只能在额外数据集、重新冻结协议或新的未触碰评测设置上重新验证。当前 test 已被多轮探索查看，结果仅是工程筛选信号，不是论文无偏结论。
