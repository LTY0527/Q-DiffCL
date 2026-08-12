# R1 频率选择性扩散 3-Seed 稳定性复核

> **FREQUENCY_SELECTIVE_R1_3SEED_VALIDATION / FIXED_R1_CONFIG / EXPLORATORY_TEP_SUBSET / NOT_FOR_PAPER_FINAL_CLAIMS**

## 结论

最终状态：`FREQUENCY_SELECTIVE_R1_3SEED_GO`。三个 Seed 全部完成：`True`。允许进入 C3：`True`。

上一阶段 NO-GO 混合了 Early Recall/Delay、correlation drift 和相对原始 C2 的 critical retention 硬条件，但 FAR 诊断已证明主要原因为 `A. INTENSITY_DOMINANT`。冻结 R1（iid，`t_noncritical=5`）在 Seed 7 相对 C1 呈现 Macro-F1 与 FAR 弱正向信号，因此本轮只验证该方向能否跨 Seed 稳定复现，不继续优化参数。

Seed 7 reused=`false`；reason=旧结果缺少完整训练代码指纹、同轮C0/C1/R1公平性元数据和分阶段test诊断，按冻结协议重新运行。

## 冻结配置与指纹

- Seeds：`[7, 42, 2026]`
- 方法：C0 传统 jitter+scaling、C1 uniform iid `t=3`、R1 selective iid `t_critical=1/t_noncritical=5`
- phase/DC：保持；C1/R1 总频谱噪声预算一致
- D/E/S=`0.5/0.3/0.2`，critical ratio=`0.30`，三个 Seed 共享同一 composite mask
- window/stride=`64/16`，MCAR=`0.30`，固定 manifest，threshold 仅由 validation 选择

- manifest_sha256: `1824e2cfa0b86ef71afe2d38913134ea418d9d7dda5bbf9e624a496faff88eb1`
- mask_sha256: `d2e1879bc012ac1326ea8c721461d31641105af2c4fe0c89eacdebac413b9395`
- frozen_config_sha256: `91d924dc2094adadf4cc99d2e97bb7dd4479de7fa0a1148be6cc33ada146883a`
- training_code_sha256: `11810e369de396f9040079dec863c6cc3c690a174533fbffd6dc1cae3e8d9b54`

## 逐 Seed 结果

| Seed | 方法 | Macro-F1 | AUPRC | Recall | FAR | Early | Middle | Stable | Mean Delay | Detected Rate | Missed Runs |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | C0 | 0.8936 | 0.9284 | 0.7878 | 0.0187 | 0.7500 | 0.8000 | 0.7893 | 111.00 | 0.8500 | 6 |
| 7 | C1 | 0.8895 | 0.9314 | 0.8021 | 0.0375 | 0.7937 | 0.8125 | 0.8007 | 103.00 | 0.8500 | 6 |
| 7 | R1 | 0.8920 | 0.9316 | 0.7979 | 0.0297 | 0.7937 | 0.8063 | 0.7964 | 103.00 | 0.8500 | 6 |
| 42 | C0 | 0.8833 | 0.9326 | 0.8202 | 0.0629 | 0.7688 | 0.8219 | 0.8257 | 104.88 | 0.8500 | 6 |
| 42 | C1 | 0.8833 | 0.9309 | 0.8106 | 0.0551 | 0.7812 | 0.8156 | 0.8129 | 121.97 | 0.8750 | 5 |
| 42 | R1 | 0.8865 | 0.9315 | 0.8165 | 0.0543 | 0.7875 | 0.8187 | 0.8193 | 121.97 | 0.8750 | 5 |
| 2026 | C0 | 0.8789 | 0.9188 | 0.7681 | 0.0281 | 0.7375 | 0.7875 | 0.7671 | 126.06 | 0.8500 | 6 |
| 2026 | C1 | 0.8810 | 0.9195 | 0.7654 | 0.0223 | 0.7375 | 0.7906 | 0.7629 | 126.06 | 0.8500 | 6 |
| 2026 | R1 | 0.8810 | 0.9195 | 0.7649 | 0.0219 | 0.7375 | 0.7875 | 0.7629 | 126.06 | 0.8500 | 6 |

每个方法的 Accuracy/AUROC、median delay、validation threshold、预算、训练历史与 Probe 历史保存在对应 `metrics.json`。

## 诊断指标

| Seed | 方法 | N→F | F→N | Normal mean | Normal P95 | Fault mean | Normal corr drift | Critical retention | Norm. L1 | Repr. Fisher | Center shift | Eff. rank |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | C1 | 0.0375 | 0.1979 | 0.3818 | 0.4146 | 0.6692 | 0.00782 | 0.9978 | 0.0517 | 1.8294 | 0.0089 | 1.1702 |
| 7 | R1 | 0.0297 | 0.2021 | 0.4535 | 0.4678 | 0.6366 | 0.00594 | 1.0013 | 0.0518 | 1.8249 | 0.0087 | 1.1700 |
| 42 | C1 | 0.0551 | 0.1894 | 0.4648 | 0.4717 | 0.5429 | 0.00766 | 0.9981 | 0.0516 | 1.8155 | 0.0052 | 1.3251 |
| 42 | R1 | 0.0543 | 0.1835 | 0.3493 | 0.3827 | 0.6290 | 0.00585 | 1.0016 | 0.0518 | 1.8170 | 0.0050 | 1.3496 |
| 2026 | C1 | 0.0223 | 0.2346 | 0.5186 | 0.5300 | 0.6184 | 0.00796 | 0.9970 | 0.0517 | 1.9225 | 0.0063 | 1.0757 |
| 2026 | R1 | 0.0219 | 0.2351 | 0.5183 | 0.5298 | 0.6183 | 0.00612 | 1.0007 | 0.0519 | 1.9223 | 0.0061 | 1.0747 |

correlation drift 与 critical retention 仅作机制分析，不参与本轮 Gate。

## Mean ± sample std（ddof=1）

| 方法 | Macro-F1 | AUPRC | Recall | FAR | Early Recall | Mean Delay |
|---|---|---|---|---|---|---|
| C0 | 0.8852 ± 0.0075 | 0.9266 ± 0.0071 | 0.7920 ± 0.0263 | 0.0366 ± 0.0233 | 0.7521 ± 0.0157 | 113.9804 ± 10.8983 |
| C1 | 0.8846 ± 0.0044 | 0.9273 ± 0.0067 | 0.7927 ± 0.0240 | 0.0383 ± 0.0164 | 0.7708 ± 0.0295 | 117.0101 ± 12.3040 |
| R1 | 0.8865 ± 0.0055 | 0.9275 ± 0.0070 | 0.7931 ± 0.0261 | 0.0353 ± 0.0169 | 0.7729 ± 0.0308 | 117.0101 ± 12.3040 |

## 配对差值

`ΔFAR<0`、`ΔDelay<0` 表示改善。

| Seed | 比较 | ΔMacro-F1 | ΔAUPRC | ΔRecall | ΔFAR | ΔEarly | ΔDelay |
|---:|---|---:|---:|---:|---:|---:|---:|
| 7 | R1-C1 | +0.0025 | +0.0002 | -0.0043 | -0.0078 | +0.0000 | +0.00 |
| 7 | R1-C0 | -0.0016 | +0.0033 | +0.0101 | +0.0109 | +0.0437 | -8.00 |
| 42 | R1-C1 | +0.0031 | +0.0006 | +0.0059 | -0.0008 | +0.0062 | +0.00 |
| 42 | R1-C0 | +0.0032 | -0.0010 | -0.0037 | -0.0086 | +0.0187 | +17.09 |
| 2026 | R1-C1 | -0.0000 | +0.0000 | -0.0005 | -0.0004 | +0.0000 | +0.00 |
| 2026 | R1-C0 | +0.0021 | +0.0007 | -0.0032 | -0.0063 | +0.0000 | +0.00 |

R1 相对 C1：Macro-F1 胜 `2/3`，FAR 胜 `3/3`，Early Recall 不下降超过 1pp 为 `3/3`。灾难性 Seed：`{'7': False, '42': False, '2026': False}`。

## Gate

- mean_macro_f1_above_c1: `True`
- macro_f1_wins_at_least_2: `True`
- mean_far_below_c1: `True`
- far_wins_at_least_2: `True`
- mean_recall_preserved: `True`
- no_single_recall_drop_over_limit: `True`
- mean_auprc_preserved: `True`
- mean_early_recall_preserved: `True`
- early_preserved_at_least_2: `True`
- mean_delay_within_one_stride: `True`
- no_catastrophic_seed: `True`

C0 辅助警告：`False`；辅助状态：`None`。C0 不是本轮 R1 vs C1 的主要因果对照，但若 R1 明显逊于 C0，不得据此进入论文主张。

## 统计与下一步边界

本轮只有 3 个 Seed，只报告 mean、sample std、配对方向与胜率，不计算 p-value，也不声称统计显著。test 未用于参数、epoch、threshold、方法或 Gate 选择；但当前 TEP test 已在之前多轮工程探索中被查看，本轮仍是探索性验证，不是论文最终无偏评测。论文级实验必须使用第二数据集或新的未触碰评测协议。

若状态为 GO，唯一下一步是冻结 R1 为新频率选择性基线，另行增量验证 C3；本轮未实现 C3。若为 UNSTABLE/NO-GO，则停止，不搜索 t=4/6，不添加 C3/C4/C5。
