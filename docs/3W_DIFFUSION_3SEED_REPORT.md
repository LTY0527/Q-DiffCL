# 3W Final Protocol 频率选择性扩散 3-Seed 稳定性复核

最终状态：`3W_FREQUENCY_SELECTIVE_R1_3SEED_INCONCLUSIVE`

本轮严格冻结 Final Primary `[0,2,8,9]`、canonical grouped split 00、Real WELL-only、Process-only、window/stride=64/32、train-only preprocessing、TCN、Hard SupCon、balanced frozen probe、Uniform `t=3` 及 R1 `t_key=1/t_nonkey=5`，仅改变随机种子 `42/43/44`。

窗口抽样固定使用 protocol seed 42，因此三个 seed 的 train/validation window refs 完全相同；模型初始化、batch order 和扩散噪声使用对应实验 seed。D/E/S 是确定性 train-data 统计，关键频率 soft mask 由 Seed 42 的 train WELL 审计产生，并由 Seed 43/44 原样复用；window refs 与 mask SHA256 在汇总前强制一致，未读取 validation/test 拟合频率统计。

## 九个正式 run

| Seed | Method | Macro-F1 | Macro Recall | Binary AUPRC | Multiclass AUPRC | FAR | Early Recall | Delay (s) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | Clean | 0.4806 | 0.4536 | 0.7843 | 0.5286 | 0.4394 | 0.9057 | 937.95 |
| 42 | Uniform | 0.4764 | 0.4483 | 0.7791 | 0.5165 | 0.4388 | 0.9019 | 965.59 |
| 42 | R1 | 0.5008 | 0.4724 | 0.7839 | 0.5244 | 0.3445 | 0.8967 | 916.14 |
| 43 | Clean | 0.5435 | 0.5539 | 0.8700 | 0.6581 | 0.4945 | 0.9405 | 315.95 |
| 43 | Uniform | 0.5194 | 0.4530 | 0.8781 | 0.6661 | 0.4733 | 0.9149 | 322.05 |
| 43 | R1 | 0.5532 | 0.5711 | 0.8851 | 0.6648 | 0.4975 | 0.9426 | 337.29 |
| 44 | Clean | 0.2656 | 0.2519 | 0.4697 | 0.2903 | 0.3086 | 0.4824 | 98.79 |
| 44 | Uniform | 0.3667 | 0.3023 | 0.5380 | 0.3232 | 0.3742 | 0.7325 | 190.79 |
| 44 | R1 | 0.4061 | 0.3551 | 0.4683 | 0.3653 | 0.3137 | 0.6927 | 257.46 |

## 各方法 3-Seed mean ± std

| Method | Macro-F1 | Macro Recall | Binary AUPRC | Multiclass AUPRC | FAR | Early Recall | Delay (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clean | 0.4299 ± 0.1190 | 0.4198 ± 0.1256 | 0.7080 ± 0.1721 | 0.4924 ± 0.1523 | 0.4142 ± 0.0780 | 0.7762 ± 0.2083 | 450.90 ± 355.63 |
| Uniform | 0.4542 ± 0.0643 | 0.4012 ± 0.0700 | 0.7317 ± 0.1428 | 0.5020 ± 0.1404 | 0.4288 ± 0.0411 | 0.8498 ± 0.0831 | 492.81 ± 338.57 |
| R1 | 0.4867 ± 0.0609 | 0.4662 ± 0.0883 | 0.7124 ± 0.1775 | 0.5182 ± 0.1223 | 0.3852 ± 0.0804 | 0.8440 ± 0.1086 | 503.63 ± 293.50 |

## Paired R1 − Uniform

| Seed | Macro-F1 | Macro Recall | Binary AUPRC | Multiclass AUPRC | FAR | Early Recall | Delay (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | +0.02447 | +0.02407 | +0.00480 | +0.00785 | -0.09428 | -0.00518 | -49.45 |
| 43 | +0.03380 | +0.11813 | +0.00704 | -0.00131 | +0.02412 | +0.02768 | +15.24 |
| 44 | +0.03945 | +0.05281 | -0.06973 | +0.04209 | -0.06044 | -0.03978 | +66.67 |
| mean ± std | +0.03258 ± 0.00617 | +0.06500 ± 0.03936 | -0.01930 ± 0.03567 | +0.01621 ± 0.01868 | -0.04353 ± 0.04979 | -0.00576 ± 0.02754 | +10.82 ± 47.51 |

Macro-F1 为 3/3 seed 改善，FAR 为 2/3 改善，Binary AUPRC 为 2/3 不差，Multiclass AUPRC 为 2/3 不差。Macro-F1 与 FAR 的均值方向支持 R1，Early Recall 均值仍在既有 0.01 容差内，Delay 有 2/3 seed 满足既有容差；但 Binary AUPRC 平均下降 0.01930，主要由 Seed 44 的 -0.06973 驱动，因此“AUPRC 整体不恶化”未通过，不能给 GO。

## Per-class 稳定性

| Method | Class 0 Recall/F1 | Class 2 Recall/F1 | Class 8 Recall/F1 | Class 9 Recall/F1 |
|---|---:|---:|---:|---:|
| Clean mean ± std | .5858±.0780 / .6326±.0313 | .3262±.0632 / .4735±.0708 | .6093±.4114 / .5747±.3690 | .1580±.2147 / .0389±.0447 |
| Uniform mean ± std | .5712±.0411 / .6473±.0113 | .3183±.0535 / .4727±.0625 | .6915±.2794 / .6874±.2016 | .0238±.0250 / .0091±.0061 |
| R1 mean ± std | .6148±.0804 / .6752±.0395 | .3369±.0785 / .4852±.0847 | .7421±.2111 / .7451±.1426 | .1710±.2331 / .0414±.0482 |

R1 相对 Uniform 的 Class 8 F1 在 3/3 seed 均改善，normal recall/FAR 在 2/3 改善；Class 2 仅 Seed 43 有明确收益。Class 9 没有严格 Recall=0，但 R1 Recall 为 Seed 42 `0.0097`、Seed 43 `0.5007`、Seed 44 `0.0028`，属于显著 seed instability，平均提升几乎全部来自 Seed 43，不能视为稳定类别收益。

## 决策

结论冻结为 `3W_FREQUENCY_SELECTIVE_R1_3SEED_INCONCLUSIVE`，不是 GO，也未达到灾难性 HOLD 条件。按停止线不自动调参、不改变 split/类别/阈值，也不立即扩展论文级强基线。下一步若有新指令，应先做只读机制审计，定位 Seed 44 binary AUPRC 下降以及 Class 9 表征/预测不稳定的来源，再决定是否值得继续投入。
