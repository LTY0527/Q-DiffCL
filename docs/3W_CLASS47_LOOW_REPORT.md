# 3W Class 4/7 跨 WELL 稳定性复核与最终协议冻结

最终状态：`3W_REAL_ONLY_GENERALIZATION_HOLD`

Base commit：`b30d7c8`。本阶段固定 Real WELL-only、Process-only、window/stride=64/32、Seed=42、TCN、Hard SupCon 20 epochs、balanced frozen probe 15 epochs，以及 train-only imputation/normalization；未调参、未改变特征、未加入 SIMULATED/DRAWN，也未运行 diffusion。

## LOOW 协议与泄漏控制

对所有包含 class 4 或 class 7 的真实 WELL 做完整 Leave-One-WELL-Out。每个 fold 将 held-out WELL 的全部实例整体作为 test；该 WELL 不进入 train、validation、预处理统计或 probe 拟合。其余 39 口真实 WELL 以固定 Seed 42 搜索得到 31 train / 8 validation，并要求六个 Strict class `0,2,4,7,8,9` 在 train 和 validation 均有覆盖。

共有 13 个 class-WELL 评估目标，因 `WELL-00001` 同时含 class 4 和 class 7，实际只需训练 12 个唯一 fold。所有 fold 均已完成并保存 per-class metrics 与 confusion matrix。

## Class 4

| Held-out WELL | Recall | F1 | Target windows |
|---|---:|---:|---:|
| WELL-00001 | 0.000000 | 0.000000 | 8,170 |
| WELL-00002 | 0.000000 | 0.000000 | 25,018 |
| WELL-00004 | 0.000000 | 0.000000 | 9,616 |
| WELL-00005 | 0.005421 | 0.010783 | 8,486 |
| WELL-00007 | 0.000000 | 0.000000 | 2,239 |
| WELL-00010 | 0.000000 | 0.000000 | 18,510 |
| WELL-00014 | 0.002347 | 0.004684 | 4,686 |

- Recall mean / median / std：0.001110 / 0 / 0.001937。
- F1 mean / median / std：0.002210 / 0 / 0.003855。
- Zero / positive Recall WELL：5 / 2。
- Best：`WELL-00005`，Recall=0.005421；worst 为其余五口 Recall=0 的 WELL（并列）。
- 判定：`SYSTEMATIC_CROSS_WELL_FAILURE`。即使两口井出现非零输出，量级仍接近零，不能视为可接受的跨井泛化。

## Class 7

| Held-out WELL | Recall | F1 | Target windows |
|---|---:|---:|---:|
| WELL-00001 | 0.000000 | 0.000000 | 1,227 |
| WELL-00006 | 0.000000 | 0.000000 | 6,984 |
| WELL-00021 | 0.000000 | 0.000000 | 1,360 |
| WELL-00022 | 0.000000 | 0.000000 | 82,631 |
| WELL-00023 | 0.000000 | 0.000000 | 42,579 |
| WELL-00024 | 0.111177 | 0.193009 | 74,620 |

- Recall mean / median / std：0.018529 / 0 / 0.041433。
- F1 mean / median / std：0.032168 / 0 / 0.071930。
- Zero / positive Recall WELL：5 / 1。
- Best：`WELL-00024`，Recall=0.111177；worst 为其余五口 Recall=0 的 WELL（并列）。
- 判定：`SYSTEMATIC_CROSS_WELL_FAILURE`。唯一非零 WELL 不足以支持稳定泛化，且 Recall 仍仅为 0.1112。

单 WELL test 通常不包含全部六个 Strict class，因此 fold 级 multiclass AUPRC 会触发 sklearn 的缺失正类 warning，fold Macro-F1 也只适合做诊断。上述 failure 判定只使用 held-out WELL 上实际存在的目标 class Recall/F1，不受该 warning 影响。每 fold 的 Macro-F1、FAR、Early Recall、Detection Delay 与检测实例率见 `3w_class47_loow_results.csv`。

## 最终协议

```text
FINAL_3W_PRIMARY_CLASSES   = [0, 2, 8, 9]
FINAL_3W_SECONDARY_CLASSES = [1, 3, 4, 5, 6, 7]
```

Class 4/7 从 Strict Primary 转入 Secondary：完整 LOOW 表明原 test split 的 zero recall 不是少数异常 WELL 所致，而是系统性跨 WELL 泛化失败。Class 1/5 因训练 target WELL 支持不足、class 3/6 因既有协议结论，继续保留在 Secondary。Primary 仍冻结为 Real WELL-only、Process-only、well-level split；Secondary 只用于高难度/探索性报告，不得与 Primary 主结论混合。

## 决策

当前证据不支持把 Real-only 3W 主协议宣告为已解决，因此冻结为 `3W_REAL_ONLY_GENERALIZATION_HOLD`。不得进入 `Uniform Diffusion 1-Seed vs Frequency-Selective R1 1-Seed`，更不得启动 diffusion 3-Seed。若后续收到新指令，应先独立验证最终 Primary `0,2,8,9` 的 grouped cross-WELL 稳定性，再决定是否解除 HOLD；不应通过调阈值、调模型或继续删类来改变本轮结论。
