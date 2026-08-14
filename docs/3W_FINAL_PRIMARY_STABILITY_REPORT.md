# 3W 最终 Primary grouped cross-WELL 稳定性验证

最终状态：`3W_FINAL_PRIMARY_STABILITY_GO`

本阶段固定 Final Primary `[0,2,8,9]`、Real WELL-only、Process-only、window/stride=64/32、Seed=42、TCN、Hard SupCon 20 epochs 与 balanced frozen linear probe 15 epochs。Class 4/7 未重新加入，未调整模型、阈值或 feature。

## Grouped split 协议

最终 Primary 实际涉及 36 口真实 WELL；仅承载 Secondary event 的 4 口 WELL 不进入该标签空间。以预注册 seeds `42/43/44` 构造三个 20/8/8 train/validation/test grouped splits，同一 WELL 不跨 split。每个 split 中，所有 fault class 在 train 至少覆盖 3 口目标 WELL，在 validation/test 至少各覆盖 2 口；三个 test WELL 集两两 Jaccard 不超过 0.5。每个 split 均重新训练模型，imputation/normalization 只拟合对应 train WELL。

## 分 split 结果

| Split | Macro-F1 | Macro Recall | Binary AUPRC | Multiclass AUPRC | FAR | Early Recall | Delay (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 / seed 42 | 0.5637 | 0.5731 | 0.7963 | 0.5205 | 0.3616 | 0.9141 | 942.04 |
| 1 / seed 43 | 0.5021 | 0.6712 | 0.5078 | 0.5705 | 0.6041 | 0.9213 | 2256.89 |
| 2 / seed 44 | 0.5165 | 0.8047 | 0.8120 | 0.6280 | 0.3700 | 0.9987 | 70.00 |
| mean ± std | 0.5274 ± 0.0263 | 0.6830 ± 0.0949 | 0.7054 ± 0.1399 | 0.5730 ± 0.0439 | 0.4452 ± 0.1124 | 0.9447 ± 0.0383 | 1089.64 ± 898.87 |

| Split | Class 0 Recall/F1 | Class 2 Recall/F1 | Class 8 Recall/F1 | Class 9 Recall/F1 |
|---:|---:|---:|---:|---:|
| 0 | 0.6384 / 0.7298 | 0.2791 / 0.4364 | 0.8990 / 0.8340 | 0.4759 / 0.2546 |
| 1 | 0.3959 / 0.5575 | 0.3663 / 0.5362 | 0.9225 / 0.8674 | 1.0000 / 0.0472 |
| 2 | 0.6300 / 0.7728 | 0.7014 / 0.1581 | 0.9672 / 0.8033 | 0.9201 / 0.3318 |

四个 Primary class 在三个独立 grouped test splits 上 Recall 均大于 0，因此通过预注册 HARD GATE。结果同时显示明显 domain variance：FAR、Delay 及 Class 2/9 F1 跨 split 波动较大；GO 表示最终标签空间未出现系统性 zero-recall，并不表示 3W cross-WELL 问题已经完全解决。

Stage A 允许进入同一协议、canonical split 00、Seed 42 的单 Seed diffusion 对比。
