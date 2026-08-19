# FINAL_QDIFFCL 5-Seed Reliability

冻结方法严格保持 `0.5D+0.5E`；本轮结果不用于重新调权重。

## 3W

| Method | Macro-F1 | FAR | Early Recall | AUPRC | Worst seed |
|---|---:|---:|---:|---:|---:|
| UNIFORM | 0.4835 ± 0.0805 | 0.4416 ± 0.0550 | 0.8646 ± 0.0774 | 0.5631 ± 0.1482 | 44 |
| CURRENT_R1 | 0.5101 ± 0.0682 | 0.3832 ± 0.0724 | 0.8509 ± 0.0946 | 0.5598 ± 0.1207 | 44 |
| FINAL_QDIFFCL | 0.5396 ± 0.0354 | 0.4042 ± 0.1192 | 0.8948 ± 0.0502 | 0.6004 ± 0.0802 | 42 |

## TEP

| Method | Macro-F1 | FAR | Early Recall | AUPRC | Worst seed |
|---|---:|---:|---:|---:|---:|
| UNIFORM | 0.8894 ± 0.0073 | 0.0311 ± 0.0161 | 0.7638 ± 0.0240 | 0.9298 ± 0.0060 | 2026 |
| CURRENT_R1 | 0.8908 ± 0.0070 | 0.0285 ± 0.0156 | 0.7638 ± 0.0255 | 0.9299 ± 0.0060 | 2026 |
| FINAL_QDIFFCL | 0.8903 ± 0.0068 | 0.0297 ± 0.0151 | 0.7638 ± 0.0240 | 0.9299 ± 0.0060 | 2026 |

## Paired reliability

### 3W

| Comparison | Metric | Mean Δ | Positive seeds | Non-worse seeds | LOSO range |
|---|---|---:|---:|---:|---:|
| FINAL_QDIFFCL-UNIFORM | macro_f1 | +0.0561 | 5/5 | 5/5 | 0.0343 |
| FINAL_QDIFFCL-UNIFORM | far | -0.0375 | 2/5 | 2/5 | 0.0380 |
| FINAL_QDIFFCL-UNIFORM | early_recall | +0.0302 | 4/5 | 4/5 | 0.0264 |
| FINAL_QDIFFCL-CURRENT_R1 | macro_f1 | +0.0295 | 4/5 | 4/5 | 0.0266 |
| FINAL_QDIFFCL-CURRENT_R1 | far | +0.0209 | 3/5 | 3/5 | 0.0628 |
| FINAL_QDIFFCL-CURRENT_R1 | early_recall | +0.0439 | 3/5 | 3/5 | 0.0356 |

### TEP

| Comparison | Metric | Mean Δ | Positive seeds | Non-worse seeds | LOSO range |
|---|---|---:|---:|---:|---:|
| FINAL_QDIFFCL-UNIFORM | macro_f1 | +0.0009 | 3/5 | 3/5 | 0.0007 |
| FINAL_QDIFFCL-UNIFORM | far | -0.0014 | 3/5 | 3/5 | 0.0019 |
| FINAL_QDIFFCL-UNIFORM | early_recall | +0.0000 | 0/5 | 5/5 | 0.0000 |
| FINAL_QDIFFCL-CURRENT_R1 | macro_f1 | -0.0005 | 0/5 | 1/5 | 0.0002 |
| FINAL_QDIFFCL-CURRENT_R1 | far | +0.0012 | 1/5 | 2/5 | 0.0012 |
| FINAL_QDIFFCL-CURRENT_R1 | early_recall | +0.0000 | 1/5 | 4/5 | 0.0031 |

## Catastrophic seed audit

按 FINAL Macro-F1 比任一基线低至少 0.10 的预注册定义：无 catastrophic seed。

## Mechanism / fairness

- 3W FINAL vs CURRENT hard-mask Jaccard `0.6834`，changed bins `82`。
- TEP FINAL vs CURRENT hard-mask Jaccard `0.8830`，changed bins `64`。
- 最大 matched-budget error `1.863e-09`；mask 跨 seed 固定且公平性哈希一致。
- test 仅用于冻结后的可靠性评估，未用于权重选择。

## Result reuse

- 直接复用既有 Uniform/CURRENT 测试结果：16 条。
- 复用 validation 搜索 checkpoint、仅新增 test 评估：6 条。
- 新训练：8 条（3W FINAL seeds 45/46；TEP seeds 43/44 的三种方法）。
- 最终组件表额外公平复用既有 D_ONLY/E_ONLY 结果 12 条，未重跑完整 8-variant 消融。

无论本轮结果好坏，FINAL 权重均不重新打开。下一阶段进入 external baseline / SOTA comparison。
