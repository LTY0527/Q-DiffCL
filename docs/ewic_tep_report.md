# EWIC TEP Early Detection 报告

结论：`TEP_EWIC_NO_GO`。

下游始终是 binary fault detection；20-fault 信息仅用于 test profile 分组，没有训练 multiclass classifier。`E_h`、lead weighting、run bootstrap reliability 与 0.5/0.3/0.2 composite 定义和 3W 完全相同。

- EWIC−R1 Early Recall：`-0.00208`，0/3 positive
- EWIC−R1 FAR：`+0.00560`
- EWIC−R1 Detection Delay：`-0.31` samples，1/3 seed 缩短
- Binary AUPRC / Fault Recall：`-0.00020` / `+0.00195`
- Early Recall 改善 faults：`[]`
- Delay 缩短 faults：`[10]`
- 明显退化 faults：`[]`

| Seed | Δ Early Recall | Δ FAR | Δ Delay(samples) |
|---:|---:|---:|---:|
| 7 | +0.00000 | +0.01211 | -0.94 |
| 42 | -0.00625 | +0.00430 | +0.00 |
| 2026 | +0.00000 | +0.00039 | +0.00 |

| Fixed-FAR OP | Δ Early Recall | Δ Delay | Δ observed test FAR |
|---|---:|---:|---:|
| far_1pct | -0.00625 | +0.78 | +0.00026 |
| far_5pct | -0.00208 | -0.16 | -0.00013 |

h1–h2 优势 bins：c40/f2, c28/f8, c26/f4, c40/f3, c10/f7；h7–h8 优势 bins：c29/f4, c29/f0, c29/f15, c17/f0, c37/f1。Reliability 过滤 `8` 个 lead Top-30% bins。Mask Jaccard=`0.89687`，changed bins=`56`。Gate：`{'early_recall_at_least_2of3_positive': False, 'delay_at_least_2of3_shorter': False, 'far_not_worse': True, 'fault_improvement_broad': False, 'fixed_far_direction_consistent': False}`。
