# HFSC 3W 报告

结论：`3W_HFSC_NO_GO`。

- HFSC−R1 Macro-F1：`+0.00196 ± 0.07574`，1/3 positive
- HFSC−R2 Macro-F1：`+0.00501 ± 0.01230`
- HFSC−R1 Multiclass AUPRC：`+0.04759`
- HFSC−R1 FAR / Early Recall：`-0.00380` / `+0.04529`

| Seed | HFSC−R1 Macro-F1 | HFSC−R2 Macro-F1 |
|---:|---:|---:|
| 42 | -0.03522 | -0.01233 |
| 43 | -0.06644 | +0.01498 |
| 44 | +0.10754 | +0.01236 |

| Fault | Δ Recall vs R1 | Δ F1 vs R1 | Recall positive seeds |
|---:|---:|---:|---:|
| 2 | -0.05612 | -0.06543 | 1/3 |
| 8 | +0.13531 | +0.08875 | 2/3 |
| 9 | -0.14866 | -0.03175 | 0/3 |

Fault 2/8/9 的配对 Recall 变化完整记录在 JSON/CSV。Shared-vs-Diagnostic Jaccard 范围为 `[0.21111, 0.26012]`；Diagnostic masks pairwise Jaccard 范围为 `[0.42951, 0.54064]`，class-specific patterns confirmed=`True`。

Gate：`{'macro_f1_mean_positive': True, 'macro_f1_at_least_2of3_positive': False, 'multiclass_auprc_nonworse': True, 'far_no_systematic_degradation': True, 'early_no_systematic_degradation': True}`。HFSC 未达到相对 R1 至少 2/3 seed 正向的稳定性要求。
