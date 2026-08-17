# HFSC TEP 21 类 Diagnosis 报告

结论：`TEP_HFSC_NO_GO`。

- HFSC−R1 Macro-F1：`+0.00510 ± 0.00791`，2/3 positive
- HFSC−R2 Macro-F1：`+0.00139 ± 0.01001`
- HFSC−R1 Multiclass AUPRC：`+0.00304`
- HFSC−R1 Binary AUPRC / Recall / FAR：`+0.01526` / `+0.01436` / `-0.00964`
- 正 mean Recall 的 fault classes：`3/20`
- 对应 fault：`[6, 8, 18]`

| Seed | HFSC−R1 Macro-F1 | HFSC−R2 Macro-F1 | Δ Binary AUPRC vs R1 | Δ FAR vs R1 |
|---:|---:|---:|---:|---:|
| 7 | +0.01501 | +0.01519 | -0.00932 | -0.00039 |
| 42 | +0.00464 | -0.00274 | +0.02433 | -0.02695 |
| 2026 | -0.00436 | -0.00827 | +0.03078 | -0.00156 |

20 类逐类 Recall/F1 与配对结果见 JSON。Shared-vs-Diagnostic Jaccard median `0.52258`；Diagnostic masks pairwise Jaccard median `0.50146`，class-specific patterns confirmed=`True`。

Gate：`{'macro_f1_mean_positive': True, 'macro_f1_majority_positive': True, 'per_fault_improvement_not_sparse': False, 'binary_auprc_preserved': True, 'fault_recall_preserved': True, 'far_preserved': True, 'early_recall_preserved': True}`。本协议仍为 exploratory，不能形成 paper-final claim。
