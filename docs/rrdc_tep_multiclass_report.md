# RRDC TEP 21 类 Diagnosis 报告

结论：`TEP_RRDC_NO_GO`。

- RRDC−R1 Macro-F1：`-0.00343 ± 0.00957`，2/3 positive
- RRDC−HFSC Macro-F1：`-0.00853`
- RRDC−R1 Multiclass / Binary AUPRC：`+0.01502` / `+0.01388`
- RRDC−R1 Fault Recall / FAR / Early Recall：`+0.01560` / `-0.00677` / `+0.00208`
- Mean Recall 改善 fault：`2/20`，类别 `[7, 8]`

| Seed | Δ R1 Macro-F1 | Δ HFSC | Δ Binary AUPRC |
|---:|---:|---:|---:|
| 7 | +0.00295 | -0.01206 | +0.00129 |
| 42 | +0.00371 | -0.00093 | +0.02745 |
| 2026 | -0.01696 | -0.01261 | +0.01289 |

Hardest-rival pairs：1→7, 2→16, 3→4, 4→3, 5→10, 6→10, 7→1, 8→13, 9→14, 10→16, 11→15, 12→8, 13→8, 14→3, 15→11, 16→10, 17→11, 18→12, 19→4, 20→15。完整逐类结果与 reliability/mask 审计见 JSON。Gate：`{'macro_f1_mean_positive': False, 'macro_f1_at_least_2of3_positive': True, 'improved_fault_count_exceeds_hfsc': False, 'improvement_not_sparse': False, 'binary_auprc_preserved': True, 'fault_recall_preserved': True, 'far_preserved': True, 'early_recall_preserved': True}`。
