# RRDC 3W 报告

结论：`3W_RRDC_NO_GO`。

- RRDC−R1 Macro-F1：`+0.01075 ± 0.06713`，2/3 positive
- RRDC−R2 / HFSC Macro-F1：`+0.01380` / `+0.00879`
- RRDC−R1 Multiclass AUPRC：`+0.05896`
- RRDC−R1 Binary AUPRC / FAR / Early Recall：`+0.13879` / `-0.03500` / `+0.03517`

| Seed | Δ R1 Macro-F1 | Δ R2 | Δ HFSC |
|---:|---:|---:|---:|
| 42 | +0.00201 | +0.02490 | +0.03723 |
| 43 | -0.06674 | +0.01468 | -0.00030 |
| 44 | +0.09698 | +0.00181 | -0.01056 |

| Fault | Mean Δ Recall vs R1 |
|---:|---:|
| 2 | -0.05597 |
| 8 | +0.12777 |
| 9 | -0.15586 |

Hardest rivals：Fault 2→9 (0.18316), Fault 8→2 (0.25984), Fault 9→2 (0.18316)。完整 reliability、mask hash、changed bins 与 Jaccard 见 JSON。Gate：`{'macro_f1_mean_positive': True, 'macro_f1_at_least_2of3_positive': True, 'multiclass_auprc_nonworse': True, 'fault_2_8_9_balanced': False, 'detection_preserved': True, 'early_preserved': True}`。
