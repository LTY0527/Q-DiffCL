# UG-R1 TEP 报告

结论：`TEP_UG_R1_GO`。下游保持 binary detection。

- UG-R1−R1 Macro-F1 / FAR：`+0.00075` / `-0.00273`
- Early Recall / Binary AUPRC：`-0.00417` / `+0.00006`
- Detection Delay / detected rate：`+0.00` / `+0.00000`

| Seed | Δ Macro-F1 | Δ FAR | Δ Early Recall | Δ AUPRC |
|---:|---:|---:|---:|---:|
| 7 | -0.00045 | +0.00078 | +0.00000 | +0.00006 |
| 42 | +0.00268 | -0.00938 | -0.01250 | +0.00012 |
| 2026 | +0.00003 | +0.00039 | +0.00000 | +0.00001 |

按 faultNumber 分层 run counts=`{'0': 128, '1': 6, '2': 6, '3': 6, '4': 6, '5': 6, '6': 6, '7': 6, '8': 6, '9': 6, '10': 6, '11': 6, '12': 6, '13': 6, '14': 6, '15': 6, '16': 6, '17': 6, '18': 6, '19': 6, '20': 6}`；p median=`0.0000`，r median=`1.0000`；changed bins=`434`，timestep MAE=`0.0100`；预算误差=`1.86e-09`。Gate：`{'mean_macro_f1_preserved': True, 'mean_far_preserved': True, 'mean_early_recall_preserved': True, 'mean_binary_auprc_preserved': True, 'no_catastrophic_seed': True}`。
