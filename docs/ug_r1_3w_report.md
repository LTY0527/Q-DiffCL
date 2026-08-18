# UG-R1 3W 报告

结论：`3W_UG_R1_NO_GO`。

- UG-R1−R1 Macro-F1：`-0.05871 ± 0.04757`，0/3 nonnegative
- FAR：`+0.02998`；Macro-F1 std R1/UG-R1=`0.06088/0.10570`
- Class 9 Recall/F1 std：R1 `0.23312/0.04824`，UG-R1 `0.07200/0.01504`

| Seed | Δ Macro-F1 | Δ FAR | Δ Class 9 Recall |
|---:|---:|---:|---:|
| 42 | -0.01966 | +0.07750 | -0.00046 |
| 43 | -0.03078 | -0.01744 | -0.34207 |
| 44 | -0.12568 | +0.02987 | +0.00000 |

完整 WELL bootstrap units=`20`；p median=`0.2188`，r median=`0.6562`；changed bins=`558`，timestep MAE=`0.1529`；预算误差=`0`。Gate：`{'mean_macro_f1_preserved': False, 'macro_f1_at_least_2of3_nonnegative': False, 'mean_far_preserved': False, 'stability_improved_10pct': True, 'class9_recall_std_not_worse_10pct': True, 'class9_f1_std_not_worse_10pct': True, 'no_catastrophic_seed': False}`。
