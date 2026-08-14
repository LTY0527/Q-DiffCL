# R3 平衡可靠多类别关键频率：3W

阶段结论：`R3_3W_NO_GO`。Stage B 不放行，不执行 TEP。

## 固定方法

R3 保持 `0.40D + 0.24E + 0.16S + 0.20M`，只将 M 替换为 Balanced + Reliable M。所有统计只使用 train run aggregate；对每类计算 one-vs-rest 分数并分别 robust normalization，类别等权平均，再乘以分层 run-bootstrap 中进入 top critical-ratio 的 selection probability。未改变 timestep、mask ratio、TCN、Hard SupCon、Original batching、split、probe、phase/DC 或总噪声预算。

## 配对结果

- R3−Uniform Macro-F1：`+0.04722 ± 0.06413`，2/3 positive
- R3−R1 Macro-F1：`+0.01465`，1/3 nonnegative
- R3−R2 Macro-F1：`+0.01769`，2/3 nonnegative
- R3−Uniform Binary AUPRC：`+0.10764`
- R3−Uniform Multiclass AUPRC：`+0.06094`
- R3−Uniform FAR：`-0.01237`

| Seed | Δ Macro-F1 | Δ Binary AUPRC | Δ Multiclass AUPRC | Δ FAR |
|---:|---:|---:|---:|---:|
| 42 | -0.00636 | -0.03234 | -0.01248 | +0.05427 |
| 43 | +0.01065 | +0.00448 | -0.00491 | +0.01006 |
| 44 | +0.13737 | +0.35079 | +0.20021 | -0.10144 |

## Class 9 与稳定性

R3 Class 9 Recall/F1 mean 为 `0.07096` / `0.02044`，高于 R2 的 `0.00736` / `0.00529`。但 R3 Recall std 为 `0.09161`，改善集中在单个 seed，属于不稳定提升，不是稳定失败，也不是稳定解决。

R3−Uniform Macro-F1 std `0.06413`，虽低于 R2 的 `0.07698`，但未达到预注册建议目标 `≤0.050`。

## Mask 与 M 审计

- R2/R3 Jaccard：`0.57971`
- changed bins：`116`；selected bins：`218`
- R2 hash：`3325c046ffc8096813d4fa9c09f8c35d655317d0889392884c60fc37b45692e5`
- R3 hash：`62374c85dbdbf5cd1152396ab4ed00f42604d8627d7892ae575c6862d1c99d63`
- bootstrap selection probability：min `0.00000`，median `0.21875`，mean `0.30028`，max `1.00000`
- train run counts：`{'0': 20, '2': 3, '8': 5, '9': 3}`

各类等权 M 贡献：

- Class 0：mean `0.62676`，P75 `0.85851`，max `8.00000`
- Class 2：mean `0.72565`，P75 `0.87137`，max `8.00000`
- Class 8：mean `0.38942`，P75 `0.37033`，max `8.00000`
- Class 9：mean `0.68564`，P75 `0.53443`，max `8.00000`

## Gate

- `macro_f1_mean_gain`：通过
- `macro_f1_wins`：通过
- `r1_macro_f1_nonnegative_seeds`：未通过
- `r1_macro_f1_mean_preserved`：通过
- `macro_f1_seed_std_reduced`：未通过
- `binary_auprc_mean_gain`：通过
- `no_large_binary_auprc_seed_drop`：未通过
- `multiclass_auprc_mean_gain`：通过
- `far_mean_preserved`：通过
- `class9_recall_improved_vs_r2`：通过
- `class9_f1_improved_vs_r2`：通过
- `class9_recall_not_near_zero`：通过
- `class9_f1_not_near_zero`：通过

主信号均值存在，但 R3−R1 seed 覆盖、配对稳定性和单 seed Binary AUPRC 红线未同时通过，因此严格判为 `R3_3W_NO_GO`。按停止线不执行 TEP、不开发 R4/R5，保留 R1/R2 并转入最终候选复核或止损。
