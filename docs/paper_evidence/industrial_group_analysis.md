# Industrial Group Analysis

本报告使用已经看过的 development test checkpoint 做只读分组重放，不参与任何新选择；不能称为 untouched paper-final evidence。

## 3W cross-WELL

| WELL | FINAL Macro-F1 | FINAL FAR |
|---|---:|---:|
| WELL-00020 | 0.0476 | 1.0000 |
| WELL-00014 | 0.2546 | 0.6264 |
| WELL-00012 | 0.4213 | 0.1641 |
| WELL-00019 | 0.5320 | 0.7902 |
| WELL-00031 | 0.5970 | 0.7036 |
| WELL-00011 | 0.6328 | 0.3358 |
| WELL-00033 | 1.0000 | 0.0000 |
| WELL-00035 | 1.0000 | 0.0000 |

最困难 WELL 为 `WELL-00020`（observed-class Macro-F1 `0.0476`）。FINAL 相对 FreRA 共救回 `14430` 个窗口、丢失 `8691`；相对 JITTER_SCALING 为 `6048` / `4959`。WELL bootstrap CI 多数跨零，因此只支持“部分 hard WELL 有收益”，不支持 universal cross-WELL improvement。

Class 2/8/9 的逐 seed F1/Recall 已保存到 `3w_difficult_classes.csv`；FINAL 五 seed 平均 F1 分别为 `0.5426`、`0.8305`、`0.1043`。Class 9 仍是明显困难类。

## TEP fault-wise

| Fault | FINAL recall | DCBR recall | SCALING recall | Δ DCBR-FINAL |
|---:|---:|---:|---:|---:|
| 1 | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| 2 | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| 3 | 0.1106 | 0.0766 | 0.0596 | -0.0340 |
| 4 | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| 5 | 0.9319 | 1.0000 | 1.0000 | +0.0681 |
| 6 | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| 7 | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| 8 | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| 9 | 0.0319 | 0.0085 | 0.0043 | -0.0234 |
| 10 | 0.6511 | 0.6298 | 0.6191 | -0.0213 |
| 11 | 0.9915 | 0.9979 | 0.9979 | +0.0064 |
| 12 | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| 13 | 0.9915 | 0.9957 | 0.9957 | +0.0043 |
| 14 | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| 15 | 0.0362 | 0.0106 | 0.0021 | -0.0255 |
| 16 | 0.6936 | 0.6511 | 0.6404 | -0.0426 |
| 17 | 0.9787 | 0.9830 | 0.9830 | +0.0043 |
| 18 | 0.9766 | 0.9809 | 0.9830 | +0.0043 |
| 19 | 0.9894 | 0.9553 | 0.9383 | -0.0340 |
| 20 | 0.9936 | 0.9915 | 0.9915 | -0.0021 |

DCBR 对 fault 5 recall 改善最大，对 3/9/15/16/19 有退化；其主要证据仍是全局 Macro-F1 稳定改善和部分 delay 缩短，而不是所有 fault 一致提高。相对 SCALING 的 fault-group recall effect 为正，但 F1 CI 跨零。

## Five-seed paired stability

以下比较使用相同 seed 配对的 development evidence；不能替代预注册 outer test。

| Dataset | Method vs reference | Mean paired Δ Macro-F1 | Positive / non-worse | Worst seed (Δ) | LOSO range |
|---|---|---:|---:|---:|---:|
| 3W | FINAL_QDIFFCL vs FRERA | +0.0033 | 2/5 / 2/5 | 42 (-0.0436) | [-0.0213, +0.0151] |
| 3W | FINAL_QDIFFCL vs JITTER_SCALING | +0.0026 | 2/5 / 2/5 | 45 (-0.0307) | [-0.0067, +0.0109] |
| TEP | DCBR vs FINAL_QDIFFCL | +0.0121 | 5/5 / 5/5 | 43 (+0.0050) | [+0.0095, +0.0139] |
| TEP | DCBR vs SCALING | -0.0025 | 2/5 / 2/5 | 44 (-0.0068) | [-0.0035, -0.0014] |

3W 的两组 paired comparison 均只有 2/5 seeds 正向，且 LOSO 区间包含负值，因此不得宣称稳定优于 FreRA 或 Jitter+Scaling。TEP DCBR 相对 FINAL 为 5/5 正向；相对 SCALING 只有 2/5 正向，不支持稳定优越。
