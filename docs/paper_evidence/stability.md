# Stability Audit

## Five-seed paired stability

以下比较使用相同 seed 配对的 development evidence；不能替代预注册 outer test。

| Dataset | Method vs reference | Mean paired Δ Macro-F1 | Positive / non-worse | Worst seed (Δ) | LOSO range |
|---|---|---:|---:|---:|---:|
| 3W | FINAL_QDIFFCL vs FRERA | +0.0033 | 2/5 / 2/5 | 42 (-0.0436) | [-0.0213, +0.0151] |
| 3W | FINAL_QDIFFCL vs JITTER_SCALING | +0.0026 | 2/5 / 2/5 | 45 (-0.0307) | [-0.0067, +0.0109] |
| TEP | DCBR vs FINAL_QDIFFCL | +0.0121 | 5/5 / 5/5 | 43 (+0.0050) | [+0.0095, +0.0139] |
| TEP | DCBR vs SCALING | -0.0025 | 2/5 / 2/5 | 44 (-0.0068) | [-0.0035, -0.0014] |

3W 的两组 paired comparison 均只有 2/5 seeds 正向，且 LOSO 区间包含负值，因此不得宣称稳定优于 FreRA 或 Jitter+Scaling。TEP DCBR 相对 FINAL 为 5/5 正向；相对 SCALING 只有 2/5 正向，不支持稳定优越。
