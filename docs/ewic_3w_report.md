# EWIC 3W Early Detection 报告

结论：`3W_EWIC_NO_GO`。

## 方法

EWIC 保留 R1 的 run/WELL-level `D` 与 `S`，仅把单一 `E=Fisher(Normal, Early)` 替换为八个 onset-relative `E_h`。每个 `E_h` 独立 Median/IQR robust normalization，按归一化后的 `exp(-0.35(h-1))` 加权得到 `E_lead`；随后用 64 次 train-WELL bootstrap 的 Top-30% selection probability 构造 `E_invariant=E_lead×R_early`。最终仍为 `0.5D+0.3E_invariant+0.2S`。

- EWIC−R1 Early Recall：`+0.05172 ± 0.06465`，3/3 positive
- EWIC−R1 FAR：`-0.01304`
- EWIC−R1 Detection Delay：`-108.94` 秒，1/3 seed 缩短
- Binary AUPRC / Fault Recall：`+0.13764` / `+0.04976`

| Seed | Δ Early Recall | Δ FAR | Δ Delay(s) |
|---:|---:|---:|---:|
| 42 | +0.00903 | +0.10634 | -721.45 |
| 43 | +0.00304 | +0.00152 | +13.71 |
| 44 | +0.14308 | -0.14696 | +380.93 |

| Fixed-FAR OP | Δ Early Recall | Δ Delay(s) | Δ observed test FAR |
|---|---:|---:|---:|
| far_1pct | +0.23178 | +194.79 | +0.01052 |
| far_5pct | +0.20827 | -240.00 | -0.02186 |

Fault 2/8/9 与 WELL 分布见 JSON。h1–h2 优势最明显的 bins：c17/f6, c17/f7, c17/f10, c17/f20, c17/f13；只在 h7–h8 更强的 bins：c21/f1, c21/f2, c21/f3, c21/f4, c21/f25。Reliability 将 `8` 个原始 lead Top-30% bins 压到 `<0.5`。Mask Jaccard=`0.78689`，changed bins=`52`。Gate：`{'early_recall_majority_positive': True, 'delay_majority_shorter': False, 'far_not_systematically_worse': True, 'fixed_far_direction_consistent': False, 'not_single_fault_class': False}`。
