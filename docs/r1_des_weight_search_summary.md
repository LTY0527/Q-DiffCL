# R1 DES 权重 Validation 搜索与最终冻结

本轮只使用 canonical train split 构建 mask，并且只使用 validation 指标筛选与冻结；未计算或读取 test 指标参与排序。

## Stage 1：12 候选单种子筛选

| Rank | Candidate | FAR gate | Relative Macro gain | FAR delta | Early delta | AUPRC delta |
|---:|---|---|---:|---:|---:|---:|
| 1 | DE_60_40 | True | +8.0904% | -0.0485 | +0.0008 | +0.0578 |
| 2 | DE_50_50 | True | +1.5700% | -0.0476 | -0.0288 | +0.0603 |
| 3 | ES | True | +1.0992% | -0.0476 | -0.0438 | +0.0298 |
| 4 | LOW_D | True | -2.4192% | -0.0039 | -0.0264 | +0.0417 |
| 5 | D_HEAVY_2 | True | -2.5404% | +0.0005 | -0.0163 | +0.0617 |
| 6 | D_HEAVY_1 | True | -3.3026% | -0.0017 | -0.0225 | +0.0423 |
| 7 | E_HEAVY_1 | False | +1.2579% | -0.0396 | -0.0089 | +0.0597 |
| 8 | DE_HEAVY | False | +1.0835% | -0.0393 | -0.0282 | +0.0571 |
| 9 | DS | False | +0.4001% | +0.0081 | +0.0428 | +0.0632 |
| 10 | BALANCED | False | -0.0534% | -0.0108 | +0.0232 | +0.0613 |
| 11 | E_HEAVY_2 | False | -1.5792% | +0.0066 | +0.0062 | +0.0045 |

Top-3：`DE_60_40`、`DE_50_50`、`ES`。

## Stage 2：Top-3 + CURRENT 三随机种子

| Variant | 3W Macro-F1 | Δ | 3W FAR | Δ | 3W Early | TEP Macro-F1 | Δ | TEP FAR | Δ | TEP Early |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CURRENT | 0.4388 ± 0.1029 | +0.0000 | 0.3380 ± 0.0864 | +0.0000 | 0.8891 ± 0.0151 | 0.9103 ± 0.0222 | +0.0000 | 0.0413 ± 0.0215 | +0.0000 | 0.8458 ± 0.0940 |
| DE_60_40 | 0.4015 ± 0.1312 | -0.0372 | 0.3873 ± 0.1299 | +0.0493 | 0.9068 ± 0.0199 | 0.9102 ± 0.0220 | -0.0002 | 0.0413 ± 0.0198 | -0.0000 | 0.8479 ± 0.0973 |
| DE_50_50 | 0.4572 ± 0.1019 | +0.0184 | 0.3014 ± 0.0366 | -0.0366 | 0.8987 ± 0.0514 | 0.9103 ± 0.0221 | -0.0000 | 0.0420 ± 0.0228 | +0.0007 | 0.8458 ± 0.0977 |
| ES | 0.4507 ± 0.0979 | +0.0119 | 0.2945 ± 0.0419 | -0.0435 | 0.8814 ± 0.0520 | 0.9104 ± 0.0222 | +0.0000 | 0.0402 ± 0.0196 | -0.0011 | 0.8438 ± 0.0944 |

表中 Δ 均为相对 CURRENT 的同 seed paired mean；FAR 越低越好。完整 AUPRC 和每 seed 数据见 CSV。

## 一次性最终决策

- DE_60_40：relative Macro gain -4.2532%；Macro floor=False，FAR=False，consistency=False，clear gain=False，最终 eligible=False。
- DE_50_50：relative Macro gain +2.0963%；Macro floor=True，FAR=True，consistency=True，clear gain=True，最终 eligible=True。
- ES：relative Macro gain +1.3530%；Macro floor=True，FAR=True，consistency=True，clear gain=True，最终 eligible=True。

最终冻结：`DE_50_50`，D/E/S = `0.500/0.500/0.000`。
决策原因：validation gates passed。

权重自此停止调参。后续 test、external baseline 或 reliability 结果不得触发再次改权重。

## 审计

- validation-only：True；test metrics read：False。
- candidate mask 跨 seed 固定：True。
- initialization / batch order / preprocessing 公平性哈希一致：True。
- 最大 matched-budget error：3.725e-09。

现有 R1 test 结果仅保留为历史开发证据，未参与本轮候选排序。下一步进入 external baseline、5-seed reliability 与 paper-final 实验。
