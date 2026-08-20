# Paper Mechanism Ablation — Validation Only

所有方法使用相同 split、初始化、batch order、TCN、Hard SupCon 和 Probe。3W seeds `42/43/44`，TEP seeds `7/42/2026`；未读取 test。

## 3W

| Variant | Macro-F1 | AUPRC | FAR | Early Recall | Expected budget | Measured L1 (domain) |
|---|---:|---:|---:|---:|---:|---:|
| UNIFORM_DIFFUSION | 0.4258 ± 0.0973 | 0.5992 ± 0.0617 | 0.3039 ± 0.0338 | 0.9095 ± 0.0651 | 0.01797364 | 0.0007 (time) |
| HARD_MASK_SELECTIVE | 0.4462 ± 0.0978 | 0.6468 ± 0.0362 | 0.3088 ± 0.0338 | 0.9030 ± 0.0442 | 0.01797364 | 0.0007 (time) |
| SOFT_MASK_SELECTIVE | 0.4870 ± 0.0824 | 0.6455 ± 0.0559 | 0.2352 ± 0.1426 | 0.8845 ± 0.0771 | 0.01797364 | 0.0007 (time) |
| SOFT_MASK_WO_BUDGET_MATCH | 0.4358 ± 0.0918 | 0.6283 ± 0.0569 | 0.3238 ± 0.0600 | 0.8414 ± 0.0957 | 0.02156851 | 0.0007 (time) |

## TEP

| Variant | Macro-F1 | AUPRC | FAR | Early Recall | Expected budget | Measured L1 (domain) |
|---|---:|---:|---:|---:|---:|---:|
| UNIFORM_DIFFUSION | 0.9727 ± 0.0027 | 0.9955 ± 0.0003 | 0.0458 ± 0.0019 | 0.9958 ± 0.0036 | 0.01797364 | 0.0566 (frequency) |
| HARD_MASK_SELECTIVE | 0.9724 ± 0.0021 | 0.9955 ± 0.0004 | 0.0432 ± 0.0017 | 0.9958 ± 0.0036 | 0.01797364 | 0.0548 (frequency) |
| SOFT_MASK_SELECTIVE | 0.9700 ± 0.0068 | 0.9953 ± 0.0004 | 0.0465 ± 0.0013 | 0.9812 ± 0.0272 | 0.01797364 | 0.0568 (frequency) |
| SOFT_MASK_WO_BUDGET_MATCH | 0.9729 ± 0.0024 | 0.9955 ± 0.0003 | 0.0465 ± 0.0026 | 0.9958 ± 0.0036 | 0.02079784 | 0.0611 (frequency) |

## Paired Soft matched delta

| Dataset | Reference | ΔMacro-F1 | Positive seeds | ΔAUPRC | ΔFAR |
|---|---|---:|---:|---:|---:|
| 3W | UNIFORM_DIFFUSION | +0.0612 | 3/3 | +0.0463 | -0.0687 |
| 3W | HARD_MASK_SELECTIVE | +0.0409 | 3/3 | -0.0013 | -0.0736 |
| 3W | SOFT_MASK_WO_BUDGET_MATCH | +0.0512 | 3/3 | +0.0172 | -0.0886 |
| TEP | UNIFORM_DIFFUSION | -0.0028 | 2/3 | -0.0001 | +0.0007 |
| TEP | HARD_MASK_SELECTIVE | -0.0024 | 2/3 | -0.0002 | +0.0033 |
| TEP | SOFT_MASK_WO_BUDGET_MATCH | -0.0029 | 2/3 | -0.0002 | -0.0000 |

Uniform、Hard 和 Soft matched 的 expected total spectral budget 数值相等：`True`。Unmatched 仅移除全局预算匹配，保留相同 soft allocation 和 timestep map。Fairness hash 对齐：`True`。
