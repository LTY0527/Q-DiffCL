# Paper Mechanism Evidence

## A1 Core mechanism

### 3W

| Method | Macro-F1 | AUPRC | FAR | Expected budget |
|---|---:|---:|---:|---:|
| UNIFORM_DIFFUSION | 0.4258 ± 0.0973 | 0.5992 | 0.3039 | 0.01797364 |
| HARD_MASK_SELECTIVE | 0.4462 ± 0.0978 | 0.6468 | 0.3088 | 0.01797364 |
| SOFT_MASK_SELECTIVE | 0.4870 ± 0.0824 | 0.6455 | 0.2352 | 0.01797364 |
| SOFT_MASK_WO_BUDGET_MATCH | 0.4358 ± 0.0918 | 0.6283 | 0.3238 | 0.02156851 |

### TEP

| Method | Macro-F1 | AUPRC | FAR | Expected budget |
|---|---:|---:|---:|---:|
| UNIFORM_DIFFUSION | 0.9727 ± 0.0027 | 0.9955 | 0.0458 | 0.01797364 |
| HARD_MASK_SELECTIVE | 0.9724 ± 0.0021 | 0.9955 | 0.0432 | 0.01797364 |
| SOFT_MASK_SELECTIVE | 0.9700 ± 0.0068 | 0.9953 | 0.0465 | 0.01797364 |
| SOFT_MASK_WO_BUDGET_MATCH | 0.9729 ± 0.0024 | 0.9955 | 0.0465 | 0.02079784 |

三 seed validation-only 公平审计已通过。3W 上 Soft matched 相对 Uniform、Hard 和 unmatched 分别为 `+0.0612/+0.0409/+0.0512` Macro-F1，均 3/3 seeds 正向；TEP 上对应差值均约为 `-0.0024~-0.0029`。因此 soft allocation/budget matching 只支持“对 3W 有明确收益”的数据集依赖表述，不支持跨数据集普遍优越。

## A2 Semantic components

| Dataset | Method | Macro-F1 | FAR | Early Recall |
|---|---|---:|---:|---:|
| 3W | UNIFORM | 0.4542 ± 0.0788 | 0.4288 | 0.8498 |
| 3W | D_ONLY | 0.5128 ± 0.0365 | 0.3749 | 0.8899 |
| 3W | E_ONLY | 0.4628 ± 0.0937 | 0.3841 | 0.8302 |
| 3W | FINAL_DE | 0.5188 ± 0.0238 | 0.3555 | 0.8879 |
| TEP | UNIFORM | 0.8846 ± 0.0044 | 0.0383 | 0.7708 |
| TEP | D_ONLY | 0.8872 ± 0.0059 | 0.0327 | 0.7688 |
| TEP | E_ONLY | 0.8837 ± 0.0050 | 0.0428 | 0.7750 |
| TEP | FINAL_DE | 0.8861 ± 0.0052 | 0.0355 | 0.7708 |

3W 中 D_ONLY 明显强于 E_ONLY，D 是主要 discriminative contributor；E 仅可描述为 complementary early-fault prior。TEP 的组件差异很小，不支持“D/E 同等必要”。

## A3 DCBR

见 `dcbr_extension_ablation.csv`。3W validation 选择 `rho=1`，保持 FINAL；TEP 选择 `rho=.75`，现有 development locked evidence 中相对 FINAL Macro-F1 `+0.0121`，但仍略低于 SCALING。GLOBAL_RHO_075 仅是 validation mechanism reference，不能与 locked-test 行直接做统计检验。
