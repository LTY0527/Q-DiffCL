# 3W R1/R2 五随机种子稳定性复核

最终判定：R1 `R1_5SEED_STABLE_CANDIDATE`；R2 `R2_EXTENSION_ONLY`。

本轮只新增 seeds 45/46 的 Uniform、R1、R2，共 6 个训练；seeds 42/43/44 的 9 个结果直接复用。未运行 R3、TEP，也未改变权重、timestep、mask ratio、sampler、TCN、loss、split 或 probe。

## Macro-F1 配对稳定性

- R1-UNIFORM Macro-F1：`+0.02659 ± 0.01455`，median `+0.03380`，range `[-0.00076, +0.03945]`，positive `4/5`
- R2-UNIFORM Macro-F1：`+0.02744 ± 0.05979`，median `+0.01877`，range `[-0.04762, +0.13462]`，positive `4/5`
- R2-R1 Macro-F1：`+0.00085 ± 0.05767`，median `-0.00614`，range `[-0.08142, +0.09517]`，positive `2/5`

| Seed | R1−Uniform | R2−Uniform | R2−R1 |
|---:|---:|---:|---:|
| 42 | +0.02447 | +0.00158 | -0.02289 |
| 43 | +0.03380 | -0.04762 | -0.08142 |
| 44 | +0.03945 | +0.13462 | +0.09517 |
| 45 | -0.00076 | +0.01877 | +0.01952 |
| 46 | +0.03600 | +0.02986 | -0.00614 |

去掉 R1 最佳增益 seed 后，R1−Uniform Macro-F1 mean 为 `+0.02338`；最大正增益 seed 占全部正增益 `29.5%`。

## 其他核心指标的五 seed 配对均值

| Comparison | Macro Recall | Binary AUPRC | Multiclass AUPRC | FAR | Early Recall | Delay (s) |
|---|---:|---:|---:|---:|---:|---:|
| R1-UNIFORM | +0.04405 | -0.00753 | -0.00327 | -0.05839 | -0.01368 | +20.43 |
| R2-UNIFORM | +0.04750 | +0.07466 | +0.03640 | -0.01042 | +0.01861 | -185.26 |
| R2-R1 | +0.00344 | +0.08219 | +0.03967 | +0.04797 | +0.03228 | -205.69 |

R1 的 Multiclass AUPRC mean 为轻微负向，但 3/5 seed 非负、median 为 `+0.00044`，不构成多数 seed 系统性反向；FAR mean 明显改善。R2 的两项 AUPRC 相对 Uniform 均为正，但相对 R1 的 Macro-F1 只有 2/5 正向且 FAR mean 恶化，因此不升级为主候选。

## Class 9

- Uniform Recall/F1：`0.09867 ± 0.11865` / `0.05855 ± 0.08985`
- R1 Recall/F1：`0.17149 ± 0.20749` / `0.07363 ± 0.09220`
- R2 Recall/F1：`0.18428 ± 0.22226` / `0.08641 ± 0.12107`

三种方法的 Class 9 Recall median 均仅约 `0.01`，且 R1/R2 的 std 都高于 mean。Class 9 仍高度不稳定，较高均值由少数 seed 主导，不能作为稳定改善结论。

## 候选结论

R1 gate：{'macro_f1_positive_at_least_4of5': True, 'macro_f1_mean_gain': True, 'multiclass_auprc_not_systematically_reversed': True, 'far_not_systematically_reversed': True, 'not_driven_only_by_best_seed': True}。

R2 gate：{'macro_f1_majority_positive_vs_uniform': True, 'macro_f1_at_least_3of5_nonworse_vs_r1': False, 'binary_auprc_mean_advantage_vs_uniform': True, 'binary_auprc_majority_nonworse_vs_uniform': True, 'multiclass_auprc_mean_advantage_vs_uniform': True, 'multiclass_auprc_majority_nonworse_vs_uniform': True}。

`paper_final_protocol_design_allowed = true`。即使允许进入下一阶段，当前 3W/TEP test 已参与方法开发，不能作为 paper-final claim；下一阶段只能设计新的独立 paper-final protocol。
