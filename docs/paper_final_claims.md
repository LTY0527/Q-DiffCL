# Q-DiffCL Paper-final Claims

## SAFE TO CLAIM

- Outer 结果不支持跨两个数据集都成立的无条件性能优越性表述；可安全陈述已完成冻结 nested/grouped evaluation。

## DATASET-DEPENDENT CLAIM

- 3W: FINAL 与 NO_AUG 的差异不确定（paired Δ NO_AUG-FINAL=-0.0238, 95% CI [-0.0376, +0.0069]）。
- TEP: FINAL 与 NO_AUG 的差异不确定（paired Δ NO_AUG-FINAL=-0.0003, 95% CI [-0.0012, +0.0006]）。
- selective/soft matched-budget mechanism 的优势仍是 3W 支持、TEP 不一致；DCBR 的作用按数据集分别表述。

## DEVELOPMENT EVIDENCE ONLY

- 2×2 contrastive interaction、critical-ratio sensitivity、TEP onset trajectory 与机制 ablation 没有在 outer matrix 重跑。

## LIMITATION

- soft allocation 跨数据集不一致；critical_ratio=0.30 不是 universal optimum。
- limited-data 与更广 missingness robustness 未完成；FRERA augmentation-only timing 缺失。
- AutoDA 仅 method-native supplementary；DiCL 存在公平复现缺口。

## DO NOT CLAIM

- 不宣称 universal Soft superiority、universal cross-WELL superiority、0.30 universal optimum，或未评估的 robustness。
