# Robustness / Sensitivity Audit

冻结设置仍为 critical ratio `0.30`、timesteps `1/5`；本审计不根据结果更换设置。

- critical-ratio 与 timestep 表仅证明 mask/budget 计算可复现；除冻结点外没有同协议下游训练结果，不能声称性能敏感性。
- limited-data 25/50/100% 已完成 grouped sampling dry-run 与 group hash，但尚无模型性能结果。
- TEP 现有结果覆盖固定 MCAR 30%；3W 保持 native missingness。MCAR 10% 仍缺失。
- 所有缺失格均在 CSV 标记 `UNSUPPORTED / DO NOT CLAIM`，paper-final outer test 未运行。

原始审计表：`docs/paper_evidence/robustness_sensitivity.csv`。
