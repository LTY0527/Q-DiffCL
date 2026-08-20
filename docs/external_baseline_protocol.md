# External Baseline 公平比较协议

## 冻结项

FINAL_QDIFFCL 保持 `0.5D+0.5E`、`S=0`、critical ratio `0.30`、`t=1/5`，未重新调参。所有 augmentation-only 方法共享冻结 split、train-only preprocessing、window、TCN、Hard SupCon、Original batching、Linear Probe、threshold 与 evaluation。FINAL/Uniform 从既有 5-seed manifest 复用。

## Baseline 与来源

- NO_AUG：clean/clean 正视图。
- JITTER：逐观测高斯噪声，std=0.03。
- SCALING：逐通道、时间恒定缩放，std=0.05。
- JITTER_SCALING：上述两者组合。参数来自仓库冻结传统增强配置，无搜索。
- FreRA：官方仓库 `Tian0426/FreRA`，commit `7236fbfc1c665f83ed5f4364cad59093ee283c14`，官方 README 参数 `f_lr=0.001 / f_temperature=0.1 / l1=0.003`。采用 shared-backbone adaptation：保留官方可学习 stochastic frequency gate、自适应 modification 与 L1，仅适配 `[B,C,L]` 和 device；共享 TCN/Hard SupCon/probe。官方 method-native FCN+SimCLR 结果未混入公平主表。

## 运行阶段

- Stage A：3W seed 42、TEP seed 7 完整轮数 sanity。
- Stage B：3W `42/43/44`，TEP `7/42/2026`，Tier 1 + FreRA。
- Stage C：3W `42/43/44/45/46`，TEP `7/42/43/44/2026`；补齐 FINAL、Uniform、最强传统增强、FreRA。

所有新增方法与 FINAL 的公平哈希检查均通过：3W=True，TEP=True。test 未参与超参数选择或 FINAL 修改。
