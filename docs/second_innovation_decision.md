# 第二创新决策

## 决策：C

3W 上 FINAL 与最强外部 baseline 基本持平（FINAL `0.5396` vs JITTER_SCALING `0.5370`）；TEP 上 SCALING 以 `+0.0145` Macro-F1、`100%` seed 正向稳定超过 FINAL，并且 AUPRC/FAR 同向更优。因此不是 A；TEP 的预注册清晰差距超过 `0.01`，判为 C 而非 B。

差距模式表明问题更像 **数据集相关的扰动分配/是否应施加扰动**，而不是回头调整 D/E 或关键频率：FINAL 在 3W 保持竞争力，但 TEP 的轻量 SCALING 与 NO_AUG 均优于 diffusion。固定非零 matched budget 对 TEP 可能过强，当前 allocator 缺少接近零预算或按域收缩的能力。

## 下一步

建议建立独立分支 `exp/budget-constrained-allocation-v2`，研究 Budget-Constrained Semantic Perturbation Allocation，并把“可收缩到零的连续预算”作为验证重点。当前轮不实现 v2，不修改 FINAL，不根据 test 重调 D/E、ratio 或 timestep。下一阶段应仅用 train/validation 学习 allocation，再锁定后做双数据集 paired 5-seed 验证。
