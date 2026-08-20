# Budget Shrinkage 第二创新判定

## GO_BUDGET_CONSTRAINED_ALLOCATION_V2

3W 满预算相对低预算的 validation Macro-F1 配对均值为 `+0.0221`（`2/3` 正向）；TEP 低预算相对满预算为 `+0.0624`（`3/3` 正向）。TEP 的 FAR 代价 `+0.0022` 未超过预注册上限 `0.02`，Early Recall 变化 `+0.1500`。

因此支持下一阶段开发独立的 `exp/budget-constrained-allocation-v2`：先仅由 train/validation 估计可收缩至 0 的 domain-safe 总预算，再由冻结 D/E 决定预算投放位置。当前阶段不实现 allocator、不修改 FINAL、不为 3W/TEP 人工冻结不同 rho，也不细搜额外 rho。
