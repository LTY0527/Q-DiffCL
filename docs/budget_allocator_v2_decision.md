# Budget-Constrained Allocation v2 Decision

## NO_GO_BUDGET_CONSTRAINED_ALLOCATION_V2

当前定义 `rho(x)=1-critical_energy_ratio(x)` 在 train-only 数据上产生相反的 domain ordering：3W - TEP mean rho 为 `-0.132002`。允许的 `gamma={0.5,1,2}` 均不能反转该方向，因此不能用 validation/test 性能为其打补丁。

停止当前 controller；保留冻结 FINAL 与 Budget Shrinkage 结论。Stage B/C/D 未执行。
