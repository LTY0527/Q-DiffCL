# Budget-Constrained Allocation v2 Summary

## Stage A：`NO_GO_SAFE_CAPACITY_DIRECTION`

参数无关 Safe Capacity 在 train-only 数据上给出 3W mean rho `0.470266`、TEP `0.602268`，方向差 `-0.132002`。该方向与 Budget Shrinkage Diagnostic 的 3W 中高预算、TEP 低预算需求相反。

依照预注册硬门，本阶段在 Stage A 停止：未将 sample-adaptive variance 接入 diffusion，未运行 Stage C 单 seed 或 Stage D 三 seed，未读取 validation/test 指标，也未修改 FINAL_QDIFFCL。

最终判定：`NO_GO_BUDGET_CONSTRAINED_ALLOCATION_V2`。
