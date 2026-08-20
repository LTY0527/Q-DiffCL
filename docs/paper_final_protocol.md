# Q-DiffCL Paper-final Protocol

状态：`PAPER_FINAL_PROTOCOL_DRY_RUN_GO`。本阶段仅完成协议锁定与 dry-run；没有训练 outer 模型，也没有读取 outer-test 指标。

## 冻结方法

- FINAL_QDIFFCL：`0.5D + 0.5E`，critical ratio `0.30`，timesteps `1/5`，soft allocation、TCN、Hard SupCon、Original batching、frozen Linear Probe。
- DCBR：domain-level validation-calibrated `rho ∈ {0,.25,.5,.75,1}`；不学习 controller，推理新增参数为 0。
- primary metric：Macro-F1；secondary：AUPRC、FAR、Early Recall、Detection Delay、per-group performance。

## 3W repeated grouped outer holdout

| Outer seed | Train groups | Inner-val groups | Outer-test groups | Disjoint |
|---:|---:|---:|---:|---|
| 31001 | 20 | 8 | 8 | True |
| 31002 | 20 | 8 | 8 | True |
| 31003 | 20 | 8 | 8 | True |

同一 WELL 严禁跨 outer-train、inner-validation、outer-test。每个 outer split 使用 20/8/8 WELL，inner validation 只校准 rho、threshold 和 early stopping。

## TEP run-level nested grouped evaluation

| Outer seed | Train groups | Inner-val groups | Outer-test groups | Disjoint |
|---:|---:|---:|---:|---|
| 32001 | 248 | 72 | 80 | True |
| 32002 | 248 | 72 | 80 | True |
| 32003 | 248 | 72 | 80 | True |

Run 是最小分组单位；同一 Run 的窗口绝不跨 split。各 fault type 与 normal 分层分配。

## Fit scope 与 leakage rule

scaler、插补、feature/criticality D/E、frequency statistics 仅由 outer-train 拟合；rho、threshold、early stopping 仅使用 inner validation；outer-test 只进行一次冻结评估。任何 outer-test 后的算法、候选网格或阈值修改均禁止。

## Seeds 与统计

- 3W model seeds：`[42, 43, 44, 45, 46]`。
- TEP model seeds：`[7, 42, 43, 44, 2026]`。
- 报告 mean±std、paired delta、positive/non-worse count、worst seed、LOSO 与 2,000 次 WELL/Run bootstrap 95% CI。

## Final freeze statement

当前长期开发 test 不得称 untouched。只有此 manifest 中预注册、未参与任何后续选择的 outer groups 才可作为 paper-final evaluation；outer 结果产生后项目进入只分析、不改算法状态。
