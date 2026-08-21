# Q-DiffCL Paper-final Protocol

状态：`PAPER_FINAL_PROTOCOL_AMENDMENT_GO`。WindowRef coverage 修订发生在任何 outer training/metric 之前。

## 3W repeated grouped outer holdout（revised）

| Outer seed | Train WELL | Inner-val WELL | Outer-test WELL | WindowRef targets complete | Candidates checked |
|---:|---:|---:|---:|---|---:|
| 31001 | 20 | 8 | 8 | True | 1783 |
| 31002 | 20 | 8 | 8 | True | 526 |
| 31003 | 20 | 8 | 8 | True | 1848 |

Coverage 以正式 runner 的冻结 label mapping、transition exclusion 与 `WindowRef.target` 为准；每个 target 在 train/validation/test 均有可用窗口。WELL 完全不相交。

## TEP run-level nested grouped evaluation

TEP 的 248/72/80 Run splits 与修订前逐元素一致，Run 仍为最小分组单位。

## Fit scope

scaler、插补、D/E criticality、frequency statistics 仅 outer-train 拟合；rho、threshold、early stopping 仅 inner-validation；outer-test 只进行冻结评估。

## Frozen boundary

方法、baseline、model seeds、outer seeds、metrics、2,000 次 group bootstrap 均未改变；split 生成未使用模型性能。
