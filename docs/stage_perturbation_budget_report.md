# 故障阶段显式扰动预算 MVP

> **STAGE_EFFECT_AUDIT / STAGE_PERTURBATION_BUDGET_MVP / FIXED_R1_BASELINE / EXPLORATORY_TEP_SUBSET / NOT_FOR_PAPER_FINAL_CLAIMS**

## 结论

Stage Effect Audit 为 `STAGE_TIMESTEP_EFFECT_WEAK`，因此执行唯一固定 beta：normal/early/middle/stable=`1.0/0.6/0.8/1.0`。Seed 7 状态：`STAGE_PERTURBATION_BUDGET_SEED7_NO_GO`；最终：`STAGE_PERTURBATION_BUDGET_SEED7_NO_GO`；3-Seed 完成：`False`。

| Seed | 方法 | Macro-F1 | AUPRC | Recall | FAR | Early | Delay |
|---:|---|---:|---:|---:|---:|---:|---:|
| 7 | R1 | 0.8920 | 0.9316 | 0.7979 | 0.0297 | 0.7937 | 103.00 |
| 7 | B3 | 0.8932 | 0.9285 | 0.7894 | 0.0207 | 0.7500 | 110.53 |

B3−R1：Macro-F1 `+0.00121`，AUPRC `-0.00314`，Recall `-0.00851`，FAR `-0.00898`，Early Recall `-0.04375`，Delay `+7.53`。`ΔFAR<0`、`ΔDelay<0` 才表示改善。

| 方法 | Overall L1 | Normal | Early | Middle | Stable | Critical L1 | Noncritical L1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| R1 | 0.05208 | 0.05577 | 0.04972 | 0.04843 | 0.04696 | 0.03081 | 0.04731 |
| B3 | 0.04927 | 0.05577 | 0.02983 | 0.03874 | 0.04696 | 0.02904 | 0.04459 |

## Fisher 与表征距离

| 方法 | Critical Fisher retention | Overall repr L2 | Normal repr L2 | Early repr L2 | Middle repr L2 | Stable repr L2 |
|---|---:|---:|---:|---:|---:|---:|
| R1 | 0.99651 | 0.01871 | 0.00748 | 0.02793 | 0.03346 | 0.03159 |
| B3 | 0.99772 | 0.01699 | 0.00748 | 0.01687 | 0.02697 | 0.03159 |

预算顺序有效：`True`。Seed 7 Gate：`{'core_preserved': True, 'early_or_delay_gain': False, 'budget_order_valid': True}`；3-Seed Gate：`{'three_seed_skipped': True}`。beta 只缩放训练 R1 residual，不进入 validation threshold、encoder、Probe 或 test 推理。

B3 虽提高 Macro-F1 并降低 FAR，但 Early Recall 明显下降且 Delay 恶化，未满足工业收益条件。当前 TEP test 已多轮查看，本结果仍是探索性而非论文最终无偏结论。NO-GO 后彻底停止 C3，不搜索新 beta/stage/horizon，不增加 C4/C5；下一步冻结 R1 并转向第二数据集、新未触碰协议、第二种退化与强基线。
