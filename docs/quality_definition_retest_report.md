# 去偏质量定义复测报告

> **QUALITY_DEFINITION_AUDIT / SINGLE_SEED / FIXED_DIFFUSION_VIEW / NOT_FOR_PAPER_CLAIMS**

## 最终状态

```text
QUALITY_DEFINITION_REDESIGN_NO_GO
前置状态：QUALITY_DEFINITION_AUDIT_NO_GO
训练复测：SKIPPED_BY_TRAIN_ONLY_AUDIT_GATE
```

Q1/Q2 均未满足“相比 Q0 明显减小 normal/fault 平均 q 差”和“不再把 fault 整体压到低权重”的前置要求。按实验协议，本轮没有运行 Hard/Q0/Q1/Q2 新训练，也没有根据 test 指标调整映射。

## 计划结果表

| 方法 | Macro-F1 | AUPRC | Fault Recall | FAR | q-label 偏置 | q-semantic 相关 | 时间 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hard SupCon | 未运行 | 未运行 | 未运行 | 未运行 | N/A | N/A | 0 |
| Q0 Abs-MAE | 未运行 | 未运行 | 未运行 | 未运行 | -0.5269 | -0.5598 | 0 |
| Q1 Relative Gain | 审计未通过 | 审计未通过 | 审计未通过 | 审计未通过 | -0.3475 | -0.4037 | 0 |
| Q2 Relative + Semantic | 审计未通过 | 审计未通过 | 审计未通过 | 审计未通过 | -0.2107 | -0.0831 | 0 |

这里的“未运行”是防止无效候选消耗一次单 seed 训练机会的设计结果，不是缺失实验。

## 历史参照（非本轮重复实验）

上一轮完全相同固定 views 的已提交结果仅用于说明 Q0 的既有表现：

| 历史方法 | Macro-F1 | AUPRC | Fault Recall | FAR |
|---|---:|---:|---:|---:|
| Hard SupCon | 0.893753 | 0.928673 | 0.796809 | 0.025781 |
| Q0 Abs-MAE | 0.883611 | 0.925578 | 0.791489 | 0.039062 |

本轮 loss-scale 审计已证明 Q0/Hard 平均 batch loss 比例为 0.999915，归一化后有效权重均值为 1；因此历史 Q0 变差不是整体梯度随 mean(q) 缩小造成。

## 为何不训练 Q1/Q2

- Q0 train normal/fault 均值差：0.070642
- Q1 train normal/fault 均值差：0.143094，较 Q0 扩大 102.6%
- Q2 train normal/fault 均值差：0.076499，较 Q0 扩大 8.3%
- Q1 fault mean 0.383401，normal mean 0.526495
- Q2 fault mean 0.369610，normal mean 0.446110
- fault 内高低组仍主要按 fault type 难度分离

虽然 Q2 的 q-label Spearman 降至 -0.2107，并使 q-semantic 相关接近 0，但这来自更宽的权重分布与语义组合，没有满足原始均值去偏要求。继续训练会再次系统性降低 fault 正样本权重，且无法区分性能变化来自“质量”还是 fault type 重加权。

## 公平性和安全性

- 固定 checkpoint/view/manifest/mask/seed 均未改变
- diffusion 未训练、views 未重生成
- 教师保持冻结
- Q1 median/IQR 只由 train 拟合
- Q2 不使用真实标签
- validation/test 未参与门控参数拟合
- 训练入口读取审计结果并返回 `training_skipped=true`

## GO / NO-GO

Q1/Q2 仍强烈编码 normal/fault 和 fault type，且没有通过训练前去偏门槛，满足 `QUALITY_DEFINITION_REDESIGN_NO_GO` 条件。停止无参考质量评分器开发，并重新评估是否保留“质量加权”作为核心创新。

## 唯一下一步建议

回到核心方法选择，使用现有固定结果比较“完全不做质量加权的 Hard SupCon”与其他不依赖恢复质量标量的视图使用方式；在新的机制假设获得独立证据前，不再设计或训练质量评分器。
