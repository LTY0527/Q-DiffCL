# 质量定义审计报告

> **QUALITY_DEFINITION_AUDIT / SINGLE_SEED / FIXED_DIFFUSION_VIEW / NOT_FOR_PAPER_CLAIMS**

## 审计结论

train-only 审计状态为 **`QUALITY_DEFINITION_AUDIT_NO_GO`**。现有 Q0 并不存在“平均 q 较低导致整体 SupCon loss/梯度同比缩小”的实现错误；Q1 相对插值改进和 Q2 相对改进+教师语义均未使 normal/fault 的原始平均权重差小于 Q0，且 fault 平均权重仍比 normal 低超过 0.05。因此两者均不具备进入训练复测的资格。

本报告使用标签、fault type 和 run_uid 只做偏置审计。Q0/Q1/Q2 的质量公式均不接收标签或 fault type，validation/test 也不参与参数拟合。

## 固定条件

- 固定视图：train 6,704、validation 1,936、test 4,440
- diffusion checkpoint SHA-256：`74ae41ca8bf45fc284557be7fa6c0859caf0e91a8ab9e642cdf5f75eeae9a22c`
- seed：7；MCAR：30%；DDPM sampling seed：3007
- 冻结教师：`outputs/rapid_idea_validation/G1_0.pt`
- 未重训 diffusion、未重生成 view/mask/manifest

## 质量权重与损失尺度审计

当前实现对每个 anchor 的正样本候选权重按正权重和归一化，即先令有效正权重均值为 1，再对正样本 log-probability 求均值。这与“加权和除以权重和”保持总尺度的写法等价，不是 `mean(q_i * loss_i)`。

| 项目 | Hard | Q0 Abs-MAE | Q0/Hard |
|---|---:|---:|---:|
| 完整 train 平均 batch loss | 5.521287 | 5.520816 | 0.999915 |
| 平均梯度范数 | 0.038434 | 0.045945 | 1.195422 |

- recovered q 均值：0.344535
- 正样本归一化前有效权重均值：0.672268
- 正样本归一化后有效权重均值：1.000000
- 判定：`overall_loss_scale_preserved=true`

因此无需运行 `ABS_MAE_NORMALIZED_RETEST`；上一轮 Q0 的 NO-GO 不能归因于整体损失尺度缩小。

## 绝对重构误差偏置

train masked diffusion MAE：

| 审计组 | n | mean | std | P25 | P50 | P75 |
|---|---:|---:|---:|---:|---:|---:|
| normal | 3,584 | 0.336320 | 0.016811 | 0.324843 | 0.335495 | 0.347536 |
| fault | 3,120 | 0.439622 | 0.182473 | 0.343122 | 0.374930 | 0.467834 |

fault-normal standardized mean difference 为 0.797241。当前 Q0 的 q-label Spearman 为 -0.526919，q-teacher hard consistency 为 -0.143257，q-semantic score 为 -0.559790。

### 连续因素 Spearman 相关

下表为 train masked diffusion MAE 与各因素的 Spearman ρ：

| 因素 | ρ |
|---|---:|
| clean window 原始方差 | 0.704630 |
| 一阶差分幅度 | 0.321507 |
| simple interpolation error | 0.327946 |
| degraded zero-fill error | 0.743109 |
| teacher clean prediction entropy | -0.611284 |
| teacher semantic score | 0.559787 |
| teacher hard consistency | 0.143256 |
| fault onset 相对起点 | 0.006742 |
| simple-relative gain | -0.664372 |

error 与信号方差、zero-fill error 和 fault type 明显耦合，而与 onset 后相对位置几乎无单调关系。

### fault type 与 run 差异

每个 fault type 均有 156 个 train 窗口：

| fault type | MAE mean±std | fault type | MAE mean±std |
|---:|---:|---:|---:|
| 1 | 0.4278±0.0392 | 11 | 0.3551±0.0160 |
| 2 | 0.5069±0.0249 | 12 | 0.4870±0.0646 |
| 3 | 0.3336±0.0168 | 13 | 0.4964±0.0664 |
| 4 | 0.3411±0.0174 | 14 | 0.3967±0.0190 |
| 5 | 0.3570±0.0272 | 15 | 0.3351±0.0181 |
| 6 | 0.8841±0.1922 | 16 | 0.3398±0.0183 |
| 7 | 0.4041±0.0623 | 17 | 0.4295±0.0271 |
| 8 | 0.4981±0.1003 | 18 | 0.8086±0.3993 |
| 9 | 0.3336±0.0163 | 19 | 0.3600±0.0190 |
| 10 | 0.3473±0.0171 | 20 | 0.3504±0.0168 |

fault type 均值范围为 0.333634（type 9）至 0.884094（type 6），跨度 0.550460。248 个 train run 的平均误差分布为 mean 0.386305、std 0.119333、P10/P50/P90 0.332408/0.339669/0.497565，run 均值全范围 0.746167。绝对误差显著编码 fault/run 难度。

## 三种质量定义

### Q0：Absolute MAE

```text
q_abs = clip(exp(-e_diff / train_median(e_diff)), 0.1, 1)
```

train-only scale 为 0.344795680。

### Q1：Relative Gain

```text
gain = (e_simple - e_diff) / (e_simple + eps)
z = (gain - train_median) / train_IQR
q_rel = 0.1 + 0.8 * sigmoid(z)
```

train-only median 为 -0.092615611，IQR 为 0.127698421。负中位数说明 diffusion 对至少一半 train 窗口不如 simple interpolation。

### Q2：Relative Gain + Semantic

```text
semantic = 1 - total_variation(teacher(clean), teacher(restored))
q_rel_sem = 0.1 + (q_rel - 0.1) * semantic
```

教师冻结；semantic 只使用预测分布距离，不使用真实标签。

## 候选偏置结果

| 定义/split | q mean±std | normal mean | fault mean | 均值差 | q-label ρ | q-semantic ρ |
|---|---:|---:|---:|---:|---:|---:|
| Q0/train | 0.3446±0.0679 | 0.3775 | 0.3068 | 0.0706 | -0.5269 | -0.5598 |
| Q1/train | 0.4599±0.1770 | 0.5265 | 0.3834 | 0.1431 | -0.3475 | -0.4037 |
| Q2/train | 0.4105±0.1720 | 0.4461 | 0.3696 | 0.0765 | -0.2107 | -0.0831 |
| Q0/validation | 0.3412±0.0674 | 0.3787 | 0.3090 | 0.0697 | -0.5645 | -0.6214 |
| Q1/validation | 0.4486±0.1808 | 0.5266 | 0.3814 | 0.1453 | -0.3733 | -0.4581 |
| Q2/validation | 0.4021±0.1743 | 0.4444 | 0.3656 | 0.0788 | -0.2291 | -0.1528 |
| Q0/test | 0.3464±0.0700 | 0.3785 | 0.3028 | 0.0757 | -0.5385 | -0.4973 |
| Q1/test | 0.4663±0.1734 | 0.5334 | 0.3751 | 0.1583 | -0.4009 | -0.4036 |
| Q2/test | 0.4091±0.1678 | 0.4479 | 0.3563 | 0.0916 | -0.2610 | -0.0246 |

Q1/Q2 的 q-label 秩相关和 SMD 确有减弱，但原始平均 q 差没有比 Q0 更小。Q1 的 train 均值差扩大 102.6%，Q2 扩大 8.3%。这说明更宽的 q 分布可以降低 SMD/秩相关，却没有消除 fault 整体降权，不能作为去偏通过证据。Q1/Q2 落在 q_min 的比例均为 2.61%，不存在整体塌缩到下界的问题。

## 同类内部高低质量审计

所有高低组均在 normal、fault 内部分半，不混用总体类别比例。

| 定义/类别 | low n / mean q | high n / mean q | low/high diffusion MAE | low/high relative gain | low/high semantic |
|---|---:|---:|---:|---:|---:|
| Q1 normal | 1792 / 0.4555 | 1792 / 0.5975 | 0.3433 / 0.3293 | -0.1220 / -0.0278 | 0.7883 / 0.8249 |
| Q1 fault | 1560 / 0.1941 | 1560 / 0.5727 | 0.5309 / 0.3484 | -1.4549 / -0.0334 | 0.9954 / 0.9422 |
| Q2 normal | 1792 / 0.3501 | 1792 / 0.5422 | 0.3419 / 0.3308 | -0.1088 / -0.0410 | 0.6843 / 0.9289 |
| Q2 fault | 1560 / 0.1900 | 1560 / 0.5492 | 0.5299 / 0.3494 | -1.4515 / -0.0368 | 0.9761 / 0.9615 |

fault 内部的高低组 fault-type 组成仍严重不同。例如 Q1-low 主要包含 type 1/2/6/8/12/13/17/18，Q1-high 主要包含 type 3/4/9/10/11/14/15/16/19/20；Q2 亦呈相同结构。也就是说，relative gain 仍主要在排序 fault type 难度，而非隔离独立的恢复质量。

完整每组 fault-type counts、每个 fault type 内 q-error Spearman、所有分位数保存在 `outputs/quality_definition_audit/quality_definition_audit.json`。该文件为运行输出，不提交 Git。

## 门控判定

Q1 与 Q2 均通过：q-semantic 关系未变差、未集中到 q_min。

Q1 与 Q2 均失败：相对 Q0 的平均权重差未减少、fault 平均 q 仍比 normal 低超过 0.05。

因此：

```text
QUALITY_DEFINITION_AUDIT_NO_GO
training_retest_allowed = false
```

审计门控发生在任何新 SupCon 训练之前，未使用 test 分类指标反向选择定义。
