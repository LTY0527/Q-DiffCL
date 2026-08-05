# 语义约束扩散增强可行域审计

> **SEMANTIC_DIFFUSION_AUGMENTATION / SINGLE_SEED / SUBSET_DATA / NOT_FOR_PAPER_CLAIMS**

## 结论

第一级审计状态为：

```text
SEMANTIC_DIFFUSION_FEASIBLE_REGION_GO
selected_t_aug = 5
feasible_timesteps = [5]
```

这是单 seed、小型真实子集上的工程筛选结果，不是论文结论。`t_aug`、语义损失系数和门控均未使用 test 数据选择；test 只评价 validation 选出的 `t_aug=5`。

## 为什么停止 MAE 质量加权路线

此前的质量定义审计与同样本候选排序审计已经表明：绝对重构误差、相对插值改进及其教师语义组合都不能形成可靠的跨样本质量权重；即使 Oracle Best-of-K 能在同一样本内找到更优候选，其 MAE 上限仍不足，最佳扩散候选继续落后于简单插值。因此本阶段不再要求扩散补全的 MAE 超过插值，也不再用 MAE 降低某个样本的 SupCon 总权重。

新路线把简单插值固定为基础补全器，把扩散模型改成正样本增强器：

```text
固定 MCAR 30% 视图 → simple interpolation 得到 x_base
→ 从 q_sample(x_base, t_aug) 开始部分扩散
→ 条件反向去噪得到 x_aug
→ (x_base, x_aug) 作为等权正样本视图
```

## 生成器与语义条件

G0/G1 共用 `SemanticPartialDiffusion1D`：50-step cosine schedule、64 个隐藏通道、3 个 residual block。输入端拼接当前带噪视图与 observation mask；timestep embedding 和冻结教师的 32 维 semantic embedding 分别线性投影后相加，并注入全部 3 个 residual block，而不是只在输入层拼接。

教师 checkpoint 为 `outputs/rapid_idea_validation/G1_0.pt`，仅由 training split 训练。教师参数完全冻结；G1 对生成视图的教师前向仍保留输入梯度。

- G0：`L_diff`
- G1：`L_diff + 0.1 L_prob + 0.1 L_feat`
- `L_prob`：`KL(stopgrad(p_teacher(x_base)) || p_teacher(x0_hat))`
- `L_feat`：`1 - cosine(stopgrad(f_teacher(x_base)), f_teacher(x0_hat))`

G0/G1 使用完全相同的初始化、epoch batch order、训练 timestep/noise seed 和审计 sampling seed。初始化 SHA-256 为 `aac008d6bffe90066de018a3e353e95058fc0b1406c2a6a7abc00b7579df91fb`。

## validation 可行域审计

每个配置生成 3 次随机增强。多样性使用相对 `x_base` 标准差归一化的 L1 和多次增强 pairwise diversity。

| 生成器 | t_aug | 一致率 | normal 一致率 | fault 一致率 | fault→normal | feature cosine | normalized L1 | pairwise diversity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G0 | 5 | 0.9180 | 0.8828 | 0.9531 | 0.0677 | 0.9951 | 0.2327 | 0.2980 |
| G1 | 5 | **0.9310** | **0.8932** | **0.9688** | **0.0521** | 0.9951 | 0.2545 | 0.3366 |
| G0 | 10 | 0.8216 | 0.6823 | 0.9609 | 0.0391 | 0.9733 | 0.3681 | 0.4781 |
| G1 | 10 | **0.8854** | **0.8073** | **0.9635** | 0.0521 | 0.9834 | 0.4256 | 0.5717 |
| G0 | 15 | 0.6602 | 0.3802 | 0.9401 | 0.0026 | 0.9235 | 0.5036 | 0.6671 |
| G1 | 15 | **0.8438** | **0.7422** | **0.9453** | 0.0130 | 0.9729 | 0.6076 | 0.8291 |

`t_aug=10/15` 的整体一致率低于 0.90，未进入可行域。`t_aug=5` 同时满足整体一致率不低于 0.90、feature cosine 不低于 0.90、非零且适度的多样性、有限值及异常尖峰比例不高于 0.1%。

最初诊断曾把异常尖峰比例要求为浮点意义上的精确零，导致已按训练通道范围裁剪、且异常比例仅 `3.29e-5` 的 `t=5` 被错误排除；传统增强该统计同样非零。修正版将数值异常上限显式配置为 0.1%，新增回归测试，并写入独立的 `audit_v2`，没有覆盖首次诊断 checkpoint。

## test 只读评价

在 validation 选定的 `t_aug=5` 上：

| 指标 | G0 | G1 |
|---|---:|---:|
| teacher consistency | 0.9251 | **0.9368** |
| normal consistency | 0.8880 | **0.9049** |
| fault consistency | 0.9622 | **0.9688** |
| normal→fault | **0.2057** | 0.2096 |
| fault→normal | 0.1719 | **0.1602** |
| probability KL | 0.0660 | **0.0588** |
| feature cosine | 0.9946 | **0.9951** |
| normalized L1 | 0.2325 | 0.2547 |
| pairwise diversity | 0.2977 | 0.3365 |

G1 没有退化为复制：normalized L1 和 pairwise diversity 均非零，且略高于 G0。所有生成值有限；G1 通道方差比均值为 0.9951，未出现整体方差坍塌。

fault type 层面，G1 在类型 3、8、10、15、16 上相对 G0 提高或保持一致率；类型 19 从 0.8788 降至 0.8485，是主要负向例外。类型 9/10/15/16/19 仍是需要关注的困难类型，说明语义保持改善并非所有 fault type 一致增益。

## 时间与显存

| 生成器 | 训练时间 | 峰值显存 |
|---|---:|---:|
| G0 | 3.86 s | 83.77 MiB |
| G1 | 4.80 s | 84.73 MiB |

运行环境为 NVIDIA GeForce RTX 4060 Laptop GPU、PyTorch 2.6.0+cu124。完整审计结果位于 `outputs/semantic_diffusion_augmentation/audit_v2/result.json`，checkpoint 与大型输出不进入 Git。

## 第一级判定

G1 相对 G0 提高 fault consistency、降低 fault→normal 翻转，并保留适度非零多样性；`t_aug=5` 进入预设可行域。因此允许执行一次 B0–B3 最小公平复测。最终方法 GO/NO-GO 由下游复测决定。
