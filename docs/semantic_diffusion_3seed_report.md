# 语义约束扩散增强 3-Seed 稳定性复核报告

> **SEMANTIC_DIFFUSION_3SEED_VALIDATION / FIXED_CONFIG / SUBSET_DATA / NOT_FOR_PAPER_FINAL_CLAIMS**

## 最终结论

```text
SEMANTIC_DIFFUSION_3SEED_NO_GO
```

在固定 seed `7/42/2026` 上，B2 相对 B1 的 Macro-F1 三次全部下降，FAR 三次全部上升。平均 AUPRC 和 Fault Recall 虽有改善，但没有转化为更好的 Macro-F1/FAR 综合表现。因此单 seed 阶段的正向结果未通过稳定性复核，停止继续扩展当前语义约束版本。

本结果仅来自固定 TEP 小型子集和 3 个 seed，不进行显著性声称，也不是论文最终结论。

## 冻结配置与指纹

复核严格使用 `configs/semantic_diffusion_3seed.yaml`：固定 split 与 MCAR 30% mask、window length 64、stride 16、standard scaler、冻结教师、TCN encoder/projection head、50-step cosine schedule、`t_aug=5`、`lambda_prob=lambda_feat=0.1`、Adam、学习率 0.001、batch size 128、8 个 SupCon epoch、8 个 probe epoch、temperature 0.1 及同一 validation threshold 算法。

| 对象 | SHA-256 |
|---|---|
| 冻结配置 | `77c6b2c75a409d152fa74df52f1263d7ab0c06b0360b8d3b6aa3d567639f6696` |
| split/fixed-view manifest | `1824e2cfa0b86ef71afe2d38913134ea418d9d7dda5bbf9e624a496faff88eb1` |
| teacher checkpoint | `290701ccf6ba74fb620874aaeada60ffc9a93c7bd3d8991de4a92f155fdc203b` |

三个 seed 共享以上全部指纹。同一 seed 内 B0/B1/B2 共享分类器初始化、batch order、优化器和探针协议；唯一差异为增强视图来源。B1/G0 与 B2/G1 还共享同一 generator 初始化来源及训练 timestep/noise 序列。

启用了 PyTorch deterministic algorithms、cuDNN deterministic、`CUBLAS_WORKSPACE_CONFIG=:4096:8`，并关闭 cuDNN benchmark。仍需承认 PyTorch 之外的 CUDA/库算子不保证跨硬件位级一致。

## Seed 7 是否复用

未复用旧结果。旧单 seed 结果没有本轮完整 config/split/teacher 三重指纹及确定性 CUDA 记录，无法证明与冻结协议完全一致；按预注册规则重新运行 seed 7。新的 seed 7 结果与其他 seed 使用完全相同的框架提交和配置指纹。

## 每个 Seed 的 B0/B1/B2 结果

| Seed | 方法 | Macro-F1 | AUPRC | Fault Recall | FAR | AUROC | threshold |
|---:|---|---:|---:|---:|---:|---:|---:|
| 7 | B0 | 0.8933 | 0.9287 | 0.7867 | 0.0184 | 0.9120 | 0.4728 |
| 7 | B1 | 0.8880 | 0.9279 | 0.8043 | 0.0418 | 0.9114 | 0.4669 |
| 7 | B2 | 0.8846 | 0.9273 | 0.8085 | 0.0512 | 0.9102 | 0.4649 |
| 42 | B0 | 0.8823 | 0.9328 | 0.8239 | 0.0676 | 0.9191 | 0.3587 |
| 42 | B1 | 0.8841 | 0.9258 | 0.7963 | 0.0422 | 0.9086 | 0.3653 |
| 42 | B2 | 0.8811 | 0.9309 | 0.8218 | 0.0680 | 0.9162 | 0.3628 |
| 2026 | B0 | 0.8791 | 0.9182 | 0.7676 | 0.0273 | 0.8981 | 0.4976 |
| 2026 | B1 | 0.8804 | 0.9177 | 0.7617 | 0.0203 | 0.8972 | 0.5010 |
| 2026 | B2 | 0.8797 | 0.9188 | 0.7670 | 0.0258 | 0.8988 | 0.4975 |

所有 test 指标只在各方法用相同算法从自身 validation 结果选定 threshold 后评价；test 没有参与配置、`t_aug`、lambda 或 threshold 规则选择。

## 3-Seed mean ± std

标准差为 3 个 seed 的样本标准差（`ddof=1`）。

| 方法 | Macro-F1 | AUPRC | Fault Recall | FAR | AUROC |
|---|---:|---:|---:|---:|---:|
| B0 传统增强 | 0.8849 ± 0.0075 | 0.9266 ± 0.0075 | 0.7927 ± 0.0287 | 0.0378 ± 0.0262 | 0.9097 ± 0.0107 |
| B1 普通扩散增强 | 0.8841 ± 0.0038 | 0.9238 ± 0.0054 | 0.7874 ± 0.0226 | 0.0348 ± 0.0125 | 0.9057 ± 0.0075 |
| B2 语义约束扩散增强 | 0.8818 ± 0.0025 | 0.9256 ± 0.0062 | 0.7991 ± 0.0286 | 0.0483 ± 0.0212 | 0.9084 ± 0.0088 |

B2 平均综合表现没有达到 B0：Macro-F1 比 B0 低 0.31 个百分点，FAR 高 1.05 个百分点。AUPRC 低 0.09 个百分点，Recall 高 0.64 个百分点。

## B2-B1 逐 Seed 配对差值

`ΔFAR = FAR(B2)-FAR(B1)`，负值才表示改善。

| Seed | ΔMacro-F1 | ΔAUPRC | ΔRecall | ΔFAR |
|---:|---:|---:|---:|---:|
| 7 | -0.0034 | -0.0006 | +0.0043 | +0.0094 |
| 42 | -0.0030 | +0.0050 | +0.0255 | +0.0258 |
| 2026 | -0.0006 | +0.0010 | +0.0053 | +0.0055 |
| mean ± std | -0.0023 ± 0.0015 | +0.0018 ± 0.0029 | +0.0117 ± 0.0120 | +0.0135 ± 0.0108 |

方向一致性：

- B2 Macro-F1 高于 B1：`0/3`；
- B2 FAR 低于 B1：`0/3`；
- B2 AUPRC 不低于 B1：`2/3`；
- B2 Recall 相对 B1 下降不超过 1 个百分点：`3/3`，实际三个 seed 均上升；
- 没有单 seed 出现 FAR 增加超过 5 个百分点或 Recall 下降超过 5 个百分点的“灾难性”崩溃，但三个 seed 的核心综合方向一致反向。

这不是“标准差大于平均提升”的不确定结果：Macro-F1 和 FAR 在 3/3 seed 上都朝不利方向变化，因此判为 NO-GO，而不是 UNSTABLE。

## 语义一致性与增强多样性

| Seed | 方法 | teacher consistency | fault→normal | normal→fault | feature cosine | normalized L1 |
|---:|---|---:|---:|---:|---:|---:|
| 7 | B1 | 0.9252 | 0.1739 | 0.1523 | 0.9945 | 0.2382 |
| 7 | B2 | 0.9117 | 0.1479 | 0.2379 | 0.9936 | 0.2572 |
| 42 | B1 | 0.8869 | 0.1521 | 0.2621 | 0.9916 | 0.2368 |
| 42 | B2 | 0.7775 | 0.0979 | 0.4605 | 0.9768 | 0.2597 |
| 2026 | B1 | 0.8869 | 0.1590 | 0.2578 | 0.9914 | 0.2364 |
| 2026 | B2 | 0.9354 | 0.1681 | 0.1844 | 0.9948 | 0.2616 |

B2 的 normalized L1 在三个 seed 都约为 0.26，不是复制视图，且所有增强有限、无 NaN/Inf。但语义约束本身不稳定：seed 7/42 的整体 teacher consistency 反而低于 B1，seed 42 尤其降至 0.7775；仅 seed 2026 明显高于 B1。B2 往往降低 fault→normal，却可能提高 normal→fault，最终体现为 FAR 恶化。语义损失改善某一翻转方向不等价于稳定的双向语义保持或下游收益。

## 表征、训练时间与显存

三个 seed 的 classifier 训练平均耗时：B0 276.66 s、B1 273.63 s、B2 275.46 s；峰值显存均约 78.14 MiB。generator 平均训练耗时：G0/B1 190.27 s，G1/B2 276.69 s；峰值显存约 131.23 MiB。

best pretrain/probe epoch 随 seed 改变，但同一 seed 的三方法使用相同 early-stopping 规则：seed 7 主要为 epoch 1/1，seed 42 为 epoch 1–3/7，seed 2026 为 epoch 0/3。完整 checkpoint、训练曲线与方法结果位于 `outputs/semantic_diffusion_3seed/seed_<seed>/<method>/`，受 Git 忽略，不进入提交。

## 稳定性判定

八项 GO 检查中，仅“平均 Recall 未下降超过 1 个百分点”“平均 AUPRC 未下降超过 0.5 个百分点”和“无灾难性单 seed”通过；平均 Macro-F1、Macro-F1 逐 seed 胜率、平均 FAR、FAR 逐 seed 胜率及相对 B0 综合表现均失败。

因为 B2 在至少 2/3 seed 上不如 B1，且 Macro-F1/FAR 在三个 seed 上一致变差，最终状态为：

```text
SEMANTIC_DIFFUSION_3SEED_NO_GO
```

## 唯一下一步建议

停止扩展当前语义约束扩散增强版本；仅基于本轮已保存的 G1 训练曲线和增强审计，定位为什么 semantic consistency 对 generator seed 高度敏感，暂不新增数据集、退化、损失或超参数搜索。
