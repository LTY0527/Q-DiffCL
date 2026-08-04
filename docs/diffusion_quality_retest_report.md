# 固定扩散视图质量加权复测报告

> **IDEA_RETEST / SINGLE_SEED / FIXED_DIFFUSION_VIEW / NOT_FOR_PAPER_CLAIMS**

## 结论

本轮结论为 **`QUALITY_WEIGHTING_IDEA_NO_GO`**。在唯一变量为 Oracle reconstruction quality 权重的公平对比中，质量加权相对 Hard SupCon 的 Macro-F1 下降 1.014 个百分点、FAR 上升 1.328 个百分点；Fault Recall 下降 0.532 个百分点、AUPRC 下降 0.310 个百分点。后两项仍在容忍范围内，但核心目标 Macro-F1 与 FAR 同时变差，因此不支持继续开发无参考质量评分网络。

该结论只适用于当前固定小型子集、seed 7 和固定 diffusion views，不构成论文结论。

## 固定资产与视图冻结

- diffusion checkpoint：`outputs/diffusion_debug/small_subset/best_diffusion.pt`
- checkpoint SHA-256：`74ae41ca8bf45fc284557be7fa6c0859caf0e91a8ab9e642cdf5f75eeae9a22c`
- source split manifest：`outputs/rapid_idea_validation/split_manifest.json`
- source manifest SHA-256：`8bc97906a566aaf71fb438647aadb9901809b417d375fb66f690355a62276ad6`
- 冻结教师：`outputs/rapid_idea_validation/G1_0.pt`
- master seed：7；DDPM sampling seed：3007
- 退化：normalized-space deterministic MCAR 30%
- diffusion：50-step cosine DDPM；未重新训练 diffusion
- 生成代码版本：`2ff336053be7f18c29c3b79d6c9896aefb0e395d`

三个 split 由同一 checkpoint 一次性生成，R1/R2 只读同一 NPZ。NPZ 内保存 `run_uid`、`window_id`、`start_sample`、`end_sample`、`clean_index`、`mask_id`、label、clean/degraded/restored window。大型 NPZ 位于 `outputs/fixed_diffusion_views/`，受 `.gitignore` 保护；小型追溯清单提交在 `configs/fixed_diffusion_views_manifest.json`。

| split | windows | runs | 实际缺失率 | masked MAE | NPZ SHA-256 |
|---|---:|---:|---:|---:|---|
| train | 6,704 | 248 | 0.300051 | 0.384396 | `bd47842e…56685715` |
| validation | 1,936 | 72 | 0.299916 | 0.382687 | `e95b1c5a…135be47` |
| test | 4,440 | 80 | 0.300091 | 0.400778 | `f2f27e4f…ea00af9` |

审计确认三个 split 的 run_uid 集合两两无交集，且运行前重新验证了文件 SHA-256、window order hash 和 mask order hash。现有 best checkpoint 未被覆盖。

## Oracle Quality

每个窗口仅在缺失位置计算：

```text
e_i = masked_MAE(x_clean_i, x_diffusion_recovered_i)
q_i = clip(exp(-e_i / scale), q_min, 1)
```

`scale` 使用 train error 中位数拟合，为 **0.34479567992607674**；`q_min=0.1`。validation/test 不参与 scale 拟合，也未使用 test label。各 split 只应用 train 得到的固定 scale。

| split | q min | q max | mean | std | P10 | P25 | P50 | P75 | P90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 0.1000 | 0.4397 | 0.3446 | 0.0679 | 0.2421 | 0.3395 | 0.3679 | 0.3839 | 0.3964 |
| validation | 0.1000 | 0.4371 | 0.3412 | 0.0674 | 0.2349 | 0.3249 | 0.3654 | 0.3825 | 0.3958 |
| test | 0.1000 | 0.4452 | 0.3464 | 0.0700 | 0.2396 | 0.3459 | 0.3694 | 0.3856 | 0.3978 |

训练 error 的 min/max/mean/std 为 0.283318 / 2.000129 / 0.384396 / 0.135286，P10/P25/P50/P75/P90 为 0.319029 / 0.330083 / 0.344796 / 0.372446 / 0.489065。

## 公平性检查

两组共享以下条件：固定 recovery views、split、seed、TCN encoder、projection head、Adam、学习率 0.001、batch size 128、8 个 SupCon epoch、temperature 0.1、相同 early-stopping 配置、8 个 linear-probe epoch、相同 probe 数据与验证阈值选择算法、同一 test set。

- 初始化 SHA-256：`dadf6884be39b2a36fdcfa57e3c0ae42dcbef60ecc9524fbcb63ea8c9484c6fb`
- batch-order SHA-256：`09824ca3f2575c286088d5dcc20c319d59b037ac6f513da1b2bca856c52a29c6`
- Hard 验证阈值：0.464335；Quality 验证阈值：0.462386
- 唯一训练变量：恢复视图作为正样本时使用 `q=1` 或 train-only Oracle `q_i`

单元测试验证了：`q=1` 与 Hard SupCon 数值一致；降低 q 会降低对应正样本梯度系数；无正样本、全零权重均无 NaN；非有限权重被拒绝；相同 seed 的 batch order 可复现；固定视图 manifest 无 run 泄漏；validation/test 的改变不会影响 fitted scale。

## 同条件结果

| 方法 | Macro-F1 | AUPRC | Fault Recall | FAR | AUROC | Teacher consistency | 时间（秒） | 峰值显存（MiB） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Diffusion + Hard SupCon | **0.893753** | **0.928673** | **0.796809** | **0.025781** | **0.911652** | 0.869369 | 13.41 | 30.81 |
| Diffusion + Oracle Quality SupCon | 0.883611 | 0.925578 | 0.791489 | 0.039062 | 0.907176 | 0.869369 | 9.36 | 30.81 |

| Quality - Hard | Macro-F1 | AUPRC | Fault Recall | FAR |
|---|---:|---:|---:|---:|
| 绝对差 | -0.010141 | -0.003095 | -0.005319 | +0.013281 |

Teacher consistency 由同一冻结教师在同一 clean/restored test views 上计算，因此两组相同。训练时间的差异不作为效果证据；两组都完成 8 个 pretrain 与 8 个 probe epoch，峰值显存相同。

## 表示诊断

| 方法 | Fisher ratio | class-center shift | effective rank |
|---|---:|---:|---:|
| Hard SupCon | 1.703515 | 0.076005 | 1.084755 |
| Oracle Quality SupCon | 1.684614 | 0.075400 | 1.094957 |

Quality 权重确实进入有效梯度：训练 batch 的 q 均值/范围由 Hard 的 1/`[1,1]` 改为 0.344606/`[0.1,0.439684]`，且两者 batch 内平均有效 anchor 均为 252.98。两组首轮及末轮 loss 均不同，排除了权重未生效的情况。

## 高、低质量分组

按 test q 中位数 0.369362 固定分成各 2,220 个窗口：

| 方法/分组 | Macro-F1 | AUPRC | Fault Recall | FAR | fault 数/总数 |
|---|---:|---:|---:|---:|---:|
| Hard / 低质量 | 0.919544 | 0.982310 | 0.900351 | 0.033962 | 1425/2220 |
| Hard / 高质量 | 0.765855 | 0.661850 | 0.472527 | 0.022096 | 455/2220 |
| Quality / 低质量 | 0.907871 | 0.980599 | 0.896140 | 0.056604 | 1425/2220 |
| Quality / 高质量 | 0.752450 | 0.655046 | 0.463736 | 0.031161 | 455/2220 |

高质量组没有优于低质量组。该比较存在明确的类别组成混杂：test normal 的平均 q 为 0.378453、masked MAE 为 0.335421；fault 的平均 q 为 0.302755、masked MAE 为 0.489775。train 中也出现同方向关系（normal q 0.377482，fault q 0.306840）。因此当前 reconstruction error 不只是“恢复可信度”，还系统性编码了故障窗口本身更难恢复这一事实；直接降权会削弱故障正样本，并与本轮 Recall 和 FAR 变差一致。

## GO / NO-GO 判定

六项检查仅三项通过：

- Macro-F1 提升：失败
- FAR 下降：失败
- Fault Recall 下降不超过 1 个百分点：通过
- AUPRC 下降不超过 0.5 个百分点：通过
- 高质量组优于低质量组：失败
- 初始化与 batch order 完全公平：通过

综合判定：**`QUALITY_WEIGHTING_IDEA_NO_GO`**。当前 Oracle reconstruction quality 即使拥有 clean reference 也没有改善目标，故不应继续投入无参考质量评分器。

## 唯一下一步建议

只进行一次 **train-only 的质量定义审计**：量化 masked reconstruction error 对 normal/fault、fault type 和 run 的依赖，并尝试在 train 内做条件化校准，使质量分数不再把“故障语义难度”误当成“恢复低质量”；在该定义通过无标签泄漏审计前，不训练质量网络、不扩大数据、不运行多 seed。

## 资源记录

- 固定 views 生成：202.05 秒，峰值显存 41.98 MiB
- R1/R2 复测总计：25.30 秒
- Hard：13.41 秒；Quality：9.36 秒
- 两组峰值显存：30.81 MiB
