# 语义保持扩散增强最小公平复测报告

> **SEMANTIC_DIFFUSION_AUGMENTATION / SINGLE_SEED / SUBSET_DATA / NOT_FOR_PAPER_CLAIMS**

## 最终状态

```text
SEMANTIC_DIFFUSION_AUGMENTATION_GO
```

本结论仅适用于当前 TEP 固定 split、固定 MCAR 30% 视图、单 seed 和小型实验预算，不可作为论文正式结论。

## 公平性协议

B0–B3 共享 split、master seed、TCN 初始化、batch order、Adam、学习率、8 个 SupCon epoch、temperature=0.1、projection head、8 个 probe epoch、validation threshold 选择逻辑和 test set。每个原始样本始终贡献一条 `x_base` 与一条增强视图，总样本权重固定为 1；唯一变量是正样本增强视图来源。

- 初始化 SHA-256：`dadf6884be39b2a36fdcfa57e3c0ae42dcbef60ecc9524fbcb63ea8c9484c6fb`
- SupCon batch-order SHA-256：`09824ca3f2575c286088d5dcc20c319d59b037ac6f513da1b2bca856c52a29c6`
- probe batch-order SHA-256：`f26379fba215925fc45bd5f1ec4d4db0470b8ae026c37e03e19ec3931aa25aa3`

`t_aug=5` 只由 validation 可行域审计选择。test 只进行一次最终评价，没有参与 `t_aug`、lambda、门控阈值或分类阈值选择。

## 核心结果

| 方法 | Macro-F1 | AUPRC | Fault Recall | FAR | 语义一致率 | normalized L1 |
|---|---:|---:|---:|---:|---:|---:|
| B0 传统增强 | 0.8931 | **0.9287** | 0.7862 | **0.0184** | **0.9795** | 0.0414 |
| B1 普通部分扩散增强 | 0.8856 | 0.9265 | **0.8059** | 0.0473 | 0.9108 | 0.2381 |
| B2 语义约束部分扩散增强 | **0.8949** | 0.9272 | 0.7968 | 0.0238 | 0.9189 | 0.2609 |
| B3 语义有效性门控 | 0.8630 | 0.9187 | 0.8000 | 0.0820 | 0.9752 | 0.2416 |

B2 相对 B1：

- Macro-F1 提高 0.93 个百分点；
- FAR 降低 2.34 个百分点，超过预设的 2 个百分点强信号参考；
- Fault Recall 下降 0.90 个百分点，仍在“不超过 1 个百分点”的保持范围；
- AUPRC 提高 0.07 个百分点；
- 语义一致率提高 0.81 个百分点，fault→normal 翻转从 15.96% 降至 15.37%；
- normalized L1 为 0.2609，证明提升不是通过复制基础视图获得。

B2 相对 B0 的 Macro-F1 高 0.18 个百分点、Fault Recall 高 1.06 个百分点；代价是 FAR 高 0.55 个百分点、AUPRC 低 0.15 个百分点。因而本轮信号支持“语义约束优于无约束部分扩散”，但尚不能宣称全面优于传统增强。

## B3 门控结果

B2 出现正向信号后才运行 B3。train/validation/test 的首次接受率分别为 83.68%/83.94%/83.38%；重采样后额外接受约 8.4%–8.9%，最终约 7.5%–7.7% 回退传统增强。

B3 将 test 增强语义一致率提高到 97.52%，但 Macro-F1 降至 0.8630、FAR 升至 8.20%，明显弱于 B2。高教师一致率并不自动等价于更好的下游决策边界，本轮不保留 B3 作为有效方法组件。

## normal、fault 与 fault type

B2 的 normal FAR 为 2.38%，显著低于 B1 的 4.73%；fault recall 为 79.68%，低于 B1 的 80.59% 但高于 B0 的 78.62%。这说明主要收益来自抑制 normal 误报，同时只付出小幅 fault recall 代价。

20 个 fault type 中，B2 相对 B1 在类型 1、2、4、6–9、11–14、17、18、20 上保持不变；主要下降来自类型 16（61.70%→55.32%）、10（58.51%→54.26%）、5（92.55%→89.36%）和 19（100%→97.87%）。类型 3、9、15 在所有方法中依然困难，不能把 overall 改善解释为所有故障类型一致改善。

早期故障窗口共 360 个：B0/B1/B2/B3 recall 分别为 77.78%/80.00%/78.61%/78.33%。B2 的早期故障表现介于传统增强与普通扩散之间，没有出现额外早期检测收益。

## 表征、时间与显存

| 方法 | Fisher ratio | class-center shift | effective rank | 训练时间 | 峰值显存 |
|---|---:|---:|---:|---:|---:|
| B0 | 1.8675 | 0.0063 | 1.1258 | 16.23 s | 31.83 MiB |
| B1 | 1.8540 | 0.0099 | 1.1305 | 12.07 s | 31.83 MiB |
| B2 | 1.8537 | 0.0163 | 1.1315 | 12.70 s | 31.83 MiB |
| B3 | 1.9016 | 0.0175 | 1.1155 | 13.01 s | 31.83 MiB |

生成器训练耗时为 G0 3.86 s、G1 4.80 s，峰值显存约 84 MiB；完整 B0–B3 v3 复测耗时 82.43 s。原始 JSON/CSV 位于 `outputs/semantic_diffusion_augmentation/retest_v3/`，不提交 checkpoint、大型 NPZ 或 outputs。

## GO 判定与边界

预注册的八项检查全部通过：B2 的语义一致率、fault 翻转、Macro-F1、FAR、Recall/AUPRC 保持、相对 B0 综合表现、公平性和非零多样性均满足当前工程门控。特别是 B2 对 B1 的 FAR 降幅达到强信号参考，因此最终状态为 `SEMANTIC_DIFFUSION_AUGMENTATION_GO`。

该 GO 只表示值得继续验证，不表示已获得稳定论文结果。单 seed 下 B2 对 B0 的优势很小，fault type 与早期窗口仍有明显异质性。

## 唯一下一步建议

严格冻结当前 `t_aug=5`、lambda、B2 结构与全部训练协议，仅增加到 3 个 seed 做稳定性复核；在完成该复核前不扩展新退化、结构约束或正式论文实验。
