# 语义扩散生成器稳定性诊断与修复结果

> **SEMANTIC_GENERATOR_STABILITY_FIX / GENERATOR_FIRST / FIXED_TEP_SUBSET / NOT_FOR_PAPER_CLAIMS**

## 结论

本轮仅完成 generator-only 诊断、最小结构修复和三随机种子审计。最终状态为：

```text
SEMANTIC_GENERATOR_FIX_NO_GO
```

修复后的生成器消除了 normal→fault 超过 30% 的异常 seed，增强幅度稳定落在目标区间，且 validation best checkpoint 在全部六个对照运行中均不差于 last checkpoint；但是 G1-fixed 只在 1/3 seed 上提高 teacher consistency，平均 balanced flip rate 也没有低于 G0。因此生成器 gate 未通过，下游 B1/B2-fixed SupCon 复测按规则跳过，不能据此提出论文有效性结论。

## 实验边界与可复现设置

- 数据范围：固定 TEP 子集、既有 split、固定 teacher、固定 mask 和固定 validation 子集；没有新增数据集或退化类型。
- generator seed：`7 / 42 / 2026`。
- 训练 timestep：`{3,4,5,6,7}`；推理 `t_aug=5`。
- 最大训练 20 epoch，early stopping patience 为 4；EMA decay 为 0.999。
- generator checkpoint 只由固定 validation 子集、固定 sampling seed 和分数 `J = balanced flip + probability distance + diversity penalty` 选择；test 不参与选择。
- seed 7 用于一次性选择 `lambda_sem` 和 `alpha`，选定后冻结到 seed 42/2026。
- 运行环境：Python 3.10.20、PyTorch 2.6.0+cu124、CUDA 12.4、NVIDIA GeForce RTX 4060 Laptop GPU。
- 原始结果保存在 `outputs/semantic_generator_stability_fix/`，未覆盖旧实验结果。

## 失败原因诊断

旧生成器的不稳定不是 seed 42 的初始状态异常，而是多个因素共同作用：

1. seed 42 在 epoch 0 的 consistency 为 0.9609、normal→fault 为 0.1172，尚属正常；到 epoch 5 分别恶化为 0.8320 和 0.4297，说明偏移主要在训练后期形成。
2. 旧语义项的加权标量只占总 loss 约 1.3%，但 probability 项的局部梯度高度波动，加权梯度峰值约 5.94，可显著超过 diffusion 梯度；问题不是简单的“语义权重太小”。
3. fault→normal 的下降经常伴随 normal→fault 上升，旧单向约束主要把边界推向 fault 一侧，没有形成 normal/fault 双向保护。
4. seed 7 和 42 的 last checkpoint 明显差于中间轮次，缺少逐 epoch validation 与 best checkpoint 会放大 seed 差异。
5. 旧 normalized L1 约为 0.25～0.32，增强幅度过强；同时存在 GroupNorm 前条件削弱、训练/使用 timestep 错配以及未显式输入基础视图的问题。

完整诊断证据见 `docs/semantic_generator_failure_diagnosis.md`。

## 最小修复内容

- 将条件注入改为真正的 GroupNorm 后 FiLM，每个 residual block 独立使用 timestep 与 semantic embedding 生成 gamma/beta，并采用保守初始化。
- generator 输入从 `[x_t, observation_mask]` 改为 `[x_t, x_base, observation_mask]`，训练与采样使用相同定义。
- 使用 `x_aug = x_base + alpha * (x_sample - x_base)` 限制残差幅度，并验证 alpha 0/1 边界。
- 将训练 timestep 收窄到 `{3,4,5,6,7}`，与实际 `t_aug=5` 对齐。
- 语义约束改为 Jensen-Shannon、SmoothL1 logit margin、feature cosine 的 normal/fault 等权聚合；教师保持冻结，并安全处理单类 batch。
- 增加固定 validation、raw/EMA 同轮评价、best/last checkpoint、early stopping 和一致的 Probe epoch/threshold 选择逻辑。

## 超参数选择

seed 7 的 validation 候选分数如下，分数越低越好：

| lambda_sem | alpha=0.4 | alpha=0.6 |
|---:|---:|---:|
| 0.03 | 0.118351 | 0.114602 |
| 0.10 | **0.105148** | 0.111694 |

因此冻结配置为：

```text
lambda_sem = 0.1
alpha = 0.4
```

## Generator-only 三随机种子结果

下表均为各运行的 validation best checkpoint。`prob.` 为 teacher probability KL，`feat.` 为 teacher feature cosine；balanced flip 为 normal→fault 与 fault→normal 的等权平均。

| Seed | 方法 | best epoch | 权重 | consistency | normal→fault | fault→normal | balanced flip | norm. L1 | prob. | feat. |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 7 | G0 | 1 | raw | 0.9844 | 0.1328 | 0.0625 | 0.0977 | 0.1012 | 0.00801 | 0.99924 |
| 7 | G1-fixed | 4 | raw | 0.9805 | 0.1250 | 0.0625 | 0.0938 | 0.1136 | 0.01140 | 0.99886 |
| 42 | G0 | 8 | EMA | 0.9844 | 0.1172 | 0.0781 | 0.0977 | 0.1289 | 0.01213 | 0.99882 |
| 42 | G1-fixed | 4 | raw | 0.9922 | 0.1328 | 0.0781 | 0.1055 | 0.1140 | 0.00938 | 0.99910 |
| 2026 | G0 | 0 | raw | 0.9727 | 0.1641 | 0.0391 | 0.1016 | 0.1266 | 0.01748 | 0.99846 |
| 2026 | G1-fixed | 0 | EMA | 0.9648 | 0.1250 | 0.0781 | 0.1016 | 0.1381 | 0.01396 | 0.99865 |

三 seed 平均 balanced flip：

| 方法 | 平均 balanced flip |
|---|---:|
| G0 | **0.098958** |
| G1-fixed | 0.100260 |

G1-fixed 的方向性结果是：normal→fault 在 2/3 seed 不高于 G0，fault→normal 在 2/3 seed 不高于 G0（相等计为满足），但 consistency 仅 seed 42 提高。语义约束仍未产生稳定的综合净收益。

## Best 与 last、EMA 与 raw

分数 `J` 越低越好：

| Seed | 方法 | best（epoch/权重/J） | last（epoch/权重/J） |
|---:|---|---|---|
| 7 | G0 | 1 / raw / 0.10566 | 5 / EMA / 0.13095 |
| 7 | G1-fixed | 4 / raw / 0.10515 | 8 / raw / 0.12728 |
| 42 | G0 | 8 / EMA / 0.10979 | 12 / EMA / 0.11659 |
| 42 | G1-fixed | 4 / raw / 0.11484 | 8 / EMA / 0.12964 |
| 2026 | G0 | 0 / raw / 0.11904 | 4 / raw / 0.12040 |
| 2026 | G1-fixed | 0 / EMA / 0.11552 | 4 / EMA / 0.11902 |

六个运行的 best 均不差于 last，证明逐 epoch validation 和 best checkpoint 保存是必要修复。EMA 只在 `42/G0` 与 `2026/G1-fixed` 被选为最优：对应最优非 EMA 分数分别为 0.11514 和 0.11712，而 EMA 为 0.10979 和 0.11552。其余四个运行选择 raw，因此 EMA 有局部收益，但没有跨 seed、跨方法的一致优势。

## 时间与显存

| Seed | 方法 | 训练时间（秒） | 峰值显存（MiB） |
|---:|---|---:|---:|
| 7 | G0 | 350.7 | 224.6 |
| 7 | G1-fixed | 695.2 | 232.8 |
| 42 | G0 | 781.5 | 234.8 |
| 42 | G1-fixed | 677.4 | 232.8 |
| 2026 | G0 | 287.2 | 223.1 |
| 2026 | G1-fixed | 382.2 | 227.0 |

G1-fixed 引入语义教师前向与分项梯度记录，耗时通常高于 G0；六个最终对照的峰值显存均低于 235 MiB。seed 7 的未选中 `lambda_sem=0.03` 候选仅用于 validation 选参，不参与三 seed 门控统计。

## Gate 审计

| 条件 | 结果 |
|---|---|
| consistency 至少 2/3 seed 优于 G0 | **失败（1/3）** |
| normal→fault 至少 2/3 seed 不高于 G0 | 通过（2/3） |
| fault→normal 至少 2/3 seed 不高于 G0 | 通过（2/3） |
| 平均 balanced flip 低于 G0 | **失败（0.100260 > 0.098958）** |
| normalized L1 稳定在 0.10～0.20 | 通过 |
| 无 normal→fault 超过 30% 的 seed | 通过 |
| best checkpoint 全部不差于 last | 通过 |

由于两项必要条件失败，最终 gate 为 `SEMANTIC_GENERATOR_FIX_NO_GO`。

## 下游执行状态与唯一下一步建议

门控脚本返回：

```text
SKIPPED_BY_GENERATOR_GATE
training_skipped = true
```

没有运行 B1/B2-fixed、SupCon 或正式论文实验。

唯一建议的下一步是：继续保持 generator-only 和当前固定数据/结构，只对语义梯度做一次受控归一化修复——将加权语义梯度相对 diffusion 梯度设置明确上限，然后用完全相同的 3-seed gate 复核；在 gate 通过前仍禁止下游训练。该建议针对诊断中“语义标量小但局部梯度可压倒 diffusion 梯度”的剩余问题，不建议继续扩大超参数搜索。

> 本报告仅用于阶段性工程诊断：**NOT_FOR_PAPER_CLAIMS**。
