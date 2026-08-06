# 语义扩散生成器多随机种子失败诊断

> **SEMANTIC_GENERATOR_STABILITY_FIX / GENERATOR_FIRST / FIXED_TEP_SUBSET / NOT_FOR_PAPER_CLAIMS**

## 诊断结论

当前问题是多项共同作用，而非单一“seed 42 初始化失败”：

1. 语义与 timestep 条件以常量偏置形式加在 GroupNorm 之前，可能被归一化削弱；
2. `L_prob/L_feat` 的标量值虽小，`L_prob` 对 generator 的梯度却跨 batch/seed 剧烈波动；
3. generator 均匀训练 `t=0–15`，实际只在 `t_aug=5` 使用；
4. 旧训练只按最后一轮保存，没有逐 epoch validation、early stopping 或 best checkpoint；
5. 完整反向样本多样性过强，validation normalized L1 始终约 0.25–0.32；
6. 单向 KL 主要压低 fault→normal，却可能把 normal→fault 推高，未形成双向平衡保护。

因此诊断为：条件结构缺陷、语义梯度失衡、训练/采样错配和模型选优缺失共同造成的训练不稳定。

## 旧日志可用性

读取了：

- `outputs/semantic_diffusion_3seed/seed_7/B2/generator_training.json`
- `outputs/semantic_diffusion_3seed/seed_42/B2/generator_training.json`
- `outputs/semantic_diffusion_3seed/seed_2026/B2/generator_training.json`

旧日志包含逐 epoch `L_total/L_diff/L_prob/L_feat`，但以下字段旧日志不可用：

```text
逐 epoch validation 指标：旧日志不可用
分项 generator 梯度范数：旧日志不可用
```

没有伪造缺失字段。使用相同 split、teacher、seed 和旧配置，只重跑了 generator-only 诊断；没有运行 SupCon 或 probe，新结果写入 `outputs/semantic_generator_stability_fix/diagnosis_old/`，未覆盖旧结果。

## 损失数值尺度

旧训练最后一轮：

| Seed | L_total | L_diff | L_prob | L_feat | 0.1×(L_prob+L_feat) 占 total |
|---:|---:|---:|---:|---:|---:|
| 7 | 0.6731 | 0.6643 | 0.0808 | 0.0063 | 1.29% |
| 42 | 0.6798 | 0.6709 | 0.0823 | 0.0065 | 1.31% |
| 2026 | 0.6919 | 0.6829 | 0.0829 | 0.0066 | 1.29% |

从标量值看，语义项远小于扩散项。但梯度并不同比例稳定：首 batch 的 `L_diff` 梯度范数约 0.08–0.63；加权 probability 梯度约 0.037–5.94，seed 42/2026 的 epoch 0 分别为 5.17/5.94，后续仍会出现 0.61、1.95 等尖峰。加权 feature 梯度通常较小，但也从 0.0018 波动到 0.252。

结论：不是简单的“语义权重太小”。旧语义项在 loss 数值上弱、在局部梯度上却可能压倒扩散梯度，属于尺度与方向均不稳定。

## Seed 42 何时开始异常

固定 validation 子集与采样 seed 的 generator-only 结果：

| Epoch | consistency | normal→fault | fault→normal | normalized L1 | feature cosine |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.9609 | 0.1172 | 0.1094 | 0.3186 | 0.9942 |
| 1 | 0.9219 | 0.2578 | 0.0469 | 0.3061 | 0.9925 |
| 2 | 0.9492 | 0.2109 | 0.0703 | 0.2875 | 0.9954 |
| 3 | 0.8945 | 0.3047 | 0.0391 | 0.2725 | 0.9896 |
| 4 | 0.9141 | 0.2656 | 0.0391 | 0.2607 | 0.9926 |
| 5 | 0.8320 | 0.4297 | 0.0391 | 0.2534 | 0.9787 |

seed 42 在 epoch 0 表现正常，从 epoch 1 开始出现 normal→fault 上升，epoch 3 后明显恶化，最后一轮是最差轮次。它不是初始结构完全失效，而是训练后期逐渐向 fault 方向偏移。

## 三个 Seed 的最后轮与中间轮

- seed 7：最佳中间轮约为 epoch 2（consistency 0.9648、normal→fault 0.1563），最后轮为 0.9336/0.2266，明显更差；
- seed 42：epoch 2 为 0.9492/0.2109，最后轮为 0.8320/0.4297，显著更差；
- seed 2026：epoch 3 曾突然降至 consistency 0.7969、normal→fault 0.4766，随后恢复；最后轮 0.9453/0.1797，不是最差，但仍不如 epoch 0–2 的平衡状态。

这证明只使用 last checkpoint 会放大 seed 差异，逐 epoch validation 与 best checkpoint 是必要修复，而不是可选优化。

## 是否只在 Fault 一侧提高语义一致性

是。三个 seed 都出现 fault→normal 随训练下降，同时 normal→fault 上升：seed 42 最后一轮 fault→normal 仅 3.9%，但 normal→fault 达 43.0%；seed 2026 epoch 3 的 fault→normal 为 0，却有 47.7% normal→fault。

旧单向 KL 并未稳定保护两类边界，而更像推动教师预测向 fault 侧偏移。这与 3-seed 下游结果“Recall 上升、FAR 同时上升”一致。

## Timestep 与多样性

补充诊断复现旧 `0–15` 均匀 timestep 训练，而推理固定使用 `t_aug=5`。每个 epoch 的直方图覆盖全部 16 个 timestep，确认训练预算大量用于不会直接使用的噪声强度。

三个 seed 的 validation normalized L1 从 epoch 0 的约 0.319 下降到最后轮约 0.251–0.255，但始终高于下一阶段目标 0.10–0.20。完整反向样本变化过强，增加了 normal 跨边界风险。

## 对五个诊断问题的回答

1. seed 42 从训练初期就异常吗？不是；epoch 0 正常，后期逐渐恶化。
2. `L_prob/L_feat` 与 `L_diff` 是否尺度悬殊？标量值明显更小，但 probability 梯度反而可能远大于 diffusion 梯度，存在严重梯度尺度波动。
3. 语义一致性提高是否只发生在 fault 一侧？是；fault→normal 降低常伴随 normal→fault 上升。
4. 最后一轮是否差于中间轮？seed 7/42 明显更差，seed 2026 存在中期崩落后恢复。
5. 更像什么问题？结构条件被归一化削弱、单向语义梯度失衡、timestep 错配、增强幅度过强和 last-checkpoint 过拟合共同作用。

## 获准的最小修复

诊断证据支持提示词规定的最小修复：真正的 GroupNorm 后 FiLM、显式 `x_base` 输入、残差幅度 `alpha∈{0.4,0.6}`、训练 timestep 收窄到 `{3,4,5,6,7}`、双向平衡语义损失，以及 validation best checkpoint/EMA。修复后必须先做 generator-only 3-seed 审计，未通过不得运行下游训练。
