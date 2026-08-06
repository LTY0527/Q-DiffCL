# 频率选择性扩散 FAR 上升诊断

> **FREQUENCY_SELECTIVE_FAR_FIX / STRUCTURE_PRESERVING_SPECTRAL_NOISE / EXPLORATORY_TEP_SUBSET / NOT_FOR_PAPER_CLAIMS**

## 结论

原因分类：`A. INTENSITY_DOMINANT`。状态：`FREQUENCY_SELECTIVE_FAR_CAUSE_RESOLVED`。

旧 MVP 的 C2 在相同总噪声预算下提高了 Recall 和 Early Recall，但 FAR 上升、Macro-F1 下降。旧输出没有保存逐样本 score 和 checkpoint，因此本次以相同 Seed 7、初始化、批次顺序、SupCon 与 Probe 协议确定性重放 C1 和 iid `t=3/5/8`。test 仅作外部描述，不参与原因分类或候选选择。

## Normal/Fault 分数漂移

| 方法 | Split | Normal mean | median | P90 | P95 | Fault mean | N→F | F→N | Val threshold |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 | validation | 0.3823 | 0.3807 | 0.4075 | 0.4160 | 0.6811 | 0.0435 | 0.1077 | 0.4178 |
| C1 | test | 0.3818 | 0.3801 | 0.4067 | 0.4146 | 0.6692 | 0.0375 | 0.1979 | 0.4178 |
| C2 t=8 | validation | 0.4138 | 0.4123 | 0.4328 | 0.4396 | 0.6679 | 0.0592 | 0.0933 | 0.4380 |
| C2 t=8 | test | 0.4133 | 0.4121 | 0.4322 | 0.4381 | 0.6597 | 0.0512 | 0.1888 | 0.4380 |

threshold 邻域数量、Brier/ECE、四阶段、Fault 1–20（含 3/9/15）以及逐 Run delay/miss 均保存在 `outputs/frequency_selective_far_fix/diagnosis/result.json`。

## Validation t=3/5/8 权衡

| t | Macro-F1 | AUPRC | Recall | FAR | Early Recall | Mean Delay | Critical retention | Normal corr drift | Norm. L1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.9219 | 0.9818 | 0.8846 | 0.0346 | 0.8375 | 138.6 | 0.9917 | 0.00706 | 0.0520 |
| 5 | 0.9225 | 0.9820 | 0.8837 | 0.0324 | 0.8375 | 138.6 | 0.9933 | 0.00651 | 0.0519 |
| 8 | 0.9223 | 0.9818 | 0.9067 | 0.0592 | 0.8688 | 134.2 | 0.9946 | 0.00608 | 0.0517 |

旧规则只优化 critical Fisher 与 early retention，未设置 FAR 约束，因此选择了 `t=8`。本次结果显示 `t=5` 的 FAR 明显低于 `t=8`，且 Macro-F1 略高；代价是 Recall 和 Early Recall 较低。

## 扰动集中与跨传感器结构

C1 validation normal corr drift=0.008308，C2 t=8=0.006078，C2/C1 比值=0.732。频域关键/非关键/全频结构漂移、每通道与每 bin 扰动能量均保存在结果 JSON。

- 低强度 `t=3/5` 是否使 validation FAR 至少下降 0.0050：`True`。
- C2 normal corr drift 是否至少为 C1 的 1.10 倍且 normal score 上移：`False`。
- normal fault score 是否上移：`True`。

因此证据支持“非关键频率扰动强度主导”，不支持“跨通道结构漂移主导”。分类为 A，允许按固定配置继续 R0–R3 受限修复；这不意味着相关结构约束必然有效。

## 结论边界

当前 TEP test 已被多轮探索查看，所有 test 指标仅是工程筛选后的外部报告，不能作为论文无偏结论。后续必须严格使用 validation 选择版本，并在新数据集、重新冻结协议或未触碰评测设置上验证论文主张。
