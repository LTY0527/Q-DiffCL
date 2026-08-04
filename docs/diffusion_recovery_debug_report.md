# 扩散恢复诊断与最小有效训练报告

```text
DIFFUSION_DEBUG
SINGLE_SEED
NOT_FOR_PAPER_CLAIMS
```

最终状态：`DIFFUSION_RECOVERY_READY_FOR_IDEA_RETEST`（状态 C）

该状态仅表示修复后的最小扩散恢复已通过 Tiny 多数条件，并在小型真实子集上表现出任务感知潜力；它不表示恢复误差已经优于线性插值，也不构成论文结论。本任务没有运行 Diffusion+SupCon 第三道闸。

## 1. 实现结构

```text
x0 clean + epsilon + timestep
          │
          ▼ q_sample（cosine schedule, 0-based t）
        noisy xt
          │
          ├──────── degraded observed values
          ├──────── observation mask（1=observed, 0=missing）
          ▼
Conv1d([xt, degraded, mask])
          │
ResidualBlock dilation=1  ◄── timestep sinusoidal embedding → Linear
ResidualBlock dilation=2  ◄── timestep embedding
ResidualBlock dilation=4  ◄── timestep embedding
ResidualBlock dilation=8  ◄── timestep embedding
          │
          ▼
predicted epsilon → estimated x0 → DDPM posterior
          │
每一步：missing 使用 generated，observed 重新钳制为原值
```

输入和输出始终为 `[B,C,L]`。四个 dilation block 加上输入/输出卷积使主要感受野覆盖 length=64。mask 与 observed values 都显式进入网络，不是仅用于外部 loss。

## 2. 训练和采样链路

统一实现位于：

- `diffusion/process.py::DiffusionSchedule`
- `diffusion/process.py::ddpm_restore`
- `scripts/train_diffusion_recovery.py::diffusion_objective`
- `models/minimal_diffusion.py::MinimalConditionalDiffusion1D`

训练目标：

```text
L_total = L_diff + lambda_rec * L_masked_rec
L_diff = missing positions epsilon MSE
L_masked_rec = missing positions SmoothL1(estimated_x0, x0)
```

没有加入 semantic、temporal、correlation、SupCon、quality scorer 或 cycle consistency。

采样使用 50-step full DDPM posterior：

1. missing 从标准高斯开始；observed 取真实观测值；
2. 每步预测 epsilon 并回算 x0；
3. x0 仅按 train clean 的逐通道范围 clipping，未使用 validation/test 统计；
4. 使用标准 posterior mean/variance 生成前一步；
5. 每步重新 clamping observed；
6. step 0 不加入额外噪声。

没有实现或依赖 DDIM。

## 3. 强制单元诊断

以下测试通过：

- q_sample 公式及 `x0 → xt → estimated_x0` 回算；
- epsilon/x0 参数化一致性；
- 0-based timestep 和终端 cosine `alpha_bar<1e-3`；
- `True=observed` mask 语义；
- observed-value 每步 clamping；
- final posterior 等于预测 x0，最后一步不加随机噪声；
- 相同 seed 的 DDPM sampling 完全确定；
- 固定 timestep/noise 的 one-batch objective 降到初始 25% 以下；
- gate 多数条件实现测试。

详细基线审计见 `docs/diffusion_debug_audit.md`。

## 4. Tiny Overfit

### 数据与配置

- 配置：`configs/diffusion_tiny_overfit.yaml`。
- 128 个固定真实 TEP Training 窗口，normal/fault 各 64。
- 固定 seed 7、固定 MCAR 30% mask；所有 epoch 复用同一窗口和 mask。
- hidden=128、dilation 1/2/4/8、50 DDPM steps。
- v2：300 epoch，每 epoch 8 个 gradient batch，`lambda_rec=1.0`。

### 第一次诊断运行

旧式 posterior/训练量下：初始 sampled MAE 869.37，200 epoch 后仍为 164.77；训练仅9.29秒。虽然下降，但采样明显爆炸，状态为 `TINY_OVERFIT_FAILED`。结果保存在 `outputs/diffusion_debug/tiny_overfit/`，没有覆盖。

### 修复后 v2

| 方法 | Masked MAE | Masked RMSE | Observed MAE | Full-window MAE | Corr error | 一阶差分误差 |
|---|---:|---:|---:|---:|---:|---:|
| degraded zero fill | 0.5360 | 0.8889 | 0.0000 | 0.1608 | 0.0898 | 0.2545 |
| simple interpolation | 0.3067 | 0.5674 | 0.0000 | 0.0920 | 0.0191 | 0.1543 |
| diffusion v2 | 0.5016 | 0.7416 | 0.0000 | 0.1505 | 0.0327 | 0.2569 |

- 未训练初始 sampled masked MAE：3.6204。
- epoch 1/50/100/150/200/250/300 sampled MAE：3.5422 / 0.9160 / 0.8045 / 0.6537 / 0.5872 / 0.5236 / 0.4992。
- 三次最终采样 MAE：0.5016 / 0.5012 / 0.5012，无整体随机漂移。
- observed MAE 始终为 0，无 NaN/Inf。
- 训练耗时：27.92 秒；含数据读取/采样总耗时：140.72 秒。
- 峰值 GPU 已分配显存：93.70 MiB。

### Tiny 判定

七项中通过六项：持续下降、observed≈0、有限值、非随机噪声、重复采样稳定、固定窗口结果优于 zero-fill。唯一未通过项是建议目标 `diffusion MAE <= simple MAE`。

按提示词“满足多数条件”的规则，最终记为 `TINY_OVERFIT_PASSED`，但明确保留其未超过 simple 的限制。

## 5. 小型真实子集

### 配置

- 配置：`configs/diffusion_small_subset.yaml`。
- manifest 与 rapid validation 相同：train 248 Run、validation 72 Run、官方 Testing 80 Run。
- 窗口：train 6704、validation 1936、test 4440。
- MCAR 30%，同一 normalized space 和确定性 realization。
- 40 epoch，按 validation sampled masked MAE 保存 best checkpoint；没有 SupCon 或质量权重。

Validation sampled MAE 在 epoch 1/5/10/15/20/25/30/35/40 为：2.8976 / 0.7094 / 0.5277 / 0.4613 / 0.4290 / 0.4295 / 0.3934 / 0.3918 / 0.3833，正常收敛。

### 恢复指标

| 方法 | Masked MAE | Masked RMSE | Observed MAE | Full-window MAE | Corr error | 一阶差分误差 |
|---|---:|---:|---:|---:|---:|---:|
| degraded zero fill | 0.5680 | 1.1088 | 0.0000 | 0.1704 | 0.1005 | 0.2685 |
| simple interpolation | 0.3134 | 0.5947 | 0.0000 | 0.0940 | 0.0173 | 0.1584 |
| diffusion | 0.4010 | 0.6917 | 0.0000 | 0.1203 | 0.0382 | 0.2006 |

三次 diffusion test sampled MAE 为 0.4010 / 0.4017 / 0.4014，采样稳定。Diffusion 相比旧实现 0.7454 明显改善，但仍比 simple 高约28%。

### 冻结教师任务指标

| 方法 | Macro-F1 | AUPRC | Fault Recall | FAR | 教师一致率 | Recall retention |
|---|---:|---:|---:|---:|---:|---:|
| degraded | 0.8433 | 0.9287 | 0.8388 | 0.1484 | 0.9086 | 1.0207 |
| simple | 0.8477 | 0.9244 | 0.8282 | 0.1320 | 0.9153 | 1.0078 |
| diffusion | 0.8057 | 0.9230 | 0.8553 | 0.2285 | 0.8694 | 1.0408 |

- Diffusion Fault Recall 比 simple 高2.71个百分点，说明没有抹除故障 Recall。
- AUPRC 比 simple 低0.14个百分点；Macro-F1低4.21个百分点；FAR高9.65个百分点。
- 教师一致率0.8694，显著高于旧实现0.4356，但仍低于 simple 0.9153。

### 训练资源

- 训练40 epoch：110.41秒。
- 含RData读取、validation/test full DDPM采样与评价：306.86秒。
- 峰值GPU已分配显存：163.15 MiB。

## 6. 第二级判定与状态

六项通过五项：

1. validation MAE正常收敛：通过；
2. test MAE在simple的10%以内：不通过；
3. Fault Recall不低于simple：通过；
4. FAR未重现旧0.900灾难且相对simple增量不超过10个百分点：通过，但非常接近边界；
5. 语义一致率明显高于旧0.4356：通过；
6. 至少一个任务指标优于simple：Fault Recall通过。

因此按“满足多数条件”输出状态C：

```text
DIFFUSION_RECOVERY_READY_FOR_IDEA_RETEST
```

这是一种“任务感知潜力存在但恢复误差仍落后”的状态，不能表述为 diffusion 优于 simple。

## 7. 唯一下一步建议

保持当前 diffusion checkpoint、manifest、MCAR mask 和单 seed 不变，只重新执行一次：

- Diffusion + Hard SupCon；
- Diffusion + Oracle Quality-weighted SupCon。

不得先加入 L_obs/L_temp/L_corr、质量评分网络、多 seed 或扩大数据。如果质量加权不能在相同 diffusion view 上改善 Recall/FAR/Macro-F1 的综合表现，则恢复为 NO-GO。
