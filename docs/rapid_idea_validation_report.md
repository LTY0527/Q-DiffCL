# Q-DiffCL 核心 Idea 三道闸快速验证报告

```text
RAPID_IDEA_VALIDATION
SINGLE_SEED
SUBSET_DATA
NOT_FOR_PAPER_CLAIMS
```

本报告记录修复前的初次快速验证，最终结论为 `NO-GO`。后续扩散训练/采样修复已达到状态 C，最新结论见 `docs/diffusion_recovery_debug_report.md`；本报告中的原始负面结果保留不删改。

本结论只适用于本次固定子集、单 seed 和最小实现，不构成论文结论。由于第二道闸失败，第三道闸按事前规则没有运行，也没有实现无参考质量评分器或扩大扩散实验。

## 1. 真实数据 Smoke Test

Smoke Test 已通过。配置为 `configs/tep_template.yaml`，真实 Rieth 2017 小子集、binary fault detection、TCN+CE、clean、seed 7、1 epoch、GPU。

- RData → `run_uid` → Run 级划分 → fault boundary → train-only scaler → window → DataLoader → TCN → CE → validation/test 全链路完成。
- Run 数：train 21、validation 21、test 42；三集合 `run_uid` 交集均为 0。
- 保留窗口：train 548、validation 548、test 2274。
- checkpoint、CSV、JSON、日志、配置和 split manifest 均成功保存至 `outputs/real_data_smoke/ce-clean-seed-7/`。
- 无 NaN/Inf。
- GPU 训练与评价耗时：4.37 秒；包含 RData 读取和预处理的脚本总耗时：82.15 秒。
- 峰值 GPU 已分配显存：18.09 MiB。

1 epoch 模型几乎全部预测为 fault，validation/test FAR 均为 1.0。这不影响工程链路验收，但证明该 Smoke 结果不能用于方法比较。

## 2. 固定快速验证子集

配置：`configs/rapid_idea_validation.yaml`。

- 唯一 seed：7。
- 官方 Training 按 `faultNumber` 分层形成 train/validation；官方 Testing 只用于 test。
- Training 候选：normal 160 Run，每个 fault 8 Run。
- 分层后：normal 128/32 train/validation；每个 fault 6/2 train/validation。
- Testing：normal 40 Run，每个 fault 2 Run。
- manifest Run 数：train 248、validation 72、test 80。
- 窗口数：train 6704、validation 1936、test 4440。
- 二分类窗口分布：train `[normal=3584, fault=3120]`；validation `[896,1040]`；test `[2560,1880]`。
- 窗口长度 64、stride 16、batch size 128。
- 唯一退化：normalized space 中确定性 MCAR missing 30%。
- epochs 上限 8，early stopping patience 2。
- 五组 G1 共享完全相同的 split manifest、窗口、标签和退化 realization。

所有产物位于 `outputs/rapid_idea_validation/`，并带有本报告开头的四个限制标记。

## 3. 第一闸配置与结果

- G1-0：clean 输入的 TCN+CE。
- G1-1：MCAR 30% degraded 输入的 TCN+CE。
- G1-2：`(x_clean,x_deg)` 等权 Hard SupCon，随后冻结 encoder 做 linear probe。
- G1-3：同一对视图，使用 `q=exp(-masked_MAE/mean_masked_MAE)` 为正样本加权；q 截断到 `[0,1]`。
- G1-4：线性插值 simple recovery，与 clean 组成正样本，并使用同样的 oracle masked-MAE quality。

Oracle quality 只使用人工遮挡位置的 clean 真值，不使用测试标签，也没有训练质量评分网络。

| 方法 | Macro-F1 | AUPRC | Fault Recall | FAR | Masked MAE | 教师一致率 | 类中心偏移 | Effective rank | 训练时间/s | 峰值显存/MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G1-0 Clean CE | 0.8959 | 0.9335 | 0.8218 | 0.0422 | N/A | 1.0000 | 0.0000 | 1.010 | 6.41 | 26.1 |
| G1-1 Degraded CE | 0.8910 | 0.9322 | 0.8074 | 0.0391 | N/A | 0.9086 | 8.3325 | 1.009 | 2.46 | 26.2 |
| G1-2 Hard SupCon | 0.8313 | 0.9358 | 0.6388 | 0.0000 | N/A | 0.9086 | 0.4302 | 1.057 | 6.73 | 30.5 |
| G1-3 Oracle Quality SupCon | 0.8419 | 0.9366 | 0.6596 | 0.0000 | N/A | 0.9086 | 0.3313 | 1.053 | 6.15 | 30.6 |
| G1-4 Simple Recovery + Quality | 0.8548 | 0.9339 | 0.6862 | 0.0008 | 0.3134 | 0.9153 | 0.0248 | 1.126 | 6.31 | 30.7 |

G1 数据准备和五组训练总耗时为 117.20 秒。

### 第一闸判断

- Hard SupCon 在 degraded positive 下的 Macro-F1 明显低于 Clean/Degraded CE，且故障 Recall 从 Degraded CE 的 0.8074 降到 0.6388，H1 得到初步支持。
- Oracle Quality 相对等权 Hard SupCon：Macro-F1 +1.06 个百分点，AUPRC +0.08 个百分点；方向为正，但没有达到预设的 1.5 个百分点复核信号。
- Simple Recovery + Quality 相对 Hard SupCon：Macro-F1 +2.36 个百分点，达到工程复核信号；masked MAE 由直接零填补的 0.5680 降为 0.3134。
- SupCon embedding 的 effective rank 均接近 1，提示本子集/训练设置存在低秩或强谱集中风险，不能据此声称质量加权机制已得到稳定验证。

第一闸记为有限 `GO`，仅允许进入最小扩散恢复筛选。

## 4. 第二闸配置与结果

最小扩散器仅支持 MCAR 30%：

- observation mask 条件；
- 轻量 1D residual epsilon predictor；
- 20 diffusion steps；
- hidden channels 64，3 个 residual blocks；
- 最多 12 epoch，early stopping patience 2；
- observed 位置在训练和恢复过程中始终钳制为原始观测值；
- 不做 OOD、噪声/漂移/spike/mixed、联合训练或质量评分网络。

使用 G1-0 的冻结 Clean CE 模型，在完全相同 test 窗口上比较：

| 输入 | Masked MAE | Masked RMSE | 相关矩阵误差 | Macro-F1 | AUPRC | Fault Recall | FAR | 教师一致率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| x_deg | 0.5680 | 1.1088 | 0.1005 | 0.8433 | 0.9287 | 0.8388 | 0.1484 | 0.9086 |
| x_simple | 0.3134 | 0.5947 | 0.0173 | 0.8477 | 0.9244 | 0.8282 | 0.1320 | 0.9153 |
| x_diff | 0.7454 | 1.1802 | 0.1100 | 0.3958 | 0.9081 | 0.9814 | 0.9000 | 0.4356 |

扩散训练 12 epoch，实测训练耗时 8.14 秒，峰值 GPU 已分配显存 86.18 MiB；包含重新读取 RData、恢复和评价的总耗时为 123.56 秒。

### 是否抹除故障语义

扩散输出没有表现为“故障 Recall 被平滑掉”：Fault Recall 反而升到 0.9814。但 FAR 从 simple 的 0.1320 暴涨到 0.9000，教师一致率从 0.9153 降到 0.4356，Macro-F1 降到 0.3958。这说明恢复结果把大量正常窗口推向 fault，而不是可靠保存故障语义；属于严重语义失真/故障幻觉，同样不可接受。

### 第二闸判断

`NO-GO`：

1. Diffusion masked MAE 0.7454，显著差于 simple 0.3134，甚至差于直接 degraded 0.5680。
2. Diffusion AUPRC 0.9081，低于 degraded 0.9287 和 simple 0.9244。
3. FAR=0.9000，语义一致率=0.4356，故障检测行为不可用。

## 5. 第三闸

第三闸状态：`NOT_RUN`。

因为第二闸失败，严格停止以下实验：

- G3-0 Diffusion + Hard SupCon；
- G3-1 Diffusion + Oracle Quality-weighted SupCon；
- 无参考质量评分器；
- 多 seed；
- 完整扩散或端到端联合训练。

因此 H3 仅在 degraded oracle 权重上观察到小幅正向变化，尚未获得扩散恢复场景支持。

## 6. 核心 Go / No-Go 表

下表中 G1 方法使用各自训练后的分类头；Diffusion 行使用冻结 G1-0 Clean CE 教师，比较口径不同，不应作论文式横向排名。

| 方法 | Macro-F1 | AUPRC | Recall | FAR | Masked MAE | 语义一致率 | 时间/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clean CE | 0.8959 | 0.9335 | 0.8218 | 0.0422 | N/A | 1.0000 | 6.41 |
| Degraded CE | 0.8910 | 0.9322 | 0.8074 | 0.0391 | N/A | 0.9086 | 2.46 |
| Hard SupCon | 0.8313 | 0.9358 | 0.6388 | 0.0000 | N/A | 0.9086 | 6.73 |
| Oracle Quality SupCon | 0.8419 | 0.9366 | 0.6596 | 0.0000 | N/A | 0.9086 | 6.15 |
| Simple Recovery + Quality | 0.8548 | 0.9339 | 0.6862 | 0.0008 | 0.3134 | 0.9153 | 6.31 |
| Diffusion Recovery + Classifier | 0.3958 | 0.9081 | 0.9814 | 0.9000 | 0.7454 | 0.4356 | 8.14 |
| Diffusion + Hard SupCon | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |
| Diffusion + Quality SupCon | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |

## 7. 最终结论与唯一建议

总体结论：`NO-GO`。

- H1：初步支持，但仅为单 seed 子集信号。
- 恢复视图：简单插值在 G1 中有一定正向信号。
- H2：最小扩散恢复不成立。
- H3：扩散场景未进入验证；oracle degraded 权重收益不足以支持开发无参考质量评分器。

唯一建议：暂停完整 Q-DiffCL、无参考质量评分器和多 seed；只在当前固定子集上审计最小扩散的训练目标与采样器，首先要求其 masked MAE/RMSE 至少不差于线性插值且 FAR 不恶化。达到这一恢复层门槛前，不再运行任何 Diffusion+SupCon 实验。

## 8. 最终工程回归

初次快速验证完成时测试为 `25 passed`。加入后续扩散诊断后的当前完整回归为 `31 passed in 128.03s`，failed=0，skipped=0；最新诊断见独立报告。
