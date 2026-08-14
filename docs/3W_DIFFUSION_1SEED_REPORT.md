# 3W Final Primary 单 Seed扩散对比

最终状态：`3W_FREQUENCY_SELECTIVE_R1_1SEED_GO`

Stage A 已为 `3W_FINAL_PRIMARY_STABILITY_GO`。本轮预先固定使用 grouped split 00（不是按 test 结果选择），仅运行 Seed 42。三臂共享 train/validation/test WELL、22 个 process feature、train-only preprocessor、TCN 初始化、Hard SupCon batch order、balanced frozen probe 与 clean test evaluation。

Clean 使用 identity second view；Uniform 与 Frequency-Selective R1 只改变 SupCon 第二正视图。关键频率 D（故障判别性）+ E（Early Fault 敏感性）+ S（跨 WELL 稳定性）全部从 3W split-00 train WELL 重算，未复用 TEP 频带。R1 固定 `t_key=1, t_nonkey=5`，Uniform 固定 `t=3`；两者 expected total spectral noise budget 均为 `0.0179736409`。R1 的关键/非关键预算为 `0.0125850/0.0202861`，符合关键频率 weak、非关键频率 moderate diffusion。

## 正式结果

| Method | Macro-F1 | Macro Recall | Binary AUPRC | Multiclass AUPRC | FAR | Early Recall | Delay (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clean Hard SupCon | 0.4806 | 0.4536 | 0.7843 | 0.5286 | 0.4394 | 0.9057 | 937.95 |
| Uniform Diffusion | 0.4764 | 0.4483 | 0.7791 | 0.5165 | 0.4388 | 0.9019 | 965.59 |
| Frequency-Selective R1 | 0.5008 | 0.4724 | 0.7839 | 0.5244 | 0.3445 | 0.8967 | 916.14 |

| Method | Class 0 Recall/F1 | Class 2 Recall/F1 | Class 8 Recall/F1 | Class 9 Recall/F1 |
|---|---:|---:|---:|---:|
| Clean | 0.5606 / 0.6668 | 0.2818 / 0.4299 | 0.9624 / 0.8111 | 0.0097 / 0.0145 |
| Uniform | 0.5612 / 0.6622 | 0.2818 / 0.4299 | 0.9406 / 0.8005 | 0.0097 / 0.0129 |
| R1 | 0.6555 / 0.7299 | 0.2818 / 0.4286 | 0.9426 / 0.8306 | 0.0097 / 0.0142 |

## R1 − Uniform

- Macro-F1：`+0.02447`
- Macro Recall：`+0.02407`
- Binary AUPRC：`+0.00480`
- Multiclass AUPRC：`+0.00785`
- FAR：`-0.09428`（改善）
- Early Recall：`-0.00518`，在预注册最大下降 0.01 内
- Detection Delay：`-49.45 s`（改善）

R1 同时改善 Macro-F1、两种 AUPRC 与 Delay，并显著降低 FAR；Early Recall 的轻微下降未超过容差，全部预注册 gate checks 通过。因此允许下一阶段做相同协议的 3-Seed 稳定性复核。

该 GO 只表示 R1 相对 Uniform 的单 Seed 工程信号成立。Class 9 Recall 在三臂均只有 0.0097，Class 2 Recall 也没有因 R1 改善；本轮信号主要来自 normal/FAR 与 Class 8 F1。下一阶段必须检查收益能否跨 seed 重现，不能把本结果表述为所有 fault class 均获益，也不得在 3-Seed 前据此调 class、test split 或阈值。
