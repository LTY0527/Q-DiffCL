# 3W Clean Hard-SupCon 单 Seed 基线

最终结论：`3W_CLEAN_BASELINE_1SEED_HOLD`

## 设置

- Seed 7；只运行一个 Seed。
- TCN：44 输入通道，hidden/projection=32，3 levels。
- Hard SupCon 20 epochs；按 validation SupCon loss 选择 encoder。
- 冻结 encoder 后 linear probe 15 epochs；按 validation Macro-F1 选择 probe。
- 训练最多每类 4,000 windows，验证最多每类 2,000 windows；固定 seed 分层抽取。测试使用全部 138,393 个有效窗口并逐 instance 流式评估。
- 无 diffusion、额外 MCAR、teacher、quality weighting、candidate ranking 或 stage curriculum。

## Test 指标

| Macro-F1 | Macro Recall | Fault Recall | Binary AUPRC | Multiclass AUPRC | FAR | Early Recall | Mean Delay | Detection rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2882 | 0.3422 | 0.7034 | 0.6165 | 0.3530 | 0.4473 | 0.9099 | 994.42 s | 0.2449 |

Early Recall/FAR/Delay 使用二元 fault-detection 口径；per-class 指标使用八分类口径。Accuracy=0.5896，仅作补充，不能作为主结论。

| 原始 class | Precision | Recall | F1 | support |
|---:|---:|---:|---:|---:|
| 0 | 0.8327 | 0.5527 | 0.6644 | 100696 |
| 1 | 0.0000 | 0.0000 | 0.0000 | 451 |
| 2 | 0.6426 | 0.6699 | 0.6560 | 306 |
| 4 | 0.0000 | 0.0000 | 0.0000 | 8170 |
| 5 | 0.0000 | 0.0000 | 0.0000 | 308 |
| 7 | 0.0000 | 0.0000 | 0.0000 | 1227 |
| 8 | 0.6741 | 0.9570 | 0.7910 | 26405 |
| 9 | 0.1175 | 0.5578 | 0.1941 | 830 |

## HOLD 诊断

训练数值有限，SupCon train loss 由 4.2013 降至约 3.70，probe train CE 由 1.8655 降至 0.3596；说明代码能学习且正常收敛。但 validation Macro-F1 最高仅 0.2807，测试中 class 1/4/5/7 recall 全为 0，混淆矩阵存在严重跨 WELL 类别塌缩。Normal recall 仅 0.5527，FAR 0.4473；49 个可评估 fault instance 中仅 12 个被检出，因此 Early Recall 虽高，不能视为可靠工业基线。

本阶段不根据 test 结果调整 feature、split、窗口、模型或训练参数，也不启动 Uniform/Frequency-Selective Diffusion。下一阶段首先应做不看 test 的 validation-based baseline failure diagnosis，例如检查跨 WELL domain shift、mask/equipment availability shortcut、linear probe 容量与 class-balanced batch；协议变更必须形成新版本，不能静默覆盖本冻结结果。

机器可读结果位于 `outputs/3w_clean_baseline_seed7/`；大模型 checkpoint 与可再生成中间结果不提交 Git，摘要与协议提交。
