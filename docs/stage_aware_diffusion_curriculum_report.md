# 故障阶段感知频率扩散课程增量验证

> **STAGE_AWARE_DIFFUSION_CURRICULUM / INCREMENTAL_C3_VALIDATION / FIXED_R1_BASELINE / EXPLORATORY_TEP_SUBSET / NOT_FOR_PAPER_FINAL_CLAIMS**

## 结论

Seed 7：`STAGE_AWARE_CURRICULUM_SEED7_NO_GO`。最终状态：`STAGE_AWARE_CURRICULUM_SEED7_NO_GO`。三个 Seed 完成：`False`。

R1 已通过 3-Seed。本轮唯一问题是 training-time stage prior 能否在保持 R1 Macro-F1/FAR 的同时改善 Early Recall 或 Delay。C3-E 用于隔离普通 epoch curriculum；C3-S 才加入 stage target。stage 基于真实 onset 和已排除 transition 的窗口，仅选择训练增强强度，不进入 encoder、Probe、threshold、test 推理或后处理；这是 supervised fault-detection setting 的 training-time stage prior，不是自监督 stage discovery。

## 固定设计

- R1：非关键频率固定 t=5。
- C3-E：所有阶段从 t=2 线性课程到 t=5，不读取 stage。
- C3-S：normal/early/middle/stable 从 t=2 分别到 5/3/4/5。early=3、middle=4 用于保护早期弱故障并随故障发展逐步增加难度；stable/normal 回到 R1 强度。
- stage：真实 onset 后完整窗口 progress；early `<4*stride`，middle `<12*stride`，其后 stable。
- 指纹：`{'manifest_sha256': '1824e2cfa0b86ef71afe2d38913134ea418d9d7dda5bbf9e624a496faff88eb1', 'mask_sha256': 'd2e1879bc012ac1326ea8c721461d31641105af2c4fe0c89eacdebac413b9395', 'r1_config_sha256': '5d8dc5692aa4291c43c1a226ead2e6e00e922ecc7580b4e5aeabb7702caf4384', 'training_code_sha256': '15126559490dad74a226369a9ceee05db755ce1b3d0afe105ee9e4f17bc48792'}`。

## 逐 Seed 指标

| Seed | 方法 | Macro-F1 | AUPRC | Recall | FAR | Early Recall | Mean Delay |
|---:|---|---:|---:|---:|---:|---:|---:|
| 7 | R1 | 0.8920 | 0.9316 | 0.7979 | 0.0297 | 0.7937 | 103.00 |
| 7 | C3-E | 0.8920 | 0.9315 | 0.7984 | 0.0301 | 0.7937 | 103.00 |
| 7 | C3-S | 0.8918 | 0.9316 | 0.7979 | 0.0301 | 0.7937 | 103.00 |

## Mean ± sample std

| 方法 | Macro-F1 | FAR | Early Recall | Mean Delay |
|---|---|---|---|---|
| R1 | 0.8920 ± 0.0000 | 0.0297 ± 0.0000 | 0.7937 ± 0.0000 | 103.00 ± 0.00 |
| C3-E | 0.8920 ± 0.0000 | 0.0301 ± 0.0000 | 0.7937 ± 0.0000 | 103.00 ± 0.00 |
| C3-S | 0.8918 ± 0.0000 | 0.0301 ± 0.0000 | 0.7937 ± 0.0000 | 103.00 ± 0.00 |

由于 Seed 7 Gate 未通过，本表只有一个 Seed，标准差记为 0；未将其伪装成 3-Seed 汇总。

## Seed 7 配对增量

`ΔFAR<0`、`ΔDelay<0` 才表示改善。

| 比较 | ΔMacro-F1 | ΔAUPRC | ΔRecall | ΔFAR | ΔEarly | ΔDelay |
|---|---:|---:|---:|---:|---:|---:|
| C3-S - R1 | -0.00023 | -0.00007 | +0.00000 | +0.00039 | +0.00000 | +0.00 |
| C3-S - C3-E | -0.00025 | +0.00005 | -0.00053 | +0.00000 | +0.00000 | +0.00 |
| C3-E - R1 | +0.00002 | -0.00012 | +0.00053 | +0.00039 | +0.00000 | +0.00 |

## 课程强度与增强坍缩审计

| 方法 | Epoch | Mean t | normal/early/middle/stable t | Overall L1 | Normal L1 | Early L1 | Middle L1 | Stable L1 | Critical L1 | Noncritical L1 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| R1 | 首 epoch | 5.000 | 5 / 5 / 5 / 5 | 0.05208 | 0.05577 | 0.04972 | 0.04843 | 0.04696 | 0.03081 | 0.04731 |
| R1 | 末 epoch | 5.000 | 5 / 5 / 5 / 5 | 0.05208 | 0.05577 | 0.04972 | 0.04843 | 0.04696 | 0.03081 | 0.04731 |
| C3-E | 首 epoch | 2.000 | 2 / 2 / 2 / 2 | 0.05221 | 0.05541 | 0.05048 | 0.04926 | 0.04754 | 0.03459 | 0.04563 |
| C3-E | 末 epoch | 5.000 | 5 / 5 / 5 / 5 | 0.05208 | 0.05577 | 0.04972 | 0.04843 | 0.04696 | 0.03081 | 0.04731 |
| C3-S | 首 epoch | 2.000 | 2 / 2 / 2 / 2 | 0.05221 | 0.05541 | 0.05048 | 0.04926 | 0.04754 | 0.03459 | 0.04563 |
| C3-S | 末 epoch | 4.714 | 5 / 3 / 4 / 5 | 0.05214 | 0.05577 | 0.05015 | 0.04863 | 0.04696 | 0.03109 | 0.04719 |

C3-S 末期 early/middle timestep 确实为 3/4，normal/stable 为 5；其 overall normalized L1 与 R1 接近，因此失败不是由整体 augmentation collapse 造成。完整 8-epoch 轨迹保存在 metrics.json。

## 分数翻转与表示诊断

| 方法 | normal→fault | fault→normal | Representation Fisher | Effective rank |
|---|---:|---:|---:|---:|
| R1 | 0.0297 | 0.2021 | 1.8249 | 1.1700 |
| C3-E | 0.0301 | 0.2016 | 1.8210 | 1.1722 |
| C3-S | 0.0301 | 0.2021 | 1.8214 | 1.1715 |

Seed 7 Gate：`{'preservation': {'macro_f1': True, 'far': True, 'recall': True, 'auprc': True}, 'early_or_delay': False, 'stage_gain_over_c3e': False, 'epoch_gain': False}`。3-Seed Gate：`{'three_seed_skipped': True}`。C3-S 保持四项核心边界，但 Early Recall/Delay 无实质改善，且不优于 C3-E；C3-E 自身也无工业改善。因此输出 NO-GO 并跳过 3-Seed。完整 Middle/Stable、median delay、检测率和 missed runs 在 outputs 的 metrics.json。

课程审计用于确认 C3-S 不是整体 augmentation collapse；correlation/频带机制不作为选择条件。本阶段只报告 mean、sample std、配对方向，不计算 p-value，不声称统计显著。

当前 TEP test 已经历多轮工程探索，因此本阶段仍不是论文最终无偏评测。本轮 C3 未通过，唯一下一步是停止该 Stage-aware 增量，不搜索新 target、t_start 或非线性 schedule，不增加 C4/C5；若未来继续研究，应优先转向第二数据集或新的未触碰评测协议。
