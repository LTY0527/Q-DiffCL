# 故障阶段—频率选择性扩散对比学习 MVP 报告

> **STAGE_FREQUENCY_DIFFUSION_MVP / FORWARD_DIFFUSION_ONLY / FIXED_TEP_SUBSET / NOT_FOR_PAPER_CLAIMS**

## 结论

频率关键性审计通过，但单 seed MVP 最终状态为：

```text
FREQUENCY_SELECTIVE_DIFFUSION_MVP_NO_GO
```

C2 的频率选择机制按设计工作：在与 C1 完全相同的期望总噪声预算下，关键频带扰动更弱、非关键频带扰动更强，关键 Fisher 与 early sensitivity 保留也略好。然而这些机制指标没有转化为综合检测收益：C2 相比 C1 的 Macro-F1 降低 0.366 个百分点，FAR 上升 1.367 个百分点，且没有达到预设强工程信号。因此自动 3-seed 复核被 gate 阻止，本阶段停止，不实现 C3/C4/C5。

## 为什么停止旧主线

既有质量评价、质量加权、多候选排序和教师语义约束路线已得到多轮 NO-GO。最后的教师审计显示分类指标尚可，但 guidance embedding effective rank 约为 1.01，无法支持继续调 JS、Margin 或 Feature loss。本阶段不再修补该链路，也没有训练反向扩散恢复器、教师网络或质量模型。

新主线的核心问题是：工业故障是否存在跨 Run 可重复并对早期故障敏感的关键频率；若存在，在相同总扰动预算下保护这些频率，是否能比统一频谱扩散产生更好的对比学习表示。

## 仓库复用与冻结协议

本实现复用了固定 MCAR 视图、简单插值 `x_base`、Run 级 split、TCN、Hard SupCon、Probe、validation threshold、表示诊断和 cosine diffusion schedule。C0/C1/C2 共享：

- 固定 split、MCAR mask 与 `x_base`；
- encoder、projection head 和 Probe 初始状态；
- SupCon/Probe batch order；
- optimizer、学习率、8+8 epoch、temperature；
- Probe epoch 选择与 validation threshold 算法；
- test set 和样本总权重 1.0。

唯一变量是正样本增强视图来源。初始化 SHA256 为 `dadf6884be39b2a36fdcfa57e3c0ae42dcbef60ecc9524fbcb63ea8c9484c6fb`；配置指纹为 `197268730294e4f86630516f75bf2e4c20e37c1d5d4b2649d01aae4e1271c701`。

## 故障阶段定义

training/testing 的真实 onset 分别为 21/161，窗口长度 64、步长 16，transition 窗口继续排除。保存原始：

```text
delta = window_end_sample - fault_onset
```

为让不同绝对 onset 下的“首 N 个完整故障窗口”含义一致，阶段进度使用：

```text
progress = delta - (window_length - 1)
```

- `progress < 4*stride`：early；
- `4*stride <= progress < 12*stride`：middle；
- 之后：stable；
- 正常 Run 或完整 onset 前窗口：prefault。

test 中 prefault/early/middle/stable 分别为 2560/160/320/1400 个窗口。stage 只用于频率审计、Early Recall 和 Detection Delay，没有参与训练权重或 test threshold 选择。

## D/E/S 与三种 Mask 审计

- D：train Run 聚合后的 normal/fault Fisher 比；
- E：train early fault Run 与 train normal Run 的 Fisher 比；
- S：fault Run 相对 normal Run 中位参考的方向一致性，并用稳健变异系数惩罚；
- Composite：train-only median/IQR 标准化后，`0.5D + 0.3E + 0.2S`。

30% critical ratio 下的结果：

| 审计项 | 数值 |
|---|---:|
| Train Run bootstrap mask overlap | 0.8579 ± 0.0244 |
| 关键/非关键 Fisher 比 | 5.6209 |
| 关键频带/随机同规模频带 early sensitivity | 2.2524 |
| Energy/Composite Jaccard | 0.0911 |
| Fisher/Composite Jaccard | 0.8198 |
| Validation 关键频率方向一致率 | 1.0000 |
| 选中项中 bin>2 的比例 | 0.8369 |

六项频率 gate 全部通过，状态为 `FREQUENCY_CRITICALITY_AUDIT_GO`。完整通道 Top 频率、Fault 3/9/15 和三张图见 `docs/frequency_criticality_audit.md`。validation/test 只做外部验证，未参与 scaler、D/E/S、mask 或权重拟合。

## C1/C2 与噪声预算公平性

C1 对 train-normalized log amplitude 的所有非 DC bin 使用统一 `t_uniform=3`。C2 使用 composite soft mask，在 `t_critical=1` 与 validation 选择的 `t_noncritical` 之间分配连续方差；候选为 3/5/8。

seed 7 validation 选择结果：

| t_noncritical | Critical Fisher retention | Early retention | Critical/Noncritical perturbation | 选择分数 |
|---:|---:|---:|---:|---:|
| 3 | 0.9907 | 0.9883 | 0.7081 | 1.97895 |
| 5 | 0.9924 | 0.9900 | 0.6515 | 1.98237 |
| 8 | **0.9937** | **0.9913** | **0.6060** | **1.98498** |

因此冻结 `t_noncritical=8`。test 未参与选择。

C1/C2 的期望总频谱噪声预算均为 `0.0179736409`。C2 的关键/非关键预算为 `0.0110358/0.0209487`；C1 对应为 `0.0177075/0.0180878`（DC 保持造成 mask 分组内轻微差异）。test 时域 normalized L1 分别为 0.05167/0.05178，实际总频域 normalized L1 分别为 0.05673/0.05692，说明 C2 并非靠减小总扰动获益。

增强均为有限值，DC 保持，逆 FFT 最大重构误差不超过 `5.73e-6`，没有 NaN/Inf。test 异常尖峰比例约为 C1 `6.03e-4`、C2 `6.04e-4`。

## Seed 7 MVP

| 方法 | Macro-F1 | AUPRC | AUROC | Recall | FAR | Early Recall | 检测率 | Mean Delay |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 传统增强 | **0.8936** | 0.9284 | 0.9118 | 0.7878 | **0.0188** | 0.7500 | 0.8500 | 111.00 |
| C1 统一频谱扩散 | 0.8895 | **0.9314** | **0.9176** | 0.8021 | 0.0375 | 0.7938 | 0.8500 | 103.00 |
| C2 频率选择性扩散 | 0.8858 | 0.9312 | 0.9174 | **0.8112** | 0.0512 | **0.8000** | 0.8500 | **101.59** |

Detection Delay 要求 onset 后连续 3 个窗口预测为 fault；40 条 test fault Run 中三种方法均检测 34 条、漏检 6 条，漏检 Run 没有从 delay 汇总中悄然删除，检测率单独报告为 0.85。

逐方法训练时间与峰值显存：

| 方法 | 训练时间（秒） | 峰值显存（MiB） |
|---|---:|---:|
| C0 | 310.8 | 76.2 |
| C1 | 299.4 | 76.2 |
| C2 | 301.6 | 76.2 |

三方法训练合计约 911.8 秒；包含频谱生成、候选选择、机制审计和汇总的本次命令墙钟时间约 949 秒。

## 差值与 Gate

| 对比 | Δ Macro-F1 | Δ AUPRC | Δ Recall | Δ FAR | Δ Early Recall |
|---|---:|---:|---:|---:|---:|
| C2 - C1 | -0.00366 | -0.00023 | +0.00904 | +0.01367 | +0.00625 |
| C2 - C0 | -0.00778 | +0.00283 | +0.02340 | +0.03242 | +0.05000 |

单 seed gate：

- 失败：Macro-F1 提高；
- 失败：FAR 降低；
- 通过：AUPRC 降幅不超过 0.5 个百分点；
- 通过：Recall 降幅不超过 1 个百分点；
- 通过：Early Recall 提高或 delay 缩短；
- 通过：关键频带保留明显优于 C1；
- 通过：总噪声预算公平；
- 通过：无数值异常；
- 失败：达到 Macro-F1 +0.5pp、FAR -1pp 或 Early Recall +1pp 的强工程信号。

C2 表现为 Recall/early sensitivity 上升，同时 FAR 上升、Macro-F1 下降。它没有优于 C1 的综合检测结果，因此输出 `FREQUENCY_SELECTIVE_DIFFUSION_MVP_NO_GO`。

## 3-Seed 与下一步

seed 7 gate 未通过，seed 42/2026 未运行，不能报告 mean±std，也不能外推稳定性。`FREQUENCY_SELECTIVE_DIFFUSION_3SEED_GO/NO_GO` 均不适用。

唯一下一步建议：停止当前故障阶段—频率选择性扩散主线，保留本次“频率假设成立但对比学习收益未成立”的负结果；不得自动加入 C3 阶段课程，也不得实现 C4/C5。

> 本实验仅基于固定小型 TEP 子集：**NOT_FOR_PAPER_CLAIMS**。
