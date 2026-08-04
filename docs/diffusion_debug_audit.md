# 扩散恢复实现审计记录

```text
DIFFUSION_DEBUG
SINGLE_SEED
NOT_FOR_PAPER_CLAIMS
```

## 审计对象

- `models/minimal_diffusion.py::MinimalConditionalDiffusion1D`
- `scripts/run_rapid_diffusion_gates.py::_schedule`
- `scripts/run_rapid_diffusion_gates.py::_diffusion_loss`
- `scripts/run_rapid_diffusion_gates.py::_restore`
- 最近结果：`outputs/rapid_idea_validation/gate2_gate3_results.json`

## 逐项结论

1. 旧 `q_sample` 使用 `sqrt(alpha_bar)*x0 + sqrt(1-alpha_bar)*epsilon`，epsilon 参数化本身正确。
2. 网络预测目标为 epsilon；训练 loss 只在 missing 位置计算，mask 语义始终为 `True/1=observed`。
3. `[B,C,L]` shape 一致；网络输入明确拼接 noisy、degraded observed values 和 observation mask，mask 并非只在外部保存。
4. timestep 使用 0-based，并通过正弦 embedding 和每个 residual block 的线性投影进入网络。
5. 旧训练与旧采样使用同一线性 alpha_bar 数组，但存在严重初态错配：20 个 `beta=1e-4..0.02` 的终端 `alpha_bar≈0.817`，训练最重噪声仍保留大量 x0；采样却从纯噪声开始。该分布错配是首要风险。
6. 旧 `_restore` 是确定性 DDIM-like 更新，不是带 DDPM posterior variance 的 full-step sampling，不能满足本轮 Tiny Overfit 对 DDPM 链路的验证要求。
7. 旧最后一步没有额外随机噪声，这一点正确；observed values 每一步都重新 clamping，也正确。
8. 旧 loss 只有 epsilon MSE，没有 `lambda_rec * masked reconstruction loss`；短 schedule 下 epsilon loss 下降不保证 x0 恢复正确。
9. 旧模型 3 个无 dilation 残差块的有效感受野明显小于 length=64，可能不足以插补长依赖。本轮改为 dilation 1/2/4/8，使主要感受野覆盖整个窗口。
10. degraded/simple/diffusion 的 MAE 使用相同 normalized space、相同 fixed mask 和相同 test 窗口，指标比较口径一致。
11. 旧训练仅 12 epoch、8.14 秒，虽然 validation epsilon loss下降，但不足以证明模型能拟合恢复目标。

## 本轮修复

- 新增 `diffusion/process.py::DiffusionSchedule.cosine`，使训练终端接近纯噪声。
- 统一提供 `q_sample`、`predict_x0`、DDPM posterior step 和 `ddpm_restore`，训练与采样共享同一 schedule。
- DDPM 每一步重新固定 observed values，step 0 不再加噪声。
- Tiny 训练使用 `L_diff + lambda_rec * L_masked_rec`。
- 增加 q_sample、x0 回算、mask/clamping、确定性采样、参数化一致性和 one-batch overfit 测试。

在 Tiny Overfit 通过前，不能判断扩散 Idea 本身无效。

## 修复后验证结论

- 第一次 Tiny 仍使用不稳定 posterior 路径，masked MAE=164.77，保留为失败证据。
- 修复为标准 x0 posterior、train-only clipping 并增加实际 gradient batch 后，Tiny masked MAE=0.5016，七项通过六项。
- 小型子集 diffusion masked MAE=0.4010，较旧0.7454明显改善；Fault Recall=0.8553，高于simple=0.8282，但MAE/AUPRC/Macro-F1/FAR仍落后simple。
- 当前状态与完整限制见 `docs/diffusion_recovery_debug_report.md`。
- 历史入口 `scripts.run_rapid_diffusion_gates` 已显式禁用，防止再次使用旧 schedule/sampler；历史 JSON 和辅助评价函数仍保留用于审计。
