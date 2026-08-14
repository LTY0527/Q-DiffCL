# 3W Strict Real-only Protocol 与 Missing-Mask 消融

最终状态：`3W_REAL_ONLY_PRIMARY_HOLD`

Base commit：`294caa5`。本阶段按上一轮预先登记、非 test-driven 的 Protocol A，把旧 Primary `0,1,2,4,5,7,8,9` 重构为 Strict Primary `0,2,4,7,8,9`；class 1/5 因各只有一口 training target WELL 转入 Secondary，class 3/6 原已在 Secondary。未使用 SIMULATED/DRAWN。

## 固定协议

新 manifest 复用旧协议的 24/8/8 WELL 分组，只缩减 class，保证同 WELL 不跨 split并维持与旧结果的最大可比性。

| class | train target WELL | validation target WELL | test target WELL |
|---:|---:|---:|---:|
| 0 | 24 | 8 | 8 |
| 2 | 3 | 2 | 2 |
| 4 | 4 | 2 | 1 |
| 7 | 4 | 1 | 1 |
| 8 | 6 | 1 | 2 |
| 9 | 4 | 1 | 2 |

冻结 window=64 s、stride=32 s、TCN hidden/projection=32、Seed=42、普通 shuffled Hard SupCon 20 epochs、sqrt-inverse balanced frozen probe 15 epochs。上一阶段 positive-anchor rate 已是 1.0 且 balanced SupCon 使 FAR/Macro-F1 恶化，因此没有强行保留 balanced SupCon sampler。

Strict train-only 重新拟合原有 22 个 process feature 的 quantile clip、median imputation、mean/std；两臂共享完全相同的 split、窗口引用、preprocessing statistics、训练顺序、probe weights 和评价代码。唯一变量是追加 22 个 native missing-mask channels。

## Mask ablation

| Condition | Macro-F1 | Macro Recall | Fault Recall | Binary AUPRC | Multiclass AUPRC | FAR | Early Recall | Mean Delay | Detection rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Process only（22ch） | 0.4209 | 0.4855 | 0.7347 | 0.6061 | 0.4839 | 0.1709 | 0.9411 | 1133.0 s | 0.2128 |
| Process + mask（44ch） | 0.3796 | 0.4515 | 0.7298 | 0.7298 | 0.4580 | 0.4633 | 0.9457 | 19.4 s | 0.2128 |

Process+mask 提高 binary AUPRC，并在同样仅 21.28% fault instance 被检出的条件下让已检出实例更早报警；但 FAR 从 0.1709 增至 0.4633，Macro-F1/Recall、multiclass AUPRC 和 normal recall 均显著下降。结合上一阶段 mask-only WELL-ID=82.10%，整体证据表明 mask 主要放大 WELL/报警 shortcut，净工业收益为负。冻结输入建议为 `PROCESS_ONLY`，但该选择不等于 baseline GO。

## Per-class

| class | Process-only Recall/F1 | Process+Mask Recall/F1 |
|---:|---:|---:|
| 0 | 0.8291 / 0.8600 | 0.5367 / 0.6554 |
| 2 | 0.6601 / 0.5950 | 0.6699 / 0.6560 |
| 4 | 0 / 0 | 0 / 0 |
| 7 | 0 / 0 | 0 / 0 |
| 8 | 0.8657 / 0.8808 | 0.9277 / 0.9096 |
| 9 | 0.5578 / 0.1898 | 0.5747 / 0.0565 |

两臂 class 4/7 都仍为 zero recall，说明移除 1/5 和 mask 并不能消除核心 cross-WELL collapse。Process-only 相比旧八类 Seed 7 D0 的 Macro-F1 0.2882/FAR 0.4473 更稳定，但本轮使用 Seed 42且 class vocabulary 已变化，只能作为协议改善的探索性证据，不能解释为严格单变量模型增益。

## 结论

`3W_REAL_ONLY_PRIMARY_HOLD`。Strict protocol 改善了总体稳定性并明确支持 `PROCESS_ONLY`，但 6 个 Primary 类中仍有 2 类 zero recall，且 detection instance rate 仅 0.2128。主要阻塞仍是 class 4/7 的跨 WELL shift 与 validation/test 仅一口 target WELL，而非 mask 本身。

当前不具备进入 Uniform Diffusion 1-Seed vs Frequency-Selective R1 1-Seed 的条件。按停止线不继续调参、不缩减更多 class、不启动 diffusion。下一步只能在新指令下决定是进一步收缩 Real-only claim、重定义论文任务，还是把当前结果定位为高难度 Secondary protocol。

大型 checkpoint、完整训练历史、confusion matrix 与机器可读 result 位于 `outputs/3w_strict_mask_ablation_seed42/`，不提交 Git。
