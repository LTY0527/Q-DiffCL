# 3W 随机性来源拆解

最终机制判定：`ENCODER_OPTIMIZATION_DOMINANT`

本轮严格冻结 Final Primary `[0,2,8,9]`、canonical grouped split 00、process-only、64/32 window/stride、protocol seed 42、train/validation window refs、train-only preprocessing、critical soft mask、TCN、Hard SupCon、balanced probe、Uniform `t=3` 与 R1 `t_key=1/t_nonkey=5`。未调参、未更换阈值、split 或类别，也未实现新方法。

共完成 18 个方法结果：A/B/C 三类审计 × 3 seeds × Uniform/R1。A 固定 encoder/probe=42、validation augmentation=42，只改变 train diffusion noise；B 固定 train/validation diffusion views=42、probe=42，只改变 encoder initialization 与 SupCon order；C 直接加载 Seed 42 encoder checkpoint，不做 SupCon、不生成扩散视图，只改变 probe initialization/order。所有记录的 window refs SHA256 与 critical mask SHA256 完全一致。

## 三类来源的 paired R1 − Uniform

| Audit | Varied seed | Macro-F1 | Binary AUPRC | Multiclass AUPRC | FAR | Class 9 Recall |
|---|---:|---:|---:|---:|---:|---:|
| A Diffusion | 42 | +0.02447 | +0.00480 | +0.00785 | -0.09428 | 0.00000 |
| A Diffusion | 43 | +0.01836 | -0.00398 | -0.01940 | -0.08611 | -0.00092 |
| A Diffusion | 44 | -0.00651 | -0.02808 | +0.03444 | +0.05520 | -0.00092 |
| B Encoder | 42 | +0.02447 | +0.00480 | +0.00785 | -0.09428 | 0.00000 |
| B Encoder | 43 | -0.02156 | +0.01207 | -0.01582 | +0.01781 | +0.30299 |
| B Encoder | 44 | +0.01046 | -0.00557 | +0.02661 | -0.03466 | 0.00000 |
| C Probe | 42 | +0.02447 | +0.00480 | +0.00785 | -0.09428 | 0.00000 |
| C Probe | 43 | +0.03236 | +0.00041 | +0.02459 | -0.18394 | -0.00046 |
| C Probe | 44 | +0.02491 | +0.00499 | +0.00440 | -0.11516 | 0.00000 |

## 波动幅度

| Source | Binary AUPRC Δ mean ± std | Class 9 Recall Δ mean ± std | Macro-F1 Δ mean ± std | FAR Δ mean ± std | Combined score |
|---|---:|---:|---:|---:|---:|
| A Diffusion | -0.00909 ± 0.01390 | -0.00061 ± 0.00043 | +0.01211 ± 0.01340 | -0.04173 ± 0.06862 | 0.00717 |
| B Encoder | +0.00377 ± 0.00724 | +0.10100 ± 0.14283 | +0.00446 ± 0.01927 | -0.03704 ± 0.04579 | 0.07503 |
| C Probe | +0.00340 ± 0.00211 | -0.00015 ± 0.00022 | +0.02725 ± 0.00362 | -0.13112 ± 0.03830 | 0.00117 |

综合分数预先定义为 Binary AUPRC delta std 与 Class 9 Recall delta std 的等权平均；B/A 比值为 10.47，超过预注册 1.25× 主导阈值，因此整体判定 `ENCODER_OPTIMIZATION_DOMINANT`。Probe 的 Macro-F1 delta 最稳定，不能解释原始 Class 9 大幅波动。

## Seed 44 Binary AUPRC 与 P(normal)

Seed 44 的 Binary AUPRC delta 在 A/B/C 分别为 `-0.02808 / -0.00557 / +0.00499`。最大下降来自 diffusion-only，但没有单独复现原耦合 3-Seed 实验的 `-0.06973`；原始异常更可能是 diffusion noise 与 encoder optimization 的交互放大，而非 probe。

| Seed-44 audit | Uniform P(normal) normal mean/median | R1 P(normal) normal mean/median | Uniform P(normal) fault mean/median | R1 P(normal) fault mean/median |
|---|---:|---:|---:|---:|
| A Diffusion | .4688 / .5169 | .4221 / .4845 | .1853 / .0672 | .2013 / .1525 |
| B Encoder | .4502 / .5423 | .4652 / .5671 | .1159 / .0136 | .1211 / .0161 |
| C Probe | .4113 / .4233 | .4510 / .4575 | .1942 / .1635 | .2026 / .1504 |

A/Seed 44 中，R1 同时降低 normal 样本的 P(normal) 并提高 fault 样本的 P(normal)，导致两类分数分布靠近，与 Binary AUPRC 下降及 FAR 上升一致。完整 5/25/50/75/95% 分位数位于 paired CSV 和 JSON。

## Class 9

Class 9 不稳定明确集中于 B Encoder：Class 9 paired Recall delta std 为 `0.14283`，而 A/C 仅为 `0.00043/0.00022`，B 相对第二名约 329.5×。B/Seed 43 中 R1 相对 Uniform 的 Class 9 Recall 提升 `+0.30299`，换到 Encoder Seed 44 后该增益消失；改变 diffusion seed 或 probe seed 都无法产生同等级变化。因此 Class 9 的主要来源是 encoder initialization + SupCon optimization，而不是 diffusion noise 或 classification head。

## 结论

- 整体机制：`ENCODER_OPTIMIZATION_DOMINANT`。
- Seed 44 Binary AUPRC 最大单项来源：diffusion randomness，但只复现约 -0.028，原 -0.0697 含耦合放大。
- Class 9 instability：encoder/SupCon optimization dominant。
- Probe randomness：不是主要根因。

按停止线，本轮不自动实现 class-aware mask、bootstrap-stable mask、backbone 升级或参数搜索。
