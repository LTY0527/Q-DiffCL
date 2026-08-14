# 3W Balanced Positive-safe SupCon 稳定性修复实验

最终判定：`BALANCED_SUPCON_NO_GO`

本轮只改变 SupCon batch construction。Final Primary `[0, 2, 8, 9]`、canonical grouped split 00、Real WELL-only、process-only、window/stride `64/32`、train-only preprocessing、TCN、Hard SupCon、balanced frozen probe、Uniform `t=3`、R1 `t_key=1/t_nonkey=5`、protocol seed 42、固定 train/validation window refs、固定 critical soft mask 与固定 validation augmentation 均保持不变。未继续调整 sampler、TCN、频率掩码、类别、split、阈值或其他超参数。

## 实验与公平性

正式比较共 12 个结果：3 seeds（42/43/44）× 2 扩散方法（Uniform/R1）× 2 batching（Original/Balanced）。其中 Original 的 6 个结果复用已冻结的 3-Seed manifest，Balanced 的 6 个结果为本轮新训练。每个 seed 内 Uniform 与 R1 使用相同初始化及相同 batch order；所有结果的 window refs 和 critical mask hash 一致。

- Window refs SHA256：`4c1f14234133bb44130cf587a2dbd6330c7c40a7ef544185951e32019545e9a3`
- Critical mask SHA256：`0832936bb5da6145eacaa3fa7ca490d6fa69eb72ced12b1614d23f16d2829939`
- Balanced batch-order SHA256（seed 42/43/44）：`1d3cc613...ed73` / `af823908...82b5` / `0ae045f7...6c0b`
- 所有 Balanced 训练 loss 与 validation SupCon loss 均为 finite。

## Balanced sampler 冻结设置

Sampler 只读取 training labels。原始类 0/2/8/9 在训练集中的 window 数分别为 4000/495/4000/3660；实现内部连续标签 0/1/2/3 分别对应原始类 0/2/8/9。

| 项目 | 冻结值 |
|---|---:|
| classes per batch（P） | 4 |
| samples per class（K） | 64 |
| batch size | 256 |
| batches per epoch | 23 |
| epochs | 20 |
| max oversampling | 3.0 |

| 原始类 | Train windows | 每 epoch 实际采样 | Oversampling factor |
|---:|---:|---:|---:|
| 0 | 4000 | 1472 | 0.368000 |
| 2 | 495 | 1472 | 2.973737 |
| 8 | 4000 | 1472 | 0.368000 |
| 9 | 3660 | 1472 | 0.402186 |

三个 seed 的 20 个 epoch 均满足：每 batch 恰含 4 类、每类 64 个 clean 样本；clean positive-anchor rate 与 paired-view positive-anchor rate 均为 1.0。最初候选的 63 batches/epoch 会令 Class 2 oversampling 达到 8.145×，已在训练前由上限检查拒绝；最终设置是满足 `<3.0` 安全约束的冻结方案，而非依据测试结果调参。

## 12 个正式结果

| Batching | Seed | Method | Macro-F1 | Binary AUPRC | Multiclass AUPRC | FAR | Early Recall | Delay (s) | Class 9 Recall | Class 9 F1 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 42 | Uniform | .47636 | .77906 | .51651 | .43875 | .90190 | 965.59 | .00966 | .01292 |
| Original | 42 | R1 | .50084 | .78386 | .52436 | .34447 | .89673 | 916.14 | .00966 | .01421 |
| Original | 43 | Uniform | .51944 | .87810 | .66613 | .47335 | .91493 | 322.05 | .05885 | .01401 |
| Original | 43 | R1 | .55324 | .88514 | .66481 | .49747 | .94261 | 337.29 | .50069 | .10919 |
| Original | 44 | Uniform | .36669 | .53803 | .32323 | .37418 | .73246 | 190.79 | .00276 | .00051 |
| Original | 44 | R1 | .40614 | .46830 | .36532 | .31374 | .69268 | 257.46 | .00276 | .00082 |
| Balanced | 42 | Uniform | .55610 | .87423 | .70197 | .43320 | .90213 | 750.79 | 1.00000 | .39556 |
| Balanced | 42 | R1 | .55626 | .87434 | .70211 | .43286 | .90210 | 750.79 | 1.00000 | .39596 |
| Balanced | 43 | Uniform | .50417 | .84512 | .66068 | .54588 | .95940 | 37.46 | .67540 | .14282 |
| Balanced | 43 | R1 | .47750 | .81151 | .63569 | .50291 | .91535 | 60.86 | .00966 | .00228 |
| Balanced | 44 | Uniform | .48059 | .86182 | .52447 | .43258 | .83149 | 258.00 | .00276 | .00080 |
| Balanced | 44 | R1 | .48115 | .86935 | .52785 | .43262 | .83689 | 258.00 | .00276 | .00079 |

完整的 Macro Recall 与各类 Recall/F1 保存在 `3w_balanced_supcon_stability_results.csv`。

## Paired R1 − Uniform

| Batching | Seed | Macro-F1 | Binary AUPRC | Multiclass AUPRC | FAR | Early Recall | Delay (s) | Class 9 Recall | Class 9 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 42 | +.02447 | +.00480 | +.00785 | -.09428 | -.00518 | -49.45 | .00000 | +.00129 |
| Original | 43 | +.03380 | +.00704 | -.00131 | +.02412 | +.02768 | +15.24 | +.44184 | +.09519 |
| Original | 44 | +.03945 | -.06973 | +.04209 | -.06044 | -.03978 | +66.67 | .00000 | +.00031 |
| Balanced | 42 | +.00016 | +.00011 | +.00014 | -.00034 | -.00003 | .00 | .00000 | +.00040 |
| Balanced | 43 | -.02667 | -.03361 | -.02499 | -.04297 | -.04406 | +23.41 | -.66575 | -.14054 |
| Balanced | 44 | +.00056 | +.00753 | +.00338 | +.00003 | +.00540 | .00 | .00000 | -.00001 |

Original batching 下 R1 的 Macro-F1 为 `+.03258 ± .00617`，3/3 seeds 改善；Balanced 下变为 `-.00865 ± .01274`，形式上 2/3 seeds 为正，但 seed 42/44 的增益都接近零，seed 43 明显下降。Seed 44 的 Binary AUPRC 异常从 `-.06973` 消失并变为 `+.00753`，但不稳定性转移到了 seed 43（`-.03361`），因此不能视为稳定修复。

## Class 9 稳定性与总体保持性

| R1 指标 | Original mean ± std | Balanced mean ± std | 结论 |
|---|---:|---:|---|
| Class 9 Recall | .17103 ± .23312 | .33747 ± .46849 | std 约翻倍 |
| Class 9 F1 | .04141 ± .04824 | .13301 ± .18593 | std 约增至 3.85 倍 |
| Macro-F1 | .48674 ± .06088 | .50497 ± .03630 | 均值保持 |
| Binary AUPRC | .71243 ± .17751 | .85174 ± .02852 | 均值保持 |
| FAR | .38523 ± .08035 | .45613 ± .03308 | 均值恶化 +.07090 |
| Early Recall | .84400 ± .10863 | .88478 ± .03429 | 均值保持 |
| Delay (s) | 503.63 ± 293.50 | 356.55 ± 290.15 | 均值保持 |

Class 9 Recall/F1 的预注册 std reduction 分别为 `-1.00965` 与 `-2.85406`，均未达到至少下降 20% 的 gate。Balanced R1 的 FAR 均值比 Original R1 高 `.07090`，超过允许的 `.05` 增幅。虽然 Binary AUPRC 均值和 seed 44 异常得到改善，但这不足以抵消 Class 9 跨 seed 波动与 FAR 的明确恶化。

## Gate 与结论

通过：paired Macro-F1 多数 seed 改善、seed 44 Binary AUPRC 异常消失、Macro-F1/Binary AUPRC/Early Recall/Delay 均值保持、训练 finite、所有类保留 positive pairs。

失败：Class 9 Recall std 未下降、Class 9 F1 std 未下降、FAR 均值未保持。

因此最终判定严格保持为：

`BALANCED_SUPCON_NO_GO`

Balanced Positive-safe batching 保证了对比正样本的结构条件，也改善了部分平均指标，但没有修复目标中的 Encoder/SupCon seed instability。按照停止线，本轮到此结束：不自动调整 sampler，不继续参数搜索，不修改 TCN，也不自动进入 class-aware frequency mask 或其他新路线。
