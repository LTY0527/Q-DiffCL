# 3W Cross-WELL-aware SupCon 验证实验

最终判定：`CROSS_WELL_SUPCON_NO_GO`

本轮只改变 SupCon batch construction 是否显式考虑 training WELL。Final Primary `[0, 2, 8, 9]`、canonical grouped split 00、Real WELL-only、process-only、window/stride `64/32`、train-only preprocessing、TCN、Hard SupCon 损失、balanced frozen probe、Uniform `t=3`、R1 `t_key=1/t_nonkey=5`、protocol seed 42、固定 train/validation refs、critical soft mask 与 validation augmentation 均未改变。没有使用 validation/test 调 sampler，也未搜索 TCN、频率 mask 或扩散参数。

## 实验范围与公平性

正式比较为 3 种 batching × 3 seeds × Uniform/R1，共 18 个结果。Original 6 个和 Balanced Positive-safe 6 个结果直接复用；本轮只新训练 Cross-WELL 6 个结果。每个 seed 内 Uniform/R1 使用完全相同的初始化和 Cross-WELL batch order。

- Window refs SHA256：`4c1f14234133bb44130cf587a2dbd6330c7c40a7ef544185951e32019545e9a3`
- Critical mask SHA256：`0832936bb5da6145eacaa3fa7ca490d6fa69eb72ced12b1614d23f16d2829939`
- Cross-WELL batch-order SHA256（42/43/44）：`8d2012c4...c7b1` / `5ed72bfa...b8a9` / `bf980c5f...d8c6`
- 所有 Cross-WELL train/validation SupCon loss finite；所有类 positive-anchor rate 为 1.0。

## Cross-WELL sampler

保持上一轮的 `P=4, K=64, batch size=256, 23 batches/epoch, 20 epochs, max class oversampling=3.0`。每个 batch 的四类各有 64 个窗口。内部 target 0/1/2/3 分别对应原始类 0/2/8/9。

对每个 class，sampler 只读取 train label 和 train `well_id`：先覆盖当前仍有未使用窗口的不同 WELL，再按各 WELL 剩余窗口数的平方根分配剩余名额；井内队列在耗尽前不重复，只有全类池耗尽后才开始下一轮。该设计避免让窗口极少的 WELL 被逐 batch 强制重复，同时仍提高跨井正对比例。每个 batch、epoch、class、WELL 的计数和比例都写入 JSON。

| 原始类 | Train windows | Train WELLs | 每 epoch 采样 | 最大逐井复用因子 | Cross paired-view ratio |
|---:|---:|---:|---:|---:|---:|
| 0 | 4000 | 20 | 1472 | 1.0 | .86721 |
| 2 | 495 | 3 | 1472 | 3.0 | .65498 |
| 8 | 4000 | 5 | 1472 | <1.0 | .77176 |
| 9 | 3660 | 3 | 1472 | <1.0 | .51712 |

四类都具备真实跨井支持，不存在静默回退。任一 batch 中多井类别最少覆盖 3 口 WELL，batch 内 exact-duplicate window rate 为 0。逐 epoch、逐 WELL 的完整采样次数和 oversampling factor 位于 `3w_cross_well_supcon.json` 的 `sampler` 字段。

## 跨井正对比例

比例按实际 Hard SupCon paired-view positives 统计，即 clean/restored 两个 view 展开后，同类且来自不同 WELL 的 positive 数占全部同类 positive 数。

| Batching | Overall paired-view cross-WELL ratio | Class 0 | Class 2 | Class 8 | Class 9 |
|---|---:|---:|---:|---:|---:|
| Original | .61575 | .75715 | .62499 | .72898 | .31180 |
| Balanced | .61046 | .75585 | .64734 | .72670 | .31096 |
| Cross-WELL | .70277 | .86721 | .65498 | .77176 | .51712 |

Cross-WELL 相对最高基线提高 `.08702`，超过预注册 `.05` gate；Class 9 比例也从约 `.311` 提高到 `.517`。因此后续 NO_GO 不能归因于 sampler 没有实际提高跨井 positives。

## 6 个新增正式结果

| Seed | Method | Macro-F1 | Macro Recall | Binary AUPRC | Multi AUPRC | FAR | Early Recall | Delay (s) | C9 Recall | C9 F1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | Uniform | .47998 | .44108 | .89807 | .58130 | .34740 | .82291 | 773.05 | .00276 | .00157 |
| 42 | R1 | .48278 | .44412 | .89697 | .57208 | .33521 | .82206 | 779.14 | .00276 | .00170 |
| 43 | Uniform | .56251 | .68858 | .87585 | .71445 | .55591 | .98180 | 38.79 | .99724 | .20128 |
| 43 | R1 | .56321 | .68936 | .87589 | .71453 | .55591 | .98180 | 38.79 | .99724 | .20128 |
| 44 | Uniform | .50172 | .44980 | .83798 | .47058 | .31973 | .83481 | 293.59 | .00276 | .00107 |
| 44 | R1 | .50258 | .45119 | .86230 | .47466 | .32332 | .84111 | 573.00 | .00276 | .00110 |

各类 0/2/8/9 的完整 Recall/F1、初始化 hash 与公平性 hash 位于 results CSV。

## Paired R1 − Uniform

| Seed | Macro-F1 | Binary AUPRC | Multi AUPRC | FAR | Early Recall | Delay (s) | C9 Recall | C9 F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | +.00281 | -.00111 | -.00922 | -.01220 | -.00084 | +6.10 | .00000 | +.00013 |
| 43 | +.00071 | +.00004 | +.00008 | .00000 | .00000 | .00 | .00000 | .00000 |
| 44 | +.00086 | +.02432 | +.00408 | +.00358 | +.00630 | +279.41 | .00000 | +.00003 |

Cross-WELL 的 paired Macro-F1 为 `+.00146 ± .00096`，3/3 为正，方向稳定性相对 Balanced（`-.00865 ± .01274`，2/3）恢复；但增益量级远低于 Original 的 `+.03258 ± .00617`。Binary AUPRC 三个 seed 均未出现超过 `.03` 的负下降，上一轮从 seed 44 转移到 seed 43 的异常已消失。FAR paired mean 为 `-.00287 ± .00676`，总体没有明显恶化。Seed 44 delay 单次增加 279.41 秒，虽未使三 seed R1 delay mean 超出预注册保持 gate，仍应保留为风险记录。

## Class 9 稳定性

| R1 Class 9 | Original mean ± std | Balanced mean ± std | Cross-WELL mean ± std |
|---|---:|---:|---:|
| Recall | .17103 ± .23312 | .33747 ± .46849 | .33425 ± .46880 |
| F1 | .04141 ± .04824 | .13301 ± .18593 | .06802 ± .09423 |

Cross-WELL 的 Class 9 F1 std 比 Balanced 降低，但仍约为 Original 的 1.95 倍；Recall std 与 Balanced 基本相同，约为 Original 的 2.01 倍。相对 Original 的 Recall/F1 std reduction 分别为 `-1.01101/-0.95315`，均未达到至少下降 20% 的 gate。Seed 43 Class 9 Recall 接近 1，而 seed 42/44 接近 0，核心 seed instability 完全保留。

## R1 三 seed 总体保持性

| R1 指标 | Original mean ± std | Balanced mean ± std | Cross-WELL mean ± std |
|---|---:|---:|---:|
| Macro-F1 | .48674 ± .06088 | .50497 ± .03630 | .51619 ± .03422 |
| Binary AUPRC | .71243 ± .17751 | .85174 ± .02852 | .87839 ± .01426 |
| Multiclass AUPRC | .51816 ± .12234 | .62188 ± .07181 | .58709 ± .09850 |
| FAR | .38523 ± .08035 | .45613 ± .03308 | .40481 ± .10695 |
| Early Recall | .84400 ± .10863 | .88478 ± .03429 | .88166 ± .07124 |
| Delay (s) | 503.63 ± 293.50 | 356.55 ± 290.15 | 463.64 ± 311.98 |

Cross-WELL 保持了 Macro-F1、Binary AUPRC、FAR、Early Recall 与 mean delay gate，并消除了 Binary AUPRC 大幅负异常；但 FAR std 和 delay std 仍较大。

## Gate 与结论

通过：跨井 positive ratio 确实提高；R1 Macro-F1 3/3 正增益；所有 seed 的 Binary AUPRC 无大幅负下降；Macro-F1、Binary AUPRC、FAR、Early Recall 与 mean delay 保持；训练 finite；positive pairs 和多 WELL 支持完整。

失败：Class 9 Recall std 未下降；Class 9 F1 std 未下降。跨井 positive ratio 的提升没有同步带来目标类别的跨 seed 稳定性改善。

因此最终判定严格为：

`CROSS_WELL_SUPCON_NO_GO`

Cross-WELL batching 是有效执行的，但它只改善了平均性能、Binary AUPRC 异常和 R1 paired 方向，没有解决本轮核心的 Class 9 seed instability。按照停止线，本轮到此结束，不自动继续调整 sampler、TCN、frequency mask 或进行大规模参数搜索。
