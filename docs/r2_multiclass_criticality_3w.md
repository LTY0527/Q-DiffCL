# R2 多类别判别关键频率：3W Stage A

最终判定：`R2_3W_GO`

R2 固定为 `0.40D + 0.24E + 0.16S + 0.20M`。M 使用 canonical train split 的原始类别 `[0,2,8,9]`，对应 train run counts `20/3/5/3`；validation/test 不参与拟合。除 criticality composite 外，Original batching、TCN、Hard SupCon、probe、split、window/stride `64/32`、Uniform `t=3`、selective `t_key=1/t_nonkey=5`、phase/DC 与 total noise budget 均冻结。

## 运行与协议审计

正式结果为 Uniform/R1/R2 × seeds 42/43/44。复用 6 个冻结 Uniform/R1，只新增 3 个正式 R2。另有 3 个预跑因 M 未包含提示词明确要求的 normal class 0 而在协议复核时作废；旧 outputs 保留但完全排除。因此 3W 实际执行 6 个新训练，其中正式计入 3 个。

正式 R2 与旧结果逐 seed 的 window refs、初始化和 Original deterministic order 一致。test 未用于权重、mask、timestep、epoch、threshold 之外的训练选择或 gate 调参。

## Mask audit

D/E/S 在 R1 与 R2 间最大绝对差均为 0，M 非零位置 561 个。R1/R2 binary mask Jaccard `.82427`，218 个 selected bins 中有 42 个改变；composite rank Spearman `.98448`，M top-mask 与 R2 mask Jaccard `.63910`。

- R1 mask SHA256：`547d63f2...ee09`
- R2 mask SHA256：`3325c046...92e5`

M 确实改变了关键频率排序与最终 mask。

## 三 seed 结果

| Seed | Method | Macro-F1 | Macro Recall | Binary AUPRC | Multi AUPRC | FAR | Early Recall | Delay (s) | C9 Recall | C9 F1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | Uniform | .47636 | .44832 | .77906 | .51651 | .43875 | .90190 | 965.59 | .00966 | .01292 |
| 42 | R1 | .50084 | .47239 | .78386 | .52436 | .34447 | .89673 | 916.14 | .00966 | .01421 |
| 42 | R2 | .47794 | .44899 | .77594 | .51757 | .43923 | .91009 | 238.32 | .00920 | .01176 |
| 43 | Uniform | .51944 | .45297 | .87810 | .66613 | .47335 | .91493 | 322.05 | .05885 | .01401 |
| 43 | R1 | .55324 | .57110 | .88514 | .66481 | .49747 | .94261 | 337.29 | .50069 | .10919 |
| 43 | R2 | .47182 | .40880 | .88378 | .64471 | .48544 | .85202 | 119.05 | .01011 | .00275 |
| 44 | Uniform | .36669 | .30225 | .53803 | .32323 | .37418 | .73246 | 190.79 | .00276 | .00051 |
| 44 | R1 | .40614 | .35506 | .46830 | .36532 | .31374 | .69268 | 257.46 | .00276 | .00082 |
| 44 | R2 | .50131 | .46184 | .90375 | .52207 | .27518 | .84687 | 225.17 | .00276 | .00138 |

完整 Class 0/2/8/9 Recall/F1 位于 results CSV。

## Paired 结果

| Comparison | Macro-F1 mean ± std | Macro Recall | Binary AUPRC | Multi AUPRC | FAR | Early Recall | Delay (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| R2−Uniform | +.02953 ± .07698 | +.03870 | +.12276 | +.05950 | −.02881 | +.01990 | −298.63 |
| R2−R1 | −.00305 ± .07345 | −.02631 | +.14206 | +.04329 | +.01472 | +.02566 | −309.45 |

R2−Uniform Macro-F1 在 seeds 42/43/44 分别为 `+.00158/−.04762/+.13462`，2/3 为正，均值超过 `+.020` gate，但 seed 方差很大。R2 相对 R1 的 Macro-F1 仅 1/3 改善且均值略低；优势主要集中于 Binary AUPRC、Multiclass AUPRC 和 delay，而非稳定的 Macro-F1 全面领先。

## Class 9

| Method | Recall mean ± std | F1 mean ± std |
|---|---:|---:|
| Uniform | .02375 ± .02498 | .00915 ± .00612 |
| R1 | .17103 ± .23312 | .04141 ± .04824 |
| R2 | .00736 ± .00327 | .00529 ± .00461 |

R2 的 Class 9 std 远低于 R1，因此稳定性 gate 通过；但它是“稳定地低召回”，不是跨 seed 稳定提升，不能当成 Class 9 已解决。

## Gate

全部通过：R2−Uniform Macro-F1 mean `+.02953 ≥ +.020`，2/3 wins；Multiclass AUPRC mean `+.05950 > 0`；Binary AUPRC mean 与单 seed 均保持；FAR mean 未恶化；Class 9 Recall/F1 std 未超过 R1 的 1.10 倍。

最终判定：`R2_3W_GO`

该 GO 允许进入 TEP transfer check，但 3W 的强 seed 异质性和 R2 对 R1 的非全面优势必须保留为论文风险。
