# R2 多类别判别关键频率：TEP Stage B

最终判定：`R2_CROSS_DATASET_PARTIAL_GO`

> **R2_CROSS_DATASET_TRANSFER_CHECK / EXPLORATORY_TEP_SUBSET / NOT_FOR_PAPER_FINAL_CLAIMS**

仅在 3W `R2_3W_GO` 后执行。复用冻结的 TEP C1/R1 6 个结果，只新增 seeds 7/42/2026 的 3 个 R2 run。window/stride `64/16`、MCAR `.30`、binary downstream、fixed views、初始化、pretrain/probe order、Uniform `t=3`、selective `t_critical=1/t_noncritical=5`、phase/DC 与 budget matching 均保持。

M 只使用 train faultNumber 0–20：normal type 0 有 128 个 train run aggregates，fault 1–20 各 6 个；validation/test faultNumber 不参与 mask 拟合。

## Mask audit

R1/R2 mask Jaccard `.78819`，515 个 selected bins 中改变 122 个。

- R1 mask SHA256：`d2e1879b...9395`
- R2 mask SHA256：`950c1502...2ef2`

## 三 seed 结果

| Seed | Method | Macro-F1 | AUPRC | Recall | FAR | Early Recall | Mean Delay |
|---:|---|---:|---:|---:|---:|---:|---:|
| 7 | C1 | .88947 | .93143 | .80213 | .03750 | .79375 | 103.00 |
| 7 | R1 | .89199 | .93164 | .79787 | .02969 | .79375 | 103.00 |
| 7 | R2 | .89197 | .93143 | .79734 | .02930 | .79375 | 103.00 |
| 42 | C1 | .88332 | .93088 | .81064 | .05508 | .78125 | 121.97 |
| 42 | R1 | .88647 | .93152 | .81649 | .05430 | .78750 | 121.97 |
| 42 | R2 | .88672 | .93149 | .81702 | .05430 | .78750 | 121.97 |
| 2026 | C1 | .88100 | .91948 | .76543 | .02227 | .73750 | 126.06 |
| 2026 | R1 | .88098 | .91948 | .76489 | .02188 | .73750 | 126.06 |
| 2026 | R2 | .88123 | .91944 | .76543 | .02188 | .73750 | 126.06 |

## Paired transfer

| Comparison | Macro-F1 mean ± sample std | AUPRC | Recall | FAR | Early Recall | Delay |
|---|---:|---:|---:|---:|---:|---:|
| R2−C1 | +.00204 ± .00163 | +.00019 | +.00053 | −.00313 | +.00208 | .00 |
| R2−R1 | +.00016 ± .00016 | −.00009 | +.00018 | −.00013 | .00000 | .00 |

R2−C1 Macro-F1 3/3 为正，AUPRC/Recall/FAR/Early/Delay 均保持且无灾难 seed；方向与 3W 一致。但 Macro-F1 mean 仅 `+.00204`，低于将“基本持平”升级为跨数据集 GO 的 `.005` 解释阈值；R2−R1 几乎为零。

因此给 `R2_CROSS_DATASET_PARTIAL_GO`：TEP 没有反向破坏，但也没有提供 R2 明显优于 R1 的跨数据集效应。该 TEP 子集已在多轮探索中使用，只能作为 transfer check，不能形成 paper-final claim。
