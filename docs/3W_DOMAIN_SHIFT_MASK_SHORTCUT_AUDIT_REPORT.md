# 3W 跨 WELL Domain Shift 与 Missing-Mask Shortcut 审计

Primary status：`3W_CROSS_WELL_SHIFT_DOMINANT`

Base commit：`ae0d94b`。本阶段只做数据统计和固定 Seed 7 的轻量线性诊断；没有修改当前 8-class Primary config、删除 mask、重跑正式 baseline 或启动 diffusion。

## A. Independent-WELL support

这里按实际监督 target window 统计；normal target 可来自故障 instance 的 pre-onset normal，因此 class 0 覆盖 40 口 WELL。event-directory support 与 target-window support 保持区分。

| class | target WELL | instance | windows | 最大单 WELL 占比 | effective WELL diversity |
|---:|---:|---:|---:|---:|---:|
| 0 | 40 | 738 | 454226 | 0.2511 | 8.47 |
| 1 | 3 | 4 | 2347 | 0.6800 | 1.94 |
| 2 | 7 | 22 | 3449 | 0.6286 | 2.31 |
| 4 | 7 | 343 | 76725 | 0.3261 | 4.80 |
| 5 | 3 | 11 | 10433 | 0.9205 | 1.18 |
| 7 | 6 | 36 | 209401 | 0.3946 | 3.07 |
| 8 | 9 | 14 | 118637 | 0.2368 | 6.16 |
| 9 | 7 | 14 | 6665 | 0.4491 | 3.12 |

class 5 有 10,433 windows，但 92.05% 来自单一 WELL；class 7 有 209,401 windows，却只有 6 口 target WELL、effective diversity 3.07。class 1/2 也高度集中。结论是 `window abundance != independent domain diversity`。

当前 split 中 class 1/5 均只有 train/validation/test 各 1 口 target WELL。class 2 为 3/2/2，class 4 为 4/2/1，class 7 为 4/1/1，class 8 为 6/1/2，class 9 为 4/1/2。无 WELL 泄漏，但对 1/5 的训练域支持不足。

建议支持标准：Strict 要求故障类至少 6 口 target WELL 且 train≥3、val/test≥1；Moderate 要求总数≥3、train/val/test≥1并明确高方差；Lenient 只确认三路非空，不足以支持论文级稳定泛化。

## B. WELL identity 与 fault shortcut

WELL-ID 使用 30 个至少有两个独立 instance 的 WELL，按 instance 严格拆分 train/test；chance=0.0333。同一 window 不重复出现。

| representation | WELL-ID Accuracy | WELL-ID Macro-F1 | Fault Macro-F1 | Fault Macro Recall |
|---|---:|---:|---:|---:|
| raw process | 0.9660 | 0.8342 | — | — |
| preprocessed process | 0.9842 | 0.9002 | 0.2311 | 0.3372 |
| mask only | 0.8210 | 0.6353 | 0.1156 | 0.2528 |
| process + mask | 0.9865 | 0.9148 | 0.1898 | 0.2043 |

process-only WELL-ID 是 chance 的 29.5 倍，train-only clipping/imputation/scaling 没有移除 WELL identity。mask-only 是 chance 的 24.6 倍，判定 `MISSING_MASK_WELL_SHORTCUT_PRESENT`。但 mask-only fault Macro-F1 低于八类 chance 0.125，且远低于 process-only，当前证据不支持独立的 `MISSING_MASK_LABEL_SHORTCUT_PRESENT`。加入 mask 没有改善轻量 fault 分类，反而降低 Macro-F1。

mask signature 的 same-WELL cosine similarity=0.9997；same-class-different-WELL=0.4941；different-class=0.5314。mask pattern 明显更像 WELL identity，而不是 fault identity。

## C. Cross-WELL shift

预处理空间中，平均同类跨 WELL / 异类欧氏距离比为 0.890。class 1=1.032、class 8=1.298，已超过异类距离；class 5=0.974、class 7=0.972，也几乎相等。cosine ratio 对 class 1/4/7 分别为 1.008/1.022/1.040。故障 representation 强烈依赖 WELL。

raw space 受官方数据中的极端哨兵值强烈影响，距离尺度不稳定；但 raw process WELL-ID 仍达 96.60%。机器可读 CSV 同时保留 raw（仅 train-median imputation）和 frozen-preprocessed 两个空间，正式结论以数值稳定的预处理空间及 classifier 一致性为主。

最 WELL-specific 的前 10 个预处理 process feature（ICC-like=`Var(well means)/(Var(well means)+mean within-well variance)`）：

| rank | feature | ICC-like |
|---:|---|---:|
| 1 | P-JUS-CKP | 0.9985 |
| 2 | ESTADO-DHSV | 0.9539 |
| 3 | ABER-CKGL | 0.8993 |
| 4 | T-MON-CKP | 0.8709 |
| 5 | P-ANULAR | 0.8672 |
| 6 | T-JUS-CKP | 0.8504 |
| 7 | QGL | 0.8318 |
| 8 | T-TPT | 0.8275 |
| 9 | ESTADO-M2 | 0.8206 |
| 10 | P-TPT | 0.8153 |

## D. Leave-One-WELL-out mini audit

使用 process summary 与 nearest class centroid，仅审计支持较好的 class 2/4/8；每次从所有训练 centroid 排除 held-out WELL。

| class | held-out WELL 数 | mean accuracy | std | worst |
|---:|---:|---:|---:|---:|
| 2 | 7 | 0.5429 | 0.3332 | 0.0000 |
| 4 | 7 | 0.6553 | 0.4362 | 0.0000 |
| 8 | 9 | 0.6697 | 0.2569 | 0.2000 |

总体 mean=0.6267、std=0.3473、worst=0，跨 WELL 方差很大；即使 class 4 有大量 windows，WELL-00010/00014 的 held-out accuracy 仍接近零。

## E. Protocol reassessment

逐 class proposal：0 KEEP；1 SECONDARY；2 KEEP（高集中风险）；4 KEEP；5 SECONDARY；7 KEEP（高集中风险）；8 KEEP；9 KEEP。没有直接修改配置。

- Protocol A — Strict Real-only：`0,2,4,7,8,9`；移除只有一口 training WELL 的 1/5，强调跨 WELL 主评测。
- Protocol B — Extended Real-only：保留当前 `0,1,2,4,5,7,8,9`，但 1/5 只能作为高方差、secondary-grade evidence，不能用 window 数宣称充足。
- Protocol C — Real + Simulated Secondary：real test 仍是唯一主测试；real Primary 建议 `0,2,4,7,8,9`；仅在 secondary train 中考虑有官方 simulated 的 `1,2,5,8,9`，DRAWN 不进 Primary，simulated identity 必须隔离且不得参与 real test/preprocessing 拟合。

## 结论与下一步

主状态为 `3W_CROSS_WELL_SHIFT_DOMINANT`。Secondary findings：mask 存在明显 WELL shortcut；当前没有明显 mask label shortcut；class 1/5 independent support 不足；class 2/7 window concentration 仍高。

下一阶段最小实验只能是冻结 proposal 后的 controlled `with-mask vs no-mask` Clean baseline，以及仅基于 train/validation 的 WELL-centered/domain-normalization 诊断；在协议选择前不得删除 mask、改变 Primary classes、启动 diffusion 或 3-Seed。
