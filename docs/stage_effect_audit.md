# C3 Stage Effect 审计

> **STAGE_EFFECT_AUDIT / STAGE_PERTURBATION_BUDGET_MVP / FIXED_R1_BASELINE / EXPLORATORY_TEP_SUBSET / NOT_FOR_PAPER_FINAL_CLAIMS**

## 结论

状态：`STAGE_TIMESTEP_EFFECT_WEAK`。允许进入固定 Budget MVP：`True`。本审计只使用固定 validation、train-only mask、同一噪声 realization 和冻结 Seed 7 R1 encoder；唯一变量为 `t_noncritical∈{3,4,5}`，critical t 始终为 1，phase/DC 保持。

## 分 Stage 时域、频域与表征

| Stage | t | Time L1 | Time L2 | MSE | Noncritical rel-L1 | SNR dB | Critical energy retention | Fisher retention | Repr cosine | Repr L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| normal | 3 | 0.05578 | 0.08670 | 0.002161 | 0.03930 | 26.93 | 1.0088 | 0.9917 | 0.99991 | 0.00741 |
| normal | 4 | 0.05587 | 0.08732 | 0.002192 | 0.03972 | 26.83 | 1.0082 | 0.9927 | 0.99991 | 0.00739 |
| normal | 5 | 0.05594 | 0.08783 | 0.002217 | 0.04004 | 26.76 | 1.0078 | 0.9933 | 0.99991 | 0.00738 |
| early | 3 | 0.05033 | 0.07894 | 0.005108 | 0.03714 | 27.41 | 0.9951 | 0.9917 | 0.99995 | 0.02836 |
| early | 4 | 0.05007 | 0.07876 | 0.005011 | 0.03754 | 27.31 | 0.9954 | 0.9927 | 0.99995 | 0.02787 |
| early | 5 | 0.04986 | 0.07863 | 0.004937 | 0.03784 | 27.24 | 0.9956 | 0.9933 | 0.99995 | 0.02749 |
| middle | 3 | 0.04881 | 0.07676 | 0.005341 | 0.03795 | 27.29 | 0.9965 | 0.9917 | 0.99996 | 0.03595 |
| middle | 4 | 0.04858 | 0.07664 | 0.005247 | 0.03836 | 27.19 | 0.9966 | 0.9927 | 0.99996 | 0.03549 |
| middle | 5 | 0.04840 | 0.07655 | 0.005176 | 0.03866 | 27.12 | 0.9968 | 0.9933 | 0.99996 | 0.03509 |
| stable | 3 | 0.04819 | 0.07569 | 0.005134 | 0.03818 | 27.27 | 0.9937 | 0.9917 | 0.99996 | 0.02764 |
| stable | 4 | 0.04801 | 0.07568 | 0.005057 | 0.03859 | 27.17 | 0.9939 | 0.9927 | 0.99996 | 0.02729 |
| stable | 5 | 0.04787 | 0.07568 | 0.004999 | 0.03890 | 27.10 | 0.9941 | 0.9933 | 0.99996 | 0.02700 |

完整 per-channel variance change、peak change、critical/noncritical/all 的 relative L1/L2、SNR、energy retention 保存在结果 JSON。

## 效应比与单调性

| Stage | Time 4/3 | Time 5/3 | Freq 4/3 | Freq 5/3 | Repr 4/3 | Repr 5/3 | cos(t3,t5) | L2(t3,t5) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| normal | 1.0017 | 1.0029 | 1.0107 | 1.0187 | 0.9977 | 0.9956 | 1.00000 | 0.00028 |
| early | 0.9948 | 0.9907 | 1.0107 | 1.0187 | 0.9830 | 0.9695 | 1.00000 | 0.00147 |
| middle | 0.9953 | 0.9916 | 1.0107 | 1.0187 | 0.9872 | 0.9760 | 1.00000 | 0.00142 |
| stable | 0.9962 | 0.9932 | 1.0107 | 1.0188 | 0.9872 | 0.9767 | 1.00000 | 0.00121 |

决策：`{'median_ratios': {'time': 0.9924073721364428, 'frequency': 1.0187438860514768, 'representation': 0.9763659826035171}, 'strong_layers': {'time': False, 'frequency': False, 'representation': False}, 'strong_layer_count': 0, 'stage_monotonic_all_layers': {'normal': False, 'early': False, 'middle': False, 'stable': False}}`。规则预先固定为四 stage 比值取中位数，时域/非关键频域/表征三层中至少两层达到 1.10 才认定 timestep effect 明确。若 effect weak，仅允许 beta=1.0/0.6/0.8/1.0 的一次 Budget MVP；否则直接停止 C3。

本结果是已多轮查看 TEP test 背景下的 validation 机制审计，不声称统计显著或论文最终无偏结论。
