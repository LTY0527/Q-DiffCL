# Paper Evidence Matrix

| Claim | Evidence | Status |
|---|---|---|
| Selective > Uniform | 3-seed validation matched-budget ablation | `SUPPORTED ON 3W; NOT SUPPORTED ON TEP` |
| D/E captures fault semantics | D_ONLY/E_ONLY/FINAL + heatmaps | `SUPPORTED with D-primary/E-complementary wording` |
| Soft allocation matters | Hard vs Soft 3-seed validation ablation | `SUPPORTED ON 3W; NOT SUPPORTED ON TEP` |
| Gain is not from less noise | Soft matched vs unmatched + equal Uniform budget | `SUPPORTED ON 3W; NOT SUPPORTED ON TEP` |
| DCBR mitigates over-augmentation | TEP FINAL/DCBR/SCALING 5-seed development evidence | `SUPPORTED AS DEVELOPMENT EVIDENCE` |
| Cross-WELL benefit | per-WELL replay + bootstrap CI | `PARTIAL; CI crosses zero` |
| Practicality | runtime/memory/parameters table | `PARTIAL; some 3W runtime fields unavailable` |
| Limited-data robustness | grouped dry-run only | `UNSUPPORTED / DO NOT CLAIM` |
| Missingness robustness | TEP MCAR30 only; 3W native missingness | `PARTIAL` |
| Sensitivity 0.2/0.3/0.4 | mask/budget audit only | `UNSUPPORTED PERFORMANCE CLAIM` |
| Generalization | nested grouped paper-final protocol | `PENDING OUTER EVALUATION` |

任何标为 UNSUPPORTED/PENDING 的 claim 不得进入摘要、贡献列表或结论。SVR 保持 `NO_GO_SVR`，不进入最终方法。
