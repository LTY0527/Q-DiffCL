# Paper-final Outer Failure Analysis

低结果先经 split、fit scope、hash、checkpoint 与 resume 审计；没有协议错误的低值作为真实 generalization evidence 保留。

## Lowest cells

| Dataset | Outer | Seed | Method | Macro-F1 |
|---|---:|---:|---|---:|
| 3W | 31002 | 43 | DCBR | 0.0977 |
| 3W | 31002 | 44 | DCBR | 0.1083 |
| 3W | 31002 | 43 | NO_AUG | 0.1091 |
| 3W | 31002 | 42 | FINAL_QDIFFCL | 0.1256 |
| 3W | 31002 | 44 | UNIFORM_DIFFUSION | 0.1285 |
| 3W | 31002 | 42 | JITTER | 0.1297 |
| 3W | 31002 | 43 | JITTER | 0.1319 |
| 3W | 31002 | 44 | JITTER | 0.1358 |
| 3W | 31002 | 42 | UNIFORM_DIFFUSION | 0.1360 |
| 3W | 31002 | 42 | FRERA | 0.1402 |
| 3W | 31002 | 46 | NO_AUG | 0.1434 |
| 3W | 31002 | 44 | SCALING | 0.1461 |

## Lowest groups

| Dataset | Method | Group | Macro-F1 | Fault |
|---|---|---|---:|---:|
| 3W | DCBR | WELL-00020 | 0.0000 | mixed |
| 3W | FINAL_QDIFFCL | WELL-00020 | 0.0000 | mixed |
| 3W | FRERA | WELL-00020 | 0.0000 | mixed |
| 3W | JITTER | WELL-00020 | 0.0000 | mixed |
| 3W | JITTER_SCALING | WELL-00020 | 0.0000 | mixed |
| 3W | NO_AUG | WELL-00020 | 0.0000 | mixed |
| 3W | SCALING | WELL-00020 | 0.0000 | mixed |
| 3W | UNIFORM_DIFFUSION | WELL-00020 | 0.0000 | mixed |
| 3W | DCBR | WELL-00020 | 0.0000 | mixed |
| 3W | FINAL_QDIFFCL | WELL-00020 | 0.0000 | mixed |
| 3W | FRERA | WELL-00020 | 0.0000 | mixed |
| 3W | JITTER | WELL-00020 | 0.0000 | mixed |
| 3W | JITTER_SCALING | WELL-00020 | 0.0000 | mixed |
| 3W | NO_AUG | WELL-00020 | 0.0000 | mixed |
| 3W | SCALING | WELL-00020 | 0.0000 | mixed |
| 3W | UNIFORM_DIFFUSION | WELL-00020 | 0.0000 | mixed |
| 3W | DCBR | WELL-00020 | 0.0000 | mixed |
| 3W | FINAL_QDIFFCL | WELL-00016 | 0.0000 | mixed |
| 3W | FINAL_QDIFFCL | WELL-00020 | 0.0000 | mixed |
| 3W | FRERA | WELL-00020 | 0.0000 | mixed |

这些结果可能反映 domain shift、hard WELL/fault、seed instability 或过增强；failure analysis 不产生新方法版本。
