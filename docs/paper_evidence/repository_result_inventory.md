# Repository / Result Inventory

本清单先于任何补充实验生成。旧 locked test 仅作为 development evidence，不用于新方法或参数选择。

| Evidence | Available | Action |
|---|---|---|
| `A1_uniform_and_soft_final` | `True` | 复用并核验公平性 |
| `A1_hard_mask_selective` | `False` | 初始缺失；本阶段已按 validation-only 公平协议补齐 |
| `A1_soft_without_budget_match` | `False` | 初始缺失；本阶段已按 validation-only 公平协议补齐 |
| `A2_semantic_components` | `True` | 复用并核验公平性 |
| `A3_dcbr_domain_calibration` | `True` | 复用并核验公平性 |
| `B1_3w_checkpoints_for_group_replay` | `True` | 复用并核验公平性 |
| `B2_tep_checkpoints_for_fault_replay` | `True` | 复用并核验公平性 |
| `B3_five_seed_metrics` | `True` | 复用并核验公平性 |
| `D1_limited_data` | `False` | 初始缺失；不得声称已支持 |
| `D2_missingness_sensitivity` | `False` | 初始缺失；不得声称已支持 |
| `D3_critical_ratio_sensitivity` | `False` | 初始缺失；不得声称已支持 |
| `D4_efficiency` | `True` | 复用并核验公平性 |

## External baseline coverage

- 3W: `['FINAL_QDIFFCL', 'FRERA', 'JITTER', 'JITTER_SCALING', 'NO_AUG', 'SCALING', 'UNIFORM_DIFFUSION']`
- TEP: `['FINAL_QDIFFCL', 'FRERA', 'JITTER', 'JITTER_SCALING', 'NO_AUG', 'SCALING', 'UNIFORM_DIFFUSION']`

当前主表已覆盖 NoAug、Jitter、Scaling、Jitter+Scaling、Uniform Diffusion、FreRA、FINAL_QDIFFCL/DCBR。自动增强与工业 diffusion-native baseline 尚无公平 shared-backbone 适配，标记为 supplementary coverage gap，不为刷榜强行加入。
