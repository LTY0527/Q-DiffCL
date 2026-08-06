# 教师可靠性与扰动稳定性审计

> **TEACHER_AND_SEMANTIC_LOSS_AUDIT / GENERATOR_ONLY / THREE_SEEDS / NOT_FOR_PAPER_CLAIMS**

## 结论

最终状态：`TEACHER_NOT_RELIABLE_FOR_SEMANTIC_GUIDANCE`。

本审计没有重新训练教师。基础性能同时记录 clean 与生成器实际使用的插值基础视图；轻度扰动只施加在 validation，test 未用于门槛选择或扰动调参。

## 基础性能

| Split / 输入 | Accuracy | Macro-F1 | AUPRC | AUROC | Fault Recall | FAR | Effective Rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation / clean | 0.9360 | 0.9356 | 0.9890 | 0.9861 | 0.9365 | 0.0647 | 1.0119 |
| validation / guidance input | 0.8874 | 0.8863 | 0.9747 | 0.9643 | 0.9173 | 0.1473 | 1.0115 |
| test / clean | 0.9002 | 0.8959 | 0.9335 | 0.9160 | 0.8218 | 0.0422 | 1.0097 |
| test / guidance input | 0.8511 | 0.8477 | 0.9244 | 0.9084 | 0.8282 | 0.1320 | 1.0094 |

validation confusion matrix：`[[764, 132], [86, 954]]`；test confusion matrix：`[[2222, 338], [323, 1557]]`。

test 早期故障 recall：0.8305555555555556（n=360）。预测置信度、signed/absolute logit margin 的完整分布已写入机器可读结果 `outputs/teacher_semantic_loss_audit/teacher_reliability/result.json`。

## Validation 轻扰动稳定性

| 扰动 | Consistency | N→F | F→N | JS mean | Margin change mean | Embedding cosine mean |
|---|---:|---:|---:|---:|---:|---:|
| jitter | 0.9954 | 0.0036 | 0.0010 | 0.000037 | 0.054657 | 0.999989 |
| scaling | 0.9954 | 0.0031 | 0.0015 | 0.000034 | 0.205152 | 0.999989 |
| light_masking | 0.9866 | 0.0103 | 0.0031 | 0.000602 | 0.234930 | 0.999815 |
| interpolation_perturbation | 0.9964 | 0.0015 | 0.0021 | 0.000082 | 0.092006 | 0.999974 |

N→F/F→N 是相对基础预测的真实方向翻转率。normal、fault 及各 fault type 的分组 consistency 已保存在结果 JSON 中。

## Test 的 20 个故障类型 Recall

| Fault type | Count | Recall |
|---:|---:|---:|
| 1 | 94 | 1.0000 |
| 2 | 94 | 1.0000 |
| 3 | 94 | 0.1489 |
| 4 | 94 | 1.0000 |
| 5 | 94 | 1.0000 |
| 6 | 94 | 1.0000 |
| 7 | 94 | 1.0000 |
| 8 | 94 | 1.0000 |
| 9 | 94 | 0.0745 |
| 10 | 94 | 0.6489 |
| 11 | 94 | 0.9894 |
| 12 | 94 | 1.0000 |
| 13 | 94 | 1.0000 |
| 14 | 94 | 1.0000 |
| 15 | 94 | 0.1383 |
| 16 | 94 | 0.7021 |
| 17 | 94 | 0.9894 |
| 18 | 94 | 0.9787 |
| 19 | 94 | 0.8936 |
| 20 | 94 | 1.0000 |

## Gate

- 通过：`all_perturbation_consistency_at_least_095`
- 通过：`directional_flips_not_abnormal`
- 通过：`normal_fault_directions_consistent`
- 通过：`major_fault_types_stable`
- 通过：`base_macro_f1_acceptable`
- 通过：`base_auprc_acceptable`
- 通过：`base_auroc_acceptable`
- **失败**：`embedding_effective_rank_acceptable`
- 通过：`fault_type_recall_not_broadly_low`

教师 gate 必须全部通过。若状态为 `TEACHER_NOT_RELIABLE_FOR_SEMANTIC_GUIDANCE`，则按提示词停止 S0–S4 语义损失训练并禁止下游 SupCon。

本次唯一 gate 失败项是 embedding effective rank：validation/test 的 guidance input 分别约为 1.0115/1.0094，低于预先固定的最低 2.0。虽然“低于 0.5 recall 的 fault type 占比不超过 25%”这一宽泛条件通过，test 的 fault 3、9、15 recall 仍分别只有 0.1489、0.0745、0.1383，不应把扰动一致性误解为覆盖所有故障语义。

> 本报告仅为固定 TEP 子集的工程审计：**NOT_FOR_PAPER_CLAIMS**。
