# 语义损失单项消融门控记录

> **TEACHER_AND_SEMANTIC_LOSS_AUDIT / GENERATOR_ONLY / THREE_SEEDS / NOT_FOR_PAPER_CLAIMS**

## 结论

本阶段的前置教师 gate 状态为：

```text
TEACHER_NOT_RELIABLE_FOR_SEMANTIC_GUIDANCE
```

因此 S0、S1 JS、S2 Margin、S3 Feature 和 S4 Balanced Best 均为：

```text
SKIPPED_BY_TEACHER_GATE
```

这不是 `TEACHER_SEMANTIC_GUIDANCE_NO_GO`：后者要求真实运行 S1/S2/S3 后确认三个单项全部失败。本次在任何生成器训练开始前就被教师可靠性 gate 阻止，不能伪造或外推单项损失结论。

## 前置 gate 证据

- validation guidance input：Macro-F1 0.8863、AUPRC 0.9747、AUROC 0.9643。
- test guidance input：Macro-F1 0.8477、AUPRC 0.9244、AUROC 0.9084。
- 四种 validation 轻扰动 prediction consistency 为 0.9866～0.9964，方向翻转率均低于预设上限。
- validation/test guidance embedding effective rank 分别只有 1.0115/1.0094，低于运行前固定的最低 2.0。
- test 的 fault 3、9、15 recall 分别为 0.1489、0.0745、0.1383；教师预测稳定不等于覆盖所有故障语义。

完整基础性能、混淆矩阵、20 个 fault type recall、早期故障 recall、置信度/logit margin 分布和分组扰动结果见 `docs/teacher_reliability_audit.md` 及未纳入 Git 的机器可读结果 `outputs/teacher_semantic_loss_audit/teacher_reliability/result.json`。

## 实验执行矩阵

| 方法 | 是否训练 | Seed | Lambda | Generator 指标 | 原因 |
|---|---|---|---|---|---|
| S0 无语义损失 | 否 | 未运行 | 0 | 无 | 第二阶段只在教师通过后执行 |
| S1 JS | 否 | 未运行 | 未选择 | 无 | 教师 gate 失败 |
| S2 Margin | 否 | 未运行 | 未选择 | 无 | 教师 gate 失败 |
| S3 Feature | 否 | 未运行 | 未选择 | 无 | 教师 gate 失败 |
| S4 Balanced Best | 否 | 未运行 | 未选择 | 无 | S1/S2/S3 未运行且无通过项 |

因此没有可报告的 mean ± std、相对 S0 差值、gradient norm、best/last、EMA/raw、耗时或显存。表中“无”表示未产生数据，不表示零。

## 安全与选择审计

- 未重新训练或修改 teacher checkpoint。
- 未修改 generator 结构、split、mask、四个 RData 或既有 outputs。
- 未查看 test 来选择扰动、lambda、loss 类型或 generator checkpoint。
- 未运行 generator-only S0–S4、下游 SupCon 或正式论文实验。
- 未产生新的 generator checkpoint、NPZ 或大型 Git 文件。
- `BALANCED_SEMANTIC_LOSS_READY_FOR_DOWNSTREAM` 与 `BALANCED_SEMANTIC_LOSS_NO_GO` 均不适用，因为 S4 没有运行。

## 路线判定

当前证据只支持停止“使用该 checkpoint 作为连续 embedding 语义指导”的路线。若未来继续，应先取得具有足够故障类型覆盖和非退化 embedding 的教师，并重新通过同一教师可靠性 gate；在此之前不应继续调 JS、Margin、Feature 或 lambda。

> 本记录仅用于固定 TEP 子集的工程决策：**NOT_FOR_PAPER_CLAIMS**。
