# 同样本候选选择下游复测报告

> **INTRA_SAMPLE_CANDIDATE_AUDIT / SINGLE_SEED / FIXED_CHECKPOINT / NOT_FOR_PAPER_CLAIMS**

## 状态

```text
第一级：INTRA_SAMPLE_CANDIDATE_RANKING_NO_GO
第二级：SKIPPED_BY_RANKABILITY_GATE
training_skipped = true
```

Oracle Best-of-5 相对随机候选的 train masked MAE 改善只有 3.82%，未达到 5% 门槛。按照预注册流程，本轮没有生成完整数据的 K=3 候选，也没有执行任何新的 SupCon 或 linear probe 训练。

## 计划对比表

| 方法 | Macro-F1 | AUPRC | Fault Recall | FAR | Teacher consistency | 时间 |
|---|---:|---:|---:|---:|---:|---:|
| R0 Fixed Single | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 | 0 |
| R1 Random Candidate | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 | 0 |
| R2 Oracle Top-1 | 审计上界不足 | 审计上界不足 | 审计上界不足 | 审计上界不足 | 审计上界不足 | 0 |
| R3 No-reference Top-1 | 未获准运行 | 未获准运行 | 未获准运行 | 未获准运行 | 未获准运行 | 0 |
| R4 Intra-sample Soft | 前置 Top-1 未运行 | 前置 Top-1 未运行 | 前置 Top-1 未运行 | 前置 Top-1 未运行 | 前置 Top-1 未运行 | 0 |

“未运行”是门控的预期输出，不是实验缺失。`run_candidate_selection_retest.py` 在加载完整候选或构建训练模型之前读取第一级结果，并返回 `training_skipped=true`。

## 公平设计（未触发）

若第一级通过，R0–R3 将共享 fixed split/mask/checkpoint、初始化、batch order、Adam、学习率、8 个 epoch、temperature、projection head、linear probe、验证阈值与 test set。所有组的样本级总权重固定为 1，唯一差异为每个样本内部选择哪个候选。相关逻辑和测试已实现，但本轮未执行。

## 结论

H1 的排序信号真实存在，但 Oracle 候选池本身的收益不足，且 Oracle 恢复仍明显落后 simple interpolation。没有必要为约 3.8% 的恢复 MAE 上界投入完整 K=3 候选生成和四组下游训练。

最终保持：

```text
INTRA_SAMPLE_CANDIDATE_RANKING_NO_GO
```

唯一下一步建议：暂停候选排序/选择路线，优先解决 diffusion 单候选恢复质量落后 simple interpolation 的问题。
