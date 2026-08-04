# 同样本多扩散候选排序审计

> **INTRA_SAMPLE_CANDIDATE_AUDIT / SINGLE_SEED / FIXED_CHECKPOINT / NOT_FOR_PAPER_CLAIMS**

## 结论

第一级闸门状态为 **`INTRA_SAMPLE_CANDIDATE_RANKING_NO_GO`**。

同一样本的 K=5 扩散候选具有可测差异，H1 候选中心一致性也能稳定优于随机排序；但 Oracle Best-of-5 相对随机候选的 masked MAE 改善仅为 3.82%，低于预先固定的 5% 工程门槛。候选选择的理论上界不足，因此没有生成完整 K=3 候选，也没有进入 SupCon 下游复测。

## 研究动机

此前跨样本 Q0/Q1/Q2 会把 fault 难度编码为低质量并降低整个 fault 样本的权重。本轮不再比较不同样本的全局 q，而是在同一个 degraded window 的 K 个候选内部排序。这样固定了样本、故障状态、mask 和输入难度，每个原始样本后续总权重仍为 1。

## 固定资产

- checkpoint：`outputs/diffusion_debug/small_subset/best_diffusion.pt`
- checkpoint SHA-256：`74ae41ca8bf45fc284557be7fa6c0859caf0e91a8ab9e642cdf5f75eeae9a22c`
- 固定 views manifest：`outputs/fixed_diffusion_views/views_manifest.json`
- 冻结教师：`outputs/rapid_idea_validation/G1_0.pt`
- master seed：7
- deterministic MCAR：30%，normalized space
- 50-step cosine DDPM
- 原窗口长度 64、stride 16、原 scaler 与 Run split 不变

三个审计 split 的 run_uid 继续两两互斥，checkpoint、既有 fixed views、mask 和 split manifest 均未修改。

## 审计子集与候选生成

| split | windows | normal/fault | K | NPZ SHA-256 | 生成秒数 |
|---|---:|---:|---:|---|---:|
| train | 512 | 256/256 | 5 | `fbeaef36…aaa595b` | 5.86 |
| validation | 256 | 128/128 | 5 | `4924f97…5a59079` | 2.65 |
| test | 512 | 256/256 | 5 | `7f801cf1…5fac2d` | 5.36 |

train/validation/test 的 fault 窗口均覆盖 fault type 1–20。test 的 normal 类还包含 fault run 中 onset 前的正常窗口，因此按 run_uid 解析的 type 统计与二分类 label 统计不完全相同，这是协议预期行为。

每个候选 seed 按以下字段做 SHA-256 确定性派生：

```text
master_seed | split | window_id | candidate_id | ddpm_candidate
```

同一窗口的五个候选共享 degraded window、mask_id、checkpoint、clip range 和 schedule，只改变 candidate seed。候选 NPZ 位于 `outputs/intra_sample_candidates/audit/`，受 `.gitignore` 保护；可追溯小型 manifest 提交在 `configs/intra_sample_candidate_audit_manifest.json`。manifest 包含每个 window 的 split、run_uid、window_id、mask_id、candidate_id 和 seed。

候选生成总计算时间 14.32 秒，峰值显存 259.10 MiB；包含 Python 启动和资产加载的命令 wall time 约 27 秒。

## Oracle 候选多样性与上界

| split/组 | candidate MAE | 同样本 std | 同样本 range | Oracle MAE | Random MAE | Oracle 改善 | Fixed MAE | 改善 vs Fixed | Simple MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train overall | 0.38916 | 0.01072 | 0.02973 | 0.37430 | 0.38916 | 3.82% | 0.38787 | 3.50% | 0.31726 |
| train normal | 0.33558 | 0.00988 | 0.02733 | 0.32203 | 0.33558 | 4.04% | 0.33460 | 3.76% | 0.31279 |
| train fault | 0.44274 | 0.01156 | 0.03212 | 0.42658 | 0.44274 | 3.65% | 0.44114 | 3.30% | 0.32173 |
| validation overall | 0.38930 | 0.01130 | 0.03140 | 0.37380 | 0.38930 | 3.98% | 0.38854 | 3.79% | 0.31152 |
| test overall | 0.42159 | 0.01136 | 0.03142 | 0.40605 | 0.42159 | 3.69% | 0.42012 | 3.35% | 0.31033 |
| test normal | 0.33632 | 0.00965 | 0.02686 | 0.32322 | 0.33632 | 3.89% | 0.33497 | 3.51% | 0.31554 |
| test fault | 0.50687 | 0.01307 | 0.03597 | 0.48889 | 0.50687 | 3.55% | 0.50527 | 3.24% | 0.30512 |

候选差异非零，且 Oracle 在 normal/fault、三个 split 中方向一致。然而改善稳定停留在约 3.5%–4.0%，没有达到 5%。更关键的是 Oracle 仍比 simple interpolation 差：train overall 差 0.05704，test overall 差 0.09572；候选选择无法弥补当前 diffusion 恢复器与 simple baseline 的主体差距。

Oracle 的 train teacher distance 为 0.10278，随机候选为 0.11267；teacher consistency 为 0.90430 vs 0.88750。Oracle 选择没有以语义退化换取 MAE 改善。

Oracle 候选的其他 train 指标：masked RMSE 0.57785、一阶差分误差 0.18784、correlation error 0.11344。

## 无参考排序

H1/H2 的接口不接收 clean 或 label；Oracle MAE 在独立函数中计算。H3 仅在 H1 初步信号出现后启用，且所有标准化都在单个窗口的 K 个候选内部进行。

| split/组 | 分数 | Spearman | Kendall Tau | Top-1 | Top-2 | MAE regret |
|---|---|---:|---:|---:|---:|---:|
| train overall | H1 center | **0.3510** | **0.2895** | **0.3594** | **0.6016** | **0.00875** |
| train overall | H2 semantic | 0.0279 | 0.0248 | 0.1914 | 0.4121 | 0.01448 |
| train overall | H3 combined | 0.2818 | 0.2270 | 0.2969 | 0.5645 | 0.00988 |
| train normal | H1 center | 0.3699 | 0.3070 | 0.3711 | 0.6094 | 0.00730 |
| train fault | H1 center | 0.3320 | 0.2719 | 0.3477 | 0.5938 | 0.01019 |
| validation overall | H1 center | 0.3691 | 0.2930 | 0.3633 | 0.5977 | 0.00912 |
| test overall | H1 center | 0.3291 | 0.2664 | 0.3340 | 0.5801 | 0.00997 |
| test normal | H1 center | 0.3242 | 0.2641 | 0.3477 | 0.5742 | 0.00830 |
| test fault | H1 center | 0.3340 | 0.2688 | 0.3203 | 0.5859 | 0.01164 |

K=5 的随机 Top-1/Top-2 基线为 0.2/0.4。H1 在三个 split 以及 normal/fault 内均高于随机并保持同方向，是稳定但不充分的排序信号。H2 几乎没有 MAE 排序能力；H3 被 H2 稀释，整体弱于 H1，因此最佳无参考分数固定为 `h1_center`。

H1 在 train 选中候选 MAE 为 0.38305，优于随机期望 0.38916，但仍距 Oracle 0.37430 有 0.00875 regret。

## fault type 审计

train 中 20 个 fault type 的 Oracle 改善均为正，范围约 2.79%–5.31%；只有 type 20 在本次小样本中超过 5%。各 type 样本数仅 6–20，不能据此作 fault-specific 结论。H1 Spearman 多数为正，但 type 4/17/19 较弱；normal/fault 汇总方向一致，排除了只对 normal 有效的情况。

完整 fault-type 分组、H1/H2/H3 排序指标和 semantic regret 保存在忽略的 `rankability_result.json/.csv`。

## 第一级门控

通过五项：候选差异非零；Oracle 语义不差于随机；H1 rank signal；H1 Top-k 高于随机；normal/fault Oracle 方向一致。

失败一项：Oracle Best-of-5 相对随机的 train MAE 改善为 3.82%，未达到预先固定的 5% 工程门槛。工程门槛要求“Oracle≥5% 且无参考排序有效”，因此尽管六项中通过多数，最终工程闸门仍失败：

```text
INTRA_SAMPLE_CANDIDATE_RANKING_NO_GO
downstream_retest_allowed = false
```

该门槛在生成候选和查看结果前已经写入配置，没有根据 test 指标修改。

## 唯一下一步建议

停止扩展质量加权与候选选择模块，回到恢复器本身：先解释并缩小 diffusion 与 simple interpolation 的恢复误差差距；在单候选恢复质量达到或接近 simple baseline 之前，不再增加候选数、排序器或下游训练。
