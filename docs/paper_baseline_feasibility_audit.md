# External Baseline Feasibility Audit

审计日期：2026-08-21。审计目标仅是判断候选能否在冻结的 TCN / Hard SupCon / grouped split / frozen Linear Probe 协议下公平复现；没有根据候选结果修改 Q-DiffCL，也没有运行 Paper-final outer evaluation。

## 结论

| Candidate | Verified source | Classification | Main-table action |
|---|---|---|---|
| AutoDA-Timeseries | [NetManAIOps/AutoDA-Timeseries](https://github.com/NetManAIOps/AutoDA-Timeseries), commit `91dbf70b54b255214b7f204d8d9f70d26f9c1fe3` | `METHOD_NATIVE_ONLY` | 不进入统一主协议；可在 supplementary 做 method-native comparison |
| DiCL（industrial diffusion + contrastive） | 未找到可核验的对应官方论文与仓库 | `NOT_FAIRLY_REPRODUCIBLE` | 不实现、不排名；保留 coverage gap |

## AutoDA-Timeseries

- 项目页：<https://netmanaiops.github.io/AutoDA-Timeseries/>。
- 官方仓库当前提供 cleaned implementation；无 release/tag，且仓库未声明 license，因此以审计 commit 固定版本。
- native classification 示例：`ROCKET` downstream、`Catch22` feature extractor、`ArticularyWordRecognition` 数据集、batch size 16、learning rate 0.005、35 epochs、patience 17。
- augmentation：三层可学习 policy；每层通过 feature-conditioned MLP 产生 transform probability 与 per-channel strength，使用 Gumbel-Softmax 选择变换。
- objective：下游 task loss，可选 entropy/diversity composite loss；策略和 downstream model 联合参与训练。
- label usage：classification task loss 直接使用标签。
- 输入适配：核心 `AugmentLayer.forward` 明确接收 `[B,C,L]`，形状可适配 Q-DiffCL 窗口。
- split / preprocessing：官方仓库自带 dataset adapter 和 Catch22 feature extraction，但没有 Q-DiffCL 的 WELL/Run grouped split；若比较必须另写 grouped adapter。

公平性判断：把 native downstream task loss 替换为 Hard SupCon、移除 Catch22 conditioning 或冻结 augmentation policy，都会实质改变 AutoDA 的核心机制。保留 native objective/backbone 又无法与 Q-DiffCL 的统一表示学习协议直接排名。因此本轮不创建所谓 “official reproduction” 或 “shared-backbone reproduction”；未来若补充，只能明确命名为 `AutoDA-Timeseries method-native adaptation`，放在 supplementary。

## DiCL

对 GitHub repository name/description 进行了 `DiCL`、`time series`、`fault diagnosis`、`industrial diffusion contrastive` 等组合查询，并对学术索引执行题名/摘要搜索。没有发现与“工业时间序列 diffusion + contrastive baseline”描述相匹配、可确定版本的官方实现。

检索到的主要同名项目均不相关：

- `jiabeiwangTJU/DICL`：Deep Intra-Image Contrastive Learning，任务是 person search；不是 diffusion 或工业时间序列。
- `tmllab/2024_ICML_DICL`：Denoising In-Context Learning，任务是视觉模型/MLLM robustness；不是 diffusion time-series augmentation。
- `abenechehab/dicl`：Disentangled In-Context Learning，用于 model-based reinforcement learning。

在无法唯一确认 paper、official repository、commit、native split、objective 和输入协议时，实现一个自定义 “DiCL-style” baseline 会成为不可审计的新方法，不是官方复现。因此分类为 `NOT_FAIRLY_REPRODUCIBLE`。

## Freeze decision

- 冻结主表继续包含：NoAug、Jitter、Scaling、Jitter+Scaling、Uniform Diffusion、FreRA shared-backbone adaptation、FINAL_QDIFFCL、DCBR。
- AutoDA-Timeseries 仅列为 method-native supplementary 候选，不进入本次 future outer run matrix。
- DiCL 保留为不可公平复现的 coverage gap。
- 本阶段新增 external-baseline 训练数：`0`。
