# R2 多类别判别关键频率：跨数据集总结

阶段结论：

- 3W：`R2_3W_GO`
- TEP：`R2_CROSS_DATASET_PARTIAL_GO`
- Paper-final claim：不允许

R2 的 M 在两个数据集都真实改变了关键频率排序：3W mask Jaccard `.82427`、42 bins 改变；TEP Jaccard `.78819`、122 bins 改变。所有 M 都仅由 train class/faultNumber 拟合。

3W 上 R2−Uniform Macro-F1 `+.02953 ± .07698`，Multiclass AUPRC `+.05950`、Binary AUPRC `+.12276`、FAR `−.02881`，达到 GO；但优势高度依赖 seed 44，且 R2−R1 Macro-F1 为 `−.00305 ± .07345`。TEP 上 R2−C1 Macro-F1 3/3 正向但只有 `+.00204 ± .00163`，R2−R1 仅 `+.00016 ± .00016`，属于基本持平。

本轮实际执行 9 个新训练：3 个 fault-only M 预跑因遗漏明确要求的 normal class 0 而作废并保留 outputs；3 个正式 3W R2；3 个正式 TEP R2。正式比较复用 12 个旧结果（每数据集 Uniform/C1 与 R1 各 3 seeds）。

## 论文主线判断

R2 值得进入一次新的、严格冻结且未触碰评测协议中的 paper-final 候选复核，因为它在 3W 通过预注册 gate，TEP 方向不反向。但现有证据不能说明 R2 比 R1 更适合作为最终方法：3W 的 R2−R1 Macro-F1 略负且波动大，TEP 几乎完全持平，Class 9 仍是稳定低召回。

因此建议仅把 R2 与 R1 一起冻结为最终候选，下一阶段若获授权应在新评测协议中做 head-to-head，而不是继续调 M 权重、timestep、sampler、TCN 或增加新机制。本轮不声明统计显著，也不直接形成论文最终结论。
