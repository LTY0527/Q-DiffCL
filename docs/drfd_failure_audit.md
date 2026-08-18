# DRFD 失败审计

Phase A 结论：`DRFD_RANK_RELIABILITY_INSUFFICIENT`。本阶段新增训练 run 为 0。

## A1 Fixed-FAR / Calibration

TEP 两 operating points 的最大 DRFD−R1 test FAR 差为 +0.012500；calibration 是否消除 FAR 问题：False。3W paired Macro-F1 非负 seed 数为 1/3。

## A2 Score trajectory

seed 44 标准协议中 DRFD AUPRC/Early Recall 上升但 delay 变差，主要因为 3W 标准评估使用 multiclass argmax、无单一 binary validation threshold，且 detected-instance 子集变化会改变 delay 均值。fixed-FAR 下 DRFD 的 seed 44 delay 反而改善；trajectory 中持续 3-window 与单窗口结果用于判断短暂波动，未修改 alarm rule。

## A3 Reliability

有效 pseudo-unseen train WELL=11，stable-rank 但出现 false-noncritical risk 的 bins=321，rank-IQR 与 unsafe-rate Spearman=0.648036。这些反例占 reliable non-critical 的 64.98%；rank reliability 能否作为 pseudo-unseen safety certificate：False。

因此 fixed-FAR 不能充分解释 paired inconsistency，且 source WELL 内 rank stability 不等价于 pseudo-unseen semantic safety。
