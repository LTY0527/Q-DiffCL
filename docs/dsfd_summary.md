# DSFD 总结

最终结论：`DSFD_DUAL_DATASET_NO_GO`；停止状态：`DSFD_SHORTCUT_GATE_NO_GO`；新增训练 run=0。

3W 存在 channel-frequency 域特异性，Fault/Domain Spearman=0.130660，Fault-low + Domain-high bins=247。但 DSFD 未降低 WELL-ID predictability：Accuracy 0.617937 → 0.618760；Fault Macro-F1 基本保持，delta=-0.000672。

因为核心 shortcut suppression 假设未得到操作性验证，Stage B/C 未执行，无法宣称改善 3W cross-WELL robustness 或完成 TEP preservation。第二创新正式冻结；下一阶段直接进入 R1 正式消融、external baselines 与 paper-final protocol。
