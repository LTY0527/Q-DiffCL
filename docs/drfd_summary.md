# DRFD 双数据集总结

最终结论：`DRFD_DUAL_DATASET_NO_GO`。Stage A=`DRFD_MECHANISM_GO`，Stage B=`DRFD_KILL_TEST_GO`，新增训练 run=`6`。

UG-R1 失败源于对称 uncertain→Uniform 会增加部分 R1 protected bins 的扰动；DRFD 通过 `t_r1<=3 => t_safe=t_r1`，并限制所有预算调整只发生在 reliable non-critical bins，避免削弱 semantic protection。

3W 可靠 critical / ambiguous / reliable non-critical 数量为 210 / 22 / 494；TEP 为 452 / 112 / 1152。Stage A 安全不变量和 2% budget Gate 均通过。

3W stability Gate=`NO-GO`；TEP preservation Gate=`NO-GO`。按预注册停止 uncertainty 方向，不搜索 rank threshold/confidence function，不开发 DRFD-v2，回退并冻结 R1。
