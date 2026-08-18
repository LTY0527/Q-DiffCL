# R1 DES 双数据集正式消融总结

## 主消融表

| variant | 3w macro f1 | 3w far | 3w early recall | tep macro f1 | tep far | tep early recall |
|---|---|---|---|---|---|---|
| UNIFORM | 0.4542 ± 0.0788 | 0.4288 ± 0.0503 | 0.8498 ± 0.1018 | 0.8846 ± 0.0044 | 0.0383 ± 0.0164 | 0.7708 ± 0.0295 |
| W/O_D | 0.5177 ± 0.0415 | 0.3530 ± 0.1726 | 0.8764 ± 0.0398 | 0.8862 ± 0.0052 | 0.0348 ± 0.0157 | 0.7708 ± 0.0295 |
| W/O_E | 0.4492 ± 0.1009 | 0.4629 ± 0.0785 | 0.8468 ± 0.1465 | 0.8862 ± 0.0060 | 0.0361 ± 0.0154 | 0.7771 ± 0.0377 |
| W/O_S | 0.5066 ± 0.0368 | 0.4135 ± 0.0957 | 0.8985 ± 0.0403 | 0.8860 ± 0.0053 | 0.0354 ± 0.0162 | 0.7708 ± 0.0295 |
| FULL_DES | 0.4867 ± 0.0746 | 0.3852 ± 0.0984 | 0.8440 ± 0.1330 | 0.8865 ± 0.0055 | 0.0353 ± 0.0169 | 0.7729 ± 0.0308 |

## 单分量补充表

| variant | 3w macro f1 | 3w far | 3w early recall | tep macro f1 | tep far | tep early recall |
|---|---|---|---|---|---|---|
| D_ONLY | 0.5128 ± 0.0365 | 0.3749 ± 0.1575 | 0.8899 ± 0.0538 | 0.8872 ± 0.0059 | 0.0327 ± 0.0095 | 0.7688 ± 0.0286 |
| E_ONLY | 0.4628 ± 0.0937 | 0.3841 ± 0.0879 | 0.8302 ± 0.1371 | 0.8837 ± 0.0050 | 0.0428 ± 0.0209 | 0.7750 ± 0.0348 |
| S_ONLY | 0.4216 ± 0.0884 | 0.4336 ± 0.0743 | 0.7861 ± 0.1669 | 0.8811 ± 0.0086 | 0.0523 ± 0.0391 | 0.7708 ± 0.0295 |
| FULL_DES | 0.4867 ± 0.0746 | 0.3852 ± 0.0984 | 0.8440 ± 0.1330 | 0.8865 ± 0.0055 | 0.0353 ± 0.0169 | 0.7729 ± 0.0308 |

## 删除分量的观测影响

- 删除 D（W/O_D）相对 Full 的 paired mean：3W macro_f1 +0.0309；3W far -0.0322；3W early_recall +0.0324；TEP macro_f1 -0.0003；TEP far -0.0005；TEP early_recall -0.0021。
- 删除 E（W/O_E）相对 Full 的 paired mean：3W macro_f1 -0.0375；3W far +0.0776；3W early_recall +0.0028；TEP macro_f1 -0.0003；TEP far +0.0008；TEP early_recall +0.0042。
- 删除 S（W/O_S）相对 Full 的 paired mean：3W macro_f1 +0.0198；3W far +0.0283；3W early_recall +0.0544；TEP macro_f1 -0.0005；TEP far +0.0001；TEP early_recall -0.0021。

## 冻结实验判读

- **D：未得到‘删除即退化’的必要性证据。** W/O_D 在 3W 的 Macro-F1、FAR、Early Recall 均值优于 Full，但 FAR std 从 0.0984 增至 0.1726；TEP 变化接近零。D 明显改变了频率选择，却不能由本轮结果宣称为不可缺少。
- **E：支持 3W 整体判别与 FAR，但没有验证预期的早期优势。** W/O_E 相对 Full 的 3W Macro-F1 降低 0.0375、FAR 增加 0.0776；然而 Early Recall 基本不变，平均检测延迟反而由 503.63 秒降至 151.89 秒。TEP 同样没有出现清晰的 Early 退化。
- **S：表现为指标权衡，稳定性假设未获直接支持。** W/O_S 的 3W Macro-F1 与 Early Recall 更高，但 FAR 更差且检测延迟升至 1143.96 秒；其 Macro-F1 std 反而小于 Full。TEP 三项主指标近似持平。
- **单分量结果：D_ONLY 最强，S_ONLY 最弱。** D_ONLY 在两数据集的主检测指标上具有竞争力；S_ONLY 的 Macro-F1/FAR 明显较差，说明单独依靠稳定性统计不足以支撑选择性扩散。
- **Full D+E+S 并非本轮双数据集上最均衡的唯一或一致最优方案。** Full 在 TEP 上保持稳定且有竞争力，但 3W 被 W/O_D 或 D_ONLY 在多项均值上超过。本轮因此不能声称三个分量均为必要，也不能声称 Full 稳定优于所有消融。

以上是冻结权重、冻结协议的描述性结论，不进行显著性外推，不据此修改权重或提出 R1-v2。

## Mask 与预算审计

### 3W

| Variant | Jaccard vs Full | Changed bins | Budget error |
|---|---:|---:|---:|
| W/O_D | 0.8793 | 28 | 0.000e+00 |
| W/O_E | 0.9039 | 22 | 3.725e-09 |
| W/O_S | 0.6834 | 82 | 0.000e+00 |
| FULL_DES | 1.0000 | 0 | 0.000e+00 |
| D_ONLY | 0.6269 | 100 | 0.000e+00 |
| E_ONLY | 0.6834 | 82 | 0.000e+00 |
| S_ONLY | 0.3711 | 200 | 3.725e-09 |

### TEP

| Variant | Jaccard vs Full | Changed bins | Budget error |
|---|---:|---:|---:|
| W/O_D | 0.7607 | 140 | 3.725e-09 |
| W/O_E | 0.8393 | 90 | 0.000e+00 |
| W/O_S | 0.8864 | 62 | 0.000e+00 |
| FULL_DES | 1.0000 | 0 | 0.000e+00 |
| D_ONLY | 0.8198 | 102 | 1.863e-09 |
| E_ONLY | 0.7547 | 144 | 1.863e-09 |
| S_ONLY | 0.3807 | 462 | 1.863e-09 |

删除 D/E/S 后，3W hard mask 分别改变 28/22/82 个 bins，TEP 分别改变 140/90/62 个 bins。3W 删除 D 的最大 composite 变化集中于 `(channel, frequency_bin)=(3,19),(6,3),(6,9),(6,15),(6,4)`；TEP 对应为 `(2,7),(7,18),(21,22),(26,4),(40,2)`。

各变体均在 canonical train split 上独立重建一次 mask，并跨模型 seed 冻结；timestep 保持 `t_critical=1`、`t_noncritical=5`，总频谱噪声预算与 Uniform `t=3` 匹配，最大误差仅 3.725e-09。完整 component/composite/soft mask、timestep map 和最大变化 bins 见 `r1_des_mask_audit.json` 及 `outputs/r1_des_ablation/masks/`。

## 完整性

Stage A 与 Stage B 均完整覆盖 3W 和 TEP 的 3 seeds：新增训练 36 runs；Uniform 与 Full 基线复用 12 runs。
