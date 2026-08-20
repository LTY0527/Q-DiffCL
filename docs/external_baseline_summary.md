# Q-DiffCL External Baseline / SOTA Comparison

Stage A/B/C 全部完成，无失败记录。FreRA shared-backbone adaptation 成功；未强行接入无法在本轮公平复现的 Tier 3 method-native 方法。

### 3W 3-seed

| Method | Macro-F1 | AUPRC | FAR | Early Recall |
|---|---:|---:|---:|---:|
| NO_AUG | 0.4299 ± 0.1457 | 0.4924 ± 0.1866 | 0.4142 ± 0.0955 | 0.7762 ± 0.2551 |
| JITTER | 0.5041 ± 0.0407 | 0.5833 ± 0.0778 | 0.4043 ± 0.1143 | 0.8841 ± 0.0326 |
| SCALING | 0.4826 ± 0.0179 | 0.5670 ± 0.0873 | 0.4512 ± 0.0583 | 0.9168 ± 0.0154 |
| JITTER_SCALING | 0.5175 ± 0.0274 | 0.5529 ± 0.0989 | 0.3590 ± 0.0933 | 0.8797 ± 0.0459 |
| UNIFORM_DIFFUSION | 0.4542 ± 0.0788 | 0.5020 ± 0.1719 | 0.4288 ± 0.0503 | 0.8498 ± 0.1018 |
| FINAL_QDIFFCL | 0.5188 ± 0.0238 | 0.5635 ± 0.0838 | 0.3555 ± 0.1279 | 0.8879 ± 0.0541 |
| FRERA | 0.5503 ± 0.0317 | 0.5966 ± 0.0757 | 0.3532 ± 0.1758 | 0.8955 ± 0.0506 |

### 3W 5-seed

| Method | Macro-F1 | AUPRC | FAR | Early Recall |
|---|---:|---:|---:|---:|
| UNIFORM_DIFFUSION | 0.4835 ± 0.0805 | 0.5631 ± 0.1482 | 0.4416 ± 0.0550 | 0.8646 ± 0.0774 |
| FINAL_QDIFFCL | 0.5396 ± 0.0354 | 0.6004 ± 0.0802 | 0.4042 ± 0.1192 | 0.8948 ± 0.0502 |
| JITTER_SCALING | 0.5370 ± 0.0500 | 0.5884 ± 0.0851 | 0.3907 ± 0.0871 | 0.8800 ± 0.0362 |
| FRERA | 0.5363 ± 0.0357 | 0.6094 ± 0.0630 | 0.4120 ± 0.1511 | 0.9058 ± 0.0435 |

### TEP 3-seed

| Method | Macro-F1 | AUPRC | FAR | Early Recall |
|---|---:|---:|---:|---:|
| NO_AUG | 0.9056 ± 0.0025 | 0.9388 ± 0.0018 | 0.0184 ± 0.0035 | 0.8042 ± 0.0095 |
| JITTER | 0.9042 ± 0.0015 | 0.9388 ± 0.0022 | 0.0214 ± 0.0039 | 0.8021 ± 0.0191 |
| SCALING | 0.9049 ± 0.0025 | 0.9389 ± 0.0003 | 0.0176 ± 0.0041 | 0.8021 ± 0.0219 |
| JITTER_SCALING | 0.9029 ± 0.0041 | 0.9369 ± 0.0033 | 0.0207 ± 0.0027 | 0.7979 ± 0.0157 |
| UNIFORM_DIFFUSION | 0.8846 ± 0.0044 | 0.9273 ± 0.0067 | 0.0383 ± 0.0164 | 0.7708 ± 0.0295 |
| FINAL_QDIFFCL | 0.8861 ± 0.0052 | 0.9275 ± 0.0070 | 0.0355 ± 0.0165 | 0.7708 ± 0.0295 |
| FRERA | 0.8920 ± 0.0172 | 0.9270 ± 0.0110 | 0.0227 ± 0.0034 | 0.7458 ± 0.0219 |

### TEP 5-seed

| Method | Macro-F1 | AUPRC | FAR | Early Recall |
|---|---:|---:|---:|---:|
| UNIFORM_DIFFUSION | 0.8894 ± 0.0073 | 0.9298 ± 0.0060 | 0.0311 ± 0.0161 | 0.7637 ± 0.0240 |
| FINAL_QDIFFCL | 0.8903 ± 0.0068 | 0.9299 ± 0.0060 | 0.0297 ± 0.0151 | 0.7637 ± 0.0240 |
| SCALING | 0.9048 ± 0.0026 | 0.9383 ± 0.0013 | 0.0179 ± 0.0049 | 0.7937 ± 0.0198 |
| FRERA | 0.8874 ± 0.0152 | 0.9245 ± 0.0107 | 0.0241 ± 0.0032 | 0.7362 ± 0.0389 |

## 主要配对结论

- 3W：FINAL `0.5396`，最强外部方法 JITTER_SCALING `0.5370`，FINAL 配对均值差 `+0.0026`，属于持平。
- TEP：最强外部方法 SCALING `0.9048`，FINAL `0.8903`，外部方法配对优势 `+0.0145`，正向种子比例 `100%`。
- catastrophic（相对 FINAL Macro-F1 下降超过 0.10）记录：`[{"dataset": "3W", "method": "NO_AUG", "seed": 44, "delta_macro_f1": -0.23998896027499145}, {"dataset": "3W", "method": "UNIFORM_DIFFUSION", "seed": 44, "delta_macro_f1": -0.13891367965760676}]`。失败/负结果未删除。

5-seed 配对均值（正值表示 method 高于 reference；FAR 负值更优）：

| Dataset | Method | Reference | ΔMacro-F1 | ΔAUPRC | ΔFAR | ΔEarly Recall |
|---|---|---|---:|---:|---:|---:|
| 3W | JITTER_SCALING | FINAL_QDIFFCL | -0.0026 | -0.0119 | -0.0134 | -0.0148 |
| 3W | JITTER_SCALING | UNIFORM_DIFFUSION | +0.0536 | +0.0253 | -0.0509 | +0.0154 |
| 3W | FRERA | FINAL_QDIFFCL | -0.0033 | +0.0090 | +0.0078 | +0.0109 |
| 3W | FRERA | UNIFORM_DIFFUSION | +0.0528 | +0.0463 | -0.0297 | +0.0411 |
| TEP | SCALING | FINAL_QDIFFCL | +0.0145 | +0.0084 | -0.0118 | +0.0300 |
| TEP | SCALING | UNIFORM_DIFFUSION | +0.0154 | +0.0085 | -0.0132 | +0.0300 |
| TEP | FRERA | FINAL_QDIFFCL | -0.0028 | -0.0054 | -0.0055 | -0.0275 |
| TEP | FRERA | UNIFORM_DIFFUSION | -0.0020 | -0.0053 | -0.0070 | -0.0275 |

## C 档失败定位

- 3W 最强外部方法 JITTER_SCALING 相对 FINAL 的 per-class F1 均值差为 `{"0": 0.005363252989442846, "2": 0.010316796937251672, "8": 0.0037947785736020647, "9": -0.029824257949799628}`；主要弱项是 class 9。故障 instance 变化：新增检测 `1`、丢失检测 `2`、更快 `36`、更慢 `6`；丢失项为 `[[43, "WELL-00014_20160304155906"], [46, "WELL-00014_20160304155906"]]`。
- TEP SCALING 的改善集中于 fault 10/11/13/16/17/18 的检测延迟，并在 fault 3 出现新增与丢失检测混合；完整计数为 `{"3": {"new_detection": 2, "lost_detection": 1}, "8": {"faster": 1}, "10": {"faster": 7, "slower": 2}, "11": {"faster": 3}, "13": {"faster": 2}, "16": {"faster": 6, "slower": 2}, "17": {"faster": 2}, "18": {"faster": 3}, "20": {"slower": 1}}`。

## 公平性与开销

所有 Stage C 新训练方法的初始化、split/window/manifest、SupCon batch order 与 probe order 哈希均与同 seed FINAL 对齐。传统增强不增加模型参数；FreRA 仅在预训练期增加 66 个频域门控参数，推理仍为相同 TCN。逐 seed runtime/GPU memory 见 `external_baseline_results.csv`。

## 未纳入主表的方法

FreRA 官方 method-native 使用 FCN+SimCLR、200 epochs 与自己的数据切分，不能与 augmentation-only 主表混排；本轮报告可审计的 shared-backbone adaptation。AutoTCL 及其他 diffusion/contrastive Tier 3 未在不改变 encoder/objective/protocol 的合理工作量内完成官方适配，按提示词不为数量强行实现。
