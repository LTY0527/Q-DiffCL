# FINAL_QDIFFCL Paper-Final Component Ablation

主表仅包含 Uniform、D_ONLY、E_ONLY、FINAL_DE；CURRENT_DES 作为历史 +S reference 放入 supplementary。

## 3W（seeds 42/43/44）

| Method | Section | Macro-F1 | FAR | Early Recall | AUPRC |
|---|---|---:|---:|---:|---:|
| UNIFORM | main | 0.4542 ± 0.0788 | 0.4288 ± 0.0503 | 0.8498 ± 0.1018 | 0.5020 ± 0.1719 |
| D_ONLY | main | 0.5128 ± 0.0365 | 0.3749 ± 0.1575 | 0.8899 ± 0.0538 | 0.5638 ± 0.0880 |
| E_ONLY | main | 0.4628 ± 0.0937 | 0.3841 ± 0.0879 | 0.8302 ± 0.1371 | 0.5081 ± 0.1781 |
| FINAL_DE | main | 0.5188 ± 0.0238 | 0.3555 ± 0.1279 | 0.8879 ± 0.0541 | 0.5635 ± 0.0838 |
| CURRENT_DES | supplementary | 0.4867 ± 0.0746 | 0.3852 ± 0.0984 | 0.8440 ± 0.1330 | 0.5182 ± 0.1498 |

## TEP（seeds 7/42/2026）

| Method | Section | Macro-F1 | FAR | Early Recall | AUPRC |
|---|---|---:|---:|---:|---:|
| UNIFORM | main | 0.8846 ± 0.0044 | 0.0383 ± 0.0164 | 0.7708 ± 0.0295 | 0.9273 ± 0.0067 |
| D_ONLY | main | 0.8872 ± 0.0059 | 0.0327 ± 0.0095 | 0.7688 ± 0.0286 | 0.9273 ± 0.0069 |
| E_ONLY | main | 0.8837 ± 0.0050 | 0.0428 ± 0.0209 | 0.7750 ± 0.0348 | 0.9274 ± 0.0069 |
| FINAL_DE | main | 0.8861 ± 0.0052 | 0.0355 ± 0.0165 | 0.7708 ± 0.0295 | 0.9275 ± 0.0070 |
| CURRENT_DES | supplementary | 0.8865 ± 0.0055 | 0.0353 ± 0.0169 | 0.7729 ± 0.0308 | 0.9275 ± 0.0070 |

D_ONLY/E_ONLY 来自协议与公平性哈希一致的既有 DES 消融；未重新运行上一轮完整 8-variant 消融。
