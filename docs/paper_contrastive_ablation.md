# Contrastive Learning Necessity Ablation

所有结果均为既有 development test 协议或同协议新增训练；不是 Paper-final outer evidence。

## 3W

| Objective | Augmentation | Macro-F1 | AUPRC | FAR | Early Recall | Delay |
|---|---|---:|---:|---:|---:|---:|
| CE_REP | NO_AUG | 0.5921 ± 0.0562 | 0.6254 | 0.3625 | 0.8927 | 950.10 |
| CE_REP | FINAL_QDIFFCL | 0.5327 ± 0.0862 | 0.6102 | 0.3662 | 0.8846 | 865.43 |
| HARD_SUPCON | NO_AUG | 0.4299 ± 0.1457 | 0.4924 | 0.4142 | 0.7762 | 450.90 |
| HARD_SUPCON | FINAL_QDIFFCL | 0.5188 ± 0.0238 | 0.5635 | 0.3555 | 0.8879 | 486.81 |

Macro-F1 paired augmentation delta：CE `-0.0594`，Hard SupCon `+0.0889`。
Interaction `(FINAL-NO_AUG)_SupCon-(FINAL-NO_AUG)_CE` = `+0.1483`，95% bootstrap CI `[+0.0515, +0.2409]`，positive/non-worse `3/3` / `3`，Cohen dz `1.564`。

## TEP

| Objective | Augmentation | Macro-F1 | AUPRC | FAR | Early Recall | Delay |
|---|---|---:|---:|---:|---:|---:|
| CE_REP | NO_AUG | 0.9001 ± 0.0028 | 0.9355 | 0.0281 | 0.7896 | 165.62 |
| CE_REP | FINAL_QDIFFCL | 0.9011 ± 0.0024 | 0.9357 | 0.0241 | 0.7812 | 155.27 |
| HARD_SUPCON | NO_AUG | 0.9056 ± 0.0025 | 0.9388 | 0.0184 | 0.8042 | 111.15 |
| HARD_SUPCON | FINAL_QDIFFCL | 0.8861 ± 0.0052 | 0.9275 | 0.0355 | 0.7708 | 117.01 |

Macro-F1 paired augmentation delta：CE `+0.0010`，Hard SupCon `-0.0196`。
Interaction `(FINAL-NO_AUG)_SupCon-(FINAL-NO_AUG)_CE` = `-0.0206`，95% bootstrap CI `[-0.0290, -0.0120]`，positive/non-worse `0/0` / `3`，Cohen dz `-2.422`。
