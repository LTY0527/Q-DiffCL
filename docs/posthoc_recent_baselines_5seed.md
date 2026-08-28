# Recent time-series baselines: unified five-seed extension

Status: `POSTHOC_BASELINE_5SEED_EXTENSION_COMPLETE`

Evidence class: `POSTHOC_BASELINE_5SEED_EXTENSION`. This is post-hoc seed completion, not preregistered Paper-final evidence.

H1 archive commit: `b3aee5a2ad0cdc0634d3ef67a86e49c2b37489c4`. Protocol-lock commit: `2e5e1544e75303d50253c0c05c06918a4d7b30a6`. Protocol hash: `89cab7e6c6b3e9a127ee51cbec28ba6c4a730013139c7d3cc4e30664cd199b51`.

All four H1 baselines were extended independently of their three-seed result direction. The 72 H1 cells were reused and exactly 48 missing-seed cells were added; no completed H1 cell was retrained.

## Comparison boundary

- Track A: AutoTCL and SoftCLT are shared-backbone/mechanism adaptations, not official reproductions.
- Track B: TF-C and TS2Vec are method-native representation comparisons and do not identify augmentation-only causality.
- Frozen Paper-final methods are reused without retraining.

## Split-first five-seed results

Each split mean first averages five matched model seeds. Mean ± sample SD is then computed across the three frozen outer splits.

| Dataset | Track | Method | Macro-F1 | AUPRC | FAR | Early recall | Delay | Worst cell (outer/seed) |
|---|---|---|---:|---:|---:|---:|---:|---|
| 3W | TRACK_B_METHOD_NATIVE_REPRESENTATION | TF-C | 0.3392 ± 0.1305 | 0.7293 | 0.7202 | 0.9842 | 264.4546 | 0.1818 (31002/43) |
| 3W | TRACK_A_MECHANISM_ADAPTATION | SoftCLT | 0.3408 ± 0.0288 | 0.7764 | 0.6099 | 0.9199 | 457.4494 | 0.2382 (31001/42) |
| 3W | TRACK_B_METHOD_NATIVE_REPRESENTATION | TS2Vec | 0.2397 ± 0.0564 | 0.5020 | 0.6629 | 0.7695 | 1265.8419 | 0.0771 (31002/42) |
| 3W | TRACK_A_MECHANISM_ADAPTATION | AutoTCL | 0.3387 ± 0.1013 | 0.7502 | 0.6895 | 0.8462 | 198.4690 | 0.0620 (31002/45) |
| 3W | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | FINAL_QDIFFCL | 0.3216 ± 0.0324 | 0.7873 | 0.6515 | 0.9114 | 915.0599 | 0.1256 (31002/42) |
| 3W | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | NO_AUG | 0.2978 ± 0.0610 | 0.7791 | 0.6383 | 0.8707 | 800.0460 | 0.1091 (31002/43) |
| 3W | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | UNIFORM_DIFFUSION | 0.3099 ± 0.0656 | 0.7771 | 0.6608 | 0.9015 | 1177.6818 | 0.1285 (31002/44) |
| 3W | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | FRERA | 0.3334 ± 0.0974 | 0.6993 | 0.6244 | 0.8445 | 2010.5176 | 0.1402 (31002/42) |
| 3W | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | JITTER | 0.3012 ± 0.0893 | 0.7736 | 0.6701 | 0.8642 | 1807.7269 | 0.1297 (31002/42) |
| 3W | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | SCALING | 0.3203 ± 0.0957 | 0.7702 | 0.7079 | 0.8719 | 1306.5137 | 0.1461 (31002/44) |
| 3W | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | JITTER_SCALING | 0.3256 ± 0.0891 | 0.7657 | 0.6558 | 0.8714 | 861.7904 | 0.1507 (31002/44) |
| 3W | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | DCBR | 0.3204 ± 0.0981 | 0.7715 | 0.6623 | 0.8808 | 861.4848 | 0.0977 (31002/43) |
| TEP | TRACK_B_METHOD_NATIVE_REPRESENTATION | TF-C | 0.8039 ± 0.0166 | 0.8882 | 0.0942 | 0.6546 | 103.0704 | 0.7759 (32003/2026) |
| TEP | TRACK_A_MECHANISM_ADAPTATION | SoftCLT | 0.7513 ± 0.0102 | 0.8407 | 0.0950 | 0.5250 | 140.2104 | 0.7316 (32003/7) |
| TEP | TRACK_B_METHOD_NATIVE_REPRESENTATION | TS2Vec | 0.7657 ± 0.0166 | 0.8579 | 0.1514 | 0.5696 | 126.5681 | 0.7267 (32003/44) |
| TEP | TRACK_A_MECHANISM_ADAPTATION | AutoTCL | 0.9357 ± 0.0252 | 0.9799 | 0.0141 | 0.8408 | 87.1535 | 0.9017 (32001/2026) |
| TEP | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | FINAL_QDIFFCL | 0.9483 ± 0.0254 | 0.9859 | 0.0195 | 0.8838 | 84.1977 | 0.9012 (32001/2026) |
| TEP | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | NO_AUG | 0.9480 ± 0.0251 | 0.9858 | 0.0175 | 0.8796 | 84.8273 | 0.9027 (32001/2026) |
| TEP | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | UNIFORM_DIFFUSION | 0.9469 ± 0.0255 | 0.9854 | 0.0197 | 0.8796 | 84.9979 | 0.9024 (32001/2026) |
| TEP | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | FRERA | 0.9496 ± 0.0171 | 0.9895 | 0.0165 | 0.8746 | 85.8579 | 0.9099 (32002/2026) |
| TEP | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | JITTER | 0.9470 ± 0.0243 | 0.9857 | 0.0181 | 0.8750 | 84.8841 | 0.9032 (32001/2026) |
| TEP | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | SCALING | 0.9458 ± 0.0254 | 0.9847 | 0.0204 | 0.8783 | 85.2729 | 0.9011 (32001/2026) |
| TEP | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | JITTER_SCALING | 0.9492 ± 0.0239 | 0.9865 | 0.0191 | 0.8838 | 84.1875 | 0.9070 (32001/2026) |
| TEP | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | DCBR | 0.9486 ± 0.0250 | 0.9861 | 0.0174 | 0.8821 | 84.3116 | 0.9037 (32001/2026) |

## Paired group-aware bootstrap versus FINAL_QDIFFCL

Positive Δ means the row method has a higher paired Macro-F1 point estimate. Resampling uses WELL on 3W and Run on TEP; windows are never independent bootstrap units.

| Dataset | Track | Method | Δ Macro-F1 | 95% CI | Above cells | Non-worse cells | Worst delta |
|---|---|---|---:|---:|---:|---:|---:|
| 3W | TRACK_B_METHOD_NATIVE_REPRESENTATION | TF-C | +0.0176 | [-0.0352, +0.0819] | 7/15 | 7/15 | -0.2891 |
| 3W | TRACK_A_MECHANISM_ADAPTATION | SoftCLT | +0.0192 | [-0.0495, +0.0557] | 10/15 | 10/15 | -0.3039 |
| 3W | TRACK_B_METHOD_NATIVE_REPRESENTATION | TS2Vec | -0.0818 | [-0.1469, -0.0365] | 3/15 | 3/15 | -0.4007 |
| 3W | TRACK_A_MECHANISM_ADAPTATION | AutoTCL | +0.0172 | [-0.0437, +0.0501] | 9/15 | 9/15 | -0.4773 |
| 3W | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | NO_AUG | -0.0238 | [-0.0382, +0.0080] | 6/15 | 6/15 | -0.1696 |
| 3W | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | UNIFORM_DIFFUSION | -0.0117 | [-0.0337, +0.0018] | 7/15 | 7/15 | -0.1453 |
| 3W | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | FRERA | +0.0118 | [-0.0505, +0.0379] | 9/15 | 9/15 | -0.3173 |
| 3W | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | JITTER | -0.0204 | [-0.0558, -0.0035] | 7/15 | 7/15 | -0.2202 |
| 3W | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | SCALING | -0.0013 | [-0.0371, +0.0123] | 6/15 | 6/15 | -0.2380 |
| 3W | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | JITTER_SCALING | +0.0040 | [-0.0263, +0.0246] | 7/15 | 7/15 | -0.2473 |
| 3W | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | DCBR | -0.0012 | [-0.0301, +0.0112] | 6/15 | 6/15 | -0.1810 |
| TEP | TRACK_B_METHOD_NATIVE_REPRESENTATION | TF-C | -0.1444 | [-0.1604, -0.1312] | 0/15 | 0/15 | -0.1920 |
| TEP | TRACK_A_MECHANISM_ADAPTATION | SoftCLT | -0.1970 | [-0.2158, -0.1804] | 0/15 | 0/15 | -0.2518 |
| TEP | TRACK_B_METHOD_NATIVE_REPRESENTATION | TS2Vec | -0.1827 | [-0.1995, -0.1695] | 0/15 | 0/15 | -0.2487 |
| TEP | TRACK_A_MECHANISM_ADAPTATION | AutoTCL | -0.0127 | [-0.0178, -0.0075] | 5/15 | 5/15 | -0.0431 |
| TEP | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | NO_AUG | -0.0003 | [-0.0013, +0.0006] | 7/15 | 8/15 | -0.0038 |
| TEP | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | UNIFORM_DIFFUSION | -0.0015 | [-0.0028, -0.0004] | 5/15 | 7/15 | -0.0196 |
| TEP | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | FRERA | +0.0012 | [-0.0029, +0.0054] | 7/15 | 7/15 | -0.0200 |
| TEP | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | JITTER | -0.0014 | [-0.0033, +0.0006] | 3/15 | 3/15 | -0.0118 |
| TEP | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | SCALING | -0.0025 | [-0.0042, -0.0008] | 5/15 | 5/15 | -0.0281 |
| TEP | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | JITTER_SCALING | +0.0008 | [-0.0007, +0.0025] | 7/15 | 7/15 | -0.0072 |
| TEP | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | DCBR | +0.0003 | [-0.0014, +0.0021] | 5/15 | 5/15 | -0.0041 |

## Explicit above/below accounting

- 3W: above Q-DiffCL by paired point estimate: TF-C, SoftCLT, AutoTCL, FRERA, JITTER_SCALING; below: TS2Vec, NO_AUG, UNIFORM_DIFFUSION, JITTER, SCALING, DCBR. Use the 95% CI, not sign alone, for uncertainty claims.
- TEP: above Q-DiffCL by paired point estimate: FRERA, JITTER_SCALING, DCBR; below: TF-C, SoftCLT, TS2Vec, AutoTCL, NO_AUG, UNIFORM_DIFFUSION, JITTER, SCALING. Use the 95% CI, not sign alone, for uncertainty claims.

## Coverage and evidence limits

- Coverage is two datasets × three grouped outer splits × five matched model seeds.
- The extension does not modify Q-DiffCL, select new baselines, search seeds, or tune on outer-test results.
- Track B remains representation-level context; the extension does not close low-data, third-dataset, or missingness-robustness gaps.
