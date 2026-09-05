# QDIFFCL_DATA_REGIME_PROGRESS_AUDIT

> 原始审计时间：2026-09-05T17:36:17.345639+00:00

## Global accounting（hash-valid cells only）

| metric | value |
|--------|-------|
| formal_cells_expected | 375 |
| formal_cells_valid | 375 |
| formal_cells_invalid | 0 |
| formal_cells_remaining | 0 |
| rho_candidate_valid | 225 |
| rho_candidate_invalid | 0 |
| rho_candidate_remaining | 0 |
| duplicate_count | 0 |
| historical_failure_count | 3 |

## Method summary（dataset × fraction）

cells=15 / method 表示 3 outers × 5 seeds。
| 3W 0.1 | CALIBRATED_RHO | 15 | 0.3677 | 0.7705 | 0.5475 | 0.9188 | 863.0 |
 | 3W 0.1 | FINAL_QDIFFCL_FIXED | 15 | 0.3695 | 0.7628 | 0.5299 | 0.9118 | 891.2 |
 | 3W 0.1 | JITTER_SCALING | 15 | 0.3574 | 0.7736 | 0.5392 | 0.8877 | 1461.8 |
 | 3W 0.1 | NO_AUG | 15 | 0.3814 | 0.7634 | 0.5182 | 0.8900 | 1527.4 |
 | 3W 0.1 | UNIFORM_DIFFUSION | 15 | 0.3723 | 0.7714 | 0.5472 | 0.9176 | 864.0 |
 | 3W 0.25 | CALIBRATED_RHO | 15 | 0.3697 | 0.7336 | 0.6089 | 0.9031 | 1033.1 |
 | 3W 0.25 | FINAL_QDIFFCL_FIXED | 15 | 0.3701 | 0.7253 | 0.6219 | 0.8940 | 1175.2 |
 | 3W 0.25 | JITTER_SCALING | 15 | 0.3671 | 0.7468 | 0.6278 | 0.8758 | 1601.6 |
 | 3W 0.25 | NO_AUG | 15 | 0.3644 | 0.7267 | 0.6256 | 0.9179 | 800.9 |
 | 3W 0.25 | UNIFORM_DIFFUSION | 15 | 0.3762 | 0.7492 | 0.5951 | 0.9212 | 606.9 |
 | 3W 1.0 | CALIBRATED_RHO | 15 | 0.3221 | 0.7795 | 0.6430 | 0.8647 | 1538.8 |
 | 3W 1.0 | FINAL_QDIFFCL_FIXED | 15 | 0.2937 | 0.7305 | 0.6672 | 0.8514 | 1438.5 |
 | 3W 1.0 | JITTER_SCALING | 15 | 0.3430 | 0.7288 | 0.5910 | 0.8326 | 1177.1 |
 | 3W 1.0 | NO_AUG | 15 | 0.3241 | 0.7734 | 0.6133 | 0.8548 | 2372.2 |
 | 3W 1.0 | UNIFORM_DIFFUSION | 15 | 0.3134 | 0.7270 | 0.6341 | 0.8173 | 3144.4 |
 | TEP 0.25 | CALIBRATED_RHO | 15 | 0.8931 | 0.9513 | 0.0257 | 0.7504 | 102.4 |
 | TEP 0.25 | FINAL_QDIFFCL_FIXED | 15 | 0.8925 | 0.9509 | 0.0276 | 0.7521 | 102.0 |
 | TEP 0.25 | JITTER_SCALING | 15 | 0.8913 | 0.9504 | 0.0285 | 0.7471 | 103.2 |
 | TEP 0.25 | NO_AUG | 15 | 0.8929 | 0.9509 | 0.0300 | 0.7538 | 102.5 |
 | TEP 0.25 | UNIFORM_DIFFUSION | 15 | 0.8930 | 0.9508 | 0.0283 | 0.7537 | 103.0 |
 | TEP 1.0 | CALIBRATED_RHO | 15 | 0.9486 | 0.9861 | 0.0174 | 0.8821 | 84.3 |
 | TEP 1.0 | FINAL_QDIFFCL_FIXED | 15 | 0.9483 | 0.9859 | 0.0195 | 0.8838 | 84.2 |
 | TEP 1.0 | JITTER_SCALING | 15 | 0.9492 | 0.9865 | 0.0191 | 0.8838 | 84.2 |
 | TEP 1.0 | NO_AUG | 15 | 0.9480 | 0.9858 | 0.0175 | 0.8796 | 84.8 |
 | TEP 1.0 | UNIFORM_DIFFUSION | 15 | 0.9469 | 0.9854 | 0.0197 | 0.8796 | 85.0 |

## TEP10

TEP 10% = E_IDENTIFIABILITY_HOLD，不计算正式 D+E cell，未进入。

