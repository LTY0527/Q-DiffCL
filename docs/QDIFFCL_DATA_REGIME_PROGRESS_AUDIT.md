# Q-DiffCL Data-Regime Progress Audit

Status: `PARTIAL_RESULTS_ARCHIVED` / `INTERIM_PARTIAL_EVIDENCE`.

The runner is stopped at audit time. This report inventories all hash-valid artifacts currently on disk.

## Cell accounting

- Formal: 325 valid / 375 expected; 50 remaining.
- Rho candidates: 195 valid / 225 expected; 30 remaining.
- Invalid: formal 0, rho 0; duplicates 0; runner failures 3.

## Dataset and fraction completion

### 3W

- 100%: 75 formal (75 new, 0 reused), 45 rho candidates, completed outers [31001, 31002, 31003], completed seeds [42, 43, 44, 45, 46], methods {'NO_AUG': 15, 'UNIFORM_DIFFUSION': 15, 'JITTER_SCALING': 15, 'FINAL_QDIFFCL_FIXED': 15, 'CALIBRATED_RHO': 15}, rho selections 3.
- 25%: 75 formal (75 new, 0 reused), 45 rho candidates, completed outers [31001, 31002, 31003], completed seeds [42, 43, 44, 45, 46], methods {'NO_AUG': 15, 'UNIFORM_DIFFUSION': 15, 'JITTER_SCALING': 15, 'FINAL_QDIFFCL_FIXED': 15, 'CALIBRATED_RHO': 15}, rho selections 3.
- 10%: 75 formal (75 new, 0 reused), 45 rho candidates, completed outers [31001, 31002, 31003], completed seeds [42, 43, 44, 45, 46], methods {'NO_AUG': 15, 'UNIFORM_DIFFUSION': 15, 'JITTER_SCALING': 15, 'FINAL_QDIFFCL_FIXED': 15, 'CALIBRATED_RHO': 15}, rho selections 3.

### TEP

- 100%: 75 formal (75 new, 0 reused), 45 rho candidates, completed outers [32001, 32002, 32003], completed seeds [7, 42, 43, 44, 2026], methods {'NO_AUG': 15, 'UNIFORM_DIFFUSION': 15, 'JITTER_SCALING': 15, 'FINAL_QDIFFCL_FIXED': 15, 'CALIBRATED_RHO': 15}, rho selections 3.
- 25%: 25 formal (25 new, 0 reused), 15 rho candidates, completed outers [32001], completed seeds [7, 42, 43, 44, 2026], methods {'NO_AUG': 5, 'UNIFORM_DIFFUSION': 5, 'JITTER_SCALING': 5, 'FINAL_QDIFFCL_FIXED': 5, 'CALIBRATED_RHO': 5}, rho selections 1.
- 10%: E_IDENTIFIABILITY_HOLD; excluded from primary matrix.

## Interim metric means (valid completed cells only)

| Dataset | Fraction | Method | Cells | Macro-F1 | AUPRC | FAR | Early Recall | Delay |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 3W | 0.10 | CALIBRATED_RHO | 15 | 0.367732 | 0.770487 | 0.547525 | 0.918797 | 863.019197 |
| 3W | 0.10 | FINAL_QDIFFCL_FIXED | 15 | 0.369540 | 0.762773 | 0.529901 | 0.911767 | 891.215268 |
| 3W | 0.10 | JITTER_SCALING | 15 | 0.357361 | 0.773648 | 0.539170 | 0.887708 | 1461.761951 |
| 3W | 0.10 | NO_AUG | 15 | 0.381413 | 0.763413 | 0.518151 | 0.889991 | 1527.431404 |
| 3W | 0.10 | UNIFORM_DIFFUSION | 15 | 0.372294 | 0.771429 | 0.547248 | 0.917645 | 864.001876 |
| 3W | 0.25 | CALIBRATED_RHO | 15 | 0.369744 | 0.733593 | 0.608886 | 0.903063 | 1033.118555 |
| 3W | 0.25 | FINAL_QDIFFCL_FIXED | 15 | 0.370121 | 0.725258 | 0.621877 | 0.894027 | 1175.183231 |
| 3W | 0.25 | JITTER_SCALING | 15 | 0.367138 | 0.746837 | 0.627800 | 0.875804 | 1601.588056 |
| 3W | 0.25 | NO_AUG | 15 | 0.364369 | 0.726660 | 0.625621 | 0.917875 | 800.891438 |
| 3W | 0.25 | UNIFORM_DIFFUSION | 15 | 0.376176 | 0.749213 | 0.595086 | 0.921208 | 606.931101 |
| 3W | 1.00 | CALIBRATED_RHO | 15 | 0.322101 | 0.779503 | 0.643038 | 0.864666 | 1538.824988 |
| 3W | 1.00 | FINAL_QDIFFCL_FIXED | 15 | 0.293659 | 0.730497 | 0.667194 | 0.851419 | 1438.524418 |
| 3W | 1.00 | JITTER_SCALING | 15 | 0.343001 | 0.728762 | 0.590977 | 0.832603 | 1177.141058 |
| 3W | 1.00 | NO_AUG | 15 | 0.324148 | 0.773382 | 0.613329 | 0.854842 | 2372.181296 |
| 3W | 1.00 | UNIFORM_DIFFUSION | 15 | 0.313388 | 0.726952 | 0.634137 | 0.817333 | 3144.420692 |
| TEP | 0.25 | CALIBRATED_RHO | 5 | 0.898996 | 0.950648 | 0.022618 | 0.713750 | 110.982361 |
| TEP | 0.25 | FINAL_QDIFFCL_FIXED | 5 | 0.896362 | 0.949064 | 0.025634 | 0.712500 | 109.861842 |
| TEP | 0.25 | JITTER_SCALING | 5 | 0.898187 | 0.950212 | 0.024949 | 0.716250 | 105.780180 |
| TEP | 0.25 | NO_AUG | 5 | 0.897539 | 0.949324 | 0.024674 | 0.707500 | 110.867267 |
| TEP | 0.25 | UNIFORM_DIFFUSION | 5 | 0.897742 | 0.948837 | 0.017820 | 0.703750 | 113.347621 |
| TEP | 1.00 | CALIBRATED_RHO | 15 | 0.948639 | 0.986054 | 0.017367 | 0.882083 | 84.311606 |
| TEP | 1.00 | FINAL_QDIFFCL_FIXED | 15 | 0.948324 | 0.985883 | 0.019507 | 0.883750 | 84.197742 |
| TEP | 1.00 | JITTER_SCALING | 15 | 0.949154 | 0.986473 | 0.019107 | 0.883750 | 84.187548 |
| TEP | 1.00 | NO_AUG | 15 | 0.948015 | 0.985783 | 0.017469 | 0.879583 | 84.827269 |
| TEP | 1.00 | UNIFORM_DIFFUSION | 15 | 0.946858 | 0.985365 | 0.019727 | 0.879583 | 84.997913 |

## Interim paired Macro-F1

| Dataset | Fraction | Contrast | Paired cells | Mean delta |
|---|---:|---|---:|---:|
| 3W | 0.10 | FINAL_QDIFFCL_FIXED - NO_AUG | 15 | -0.011873 |
| 3W | 0.10 | FINAL_QDIFFCL_FIXED - UNIFORM_DIFFUSION | 15 | -0.002754 |
| 3W | 0.10 | FINAL_QDIFFCL_FIXED - JITTER_SCALING | 15 | 0.012179 |
| 3W | 0.10 | CALIBRATED_RHO - FINAL_QDIFFCL_FIXED | 15 | -0.001808 |
| 3W | 0.10 | CALIBRATED_RHO - UNIFORM_DIFFUSION | 15 | -0.004562 |
| 3W | 0.25 | FINAL_QDIFFCL_FIXED - NO_AUG | 15 | 0.005751 |
| 3W | 0.25 | FINAL_QDIFFCL_FIXED - UNIFORM_DIFFUSION | 15 | -0.006055 |
| 3W | 0.25 | FINAL_QDIFFCL_FIXED - JITTER_SCALING | 15 | 0.002983 |
| 3W | 0.25 | CALIBRATED_RHO - FINAL_QDIFFCL_FIXED | 15 | -0.000377 |
| 3W | 0.25 | CALIBRATED_RHO - UNIFORM_DIFFUSION | 15 | -0.006432 |
| 3W | 1.00 | FINAL_QDIFFCL_FIXED - NO_AUG | 15 | -0.030489 |
| 3W | 1.00 | FINAL_QDIFFCL_FIXED - UNIFORM_DIFFUSION | 15 | -0.019729 |
| 3W | 1.00 | FINAL_QDIFFCL_FIXED - JITTER_SCALING | 15 | -0.049342 |
| 3W | 1.00 | CALIBRATED_RHO - FINAL_QDIFFCL_FIXED | 15 | 0.028442 |
| 3W | 1.00 | CALIBRATED_RHO - UNIFORM_DIFFUSION | 15 | 0.008713 |
| TEP | 0.25 | FINAL_QDIFFCL_FIXED - NO_AUG | 5 | -0.001177 |
| TEP | 0.25 | FINAL_QDIFFCL_FIXED - UNIFORM_DIFFUSION | 5 | -0.001380 |
| TEP | 0.25 | FINAL_QDIFFCL_FIXED - JITTER_SCALING | 5 | -0.001825 |
| TEP | 0.25 | CALIBRATED_RHO - FINAL_QDIFFCL_FIXED | 5 | 0.002634 |
| TEP | 0.25 | CALIBRATED_RHO - UNIFORM_DIFFUSION | 5 | 0.001254 |
| TEP | 1.00 | FINAL_QDIFFCL_FIXED - NO_AUG | 15 | 0.000309 |
| TEP | 1.00 | FINAL_QDIFFCL_FIXED - UNIFORM_DIFFUSION | 15 | 0.001466 |
| TEP | 1.00 | FINAL_QDIFFCL_FIXED - JITTER_SCALING | 15 | -0.000830 |
| TEP | 1.00 | CALIBRATED_RHO - FINAL_QDIFFCL_FIXED | 15 | 0.000314 |
| TEP | 1.00 | CALIBRATED_RHO - UNIFORM_DIFFUSION | 15 | 0.001781 |

These are stage results, not paper-final cross-dataset claims. Only completed hash-valid locked-test cells are included.

## Rho selections

- 3W 0.10 outer 31001: DATA_REGIME_RHO_STAR=1.0, candidates=15/15, seeds=[42, 43, 44], validation Macro-F1=0.646776, AUPRC=0.946015, FAR=0.207333, test_used_for_selection=false.
- 3W 0.10 outer 31002: DATA_REGIME_RHO_STAR=1.0, candidates=15/15, seeds=[42, 43, 44], validation Macro-F1=0.346855, AUPRC=0.761171, FAR=0.761333, test_used_for_selection=false.
- 3W 0.10 outer 31003: DATA_REGIME_RHO_STAR=0.0, candidates=15/15, seeds=[42, 43, 44], validation Macro-F1=0.849711, AUPRC=0.985142, FAR=0.107167, test_used_for_selection=false.
- 3W 0.25 outer 31001: DATA_REGIME_RHO_STAR=0.0, candidates=15/15, seeds=[42, 43, 44], validation Macro-F1=0.666912, AUPRC=0.913047, FAR=0.264500, test_used_for_selection=false.
- 3W 0.25 outer 31002: DATA_REGIME_RHO_STAR=0.75, candidates=15/15, seeds=[42, 43, 44], validation Macro-F1=0.290257, AUPRC=0.647239, FAR=0.840167, test_used_for_selection=false.
- 3W 0.25 outer 31003: DATA_REGIME_RHO_STAR=0.75, candidates=15/15, seeds=[42, 43, 44], validation Macro-F1=0.847734, AUPRC=0.970740, FAR=0.176167, test_used_for_selection=false.
- 3W 1.00 outer 31001: DATA_REGIME_RHO_STAR=0.0, candidates=15/15, seeds=[42, 43, 44], validation Macro-F1=0.709779, AUPRC=0.918761, FAR=0.247500, test_used_for_selection=false.
- 3W 1.00 outer 31002: DATA_REGIME_RHO_STAR=0.75, candidates=15/15, seeds=[42, 43, 44], validation Macro-F1=0.278721, AUPRC=0.664914, FAR=0.826000, test_used_for_selection=false.
- 3W 1.00 outer 31003: DATA_REGIME_RHO_STAR=1.0, candidates=15/15, seeds=[42, 43, 44], validation Macro-F1=0.851350, AUPRC=0.926635, FAR=0.114833, test_used_for_selection=false.
- TEP 0.25 outer 32001: DATA_REGIME_RHO_STAR=0.25, candidates=15/15, seeds=[7, 42, 2026], validation Macro-F1=0.895625, AUPRC=0.966632, FAR=0.033555, test_used_for_selection=false.
- TEP 1.00 outer 32001: DATA_REGIME_RHO_STAR=0.5, candidates=15/15, seeds=[7, 42, 2026], validation Macro-F1=0.925485, AUPRC=0.982242, FAR=0.054764, test_used_for_selection=false.
- TEP 1.00 outer 32002: DATA_REGIME_RHO_STAR=0.25, candidates=15/15, seeds=[7, 42, 2026], validation Macro-F1=0.904878, AUPRC=0.964991, FAR=0.030769, test_used_for_selection=false.
- TEP 1.00 outer 32003: DATA_REGIME_RHO_STAR=0.75, candidates=15/15, seeds=[7, 42, 2026], validation Macro-F1=0.964872, AUPRC=0.995927, FAR=0.043176, test_used_for_selection=false.

`DATA_REGIME_RHO_STAR` values above are outer-specific validation-only choices. They are distinct from historical `HISTORICAL_DCBR_GLOBAL_RHO` lineage values.

## Local artifact archive

- `outputs\qdiffcl_data_regime_v1` contains 1906 files (367293798 bytes); it remains local and Git-ignored.
- Files larger than 50 MiB: 0. Checkpoint, prediction, and result hashes are recorded per completed cell in `analysis/results/qdiffcl_data_regime_progress_audit.csv`.
- No local training artifact was deleted or staged for Git.

## Resume

No resume was executed. Registered command: `E:\anaconda\envs\qdiffcl\python.exe -u -m scripts.run_qdiffcl_data_regime --stage all --device cuda`.
Remaining work: 50 formal cells and 30 rho candidates. Remaining GPU time is not estimated by this artifact audit.
The TEP memory-safe loader is registered as a numerically equivalent runtime amendment; each new outer remains subject to the supervised RAM preflight.
