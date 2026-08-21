# Q-DiffCL Paper-final Freeze

状态：`PAPER_FINAL_FREEZE_READY`。这是 pre-outer 冻结快照；没有训练 outer model，也没有读取 outer-test metric。

## Version

- Source branch before freeze commit: `exp/paper-final-outer`
- Source HEAD used to generate snapshot: `848b49302f21af39253243e5655ce7145e919ce6`
- Freeze commit: 由包含本快照的 Git commit/tag 确定，并写入 outer manifest。
- Freeze tag: `paper-final-freeze-pre-outer-v3`
- Outer branch after tag: `exp/paper-final-outer`
- Tests: `271 passed in 97.76s`
- Environment: Python 3.10.20, PyTorch 2.6.0+cu124, CUDA 12.4, NVIDIA GeForce RTX 4060 Laptop GPU

## Frozen method

- FINAL_QDIFFCL: `0.5D + 0.5E`, `S=0`, `critical_ratio=0.30`, selective timesteps `1/5`, soft channel-frequency allocation.
- TCN encoder, Hard SupCon, Original batching, frozen Linear Probe.
- DCBR: inner-validation domain calibration over `rho ∈ {0,.25,.5,.75,1}`; no learned controller and 0 inference parameters. Development references remain 3W `rho=1`, TEP `rho=.75`.
- SVR/router/controller: `NO_GO_SVR`, excluded.

## Frozen baseline set

`NO_AUG`, `JITTER`, `SCALING`, `JITTER_SCALING`, `UNIFORM_DIFFUSION`, `FRERA` shared-backbone adaptation, `FINAL_QDIFFCL`, `DCBR`.
AutoDA-Timeseries is method-native supplementary only; DiCL is not fairly reproducible. Neither enters the outer main-table matrix.

## Frozen evaluation protocol

- 3W: repeated grouped outer holdout, outer seeds `31001/31002/31003`, grouping unit WELL, per split 20 train / 8 inner-val / 8 outer-test WELL.
- TEP: repeated stratified Run-level outer holdout, outer seeds `32001/32002/32003`, per split 248 train / 72 inner-val / 80 outer-test Runs.
- Model seeds: 3W `42/43/44/45/46`; TEP `7/42/43/44/2026`.
- scaler/imputation/D/E/frequency statistics fit on outer-train only; rho/threshold/early stopping use inner validation only; outer-test is evaluated once.
- Primary metric Macro-F1; secondary AUPRC, FAR, Early Recall, Detection Delay, per-group/per-fault metrics; 2,000 group bootstrap repeats.

## Exact future run matrix

- 3W: `3 outer splits × 5 model seeds × 7 unique trained methods = 105` training/evaluation cells; DCBR `rho=1` may be emitted as 15 exact FINAL alias rows only when inner validation selects 1.
- TEP: `3 outer splits × 5 model seeds × 8 methods = 120` training/evaluation cells.
- Each cell writes config, split IDs, checkpoint, validation selection record, raw scores/predictions, per-group metrics and environment metadata.
- Expected roots: `outputs/paper_final/3w/outer_{seed}/model_seed_{seed}/{method}/` and `outputs/paper_final/tep/outer_{seed}/model_seed_{seed}/{method}/`.

## Hash audit

- 3W content collection: `2fb77ef15859e26011107c65b66c2050f6d0f097ecd5bf1368c98d89aa931dbe` (2229 files, 1873136197 bytes).
- TEP content collection: `a770c036cd6dc0cc33c792b3238d038392cb892ac3277e6fc92bf819c2d795dd` (4 files, 1402950911 bytes).
- Dry-run split manifest: `e4ef68e85b57f0ae90359f8ad7a83e2949a363fdff3945297c44bd102bd66207`.
- Leakage audit: `9c07814cc9cdfa38dc0e7728e7fffc4cd36d8d41a4904f15752a394e7618a3e4`; status `PAPER_FINAL_PROTOCOL_AMENDMENT_GO`, outer metrics `null`.
- Per-file data and config hashes are stored in `outputs/paper_final_freeze/freeze_manifest.json`.

## Resume and stopping policy

- Resume only an incomplete cell whose config, data, split, initialization and epoch-order hashes match exactly; otherwise fail closed.
- Completed outer-test cells are immutable and never rerun for selection.
- After the first outer metric is produced, algorithm structure, candidate grids, thresholds and baseline membership are locked; only predeclared analysis may continue.

## A–E inventory

- Reused existing experiment cells: `27`.
- New training cells: `27`.
- Evaluation-only checkpoint replays: `4`.
- Audit-only candidates: `2`.

## Remaining claim boundaries

- Paper-final generalization remains pending until the frozen outer matrix runs once.
- Limited-data and broader missingness robustness remain unsupported.
- 3W universal cross-WELL superiority remains unsupported; existing bootstrap CI crosses zero.
- TEP ratio 0.30 is a local sensitivity trough; the frozen parameter is not reopened, and no universal optimum claim is allowed.
- FRERA augmentation-only timing remains unavailable because its standalone augmenter checkpoint was not preserved.
