# Q-DiffCL Data-Regime Protocol V1

Status before the lock commit: `DATA_REGIME_SANITY_GO`.

This protocol measures **training-data scarcity under a fixed validation protocol**. It is not a fully label-scarce setting. The validation set is deliberately kept fixed to isolate training-side information scarcity; it is never used for parameter fitting, and is used only for frozen early stopping, threshold selection, and validation-only rho selection. Outer test groups remain frozen and are read once after all selection is locked.

## Frozen scientific definition

- Evidence class: `DATA_REGIME_GENERALIZATION_V1`
- Datasets: 3W and TEP
- Training fractions: 1.00, 0.25, 0.10
- Methods: NO_AUG, UNIFORM_DIFFUSION, JITTER_SCALING, FINAL_QDIFFCL_FIXED, CALIBRATED_RHO
- FINAL_QDIFFCL_FIXED: `0.5D + 0.5E`, `S=0`, critical ratio 0.30, diffusion steps 50, uniform timestep 3, critical/noncritical timesteps 1/5, phase/DC preservation, iid noise, seed offset 61000, clip quantile 0.999
- Encoder/objective/probe: TCN, Hard SupCon, original batching, frozen linear probe
- Model seeds: 3W `[42,43,44,45,46]`; TEP `[7,42,43,44,2026]`
- Rho selection seeds: 3W `[42,43,44]`; TEP `[7,42,2026]`
- Rho grid: `[0,.25,.5,.75,1]`
- Rho order: validation Macro-F1, validation AUPRC, lower validation FAR, then smaller rho

`CALIBRATED_RHO` is the DCBR mechanism extended to the data-regime axis. `HISTORICAL_DCBR_GLOBAL_RHO` (3W=1.00, TEP=0.75) is lineage context only. `DATA_REGIME_RHO_STAR(dataset,fraction,outer)` is independently selected for every outer using only that outer's fraction-local train set and frozen validation set. This includes fraction 1.00.

## Frozen outer and low-data units

- 3W outer IDs: 31001, 31002, 31003; grouping unit WELL. Training reduction occurs at `instance_id` trajectory level inside frozen train wells, before windows.
- TEP outer IDs: 32001, 32002, 32003; grouping unit Run. Training reduction occurs at `run_uid` trajectory level, before windows.
- Fraction manifests are deterministic, model-seed independent, and nested: 10% is a strict subset of 25%, which is a strict subset of 100%.
- Validation and test group identities and sizes never change with fraction.

Every train-derived object is fraction-local: normalization, spectral statistics, D, E, final mask, class statistics, augmentation views, and encoder/probe fitting. No unused portion of outer train, validation, or test data may be used to fit these objects.

The frozen 3W feature dimension is retained across fractions. If a frozen sensor channel has no finite observation in a selected fraction, it is represented by a neutral all-zero normalized channel and recorded in the context audit; no full-data statistic is borrowed.

## E-identifiability outcome

The preregistered floor is two independent onset-bearing and early-stage units per E-required fault class. 3W 100/25/10 and TEP 100/25 pass. TEP 10% has only one Run per fault class and therefore triggers `E_IDENTIFIABILITY_HOLD`. It is excluded from the primary complete D+E/CALIBRATED matrix without resampling, threshold changes, or fraction expansion. Other legal fractions continue unchanged.

## Fixed training budget

There is no optimizer-step compensation. Epoch caps, patience, batch size, optimizer, learning rate, temperature, and probe policy do not vary by fraction or method.

- 3W: pretrain 20 epochs, probe 15 epochs, batch 256, learning rate 0.001, temperature 0.1, patience 20.
- TEP: pretrain 8 epochs, probe 8 epochs, batch 128, learning rate 0.001, temperature 0.1, pretrain/probe patience 8.

Actual epochs, optimizer steps, examples seen, batch size, stopping epoch, and stopping reason are recorded per training artifact. Low fractions naturally execute fewer steps per epoch.

## Test boundary, resume, and accounting

Two validation-only smoke tests precede the protocol lock: NO_AUG plus FINAL on one low-data cell, and a two-candidate rho plumbing smoke in the separate `SMOKE_ONLY` namespace. Formal selection restores the full five-rho grid.

After the local lock commit, each formal `(dataset,fraction,outer,method,seed)` cell has one ID and an outer-test-started guard. Completed artifacts are skipped only after checkpoint and prediction hashes validate. Atomic JSON manifests and a runtime heartbeat support resumption. Historical 100% cells are reused only if every registered compatibility hash matches; otherwise necessary 100% cells are rerun.

After the TEP 10% hold, expected primary accounting is 375 locked-test cells and 225 validation rho-candidate cells. Historical exact reuse is initially zero pending cell-level proof.

## Statistics

Primary metric is Macro-F1; secondary metrics are AUPRC, FAR, Early Recall, and Detection Delay. Aggregation is split-first over five paired seeds and then across three frozen outers. Bootstrap units are WELL for 3W and Run for TEP. Reports include paired deltas, 95% group-aware CIs, positive/non-worse counts, worst seed/outer, scarcity difference-of-differences, outer-direction consistency, and paired-seed direction win rate. Direction consistency is descriptive and never substitutes for a CI.

Protocol-lock commit, protocol hash, and fraction-manifest hashes are written by the post-commit `lock-record` step before any formal test access. No push is permitted.
