# Q-DiffCL Paper Evidence Chain Summary

## Frozen method

FINAL_QDIFFCL remains `0.5D + 0.5E`, critical ratio `0.30`, selective timesteps `1/5`, soft channel-frequency allocation, TCN, Hard SupCon, Original batching and frozen Linear Probe. DCBR remains a validation-calibrated domain-level scalar (`3W rho=1`, `TEP rho=.75` as development references) with zero inference parameters. SVR remains `NO_GO_SVR` and is excluded from the final method.

## Evidence scope after A–E

- Core mechanism: 3-seed validation-only Uniform/Hard/Soft/unmatched ablation; Soft matched is strongly supported on 3W but not on TEP.
- Semantic components: D is the primary discriminative contributor; E is complementary, not equally necessary.
- Contrastive necessity: 3W shows a positive 2×2 interaction (`+0.1483`, 3/3 positive); TEP shows an inverse interaction (`-0.0206`, 0/3 positive). The complementarity claim is dataset-dependent.
- Ratio sensitivity: 3W has a 0.30–0.40 plateau with 0.20 weaker; TEP has 0.30 as a local trough. The frozen value is not reopened and is not claimed universally optimal.
- Practicality: dual-dataset training time, peak GPU memory, parameters and 3-repeat lightweight benchmark are complete. DCBR adds zero inference parameters; standalone FRERA augmentation timing remains unavailable.
- Early-fault trajectory: existing seed-7 checkpoints replayed across all 40 TEP fault runs with onset alignment and run bootstrap bands; no representative-case cherry-picking.
- External audit: AutoDA-Timeseries is method-native supplementary only; no verifiable industrial diffusion+contrastive DiCL implementation was found, so it is not ranked.
- Industrial analysis: existing development checkpoints replayed per WELL/fault with group bootstrap; these are not untouched paper-final results.
- Robustness gaps: limited-data performance and broader missingness cells remain unsupported.

## Paper-final freeze

The nested grouped protocol again passed dry-run, full repository tests passed (`264 passed`), and content hashes were recorded for 2,229 3W data files and four TEP RData files. `docs/paper_final_freeze.md` is the auditable pre-outer snapshot. At that freeze point, no outer model or outer metric had been produced; the completed outer evaluation is recorded below.

## Reporting boundary

The paper may claim dataset-dependent mechanism evidence, completed frozen nested/grouped outer evaluation, dual-dataset sensitivity characterization, efficiency evidence, and TEP development onset trajectories. The outer results support only dataset-specific generalization statements. The paper may not claim universal Soft superiority, universal cross-WELL gains, 0.30 as a universal optimum, completed limited-data robustness, or unconditional performance superiority.

See `docs/paper_evidence_matrix.md`, `docs/paper_final_protocol.md`, `docs/paper_final_freeze.md`, and raw CSV/JSON under `analysis/results/`, `docs/paper_evidence/`, and `outputs/`.

## Paper-final outer evaluation

The frozen nested/grouped outer matrix is complete. Statistics retain WELL/Run grouping and aggregate model seeds inside each split before the three-split summary.

- 3W: FINAL Macro-F1 `0.3216`; DCBR `0.3204`; paired DCBR-FINAL `-0.0012` (95% CI `-0.0290` to `+0.0118`).
- TEP: FINAL Macro-F1 `0.9483`; DCBR `0.9486`; paired DCBR-FINAL `+0.0003` (95% CI `-0.0013` to `+0.0020`).

Claim categories and limitations are frozen in `docs/paper_final_claims.md`.
