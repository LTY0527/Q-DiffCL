# Q-DiffCL Paper Evidence Chain Summary

## Frozen method

FINAL_QDIFFCL remains `0.5D + 0.5E`, critical ratio `0.30`, selective timesteps `1/5`, soft channel-frequency allocation, TCN, Hard SupCon, Original batching and frozen Linear Probe. DCBR remains a validation-calibrated domain-level scalar (`3W rho=1`, `TEP rho=.75`) with zero inference parameters. SVR remains `NO_GO_SVR` and is excluded from the final method.

## Evidence scope

- Core mechanism: new 3-seed validation-only Uniform/Hard/Soft/unmatched ablation; Soft matched is strongly supported on 3W but not on TEP.
- Semantic components: D is the primary discriminative contributor; E is complementary, not equally necessary.
- Industrial analysis: existing development checkpoints replayed per WELL/fault with group bootstrap; these are not untouched paper-final results.
- Robustness: grouped limited-data dry-run and ratio/timestep budget audit completed; missing downstream cells remain explicitly unsupported.
- Generalization: nested grouped Paper-final protocol passed dry-run; no outer model or outer metric has been run.

## Reporting boundary

The paper may claim dataset-dependent mechanism evidence and a TEP over-augmentation mitigation role for DCBR. It may not claim universal Soft superiority, universal cross-WELL gains, completed limited-data robustness, or paper-final generalization until the frozen outer evaluation is executed once.

See `docs/paper_evidence_matrix.md`, `docs/paper_final_protocol.md`, and the raw CSV/JSON under `docs/paper_evidence/`.
