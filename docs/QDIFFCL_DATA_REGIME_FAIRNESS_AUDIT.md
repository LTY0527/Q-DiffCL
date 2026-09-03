# Q-DiffCL Data-Regime Fairness Audit

Current status: `DATA_REGIME_SANITY_GO`.

The six frozen outer manifests were copied into a dedicated namespace with their source hash and unchanged train/validation/test group identities. Training subsets are selected at 3W instance and TEP Run level before windowing. Selection is deterministic and independent of model seed.

The runner fits normalization and D/E criticality from the selected fraction only. Validation windows remain fixed by design; test data are not materialized during manifest generation, lineage audit, rho selection, or either smoke. D/E weights, S, timesteps, critical ratio, training caps, patience, batch sizes, learning rates, and seeds are invariant across fractions and methods.

For a frozen 3W channel with zero finite observations in a low-data subset, the registered fallback is an all-zero neutral normalized channel. The channel remains in the model input and its name is written to the context audit; statistics from the unused train pool are never imported.

TEP 10% is a preregistered scientific hold because 25 total Runs cannot provide two independent units for each of 20 E-required fault classes while remaining a 10% subset. It is excluded from the primary D+E matrix rather than repaired post hoc.

Tracked, source-equivalent archival implementations of the previously untracked `utils.py` and `degradations/` runtime dependencies make the new worktree self-contained without modifying the originals. Their new-worktree SHA-256 values are recorded in lineage; byte identity is not claimed because formatting was normalized when the files were added.

Both validation-only smoke tests passed, including immutable resume of the rho candidate. Full pytest passed (317 tests), `git diff --check` passed, and expected cell accounting is 375 formal plus 225 validation candidates. Formal execution still requires the local protocol-lock commit and matching post-commit protocol hash.
