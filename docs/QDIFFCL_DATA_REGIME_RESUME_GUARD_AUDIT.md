# Q-DiffCL Data-Regime Resume Guard Audit

Status before repair: `RUNTIME_GUARD_SELF_LOCKED`; no new formal cell and no outer-test metric was produced by either guard failure.

## Failure replay

1. Historical failure: `numpy._core._exceptions._ArrayMemoryError` requested 3.79 GiB for a `(53, 9600000)` float64 consolidation in the TEP RData loader. This was a host-RAM failure, not CUDA OOM. The memory-safe loader subsequently passed `TEP_MEMORY_SAFE_LOADER_EQUIVALENCE_GO`.
2. Runtime-amendment guard failure: `validate_protocol()` in `scripts/run_qdiffcl_data_regime.py` raised `DATA_REGIME_PROTOCOL_LOCK_HOLD: protocol hash changed`. Expected was the immutable lock hash `f633e84c...`; actual was the post-repair legacy composite hash. The composite mixed scientific inputs with runner, audit, summary, and test implementation files, so this mismatch was a runtime-implementation change rather than a scientific-protocol change.
3. Run-manifest guard failure: after wiring the amendment into `validate_protocol()`, `build_run_manifest()` raised `run manifest protocol hash mismatch`. The existing manifest correctly remained anchored to `f633e84c...`, while the guard incorrectly compared it with the current legacy composite runtime hash.

## Root cause

The legacy `protocol_hash()` combined immutable scientific inputs and mutable runtime implementation. Updating the guard changed the runner hash again, creating a self-reference cycle: amendment records runner hash, runner is edited to validate amendment, and the recorded hash becomes stale. The formal manifest independently and correctly retained the original protocol-lock hash.

## Repair boundary

- The original protocol-lock manifest and commit `5d88cb9d559ab9353df0b835e2d39f6f9966d77f` remain immutable.
- Frozen config, fraction manifests, outer splits, D/E/S weights, rho grid, budgets, seeds, and `test_used_for_selection=false` remain guarded.
- `scientific_inputs_hash` covers only the frozen config and six fraction/outer manifests.
- `runtime_implementation_hash` separately covers the TEP loader and formal runner/guard implementation.
- The approved runtime amendment must bind the parent lock, original scientific protocol hash, scientific-input hash, exact runtime files, equivalence evidence, and test blindness.
- Existing completed cells and the existing formal manifest are not rewritten. New cells record the original scientific lock plus runtime amendment ID and runtime implementation hash.

This separation does not bypass or weaken a scientific guard. It prevents engineering-only changes from redefining the immutable scientific protocol while making the approved numerical-equivalence amendment explicit and auditable.
