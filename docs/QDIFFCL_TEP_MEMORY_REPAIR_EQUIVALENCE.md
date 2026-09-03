# Q-DiffCL TEP Memory Repair Equivalence

Status: `TEP_MEMORY_SAFE_LOADER_EQUIVALENCE_GO`.

## Exact checks already passed

On the registered `TEP_FaultFree_Training.RData` file, the legacy BlockManager reader and the repaired column-managed reader have exactly equal:

- shape `(250000, 55)`, column order, index, and per-column dtype;
- finite/NaN placement and every numeric value (`rtol=0`, `atol=0` semantics);
- SHA-256 over ordered column bytes: `3091b6aa216b1cc0b1eae15ada9bacfb647294dcf8f4e4fe36f4c5eb2bded395`;
- selected Run count/order, `run_uid`, feature shape/count, sample arrays, fault labels, and onset metadata;
- prefilter-on versus legacy post-group filter Run-array hashes.

A mixed normal/fault fixture separately verifies exact preservation of fault IDs, limits, order, NaNs, and testing onset boundary `161`.

The frozen Data-Regime config and all six fraction manifests are byte-unchanged. The test suite also verifies `0.5D + 0.5E`, `S=0`, critical ratio, timesteps, rho grid, and the TEP 10% hold.

## Full-context checks

TEP 100% and 25% each built/released twice with identical within-fraction train, validation, test-context, ID, finite-count, shape, fraction, D/E input, mask, and context signatures. Against the archived pre-repair TEP 100% outer-32001 evidence, the following are exact:

- context hash: `a5d4ac39ad9f8022949f2f693fecd31f65690a191450498f4de87bc629b3ca38`;
- criticality NPZ SHA-256: `ad6a64bca2e780e3fc9e1bf814dd71a51c6fcd929f6f301f2156cec0ea74db0d`;
- fraction manifest hash, scaler arrays, criticality mask, selected train IDs, validation/test IDs, and window counts.

No outer-test metric was computed or read. The RAM and equivalence gates therefore permit the supervised formal resume.
