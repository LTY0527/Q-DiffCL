# H1.5 interruption recovery and finalization audit

Audit date: 2026-08-28

## Recovery decision

- Branch: `exp/posthoc-baseline-expansion`.
- H1 archive commit: `b3aee5a2ad0cdc0634d3ef67a86e49c2b37489c4`.
- H1.5 protocol-lock commit: `2e5e1544e75303d50253c0c05c06918a4d7b30a6`.
- Frozen protocol hash: `89cab7e6c6b3e9a127ee51cbec28ba6c4a730013139c7d3cc4e30664cd199b51`.
- No H1.5 runner remained active at audit time.
- Disk and manifest validation found all 48/48 extension cells complete: 24/24 on 3W and 24/24 on TEP, with no failures, duplicates, or orphan results.
- The UI-interrupted cell `tep-outer32001-seed43-ts2vec` had complete result, prediction, checkpoint, provenance, and hashes. It was accepted as complete and was not retrained.
- No checkpoint resume and no single-cell restart were required. All 48 validation records report a fresh original execution (`resumed=false`).

## Result-independent finalization amendment

The prepared summarizer used `row.get("evidence_source", source_by_id[row["run_id"]])`. Python evaluates the default expression eagerly, so a frozen Paper-final reference row could raise `KeyError` even though that row already supplied `evidence_source`.

The lookup was changed to an explicit conditional that reads `source_by_id` only when `evidence_source` is absent. A regression test covers a frozen Paper-final reference whose run ID is intentionally absent from the H1/extension source map. Corrupted report-label glyphs are normalized back to `±`, `Δ`, and `×` before writing; this affects presentation only. The summarizer also accepts its own final `POSTHOC_BASELINE_5SEED_EXTENSION_COMPLETE` state, making report regeneration idempotent without reopening training or outer evaluation.

- Prepared summarizer SHA-256 recorded in the manifest: `c177f3027baaa920f4a6f239e4db5c1c104bc0d7ffe7a8becbcc1c9ba22bff37`.
- Amended summarizer SHA-256: `b21f9d6305d0bcff26a3899204bac64f1e14e10a36417a3baa4d065ba4e7a6ab`.
- Scientific protocol hash remains unchanged.
- This amendment changes no training, split, seed, metric, grouping unit, bootstrap repeat, model selection, or outer-test semantics.

The amendment is append-only provenance for finalization and does not retroactively alter any completed cell artifact.
