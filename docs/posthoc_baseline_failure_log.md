# Post-hoc Baseline Failure Log

Evidence class: `POSTHOC_BASELINE_EVIDENCE`.

## Pre-outer engineering findings

| Date | Method | Stage | Finding | Resolution | Outer metric read |
|---|---|---|---|---|---|
| 2026-08-27 | TimesURL | sanity | official unnamespaced modules collide with Q-DiffCL; isolated import required | isolated official commit modules; sanity passed | no |
| 2026-08-27 | MF-CLR | sanity | official `full_series` encode pools only the fine branch, then concatenates a length-64 coarse branch | adapter pools both branches as documented full-series representation; sanity passed | no |
| 2026-08-27 | REBAR | sanity | implicit namespace packages, optional tensorboard dependency, and Windows worker spawning block the official path | isolated namespaces, no-op logging, and numeric-neutral `num_workers=0`; sanity passed | no |
| 2026-08-27 | TimesURL | post-sanity cost gate | conservative formal estimate 63.16 h, versus 12–20 h aggregate lock estimate | replaced by first fallback TF-C | no |
| 2026-08-27 | MF-CLR | post-sanity cost gate | conservative formal estimate 104.23 h | replaced by second fallback SoftCLT | no |
| 2026-08-27 | REBAR | post-sanity cost gate | conservative formal estimate 74.78 h | replaced by third fallback TS2Vec | no |

The first three entries are resolved compatibility findings. The final three are the allowed `estimated cost far beyond audit estimate` replacement condition, not numerical-performance failures.

## Formal-run numerical recovery

| Date | Cell | Stage | Finding | Resolution | Outer-test use in resolution |
|---|---|---|---|---|---|
| 2026-08-27 | TEP / outer 32001 / seed 7 / AutoTCL | representation training | the learned sigmoid gate saturated to an exact probability boundary; the following unconstrained `logit` derivative produced a non-finite SupCon update | clamp only the sampling probability before `logit` and add explicit finite audits for gate probability, changed view, both projections, and gradients; saturated-gate regression test added | none; the cell was recovered for 8 epochs using outer-train plus inner-validation only |
| 2026-08-27 | same cell | checkpoint resume audit | the no-test preparation helper initially omitted the train-only critical-mask and scaler fit-group fields used by the formal context hash | reproduce the canonical hash from outer-train only; preserve the first checkpoint under `outputs/posthoc_recent_baselines/diagnostics/` and retrain rather than rewriting binary metadata | none; canonical hash matched before formal resume |

The canonical recovery record has `finite_loss=true`, `outer_test_read=false`, and `outer_test_materialized=false`. The formal runner subsequently performed the cell's single outer evaluation and advanced the manifest from 39 to 40 completed cells. No outer score was used to select or tune the stability fix.
