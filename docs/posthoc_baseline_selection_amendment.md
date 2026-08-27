# Post-hoc Baseline Selection Amendment

This document is append-only. All amendments are based on engineering evidence collected before any candidate outer-test metric was materialized or read. The original selection lock and hash remain unchanged.

## 2026-08-27 — native runtime correction

Evidence class: `POSTHOC_BASELINE_EVIDENCE`.

All eight original sanity cells passed finite-loss, GPU, representation-shape, grouped-leakage, linear-probe, and checkpoint round-trip checks. Each used only outer-train plus inner-validation; `outer_test_materialized=false` and `outer_test_metric_read=false` for every row. Validation scores were not used in the decision below.

The audit estimated 12–20 GPU hours for the whole 72-cell benchmark. Measured 512-window sanity runtimes were extrapolated using only full train-window count, frozen epoch count, and nine formal cells per dataset:

| Locked method | 3W estimate | TEP estimate | Total | Engineering disposition |
|---|---:|---:|---:|---|
| TimesURL | 14.87 h | 48.29 h | 63.16 h | replace: cost far beyond audit estimate |
| MF-CLR | 54.63 h | 49.60 h | 104.23 h | replace: cost far beyond audit estimate |
| REBAR | 19.99 h | 54.79 h | 74.78 h | replace: cost far beyond audit estimate |

The extrapolation is deliberately simple and performance-blind. Even large sublinear speedups would not bring the three native methods together inside the locked night-run budget. Their successful sanity records remain in `analysis/results/posthoc_baseline_sanity.csv` as reproducibility evidence.

Fallbacks are consumed in the pre-locked order:

1. TimesURL → TF-C, Track B method-native representation.
2. MF-CLR → SoftCLT, Track A independently implemented mechanism adaptation.
3. REBAR → TS2Vec, Track B method-native representation.

The active four-method set is therefore TF-C, SoftCLT, TS2Vec, and AutoTCL. Each replacement must pass the same two-dataset train/validation-only sanity gate before formal evaluation. No outer-test metric existed at this amendment.
