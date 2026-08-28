# Recent Time-series Baselines: Post-hoc Fair Benchmark

Status: `POSTHOC_BASELINE_BENCHMARK_COMPLETE`

Evidence class: `POSTHOC_BASELINE_EVIDENCE`. These results were produced after the frozen Paper-final evaluation and are not preregistered Paper-final evidence.

## Candidate audit and selection lock

The selection was performance-blind and completed before candidate outer metrics. The original selection hash remains `07d92df216fcd254cad6caf8b3688dc2a98781586c0bf49b0d35970a56614838`.

| Rank | Method | Score | Locked | Final disposition |
|---:|---|---:|---|---|
| 1 | TimesURL | 16 | yes | TF-C |
| 2 | MF-CLR | 16 | yes | SoftCLT |
| 3 | REBAR | 15 | yes | TS2Vec |
| 4 | AutoTCL | 15 | yes | retained |
| 5 | TF-C | 14 | no | active fallback |
| 6 | SoftCLT | 14 | no | active fallback |
| 7 | TS2Vec | 14 | no | active fallback |
| 8 | AutoDA-Timeseries | 13 | no | unconsumed fallback rank 4 |
| 9 | InfoTS | 12 | no | unconsumed fallback rank 5 |

The append-only cost amendment replaced TimesURL, MF-CLR, and REBAR by TF-C, SoftCLT, and TS2Vec respectively. Conservative estimates were 63.16 h, 104.23 h, and 74.78 h; no validation or outer score was used. AutoTCL was retained. See `posthoc_baseline_selection_amendment.md` and `posthoc_baseline_failure_log.md`.

## Comparison boundary

| Track | Methods | Interpretation |
|---|---|---|
| Track A | AutoTCL, SoftCLT; frozen Paper-final references | shared-backbone/mechanism comparison; direct paired comparison is permitted |
| Track B | TF-C, TS2Vec | method-native representation context; do not claim augmentation-only causality |

## Split-first results

Each outer-split value first averages its three matched model seeds; the table reports mean ± sample SD across the three frozen outer splits.

| Dataset | Track | Method | Macro-F1 | AUPRC | FAR | Early recall | Delay | Worst cell |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 3W | TRACK_B_METHOD_NATIVE_REPRESENTATION | TF-C | 0.3391 ± 0.1296 | 0.7377 | 0.7065 | 0.9822 | 183.2275 | 0.1818 |
| 3W | TRACK_A_MECHANISM_ADAPTATION | SoftCLT | 0.3324 ± 0.0203 | 0.7857 | 0.6067 | 0.9059 | 165.1648 | 0.2382 |
| 3W | TRACK_B_METHOD_NATIVE_REPRESENTATION | TS2Vec | 0.2305 ± 0.0375 | 0.4714 | 0.7305 | 0.7747 | 1428.8206 | 0.0771 |
| 3W | TRACK_A_MECHANISM_ADAPTATION | AutoTCL | 0.3537 ± 0.0715 | 0.7782 | 0.7585 | 0.9108 | 225.5951 | 0.2488 |
| 3W | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | FINAL_QDIFFCL | 0.3045 ± 0.0938 | 0.7885 | 0.6979 | 0.9103 | 1195.9356 | 0.1256 |
| 3W | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | NO_AUG | 0.2851 ± 0.1105 | 0.7792 | 0.6478 | 0.8433 | 1061.3377 | 0.1091 |
| 3W | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | UNIFORM_DIFFUSION | 0.3018 ± 0.1146 | 0.7898 | 0.7010 | 0.9070 | 1321.0909 | 0.1285 |
| 3W | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | FRERA | 0.3265 ± 0.1066 | 0.7610 | 0.6509 | 0.8590 | 724.8668 | 0.1402 |
| TEP | TRACK_B_METHOD_NATIVE_REPRESENTATION | TF-C | 0.8031 ± 0.0186 | 0.8892 | 0.1049 | 0.6653 | 101.1570 | 0.7759 |
| TEP | TRACK_A_MECHANISM_ADAPTATION | SoftCLT | 0.7496 ± 0.0140 | 0.8409 | 0.0907 | 0.5222 | 143.5543 | 0.7316 |
| TEP | TRACK_B_METHOD_NATIVE_REPRESENTATION | TS2Vec | 0.7696 ± 0.0112 | 0.8612 | 0.1527 | 0.5771 | 121.9808 | 0.7535 |
| TEP | TRACK_A_MECHANISM_ADAPTATION | AutoTCL | 0.9331 ± 0.0270 | 0.9794 | 0.0135 | 0.8340 | 87.6405 | 0.9017 |
| TEP | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | FINAL_QDIFFCL | 0.9457 ± 0.0275 | 0.9836 | 0.0228 | 0.8778 | 84.1182 | 0.9012 |
| TEP | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | NO_AUG | 0.9450 ± 0.0275 | 0.9836 | 0.0204 | 0.8708 | 85.3007 | 0.9027 |
| TEP | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | UNIFORM_DIFFUSION | 0.9434 ± 0.0279 | 0.9828 | 0.0238 | 0.8722 | 85.3173 | 0.9024 |
| TEP | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | FRERA | 0.9504 ± 0.0209 | 0.9891 | 0.0179 | 0.8785 | 86.4200 | 0.9099 |

## Paired group-aware bootstrap versus FINAL_QDIFFCL

Positive Δ means the row method is above FINAL_QDIFFCL. Resampling uses WELL on 3W and Run on TEP; windows are never treated as independent.

| Dataset | Track | Method | Δ Macro-F1 | 95% CI | Above cells | Non-worse cells |
|---|---|---|---:|---:|---:|---:|
| 3W | TRACK_B_METHOD_NATIVE_REPRESENTATION | TF-C | +0.0347 | [-0.0401, +0.1075] | 5/9 | 5/9 |
| 3W | TRACK_A_MECHANISM_ADAPTATION | SoftCLT | +0.0280 | [-0.0716, +0.0741] | 6/9 | 6/9 |
| 3W | TRACK_B_METHOD_NATIVE_REPRESENTATION | TS2Vec | -0.0739 | [-0.1629, -0.0316] | 2/9 | 2/9 |
| 3W | TRACK_A_MECHANISM_ADAPTATION | AutoTCL | +0.0492 | [-0.0316, +0.0929] | 6/9 | 6/9 |
| 3W | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | NO_AUG | -0.0193 | [-0.0377, +0.0139] | 4/9 | 4/9 |
| 3W | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | UNIFORM_DIFFUSION | -0.0027 | [-0.0236, +0.0088] | 5/9 | 5/9 |
| 3W | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | FRERA | +0.0221 | [-0.0613, +0.0534] | 6/9 | 6/9 |
| TEP | TRACK_B_METHOD_NATIVE_REPRESENTATION | TF-C | -0.1427 | [-0.1609, -0.1274] | 0/9 | 0/9 |
| TEP | TRACK_A_MECHANISM_ADAPTATION | SoftCLT | -0.1961 | [-0.2208, -0.1751] | 0/9 | 0/9 |
| TEP | TRACK_B_METHOD_NATIVE_REPRESENTATION | TS2Vec | -0.1761 | [-0.1961, -0.1588] | 0/9 | 0/9 |
| TEP | TRACK_A_MECHANISM_ADAPTATION | AutoTCL | -0.0127 | [-0.0198, -0.0060] | 4/9 | 4/9 |
| TEP | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | NO_AUG | -0.0007 | [-0.0022, +0.0006] | 3/9 | 3/9 |
| TEP | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | UNIFORM_DIFFUSION | -0.0023 | [-0.0043, -0.0007] | 4/9 | 4/9 |
| TEP | TRACK_A_FROZEN_PAPER_FINAL_REFERENCE | FRERA | +0.0046 | [-0.0004, +0.0100] | 4/9 | 4/9 |

## Explicit above/below accounting

- 3W: above Q-DiffCL by point estimate: TF-C, SoftCLT, AutoTCL, FRERA; below: TS2Vec, NO_AUG, UNIFORM_DIFFUSION. Statistical uncertainty is governed by the CI table, not the sign alone.
- TEP: above Q-DiffCL by point estimate: FRERA; below: TF-C, SoftCLT, TS2Vec, AutoTCL, NO_AUG, UNIFORM_DIFFUSION. Statistical uncertainty is governed by the CI table, not the sign alone.

## Remaining coverage gaps

- The active benchmark covers two datasets, three grouped outer splits, and three matched seeds; it is post-hoc evidence, not a new preregistration.
- Track B provides representation-level context and cannot isolate augmentation causality.
- TimesURL, MF-CLR, and REBAR retain successful sanity evidence but lack formal outer results due to the predeclared non-performance cost rule.
- AutoDA-Timeseries remains method-native supplementary only; InfoTS remains audit/fallback coverage.
- No low-data study, third dataset, or broader missingness robustness evaluation was added in H1.

## Five-seed extension

The four active recent baselines were subsequently completed to the frozen Paper-final five-seed sets under `POSTHOC_BASELINE_5SEED_EXTENSION`. See `posthoc_recent_baselines_5seed.md`; the original H1 three-seed results and evidence classification remain unchanged.
