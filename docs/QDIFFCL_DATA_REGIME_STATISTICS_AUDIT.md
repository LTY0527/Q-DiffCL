# Q-DiffCL Data-Regime Statistics Audit

Status: `PAIRED_ESTIMANDS_DISAMBIGUATED`; no model was retrained.

The former CSV placed a cell-level paired point estimate beside a group-bootstrap interval. Those quantities use different estimands:

- `paired_delta`: mean of paired outer-test aggregate Macro-F1 differences over outer/seed cells;
- group bootstrap: outer-first mean of per-WELL/per-Run Macro-F1 differences, resampling outer and group units.

Macro-F1 is nonlinear, and equal group weighting is not equivalent to recomputing aggregate Macro-F1 from pooled windows. Consequently a cell-level point estimate could legitimately fall outside an interval centered on the group-level estimand. Calling that interval the confidence interval of `paired_delta` was ambiguous.

The summarizer now retains the preregistered aggregate `paired_delta`, adds the matching `group_macro_f1_delta` point estimate, and names its interval `group_bootstrap_ci_low/high`. Scarcity DoD receives the same explicit separation. The bootstrap algorithm, repeats, seed, group units, completed model artifacts, and scientific protocol are unchanged.

Example at 3W 100%, `CALIBRATED_RHO - FINAL_QDIFFCL_FIXED`:

- aggregate paired delta: `+0.028442`;
- group-level point estimate: `-0.007962`;
- group-bootstrap interval for the group-level estimand: `[-0.030613, +0.008006]`.

The corrected labels prevent the interval from being attributed to the different aggregate point estimate. Current evidence remains partial because TEP formal cells are absent.
