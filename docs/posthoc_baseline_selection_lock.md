# Post-hoc Recent-Baseline Selection Lock

Status: `POSTHOC_BASELINE_SELECTION_LOCKED`.

This lock was produced before any candidate outer-test metric. Its ranking uses only the frozen audit in `posthoc_baseline_candidate_matrix.csv`. The selection hash is `07d92df216fcd254cad6caf8b3688dc2a98781586c0bf49b0d35970a56614838`.

## Locked set

| Method | Score | Track | Non-performance selection reason |
|---|---:|---|---|
| TimesURL | 16 | Track B, method-native representation | recent frequency-temporal universal representation; complete MIT code |
| MF-CLR | 16 | Track B, method-native representation | recent multi-frequency representation; classification path and complete MIT code |
| REBAR | 15 | Track B, method-native representation | distinctive reconstruction-based positive pairing; linear probe and complete MIT code |
| AutoTCL | 15 | Track A, mechanism adaptation | directly augmentation-focused and encoder-agnostic; independent adaptation because the repository has no license file |

Four methods are locked because the aggregate native engineering/GPU estimate is high. The set covers four 2024 methods, learned augmentation, frequency-aware learning, positive-pair construction, and recent representation learning without maximizing method count.

## Fallback order

1. TF-C — method-native representation.
2. SoftCLT — independently implemented objective/mechanism adaptation only.
3. TS2Vec — method-native representation.
4. AutoDA-Timeseries — method-native supplementary only.
5. InfoTS — audit-only; usable only if reproducibility and licensing gaps are independently resolved.

A locked method can be replaced only for a documented engineering, dependency, license, semantic-adaptation, NaN, checkpoint/provenance, or excessive-cost failure discovered without reading its outer-test metric. Low validation performance is not a replacement reason. Every replacement is append-only in `docs/posthoc_baseline_selection_amendment.md`.

## Frozen benchmark matrix

- Datasets: 3W and TEP.
- Frozen outer splits: 3 per dataset.
- Seeds: 3W `42, 43, 44`; TEP `7, 42, 2026`.
- Per method: `2 × 3 × 3 = 18` logical cells.
- Locked total: `4 × 18 = 72` logical cells.
- Q-DiffCL, NO_AUG, UNIFORM_DIFFUSION, and FRERA are reused from matched Paper-final rows and not retrained.

