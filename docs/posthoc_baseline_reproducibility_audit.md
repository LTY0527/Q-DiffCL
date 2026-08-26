# Recent Time-series Baseline Reproducibility Audit

Status: `POSTHOC_BASELINE_AUDIT_COMPLETE`.

Audit date: 2026-08-27. Selection evidence is restricted to public provenance, scientific scope, code/license completeness, protocol compatibility, and estimated cost. No candidate outer-test metric was read or generated.

## Repositories and immutable revisions

| Rank | Method | Venue/year | Official revision | License | Score | Fairness class |
|---:|---|---|---|---|---:|---|
| 1 | TimesURL | AAAI 2024 | `d3533e45cb28efe8c986f13ce8d80926d0e9254e` | MIT | 16 | `METHOD_NATIVE_REPRESENTATION` |
| 2 | MF-CLR | ICML 2024 | `c40fc8d265947f7a194ac43b8256c2b5d9febe01` | MIT | 16 | `METHOD_NATIVE_REPRESENTATION` |
| 3 | REBAR | ICLR 2024 | `74cd46b56262488378f49ebe6ea40ee59ff577dc` | MIT | 15 | `METHOD_NATIVE_REPRESENTATION` |
| 4 | AutoTCL | ICLR 2024 | `2ca00603734d9d339e74f42b22633df6c91c6256` | no license file | 15 | `MECHANISM_ADAPTATION_ONLY` |
| 5 | TF-C | NeurIPS 2022 | `96675826e9ef234a9b01cc63d484c66cb0441bc0` | MIT | 14 | `METHOD_NATIVE_REPRESENTATION` |
| 6 | SoftCLT | ICLR 2024 | `14c638979b129075d7a1111e9f529b9a275ea394` | no license file | 14 | `MECHANISM_ADAPTATION_ONLY` |
| 7 | TS2Vec | AAAI 2022 | `b0088e14a99706c05451316dc6db8d3da9351163` | MIT | 14 | `SHARED_PROTOCOL_COMPATIBLE` |
| 8 | AutoDA-Timeseries | ICLR 2026 | `91dbf70b54b255214b7f204d8d9f70d26f9c1fe3` | no license file | 13 | `METHOD_NATIVE_REPRESENTATION` |
| 9 | InfoTS | AAAI 2023 | `11775205159e87e767aaa61c23e64ae4ef11c6fd` | no license file | 12 | `NOT_FAIRLY_REPRODUCIBLE` |

The complete field-level audit is in `analysis/results/posthoc_baseline_candidate_matrix.csv`.

## Method findings

- **TimesURL** provides training, classification, forecasting, imputation and anomaly code, accepts the TS2Vec-style `[N,T,C]` representation interface, and combines temporal interpolation/masking with frequency-aware contrastive representation learning. Its official MIT code is traceable and adaptable to frozen grouped IDs.
- **MF-CLR** provides an MIT-licensed multi-frequency encoder, classification entry point and bundled comparison implementations. The native repository is broad and its experiment script mixes several downstream choices, so the Q-DiffCL adapter must isolate MF-CLR itself and record every deviation.
- **REBAR** provides an MIT-licensed modular SSL framework and explicit linear-probe classification. Its retrieval-by-reconstruction positive-pair semantics are relevant and distinctive, but the official environment is Python 3.8/PyTorch 2.0/CUDA 11.7 and native adaptation is high cost.
- **AutoTCL** supplies the parametric augmentation network, encoder training code and a classification task module and explicitly describes the augmentation as encoder-agnostic. The repository has no license file and an over-inclusive pinned requirements list, so only an independently implemented and explicitly named mechanism adaptation is eligible for Track A.
- **TF-C** is a complete MIT-licensed time/frequency representation baseline with classification transfer support. Its older multi-environment workflow makes it the first fallback rather than a primary night-run method.
- **SoftCLT** has complete TS2Vec and CA-TCC classification paths, but no repository license or consolidated environment. It remains a mechanism-adaptation fallback, not an unqualified official reproduction.
- **TS2Vec** is the lowest-cost, well-licensed fallback with a stable `[N,T,C]` API and linear classification, but is older and less targeted to the recent-baseline coverage gap.
- **AutoDA-Timeseries** retains Catch22-conditioned policy learning and a native supervised downstream objective. Removing those components would change the method, so it remains method-native supplementary only and is not mixed into the shared-protocol table.
- **InfoTS** exposes its source only through a zip with a minimal README and no license. It is audit-only under this protocol.

## Audit-only names

No uniquely verifiable official repository was introduced for AutoCL, CAAP, or DiCL. They are not treated as official implementations and are excluded from training.

## Fixed scoring

Total score is the sum of scientific relevance (0–4), reproducibility (0–4), protocol compatibility (0–4), recency/venue (0–4), and engineering cost (0–2, where higher is cheaper). Ties are ordered by scientific relevance, reproducibility, protocol compatibility, recency, engineering cost, then method name. Scores never use 3W/TEP validation or outer-test performance.

