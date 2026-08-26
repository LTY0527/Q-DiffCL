# Q-DiffCL Post-hoc Recent-Baseline Protocol

Status: `POSTHOC_BASELINE_AUDIT_IN_PROGRESS`.

## Provenance boundary

- Frozen paper-final source commit: `276416ff3114ab40a41cf48bbad16e9a7368732d`.
- Frozen paper-final tag: `paper-final-outer-complete`.
- Expansion branch: `exp/posthoc-baseline-expansion`.
- All results created in this branch are `POSTHOC_BASELINE_EVIDENCE`; they do not amend the preregistered Paper-final matrix, method, splits, seeds, results, or claims.
- Baseline selection must be committed before any candidate outer-test metric is read.

## Execution environment

- Project: `E:\Code\Q-DiffCL`.
- Python: `E:\anaconda\envs\qdiffcl\python.exe`.
- Frozen environment at branch creation: PyTorch `2.6.0+cu124`, CUDA `12.4`, NVIDIA GeForce RTX 4060 Laptop GPU.
- Public repositories are inspected under ignored `external_baselines/`; clones, environments, checkpoints, predictions, datasets, and caches are not committed.

## Fairness rules

- Candidate ranking uses only scientific relevance, public provenance, license/code completeness, protocol compatibility, recency, and estimated engineering/GPU cost.
- Validation or outer-test performance is forbidden as a selection or replacement signal.
- Shared-protocol mechanism adaptations and method-native representations are reported in separate tracks.
- Completed Paper-final Q-DiffCL, NO_AUG, UNIFORM_DIFFUSION, and FRERA cells are reused only for matched reporting and are never retrained here.
- A locked method may be replaced only after a documented engineering, dependency, licensing, semantic-adaptation, NaN, provenance, or excessive-cost failure; the next locked fallback is used without consulting outer-test performance.

