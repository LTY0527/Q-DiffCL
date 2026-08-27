# Recent-baseline five-seed extension protocol

Status: `POSTHOC_BASELINE_5SEED_EXTENSION_PREPARED`

Evidence class: `POSTHOC_BASELINE_5SEED_EXTENSION`.

This extension was locked after the complete H1 three-seed benchmark and before any new seed 45/46/43/44 outer-test metric was produced. It aligns the four H1 baselines with the frozen five-seed Paper-final protocol. The direction or magnitude of the H1 results was not used to retain, remove, or replace a method.

## Frozen scope

- H1 archive commit: `b3aee5a2ad0cdc0634d3ef67a86e49c2b37489c4`.
- Paper-final source commit: `276416ff3114ab40a41cf48bbad16e9a7368732d`.
- Methods: AutoTCL, SoftCLT, TF-C, TS2Vec; no additions or removals.
- Track A mechanism adaptations: AutoTCL and SoftCLT.
- Track B method-native representations: TF-C and TS2Vec.
- Track B is representation-level context and cannot support augmentation-only causal claims.
- Existing H1 cells are reused, never retrained: 72 cells from three outer splits and three seeds.
- New cells: exactly 48, isolated under `outputs/posthoc_baseline_5seed_extension`.

## Frozen split and seed matrix

| Dataset | Outer splits | H1-complete seeds | Missing seeds added here | Full five-seed set |
|---|---|---|---|---|
| 3W | 31001, 31002, 31003 | 42, 43, 44 | 45, 46 | 42, 43, 44, 45, 46 |
| TEP | 32001, 32002, 32003 | 7, 42, 2026 | 43, 44 | 7, 42, 43, 44, 2026 |

For every method, the extension adds six 3W and six TEP cells. Four methods therefore add `4 × 12 = 48` cells. The grouped Paper-final split definitions remain unchanged: WELL is the 3W grouping unit and Run is the TEP grouping unit.

## Numerical and provenance invariants

The extension runner imports the archived H1 training/evaluation functions. It does not modify the adapter, model structure, objective, preprocessing, epochs, batch size, probe, threshold selection, or outer evaluation semantics. The extension config locks SHA-256 values for the H1 adapter, config, runner, H1 manifest/raw result, and Paper-final config/raw result. Official source commits remain:

| Method | Official source commit | Execution boundary |
|---|---|---|
| AutoTCL | `2ca00603734d9d339e74f42b22633df6c91c6256` | independent shared-TCN mechanism adaptation; not an official reproduction |
| SoftCLT | `14c638979b129075d7a1111e9f529b9a275ea394` | shared-TCN soft-objective mechanism adaptation; not an official reproduction |
| TF-C | `96675826e9ef234a9b01cc63d484c66cb0441bc0` | method-native dual time/frequency representation |
| TS2Vec | `b0088e14a99706c05451316dc6db8d3da9351163` | method-native hierarchical representation |

Every new cell has a deterministic run ID, checkpoint/validation pair, prediction hash, checkpoint hash, protocol hash, protocol-lock commit, and outer-test-once marker. Existing complete results are hash-validated and reused; partial artifact pairs fail closed.

## Evaluation and statistics lock

- Formal interpreter: `E:\anaconda\envs\qdiffcl\python.exe`; CUDA is mandatory.
- Test is evaluated only after training and validation-only selection complete.
- Original Paper-final methods are reused from frozen raw results and are not retrained.
- The unified result first averages five model seeds within each outer split and then reports mean ± sample SD across the three splits.
- Paired comparisons use the same dataset, split, and seed.
- Group-aware bootstrap uses 2,000 resamples: WELL for 3W and Run for TEP; windows are never treated as independent.
- Reports include effect size, 95% CI, positive/non-worse cell counts, and worst cell/seed.

The complete reference extraction includes FINAL_QDIFFCL, NO_AUG, UNIFORM_DIFFUSION, FRERA, JITTER, SCALING, JITTER_SCALING, and DCBR. All four recent baselines remain in the final report regardless of whether their five-seed result is above or below Q-DiffCL.

Protocol hash is stored in `analysis/results/posthoc_baseline_5seed_extension_manifest.json` and is computed only from the frozen scientific/provenance fields in the extension config.
