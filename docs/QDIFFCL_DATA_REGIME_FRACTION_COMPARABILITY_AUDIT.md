# Q-DiffCL Data-Regime Fraction Comparability Audit

Status: `NESTED_SOURCE_UNITS_CONFIRMED_WITH_COMPOSITION_CONFOUND`.

All six dataset/outer manifests satisfy strict `10% ⊂ 25% ⊂ 100%` nesting. Sampling is deterministic, seed-independent, and performed on source units before windowing. Detailed per-class and per-group counts are stored in `qdiffcl_data_regime_fraction_composition.csv` and `qdiffcl_data_regime_fraction_groups.csv`.

This experiment varies source-unit diversity, not proportional window count. The training pipeline retains fixed per-class window caps. Minority fault trajectories are protected by class-stratified minimum-unit rules, so their retention rate is much higher than the normal-class retention rate. For example, 3W outer 31001 changes from full class counts `188/4/4/38` to 10% counts `17/2/2/2` for original classes `0/2/8/9`.

The completed 3W NO_AUG per-class means show that its 10% improvement is concentrated in some classes rather than uniform:

| Original class | 100% F1 | 25% F1 | 10% F1 |
|---:|---:|---:|---:|
| 0 | 0.460504 | 0.434378 | 0.562940 |
| 2 | 0.128455 | 0.292440 | 0.282266 |
| 8 | 0.541186 | 0.582274 | 0.562475 |
| 9 | 0.166446 | 0.148386 | 0.117971 |

Thus the regime axis combines reduced independent trajectory diversity with a class-composition shift imposed by the preregistered stratification policy. This is a real interpretive confound and must accompany scarcity claims. Frozen manifests are not modified, and no more favorable subset is selected.
