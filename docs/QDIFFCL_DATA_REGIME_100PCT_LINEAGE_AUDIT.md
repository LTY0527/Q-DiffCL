# Q-DiffCL Data-Regime 100% Lineage Audit

Status: `PROVENANCE_EXPLAINED_NO_RESULT_SELECTION`.

The Data-Regime 3W 100% results and historical Paper-final/Posthoc full-data results use the same three outer IDs, train/validation/test WELL assignments, five model seeds, raw source universe, frozen 22 features, 4,000/2,000 train/validation windows per class, 20/15 epoch caps, batch size 256, TCN, Hard SupCon, original batching, and outer-test aggregation family. Historical metrics were not reused because compatibility hashes did not match.

The important provenance difference is the source-unit definition used inside train-derived criticality. Paper-final represents 3W windows by WELL in the criticality bundle. Data-Regime defines its scarcity axis at `instance_id` trajectory level and therefore represents even its 100% anchor by instance ID. D and E aggregate by `run_uid`; changing WELL-level aggregation to instance-level aggregation changes D/E estimates and the soft mask despite `S=0`.

For outer 31001:

- Paper-final context hash: `ab66ebc8673b49844c7563328f1dff2f379f86b81f37445013af200dcfca89d7`;
- Data-Regime 100% context hash: `eefe4c06d4ac89e050afc7eb21745708690347149d7c7f63ec4b2411b893290c`;
- Paper-final soft-mask hash: `16f358ea75ad7b24ed2e4c4a92b1a05e4e0ad2eb6bccbf01e11b7777c7d4197f`;
- Data-Regime soft-mask hash: `8802d33e706c776ce76072a7057ac30902ba89618d551b2a1d86640a7a181370`.

The Data-Regime preprocessor additionally records fraction-local empty-channel policy and fraction provenance, so its metadata/context hash is intentionally distinct even at 100%. Augmentation is regenerated under the Data-Regime context and protocol hash rather than borrowed from Posthoc. These differences explain why the 100% numbers are not byte-compatible; neither result family is selected for being more favorable.
