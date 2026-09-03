# Q-DiffCL Data-Regime Evidence Lineage

- Base branch: `exp/posthoc-baseline-expansion`
- Base commit: `ea7909987998a865b7cfdf9467465e8c13ea288c`
- Development branch: `exp/qdiffcl-data-regime`
- Frozen outer source: `E:/Code/Q-DiffCL/outputs/paper_final_protocol/dry_run_manifest.json`
- Frozen outer source SHA-256: `e4ef68e85b57f0ae90359f8ad7a83e2949a363fdff3945297c44bd102bd66207`
- Paper-final result manifest exists: `True`
- Paper-final output source exists: `True`

## Frozen config hashes

- `configs/qdiffcl_final.yaml`: `e50c904484b28ef90fe7da3c774c6b127796861e0bc5de02854960cc4dbe8184`
- `configs/qdiffcl_final_5seed.yaml`: `051739e9822987f5088481607e288ab5c51294f891e292963958446b8fd54034`
- `configs/domain_calibrated_budget_routing.yaml`: `88d5b09046500f38c714afd40d06210da4fd0200a52e64124594336f15f65656`
- `configs/domain_calibrated_budget_routing_final.yaml`: `13f646d30448e9f7a440b5aebc2a236d553401cd5efc1fe9278bbf36fa6bf817`
- `configs/paper_final_outer.yaml`: `90bc91b97fd643323064f2352cf39ef23c004f62430b89dd68d8c9f6b8b7a8a8`
- `E:/Code/Q-DiffCL/outputs/paper_final_protocol/dry_run_manifest.json`: `e4ef68e85b57f0ae90359f8ad7a83e2949a363fdff3945297c44bd102bd66207`
- `utils.py`: `cee7ca5a9a23ad788d9899fbb0f270630fe919d5a76f9cb64061211dac1758da`
- `degradations/__init__.py`: `9a8504c702fbb6fd9a02b8fb8add1e0418e2a9f37f22b8d36661373c3e41b13e`
- `degradations/core.py`: `5fdb30b79cf156ed68faab788ffbc3bfaec5e8f71ebc78100b88bf86a7acf141`
- `utils.py` and `degradations/` are tracked, source-equivalent archival implementations of previously untracked runtime dependencies; their new-worktree hashes are listed above.

## Integrity

- FINAL_QDIFFCL is frozen at `0.5D + 0.5E`; `S=0`.
- Historical DCBR is validation-only with global rho 3W=1.00 and TEP=0.75.
- Data-Regime rho is a distinct outer-specific validation-only selection.
- The split source file contains completed historical outer metrics. The generator reads only its `three_w`/`tep` group records; the new selector is forbidden from reading historical or current test metrics.

## Fraction manifests

- 3W outer 31001: `20ea9e6a68364802a4fee1c8fd04e012e9e3d9866e3ef120aabc3a1c471e0e2f`
- 3W outer 31002: `cd30c5b396a9e96215997c1835daa2b74c356843256df20785d53103e4cdb67e`
- 3W outer 31003: `311171432bb13ccf8c5976a530d671c4c9dd5747abd9c4403a9acfa342226933`
- TEP outer 32001: `aec9d92e99472f21ba474e0096b962d8a75bf0d3c43f9e73c947d1be9ba543a9`
- TEP outer 32002: `cb2995360d5da434172afe159995f7ff04383d4e952914eb9010a2949bf1be3c`
- TEP outer 32003: `89b39d43583b85cb20661c6a418834cfe3454b3a8eb090d172e841a3813f1c2e`

## 100% reuse status

Historical 100% metrics are lineage context only at protocol construction time. Exact reuse remains disabled until the runner proves matching train subset, preprocessing, training-budget, context, checkpoint, prediction, and protocol hashes for each cell. The new outer-specific rho selection also makes historical global DCBR non-interchangeable with CALIBRATED_RHO.
