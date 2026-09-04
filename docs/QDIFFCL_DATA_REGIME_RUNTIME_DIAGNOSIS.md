# Q-DiffCL Data-Regime Runtime Diagnosis

Status: `PROCESS_EXITED_SUCCESS`; artifacts remain resumable.

- Audit time: `2026-09-04T16:35:04.433771+00:00`
- Historical PID 45408 exists: `false`
- Runner command: `E:\anaconda\envs\qdiffcl\python.exe -u -m scripts.run_qdiffcl_data_regime --stage all --device cuda`
- CPU state: process exited
- GPU current: `NVIDIA GeForce RTX 4060 Laptop GPU, 0 %, 128 MiB, 8188 MiB, 53`
- Runtime status timestamp: `2026-09-04T16:34:09.929475+00:00`
- Stdout mtime UTC: `2026-09-03T09:26:07.604392+00:00`
- Stderr mtime UTC: `2026-09-03T16:26:07.196388+00:00`
- Last artifact: `outputs\qdiffcl_data_regime_v1\DATA_REGIME_GENERALIZATION_V1\tep\f025\outer_32002\model_seed_2026\CALIBRATED_RHO\result.json`
- Last formal completed cells: `350`
- Last formal cell: `TEP / 25% / outer 32002 / CALIBRATED_RHO / seed 2026`
- Last validation candidate: `TEP / 100% / outer 32001 / seed 2026 / rho 1.0 (15/15 completed)`
- Latest supervised runner exit code: `0`
- Historical traceback retained: `yes`
- Historical Python exception: `numpy._core._exceptions._ArrayMemoryError`
- Allocation request: `3.79 GiB`, shape `(53, 9600000)`, dtype `float64`
- CUDA OOM: `no`
- Metric NaN exception: `no`; stderr contains expected single-class group metric warnings only
- KeyboardInterrupt/external termination: `no evidence`
- Test-read guard violation: `no`
- Manifest/hash failure: `no`
- Failure count: `3`

- Remaining: `25` formal cells and `15` rho candidates
- Remaining GPU time: not estimated by this artifact audit

The historical RAM failure occurred before any TEP outer-test result. The memory-safe runtime amendment is now active; the latest supervised stage status above is authoritative for current execution state. This audit itself does not restart the runner.
