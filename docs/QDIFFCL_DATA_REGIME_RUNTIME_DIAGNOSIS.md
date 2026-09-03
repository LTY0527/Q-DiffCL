# Q-DiffCL Data-Regime Runtime Diagnosis

Status: `PROCESS_EXITED_FAILURE`; artifacts remain resumable.

- Audit time: `2026-09-03T19:21:35.241004+00:00`
- Historical PID 45408 exists: `false`
- Runner command: `E:\anaconda\envs\qdiffcl\python.exe -u -m scripts.run_qdiffcl_data_regime --stage all --device cuda`
- CPU state: process exited
- GPU current: `NVIDIA GeForce RTX 4060 Laptop GPU, 0 %, 341 MiB, 8188 MiB, 55`
- Runtime status timestamp: `2026-09-03T16:17:17.067661+00:00`
- Stdout mtime UTC: `2026-09-03T09:26:07.604392+00:00`
- Stderr mtime UTC: `2026-09-03T16:26:07.196388+00:00`
- Last artifact: `outputs\qdiffcl_data_regime_v1\DATA_REGIME_GENERALIZATION_V1\tep\f100\outer_32001\rho_selection.json`
- Last formal completed cells: `225`
- Last formal cell: `3W / 10% / outer 31003 / CALIBRATED_RHO / seed 46`
- Last validation candidate: `TEP / 100% / outer 32001 / seed 2026 / rho 1.0 (15/15 completed)`
- Traceback: `yes`
- Python exception: `numpy._core._exceptions._ArrayMemoryError`
- Allocation request: `3.79 GiB`, shape `(53, 9600000)`, dtype `float64`
- CUDA OOM: `no`
- Metric NaN exception: `no`; stderr contains expected single-class group metric warnings only
- KeyboardInterrupt/external termination: `no evidence`
- Test-read guard violation: `no`
- Manifest/hash failure: `no`
- Failure count: `1`

- Remaining: `150` formal cells and `75` rho candidates
- Remaining GPU time: `UNAVAILABLE` until the host-RAM loader failure is resolved; the preliminary ETA file predates current progress and is stale

The failure occurred while reloading the TEP RData context after validation-only rho selection. No TEP outer-test result was produced. This audit does not restart the runner.
