# Q-DiffCL TEP Memory Repair Audit

Status: `TEP_MEMORY_SAFE_LOADER_EQUIVALENCE_GO`.

## Failure root cause

The failing allocation is not a CUDA tensor and is not created by Q-DiffCL training. The traceback reaches `datasets.tep.read_rdata_frame`, then `pyreadr.read_r`, `pandas.DataFrame.from_dict`, BlockManager consolidation, and finally `numpy.vstack`. Pandas attempts to consolidate 53 decoded float64 columns with 9,600,000 rows into one additional contiguous `(53, 9600000)` array. That redundant full-table copy requests about 3.79 GiB while the individual decoded columns already exist.

The RData source and downstream Run values are float64 before the frozen window pipeline converts normalized windows to float32. The repair does not move or remove that frozen conversion and does not lower source precision.

## Repeated copies and lifetime

- `pyreadr` already decodes each R column; BlockManager consolidation adds a second full float64 representation.
- The old `frame_to_runs` grouped the entire frame before rejecting Runs above the registered limits, causing unnecessary group materialization for excluded Runs.
- The Data-Regime CLI prepared a context in `main`, then `run_formal_context` prepared the same dataset/fraction/outer context again. With `--stage all`, the validation-selection context remained live when the formal context was reloaded. This is the direct lifecycle explanation for the observed failure immediately after the completed TEP outer-32001 rho selection.

## Numerically neutral repair

1. `read_rdata_frame` asks pandas to retain pyreadr's decoded columns in `ArrayManager`, removing only the consolidation copy.
2. `frame_to_runs` applies the same frozen `simulationRun <= limit` predicate before groupby. Selected row order, Run sort order, values, samples, labels, and boundaries remain unchanged.
3. The formal runner receives and reuses the already prepared context. It does not load the same TEP context a second time.

No fraction, split, seed, rho grid, D/E/S weight, timestep, critical ratio, backbone, loss, probe, epoch, patience, or batch-size setting changed. No test metric or outer result was consulted when choosing the repair.

## Gate result

The real small RData equivalence test passes exactly. After available system RAM exceeded the registered 2 GiB floor, TEP 100% and 25% contexts each built and released twice. Repeated signatures were exact; post-release working-set changes were `-28 MiB` and `+4 MiB`, so there was no linear retention. The pre/post outer-32001 100% context, scaler, criticality mask, criticality NPZ, selected IDs, and window-count evidence matches exactly. See `analysis/results/qdiffcl_tep_memory_smoke.json`.
