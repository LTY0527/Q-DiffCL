# QDIFFCL_DATA_REGIME_RUNTIME_DIAGNOSIS

## 审计时间
2026-09-06 01:37 (本机)

## Runner
- runner status: DATA_REGIME_COMPLETE
- active PID: NONE
- last supervisor PID: 65116 (exited 0, completed outer32003 2026-09-05 01:56 UTC)
- last runtime_status: outputs/qdiffcl_data_regime_v1/runtime_status.json
- last artifact: outputs/.../CALIBRATED_RHO/result.json (seed_2026, TEP 25% outer32003)

## Accounting (hash-valid, audit script)
- formal valid / expected: 375 / 375
- formal remaining: 0
- rho candidate valid / expected: 225 / 225
- rho remaining: 0
- failures (historical): 3
- duplicates: 0

## Protocol
- TEP fractions: 100%, 25% （TEP 10% = E_IDENTIFIABILITY_HOLD 永久解除禁止）
- 3W fractions: 100%, 25%, 10%
- rho grid: [0, 0.25, 0.5, 0.75, 1.0]
- critical_ratio: 0.30
- FINAL_QDIFFCL_FIXED = 0.5 D + 0.5 E

## Origin / Git
- Data-Regime root: E:\Code\Q-DiffCL-data-regime
- Data-Regime branch: exp/qdiffcl-data-regime @ 55f48e28c36f0c49b9d0def8f7d58ff18794c61a
- remote origin: https://github.com/LTY0527/Q-DiffCL.git (same as root)
- working tree: CLEAN (git status, diff --check, bigfile scan, secrets scan)
