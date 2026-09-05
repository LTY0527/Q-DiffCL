# Q-DiffCL｜Reliability (R) 可行性审计报告

- **判定**: `QDIFFCL_R_FEASIBILITY_NO_GO`
- **时间**: 2026-09-06 (UTC+8)
- **分支**: `exp/qdiffcl-r-reliability-feasibility` @ data-regime archive HEAD
- **研究定位**: 仅证伪优先可行性研究（R = criticality estimator reliability，非新 fault score；今晚不训练新模型、不改 FINAL 公式、不碰 TEST rows）

---

## 1. 冻结协议摘要（全程未改）

| 项目 | 值 |
|---|---|
| FINAL composition | `0.5·D + 0.5·E`, `S = 0` |
| critical_ratio | `0.30` |
| rho_grid | `[0, .25, .5, .75, 1]` |
| ρ-selection utility（exact，来自 `run_qdiffcl_data_regime.py:91`） | `lex( Macro-F1 ↑, AUPRC ↑, −FAR ↑, −ρ ↑ )` |
| Regret_rho1 | `max_ρ V(ρ) − V(ρ=1)` |
| R formula（预注册 fixed） | `R = 0.5·max(0,R_rank) + 0.5·R_mask` |
| R_rank | `median_b Spearman(C_b, C_ref)` |
| R_mask | `median_b Jaccard(M_b, M_ref)` |
| Bootstrap unit（严格 grouped） | 3W: `instance_id` / TEP: `run_uid`，禁止 window iid |
| Bootstrap repeats / seed | `64` / `22042` |
| Association metric | pooled outer-aware stratified bootstrap Spearman(R, Regret_rho1) |

---

## 2. Artifact Inventory / Test Leakage Gate

- 产出文件: `analysis/results/qdiffcl_r_reliability_artifact_inventory.json`
- 总 Artifacts: `247`（train=30, validation=217, test=0）
  - `audit_context` 15 (split=train, used_for_R=YES)
  - `criticality_train_priors` 15 (split=train, used_for_R=YES)
  - `rho_selection_validation_meta` 15 (split=validation, used_for_R=YES)
  - `rho_candidate_validation_scores` 202 (split=validation, used_for_R=YES)
- **TEST LEAKAGE GATE: PASS**
  - `split_role=test ∧ used_for_R=YES`: 0 条
  - `R_TEST_LEAKAGE_HOLD = false`
  - `leaking_artifacts = []`
- **TEP 10% HOLD**: 永久尊重，未进入 pipeline

---

## 3. Reliability Cells（15/15，合法 3W×9 + TEP×6）

产出: `analysis/results/qdiffcl_r_reliability_cells.csv`

| dataset | fractions × outers | n | R median | R IQR | Regret_rho1 (median) |
|---|---|---|---|---|---|
| 3W | f100,f025,f010 × 31001..31003 | 9 | 0.97–0.99 | 0.013 | 0 – 1156 |
| TEP | f100,f025 × 32001..32003 | 6 | 0.996–0.999 | 0.004 | 0 – 12 |

- R non-degenerate：所有 cell R ∈ [0.957, 0.999]，finite, deterministic。
- 所有 cell：`r_rank > 0.99`, `r_mask ∈ [0.93, 1.0]`, `bootstrap_repeats=64`。
- reference_map_hash / reference_mask_hash 稳定；source_artifact_hashes 完整。
- E_identifiable：TEP 10% 行未产出（HOLD）。

---

## 4. Validation-only Association: R ↔ Regret_rho1

产出: `analysis/results/qdiffcl_r_reliability_validation_association.json`
Exact utility: `(macro_f1, auprc, -far, -rho) lex`（逐行从 `rho_selection.json: candidate_rows` 聚合，无 test leakage）

| 范围 | n | Spearman | bootstrap q05 | q50 | q95 | P(assoc < 0) |
|---|---|---|---|---|---|---|
| pooled **(primary)** | 15 | **−0.293** | −0.503 | −0.217 | +0.071 | 0.874 |
| 3W | 9 | **−0.267** | −0.714 | +0.029 | +0.500 | — |
| TEP | 6 | **+0.543** ⚠️ | −0.200 | +0.800 | +1.000 | — |

Bootstrap: outer-aware stratified（1000 repeats，seed 22042），非 candidate-row iid。

---

## 5. 预注册 Gate 核对（逐项严格）

预注册 GO 要求 **全部同时成立**：

| Gate 条件 | 实际值 | 是否满足 |
|---|---|---|
| R finite / non-degenerate / deterministic | ✓ 所有 cell R∈[0.957, 0.999]，完全可重复 | ✅ |
| pooled Spearman(R, Regret) ≤ **−0.30** | −0.293（差 0.007） | ❌ *missed by 0.007* |
| P(assoc < 0) ≥ 0.80 | 0.874 | ✅ |
| ≥1 dataset-specific ≤ −0.30 | 3W=−0.267, TEP=+0.543（均未达标） | ❌ |
| 另一个 dataset assoc ≤ +0.10 | **TEP = +0.543 > +0.10**（硬违反） | ❌ ⚠️ |
| 无 test leakage | audit/artifact gate 双 PASS；`test_rows_used=false` | ✅ |
| 非单个 outer / fraction 独占驱动 | 3W 9 cells 分布跨 3 outer × 3 fractions；TEP 6 cells 跨 2 fractions——**但 TEP 方向整体反向，说明 R 在 TEP 不具备预期负向关联** | ❌ |

### 判定

```text
QDIFFCL_R_FEASIBILITY_NO_GO
```

- **硬违反项**：TEP-specific assoc = +0.543，**远超** +0.10 上限。这表明在 TEP dataset 上，R（criticality map stability）与 ρ=1 的 validation regret 之间是**正相关**（R 高反而 regret 高），与预期 `R↑→regret↓` 方向完全相反。
- 次要违反：pooled Spearman 仅差 0.007（−0.293 vs −0.30），3W-specific 仅差 0.033（−0.267 vs −0.30）——但因 TEP 硬违反，这两项不做补偿。
- **绝不因结果不好进行任何 metric/weight/association metric 调整**：保持 0.5/0.5 权重、Spearman、outer-aware bootstrap、ρ utility 全部冻结，按 NO-GO 立即停止 R 公式扩展。

---

## 6. NO-GO 结论

今晚核心证伪问题 **"Can we trust the estimated channel-frequency criticality map to predict ρ=1 safety regret?"** → **当前证据下不能**：
1. R 本身在两个 dataset 上都表现出很高的数值稳定性（rank/mask median > 0.93），说明**关键图本身 bootstrap 可靠**；
2. 但这种 bootstrap stability **并不能预测**下游 full-ρ vs ρ=1 validation regret：
   - 3W 上弱负向（接近门槛），TEP 上强正向反向；
   - pooled 方向正确但量级略低于门槛，且被 TEP 反向效应显著拖累。

### 下一步不自动做
- ❌ 不写 DESIGN_NOTE（NO-GO 触发 §16 停止规则）
- ❌ 不产 `docs/assets/qdiffcl_r_reliability/` 图
- ❌ 不改 R 权重（0.7/0.3 etc.）、不换 association metric、不删 outer、不换 bootstrap strategy
- ❌ 不启动任何 "rho_eff = R·rho_val" 的训练实验

### 可以考虑的后续非承诺方向（需独立协议 lock）
如果未来要重新研究 R，需要：
- 重新界定 **regret 的标准化/对齐**（当前 3W regret 数值跨度 [0,1156] 远大于 TEP 的 [0,12]，pooled Spearman 被 3W 的绝对数值主导）；
- TEP 上 R 的 rank/mask 稳定性高但 ρ selection 无差异（多数 regret=0 或极低），可能需要 **TEP-specific regret floor / subset restriction**；
- 或转而去研究 **"R 与 method contrast NO_AUG vs CALIBRATED_RHO 的 validation improvement"** 这一关联（不是 ρ-grid regret）。

---

## 7. 代码产出与 Reproducibility

### 新增文件
- `analysis/qdiffcl_reliability.py` — 4 个核心纯函数（`spearman_rank_reliability`, `mask_reliability`, `combine_reliability`, `summarize_bootstrap_reliability`）+ 辅助（`choose_rho_utility`, `compute_full_rho_regret`, `outer_aware_association`）
- `scripts/audit_qdiffcl_reliability.py` — 只读 CLI，`--help` 正常，默认 seed 22042，拒绝 `--force-rebuild`，过滤 TEP10，test-leakage 抛 RuntimeError
- `tests/test_qdiffcl_reliability.py` — §19 要求 10 条 test case，**10/10 PASSED**：
  1. identical maps → rank reliability = 1
  2. identical masks → Jaccard = 1
  3. 0 ≤ R ≤ 1
  4. reversed ranking handled (Spearman = −1)
  5. ties deterministic
  6. fixed bootstrap seed deterministic
  7. window_iid bootstrap path explicitly raises ValueError
  8. TEP10 HOLD cannot produce full R
  9. test rows (rho_selection `outer_test_read=true`) rejected via RuntimeError
  10. frozen historical criticality top_mask ratio=0.30 behavior unchanged

### 数据产出
- `analysis/results/qdiffcl_r_reliability_artifact_inventory.json` — 247 artifacts / split_role / sha256
- `analysis/results/qdiffcl_r_reliability_cells.csv` — 15 cells（3W 9 + TEP 6），所有 required 字段
- `analysis/results/qdiffcl_r_reliability_validation_association.json` — exact utility + 3W/TEP/pooled spearman + outer-aware bootstrap quantiles
- `analysis/results/qdiffcl_r_reliability_run_manifest.json` — config hash / deterministic seed / hold flags

### Reproduce
```powershell
$PY = "E:\anaconda\envs\qdiffcl\python.exe"
& $PY -m pytest tests/test_qdiffcl_reliability.py -q   # 10 passed
& $PY -m scripts.audit_qdiffcl_reliability             # 复现 15 cells + association; output paths as above
```

Expected output（deterministic seed 22042）：
```
pooled Spearman = -0.29285714285714287
P(assoc<0) = 0.874
```

---

## 8. Paderborn 保护 & Worktree 管理

| 项目 | 状态 |
|---|---|
| Paderborn worktree (`E:\Code\Q-DiffCL`) modified? | **NO** |
| Paderborn training launched? | **NO** |
| pseudo-E created? | **NO** |
| NEW_WORKTREE_CREATED | **NO**（Case A，同目录 switch branch 安全） |
| Data-Regime worktree history | intact @ `exp/qdiffcl-data-regime` `dc8d12c`（archive commit 已推 origin） |
