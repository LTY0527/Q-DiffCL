#!/usr/bin/env python
"""CLI audit script: read-only Data-Regime artifacts → R reliability audit.

Reads frozen train-only criticality.npz / audit.json + validation-only rho_selection.json
and produces qdiffcl_r_reliability_cells.csv plus validation association JSON.

Deterministic (seed 22042). Rejects test rows; TEP 10% held out.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.qdiffcl_reliability import (
    combine_reliability,
    compute_full_rho_regret,
    mask_reliability,
    outer_aware_association,
    spearman_rank_reliability,
    summarize_bootstrap_reliability,
    _robust_normalize,
    _top_mask,
)


ROOT = REPO_ROOT / "outputs" / "qdiffcl_data_regime_v1" / "DATA_REGIME_GENERALIZATION_V1"
RESULTS = REPO_ROOT / "analysis" / "results"
CONFIG_PATH = REPO_ROOT / "configs" / "qdiffcl_data_regime_v1.yaml"

FRACTIONS: dict[str, list[str]] = {
    "3W": ["f100", "f025", "f010"],
    "TEP": ["f100", "f025"],
}
OUTERS: dict[str, list[int]] = {
    "3W": [31001, 31002, 31003],
    "TEP": [32001, 32002, 32003],
}


def load_yaml_config(path: Path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fraction_numeric(folder: str) -> float:
    return {"f010": 0.10, "f025": 0.25, "f100": 1.00}[folder]


def load_criticality_artifact(crit_path: Path) -> dict:
    data = np.load(crit_path, allow_pickle=True)
    out: dict = {}
    for k in data.files:
        v = data[k]
        try:
            if isinstance(v, np.ndarray) and v.dtype == object and v.ndim == 0:
                out[k] = v.item()
            else:
                out[k] = v
        except Exception:
            out[k] = v
    return out


def build_cells_from_context(config: dict, algorithm: dict, force_rebuild: bool = False) -> list[dict]:
    results: list[dict] = []
    dataset_to_config_key = {"3W": "three_w", "TEP": "tep"}
    for dataset, fracs in FRACTIONS.items():
        ds_key = dataset_to_config_key.get(dataset, dataset.lower())
        if ds_key not in config:
            continue
        sampling_unit = config[ds_key]["sampling_unit"]
        for frac_folder in fracs:
            frac_num = fraction_numeric(frac_folder)
            if dataset == "TEP" and abs(frac_num - 0.10) < 1e-9:
                continue
            for outer_id in OUTERS[dataset]:
                cell_dir = ROOT / dataset.lower() / frac_folder / f"outer_{outer_id}"
                audit_path = cell_dir / "_context" / "audit.json"
                crit_path = cell_dir / "_context" / "criticality.npz"
                if not (audit_path.exists() and crit_path.exists()):
                    continue
                with open(audit_path, "r", encoding="utf-8") as f:
                    audit = json.load(f)
                if audit.get("outer_test_read", False):
                    raise RuntimeError(f"test leakage detected at {audit_path}")
                e_identifiable = not (dataset == "TEP" and abs(frac_num - 0.10) < 1e-9)
                artifact_crit = load_criticality_artifact(crit_path)
                composite_ref = np.asarray(artifact_crit.get("composite"), dtype=np.float64)
                discriminative = np.asarray(artifact_crit.get("discriminative"), dtype=np.float64)
                early_map = np.asarray(artifact_crit.get("early"), dtype=np.float64)
                run_counts = artifact_crit.get("run_counts", {})
                ref_hash = hashlib.sha256(np.ascontiguousarray(composite_ref).tobytes()).hexdigest()
                if force_rebuild:
                    raise NotImplementedError("--force-rebuild requires full train bundle reload; disabled in this audit")
                else:
                    ratio = float(algorithm["critical_ratio"])
                    w_d = float(algorithm["criticality_weights"]["weight_discriminative"])
                    w_e = float(algorithm["criticality_weights"]["weight_early"])
                    c_ref_proj = w_d * _robust_normalize(discriminative) + w_e * _robust_normalize(early_map)
                    m_ref = _top_mask(c_ref_proj, ratio)
                    rng = np.random.default_rng(int(algorithm["bootstrap_seed"]))
                    n_repeats = int(algorithm["bootstrap_repeats"])
                    rank_vals: list[float] = []
                    mask_vals: list[float] = []
                    comb_vals: list[float] = []
                    norm_runs_normal = int(run_counts.get("normal", 0))
                    norm_runs_fault = int(run_counts.get("fault", 0))
                    norm_runs_early = int(run_counts.get("early_fault", 0))
                    noise_d = max(1e-6, float(discriminative.std() * 0.01)) if discriminative.size else 1e-6
                    noise_e = max(1e-6, float(early_map.std() * 0.01)) if early_map.size else 1e-6
                    for _ in range(n_repeats):
                        perm_d = discriminative + rng.normal(0.0, noise_d, discriminative.shape)
                        perm_e = early_map + rng.normal(0.0, noise_e, early_map.shape)
                        c_b = w_d * _robust_normalize(perm_d) + w_e * _robust_normalize(perm_e)
                        m_b = _top_mask(c_b, ratio)
                        sr = spearman_rank_reliability(c_b, composite_ref)
                        mr = mask_reliability(m_b, m_ref)
                        cr = combine_reliability(sr, mr)
                        rank_vals.append(sr)
                        mask_vals.append(mr)
                        comb_vals.append(cr)
                    rank_arr = np.array(rank_vals, dtype=np.float64)
                    mask_arr = np.array(mask_vals, dtype=np.float64)
                    comb_arr = np.array(comb_vals, dtype=np.float64)
                    r_rank = float(np.nanmedian(rank_arr)) if rank_arr.size else float("nan")
                    r_mask = float(np.nanmedian(mask_arr)) if mask_arr.size else float("nan")
                    def q(arr, p): return float(np.nanquantile(arr, p)) if arr.size else float("nan")
                    summary = {
                        "r_rank": r_rank,
                        "r_mask": r_mask,
                        "R": combine_reliability(r_rank, r_mask),
                        "r_rank_quantiles": {"q05": q(rank_arr, .05), "q25": q(rank_arr, .25), "q50": q(rank_arr, .5), "q75": q(rank_arr, .75), "q95": q(rank_arr, .95)},
                        "r_mask_quantiles": {"q05": q(mask_arr, .05), "q25": q(mask_arr, .25), "q50": q(mask_arr, .5), "q75": q(mask_arr, .75), "q95": q(mask_arr, .95)},
                        "R_quantiles": {"q05": q(comb_arr, .05), "q25": q(comb_arr, .25), "q50": q(comb_arr, .5), "q75": q(comb_arr, .75), "q95": q(comb_arr, .95)},
                        "bootstrap_repeats": n_repeats,
                        "bootstrap_seed": int(algorithm["bootstrap_seed"]),
                        "critical_ratio": ratio,
                        "weights": {"weight_discriminative": w_d, "weight_early": w_e, "weight_run_stability": float(algorithm["criticality_weights"]["weight_run_stability"])},
                        "sampling_unit": sampling_unit,
                        "n_normal_runs": norm_runs_normal,
                        "n_fault_runs": norm_runs_fault,
                        "n_early_runs": norm_runs_early,
                        "reference_map_sha256": ref_hash,
                        "reference_mask_sha256": hashlib.sha256(np.ascontiguousarray(m_ref, dtype=bool).view(np.uint8).tobytes()).hexdigest(),
                        "c_ref_shape": list(composite_ref.shape),
                        "m_ref_true_count": int(m_ref.sum()),
                    }
                results.append({
                    "dataset": dataset,
                    "outer_id": outer_id,
                    "fraction": frac_folder,
                    "fraction_numeric": frac_num,
                    "E_identifiable": bool(e_identifiable),
                    "independent_train_groups": int(run_counts.get("normal", 0) + run_counts.get("fault", 0)),
                    "normal_groups": int(run_counts.get("normal", 0)),
                    "fault_groups": int(run_counts.get("fault", 0)),
                    "early_fault_groups": int(run_counts.get("early_fault", 0)),
                    "r_rank": summary["r_rank"],
                    "r_rank_q05": summary["r_rank_quantiles"]["q05"],
                    "r_rank_q25": summary["r_rank_quantiles"]["q25"],
                    "r_rank_q75": summary["r_rank_quantiles"]["q75"],
                    "r_rank_q95": summary["r_rank_quantiles"]["q95"],
                    "r_mask": summary["r_mask"],
                    "r_mask_q05": summary["r_mask_quantiles"]["q05"],
                    "r_mask_q25": summary["r_mask_quantiles"]["q25"],
                    "r_mask_q75": summary["r_mask_quantiles"]["q75"],
                    "r_mask_q95": summary["r_mask_quantiles"]["q95"],
                    "R": summary["R"],
                    "R_q05": summary["R_quantiles"]["q05"],
                    "R_q50": summary["R_quantiles"]["q50"],
                    "R_q75": summary["R_quantiles"]["q75"],
                    "R_q95": summary["R_quantiles"]["q95"],
                    "bootstrap_repeats": summary["bootstrap_repeats"],
                    "reference_map_hash": summary["reference_map_sha256"],
                    "reference_mask_hash": summary["reference_mask_sha256"],
                    "source_artifact_hashes": json.dumps({
                        "audit_sha256": sha256_file(audit_path),
                        "criticality_sha256": sha256_file(crit_path),
                    }, sort_keys=True),
                    "_rho_dir": str(cell_dir),
                    "_sampling_unit": sampling_unit,
                })
    return results


def attach_regret(cells: list[dict]) -> list[dict]:
    out = []
    for c in cells:
        rho_path = Path(c["_rho_dir"]) / "rho_selection.json"
        c2 = dict(c)
        if not rho_path.exists():
            c2["regret_rho1"] = float("nan")
            c2["rho_opt"] = float("nan")
            c2["V_max"] = float("nan")
            c2["V_ref"] = float("nan")
            out.append(c2)
            continue
        with open(rho_path, "r", encoding="utf-8") as f:
            sel = json.load(f)
        if sel.get("outer_test_read", False):
            raise RuntimeError(f"rho selection used test: {rho_path}")
        regret = compute_full_rho_regret(sel.get("candidate_rows", []))
        c2["regret_rho1"] = regret["regret_rho1"]
        c2["rho_opt"] = regret["rho_opt"]
        c2["V_max"] = regret["V_max"]
        c2["V_ref"] = regret["V_ref"]
        c2["selection_sha256"] = sha256_file(rho_path)
        out.append(c2)
    return out


def write_cells_csv(cells: list[dict], csv_path: Path) -> None:
    cols = ["dataset", "outer_id", "fraction", "fraction_numeric", "E_identifiable",
            "independent_train_groups", "normal_groups", "fault_groups", "early_fault_groups",
            "r_rank", "r_rank_q05", "r_rank_q25", "r_rank_q75", "r_rank_q95",
            "r_mask", "r_mask_q05", "r_mask_q25", "r_mask_q75", "r_mask_q95",
            "R", "R_q05", "R_q50", "R_q75", "R_q95",
            "bootstrap_repeats", "reference_map_hash", "reference_mask_hash", "source_artifact_hashes",
            "regret_rho1", "rho_opt", "V_max", "V_ref"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for c in cells:
            row = {k: c.get(k) for k in cols}
            writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Q-DiffCL reliability audit (read-only, deterministic)")
    ap.add_argument("--config", default=str(CONFIG_PATH), help="Path to qdiffcl_data_regime_v1 yaml")
    ap.add_argument("--output-csv", default=str(RESULTS / "qdiffcl_r_reliability_cells.csv"))
    ap.add_argument("--output-association", default=str(RESULTS / "qdiffcl_r_reliability_validation_association.json"))
    ap.add_argument("--output-manifest", default=str(RESULTS / "qdiffcl_r_reliability_run_manifest.json"))
    ap.add_argument("--force-rebuild", action="store_true",
                    help="Rebuild grouped-bootstrap R from raw train features instead of criticality.npz priors (much slower).")
    ap.add_argument("--association-bootstrap", type=int, default=1000)
    args = ap.parse_args(argv)
    if args.force_rebuild:
        print("--force-rebuild not available in audit-only mode; use raw training pipeline", file=sys.stderr)
        return 2
    config = load_yaml_config(Path(args.config))
    algorithm = config["algorithm"]
    RESULTS.mkdir(parents=True, exist_ok=True)
    cells = build_cells_from_context(config, algorithm, force_rebuild=False)
    cells = attach_regret(cells)
    csv_path = Path(args.output_csv)
    write_cells_csv(cells, csv_path)
    assoc = outer_aware_association(cells, bootstrap_seed=22042, bootstrap_repeats=args.association_bootstrap)
    assoc["exact_validation_utility"] = "(macro_f1, auprc, -far, -rho) lex"
    with open(args.output_association, "w", encoding="utf-8") as f:
        json.dump(assoc, f, indent=2, sort_keys=True, default=str)
    manifest = {
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "config_hash": sha256_file(Path(args.config)),
        "n_cells": len(cells),
        "cells_path": str(csv_path.relative_to(REPO_ROOT)),
        "association_path": str(Path(args.output_association).relative_to(REPO_ROOT)),
        "force_rebuild": False,
        "deterministic_seed": 22042,
        "test_leakage_hold": False,
        "tep10_hold": True,
    }
    with open(args.output_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"Wrote {len(cells)} cells → {csv_path}")
    print(f"Association → {args.output_association}")
    print(f"pooled Spearman = {assoc.get('pooled_spearman')}")
    print(f"P(assoc<0) = {assoc.get('P_assoc_lt_zero')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
