from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from scripts.run_3w_diffusion_1seed import METHODS
from scripts.run_3w_seed_source_audit import AUDITS
from scripts.summarize_3w_diffusion_1seed import classify


STATUS = {"A_DIFFUSION": "DIFFUSION_RANDOMNESS_DOMINANT", "B_ENCODER": "ENCODER_OPTIMIZATION_DOMINANT", "C_PROBE": "PROBE_RANDOMNESS_DOMINANT"}


def mechanism_decision(source_summary: dict, dominance_ratio: float) -> tuple[str, dict]:
    scores = {name: (row["binary_auprc_std"] + row["class_9_recall_std"]) / 2 for name, row in source_summary.items()}
    ordered = sorted(scores, key=scores.get, reverse=True)
    ratio = scores[ordered[0]] / max(scores[ordered[1]], 1e-12)
    status = STATUS[ordered[0]] if ratio >= dominance_ratio else "MIXED_SEED_INSTABILITY"
    class9_order = sorted(source_summary, key=lambda name: source_summary[name]["class_9_recall_std"], reverse=True)
    class9_ratio = source_summary[class9_order[0]]["class_9_recall_std"] / max(source_summary[class9_order[1]]["class_9_recall_std"], 1e-12)
    class9_source = class9_order[0] if class9_ratio >= dominance_ratio else "MIXED"
    return status, {"instability_scores": scores, "dominance_ratio_observed": ratio,
                    "class9_primary_source": class9_source, "class9_dominance_ratio": class9_ratio}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/3w_seed_source_audit.yaml")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/3w_seed_source_audit"))
    parser.add_argument("--csv", type=Path, default=Path("docs/3w_seed_source_audit_paired.csv"))
    parser.add_argument("--json", type=Path, default=Path("docs/3w_seed_source_audit.json"))
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    single = yaml.safe_load(Path(config["base_config"]).read_text(encoding="utf-8"))
    records = {}; reference = None
    for audit in AUDITS:
        records[audit] = {}
        for seed in map(int, config["seeds"]):
            records[audit][seed] = {}
            for method in config["methods"]:
                path = args.output_dir / "runs" / f"{audit}_seed{seed}_{method}.json"
                if not path.exists(): raise RuntimeError(f"missing seed-source audit run: {path}")
                record = json.loads(path.read_text(encoding="utf-8")); current = (record["window_refs_sha256"], record["critical_mask_sha256"])
                if reference is not None and current != reference: raise RuntimeError("audit changed frozen window refs or critical mask")
                reference = current; records[audit][seed][method] = record
    rows = []; source_summary = {}
    for audit in AUDITS:
        deltas = []
        for seed in map(int, config["seeds"]):
            uniform = records[audit][seed][METHODS[1]]["metrics"]; r1 = records[audit][seed][METHODS[2]]["metrics"]
            _, delta, _ = classify(uniform, r1, single["gate"]); deltas.append(delta)
            row = {"audit": audit, "varied_seed": seed, **delta}
            for method in (METHODS[1], METHODS[2]):
                profile = records[audit][seed][method]["p_normal"]
                prefix = "uniform" if method == METHODS[1] else "r1"
                for group in ("normal", "fault"):
                    row[f"{prefix}_pnormal_{group}_mean"] = profile[group]["mean"]
                    for q, value in profile[group]["quantiles"].items(): row[f"{prefix}_pnormal_{group}_q{q}"] = value
            rows.append(row)
        binary = [row["binary_auprc"] for row in deltas]; class9 = [row["class_9_recall"] for row in deltas]
        source_summary[audit] = {
            "binary_auprc_mean": float(np.mean(binary)), "binary_auprc_std": float(np.std(binary)),
            "class_9_recall_mean": float(np.mean(class9)), "class_9_recall_std": float(np.std(class9)),
            "macro_f1_mean": float(np.mean([row["macro_f1"] for row in deltas])),
            "macro_f1_std": float(np.std([row["macro_f1"] for row in deltas])),
            "far_mean": float(np.mean([row["far"] for row in deltas])),
            "far_std": float(np.std([row["far"] for row in deltas])),
            "seed44_binary_auprc_delta": float(deltas[2]["binary_auprc"]),
            "seed44_class9_recall_delta": float(deltas[2]["class_9_recall"]),
        }
    status, decision = mechanism_decision(source_summary, float(config["dominance_ratio"]))
    seed44_source = min(source_summary, key=lambda name: source_summary[name]["seed44_binary_auprc_delta"])
    payload = {"status": status, "source_summary": source_summary, **decision,
               "seed44_binary_auprc_largest_drop_source": seed44_source,
               "window_refs_sha256": reference[0], "critical_mask_sha256": reference[1],
               "runs": records}
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("status", "source_summary", "instability_scores", "class9_primary_source", "seed44_binary_auprc_largest_drop_source")}, ensure_ascii=False))


if __name__ == "__main__": main()
