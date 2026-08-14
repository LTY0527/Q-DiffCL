from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

import scripts.run_3w_clean_baseline as base3w
from datasets.three_w import discover_instances
from scripts.run_3w_clean_collapse_diagnosis import train_probe
from scripts.run_3w_diffusion_1seed import METHODS, run
from scripts.run_3w_final_primary_grouped import FINAL_PRIMARY_CLASSES, build_model
from trainers.balanced import sqrt_inverse_frequency_weights
from utils import select_device


AUDITS = ("A_DIFFUSION", "B_ENCODER", "C_PROBE")


def prepare(base_config: dict, data_root: Path):
    grouped = Path(base_config["grouped_output"])
    split_index = int(base_config["canonical_split_index"])
    manifest = json.loads((grouped / "grouped_split_manifest.json").read_text(encoding="utf-8"))
    split = {name: set(wells) for name, wells in manifest["splits"][split_index]["wells"].items()}
    final = yaml.safe_load(Path(base_config["base_config"]).read_text(encoding="utf-8"))
    base = yaml.safe_load(Path(final["base_config"]).read_text(encoding="utf-8"))
    base3w.PRIMARY_CLASSES = FINAL_PRIMARY_CLASSES
    base3w.CLASS_TO_TARGET = {value: index for index, value in enumerate(FINAL_PRIMARY_CLASSES)}
    instances = [item for item in discover_instances(data_root) if item.source == "WELL" and item.event_class in FINAL_PRIMARY_CLASSES]
    by_instance = {item.instance_id: item for item in instances}
    by_split = {name: [item for item in instances if item.well_id in wells] for name, wells in split.items()}
    preprocessor = json.loads((grouped / f"split_{split_index:02d}" / "preprocessor.json").read_text(encoding="utf-8"))
    refs_by_split = {}; refs_by_instance = {}
    for name, items in by_split.items():
        refs = []
        for item in items:
            current = base3w.instance_refs(item, int(base["protocol"]["window_length"]), int(base["protocol"]["stride"]), int(base["protocol"]["transient_offset"]))
            refs.extend(current); refs_by_instance[item.instance_id] = current
        refs_by_split[name] = refs
    protocol_seed = int(base_config.get("protocol_seed", 42))
    train_refs = base3w.stratified_refs(refs_by_split["train"], int(final["train_windows_per_class"]), protocol_seed)
    validation_refs = base3w.stratified_refs(refs_by_split["validation"], int(final["validation_windows_per_class"]), protocol_seed + 1)
    length = int(base["protocol"]["window_length"])
    train_x, train_y = base3w.materialize(train_refs, by_instance, preprocessor, length, False)
    validation_x, validation_y = base3w.materialize(validation_refs, by_instance, preprocessor, length, False)
    evaluation = copy.deepcopy(base); evaluation["protocol"]["append_missing_mask"] = False
    evaluation["training"]["batch_size"] = int(base_config["training"]["batch_size"])
    return {"base": base, "by_split": by_split, "by_instance": by_instance, "refs_by_instance": refs_by_instance,
            "preprocessor": preprocessor, "train_x": train_x, "train_y": train_y,
            "validation_x": validation_x, "validation_y": validation_y, "evaluation": evaluation,
            "window_refs_sha256": hashlib.sha256("\n".join(f"{ref.instance_id}:{ref.start}:{ref.target}" for ref in train_refs + validation_refs).encode()).hexdigest()}


def probability_profile(model, prepared, device: str) -> dict:
    values = {"normal": [], "fault": []}
    for instance in prepared["by_split"]["test"]:
        refs = prepared["refs_by_instance"].get(instance.instance_id, [])
        if not refs: continue
        x, y = base3w.materialize(refs, {instance.instance_id: instance}, prepared["preprocessor"],
                                  int(prepared["base"]["protocol"]["window_length"]), False)
        probability = base3w.probabilities(model, x, y, int(prepared["evaluation"]["training"]["batch_size"]), device)[:, 0]
        values["normal"].extend(probability[y == 0].tolist()); values["fault"].extend(probability[y != 0].tolist())
    result = {}
    for name, rows in values.items():
        array = np.asarray(rows, dtype=np.float64)
        result[name] = {"count": len(array), "mean": float(array.mean()),
                        "quantiles": {str(q): float(np.quantile(array, q)) for q in (.05, .25, .5, .75, .95)}}
    return result


def checkpoint_record(audit: str, varied_seed: int, method: str, checkpoint: Path, result: dict,
                      prepared, base_config: dict, seeds: dict, output: Path) -> dict:
    device = select_device(str(base_config["device"])); model = build_model(prepared["base"]["training"]["model"], prepared["train_x"].shape[1], device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    metrics, _ = base3w.evaluate_stream(model, prepared["by_split"]["test"], prepared["refs_by_instance"],
                                        prepared["preprocessor"], prepared["evaluation"], device)
    source = json.loads(Path(base_config.get("criticality_source", base_config["output_dir"] + "/result.json")).read_text(encoding="utf-8"))
    record = {"audit": audit, "varied_seed": varied_seed, "method": method, **seeds,
              "metrics": metrics, "p_normal": probability_profile(model, prepared, device),
              "initialization_sha256": result["methods"][method]["initialization_sha256"],
              "window_refs_sha256": prepared["window_refs_sha256"],
              "critical_mask_sha256": source["fairness"]["critical_soft_mask_sha256"],
              "checkpoint": str(checkpoint)}
    path = output / "runs" / f"{audit}_seed{varied_seed}_{method}.json"; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"); return record


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/3w_seed_source_audit.yaml")
    parser.add_argument("--data-root", type=Path, required=True); parser.add_argument("--audit", choices=AUDITS, required=True)
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base_config = yaml.safe_load(Path(config["base_config"]).read_text(encoding="utf-8")); output = Path(config["output_dir"])
    selected = list(map(int, args.seeds if args.seeds is not None else config["seeds"])); prepared = prepare(base_config, args.data_root)
    critical_source = json.loads(Path(config["criticality_source"]).read_text(encoding="utf-8"))
    critical_hash = critical_source["fairness"]["critical_soft_mask_sha256"]
    for varied_seed in selected:
        if args.audit != "C_PROBE":
            current = copy.deepcopy(base_config); current["methods"] = list(config["methods"]); current["seed"] = varied_seed
            current["protocol_seed"] = int(config["protocol_seed"]); current["criticality_source"] = config["criticality_source"]
            if args.audit == "A_DIFFUSION":
                current.update({"diffusion_seed": varied_seed, "validation_diffusion_seed": int(config["fixed_validation_diffusion_seed"]),
                                "encoder_seed": int(config["fixed_encoder_seed"]), "probe_seed": int(config["fixed_probe_seed"])})
            else:
                current.update({"diffusion_seed": 42, "validation_diffusion_seed": 42,
                                "encoder_seed": varied_seed, "probe_seed": int(config["fixed_probe_seed"])})
            current["output_dir"] = str(output / "training" / f"{args.audit}_seed{varied_seed}")
            if varied_seed == 42:
                result = json.loads(Path(config["seed42_result"]).read_text(encoding="utf-8"))
            else:
                result = run(current, args.data_root)
            for method in config["methods"]:
                record_path = output / "runs" / f"{args.audit}_seed{varied_seed}_{method}.json"
                if record_path.exists(): print("skip", record_path.name, flush=True); continue
                checkpoint_root = Path(config["seed42_output"]) if varied_seed == 42 else Path(current["output_dir"])
                checkpoint = checkpoint_root / f"{method}_model.pt"
                seeds = {name: int(current[name]) for name in ("diffusion_seed", "encoder_seed", "probe_seed")}
                checkpoint_record(args.audit, varied_seed, method, checkpoint, result, prepared, current, seeds, output)
                print("done", args.audit, varied_seed, method, flush=True)
            continue
        for method in config["methods"]:
            record_path = output / "runs" / f"{args.audit}_seed{varied_seed}_{method}.json"
            if record_path.exists(): print("skip", record_path.name, flush=True); continue
            source_checkpoint = Path(config["seed42_output"]) / f"{method}_model.pt"
            device = select_device(str(base_config["device"])); model = build_model(prepared["base"]["training"]["model"], prepared["train_x"].shape[1], device)
            model.load_state_dict(torch.load(source_checkpoint, map_location=device, weights_only=True))
            if varied_seed == 42:
                checkpoint = source_checkpoint
            else:
                weights = sqrt_inverse_frequency_weights(prepared["train_y"])
                train_probe(model, prepared["train_x"], prepared["train_y"], prepared["validation_x"], prepared["validation_y"], weights,
                            int(base_config["training"]["probe_epochs"]), float(base_config["training"]["learning_rate"]),
                            int(base_config["training"]["batch_size"]), varied_seed, device)
                checkpoint = output / "checkpoints" / f"C_PROBE_seed{varied_seed}_{method}.pt"; checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint)
            metrics, _ = base3w.evaluate_stream(model, prepared["by_split"]["test"], prepared["refs_by_instance"], prepared["preprocessor"], prepared["evaluation"], device)
            source_result = json.loads(Path(config["seed42_result"]).read_text(encoding="utf-8"))
            record = {"audit": args.audit, "varied_seed": varied_seed, "method": method,
                      "diffusion_seed": 42, "encoder_seed": 42, "probe_seed": varied_seed,
                      "metrics": metrics, "p_normal": probability_profile(model, prepared, device),
                      "initialization_sha256": source_result["methods"][method]["initialization_sha256"],
                      "window_refs_sha256": prepared["window_refs_sha256"], "critical_mask_sha256": critical_hash,
                      "checkpoint": str(checkpoint)}
            record_path.parent.mkdir(parents=True, exist_ok=True); record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            print("done", args.audit, varied_seed, method, flush=True)


if __name__ == "__main__": main()
