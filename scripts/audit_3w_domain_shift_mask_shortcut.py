from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from datasets.three_w import discover_instances
from scripts.run_3w_clean_baseline import (CLASS_TO_TARGET, PRIMARY_CLASSES, instance_refs,
                                            load_split, read_frame, transform_frame)


def stable_sample(items, limit: int, key: str):
    ranked = sorted(items, key=lambda item: hashlib.sha256(f"{key}|{item.start}".encode()).digest())
    return ranked[:limit]


def summary_views(raw: np.ndarray, processed: np.ndarray, mask: np.ndarray, start: int, length: int):
    raw_window = raw[start:start + length]; process_window = processed[start:start + length]; mask_window = mask[start:start + length]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw_summary = np.r_[np.nanmean(raw_window, axis=0), np.nanstd(raw_window, axis=0)]
    process_summary = np.r_[process_window.mean(0), process_window.std(0)]
    mask_summary = mask_window.mean(0).astype(np.float64)
    return raw_summary, process_summary, mask_summary, np.r_[process_summary, mask_summary]


def fit_light_classifier(train_x, train_y, test_x, test_y, seed: int):
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(max_iter=1500, class_weight="balanced", random_state=seed))
    model.fit(train_x, train_y); prediction = model.predict(test_x)
    labels = np.unique(np.r_[train_y, test_y]); precision, recall, f1, support = precision_recall_fscore_support(test_y, prediction, labels=labels, zero_division=0)
    return {"accuracy": float(accuracy_score(test_y, prediction)), "macro_f1": float(f1_score(test_y, prediction, average="macro", zero_division=0)),
            "macro_recall": float(np.mean(recall)), "chance": float(1 / len(labels)),
            "labels": [int(x) for x in labels], "confusion_matrix": confusion_matrix(test_y, prediction, labels=labels).tolist(),
            "per_class": [{"class": int(c), "precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)} for c, p, r, f, s in zip(labels, precision, recall, f1, support)]}


def cosine(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator else 1.0


def distance_audit(records, view_key="process"):
    grouped = defaultdict(list)
    for row in records: grouped[(row["target"], row["well_id"])].append(row[view_key])
    centroids = {key: np.mean(values, axis=0) for key, values in grouped.items()}
    within_euclidean, within_cosine, between_euclidean, between_cosine = defaultdict(list), defaultdict(list), [], []
    items = list(centroids.items())
    for ((class_a, well_a), a), ((class_b, well_b), b) in itertools.combinations(items, 2):
        e = float(np.linalg.norm(a - b)); c = 1 - cosine(a, b)
        if class_a == class_b and well_a != well_b:
            within_euclidean[class_a].append(e); within_cosine[class_a].append(c)
        elif class_a != class_b:
            between_euclidean.append(e); between_cosine.append(c)
    overall_between_e = float(np.mean(between_euclidean)); overall_between_c = float(np.mean(between_cosine))
    rows = []
    for target in sorted({row["target"] for row in records}):
        e = float(np.mean(within_euclidean[target])) if within_euclidean[target] else None
        c = float(np.mean(within_cosine[target])) if within_cosine[target] else None
        rows.append({"target": target, "original_class": PRIMARY_CLASSES[target], "within_cross_well_euclidean": e,
                     "between_class_euclidean": overall_between_e, "euclidean_ratio": e / overall_between_e if e else None,
                     "within_cross_well_cosine": c, "between_class_cosine": overall_between_c,
                     "cosine_ratio": c / overall_between_c if c else None})
    return rows


def feature_shift(records, features, view_key="process"):
    rows = []
    for index, feature in enumerate(features):
        by_well = defaultdict(list)
        for row in records: by_well[row["well_id"]].append(float(row[view_key][index]))
        well_means = np.asarray([np.mean(values) for values in by_well.values()]); between = float(np.var(well_means))
        within = float(np.mean([np.var(values) for values in by_well.values()])); denominator = between + within
        rows.append({"feature": feature, "between_well_variance": between, "within_well_variance": within,
                     "icc_like_ratio": between / denominator if denominator else 0.0})
    return sorted(rows, key=lambda row: row["icc_like_ratio"], reverse=True)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--data-root", type=Path, required=True); parser.add_argument("--output-dir", type=Path, default=Path("docs/3w_domain_audit")); parser.add_argument("--seed", type=int, default=7); parser.add_argument("--windows-per-instance", type=int, default=8)
    args = parser.parse_args(); output = args.output_dir; output.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(Path("configs/3w_clean_baseline.yaml").read_text(encoding="utf-8")); preprocessor = json.loads(Path("outputs/3w_clean_baseline_seed7/preprocessor.json").read_text(encoding="utf-8"))
    split_wells = load_split(Path(base["protocol"]["split_manifest"])); well_to_split = {well: split for split, wells in split_wells.items() for well in wells}
    instances = [item for item in discover_instances(args.data_root) if item.source == "WELL" and item.event_class in PRIMARY_CLASSES]
    features = preprocessor["retained_features"]; length = int(base["protocol"]["window_length"]); stride = int(base["protocol"]["stride"]); offset = int(base["protocol"]["transient_offset"])
    records, instance_signatures, instance_rows = [], [], []
    support = defaultdict(lambda: {"windows": 0, "observations": 0, "instances": set(), "wells": set(), "well_windows": Counter()})
    split_support = defaultdict(lambda: {"windows": 0, "instances": set(), "wells": set()})
    for number, instance in enumerate(instances, 1):
        frame = read_frame(instance, list(features) + ["class"]); raw = frame[features].to_numpy(dtype=np.float64, na_value=np.nan)
        processed, mask = transform_frame(frame[features], preprocessor); refs = instance_refs(instance, length, stride, offset)
        per_target = Counter(ref.target for ref in refs)
        for target, count in per_target.items():
            item = support[target]; item["windows"] += count; item["observations"] += len(frame); item["instances"].add(instance.instance_id); item["wells"].add(instance.well_id); item["well_windows"][instance.well_id] += count
            split_item = split_support[(well_to_split[instance.well_id], target)]; split_item["windows"] += count; split_item["instances"].add(instance.instance_id); split_item["wells"].add(instance.well_id)
        signature = mask.mean(0).astype(np.float64); instance_signatures.append({"well_id": instance.well_id, "instance": instance.instance_id, "event_class": instance.event_class, "signature": signature})
        instance_rows.append({"well_id": instance.well_id, "instance": instance.instance_id, "event_class": instance.event_class, "observations": len(frame), "signature": signature})
        for ref in stable_sample(refs, args.windows_per_instance, instance.instance_id):
            raw_s, process_s, mask_s, combined = summary_views(raw, processed, mask, ref.start, length)
            records.append({"well_id": instance.well_id, "instance": instance.instance_id, "event_class": instance.event_class,
                            "split": well_to_split[instance.well_id], "target": ref.target, "stage": ref.stage,
                            "raw": raw_s, "process": process_s, "mask": mask_s, "combined": combined})
        if number % 100 == 0: print("instances", number, len(instances), flush=True)

    class_rows = []
    instance_to_well = {row["instance"]: row["well_id"] for row in instance_rows}
    for target, original in enumerate(PRIMARY_CLASSES):
        item = support[target]; counts = np.asarray(list(item["well_windows"].values()), dtype=float); fractions = counts / counts.sum()
        instances_per_well = Counter(instance_to_well[instance_id] for instance_id in item["instances"])
        class_rows.append({"class": original, "target": target, "distinct_well_count": len(item["wells"]), "instance_count": len(item["instances"]),
                           "observation_count_instance_total": item["observations"], "window_count": item["windows"],
                           "median_instances_per_well": float(np.median(list(instances_per_well.values()))) if instances_per_well else 0,
                           "median_windows_per_well": float(np.median(counts)), "max_windows_from_single_well": int(counts.max()),
                           "largest_well_window_fraction": float(fractions.max()), "effective_well_diversity": float(1 / np.square(fractions).sum())})
    split_rows = []
    for split in ("train", "validation", "test"):
        total = sum(split_support[(split, target)]["windows"] for target in range(len(PRIMARY_CLASSES)))
        for target, original in enumerate(PRIMARY_CLASSES):
            item = split_support[(split, target)]
            split_rows.append({"split": split, "class": original, "target": target, "distinct_wells": len(item["wells"]),
                               "instances": len(item["instances"]), "windows": item["windows"], "window_percentage": item["windows"] / total,
                               "split_wells_over_total_wells": f"{len(split_wells[split])}/40"})

    # Strict instance-level partition for WELL-ID: each target WELL must have disjoint train/test instances.
    well_train, well_test = [], []
    for well in sorted({row["well_id"] for row in records}):
        ids = sorted({row["instance"] for row in records if row["well_id"] == well})
        if len(ids) < 2: continue
        cut = max(1, min(len(ids) - 1, round(len(ids) * .7))); train_ids = set(ids[:cut])
        well_train.extend(row for row in records if row["well_id"] == well and row["instance"] in train_ids)
        well_test.extend(row for row in records if row["well_id"] == well and row["instance"] not in train_ids)
    wells = sorted({row["well_id"] for row in well_train} & {row["well_id"] for row in well_test}); well_map = {well: i for i, well in enumerate(wells)}
    well_train = [row for row in well_train if row["well_id"] in well_map]; well_test = [row for row in well_test if row["well_id"] in well_map]
    def matrix(rows, key): return np.stack([row[key] for row in rows])
    well_results = {key: fit_light_classifier(matrix(well_train, key), np.asarray([well_map[r["well_id"]] for r in well_train]), matrix(well_test, key), np.asarray([well_map[r["well_id"]] for r in well_test]), args.seed) for key in ("raw", "process", "mask", "combined")}
    fault_train = [row for row in records if row["split"] == "train"]; fault_test = [row for row in records if row["split"] == "test"]
    fault_results = {key: fit_light_classifier(matrix(fault_train, key), np.asarray([r["target"] for r in fault_train]), matrix(fault_test, key), np.asarray([r["target"] for r in fault_test]), args.seed) for key in ("process", "mask", "combined")}
    comparison = [{"representation": key, "fault_macro_f1": fault_results[key]["macro_f1"], "fault_macro_recall": fault_results[key]["macro_recall"],
                   "well_id_accuracy": well_results[key]["accuracy"], "well_id_macro_f1": well_results[key]["macro_f1"], "well_id_chance": well_results[key]["chance"]} for key in ("process", "mask", "combined")]

    # Raw summaries use train-only median imputation solely to make distances finite; no test statistic is fitted.
    raw_imputer = SimpleImputer(strategy="median").fit(matrix(fault_train, "raw"))
    raw_filled = raw_imputer.transform(matrix(records, "raw"))
    for row, value in zip(records, raw_filled): row["raw_filled"] = value
    distance_rows = [{"space": "preprocessed", **row} for row in distance_audit(records)]
    distance_rows += [{"space": "raw_train_median_imputed", **row} for row in distance_audit(records, "raw_filled")]
    feature_rows = [{"space": "preprocessed", **row} for row in feature_shift(records, features)]
    feature_rows += [{"space": "raw_train_median_imputed", **row} for row in feature_shift(records, features, "raw_filled")]
    signature_rows = [{"well_id": row["well_id"], "instance": row["instance"], "event_class": row["event_class"],
                       **{feature: 1 - float(value) for feature, value in zip(features, row["signature"])}} for row in instance_signatures]
    similarities = {"same_well": [], "same_class_different_well": [], "different_class": []}
    for a, b in itertools.combinations(instance_signatures, 2):
        similarity = cosine(a["signature"], b["signature"])
        if a["well_id"] == b["well_id"]: similarities["same_well"].append(similarity)
        elif a["event_class"] == b["event_class"]: similarities["same_class_different_well"].append(similarity)
        else: similarities["different_class"].append(similarity)
    similarity_rows = [{"group": key, "pairs": len(values), "mean_cosine_similarity": float(np.mean(values)), "std": float(np.std(values)),
                        "p25": float(np.percentile(values, 25)), "median": float(np.median(values)), "p75": float(np.percentile(values, 75))} for key, values in similarities.items()]

    # LO-WELL-out nearest-centroid audit for the three best-supported fault classes.
    loo_rows = []
    for target in (CLASS_TO_TARGET[2], CLASS_TO_TARGET[4], CLASS_TO_TARGET[8]):
        for well in sorted({r["well_id"] for r in records if r["target"] == target}):
            train = [r for r in records if r["well_id"] != well and r["target"] in (CLASS_TO_TARGET[2], CLASS_TO_TARGET[4], CLASS_TO_TARGET[8])]
            test = [r for r in records if r["well_id"] == well and r["target"] == target]
            centroids = {c: np.mean([r["process"] for r in train if r["target"] == c], axis=0) for c in (CLASS_TO_TARGET[2], CLASS_TO_TARGET[4], CLASS_TO_TARGET[8])}
            prediction = [min(centroids, key=lambda c: np.linalg.norm(r["process"] - centroids[c])) for r in test]
            loo_rows.append({"original_class": PRIMARY_CLASSES[target], "held_out_well": well, "windows": len(test), "accuracy": float(np.mean(np.asarray(prediction) == target))})

    recommendations = []
    for row in class_rows:
        split_class = [x for x in split_rows if x["class"] == row["class"]]
        train_wells = next(x["distinct_wells"] for x in split_class if x["split"] == "train")
        if row["class"] in (1, 5): recommendation, reason = "SECONDARY", "only one training WELL despite abundant windows"
        elif train_wells >= 3 and row["distinct_well_count"] >= 6: recommendation, reason = "KEEP", "multiple independent train WELLs and at least six total target-support WELLs"
        else: recommendation, reason = "SECONDARY", "limited independent cross-WELL support"
        recommendations.append({"class": row["class"], "recommendation": recommendation, "reason": reason,
                                "distinct_target_wells": row["distinct_well_count"], "largest_well_window_fraction": row["largest_well_window_fraction"],
                                "effective_well_diversity": row["effective_well_diversity"]})
    candidates = {"protocol_a_strict_real_only": {"classes": [0, 2, 4, 7, 8, 9], "rule": "exclude fault classes with only one training WELL"},
                  "protocol_b_extended_real_only": {"classes": list(PRIMARY_CLASSES), "rule": "retain current classes but mark 1/5 high-variance secondary-grade evidence"},
                  "protocol_c_real_plus_simulated_secondary": {"real_test_only": True, "real_primary_classes": [0, 2, 4, 7, 8, 9], "simulated_train_secondary_classes": [1, 2, 5, 8, 9],
                                                                  "drawn_in_primary": False, "leakage_control": "simulated instance identity disjoint; fit preprocessing on real-train unless explicitly reported as domain adaptation"}}
    process_well = well_results["process"]["accuracy"]; mask_well = well_results["mask"]["accuracy"]
    primary_status = "3W_CROSS_WELL_SHIFT_DOMINANT" if process_well > well_results["process"]["chance"] * 3 else "3W_REAL_ONLY_PROTOCOL_REDUCTION_REQUIRED"
    result = {"status": primary_status, "base_commit": "ae0d94b", "seed": args.seed, "sampled_windows": len(records), "eligible_well_id_classes": len(wells),
              "well_id": well_results, "fault": fault_results, "representation_comparison": comparison, "distance": distance_rows,
              "feature_shift_top10": [row for row in feature_rows if row["space"] == "preprocessed"][:10], "mask_similarity": similarity_rows, "loo": loo_rows,
              "recommendations": recommendations, "protocol_candidates": candidates,
              "secondary_findings": {"missing_mask_well_shortcut_present": mask_well > well_results["mask"]["chance"] * 3,
                                      "missing_mask_label_shortcut_present": fault_results["mask"]["macro_f1"] > 1 / len(PRIMARY_CLASSES) * 1.5,
                                      "window_abundance_not_domain_diversity": True}}
    write_csv(output / "3w_class_well_support.csv", class_rows); write_csv(output / "3w_split_well_support.csv", split_rows)
    write_csv(output / "3w_cross_well_feature_shift.csv", feature_rows); write_csv(output / "3w_cross_well_distance.csv", distance_rows)
    write_csv(output / "3w_mask_signature.csv", signature_rows); write_csv(output / "3w_mask_similarity_summary.csv", similarity_rows)
    write_csv(output / "3w_representation_shortcut_comparison.csv", comparison); write_csv(output / "3w_protocol_class_recommendation.csv", recommendations); write_csv(output / "3w_leave_one_well_out.csv", loo_rows)
    (output / "3w_protocol_candidates.json").write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": primary_status, "well_id": comparison, "fault": comparison, "secondary": result["secondary_findings"]}, ensure_ascii=False))


if __name__ == "__main__": main()
