#!/usr/bin/env python3
"""One-time held-out validation using the frozen Phase 8/9 pipeline.

Commands are deliberately staged: ``prepare-address`` creates only the missing
validation Address K10 file, ``score`` applies the frozen Phase 8 models to the
V1 union Address K10 candidates, and ``evaluate`` scores the frozen Phase 9
policy against validation ground truth. Nothing in Phases 4--9 is written.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import resource
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from baseline import TargetStore, load_s1_records
from diagnostics import DEFAULT_DATASET_ROOT
from phase6d_retrieval import address_retrieve
from phase8_model import (
    FEATURE_NAMES, SCORE_HEADER, extract_features, iter_union_groups,
    load_models, union_candidate_counts,
)
from phase9_policy import (
    DecisionPolicy, evaluate_policy, read_score_file, write_predictions,
)
from scoring import load_ground_truth, load_id_file, precision_recall_diagnostics, LinkCounts, count_link_errors


BASE = Path(__file__).resolve().parents[1]
ART = BASE / "artifacts"
OUT = ART / "validation_evaluation"
VAL_IDS = ART / "splits/val_s1_ids.txt"
V1_CANDIDATES = ART / "blocking/v1_validation_candidates.tsv.gz"
V1_METADATA = ART / "blocking/v1_validation_metadata.tsv.gz"
TARGET_INDEX = ART / "blocking/v1_index.sqlite"
ADDRESS_INDEX = ART / "retrieval_diagnosis/phase6d/address_postings.sqlite"
ADDRESS_CANDIDATES = OUT / "address_top10_validation.tsv.gz"
PHASE8 = ART / "model/phase8"
PHASE9 = ART / "model/phase9"
POLICY_PATH = PHASE9 / "decision_policy.json"
GROUND_TRUTH = DEFAULT_DATASET_ROOT / "train/train_ground_truth.tsv"
EXPECTED_VALIDATION_IDS = 220_683
FROZEN_PATHS = (
    V1_CANDIDATES, V1_METADATA, TARGET_INDEX, ADDRESS_INDEX,
    PHASE8 / "feature_manifest.json", PHASE8 / "model_s2.txt", PHASE8 / "model_s3.txt",
    POLICY_PATH,
)


def peak_rss_mb() -> float:
    """Return process peak RSS in MiB on Linux."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def frozen_snapshot(paths: Iterable[Path] = FROZEN_PATHS) -> dict[str, dict[str, int]]:
    """Cheap immutability guard for frozen inputs (no multi-GB rehashing)."""
    return {str(path): {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in paths}


def assert_frozen_unchanged(before: Mapping[str, Mapping[str, int]]) -> None:
    after = frozen_snapshot(Path(path) for path in before)
    if dict(before) != after:
        raise RuntimeError("a frozen Phase 4--9 artifact changed during validation evaluation")


def validation_ids(path: Path = VAL_IDS) -> list[str]:
    ids = load_id_file(path)
    if len(ids) != EXPECTED_VALIDATION_IDS:
        raise ValueError(f"expected {EXPECTED_VALIDATION_IDS} validation IDs, got {len(ids)}")
    if ids != sorted(ids):
        raise ValueError("validation IDs must be sorted for deterministic candidate streams")
    if any(not value.startswith("S1-") for value in ids):
        raise ValueError("validation ID file contains a non-S1 ID")
    return ids


def load_raw_validation_s1(dataset_root: Path, ids: list[str]) -> dict[str, dict[str, str]]:
    """Load raw S1 fields required by the existing Phase 6E retrieval function."""
    wanted = set(ids)
    rows: dict[str, dict[str, str]] = {}
    path = dataset_root / "train/train_source1.tsv"
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"entity_id", "business_name", "business_address", "country"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"unexpected Source 1 columns in {path}")
        for row in reader:
            entity_id = row["entity_id"]
            if entity_id in wanted:
                rows[entity_id] = row
    if rows.keys() != wanted:
        raise ValueError(f"Source 1 coverage mismatch: missing {len(wanted - rows.keys())} records")
    return rows


def frozen_policy(path: Path = POLICY_PATH) -> tuple[DecisionPolicy, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    policy_data = payload.get("policy", {})
    policy = DecisionPolicy(
        s2_threshold=float(policy_data["s2_threshold"]),
        s3_threshold=float(policy_data["s3_threshold"]),
        open_threshold=policy_data.get("open_threshold"),
        conflict_policy=str(policy_data["conflict_policy"]),
        conflict_margin=policy_data.get("conflict_margin"),
    )
    if (policy.s2_threshold, policy.s3_threshold, policy.open_threshold,
            policy.conflict_policy, policy.conflict_margin) != (0.93, 0.97, None, "highest", None):
        raise ValueError("frozen Phase 9 policy does not match the approved validation policy")
    if payload.get("feature_manifest") != str(PHASE8 / "feature_manifest.json"):
        raise ValueError("Phase 9 policy references a different feature manifest")
    expected_models = {source: str(PHASE8 / f"model_{source.lower()}.txt") for source in ("S2", "S3")}
    if payload.get("models") != expected_models:
        raise ValueError("Phase 9 policy references different model artifacts")
    return policy, payload


def validate_phase8_artifacts() -> dict:
    manifest_path = PHASE8 / "feature_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("feature_count") != 48 or tuple(manifest.get("feature_order", ())) != FEATURE_NAMES:
        raise ValueError("saved feature manifest does not match the Phase 8 48-feature schema")
    models = load_models(PHASE8)
    if any(tuple(models[source].feature_name()) != FEATURE_NAMES for source in ("S2", "S3")):
        raise ValueError("saved model feature order differs from the frozen manifest")
    policy, _ = frozen_policy()
    return {"feature_count": len(FEATURE_NAMES), "policy": asdict(policy),
            "manifest": str(manifest_path), "models": {s: str(PHASE8 / f"model_{s.lower()}.txt") for s in ("S2", "S3")}}


def prepare_address(output_dir: Path = OUT, dataset_root: Path = DEFAULT_DATASET_ROOT) -> dict:
    """Generate only the missing validation Address K10 artifact."""
    before = frozen_snapshot()
    for path in (V1_CANDIDATES, V1_METADATA, TARGET_INDEX, ADDRESS_INDEX):
        if not path.is_file():
            raise FileNotFoundError(path)
    ids = validation_ids()
    raw_s1 = load_raw_validation_s1(dataset_root, ids)
    output_dir.mkdir(parents=True, exist_ok=True)
    address_path = output_dir / ADDRESS_CANDIDATES.name
    started = time.perf_counter()
    retrieval = address_retrieve(ids, raw_s1, TARGET_INDEX, ADDRESS_INDEX, address_path, top_k=10)
    assert_frozen_unchanged(before)
    result = {
        "stage": "prepare-address", "validation_s1_count": len(ids),
        "v1_candidates_reused": str(V1_CANDIDATES), "v1_metadata_reused": str(V1_METADATA),
        "address_index_reused": str(ADDRESS_INDEX), "address_candidates": str(address_path),
        "address_k": 10, "retrieval": retrieval,
        "runtime_seconds": time.perf_counter() - started, "peak_rss_mb": peak_rss_mb(),
        "frozen_artifacts_unchanged": True,
    }
    (output_dir / "address_generation_report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def score_validation(output_dir: Path = OUT, dataset_root: Path = DEFAULT_DATASET_ROOT) -> dict:
    """Score every V1 union Address K10 candidate with frozen Phase 8 models."""
    before = frozen_snapshot()
    ids = validation_ids()
    validate_phase8_artifacts()
    address_path = output_dir / ADDRESS_CANDIDATES.name
    if not address_path.is_file():
        raise FileNotFoundError(f"run prepare-address first: {address_path}")
    counts = union_candidate_counts(ids, V1_CANDIDATES, V1_METADATA, address_path)
    if sum(sum(value.values()) for value in counts.values()) == 0:
        raise ValueError("validation candidate union is empty")
    s1_records = load_s1_records(dataset_root, ids)
    models = load_models(PHASE8)
    score_paths = {source: output_dir / f"validation_scores_{source.lower()}.tsv.gz" for source in ("S2", "S3")}
    rows = Counter()
    started = time.perf_counter()
    store = TargetStore(TARGET_INDEX)
    try:
        with gzip.open(score_paths["S2"], "wt", encoding="utf-8", newline="") as s2_handle, \
             gzip.open(score_paths["S3"], "wt", encoding="utf-8", newline="") as s3_handle:
            writers = {source: csv.DictWriter(handle, fieldnames=SCORE_HEADER, delimiter="\t", lineterminator="\n")
                       for source, handle in (("S2", s2_handle), ("S3", s3_handle))}
            for writer in writers.values():
                writer.writeheader()
            for position, (s1_id, pairs) in enumerate(iter_union_groups(ids, V1_CANDIDATES, V1_METADATA, address_path), 1):
                target_ids = sorted(pairs)
                targets = store.get_many(target_ids) if target_ids else {}
                for source in ("S2", "S3"):
                    group = [pairs[target] for target in target_ids if pairs[target]["target_source"] == source]
                    if not group:
                        continue
                    features = np.vstack([
                        extract_features(s1_records[s1_id], targets[pair["candidate_entity_id"]], pair,
                                         counts[s1_id][source]) for pair in group
                    ])
                    probabilities = models[source].predict(features)
                    for pair, probability in zip(group, probabilities):
                        writers[source].writerow({
                            "source1_entity_id": s1_id, "candidate_entity_id": pair["candidate_entity_id"],
                            "score": f"{float(probability):.10f}", "from_v1": pair["from_v1"],
                            "from_address": pair["from_address"], "address_rank": pair["address_rank"],
                            "address_score": pair["address_score"],
                        })
                        rows[source] += 1
                if position % 10_000 == 0:
                    print(f"Scored {position:,}/{len(ids):,} validation S1; pairs={sum(rows.values()):,}", flush=True)
    finally:
        store.close()
    assert_frozen_unchanged(before)
    result = {
        "stage": "score", "validation_s1_count": len(ids), "rows": dict(rows),
        "candidate_pairs": sum(rows.values()), "score_paths": {s: str(p) for s, p in score_paths.items()},
        "score_bytes": {s: p.stat().st_size for s, p in score_paths.items()},
        "runtime_seconds": time.perf_counter() - started, "peak_rss_mb": peak_rss_mb(),
        "target_rows_fetched": store.rows_fetched, "target_queries": store.queries,
        "candidate_membership": "scored pair set equals V1 union Address K10", "frozen_artifacts_unchanged": True,
    }
    (output_dir / "scoring_report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def validate_scored_membership(ids: list[str], address_path: Path, s2, s3,
                               v1_candidates: Path = V1_CANDIDATES,
                               v1_metadata: Path = V1_METADATA) -> None:
    if s2.s1_ids != tuple(ids) or s3.s1_ids != tuple(ids):
        raise ValueError("validation score S1 coverage/order mismatch")
    for index, (_, candidates) in enumerate(iter_union_groups(ids, v1_candidates, v1_metadata, address_path)):
        expected_s2 = {target for target, pair in candidates.items() if pair["target_source"] == "S2"}
        expected_s3 = {target for target, pair in candidates.items() if pair["target_source"] == "S3"}
        scored_s2 = {value.decode("ascii") for value in s2.group(index)[0]}
        scored_s3 = {value.decode("ascii") for value in s3.group(index)[0]}
        if expected_s2 != scored_s2 or expected_s3 != scored_s3:
            raise ValueError(f"score/candidate pair mismatch for {ids[index]}")


def candidate_metrics(ids: list[str], truth: Mapping[str, set[str]], s2, s3) -> dict:
    source_totals = {source: {"true_links": 0, "recovered": 0, "blocking_misses": 0, "candidate_pairs": 0}
                     for source in ("S2", "S3")}
    candidate_sizes: list[int] = []
    for index, s1_id in enumerate(ids):
        targets2, _ = s2.group(index); targets3, _ = s3.group(index)
        candidates2 = {value.decode("ascii") for value in targets2}
        candidates3 = {value.decode("ascii") for value in targets3}
        candidate_sizes.append(len(candidates2) + len(candidates3))
        for source, candidates in (("S2", candidates2), ("S3", candidates3)):
            actual = {target for target in truth[s1_id] if target.startswith(source + "-")}
            hit = len(actual & candidates)
            item = source_totals[source]
            item["true_links"] += len(actual); item["recovered"] += hit
            item["blocking_misses"] += len(actual) - hit; item["candidate_pairs"] += len(candidates)
    total_true = sum(v["true_links"] for v in source_totals.values())
    total_hit = sum(v["recovered"] for v in source_totals.values())
    for item in source_totals.values():
        item["candidate_recall"] = item["recovered"] / item["true_links"] if item["true_links"] else 0.0
    return {
        "overall": {"true_links": total_true, "recovered": total_hit,
                    "blocking_misses": total_true - total_hit,
                    "candidate_recall": total_hit / total_true if total_true else 0.0,
                    "candidate_pairs": sum(x["candidate_pairs"] for x in source_totals.values())},
        "by_source": source_totals,
        "candidate_count_per_s1": {
            "mean": float(np.mean(candidate_sizes)), "median": float(np.percentile(candidate_sizes, 50)),
            "p95": float(np.percentile(candidate_sizes, 95)), "p99": float(np.percentile(candidate_sizes, 99)),
            "max": max(candidate_sizes, default=0), "zero_candidate_s1": sum(v == 0 for v in candidate_sizes),
        },
    }


def evaluate_validation(output_dir: Path = OUT) -> dict:
    before = frozen_snapshot()
    started = time.perf_counter()
    ids = validation_ids()
    address_path = output_dir / ADDRESS_CANDIDATES.name
    score_report = json.loads((output_dir / "scoring_report.json").read_text(encoding="utf-8"))
    s2 = read_score_file(output_dir / "validation_scores_s2.tsv.gz", "S2", ids, int(score_report["rows"]["S2"]))
    s3 = read_score_file(output_dir / "validation_scores_s3.tsv.gz", "S3", ids, int(score_report["rows"]["S3"]))
    validate_scored_membership(ids, address_path, s2, s3)
    truth = load_ground_truth(GROUND_TRUTH, ids)
    candidate_report = candidate_metrics(ids, truth, s2, s3)
    policy, _ = frozen_policy()
    result = evaluate_policy(s2, s3, truth, policy, candidate_report["overall"]["blocking_misses"])
    predictions = result.pop("predictions")
    s2_counts = LinkCounts(*(sum(getattr(count_link_errors(truth[s1], predictions[s1], "S2-"), k)
                                 for s1 in ids) for k in ("tp", "fp", "fn")))
    s3_counts = LinkCounts(*(sum(getattr(count_link_errors(truth[s1], predictions[s1], "S3-"), k)
                                 for s1 in ids) for k in ("tp", "fp", "fn")))
    pred_cardinality = Counter(len(predictions[s1]) for s1 in ids)
    prediction_path = output_dir / "validation_predictions.tsv.gz"
    write_predictions(prediction_path, predictions, ids)
    source_counts = {}
    for source, counts in (("S2", s2_counts), ("S3", s3_counts)):
        precision, recall = precision_recall_diagnostics(counts)
        source_counts[source] = {**asdict(counts), "precision_diagnostic": precision, "recall_diagnostic": recall}
    address_report_path = output_dir / "address_generation_report.json"
    address_report = json.loads(address_report_path.read_text(encoding="utf-8")) if address_report_path.exists() else {}
    summary = {
        "phase": "held-out-validation-evaluation", "scope": "single frozen-policy evaluation; no tuning or test inference",
        "validation_s1_count": len(ids), "frozen_policy": asdict(policy),
        "candidate_metrics": candidate_report,
        "prediction_metrics": {**result, "S2": source_counts["S2"], "S3": source_counts["S3"],
                               "predicted_cardinality_distribution": {str(k): pred_cardinality[k] for k in sorted(pred_cardinality)}},
        "singleton_metrics": {"true": result["true_singletons"], "predicted": result["predicted_singletons"],
                              "correct": result["correctly_predicted_singletons"],
                              "false_merges": result["singleton_false_merges"], "accuracy": result["singleton_accuracy"]},
        "blocking_misses": candidate_report["overall"]["blocking_misses"],
        "model_decision_misses": result["decision_misses"],
        "files": {
            "address_candidates": str(address_path), "scores_s2": str(output_dir / "validation_scores_s2.tsv.gz"),
            "scores_s3": str(output_dir / "validation_scores_s3.tsv.gz"), "predictions": str(prediction_path),
        },
        "disk_bytes": {
            "address_candidates": address_path.stat().st_size,
            "scores_s2": (output_dir / "validation_scores_s2.tsv.gz").stat().st_size,
            "scores_s3": (output_dir / "validation_scores_s3.tsv.gz").stat().st_size,
            "predictions": prediction_path.stat().st_size,
        },
        "runtime_seconds_evaluation": time.perf_counter() - started,
        "runtime_seconds_address_generation": address_report.get("runtime_seconds"),
        "runtime_seconds_scoring": score_report.get("runtime_seconds"),
        "evaluation_peak_rss_mb": peak_rss_mb(), "address_generation_peak_rss_mb": address_report.get("peak_rss_mb"),
        "scoring_peak_rss_mb": score_report.get("peak_rss_mb"),
        "candidate_pairs": candidate_report["overall"]["candidate_pairs"],
        "runtime_seconds_total": float((address_report.get("runtime_seconds") or 0.0) + (score_report.get("runtime_seconds") or 0.0) + (time.perf_counter() - started)),
        "peak_rss_mb": max(float(address_report.get("peak_rss_mb") or 0.0), float(score_report.get("peak_rss_mb") or 0.0), peak_rss_mb()),
        "disk_bytes_total": sum((output_dir / name).stat().st_size for name in (ADDRESS_CANDIDATES.name, "validation_scores_s2.tsv.gz", "validation_scores_s3.tsv.gz", prediction_path.name)),
        "score_rows": score_report["rows"], "policy_artifact": str(POLICY_PATH),
    }
    summary_path = output_dir / "validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "validation_summary.txt").write_text(format_summary(summary), encoding="utf-8")
    assert_frozen_unchanged(before)
    return summary


def format_summary(summary: Mapping) -> str:
    metrics = summary["prediction_metrics"]
    candidate = summary["candidate_metrics"]
    lines = [
        "HELD-OUT VALIDATION EVALUATION", "",
        f"S1 entities: {summary['validation_s1_count']:,}",
        f"Candidate pairs: {summary['candidate_pairs']:,}",
        f"Candidate recall: {candidate['overall']['candidate_recall']:.6%}",
        f"S2 candidate recall: {candidate['by_source']['S2']['candidate_recall']:.6%}",
        f"S3 candidate recall: {candidate['by_source']['S3']['candidate_recall']:.6%}",
        f"Macro entity-level F0.5: {metrics['macro_f0_5']:.8f}",
        f"TP / FP / FN: {metrics['tp']:,} / {metrics['fp']:,} / {metrics['fn']:,}",
        f"Precision / recall diagnostics: {metrics['micro_precision_diagnostic']:.6f} / {metrics['micro_recall_diagnostic']:.6f}",
        f"Blocking misses / model-decision misses: {summary['blocking_misses']:,} / {summary['model_decision_misses']:,}",
        f"Singleton true / predicted / correct / false merges: {summary['singleton_metrics']['true']:,} / {summary['singleton_metrics']['predicted']:,} / {summary['singleton_metrics']['correct']:,} / {summary['singleton_metrics']['false_merges']:,}",
        f"Singleton accuracy: {summary['singleton_metrics']['accuracy']:.6f}",
        f"Predictions per S1 mean / P95 / P99 / max: {metrics['predicted_mean']:.4f} / {metrics['predicted_p95']:.1f} / {metrics['predicted_p99']:.1f} / {metrics['predicted_max']}",
        "Prediction cardinality histogram: " + ", ".join(f"{k}={v}" for k, v in metrics["predicted_cardinality_distribution"].items()),
        f"Runtime evaluation / score / Address K10: {summary['runtime_seconds_evaluation']:.1f}s / {summary['runtime_seconds_scoring']:.1f}s / {summary['runtime_seconds_address_generation']:.1f}s",
        f"Peak RSS overall / evaluation: {summary['peak_rss_mb']:.1f} / {summary['evaluation_peak_rss_mb']:.1f} MiB",
        f"Output disk total: {summary['disk_bytes_total']:,} bytes",
        "", "Source diagnostics (precision/recall are micro diagnostics):",
    ]
    for source in ("S2", "S3"):
        item = metrics[source]
        lines.append(f"{source}: TP={item['tp']:,} FP={item['fp']:,} FN={item['fn']:,}; precision={item['precision_diagnostic']:.6f}; recall={item['recall_diagnostic']:.6f}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare-address", "score", "evaluate"))
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    args = parser.parse_args(argv)
    functions = {
        "prepare-address": lambda: prepare_address(args.output_dir, args.dataset_root),
        "score": lambda: score_validation(args.output_dir, args.dataset_root),
        "evaluate": lambda: evaluate_validation(args.output_dir),
    }
    print(json.dumps(functions[args.command](), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
