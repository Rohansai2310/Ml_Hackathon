#!/usr/bin/env python3
"""Tune-only Phase 9 entity-level decision-policy evaluation."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from diagnostics import DEFAULT_DATASET_ROOT
from phase8_model import TUNE_ADDRESS, TUNE_V1_CANDIDATES, TUNE_V1_METADATA, iter_union_groups
from scoring import PREDICTION_HEADER, LinkCounts, count_link_errors, load_ground_truth, load_id_file, precision_recall_diagnostics, score_predictions

BASE = Path(__file__).resolve().parents[1]
ART = BASE / "artifacts"
PHASE8 = ART / "model/phase8"
PHASE9 = ART / "model/phase9"
TUNE_IDS = ART / "splits/tune_s1_ids.txt"
GROUND_TRUTH = DEFAULT_DATASET_ROOT / "train/train_ground_truth.tsv"
SCORE_PATHS = {"S2": PHASE8 / "tune_scores_s2.tsv.gz", "S3": PHASE8 / "tune_scores_s3.tsv.gz"}
EXPECTED_ROWS = {"S2": 7_227_686, "S3": 7_323_701}
SCORE_HEADER = ["source1_entity_id", "candidate_entity_id", "score", "from_v1", "from_address", "address_rank", "address_score"]
SWEEP_HEADER = ["stage", "label", "s2_threshold", "s3_threshold", "open_threshold", "conflict_policy", "conflict_margin", "macro_f0_5", "precision_diag", "recall_diag", "tp", "fp", "fn", "predicted_links", "predicted_mean", "predicted_median", "predicted_p95", "predicted_p99", "predicted_max", "true_singletons", "predicted_singletons", "correct_singletons", "singleton_false_merges", "singleton_accuracy", "s2_precision", "s2_recall", "s3_precision", "s3_recall", "blocking_misses", "decision_misses"]


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


@dataclass(frozen=True)
class DecisionPolicy:
    s2_threshold: float
    s3_threshold: float
    open_threshold: float | None = None
    conflict_policy: str = "none"
    conflict_margin: float | None = None


@dataclass(frozen=True)
class ScoreCache:
    source: str
    s1_ids: tuple[str, ...]
    offsets: np.ndarray
    targets: np.ndarray
    scores: np.ndarray
    from_v1: np.ndarray
    from_address: np.ndarray
    address_rank: np.ndarray
    address_score: np.ndarray

    def group(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return self.targets[self.offsets[index]:self.offsets[index + 1]], self.scores[self.offsets[index]:self.offsets[index + 1]]


def _decode(values: np.ndarray) -> set[str]:
    return {value.decode("ascii") for value in values}


def _number(value: str, field: str, row: int, optional: bool = False) -> float:
    if optional and value == "":
        return 0.0
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"invalid {field} at score row {row}") from exc
    if not math.isfinite(result):
        raise ValueError(f"non-finite {field} at score row {row}")
    return result


def read_score_file(path: Path, source: str, s1_ids: list[str], expected_rows: int) -> ScoreCache:
    """Read a Phase 8 score TSV once into source-local compact arrays."""
    if source not in ("S2", "S3"):
        raise ValueError("source must be S2 or S3")
    id_index = {value: index for index, value in enumerate(s1_ids)}
    targets = np.empty(expected_rows, dtype="S20")
    scores = np.empty(expected_rows, dtype=np.float32)
    from_v1 = np.empty(expected_rows, dtype=np.uint8)
    from_address = np.empty(expected_rows, dtype=np.uint8)
    address_rank = np.empty(expected_rows, dtype=np.float32)
    address_score = np.empty(expected_rows, dtype=np.float32)
    counts = np.zeros(len(s1_ids), dtype=np.int64)
    position, previous = 0, -1
    seen: set[str] = set()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != SCORE_HEADER:
            raise ValueError(f"unexpected score header in {path}: {reader.fieldnames!r}")
        for row_number, row in enumerate(reader, start=2):
            if position >= expected_rows:
                raise ValueError(f"too many score rows in {path}")
            s1_id, target = row["source1_entity_id"], row["candidate_entity_id"]
            index = id_index.get(s1_id)
            if index is None:
                raise ValueError(f"score row has non-tune S1 ID: {s1_id}")
            if index < previous:
                raise ValueError(f"score rows are not sorted by S1 at row {row_number}")
            if index != previous:
                seen.clear(); previous = index
            if target in seen:
                raise ValueError(f"duplicate score pair at row {row_number}: {s1_id},{target}")
            seen.add(target)
            if not target.startswith(source + "-"):
                raise ValueError(f"target source prefix mismatch at row {row_number}: {target}")
            score = _number(row["score"], "score", row_number)
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"score outside [0, 1] at row {row_number}")
            targets[position] = target.encode("ascii")
            scores[position] = score
            from_v1[position] = int(row["from_v1"])
            from_address[position] = int(row["from_address"])
            address_rank[position] = _number(row["address_rank"], "address_rank", row_number, optional=True)
            address_score[position] = _number(row["address_score"], "address_score", row_number, optional=True)
            counts[index] += 1; position += 1
    if position != expected_rows:
        raise ValueError(f"score row count mismatch in {path}: expected={expected_rows}, actual={position}")
    offsets = np.zeros(len(s1_ids) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    return ScoreCache(source, tuple(s1_ids), offsets, targets, scores, from_v1, from_address, address_rank, address_score)


def save_cache(cache: ScoreCache, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, source=np.asarray([cache.source]), s1_ids=np.asarray(cache.s1_ids, dtype="S20"), offsets=cache.offsets, targets=cache.targets, scores=cache.scores, from_v1=cache.from_v1, from_address=cache.from_address, address_rank=cache.address_rank, address_score=cache.address_score)


def load_cache(path: Path) -> ScoreCache:
    with np.load(path, allow_pickle=False) as data:
        return ScoreCache(str(data["source"][0]), tuple(value.decode("ascii") for value in data["s1_ids"]), data["offsets"], data["targets"], data["scores"], data["from_v1"], data["from_address"], data["address_rank"], data["address_score"])


def validate_membership(s2: ScoreCache, s3: ScoreCache) -> None:
    """Verify score-pair sets against the frozen V1 union Address K10 stream."""
    if s2.s1_ids != s3.s1_ids:
        raise ValueError("S2/S3 score caches use different tune ID order")
    for index, (_, candidates) in enumerate(iter_union_groups(list(s2.s1_ids), TUNE_V1_CANDIDATES, TUNE_V1_METADATA, TUNE_ADDRESS)):
        expected_s2 = {target for target, row in candidates.items() if row["target_source"] == "S2"}
        expected_s3 = {target for target, row in candidates.items() if row["target_source"] == "S3"}
        if _decode(s2.group(index)[0]) != expected_s2:
            raise ValueError(f"S2 score/candidate mismatch for {s2.s1_ids[index]}")
        if _decode(s3.group(index)[0]) != expected_s3:
            raise ValueError(f"S3 score/candidate mismatch for {s2.s1_ids[index]}")


def preprocess(output_dir: Path = PHASE9) -> dict:
    start = time.perf_counter()
    ids = load_id_file(TUNE_IDS)
    if len(ids) != 100_000 or len(ids) != len(set(ids)):
        raise ValueError("tune split must contain exactly 100,000 unique IDs")
    caches = {source: read_score_file(SCORE_PATHS[source], source, ids, EXPECTED_ROWS[source]) for source in ("S2", "S3")}
    validate_membership(caches["S2"], caches["S3"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for source, cache in caches.items():
        save_cache(cache, output_dir / f"score_cache_{source.lower()}.npz")
    result = {"s1_count": len(ids), "rows": {source: int(len(cache.targets)) for source, cache in caches.items()}, "score_paths": {source: str(SCORE_PATHS[source]) for source in caches}, "cache_paths": {source: str(output_dir / f"score_cache_{source.lower()}.npz") for source in caches}, "candidate_membership": "exact V1 union Address K10 match", "runtime_seconds": time.perf_counter() - start, "peak_rss_mb": peak_rss_mb()}
    (output_dir / "preprocess_report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _selected_scores(s2: ScoreCache, s3: ScoreCache, policy: DecisionPolicy) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for index, s1_id in enumerate(s2.s1_ids):
        t2, v2 = s2.group(index); t3, v3 = s3.group(index)
        maximum = max(float(v2.max(initial=0.0)), float(v3.max(initial=0.0)))
        selected: dict[str, float] = {}
        if policy.open_threshold is None or maximum >= policy.open_threshold:
            for target, score in zip(t2[v2 >= policy.s2_threshold], v2[v2 >= policy.s2_threshold]):
                selected[target.decode("ascii")] = float(score)
            for target, score in zip(t3[v3 >= policy.s3_threshold], v3[v3 >= policy.s3_threshold]):
                selected[target.decode("ascii")] = float(score)
        result[s1_id] = selected
    return result


def _apply_conflicts(selected: dict[str, dict[str, float]], policy: DecisionPolicy) -> dict[str, set[str]]:
    if policy.conflict_policy == "none":
        return {s1_id: set(values) for s1_id, values in selected.items()}
    by_target: dict[str, list[tuple[str, float]]] = {}
    for s1_id, targets in selected.items():
        for target, score in targets.items():
            by_target.setdefault(target, []).append((s1_id, score))
    predictions = {s1_id: set() for s1_id in selected}
    for target, contenders in by_target.items():
        contenders.sort(key=lambda item: (-item[1], item[0]))
        keep = contenders
        if policy.conflict_policy == "highest":
            keep = contenders[:1]
        elif policy.conflict_policy == "margin":
            margin = policy.conflict_margin if policy.conflict_margin is not None else 0.0
            second = contenders[1][1] if len(contenders) > 1 else float("-inf")
            if contenders[0][1] - second >= margin:
                keep = contenders[:1]
        elif policy.conflict_policy != "none":
            raise ValueError(f"unknown conflict policy: {policy.conflict_policy}")
        for s1_id, _ in keep:
            predictions[s1_id].add(target)
    return predictions


def predictions_for_policy(s2: ScoreCache, s3: ScoreCache, policy: DecisionPolicy) -> dict[str, set[str]]:
    return _apply_conflicts(_selected_scores(s2, s3, policy), policy)


def _percentile(values: list[int], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else 0.0


def blocking_miss_count(s2: ScoreCache, s3: ScoreCache, truth: Mapping[str, set[str]]) -> int:
    """Count immutable retrieval misses once per sweep, not once per threshold."""
    total = 0
    for index, s1_id in enumerate(s2.s1_ids):
        candidates = _decode(s2.group(index)[0]) | _decode(s3.group(index)[0])
        total += len(truth[s1_id] - candidates)
    return total


def evaluate_policy(s2: ScoreCache, s3: ScoreCache, truth: Mapping[str, set[str]], policy: DecisionPolicy,
                    blocking_misses: int | None = None) -> dict:
    """Evaluate with scoring.py's exact macro entity-level F0.5 implementation."""
    predictions = predictions_for_policy(s2, s3, policy)
    report = score_predictions(truth, predictions, s2.s1_ids)
    s2_counts = [count_link_errors(truth[s1], predictions[s1], "S2-") for s1 in s2.s1_ids]
    s3_counts = [count_link_errors(truth[s1], predictions[s1], "S3-") for s1 in s2.s1_ids]
    s2_total = LinkCounts(*(sum(getattr(value, key) for value in s2_counts) for key in ("tp", "fp", "fn")))
    s3_total = LinkCounts(*(sum(getattr(value, key) for value in s3_counts) for key in ("tp", "fp", "fn")))
    s2_precision, s2_recall = precision_recall_diagnostics(s2_total)
    s3_precision, s3_recall = precision_recall_diagnostics(s3_total)
    sizes = [len(predictions[s1]) for s1 in s2.s1_ids]
    if blocking_misses is None:
        blocking_misses = blocking_miss_count(s2, s3, truth)
    decision_misses = report.fn - blocking_misses
    if decision_misses < 0:
        raise AssertionError("blocking misses exceed exact scorer FN count")
    return {**asdict(report), "predicted_singletons": sum(size == 0 for size in sizes), "singleton_false_merges": sum(not truth[s1] and bool(predictions[s1]) for s1 in s2.s1_ids), "predicted_mean": float(np.mean(sizes)), "predicted_median": _percentile(sizes, 50), "predicted_p95": _percentile(sizes, 95), "predicted_p99": _percentile(sizes, 99), "predicted_max": max(sizes, default=0), "s2_precision": s2_precision, "s2_recall": s2_recall, "s3_precision": s3_precision, "s3_recall": s3_recall, "blocking_misses": blocking_misses, "decision_misses": decision_misses, "policy": asdict(policy), "predictions": predictions}


def _row(stage: str, label: str, result: Mapping[str, object]) -> dict[str, object]:
    policy = result["policy"]
    assert isinstance(policy, dict)
    return {"stage": stage, "label": label, "s2_threshold": policy["s2_threshold"], "s3_threshold": policy["s3_threshold"], "open_threshold": "" if policy["open_threshold"] is None else policy["open_threshold"], "conflict_policy": policy["conflict_policy"], "conflict_margin": "" if policy["conflict_margin"] is None else policy["conflict_margin"], "macro_f0_5": result["macro_f0_5"], "precision_diag": result["micro_precision_diagnostic"], "recall_diag": result["micro_recall_diagnostic"], "tp": result["tp"], "fp": result["fp"], "fn": result["fn"], "predicted_links": result["predicted_links"], "predicted_mean": result["predicted_mean"], "predicted_median": result["predicted_median"], "predicted_p95": result["predicted_p95"], "predicted_p99": result["predicted_p99"], "predicted_max": result["predicted_max"], "true_singletons": result["true_singletons"], "predicted_singletons": result["predicted_singletons"], "correct_singletons": result["correctly_predicted_singletons"], "singleton_false_merges": result["singleton_false_merges"], "singleton_accuracy": result["singleton_accuracy"], "s2_precision": result["s2_precision"], "s2_recall": result["s2_recall"], "s3_precision": result["s3_precision"], "s3_recall": result["s3_recall"], "blocking_misses": result["blocking_misses"], "decision_misses": result["decision_misses"]}


def _load_inputs(output_dir: Path) -> tuple[ScoreCache, ScoreCache, dict[str, set[str]]]:
    s2, s3 = load_cache(output_dir / "score_cache_s2.npz"), load_cache(output_dir / "score_cache_s3.npz")
    if s2.s1_ids != s3.s1_ids:
        raise ValueError("cache S1 order mismatch")
    return s2, s3, load_ground_truth(GROUND_TRUTH, s2.s1_ids)


def _write_sweep(path: Path, rows: Iterable[dict[str, object]]) -> None:
    existing: list[dict[str, str]] = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle, delimiter="\t"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SWEEP_HEADER, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(existing); writer.writerows(rows)


def _rank(row: Mapping[str, object]) -> tuple[float, float, int, int]:
    return (float(row["macro_f0_5"]), float(row["precision_diag"]), -int(row["singleton_false_merges"]), -int(row["predicted_links"]))


def _best_sweep_row(path: Path, stages: set[str]) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle, delimiter="\t") if row["stage"] in stages]
    if not rows:
        raise ValueError(f"no sweep rows found for {sorted(stages)}")
    return max(rows, key=_rank)


def run_policies(stage: str, policies: Iterable[DecisionPolicy], output_dir: Path = PHASE9) -> list[dict[str, object]]:
    s2, s3, truth = _load_inputs(output_dir)
    misses = blocking_miss_count(s2, s3, truth)
    rows: list[dict[str, object]] = []
    for number, policy in enumerate(policies, start=1):
        rows.append(_row(stage, f"{stage}_{number:03d}", evaluate_policy(s2, s3, truth, policy, misses)))
    _write_sweep(output_dir / "threshold_sweep.tsv", rows)
    return rows


def coarse(output_dir: Path = PHASE9) -> list[dict[str, object]]:
    values = (0.50, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995)
    return run_policies("coarse_shared", (DecisionPolicy(value, value) for value in values), output_dir)


def fine(output_dir: Path = PHASE9) -> list[dict[str, object]]:
    center = float(_best_sweep_row(output_dir / "threshold_sweep.tsv", {"coarse_shared"})["s2_threshold"])
    shared = [DecisionPolicy(round(value, 3), round(value, 3)) for value in np.arange(max(0.0, center - 0.02), min(1.0, center + 0.02) + 0.0001, 0.001)]
    local = [round(min(1.0, max(0.0, center + delta)), 3) for delta in (-0.02, -0.01, 0.0, 0.01, 0.02)]
    return run_policies("fine", [*shared, *(DecisionPolicy(left, right) for left in local for right in local)], output_dir)


def open_experiment(output_dir: Path = PHASE9) -> list[dict[str, object]]:
    best = _best_sweep_row(output_dir / "threshold_sweep.tsv", {"coarse_shared", "fine"})
    values = (0.50, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99)
    return run_policies("open", (DecisionPolicy(float(best["s2_threshold"]), float(best["s3_threshold"]), value) for value in values), output_dir)


def conflict_experiment(output_dir: Path = PHASE9) -> list[dict[str, object]]:
    best = _best_sweep_row(output_dir / "threshold_sweep.tsv", {"coarse_shared", "fine", "open"})
    open_value = None if best["open_threshold"] == "" else float(best["open_threshold"])
    s2, s3 = float(best["s2_threshold"]), float(best["s3_threshold"])
    policies = [DecisionPolicy(s2, s3, open_value), DecisionPolicy(s2, s3, open_value, "highest")]
    policies.extend(DecisionPolicy(s2, s3, open_value, "margin", margin) for margin in (0.01, 0.02, 0.05))
    return run_policies("conflict", policies, output_dir)


def write_predictions(path: Path, predictions: Mapping[str, set[str]], s1_ids: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(PREDICTION_HEADER)
        for s1_id in s1_ids:
            writer.writerow([s1_id, ",".join(sorted(predictions[s1_id]))])


def freeze(output_dir: Path = PHASE9) -> dict:
    best = _best_sweep_row(output_dir / "threshold_sweep.tsv", {"coarse_shared", "fine", "open", "conflict"})
    policy = DecisionPolicy(float(best["s2_threshold"]), float(best["s3_threshold"]), None if best["open_threshold"] == "" else float(best["open_threshold"]), best["conflict_policy"], None if best["conflict_margin"] == "" else float(best["conflict_margin"]))
    s2, s3, truth = _load_inputs(output_dir)
    result = evaluate_policy(s2, s3, truth, policy, blocking_miss_count(s2, s3, truth))
    predictions_path = output_dir / "tune_predictions.tsv.gz"
    write_predictions(predictions_path, result.pop("predictions"), s2.s1_ids)
    payload = {"policy": asdict(policy), "selection_rule": "macro F0.5, precision, fewer singleton false merges, fewer predicted links", "selected_metrics": result, "models": {source: str(PHASE8 / f"model_{source.lower()}.txt") for source in ("S2", "S3")}, "feature_manifest": str(PHASE8 / "feature_manifest.json"), "score_paths": {source: str(SCORE_PATHS[source]) for source in SCORE_PATHS}, "retrieval": {"v1_candidates": str(TUNE_V1_CANDIDATES), "v1_metadata": str(TUNE_V1_METADATA), "address_k10": str(TUNE_ADDRESS)}, "tune_predictions": str(predictions_path)}
    (output_dir / "decision_policy.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def report(output_dir: Path = PHASE9) -> dict:
    policy = json.loads((output_dir / "decision_policy.json").read_text(encoding="utf-8"))
    integrity = json.loads((output_dir / "preprocess_report.json").read_text(encoding="utf-8"))
    caches = {source: output_dir / f"score_cache_{source.lower()}.npz" for source in ("S2", "S3")}
    payload = {"phase": 9, "policy": policy, "input_integrity": integrity,
               "candidate_recall_context": {"overall": 0.930380, "S2": 0.930718, "S3": 0.930064},
               "cache_bytes": {source: path.stat().st_size for source, path in caches.items()},
               "scope": "Tune-only; no validation, test inference, or submission."}
    (output_dir / "phase9_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preprocess", "sweep-coarse", "sweep-fine", "experiment-open", "experiment-conflict", "freeze", "report"))
    parser.add_argument("--output-dir", type=Path, default=PHASE9)
    args = parser.parse_args(argv)
    functions = {"preprocess": preprocess, "sweep-coarse": coarse, "sweep-fine": fine, "experiment-open": open_experiment, "experiment-conflict": conflict_experiment, "freeze": freeze, "report": report}
    print(json.dumps(functions[args.command](args.output_dir), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
