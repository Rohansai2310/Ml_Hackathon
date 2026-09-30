#!/usr/bin/env python3
"""Phase 11.2A: tune-selected grouped decision policies; never trains or retrieves."""
from __future__ import annotations

import argparse
import csv
import hashlib
import gzip
import json
import math
import os
import resource
import sqlite3
import time
from collections import defaultdict
from itertools import groupby
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from diagnostics import DEFAULT_DATASET_ROOT
from phase9_policy import (
    DecisionPolicy, ScoreCache, _apply_conflicts, evaluate_policy, load_cache,
    read_score_file,
)
from scoring import load_ground_truth, load_id_file, score_predictions

BASE = Path(__file__).resolve().parents[1]
ART = BASE / "artifacts"
OUT = ART / "phase112_group_policy"
PHASE9 = ART / "model/phase9"
PHASE8 = ART / "model/phase8"
PHASE12 = ART / "test_inference/phase12"
VAL = ART / "validation_evaluation"
SPLITS = ART / "splits"
GROUND_TRUTH = DEFAULT_DATASET_ROOT / "train/train_ground_truth.tsv"
POLICY_PATH = PHASE9 / "decision_policy.json"
TUNE_CACHE = {s: PHASE9 / f"score_cache_{s.lower()}.npz" for s in ("S2", "S3")}
VAL_SCORE = {s: VAL / f"validation_scores_{s.lower()}.tsv.gz" for s in ("S2", "S3")}
EXPECTED_TUNE = {"S2": 7_227_686, "S3": 7_323_701}
EXPECTED_VAL = {"S2": 16_098_984, "S3": 16_233_279}
BASELINE_MACRO = 0.8676284617025911
CONFIRMED_VALIDATION_MACRO = 0.8731357909723747
BASELINE_SUBMISSION_SHA256 = "c7bed709aa2058e1fa08467096e1c2afca0f6d820baab0749bdd88b73ffa2da8"
BASE_POLICY = DecisionPolicy(.93, .97, None, "highest", None)
COUNT_BINS = ((1, 25, "1-25"), (26, 100, "26-100"), (101, 500, "101-500"), (501, 10**18, "501+"))
FROZEN_PATHS = (
    PHASE8 / "model_s2.txt", PHASE8 / "model_s3.txt", PHASE8 / "feature_manifest.json",
    TUNE_CACHE["S2"], TUNE_CACHE["S3"], VAL_SCORE["S2"], VAL_SCORE["S3"],
    POLICY_PATH, PHASE12 / "candidates/candidate_pairs_long.tsv.gz",
    PHASE12 / "candidates/candidate_summary.json", PHASE12 / "scores/score_manifest.json",
    BASE.parents[1] / "output/candidate_pairs.tsv", BASE.parents[1] / "output/matching_results.tsv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_snapshot() -> dict[str, dict[str, object]]:
    """Hash small files; use size/mtime for multi-GB files and their hash manifests."""
    result = {}
    for path in FROZEN_PATHS:
        if not path.is_file():
            result[str(path)] = {"exists": False}
            continue
        stat = path.stat()
        value: dict[str, object] = {"exists": True, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if stat.st_size <= 100_000_000:
            value["sha256"] = _sha256(path)
        result[str(path)] = value
    return result


def _assert_snapshot(before: Mapping[str, Mapping[str, object]]) -> None:
    if dict(before) != frozen_snapshot():
        raise RuntimeError("frozen Phase 8/9/12 artifacts changed during Phase 11.2A")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _policy_raw() -> dict:
    raw = json.loads(POLICY_PATH.read_text(encoding="utf-8"))["policy"]
    actual = DecisionPolicy(float(raw["s2_threshold"]), float(raw["s3_threshold"]), raw.get("open_threshold"), raw["conflict_policy"], raw.get("conflict_margin"))
    if actual != BASE_POLICY:
        raise ValueError(f"Phase 11.2A requires frozen policy {asdict(BASE_POLICY)}, found {asdict(actual)}")
    return raw


def _tune_inputs() -> tuple[ScoreCache, ScoreCache, dict[str, set[str]]]:
    ids = load_id_file(SPLITS / "tune_s1_ids.txt")
    if len(ids) != 100_000:
        raise ValueError(f"expected 100,000 tune S1 IDs, got {len(ids)}")
    s2, s3 = (load_cache(TUNE_CACHE[s]) for s in ("S2", "S3"))
    for source, cache in (("S2", s2), ("S3", s3)):
        if cache.source != source or cache.s1_ids != tuple(ids) or len(cache.targets) != EXPECTED_TUNE[source]:
            raise ValueError(f"Phase 9 {source} cache order/count mismatch")
    return s2, s3, load_ground_truth(GROUND_TRUTH, ids)


def _val_inputs() -> tuple[ScoreCache, ScoreCache, dict[str, set[str]]]:
    ids = load_id_file(SPLITS / "val_s1_ids.txt")
    if ids != sorted(ids) or len(ids) != 220_683:
        raise ValueError("validation S1 ID list does not match frozen 220,683-ID ordering")
    caches = {s: read_score_file(VAL_SCORE[s], s, ids, EXPECTED_VAL[s]) for s in ("S2", "S3")}
    return caches["S2"], caches["S3"], load_ground_truth(GROUND_TRUTH, ids)


def inventory(output_dir: Path = OUT) -> dict:
    before = frozen_snapshot()
    _policy_raw()
    ids = load_id_file(SPLITS / "tune_s1_ids.txt")
    caches = {s: load_cache(TUNE_CACHE[s]) for s in ("S2", "S3")}
    value = {
        "phase": "11.2A", "selection_split": "tune only", "tune_s1": len(ids),
        "tune_rows": {s: len(caches[s].targets) for s in caches},
        "tune_score_cache_bytes": {s: TUNE_CACHE[s].stat().st_size for s in caches},
        "validation_score_files": {s: {"path": str(VAL_SCORE[s]), "exists": VAL_SCORE[s].is_file(), "bytes": VAL_SCORE[s].stat().st_size if VAL_SCORE[s].exists() else None} for s in VAL_SCORE},
        "validation_s1_expected": 220_683, "frozen_policy": _policy_raw(),
        "phase12_candidate_summary": str(PHASE12 / "candidates/candidate_summary.json"),
        "phase12_score_manifest": str(PHASE12 / "scores/score_manifest.json"),
        "frozen_snapshot": before, "peak_rss_mb": _rss_mb(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "phase112_inventory.json", value)
    _assert_snapshot(before)
    return value


def baseline_check(output_dir: Path = OUT) -> dict:
    before = frozen_snapshot()
    s2, s3, truth = _tune_inputs()
    result = evaluate_policy(s2, s3, truth, BASE_POLICY)
    if result["macro_f0_5"] != BASELINE_MACRO:
        raise RuntimeError(f"baseline mismatch: {result['macro_f0_5']!r} != {BASELINE_MACRO!r}")
    report = {
        "macro_f0_5": result["macro_f0_5"], "expected_macro_f0_5": BASELINE_MACRO,
        "exact": True, "policy": asdict(BASE_POLICY), "s1_count": len(s2.s1_ids),
        "candidate_rows": {"S2": len(s2.targets), "S3": len(s3.targets)},
        "tp": result["tp"], "fp": result["fp"], "fn": result["fn"],
        "phase12_candidate_hash_reference": _read_phase12_candidate_hash(),
        "frozen_snapshot": before, "frozen_artifacts_unchanged": True,
    }
    _atomic_json(output_dir / "baseline_reproduction.json", report)
    _assert_snapshot(before)
    return report


def _read_phase12_candidate_hash() -> str | None:
    path = PHASE12 / "candidates/candidate_summary.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value.get("membership_sha256") or value.get("sha256")


def _bin_name(count: int) -> str:
    if count <= 0:
        return "0"
    for lower, upper, name in COUNT_BINS:
        if lower <= count <= upper:
            return name
    raise AssertionError(count)


def _ordered(targets: np.ndarray, scores: np.ndarray) -> np.ndarray:
    # Stable policy rank: highest score, then lexicographically smallest ID.
    return np.lexsort((targets, -scores))


def _base_threshold(source: str) -> float:
    return BASE_POLICY.s2_threshold if source == "S2" else BASE_POLICY.s3_threshold


def _candidate_count(cache: ScoreCache, index: int) -> int:
    return int(cache.offsets[index + 1] - cache.offsets[index])


@dataclass(frozen=True)
class GroupPolicy:
    """Serializable policy description; thresholds remain anchored to Phase 9."""
    kind: str
    params: Mapping[str, object]


def shared_offset_grid() -> list[GroupPolicy]:
    return [GroupPolicy("A", {"low_offset": low, "crowded_offset": crowded})
            for low in (-.015, -.010, -.005, 0.0)
            for crowded in (0.0, .005, .010, .015, .020)]


def _a_refinements(best: GroupPolicy) -> list[GroupPolicy]:
    low, crowded = float(best.params["low_offset"]), float(best.params["crowded_offset"])
    return [GroupPolicy("A_source", {"base_low_offset": low, "base_crowded_offset": crowded,
                                      "source_offsets": {"S2": d2, "S3": d3}})
            for d2 in (-.005, 0.0, .005) for d3 in (-.005, 0.0, .005)
            if (d2, d3) != (0.0, 0.0)]


def gap_grid() -> list[GroupPolicy]:
    rows = [GroupPolicy("B_top_gap", {"relax": relax, "minimum_gap": gap})
            for relax in (.005, .010, .020) for gap in (.01, .02, .03, .05, .10)]
    rows.extend(GroupPolicy("B_crowded_tail", {"tail_raise": delta}) for delta in (.005, .010, .015))
    return rows


def _threshold(cache: ScoreCache, index: int, local: int, policy: GroupPolicy) -> float:
    source = cache.source
    count = _candidate_count(cache, index)
    name = _bin_name(count)
    base = _base_threshold(source)
    if policy.kind == "A":
        offset = float(policy.params["low_offset"] if count <= 100 else policy.params["crowded_offset"])
        return min(1.0, max(0.0, base + offset))
    if policy.kind == "A_source":
        base_offset = float(policy.params["base_low_offset"] if count <= 100 else policy.params["base_crowded_offset"])
        offset = float(policy.params["source_offsets"][source])
        return min(1.0, max(0.0, base + base_offset + offset))
    if policy.kind == "B_crowded_tail":
        targets, scores = cache.group(index)
        order = _ordered(targets, scores)
        if count >= 101 and local != int(order[0]):
            return min(1.0, base + float(policy.params["tail_raise"]))
        return base
    if policy.kind == "B_top_gap":
        targets, scores = cache.group(index)
        order = _ordered(targets, scores)
        if local == int(order[0]) and len(order) > 1:
            lead = float(scores[order[0]] - scores[order[1]])
            if lead >= float(policy.params["minimum_gap"]):
                return max(0.0, base - float(policy.params["relax"]))
        return base
    raise ValueError(f"not a threshold policy: {policy.kind}")


def predictions_for_group_policy(s2: ScoreCache, s3: ScoreCache, policy: GroupPolicy) -> dict[str, set[str]]:
    if s2.s1_ids != s3.s1_ids:
        raise ValueError("S2/S3 cache S1 ordering differs")
    selected: dict[str, dict[str, float]] = {s1: {} for s1 in s2.s1_ids}
    for cache in (s2, s3):
        for index, s1 in enumerate(cache.s1_ids):
            targets, scores = cache.group(index)
            if policy.kind == "B_top_gap":
                threshold = _base_threshold(cache.source)
                if len(scores):
                    order = _ordered(targets, scores)
                    lead_ok = len(order) > 1 and float(scores[order[0]] - scores[order[1]]) >= float(policy.params["minimum_gap"])
                    if lead_ok:
                        threshold = max(0.0, threshold - float(policy.params["relax"]))
                mask = scores >= threshold
            elif policy.kind == "B_crowded_tail":
                if len(scores) >= 101:
                    order = _ordered(targets, scores)
                    thresholds = np.full(len(scores), min(1.0, _base_threshold(cache.source) + float(policy.params["tail_raise"])), dtype=np.float32)
                    thresholds[order[0]] = _base_threshold(cache.source)
                    mask = scores >= thresholds
                else:
                    mask = scores >= _base_threshold(cache.source)
            elif policy.kind in ("A", "A_source"):
                count = len(scores)
                base_offset = float(policy.params["low_offset"] if policy.kind == "A" and count <= 100 else policy.params.get("crowded_offset", 0.0) if policy.kind == "A" else policy.params["base_low_offset"] if count <= 100 else policy.params["base_crowded_offset"])
                local = float(policy.params.get("source_offsets", {}).get(cache.source, 0.0))
                threshold = min(1.0, max(0.0, _base_threshold(cache.source) + base_offset + local))
                mask = scores >= threshold
            else:
                raise ValueError(f"unsupported group policy {policy.kind}")
            if np.any(mask):
                selected[s1].update((target.decode("ascii"), float(score)) for target, score in zip(targets[mask], scores[mask]))
    return _apply_conflicts(selected, BASE_POLICY)


def _slice_metrics(truth: Mapping[str, set[str]], predictions: Mapping[str, set[str]],
                   s2: ScoreCache, s3: ScoreCache) -> list[dict[str, object]]:
    output = []
    for source, cache in (("S2", s2), ("S3", s3)):
        for _, _, label in COUNT_BINS:
            selected_ids = [s1 for i, s1 in enumerate(cache.s1_ids) if _bin_name(_candidate_count(cache, i)) == label]
            if not selected_ids:
                continue
            prefix = source + "-"
            subset_truth = {s1: {x for x in truth[s1] if x.startswith(prefix)} for s1 in selected_ids}
            subset_pred = {s1: {x for x in predictions[s1] if x.startswith(prefix)} for s1 in selected_ids}
            report = score_predictions(subset_truth, subset_pred, selected_ids)
            output.append({"source": source, "candidate_count_bin": label, "s1_count": len(selected_ids),
                           "macro_f0_5": report.macro_f0_5, "tp": report.tp, "fp": report.fp, "fn": report.fn,
                           "precision": report.micro_precision_diagnostic, "recall": report.micro_recall_diagnostic})
    return output


def _prediction_diagnostics(truth: Mapping[str, set[str]], predictions: Mapping[str, set[str]], ids: tuple[str, ...]) -> dict:
    sizes = [len(predictions[s1]) for s1 in ids]
    source_metrics = {}
    for source in ("S2", "S3"):
        prefix = source + "-"
        tp = fp = fn = 0
        for s1 in ids:
            t = {x for x in truth[s1] if x.startswith(prefix)}
            p = {x for x in predictions[s1] if x.startswith(prefix)}
            tp += len(t & p); fp += len(p - t); fn += len(t - p)
        source_metrics[source] = {"tp": tp, "fp": fp, "fn": fn,
                                  "precision": tp / (tp + fp) if tp + fp else 0.0,
                                  "recall": tp / (tp + fn) if tp + fn else 0.0}
    return {"source_metrics": source_metrics,
            "predicted_singletons": sum(size == 0 for size in sizes),
            "singleton_false_merges": sum(not truth[s1] and bool(predictions[s1]) for s1 in ids),
            "predicted_mean": float(np.mean(sizes)) if sizes else 0.0,
            "predicted_median": float(np.percentile(sizes, 50)) if sizes else 0.0,
            "predicted_p95": float(np.percentile(sizes, 95)) if sizes else 0.0,
            "predicted_p99": float(np.percentile(sizes, 99)) if sizes else 0.0,
            "predicted_max": max(sizes, default=0)}


def _evaluate(s2: ScoreCache, s3: ScoreCache, truth: Mapping[str, set[str]], policy: GroupPolicy) -> tuple[dict, list[dict]]:
    predictions = predictions_for_group_policy(s2, s3, policy)
    report = score_predictions(truth, predictions, s2.s1_ids)
    rows = _slice_metrics(truth, predictions, s2, s3)
    result = {**asdict(report), **_prediction_diagnostics(truth, predictions, s2.s1_ids),
              "variant": "", "policy": {"kind": policy.kind, "params": dict(policy.params)},
              "candidate_count_slices": rows, "predictions": predictions}
    return result, rows


def _result_metrics(result: Mapping[str, object], variant: str, stage: str) -> dict[str, object]:
    fields = ("macro_f0_5", "entities_evaluated", "true_links", "predicted_links", "tp", "fp", "fn",
              "micro_precision_diagnostic", "micro_recall_diagnostic", "true_singletons",
              "correctly_predicted_singletons", "predicted_singletons", "singleton_false_merges",
              "singleton_accuracy", "perfect_match_sets", "partial_match_sets", "completely_missed_nonempty_truth",
              "predicted_mean", "predicted_median", "predicted_p95", "predicted_p99", "predicted_max", "source_metrics")
    row = {key: result[key] for key in fields}
    row.update({"variant": variant, "stage": stage, "delta_vs_baseline": float(result["macro_f0_5"]) - BASELINE_MACRO,
                "policy": result["policy"], "slices": result["candidate_count_slices"]})
    return row


def _save_evaluation(output_dir: Path, variant: str, stage: str, result: dict, slices: list[dict]) -> dict:
    result["variant"] = variant
    row = _result_metrics(result, variant, stage)
    root = output_dir / variant
    _atomic_json(root / "metrics.json", row)
    slice_path = output_dir / f"{stage}_slice_results.tsv"
    slice_header = ("variant", "source", "candidate_count_bin", "s1_count", "macro_f0_5", "tp", "fp", "fn", "precision", "recall")
    existing_slices = []
    if slice_path.exists():
        with slice_path.open(encoding="utf-8", newline="") as handle:
            existing_slices = [r for r in csv.DictReader(handle, delimiter="\t") if r["variant"] != variant]
    with slice_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=slice_header, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(existing_slices)
        for item in slices:
            writer.writerow({"variant": variant, **item})
    path = output_dir / f"{stage}_results.tsv"
    header = ("variant", "stage", "macro_f0_5", "delta_vs_baseline", "tp", "fp", "fn", "predicted_links", "micro_precision_diagnostic", "micro_recall_diagnostic", "true_singletons", "predicted_singletons", "correctly_predicted_singletons", "singleton_false_merges", "singleton_accuracy", "predicted_mean", "predicted_median", "predicted_p95", "predicted_p99", "predicted_max", "source_metrics", "policy", "slices")
    existing = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as handle:
            existing = [r for r in csv.DictReader(handle, delimiter="\t") if r["variant"] != variant]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(existing)
        writer.writerow({k: json.dumps(row[k], sort_keys=True) if k in ("policy", "slices") else row.get(k, "") for k in header})
    result.pop("predictions", None)
    return row


def experiment_a(output_dir: Path = OUT) -> dict:
    before = frozen_snapshot(); s2, s3, truth = _tune_inputs()
    results = []
    for n, policy in enumerate(shared_offset_grid(), 1):
        result, slices = _evaluate(s2, s3, truth, policy)
        results.append((policy, result))
        _save_evaluation(output_dir, f"A_shared_{n:02d}", "tune", result, slices)
    best_policy, best_result = max(results, key=lambda pair: _rank_result(pair[1]))
    refinements = []
    if float(best_result["macro_f0_5"]) >= BASELINE_MACRO + .0005:
        for n, policy in enumerate(_a_refinements(best_policy), 1):
            result, slices = _evaluate(s2, s3, truth, policy)
            _save_evaluation(output_dir, f"A_source_{n:02d}", "tune", result, slices)
            refinements.append(result)
    _assert_snapshot(before)
    return {"shared_policies": len(results), "source_refinements": len(refinements), "best_shared": best_result["macro_f0_5"], "frozen_artifacts_unchanged": True}


def experiment_b(output_dir: Path = OUT) -> dict:
    before = frozen_snapshot(); s2, s3, truth = _tune_inputs(); results = []
    for n, policy in enumerate(gap_grid(), 1):
        result, slices = _evaluate(s2, s3, truth, policy)
        _save_evaluation(output_dir, f"B_{n:02d}", "tune", result, slices); results.append(result)
    _assert_snapshot(before)
    return {"policies": len(results), "best_macro_f0_5": max(float(x["macro_f0_5"]) for x in results), "frozen_artifacts_unchanged": True}


def _stable_fold(s1_id: str, folds: int = 5) -> int:
    return int.from_bytes(hashlib.sha256(s1_id.encode("utf-8")).digest()[:8], "big") % folds


def _label_array(cache: ScoreCache, truth: Mapping[str, set[str]]) -> np.ndarray:
    labels = np.zeros(len(cache.targets), dtype=np.uint8)
    for i, s1 in enumerate(cache.s1_ids):
        start, end = int(cache.offsets[i]), int(cache.offsets[i + 1])
        if end > start:
            positives = truth[s1]
            labels[start:end] = np.fromiter((target.decode("ascii") in positives for target in cache.targets[start:end]), dtype=np.uint8, count=end-start)
    return labels


def _fit_isotonic(scores: np.ndarray, labels: np.ndarray):
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError as exc:
        raise RuntimeError("Experiment C requires scikit-learn's IsotonicRegression") from exc
    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    model.fit(scores.astype(np.float64), labels.astype(np.float64))
    return model


def _degree_key(c2: int, c3: int) -> tuple[int, int]:
    return (COUNT_BINS.index(next(x for x in COUNT_BINS if x[0] <= c2 <= x[1])) if c2 else -1,
            COUNT_BINS.index(next(x for x in COUNT_BINS if x[0] <= c3 <= x[1])) if c3 else -1)


def _crossfit_policy(s2: ScoreCache, s3: ScoreCache, truth: Mapping[str, set[str]], output_dir: Path) -> tuple[dict[str, set[str]], dict]:
    from sklearn.isotonic import IsotonicRegression
    labels = {"S2": _label_array(s2, truth), "S3": _label_array(s3, truth)}
    caches = {"S2": s2, "S3": s3}
    folds = np.asarray([_stable_fold(s1) for s1 in s2.s1_ids], dtype=np.int8)
    predictions: dict[str, set[str]] = {s1: set() for s1 in s2.s1_ids}
    oof_proxy = []
    for fold in range(5):
        train_ids = np.flatnonzero(folds != fold); held_ids = np.flatnonzero(folds == fold)
        if not len(held_ids):
            continue
        models = {}
        for source, cache in caches.items():
            train_mask = np.zeros(len(cache.targets), dtype=bool)
            for idx in train_ids:
                train_mask[int(cache.offsets[idx]):int(cache.offsets[idx + 1])] = True
            model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            model.fit(cache.scores[train_mask].astype(np.float64), labels[source][train_mask].astype(np.float64))
            models[source] = model
        means: dict[tuple[int, int], list[int]] = defaultdict(list)
        for idx in train_ids:
            s1 = s2.s1_ids[idx]
            means[_degree_key(_candidate_count(s2, idx), _candidate_count(s3, idx))].append(len(truth[s1]))
        fallback = float(np.mean([len(truth[s2.s1_ids[idx]]) for idx in train_ids]))
        expected = {key: float(np.mean(values)) for key, values in means.items()}
        for idx in held_ids:
            s1 = s2.s1_ids[int(idx)]; groups = {}
            for source, cache in caches.items():
                targets, raw = cache.group(int(idx))
                order = _ordered(targets, raw)[:10]
                p = models[source].predict(raw[order].astype(np.float64)) if len(order) else np.empty(0)
                groups[source] = (targets[order], p)
            m_hat = expected.get(_degree_key(len(s2.group(int(idx))[0]), len(s3.group(int(idx))[0])), fallback)
            best = (-1.0, 0, 0)
            for k2 in range(len(groups["S2"][0]) + 1):
                t2 = float(groups["S2"][1][:k2].sum())
                for k3 in range(len(groups["S3"][0]) + 1):
                    count = k2 + k3; tp_hat = t2 + float(groups["S3"][1][:k3].sum())
                    if count == 0 and m_hat <= 0:
                        utility = 1.0
                    else:
                        denominator = tp_hat + .25 * m_hat + .75 * count
                        utility = 1.25 * tp_hat / denominator if denominator else 0.0
                    candidate = (utility, -count, -k2)
                    if candidate > best:
                        best = candidate; best_k2, best_k3 = k2, k3
            targets2 = groups["S2"][0][:best_k2]; targets3 = groups["S3"][0][:best_k3]
            predictions[s1].update(x.decode("ascii") for x in targets2)
            predictions[s1].update(x.decode("ascii") for x in targets3)
            oof_proxy.append(best[0])
    final_degree: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, s1 in enumerate(s2.s1_ids):
        final_degree[_degree_key(_candidate_count(s2, idx), _candidate_count(s3, idx))].append(len(truth[s1]))
    degree_payload = {f"{k[0]},{k[1]}": float(np.mean(v)) for k, v in final_degree.items()}
    fallback_all = float(np.mean([len(truth[s1]) for s1 in s2.s1_ids]))
    result = {"method": "5-fold S1-hash cross-fit; per-source isotonic calibration; top-10 source prefixes; maximize plug-in expected F0.5 = 1.25 ETP / (ETP + 0.25 E[truth links] + 0.75 predictions); tie-break fewer predictions then fewer S2 predictions", "folds": 5, "oof_mean_proxy": float(np.mean(oof_proxy)) if oof_proxy else 0.0,
              "expected_truth_links_by_source_count_bins": degree_payload, "fallback_expected_truth_links": fallback_all}
    _atomic_json(output_dir / "C_crossfit_selection.json", result)
    return _apply_conflicts({s1: {target: 1.0 for target in values} for s1, values in predictions.items()}, BASE_POLICY), result


def experiment_c(output_dir: Path = OUT) -> dict:
    before = frozen_snapshot(); s2, s3, truth = _tune_inputs()
    predictions, methodology = _crossfit_policy(s2, s3, truth, output_dir)
    report = score_predictions(truth, predictions, s2.s1_ids)
    result = {**asdict(report), **_prediction_diagnostics(truth, predictions, s2.s1_ids),
              "variant": "C_crossfit", "policy": {"kind": "C_crossfit", "params": methodology},
              "candidate_count_slices": _slice_metrics(truth, predictions, s2, s3)}
    _save_evaluation(output_dir, "C_crossfit", "tune", result, result["candidate_count_slices"])
    # Store a final tune-fitted calibrator for later locked validation/test use.
    models = {source: _fit_isotonic(cache.scores, _label_array(cache, truth)) for source, cache in (("S2", s2), ("S3", s3))}
    _atomic_json(output_dir / "C_crossfit_calibrators.json", {"models": {s: {"x": model.X_thresholds_.tolist(), "y": model.y_thresholds_.tolist()} for s, model in models.items()},
                                                                "expected_truth_links_by_source_count_bins": methodology["expected_truth_links_by_source_count_bins"],
                                                                "fallback_expected_truth_links": methodology["fallback_expected_truth_links"]})
    _assert_snapshot(before)
    return {"variant": "C_crossfit", "macro_f0_5": report.macro_f0_5, "delta_vs_baseline": report.macro_f0_5 - BASELINE_MACRO, "method": methodology, "frozen_artifacts_unchanged": True}


def _rank_result(result: Mapping[str, object]) -> tuple[float, float, int, int]:
    tp, fp = int(result["tp"]), int(result["fp"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    return (float(result["macro_f0_5"]), precision, -int(result.get("singleton_false_merges", 0)), -int(result["predicted_links"]))


def compare_tune(output_dir: Path = OUT) -> dict:
    before = frozen_snapshot()
    path = output_dir / "tune_results.tsv"
    if not path.exists():
        raise FileNotFoundError("run experiment-a/b/c before compare-tune")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        row["macro_f0_5"] = float(row["macro_f0_5"]); row["delta_vs_baseline"] = float(row["delta_vs_baseline"])
    rows.sort(key=lambda x: (-x["macro_f0_5"], -float(x["micro_precision_diagnostic"]),
                             int(x["singleton_false_merges"]), int(x["predicted_links"]), x["variant"]))
    result = {"baseline_macro_f0_5": BASELINE_MACRO, "ranked_variants": rows,
              "selection": "tune-only; macro F0.5, then precision, fewer singleton false merges, fewer links",
              "validation_used": False}
    _atomic_json(output_dir / "tune_comparison.json", result)
    _assert_snapshot(before)
    return {"variants": len(rows), "best": rows[0] if rows else None, "path": str(output_dir / "tune_comparison.json")}


def _read_tune_rows(output_dir: Path) -> list[dict[str, str]]:
    path = output_dir / "tune_results.tsv"
    if not path.exists():
        raise FileNotFoundError("no tune results exist")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def lock_winner(output_dir: Path = OUT) -> dict:
    before = frozen_snapshot()
    if (output_dir / "locked_winner.json").exists() or (output_dir / "no_winner.json").exists():
        raise RuntimeError("winner decision is already locked")
    rows = _read_tune_rows(output_dir)
    def numeric(row, field): return float(row[field])
    rows.sort(key=lambda r: (-numeric(r, "macro_f0_5"), -numeric(r, "micro_precision_diagnostic"),
                             int(r["singleton_false_merges"]), int(r["predicted_links"]), r["variant"]))
    candidate = rows[0]
    delta = numeric(candidate, "macro_f0_5") - BASELINE_MACRO
    eligible = delta >= .0015
    rationale = "macro gain meets +0.0015 tune gate"
    if not eligible and delta >= .0010:
        slices = json.loads(candidate["slices"])
        rows_by_variant = {row["variant"]: row for row in rows}
        baseline = rows_by_variant.get("A_shared_16")
        baseline_fps = int(baseline["fp"]) if baseline else 23_544
        baseline_singleton_fp = int(baseline["singleton_false_merges"]) if baseline else 751
        fp_increase = int(candidate["fp"]) - baseline_fps
        base_slices = {(x["source"], x["candidate_count_bin"]): x for x in json.loads(baseline["slices"])} if baseline else {}
        comparable = [x for x in slices if (x["source"], x["candidate_count_bin"]) in base_slices and int(x["s1_count"]) >= 1_000]
        broad_gain = sum(float(x["macro_f0_5"]) > float(base_slices[(x["source"], x["candidate_count_bin"])]["macro_f0_5"]) for x in comparable) >= min(4, len(comparable)) and len(comparable) >= 4
        singleton_ok = int(candidate["singleton_false_merges"]) <= baseline_singleton_fp
        controlled_fp = fp_increase <= max(100, baseline_fps * .02)
        eligible = broad_gain and singleton_ok and controlled_fp
        rationale = "secondary +0.001 gate: broad slice gains, controlled FP growth, no singleton false-merge regression" if eligible else "secondary gate failed"
    if not eligible:
        payload = {"winner": None, "best_tune_candidate": candidate, "delta_vs_baseline": delta,
                   "reason": rationale, "validation_permitted": False, "frozen_snapshot": before}
        _atomic_json(output_dir / "no_winner.json", payload); _assert_snapshot(before); return payload
    policy = json.loads(candidate["policy"])
    payload = {"experiment_id": candidate["variant"], "policy": policy,
               "tune_metrics": {k: candidate[k] for k in ("macro_f0_5", "delta_vs_baseline", "tp", "fp", "fn", "predicted_links", "singleton_false_merges", "slices")},
               "selection_rationale": rationale, "frozen_policy": asdict(BASE_POLICY),
               "frozen_snapshot": before, "locked_at_ns": time.time_ns(), "validation_confirmation_used": False}
    _atomic_json(output_dir / "locked_winner.json", payload)
    _assert_snapshot(before)
    return payload


def _predict_c(s2: ScoreCache, s3: ScoreCache, calibrator_path: Path) -> dict[str, set[str]]:
    saved = json.loads(calibrator_path.read_text(encoding="utf-8"))
    calibrators = saved["models"]
    degree = saved["expected_truth_links_by_source_count_bins"]
    selected = {s1: {} for s1 in s2.s1_ids}
    for i, s1 in enumerate(s2.s1_ids):
        groups = {}
        for source, cache in (("S2", s2), ("S3", s3)):
            targets, scores = cache.group(i)
            order = _ordered(targets, scores)[:10]
            entry = calibrators[source]
            probs = np.interp(scores[order], entry["x"], entry["y"])
            groups[source] = (targets[order], scores[order], probs)
        bin_key = _degree_key(len(s2.group(i)[0]), len(s3.group(i)[0]))
        m_hat = float(degree.get(f"{bin_key[0]},{bin_key[1]}", saved["fallback_expected_truth_links"]))
        best = None
        best_k2 = best_k3 = 0
        for k2 in range(len(groups["S2"][0]) + 1):
            tp2 = float(groups["S2"][2][:k2].sum())
            for k3 in range(len(groups["S3"][0]) + 1):
                count = k2 + k3
                tp_hat = tp2 + float(groups["S3"][2][:k3].sum())
                denom = tp_hat + .25 * m_hat + .75 * count
                utility = 1.0 if count == 0 and m_hat <= 0 else (1.25 * tp_hat / denom if denom else 0.0)
                candidate = (utility, -count, -k2)
                if best is None or candidate > best:
                    best = candidate
                    best_k2, best_k3 = k2, k3
        selected[s1].update((x.decode("ascii"), float(score)) for x, score in zip(groups["S2"][0][:best_k2], groups["S2"][1][:best_k2]))
        selected[s1].update((x.decode("ascii"), float(score)) for x, score in zip(groups["S3"][0][:best_k3], groups["S3"][1][:best_k3]))
    return _apply_conflicts(selected, BASE_POLICY)


def confirm_validation(output_dir: Path = OUT) -> dict:
    before = frozen_snapshot()
    locked_path = output_dir / "locked_winner.json"
    if not locked_path.exists():
        raise RuntimeError("validation is allowed only after tune winner lock")
    if (output_dir / "validation_confirmation.json").exists():
        raise RuntimeError("one-shot validation confirmation has already been run")
    locked = json.loads(locked_path.read_text(encoding="utf-8"))
    s2, s3, truth = _val_inputs()
    policy = locked["policy"]
    if policy["kind"] == "C_crossfit":
        predictions = _predict_c(s2, s3, output_dir / "C_crossfit_calibrators.json")
        report = score_predictions(truth, predictions, s2.s1_ids)
        result = {**asdict(report), **_prediction_diagnostics(truth, predictions, s2.s1_ids),
                  "candidate_count_slices": _slice_metrics(truth, predictions, s2, s3)}
    else:
        group_policy = GroupPolicy(policy["kind"], policy["params"])
        result, _ = _evaluate(s2, s3, truth, group_policy)
    delta = float(result["macro_f0_5"]) - .8700057896268494
    state = "CONFIRMED" if delta >= -.001 and float(locked["tune_metrics"]["delta_vs_baseline"]) > 0 else "MIXED" if delta >= -.003 else "FAILED_TO_GENERALIZE"
    value = {"status": state, "validation_macro_f0_5": result["macro_f0_5"], "baseline_validation_macro_f0_5": .8700057896268494,
             "delta_vs_baseline": delta, "tp": result["tp"], "fp": result["fp"], "fn": result["fn"],
             "micro_precision_diagnostic": result["micro_precision_diagnostic"],
             "micro_recall_diagnostic": result["micro_recall_diagnostic"],
             "source_metrics": result["source_metrics"], "candidate_count_slices": result["candidate_count_slices"],
             "true_singletons": result["true_singletons"],
             "correctly_predicted_singletons": result["correctly_predicted_singletons"],
             "predicted_singletons": result["predicted_singletons"], "singleton_false_merges": result["singleton_false_merges"],
             "prediction_cardinality": {key: result[key] for key in ("predicted_mean", "predicted_median", "predicted_p95", "predicted_p99", "predicted_max")},
             "s2_rows": len(s2.targets), "s3_rows": len(s3.targets), "policy": policy,
             "validation_confirmation_used": True, "frozen_snapshot": before}
    _atomic_json(output_dir / "validation_confirmation.json", value)
    _assert_snapshot(before)
    return value


def _iter_test_score_groups(manifest: Mapping[str, object], source: str):
    """Yield (S1, candidates) in Phase 12 shard order without candidate regeneration."""
    expected_header = ["source1_entity_id", "candidate_entity_id", "score", "from_v1", "from_address", "address_rank", "address_score"]
    for shard in sorted(manifest["shards"], key=lambda x: int(x["shard"])):
        entry = shard["outputs"][source]
        path = Path(entry["path"])
        if not path.is_file() or path.stat().st_size != int(entry["bytes"]):
            raise RuntimeError(f"Phase 12 {source} score shard is missing/size-mismatched: {path}")
        if entry.get("sha256") and _sha256(path) != entry["sha256"]:
            raise RuntimeError(f"Phase 12 {source} score shard hash mismatch: {path}")
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames != expected_header:
                raise ValueError(f"unexpected Phase 12 score header in {path}")
            for s1, rows in groupby(reader, key=lambda row: row["source1_entity_id"]):
                group = []
                for row in rows:
                    target = row["candidate_entity_id"]
                    score = float(row["score"])
                    if not target.startswith(source + "-") or not 0.0 <= score <= 1.0 or not math.isfinite(score):
                        raise ValueError(f"invalid {source} score row for {s1}")
                    group.append((target, score))
                yield s1, group


def _select_group_pairs(s2_group: list[tuple[str, float]], s3_group: list[tuple[str, float]], policy: GroupPolicy,
                        calibrators: Mapping[str, object] | None = None) -> list[tuple[str, float]]:
    groups = {"S2": s2_group, "S3": s3_group}
    selected = []
    if policy.kind == "C_crossfit":
        if calibrators is None:
            raise ValueError("C policy requires its saved tune calibration")
        saved_models = calibrators["models"]
        key_bins = _degree_key(len(s2_group), len(s3_group))
        m_hat = float(calibrators["expected_truth_links_by_source_count_bins"].get(
            f"{key_bins[0]},{key_bins[1]}", calibrators["fallback_expected_truth_links"]))
        ranked = {}
        for source, pairs in groups.items():
            pairs = sorted(pairs, key=lambda x: (-x[1], x[0]))[:10]
            model = saved_models[source]
            probs = np.interp(np.asarray([x[1] for x in pairs]), model["x"], model["y"]) if pairs else np.empty(0)
            ranked[source] = (pairs, probs)
        best = None; best_k2 = best_k3 = 0
        for k2 in range(len(ranked["S2"][0]) + 1):
            tp2 = float(ranked["S2"][1][:k2].sum())
            for k3 in range(len(ranked["S3"][0]) + 1):
                count = k2 + k3; tp_hat = tp2 + float(ranked["S3"][1][:k3].sum())
                denom = tp_hat + .25 * m_hat + .75 * count
                utility = 1.0 if count == 0 and m_hat <= 0 else (1.25 * tp_hat / denom if denom else 0.0)
                choice = (utility, -count, -k2)
                if best is None or choice > best:
                    best = choice; best_k2, best_k3 = k2, k3
        return ranked["S2"][0][:best_k2] + ranked["S3"][0][:best_k3]
    for source, pairs in groups.items():
        count = len(pairs); base = _base_threshold(source)
        if policy.kind in ("A", "A_source"):
            if policy.kind == "A":
                offset = float(policy.params["low_offset"] if count <= 100 else policy.params["crowded_offset"])
            else:
                offset = float(policy.params["base_low_offset"] if count <= 100 else policy.params["base_crowded_offset"])
                offset += float(policy.params["source_offsets"][source])
            threshold = min(1.0, max(0.0, base + offset))
            selected.extend((target, score) for target, score in pairs if score >= threshold)
        elif policy.kind == "B_top_gap":
            ranked = sorted(pairs, key=lambda x: (-x[1], x[0]))
            threshold = base
            if len(ranked) > 1 and ranked[0][1] - ranked[1][1] >= float(policy.params["minimum_gap"]):
                threshold = max(0.0, base - float(policy.params["relax"]))
            selected.extend((target, score) for target, score in pairs if score >= threshold)
        elif policy.kind == "B_crowded_tail":
            ranked = sorted(pairs, key=lambda x: (-x[1], x[0]))
            top = ranked[0][0] if ranked else None
            raised = min(1.0, base + float(policy.params["tail_raise"])) if count >= 101 else base
            selected.extend((target, score) for target, score in pairs if score >= (base if target == top else raised))
        else:
            raise ValueError(f"unsupported policy kind {policy.kind}")
    return selected


def apply_test(output_dir: Path = OUT) -> dict:
    """Apply a tune-locked, validation-confirmed rule to existing Phase 12 scores only."""
    started = time.perf_counter()
    before = frozen_snapshot()
    lock_path = output_dir / "locked_winner.json"
    confirmation_path = output_dir / "validation_confirmation.json"
    if not lock_path.is_file() or not confirmation_path.is_file():
        raise RuntimeError("test policy requires a locked tune winner and completed validation confirmation")
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    locked = json.loads(lock_path.read_text(encoding="utf-8"))
    expected_policy = {
        "kind": "A_source",
        "params": {"base_crowded_offset": 0.015, "base_low_offset": -0.015,
                   "source_offsets": {"S2": -0.005, "S3": 0.0}},
    }
    if confirmation.get("status") != "CONFIRMED" or float(confirmation.get("validation_macro_f0_5", -1)) != CONFIRMED_VALIDATION_MACRO:
        raise RuntimeError("test policy requires the exact one-shot CONFIRMED validation result")
    if confirmation.get("policy") != expected_policy:
        raise RuntimeError("validation confirmation does not match A_source_02")
    if locked.get("experiment_id") != "A_source_02" or locked.get("policy") != expected_policy:
        raise RuntimeError("locked winner does not match A_source_02")
    if not locked.get("frozen_snapshot") or before != locked["frozen_snapshot"]:
        raise RuntimeError("current frozen inputs do not match the locked winner snapshot")
    submission1 = BASE.parents[1] / "output/matching_results.tsv"
    if _sha256(submission1) != BASELINE_SUBMISSION_SHA256:
        raise RuntimeError("Submission #1 SHA256 differs from the required frozen hash")
    output_path = BASE.parents[1] / "output/phase112/matching_results.tsv"
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite existing Phase 11.2 output: {output_path}")
    if not (BASE.parents[1] / "output/candidate_pairs.tsv").is_file():
        raise RuntimeError("frozen candidate_pairs.tsv is missing")
    policy_raw = locked["policy"]
    policy = GroupPolicy(policy_raw["kind"], policy_raw["params"])
    manifest_path = PHASE12 / "scores/score_manifest.json"
    candidate_summary_path = PHASE12 / "candidates/candidate_summary.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates = json.loads(candidate_summary_path.read_text(encoding="utf-8"))
    if int(manifest["total"]) != int(candidates["pairs"]) or manifest.get("candidate_membership_sha256") != candidates.get("membership_sha256"):
        raise RuntimeError("Phase 12 score manifest does not certify the frozen canonical candidate set")
    if int(manifest["total"]) != 309_551_055 or int(candidates["s1_count"]) != 1_732_544:
        raise RuntimeError("Phase 12 candidate counts do not match the frozen run")
    canonical = PHASE12 / "candidates/candidate_pairs_long.tsv.gz"
    if canonical.stat().st_size != int(candidates["canonical"]["bytes"]) or _sha256(canonical) != candidates["canonical"]["sha256"]:
        raise RuntimeError("canonical candidate file does not match its recorded SHA256")
    candidate_export = BASE.parents[1] / "output/candidate_pairs.tsv"
    candidate_export_sha_before = _sha256(candidate_export)
    if _sha256(submission1) != BASELINE_SUBMISSION_SHA256:
        raise RuntimeError("Submission #1 changed during preflight")
    calibrators = json.loads((output_dir / "C_crossfit_calibrators.json").read_text()) if policy.kind == "C_crossfit" else None
    # Phase 12 score shards are immutable inputs; this stage writes under a new policy namespace only.
    dataset_test_s1 = DEFAULT_DATASET_ROOT / "test/test_source1.tsv"
    test_rows = []
    with dataset_test_s1.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "entity_id" not in reader.fieldnames:
            raise ValueError("unexpected test Source 1 header")
        test_rows = [row["entity_id"] for row in reader]
    if len(test_rows) != 1_732_544 or len(set(test_rows)) != len(test_rows):
        raise ValueError("test Source 1 coverage does not match the frozen inventory")
    id_position = {s1: i for i, s1 in enumerate(test_rows)}
    groups = {source: iter(_iter_test_score_groups(manifest, source)) for source in ("S2", "S3")}
    current = {source: next(groups[source], None) for source in groups}
    counters = {"S2": 0, "S3": 0}; accepted_by_source = {"S2": 0, "S3": 0}
    db_path = output_dir / "test_ownership.sqlite"
    if db_path.exists():
        raise RuntimeError(f"refusing to overwrite existing ownership database: {db_path}")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(db_path)
    try:
        db.execute("CREATE TABLE accepted (target TEXT, s1 TEXT, score REAL)")
        for i, s1 in enumerate(test_rows):
            picked = {}
            for source in ("S2", "S3"):
                item = current[source]
                group = []
                if item is not None:
                    pos = id_position.get(item[0])
                    if pos is None or pos < i:
                        raise ValueError(f"score S1 order/coverage mismatch at {item[0]}")
                    if pos == i:
                        group = item[1]
                        counters[source] += len(group)
                        current[source] = next(groups[source], None)
                picked[source] = group
            chosen = _select_group_pairs(picked["S2"], picked["S3"], policy, calibrators)
            if chosen:
                for target, _score in chosen:
                    accepted_by_source["S2" if target.startswith("S2-") else "S3"] += 1
                db.executemany("INSERT INTO accepted VALUES (?,?,?)", ((target, s1, score) for target, score in chosen))
            if (i + 1) % 100_000 == 0:
                db.commit()
                print(f"Policy predictions: {i + 1:,}/{len(test_rows):,} S1; scored pairs read={sum(counters.values()):,}", flush=True)
        if any(current.values()):
            raise ValueError("score shards contain S1 rows absent from test Source 1")
        db.execute("CREATE INDEX accepted_target ON accepted(target, score DESC, s1)")
        db.execute("CREATE TABLE winners AS SELECT target,s1,score FROM (SELECT target,s1,score,ROW_NUMBER() OVER (PARTITION BY target ORDER BY score DESC,s1) AS rn FROM accepted) WHERE rn=1")
        db.execute("CREATE INDEX winners_s1 ON winners(s1,target)"); db.commit()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(".tsv.tmp")
        sizes = []; links = 0
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["source1_entity_id", "matched_entity_ids"])
            for s1 in test_rows:
                matches = [row[0] for row in db.execute("SELECT target FROM winners WHERE s1=? ORDER BY target", (s1,))]
                writer.writerow((s1, ",".join(matches))); sizes.append(len(matches)); links += len(matches)
        temporary.replace(output_path)
        conflicts = int(db.execute("SELECT count(*) FROM (SELECT target FROM accepted GROUP BY target HAVING count(*)>1)").fetchone()[0])
        final_links_by_source = {
            source: int(db.execute("SELECT count(*) FROM winners WHERE target LIKE ?", (source + "-%",)).fetchone()[0])
            for source in ("S2", "S3")
        }
    finally:
        db.close(); db_path.unlink(missing_ok=True)
    expected_rows = sum(int(item["total"]) for item in manifest["shards"])
    if sum(counters.values()) != expected_rows or expected_rows != int(manifest["total"]):
        raise RuntimeError("test policy did not consume each frozen scored candidate exactly once")
    if len(sizes) != 1_732_544:
        raise RuntimeError(f"prediction row count mismatch: {len(sizes)}")
    if _sha256(submission1) != BASELINE_SUBMISSION_SHA256:
        raise RuntimeError("Submission #1 changed during policy application")
    candidate_export_sha_after = _sha256(candidate_export)
    if candidate_export_sha_after != candidate_export_sha_before:
        raise RuntimeError("candidate_pairs.tsv changed during policy application")
    _assert_snapshot(before)
    comparison = _compare_prediction_files(submission1, output_path)
    value = {"experiment_id": locked["experiment_id"], "policy": policy_raw, "score_rows_consumed": counters,
             "accepted_links_before_ownership_by_source": accepted_by_source,
             "final_links_by_source": final_links_by_source,
             "candidate_pairs": expected_rows, "candidate_membership_sha256": candidates["membership_sha256"],
             "predicted_links": links, "target_conflicts_resolved": conflicts,
             "prediction_cardinality": {"mean": float(np.mean(sizes)), "median": float(np.percentile(sizes, 50)),
                                        "p95": float(np.percentile(sizes, 95)), "p99": float(np.percentile(sizes, 99)),
                                        "max": max(sizes, default=0), "singletons": sum(x == 0 for x in sizes)},
             "output": {"path": str(output_path), "bytes": output_path.stat().st_size, "sha256": _sha256(output_path),
                        "s1_rows": len(sizes), "unique_s1_rows": len(test_rows)},
             "comparison_to_submission_1": comparison,
             "submission_1_sha256": _sha256(submission1), "submission_1_unchanged": True,
             "candidate_pairs_sha256": candidate_export_sha_after, "candidate_pairs_unchanged": True,
             "score_manifest_sha256": _sha256(manifest_path),
             "candidate_summary_sha256": _sha256(candidate_summary_path),
             "frozen_artifacts_unchanged": True, "test_labels_accessed": False,
             "runtime_seconds": time.perf_counter() - started, "peak_rss_mb": _rss_mb()}
    _atomic_json(output_dir / "test_policy_summary.json", value)
    _assert_snapshot(before)
    return value


def _read_prediction_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["source1_entity_id", "matched_entity_ids"]:
            raise ValueError(f"unexpected prediction schema in {path}")
        seen = set()
        for row in reader:
            s1 = row["source1_entity_id"]
            if s1 in seen:
                raise ValueError(f"duplicate S1 prediction row in {path}: {s1}")
            seen.add(s1)
            matches = row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else []
            if len(matches) != len(set(matches)):
                raise ValueError(f"duplicate matched ID for {s1} in {path}")
            yield s1, set(matches)


def _compare_prediction_files(old_path: Path, new_path: Path, expected_rows: int = 1_732_544) -> dict:
    identical = changed = added = removed = singleton_to_multi = multi_to_singleton = rows = 0
    old_iter = _read_prediction_rows(old_path); new_iter = _read_prediction_rows(new_path)
    sentinel = object()
    while True:
        old = next(old_iter, sentinel); new = next(new_iter, sentinel)
        if old is sentinel and new is sentinel:
            break
        if old is sentinel or new is sentinel:
            raise ValueError("Submission #1 and #2 have different S1 coverage")
        if old[0] != new[0]:
            raise ValueError(f"Submission S1 ordering/coverage mismatch: {old[0]} != {new[0]}")
        rows += 1
        old_matches, new_matches = old[1], new[1]
        if old_matches == new_matches:
            identical += 1
        else:
            changed += 1
        added += len(new_matches - old_matches); removed += len(old_matches - new_matches)
        singleton_to_multi += not old_matches and bool(new_matches)
        multi_to_singleton += bool(old_matches) and not new_matches
    if rows != expected_rows:
        raise ValueError(f"expected {expected_rows:,} prediction rows in comparison, found {rows}")
    return {"s1_rows": rows, "identical_predictions": identical, "changed_predictions": changed,
            "links_added": added, "links_removed": removed,
            "singleton_to_non_singleton": singleton_to_multi,
            "non_singleton_to_singleton": multi_to_singleton}


def report(output_dir: Path = OUT) -> dict:
    before = frozen_snapshot()
    comparison = json.loads((output_dir / "tune_comparison.json").read_text()) if (output_dir / "tune_comparison.json").exists() else None
    lock_path = output_dir / "locked_winner.json"
    no_winner = output_dir / "no_winner.json"
    confirmation = output_dir / "validation_confirmation.json"
    value = {"phase": "11.2A", "baseline_macro_f0_5": BASELINE_MACRO, "tune_comparison": comparison,
             "locked_winner": json.loads(lock_path.read_text()) if lock_path.exists() else None,
             "no_winner": json.loads(no_winner.read_text()) if no_winner.exists() else None,
             "validation_confirmation": json.loads(confirmation.read_text()) if confirmation.exists() else None,
             "future_test_policy": "If validation confirmation is approved, apply the locked policy by streaming existing Phase 12 score shards; do not regenerate candidates or rescore.",
             "test_scores_reused": True, "candidate_pairs_unchanged": True}
    _atomic_json(output_dir / "phase112_summary.json", value)
    lines = ["# Phase 11.2A Group Policy Report", "", f"Baseline tune macro F0.5: {BASELINE_MACRO:.12f}", "",
             "Selection used tune only. Frozen thresholds and highest-score ownership apply to A/B. C uses a five-fold cross-fitted calibrated prefix selector. No Phase 12 candidates or test scores were used to choose a policy.", "",
             f"Winner: {(value['locked_winner'] or {}).get('experiment_id', 'none')}",
             f"Validation status: {(value['validation_confirmation'] or {}).get('status', 'not run')}",
             "A confirmed policy can reuse Phase 12 score shards; candidate retrieval is not rerun."]
    (output_dir / "phase112_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _assert_snapshot(before)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inventory", "baseline-check", "experiment-a", "experiment-b", "experiment-c", "compare-tune", "lock-winner", "confirm-validation", "apply-test", "report"))
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args(argv)
    fn = {"inventory": inventory, "baseline-check": baseline_check, "experiment-a": experiment_a,
          "experiment-b": experiment_b, "experiment-c": experiment_c, "compare-tune": compare_tune,
          "lock-winner": lock_winner, "confirm-validation": confirm_validation, "apply-test": apply_test, "report": report}[args.command]
    started = time.perf_counter(); value = fn(args.output_dir)
    if isinstance(value, dict):
        value.setdefault("runtime_seconds", time.perf_counter() - started); value.setdefault("peak_rss_mb", _rss_mb())
    print(json.dumps(value, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
