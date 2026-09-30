#!/usr/bin/env python3
"""Central validation diagnostics for entity-resolution experiments."""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from scoring import (
    LinkCounts,
    ScoreReport,
    count_link_errors,
    load_ground_truth,
    load_id_file,
    load_predictions,
    parse_match_list,
    precision_recall_diagnostics,
    score_predictions,
)

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[3] / "6ab10eb3b23ba_student_resource/student_resource/dataset"
DEFAULT_GROUND_TRUTH = DEFAULT_DATASET_ROOT / "train/train_ground_truth.tsv"
DEFAULT_S1_IDS = BASE_DIR / "artifacts/splits/val_s1_ids.txt"
DEFAULT_REPORT = BASE_DIR / "artifacts/diagnostics/summary.txt"
DEFAULT_EXPERIMENTS = BASE_DIR / "artifacts/experiments.csv"
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]
EXPERIMENT_COLUMNS = [
    "experiment_name", "timestamp", "notes", "macro_f0_5",
    "precision_diag", "recall_diag", "singleton_accuracy",
    "singleton_false_merges", "candidate_recall",
    "s2_candidate_recall", "s3_candidate_recall", "avg_candidates",
    "p95_candidates", "p99_candidates", "runtime_seconds",
    "peak_memory_mb",
]


@dataclass(frozen=True)
class LinkDiagnostics:
    true_links: int
    predicted_links: int
    tp: int
    fp: int
    fn: int
    precision_diagnostic: float
    recall_diagnostic: float


@dataclass(frozen=True)
class CandidateDiagnostics:
    candidate_recall: float
    s2_candidate_recall: float
    s3_candidate_recall: float
    blocking_misses: int
    model_misses: int
    mean_candidates: float
    median_candidates: float
    p90_candidates: float
    p95_candidates: float
    p99_candidates: float
    max_candidates: int


@dataclass(frozen=True)
class DiagnosticsResult:
    score: ScoreReport
    predicted_singletons: int
    singleton_false_merges: int
    source2: LinkDiagnostics
    source3: LinkDiagnostics
    false_positive_entities: int
    candidates: CandidateDiagnostics | None


def _sum_link_counts(
    truth: Mapping[str, set[str]],
    prediction: Mapping[str, set[str]],
    prefix: str | None = None,
) -> LinkCounts:
    """Sum set comparisons per S1 so repeated target IDs remain distinct pairs."""
    tp = fp = fn = 0
    for s1_id in truth:
        counts = count_link_errors(truth[s1_id], prediction[s1_id], prefix=prefix)
        tp += counts.tp
        fp += counts.fp
        fn += counts.fn
    return LinkCounts(tp=tp, fp=fp, fn=fn)


def _link_diagnostics(
    truth: Mapping[str, set[str]], prediction: Mapping[str, set[str]], prefix: str
) -> LinkDiagnostics:
    truth_links = sum(1 for links in truth.values() for value in links if value.startswith(prefix))
    predicted_links = sum(1 for links in prediction.values() for value in links if value.startswith(prefix))
    counts = _sum_link_counts(truth, prediction, prefix=prefix)
    precision, recall = precision_recall_diagnostics(counts)
    return LinkDiagnostics(truth_links, predicted_links, counts.tp, counts.fp, counts.fn, precision, recall)


def _percentile(sorted_values: list[int], percentile: float) -> float:
    """Linear percentile, matching the default NumPy percentile convention."""
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _candidate_diagnostics(
    truth: Mapping[str, set[str]],
    prediction: Mapping[str, set[str]],
    candidates: Mapping[str, set[str]],
) -> CandidateDiagnostics:
    if truth.keys() != candidates.keys():
        raise ValueError(
            f"Candidate S1 coverage mismatch: missing={len(truth.keys()-candidates.keys())}, "
            f"extra={len(candidates.keys()-truth.keys())}"
        )
    candidate_link_counts = _sum_link_counts(truth, candidates)
    s2_counts = _sum_link_counts(truth, candidates, prefix="S2-")
    s3_counts = _sum_link_counts(truth, candidates, prefix="S3-")
    total_true = sum(len(links) for links in truth.values())
    s2_true = sum(1 for links in truth.values() for value in links if value.startswith("S2-"))
    s3_true = sum(1 for links in truth.values() for value in links if value.startswith("S3-"))
    blocking_misses = 0
    model_misses = 0
    for s1_id, true_links in truth.items():
        candidate_links = candidates[s1_id]
        predicted_links = prediction[s1_id]
        blocking_misses += len(true_links - candidate_links)
        model_misses += len((true_links & candidate_links) - predicted_links)
    counts = sorted(len(candidates[s1_id]) for s1_id in sorted(truth))
    return CandidateDiagnostics(
        candidate_recall=candidate_link_counts.tp / total_true if total_true else 0.0,
        s2_candidate_recall=s2_counts.tp / s2_true if s2_true else 0.0,
        s3_candidate_recall=s3_counts.tp / s3_true if s3_true else 0.0,
        blocking_misses=blocking_misses,
        model_misses=model_misses,
        mean_candidates=sum(counts) / len(counts) if counts else 0.0,
        median_candidates=_percentile(counts, 50),
        p90_candidates=_percentile(counts, 90),
        p95_candidates=_percentile(counts, 95),
        p99_candidates=_percentile(counts, 99),
        max_candidates=max(counts, default=0),
    )


def build_diagnostics(
    ground_truth: Mapping[str, Iterable[str]],
    predictions: Mapping[str, Iterable[str]],
    candidates: Mapping[str, Iterable[str]] | CandidateDiagnostics | None = None,
) -> DiagnosticsResult:
    """Build diagnostics while delegating official macro F0.5 to scoring.py."""
    truth = {key: value if isinstance(value, set) else set(value) for key, value in ground_truth.items()}
    predicted = {key: value if isinstance(value, set) else set(value) for key, value in predictions.items()}
    score = score_predictions(truth, predicted)
    predicted_singletons = sum(not predicted[s1_id] for s1_id in truth)
    singleton_false_merges = sum(not truth[s1_id] and bool(predicted[s1_id]) for s1_id in truth)
    false_positive_entities = sum(bool(predicted[s1_id] - truth[s1_id]) for s1_id in truth)
    source2 = _link_diagnostics(truth, predicted, "S2-")
    source3 = _link_diagnostics(truth, predicted, "S3-")
    candidate_result = None
    if candidates is not None:
        if isinstance(candidates, CandidateDiagnostics):
            candidate_result = candidates
        else:
            candidate_sets = {key: value if isinstance(value, set) else set(value) for key, value in candidates.items()}
            candidate_result = _candidate_diagnostics(truth, predicted, candidate_sets)
    return DiagnosticsResult(
        score=score,
        predicted_singletons=predicted_singletons,
        singleton_false_merges=singleton_false_merges,
        source2=source2,
        source3=source3,
        false_positive_entities=false_positive_entities,
        candidates=candidate_result,
    )


def load_candidates(path: Path, expected_s1_ids: Iterable[str]) -> dict[str, set[str]]:
    """Read a candidate TSV and require exactly one row per evaluation S1."""
    expected = set(expected_s1_ids)
    result: dict[str, set[str]] = {}
    seen: set[str] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if header != CANDIDATE_HEADER:
            raise ValueError(f"Unexpected candidate columns: {header!r}; expected {CANDIDATE_HEADER!r}")
        for row_num, row in enumerate(reader, start=2):
            if len(row) != 2:
                raise ValueError(f"Malformed candidate row at {path}:{row_num}")
            s1_id, raw_ids = row
            if not s1_id.startswith("S1-") or len(s1_id) <= 3:
                raise ValueError(f"Malformed candidate S1 ID at {path}:{row_num}: {s1_id!r}")
            if s1_id in seen:
                raise ValueError(f"Duplicate candidate S1 row at {path}:{row_num}: {s1_id}")
            seen.add(s1_id)
            if s1_id in expected:
                result[s1_id] = parse_match_list(raw_ids)
    missing, extra = expected - result.keys(), seen - expected
    if missing or extra:
        raise ValueError(f"Candidate S1 coverage mismatch: missing={len(missing)}, extra={len(extra)}")
    return result



def stream_candidate_diagnostics(truth: Mapping[str, set[str]],
                                 prediction: Mapping[str, set[str]],
                                 path: Path) -> CandidateDiagnostics:
    """Stream .tsv or .tsv.gz candidates without storing all candidate pairs."""
    if truth.keys() != prediction.keys():
        raise ValueError("Prediction S1 coverage mismatch")
    opener = gzip.open if path.suffix == ".gz" else open
    seen: set[str] = set()
    sizes: list[int] = []
    recovered = s2_recovered = s3_recovered = blocking_misses = model_misses = 0
    true_links = s2_true = s3_true = 0
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        if next(reader, None) != CANDIDATE_HEADER:
            raise ValueError(f"Unexpected candidate columns in {path}")
        for row_number, row in enumerate(reader, 2):
            if len(row) != 2:
                raise ValueError(f"Malformed candidate row at {path}:{row_number}")
            s1_id, raw = row
            if s1_id not in truth or s1_id in seen:
                raise ValueError(f"Unexpected or duplicate candidate S1: {s1_id}")
            seen.add(s1_id)
            candidates = parse_match_list(raw)
            predicted = prediction[s1_id]
            if not predicted <= candidates:
                raise ValueError(f"Predicted match outside candidates for {s1_id}")
            actual = truth[s1_id]
            matched = actual & candidates
            true_links += len(actual)
            s2_true += sum(value.startswith("S2-") for value in actual)
            s3_true += sum(value.startswith("S3-") for value in actual)
            recovered += len(matched)
            s2_recovered += sum(value.startswith("S2-") for value in matched)
            s3_recovered += sum(value.startswith("S3-") for value in matched)
            blocking_misses += len(actual - candidates)
            model_misses += len(matched - predicted)
            sizes.append(len(candidates))
    if seen != truth.keys():
        raise ValueError(f"Candidate S1 coverage mismatch: missing={len(truth.keys()-seen)}")
    sizes.sort()
    return CandidateDiagnostics(
        candidate_recall=recovered / true_links if true_links else 0.0,
        s2_candidate_recall=s2_recovered / s2_true if s2_true else 0.0,
        s3_candidate_recall=s3_recovered / s3_true if s3_true else 0.0,
        blocking_misses=blocking_misses, model_misses=model_misses,
        mean_candidates=sum(sizes) / len(sizes) if sizes else 0.0,
        median_candidates=_percentile(sizes, 50),
        p90_candidates=_percentile(sizes, 90),
        p95_candidates=_percentile(sizes, 95),
        p99_candidates=_percentile(sizes, 99), max_candidates=max(sizes, default=0),
    )


def _rate(value: float) -> str:
    return f"{value:.8f}"


def render_report(result: DiagnosticsResult) -> str:
    """Render stable, deterministic human-readable report text."""
    score = result.score
    lines = [
        "ENTITY RESOLUTION DIAGNOSTICS",
        "",
        "## OVERALL",
        f"S1 entities evaluated: {score.entities_evaluated:,}",
        f"True links: {score.true_links:,}",
        f"Predicted links: {score.predicted_links:,}",
        f"Macro entity-level F0.5 (primary challenge metric): {_rate(score.macro_f0_5)}",
        f"Total TP: {score.tp:,}",
        f"Total FP: {score.fp:,}",
        f"Total FN: {score.fn:,}",
        f"Micro-style precision diagnostic: {_rate(score.micro_precision_diagnostic)}",
        f"Micro-style recall diagnostic: {_rate(score.micro_recall_diagnostic)}",
        "",
        "## SINGLETONS",
        f"True singleton count: {score.true_singletons:,}",
        f"Predicted singleton count: {result.predicted_singletons:,}",
        f"Correctly predicted singleton count: {score.correctly_predicted_singletons:,}",
        f"SINGLETON_FALSE_MERGE count: {result.singleton_false_merges:,}",
        f"Singleton accuracy: {_rate(score.singleton_accuracy)}",
        "",
    ]
    for label, metrics in (("S2", result.source2), ("S3", result.source3)):
        lines.extend([
            f"## {label} LINK DIAGNOSTICS (diagnostic only)",
            f"True {label} links: {metrics.true_links:,}",
            f"Predicted {label} links: {metrics.predicted_links:,}",
            f"TP: {metrics.tp:,}",
            f"FP: {metrics.fp:,}",
            f"FN: {metrics.fn:,}",
            f"Precision diagnostic: {_rate(metrics.precision_diagnostic)}",
            f"Recall diagnostic: {_rate(metrics.recall_diagnostic)}",
            "",
        ])
    lines.extend([
        "## ENTITY-LEVEL MATCH QUALITY (diagnostic only)",
        f"Perfect match-set entities: {score.perfect_match_sets:,}",
        f"Partial match entities: {score.partial_match_sets:,}",
        f"Completely missed linked entities (no true target recovered): {score.completely_missed_nonempty_truth:,}",
        f"Entities with at least one false positive: {result.false_positive_entities:,}",
        f"FALSE_POSITIVE / FALSE_MERGE predicted links outside truth: {score.fp:,}",
        "",
    ])
    if result.candidates is None:
        lines.extend(["## BLOCKING DIAGNOSTICS", "Not available yet (no candidate file supplied).", ""])
    else:
        c = result.candidates
        lines.extend([
            "## BLOCKING DIAGNOSTICS (diagnostic only)",
            f"Overall candidate recall / pair completeness: {_rate(c.candidate_recall)}",
            f"S2 candidate recall: {_rate(c.s2_candidate_recall)}",
            f"S3 candidate recall: {_rate(c.s3_candidate_recall)}",
            f"BLOCKING_MISS true links absent from candidates: {c.blocking_misses:,}",
            f"MODEL_MISS candidate true links absent from predictions: {c.model_misses:,}",
            f"Mean candidates per S1: {_rate(c.mean_candidates)}",
            f"Median candidates per S1: {_rate(c.median_candidates)}",
            f"P90 candidates per S1: {_rate(c.p90_candidates)}",
            f"P95 candidates per S1: {_rate(c.p95_candidates)}",
            f"P99 candidates per S1: {_rate(c.p99_candidates)}",
            f"Maximum candidates for one S1: {c.max_candidates:,}",
            "",
        ])
    return "\n".join(lines)


def append_experiment(
    path: Path,
    experiment_name: str,
    notes: str,
    result: DiagnosticsResult,
    runtime_seconds: float,
    allow_duplicate: bool = False,
    timestamp: str | None = None,
    peak_memory_mb: float | None = None,
) -> None:
    """Atomically append one CSV row, preserving history and rejecting name clashes."""
    if not experiment_name.strip():
        raise ValueError("experiment name must not be blank")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    if path.exists() and path.stat().st_size:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != EXPERIMENT_COLUMNS:
                raise ValueError(f"Unexpected experiment CSV schema in {path}: {reader.fieldnames!r}")
            rows = list(reader)
        if not allow_duplicate and any(row["experiment_name"] == experiment_name for row in rows):
            raise ValueError(f"Experiment name already exists: {experiment_name!r}")
    else:
        rows = []
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    candidate = result.candidates
    values: dict[str, object] = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "notes": notes,
        "macro_f0_5": result.score.macro_f0_5,
        "precision_diag": result.score.micro_precision_diagnostic,
        "recall_diag": result.score.micro_recall_diagnostic,
        "singleton_accuracy": result.score.singleton_accuracy,
        "singleton_false_merges": result.singleton_false_merges,
        "candidate_recall": candidate.candidate_recall if candidate else "",
        "s2_candidate_recall": candidate.s2_candidate_recall if candidate else "",
        "s3_candidate_recall": candidate.s3_candidate_recall if candidate else "",
        "avg_candidates": candidate.mean_candidates if candidate else "",
        "p95_candidates": candidate.p95_candidates if candidate else "",
        "p99_candidates": candidate.p99_candidates if candidate else "",
        "runtime_seconds": runtime_seconds,
        "peak_memory_mb": peak_memory_mb if peak_memory_mb is not None else "",
    }
    row = {key: (f"{value:.8f}" if isinstance(value, float) else str(value)) for key, value in values.items()}
    rows.append(row)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=EXPERIMENT_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise




def evaluate_candidates_only(truth: Mapping[str, set[str]], path: Path) -> dict[str, float | int]:
    """Stream a candidate TSV, including .gz, without creating model predictions."""
    opener = gzip.open if path.suffix == ".gz" else open
    seen: set[str] = set()
    sizes: list[int] = []
    true_links = s2_true = s3_true = recovered = s2_found = s3_found = 0
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        if next(reader, None) != CANDIDATE_HEADER:
            raise ValueError(f"Unexpected candidate columns in {path}")
        for row_number, row in enumerate(reader, 2):
            if len(row) != 2:
                raise ValueError(f"Malformed candidate row: {path}:{row_number}")
            s1_id, raw = row
            if s1_id not in truth or s1_id in seen:
                raise ValueError(f"Unexpected or duplicate candidate S1: {s1_id}")
            seen.add(s1_id)
            candidates = parse_match_list(raw)
            links = truth[s1_id]
            counts = count_link_errors(links, candidates)
            s2_counts = count_link_errors(links, candidates, prefix="S2-")
            s3_counts = count_link_errors(links, candidates, prefix="S3-")
            true_links += len(links)
            s2_true += sum(value.startswith("S2-") for value in links)
            s3_true += sum(value.startswith("S3-") for value in links)
            recovered += counts.tp
            s2_found += s2_counts.tp
            s3_found += s3_counts.tp
            sizes.append(len(candidates))
    if seen != truth.keys():
        raise ValueError(f"Candidate S1 coverage mismatch: missing={len(truth.keys() - seen)}")
    sizes.sort()
    return {
        "s1_entities": len(seen), "true_links": true_links,
        "s2_true_links": s2_true, "s3_true_links": s3_true,
        "recovered_links": recovered, "s2_recovered_links": s2_found,
        "s3_recovered_links": s3_found, "blocking_misses": true_links - recovered,
        "s2_blocking_misses": s2_true - s2_found, "s3_blocking_misses": s3_true - s3_found,
        "candidate_recall": recovered / true_links if true_links else 0.0,
        "s2_candidate_recall": s2_found / s2_true if s2_true else 0.0,
        "s3_candidate_recall": s3_found / s3_true if s3_true else 0.0,
        "total_candidate_pairs": sum(sizes),
        "mean_candidates": sum(sizes) / len(sizes) if sizes else 0.0,
        "median_candidates": _percentile(sizes, 50),
        "p90_candidates": _percentile(sizes, 90),
        "p95_candidates": _percentile(sizes, 95),
        "p99_candidates": _percentile(sizes, 99),
        "max_candidates": max(sizes, default=0),
        "zero_candidate_pct": 100 * sizes.count(0) / len(sizes) if sizes else 0.0,
        "over_1000_candidate_pct": 100 * sum(value > 1000 for value in sizes) / len(sizes) if sizes else 0.0,
    }



def compare_candidate_files(truth: Mapping[str, set[str]], v1_path: Path, v2_path: Path) -> dict[str, int | float]:
    """Stream V1/V2 candidate lists and count recovered V1 misses and pair growth."""
    def opener(path: Path):
        return gzip.open(path, "rt", encoding="utf-8", newline="") if path.suffix == ".gz" else path.open(encoding="utf-8", newline="")
    recovered = {"overall": 0, "S2": 0, "S3": 0}
    v1_misses = {"overall": 0, "S2": 0, "S3": 0}
    remaining = {"overall": 0, "S2": 0, "S3": 0}
    v1_pairs = v2_pairs = 0
    with opener(v1_path) as h1, opener(v2_path) as h2:
        r1, r2 = csv.reader(h1, delimiter="\t"), csv.reader(h2, delimiter="\t")
        if next(r1, None) != CANDIDATE_HEADER or next(r2, None) != CANDIDATE_HEADER:
            raise ValueError("Unexpected V1/V2 candidate header")
        for s1_id in sorted(truth):
            row1, row2 = next(r1, None), next(r2, None)
            if row1 is None or row2 is None or len(row1) != 2 or len(row2) != 2 or row1[0] != s1_id or row2[0] != s1_id:
                raise ValueError(f"V1/V2 candidate coverage mismatch at {s1_id}")
            old, new = parse_match_list(row1[1]), parse_match_list(row2[1])
            if not old <= new:
                raise ValueError(f"V1 candidate absent from V2 for {s1_id}")
            v1_pairs += len(old)
            v2_pairs += len(new)
            missing_before = truth[s1_id] - old
            missing_after = truth[s1_id] - new
            recovered_here = missing_before - missing_after
            for key, values in (("overall", missing_before), ("S2", {v for v in missing_before if v.startswith("S2-")}),
                                ("S3", {v for v in missing_before if v.startswith("S3-")})):
                v1_misses[key] += len(values)
            for key, values in (("overall", recovered_here), ("S2", {v for v in recovered_here if v.startswith("S2-")}),
                                ("S3", {v for v in recovered_here if v.startswith("S3-")})):
                recovered[key] += len(values)
            for key, values in (("overall", missing_after), ("S2", {v for v in missing_after if v.startswith("S2-")}),
                                ("S3", {v for v in missing_after if v.startswith("S3-")})):
                remaining[key] += len(values)
        if next(r1, None) is not None or next(r2, None) is not None:
            raise ValueError("Extra S1 rows in V1/V2 candidate files")
    def ratio(n: int, d: int) -> float:
        return n / d if d else 0.0
    return {
        "v1_candidate_pairs": v1_pairs, "v2_candidate_pairs": v2_pairs,
        "candidate_pair_growth": v2_pairs - v1_pairs,
        "candidate_pair_growth_pct": 100 * ratio(v2_pairs - v1_pairs, v1_pairs),
        "v1_blocking_misses": v1_misses["overall"],
        "v1_misses_recovered": recovered["overall"],
        "v1_miss_recovery_pct": 100 * ratio(recovered["overall"], v1_misses["overall"]),
        "v2_blocking_misses": remaining["overall"],
        "s2_v1_blocking_misses": v1_misses["S2"],
        "s2_v1_misses_recovered": recovered["S2"],
        "s2_v1_miss_recovery_pct": 100 * ratio(recovered["S2"], v1_misses["S2"]),
        "s2_v2_blocking_misses": remaining["S2"],
        "s3_v1_blocking_misses": v1_misses["S3"],
        "s3_v1_misses_recovered": recovered["S3"],
        "s3_v1_miss_recovery_pct": 100 * ratio(recovered["S3"], v1_misses["S3"]),
        "s3_v2_blocking_misses": remaining["S3"],
    }

def render_candidate_only_report(result: Mapping[str, float | int]) -> str:
    lines = ["CANDIDATE GENERATION DIAGNOSTICS (no match predictions)"]
    lines.extend(f"{key}: {value:.8f}" if isinstance(value, float) else f"{key}: {value:,}" for key, value in result.items())
    return "\n".join(lines) + "\n"


def append_candidate_experiment(path: Path, name: str, notes: str, metrics: Mapping[str, float | int], runtime_seconds: float, peak_memory_mb: float | None = None) -> None:
    """Append blocking metrics to the existing schema with prediction fields blank."""
    if not name.strip():
        raise ValueError("experiment name must not be blank")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if path.exists() and path.stat().st_size:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != EXPERIMENT_COLUMNS:
                raise ValueError("Unexpected experiment CSV schema")
            rows = list(reader)
    if any(row["experiment_name"] == name for row in rows):
        raise ValueError(f"Experiment name already exists: {name}")
    values = {key: "" for key in EXPERIMENT_COLUMNS}
    values.update({
        "experiment_name": name,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": notes,
        "candidate_recall": metrics["candidate_recall"],
        "s2_candidate_recall": metrics["s2_candidate_recall"],
        "s3_candidate_recall": metrics["s3_candidate_recall"],
        "avg_candidates": metrics["mean_candidates"],
        "p95_candidates": metrics["p95_candidates"],
        "p99_candidates": metrics["p99_candidates"],
        "runtime_seconds": runtime_seconds,
        "peak_memory_mb": peak_memory_mb if peak_memory_mb is not None else "",
    })
    rows.append({key: str(values[key]) for key in EXPERIMENT_COLUMNS})
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=EXPERIMENT_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--candidates-only", action="store_true", help="Evaluate blocking without match predictions")
    parser.add_argument("--runtime-seconds", type=float, help="Measured generator runtime for experiment metadata")
    parser.add_argument("--peak-memory-mb", type=float, help="Measured generator peak RSS for experiment metadata")
    parser.add_argument("--s1-ids", type=Path, default=DEFAULT_S1_IDS)
    parser.add_argument("--candidates", type=Path, help="Optional candidate_pairs-style TSV")
    parser.add_argument("--v1-candidates", type=Path, help="Optional frozen Phase 4 candidates for V1-to-V2 comparison")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--experiment-name")
    parser.add_argument("--notes", default="")
    parser.add_argument("--experiments-file", type=Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--allow-duplicate-experiment", action="store_true")
    args = parser.parse_args(argv)
    if args.notes and not args.experiment_name:
        parser.error("--notes requires --experiment-name")
    if args.candidates_only and not args.candidates:
        parser.error("--candidates-only requires --candidates")
    if args.v1_candidates and not args.candidates_only:
        parser.error("--v1-candidates is available with --candidates-only")
    if not args.candidates_only and not args.predictions:
        parser.error("--predictions is required unless --candidates-only is used")
    started = time.perf_counter()
    try:
        s1_ids = load_id_file(args.s1_ids)
        truth = load_ground_truth(args.ground_truth, s1_ids)
        if args.candidates_only:
            metrics = evaluate_candidates_only(truth, args.candidates)
            if args.v1_candidates:
                metrics.update(compare_candidate_files(truth, args.v1_candidates, args.candidates))
            report = render_candidate_only_report(metrics)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report, encoding="utf-8")
            print(report, end="")
            if args.experiment_name:
                append_candidate_experiment(args.experiments_file, args.experiment_name, args.notes, metrics, args.runtime_seconds if args.runtime_seconds is not None else time.perf_counter()-started, args.peak_memory_mb)
            return 0
        predictions = load_predictions(args.predictions)
        candidates = stream_candidate_diagnostics(truth, predictions, args.candidates) if args.candidates else None
        result = build_diagnostics(truth, predictions, candidates)
        report = render_report(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(report, end="")
        if args.experiment_name:
            append_experiment(
                args.experiments_file, args.experiment_name, args.notes, result,
                args.runtime_seconds if args.runtime_seconds is not None else time.perf_counter() - started,
                args.allow_duplicate_experiment, peak_memory_mb=args.peak_memory_mb,
            )
    except (OSError, ValueError, csv.Error) as exc:
        print(f"FAIL: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
