#!/usr/bin/env python3
"""Exact challenge-style macro entity-level F0.5 scorer."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping


GROUND_TRUTH_HEADER = ["source1_entity_id", "matched_entity_ids"]
PREDICTION_HEADER = GROUND_TRUTH_HEADER
BETA_SQUARED = 0.25


def parse_match_list(value: str) -> set[str]:
    """Parse comma-separated IDs; empty means no matches, duplicates are invalid."""
    if value == "":
        return set()
    ids = value.split(",")
    if any(not item for item in ids):
        raise ValueError("match list contains an empty ID")
    if len(ids) != len(set(ids)):
        raise ValueError("match list contains duplicate IDs")
    for item in ids:
        if not (item.startswith("S2-") or item.startswith("S3-")) or len(item) <= 3:
            raise ValueError(f"invalid matched ID: {item!r}")
    return set(ids)


def score_entity(truth: Iterable[str], prediction: Iterable[str]) -> float:
    """Return one entity's F0.5, including the challenge's singleton rule."""
    truth_set, prediction_set = set(truth), set(prediction)
    if not truth_set:
        return 1.0 if not prediction_set else 0.0
    if not prediction_set:
        return 0.0
    tp = len(truth_set & prediction_set)
    fp = len(prediction_set - truth_set)
    fn = len(truth_set - prediction_set)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    denominator = BETA_SQUARED * precision + recall
    return (1.0 + BETA_SQUARED) * precision * recall / denominator if denominator else 0.0


@dataclass(frozen=True)
class ScoreReport:
    macro_f0_5: float
    entities_evaluated: int
    true_links: int
    predicted_links: int
    tp: int
    fp: int
    fn: int
    micro_precision_diagnostic: float
    micro_recall_diagnostic: float
    true_singletons: int
    correctly_predicted_singletons: int
    singleton_accuracy: float
    perfect_match_sets: int
    partial_match_sets: int
    completely_missed_nonempty_truth: int


@dataclass(frozen=True)
class LinkCounts:
    tp: int
    fp: int
    fn: int


def count_link_errors(
    truth: Iterable[str], prediction: Iterable[str], prefix: str | None = None
) -> LinkCounts:
    """Return set-based TP/FP/FN, optionally restricted to one target source."""
    truth_set, prediction_set = set(truth), set(prediction)
    if prefix is not None:
        truth_set = {value for value in truth_set if value.startswith(prefix)}
        prediction_set = {value for value in prediction_set if value.startswith(prefix)}
    return LinkCounts(
        tp=len(truth_set & prediction_set),
        fp=len(prediction_set - truth_set),
        fn=len(truth_set - prediction_set),
    )


def precision_recall_diagnostics(counts: LinkCounts) -> tuple[float, float]:
    """Return micro-style precision and recall diagnostics with zero-safe rates."""
    precision = counts.tp / (counts.tp + counts.fp) if counts.tp + counts.fp else 0.0
    recall = counts.tp / (counts.tp + counts.fn) if counts.tp + counts.fn else 0.0
    return precision, recall


def score_predictions(
    ground_truth: Mapping[str, Iterable[str]],
    predictions: Mapping[str, Iterable[str]],
    s1_ids: Iterable[str] | None = None,
) -> ScoreReport:
    """Score exact S1 coverage; omitted rows must not silently become negatives."""
    evaluation_ids = set(s1_ids) if s1_ids is not None else set(ground_truth)
    missing_truth = evaluation_ids - ground_truth.keys()
    extra_truth = ground_truth.keys() - evaluation_ids
    if missing_truth or extra_truth:
        raise ValueError(
            f"Ground-truth S1 coverage mismatch: missing={len(missing_truth)}, extra={len(extra_truth)}"
        )
    missing_predictions = evaluation_ids - predictions.keys()
    extra_predictions = predictions.keys() - evaluation_ids
    if missing_predictions or extra_predictions:
        raise ValueError(
            "Prediction S1 coverage mismatch: "
            f"missing={len(missing_predictions)}, extra={len(extra_predictions)}"
        )
    if not evaluation_ids:
        raise ValueError("Cannot score an empty S1 evaluation set")

    per_entity: list[float] = []
    tp_total = fp_total = fn_total = truth_links = pred_links = 0
    true_singletons = correct_singletons = perfect = partial = missed = 0
    for s1_id in evaluation_ids:
        truth = set(ground_truth[s1_id])
        predicted = set(predictions[s1_id])
        per_entity.append(score_entity(truth, predicted))
        counts = count_link_errors(truth, predicted)
        tp_total += counts.tp
        fp_total += counts.fp
        fn_total += counts.fn
        truth_links += len(truth)
        pred_links += len(predicted)
        if not truth:
            true_singletons += 1
            correct_singletons += int(not predicted)
        if truth == predicted:
            perfect += 1
        elif counts.tp:
            partial += 1
        elif truth:
            missed += 1

    micro_precision, micro_recall = precision_recall_diagnostics(
        LinkCounts(tp=tp_total, fp=fp_total, fn=fn_total)
    )
    return ScoreReport(
        macro_f0_5=sum(per_entity) / len(per_entity),
        entities_evaluated=len(evaluation_ids),
        true_links=truth_links,
        predicted_links=pred_links,
        tp=tp_total,
        fp=fp_total,
        fn=fn_total,
        micro_precision_diagnostic=micro_precision,
        micro_recall_diagnostic=micro_recall,
        true_singletons=true_singletons,
        correctly_predicted_singletons=correct_singletons,
        singleton_accuracy=correct_singletons / true_singletons if true_singletons else 0.0,
        perfect_match_sets=perfect,
        partial_match_sets=partial,
        completely_missed_nonempty_truth=missed,
    )


def load_id_file(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        ids = [line.rstrip("\r\n") for line in handle if line.rstrip("\r\n")]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate S1 IDs in {path}")
    return ids


def _load_pairs(path: Path, expected_header: list[str], wanted: set[str] | None = None) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if header != expected_header:
            raise ValueError(f"Unexpected columns in {path}: {header!r}; expected {expected_header!r}")
        for row_num, row in enumerate(reader, start=2):
            if len(row) != 2:
                raise ValueError(f"Malformed row at {path}:{row_num}")
            s1_id, raw_ids = row
            if not s1_id.startswith("S1-") or len(s1_id) <= 3:
                raise ValueError(f"Malformed S1 ID at {path}:{row_num}: {s1_id!r}")
            if s1_id in result:
                raise ValueError(f"Duplicate S1 row in {path}:{row_num}: {s1_id}")
            if wanted is None or s1_id in wanted:
                result[s1_id] = parse_match_list(raw_ids)
            elif path.name != "train_ground_truth.tsv":
                result[s1_id] = parse_match_list(raw_ids)
    return result


def load_ground_truth(path: Path, s1_ids: Iterable[str] | None = None) -> dict[str, set[str]]:
    wanted = set(s1_ids) if s1_ids is not None else None
    result = _load_pairs(path, GROUND_TRUTH_HEADER, wanted)
    if wanted is not None and result.keys() != wanted:
        raise ValueError(f"Ground-truth S1 coverage mismatch: missing={len(wanted-result.keys())}, extra={len(result.keys()-wanted)}")
    return result


def load_predictions(path: Path) -> dict[str, set[str]]:
    return _load_pairs(path, PREDICTION_HEADER)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--s1-ids", type=Path, help="Optional one-ID-per-line evaluation set")
    args = parser.parse_args()
    try:
        s1_ids = load_id_file(args.s1_ids) if args.s1_ids else None
        truth = load_ground_truth(args.ground_truth, s1_ids)
        predictions = load_predictions(args.predictions)
        report = score_predictions(truth, predictions, s1_ids)
    except (OSError, ValueError, csv.Error) as exc:
        print(f"FAIL: {exc}")
        return 1
    values = asdict(report)
    print(f"PRIMARY challenge metric — macro entity-level F0.5: {report.macro_f0_5:.8f}")
    for key, value in values.items():
        if key == "macro_f0_5":
            continue
        print(f"{key}: {value:.8f}" if isinstance(value, float) else f"{key}: {value:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
