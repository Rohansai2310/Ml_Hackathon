#!/usr/bin/env python3
"""Phase 8: deterministic pair features and separate S2/S3 LightGBM models.

This module consumes frozen Phase 7 pairs and the frozen V1 + Address K10 tune
candidate artifacts. It deliberately emits unfiltered pair scores: entity-level
thresholding, singleton handling, and target conflicts belong to Phase 9.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import resource
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import lightgbm as lgb
import numpy as np
from rapidfuzz import fuzz
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from baseline import Record, TargetStore, load_s1_records
from blocking import METADATA_HEADER
from diagnostics import CANDIDATE_HEADER, DEFAULT_DATASET_ROOT
from phase7_pairs import merge_candidate_routes
from scoring import load_id_file


BASE = Path(__file__).resolve().parents[1]
ART = BASE / "artifacts"
PHASE7 = ART / "model_data/phase7"
PHASE8 = ART / "model/phase8"
SPLITS = ART / "splits"
TUNE_IDS = SPLITS / "tune_s1_ids.txt"
VAL_IDS = SPLITS / "val_s1_ids.txt"
TARGET_INDEX = ART / "blocking/v1_index.sqlite"
TUNE_V1_CANDIDATES = ART / "baseline/tune_candidates.tsv.gz"
TUNE_V1_METADATA = ART / "baseline/tune_metadata.tsv.gz"
TUNE_ADDRESS = ART / "retrieval_diagnosis/phase6e/address_top10_100k.tsv.gz"
SEED = 20260925
THREADS = 4
TRAIN_BATCH_ROWS = 25_000
SCORE_BENCHMARK_S1 = 1_000
SCORE_HEADER = [
    "source1_entity_id", "candidate_entity_id", "score", "from_v1",
    "from_address", "address_rank", "address_score",
]
PAIR_REQUIRED_COLUMNS = [
    "source1_entity_id", "candidate_entity_id", "target_source", "label",
    "from_v1", "from_address", "address_rank", "address_score", "negative_reason",
]
ADDRESS_HEADER = ["source1_entity_id", "candidate_entity_id", "target_source", "route", "score", "rank"]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    description: str
    missing: str = "0.0"


FEATURE_SPECS = (
    FeatureSpec("name_basic_exact", "Non-empty normalized business names are equal."),
    FeatureSpec("name_core_exact", "Non-empty legal-suffix-normalized business names are equal."),
    FeatureSpec("name_ratio", "RapidFuzz character ratio on normalized names."),
    FeatureSpec("name_token_sort_ratio", "RapidFuzz token-sort ratio on core names."),
    FeatureSpec("name_token_set_ratio", "RapidFuzz token-set ratio on core names."),
    FeatureSpec("name_token_jaccard", "Jaccard overlap of core-name token sets."),
    FeatureSpec("name_shared_token_count", "Count of shared core-name tokens."),
    FeatureSpec("name_shared_fraction_left", "Shared core tokens divided by S1 core-token count."),
    FeatureSpec("name_shared_fraction_right", "Shared core tokens divided by target core-token count."),
    FeatureSpec("name_length_ratio", "Shorter/longer normalized-name length."),
    FeatureSpec("name_prefix_agree", "First non-empty core-name tokens agree."),
    FeatureSpec("name_suffix_agree", "Last non-empty core-name tokens agree."),
    FeatureSpec("name_missing_left", "S1 normalized name is blank."),
    FeatureSpec("name_missing_right", "Target normalized name is blank."),
    FeatureSpec("name_missing_both", "Both normalized names are blank."),
    FeatureSpec("address_exact", "Non-empty normalized addresses are equal."),
    FeatureSpec("address_ratio", "RapidFuzz character ratio on normalized addresses."),
    FeatureSpec("address_token_jaccard", "Jaccard overlap of standardized address tokens."),
    FeatureSpec("address_shared_token_count", "Count of shared standardized address tokens."),
    FeatureSpec("address_shared_fraction_left", "Shared address tokens divided by S1 token count."),
    FeatureSpec("address_shared_fraction_right", "Shared address tokens divided by target token count."),
    FeatureSpec("address_length_ratio", "Shorter/longer normalized-address length."),
    FeatureSpec("address_missing_left", "S1 normalized address is blank."),
    FeatureSpec("address_missing_right", "Target normalized address is blank."),
    FeatureSpec("address_missing_both", "Both normalized addresses are blank."),
    FeatureSpec("numeric_count_left", "Number of non-postal numeric tokens on S1."),
    FeatureSpec("numeric_count_right", "Number of non-postal numeric tokens on target."),
    FeatureSpec("numeric_shared_count", "Count of shared non-postal numeric tokens."),
    FeatureSpec("numeric_shared", "At least one non-postal numeric token agrees."),
    FeatureSpec("numeric_conflict", "Both sides have numeric evidence but none agrees."),
    FeatureSpec("postal_shared", "At least one heuristic postal candidate agrees."),
    FeatureSpec("postal_conflict", "Both sides have postals but none agrees."),
    FeatureSpec("postal_missing_left", "S1 has no heuristic postal candidate."),
    FeatureSpec("postal_missing_right", "Target has no heuristic postal candidate."),
    FeatureSpec("country_equal", "Normalized countries agree."),
    FeatureSpec("country_missing_left", "S1 normalized country is blank."),
    FeatureSpec("country_missing_right", "Target normalized country is blank."),
    FeatureSpec("from_v1", "Candidate was retrieved by frozen V1 blocking."),
    FeatureSpec("from_address", "Candidate was retrieved by frozen Address K10."),
    FeatureSpec("from_v1_and_address", "Candidate was retrieved by both frozen routes."),
    FeatureSpec("address_rank", "Address-route rank; zero when no address route."),
    FeatureSpec("address_rank_reciprocal", "One divided by Address-route rank; zero when absent."),
    FeatureSpec("address_rank_log1p", "log1p(Address-route rank); zero when absent."),
    FeatureSpec("address_rank_missing", "Candidate has no Address-route rank."),
    FeatureSpec("address_score", "Address-route retrieval score; zero when absent."),
    FeatureSpec("address_top_candidate", "Candidate has Address-route rank one."),
    FeatureSpec("candidate_count_source", "Full union candidate count for this S1 and target source."),
    FeatureSpec("candidate_count_log1p", "log1p(full union candidate count for source)."),
)
FEATURE_NAMES = tuple(spec.name for spec in FEATURE_SPECS)


MODEL_CONFIGS = (
    {"name": "balanced_63", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100,
     "feature_fraction": 0.90, "bagging_fraction": 0.90, "lambda_l2": 0.5},
    {"name": "shallow_31", "learning_rate": 0.06, "num_leaves": 31, "min_data_in_leaf": 200,
     "feature_fraction": 1.00, "bagging_fraction": 0.90, "lambda_l2": 1.0},
    {"name": "wide_95", "learning_rate": 0.03, "num_leaves": 95, "min_data_in_leaf": 150,
     "feature_fraction": 0.90, "bagging_fraction": 0.85, "lambda_l2": 1.0},
)


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _ratio(left: str, right: str) -> float:
    return fuzz.ratio(left, right) / 100.0 if left and right else 0.0


def _token_ratio(method, left: str, right: str) -> float:
    return method(left, right) / 100.0 if left and right else 0.0


def _length_ratio(left: str, right: str) -> float:
    return min(len(left), len(right)) / max(len(left), len(right)) if left and right else 0.0


def _fraction(shared: int, total: int) -> float:
    return shared / total if total else 0.0


def _as_float(value: str | int | float | None) -> float:
    try:
        number = float(value) if value not in (None, "") else 0.0
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _as_int(value: str | int | float | None) -> int:
    try:
        return int(float(value)) if value not in (None, "") else 0
    except (TypeError, ValueError):
        return 0


def validate_feature_names(names: Iterable[str]) -> None:
    values = tuple(names)
    forbidden = {"source1_entity_id", "candidate_entity_id", "label", "negative_reason", "target_source"}
    if values != FEATURE_NAMES:
        raise ValueError("feature order does not match the frozen Phase 8 schema")
    if forbidden & set(values):
        raise ValueError("entity IDs, labels, or sampling metadata cannot be model features")
    if len(values) != len(set(values)):
        raise ValueError("feature names are not unique")


def extract_features(left: Record, right: Record, evidence: Mapping[str, str | int | float],
                     candidate_count_source: int) -> np.ndarray:
    """Create the complete ordered float32 feature vector without labels or IDs."""
    name_shared = left.core_tokens & right.core_tokens
    address_shared = left.address_words & right.address_words
    numeric_shared = left.numbers & right.numbers
    postal_shared = left.postals & right.postals
    basic_left, basic_right = left.basic, right.basic
    address_left, address_right = left.address_basic, right.address_basic
    rank = _as_int(evidence.get("address_rank"))
    score = _as_float(evidence.get("address_score"))
    from_v1 = float(bool(_as_int(evidence.get("from_v1"))))
    from_address = float(bool(_as_int(evidence.get("from_address"))))
    values = (
        float(bool(basic_left and basic_left == basic_right)),
        float(bool(left.core and left.core == right.core)),
        _ratio(basic_left, basic_right),
        _token_ratio(fuzz.token_sort_ratio, left.core, right.core),
        _token_ratio(fuzz.token_set_ratio, left.core, right.core),
        _fraction(len(name_shared), len(left.core_tokens | right.core_tokens)),
        float(len(name_shared)),
        _fraction(len(name_shared), len(left.core_tokens)),
        _fraction(len(name_shared), len(right.core_tokens)),
        _length_ratio(basic_left, basic_right),
        float(bool(left.core and right.core and left.core.split()[0] == right.core.split()[0])),
        float(bool(left.core and right.core and left.core.split()[-1] == right.core.split()[-1])),
        float(not basic_left), float(not basic_right), float(not basic_left and not basic_right),
        float(bool(address_left and address_left == address_right)),
        _ratio(address_left, address_right),
        _fraction(len(address_shared), len(left.address_words | right.address_words)),
        float(len(address_shared)),
        _fraction(len(address_shared), len(left.address_words)),
        _fraction(len(address_shared), len(right.address_words)),
        _length_ratio(address_left, address_right),
        float(not address_left), float(not address_right), float(not address_left and not address_right),
        float(len(left.numbers)), float(len(right.numbers)), float(len(numeric_shared)),
        float(bool(numeric_shared)), float(bool(left.numbers and right.numbers and not numeric_shared)),
        float(bool(postal_shared)), float(bool(left.postals and right.postals and not postal_shared)),
        float(not left.postals), float(not right.postals),
        float(left.country == right.country), float(not left.country), float(not right.country),
        from_v1, from_address, float(bool(from_v1 and from_address)),
        float(rank), 1.0 / rank if rank > 0 else 0.0, math.log1p(rank) if rank > 0 else 0.0,
        float(rank == 0), score, float(rank == 1), float(candidate_count_source), math.log1p(candidate_count_source),
    )
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError("feature extractor produced invalid features")
    return result


def feature_manifest() -> dict:
    validate_feature_names(FEATURE_NAMES)
    return {
        "phase": "8", "dtype": "float32", "feature_count": len(FEATURE_NAMES),
        "feature_order": list(FEATURE_NAMES),
        "features": [{"name": spec.name, "dtype": "float32", "missing": spec.missing,
                      "description": spec.description} for spec in FEATURE_SPECS],
        "excluded_fields": ["source1_entity_id", "candidate_entity_id", "label", "negative_reason", "target_source"],
        "normalization_source": "normalize.py through baseline.make_record",
    }


def write_manifest(output_dir: Path = PHASE8) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "feature_manifest.json"
    path.write_text(json.dumps(feature_manifest(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def frozen_snapshot(paths: Iterable[Path]) -> dict[str, dict[str, int]]:
    result = {}
    for path in paths:
        stat = path.stat()
        result[str(path)] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return result


def frozen_paths() -> tuple[Path, ...]:
    return (
        PHASE7 / "train_pairs_s2.tsv.gz", PHASE7 / "train_pairs_s3.tsv.gz",
        PHASE7 / "train_subset_ids.txt", PHASE7 / "v1_candidates.tsv.gz",
        PHASE7 / "v1_metadata.tsv.gz", PHASE7 / "address_k10_candidates.tsv.gz",
        TUNE_IDS, TUNE_V1_CANDIDATES, TUNE_V1_METADATA, TUNE_ADDRESS, TARGET_INDEX,
    )


def internal_s1_split(ids: Iterable[str], seed: int = SEED) -> tuple[set[str], set[str]]:
    ordered = sorted(ids)
    if len(ordered) != len(set(ordered)):
        raise ValueError("internal model-selection IDs contain duplicates")
    indices = np.random.default_rng(seed).permutation(len(ordered))
    fit_size = int(0.8 * len(ordered))
    fit = {ordered[int(i)] for i in indices[:fit_size]}
    selection = {ordered[int(i)] for i in indices[fit_size:]}
    if fit & selection or fit | selection != set(ordered):
        raise AssertionError("internal S1 split is invalid")
    return fit, selection


def _next_group(reader: Iterator[dict[str, str]], current: dict[str, str] | None,
                s1_id: str) -> tuple[list[dict[str, str]], dict[str, str] | None]:
    rows: list[dict[str, str]] = []
    while current is not None and current["source1_entity_id"] == s1_id:
        rows.append(current)
        current = next(reader, None)
    if current is not None and current["source1_entity_id"] < s1_id:
        raise ValueError("candidate stream is not sorted by source1_entity_id")
    return rows, current


def iter_union_groups(ids: list[str], v1_candidates: Path, v1_metadata: Path,
                      address_candidates: Path, *,
                      require_exhausted: bool = True) -> Iterator[tuple[str, dict[str, dict]]]:
    """Yield the deterministic V1 union Address K10 map for every requested S1."""
    with gzip.open(v1_candidates, "rt", encoding="utf-8", newline="") as ch, \
         gzip.open(v1_metadata, "rt", encoding="utf-8", newline="") as mh, \
         gzip.open(address_candidates, "rt", encoding="utf-8", newline="") as ah:
        cr, mr, ar = (csv.DictReader(ch, delimiter="\t"), csv.DictReader(mh, delimiter="\t"), csv.DictReader(ah, delimiter="\t"))
        if cr.fieldnames != CANDIDATE_HEADER or mr.fieldnames != METADATA_HEADER or ar.fieldnames != ADDRESS_HEADER:
            raise ValueError("unexpected frozen tune candidate headers")
        meta_current, address_current = next(mr, None), next(ar, None)
        for s1_id in ids:
            candidate = next(cr, None)
            if candidate is None or candidate["source1_entity_id"] != s1_id:
                raise ValueError(f"V1 tune candidate coverage/order mismatch at {s1_id}")
            target_ids = set(candidate["candidate_entity_ids"].split(",")) if candidate["candidate_entity_ids"] else set()
            meta_rows, meta_current = _next_group(mr, meta_current, s1_id)
            if {row["candidate_entity_id"] for row in meta_rows} != target_ids:
                raise ValueError(f"V1 tune candidate/metadata mismatch at {s1_id}")
            address_rows, address_current = _next_group(ar, address_current, s1_id)
            yield s1_id, merge_candidate_routes(meta_rows, address_rows)
        if require_exhausted and (next(cr, None) is not None or meta_current is not None or address_current is not None):
            raise ValueError("frozen tune candidate streams include extra S1 rows")


def union_candidate_counts(ids: list[str], v1_candidates: Path, v1_metadata: Path,
                           address_candidates: Path, *,
                           require_exhausted: bool = True) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for s1_id, pairs in iter_union_groups(
            ids, v1_candidates, v1_metadata, address_candidates,
            require_exhausted=require_exhausted):
        counts[s1_id] = {source: sum(row["target_source"] == source for row in pairs.values())
                         for source in ("S2", "S3")}
    return counts


def _iter_pair_rows(path: Path) -> Iterator[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != PAIR_REQUIRED_COLUMNS:
            raise ValueError(f"unexpected Phase 7 pair header in {path}")
        for row in reader:
            if row["target_source"] not in ("S2", "S3"):
                raise ValueError("unexpected target source in Phase 7 pairs")
            if row["negative_reason"] and row["label"] == "1":
                raise ValueError("positive pair carries sampling metadata")
            yield row


def _chunks(rows: Iterable[dict[str, str]], size: int) -> Iterator[list[dict[str, str]]]:
    batch: list[dict[str, str]] = []
    for row in rows:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_training_matrices(source: str, dataset_root: Path = DEFAULT_DATASET_ROOT,
                            output_dir: Path = PHASE8) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Build bounded batches of sampled Phase 7 features for one target source."""
    if source not in ("S2", "S3"):
        raise ValueError("source must be S2 or S3")
    pair_path = PHASE7 / f"train_pairs_{source.lower()}.tsv.gz"
    ids = load_id_file(PHASE7 / "train_subset_ids.txt")
    s1 = load_s1_records(dataset_root, ids)
    counts = union_candidate_counts(ids, PHASE7 / "v1_candidates.tsv.gz", PHASE7 / "v1_metadata.tsv.gz",
                                    PHASE7 / "address_k10_candidates.tsv.gz")
    fit_ids, selection_ids = internal_s1_split(ids)
    x_fit: list[np.ndarray] = []; y_fit: list[np.ndarray] = []
    x_selection: list[np.ndarray] = []; y_selection: list[np.ndarray] = []
    store = TargetStore(TARGET_INDEX)
    start = time.perf_counter(); total = Counter(); sums = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    sums_sq = np.zeros(len(FEATURE_NAMES), dtype=np.float64); mins = np.full(len(FEATURE_NAMES), np.inf)
    maxs = np.full(len(FEATURE_NAMES), -np.inf); by_label_sum = {0: np.zeros(len(FEATURE_NAMES)), 1: np.zeros(len(FEATURE_NAMES))}; by_label_n = Counter()
    try:
        for batch in _chunks(_iter_pair_rows(pair_path), TRAIN_BATCH_ROWS):
            if any(row["target_source"] != source for row in batch):
                raise ValueError("Phase 7 pair file mixes target sources")
            target_ids = sorted({row["candidate_entity_id"] for row in batch})
            targets = store.get_many(target_ids)
            features = np.vstack([extract_features(s1[row["source1_entity_id"]], targets[row["candidate_entity_id"]], row,
                                                    counts[row["source1_entity_id"]][source]) for row in batch])
            labels = np.asarray([int(row["label"]) for row in batch], dtype=np.int8)
            if not np.isfinite(features).all():
                raise ValueError("training matrix contains NaN or infinity")
            sums += features.sum(axis=0); sums_sq += np.square(features, dtype=np.float64).sum(axis=0)
            mins = np.minimum(mins, features.min(axis=0)); maxs = np.maximum(maxs, features.max(axis=0))
            for label in (0, 1):
                mask = labels == label
                if mask.any():
                    by_label_sum[label] += features[mask].sum(axis=0); by_label_n[label] += int(mask.sum())
            fit_mask = np.asarray([row["source1_entity_id"] in fit_ids for row in batch], dtype=bool)
            x_fit.append(features[fit_mask]); y_fit.append(labels[fit_mask])
            x_selection.append(features[~fit_mask]); y_selection.append(labels[~fit_mask])
            total["rows"] += len(batch)
    finally:
        store.close()
    xf, yf = np.concatenate(x_fit), np.concatenate(y_fit)
    xs, ys = np.concatenate(x_selection), np.concatenate(y_selection)
    mean = sums / total["rows"]; variance = np.maximum(0.0, sums_sq / total["rows"] - np.square(mean))
    quality = {
        "source": source, "rows": int(total["rows"]), "fit_rows": int(len(yf)), "selection_rows": int(len(ys)),
        "fit_s1_count": len(fit_ids), "selection_s1_count": len(selection_ids),
        "label_counts": {str(label): int(by_label_n[label]) for label in (0, 1)},
        "finite": True, "constant_features": [FEATURE_NAMES[i] for i, value in enumerate(variance) if value == 0.0],
        "feature_summary": {FEATURE_NAMES[i]: {"min": float(mins[i]), "max": float(maxs[i]), "mean": float(mean[i]),
                                                   "positive_mean": float(by_label_sum[1][i] / by_label_n[1]) if by_label_n[1] else 0.0,
                                                   "negative_mean": float(by_label_sum[0][i] / by_label_n[0]) if by_label_n[0] else 0.0}
                            for i in range(len(FEATURE_NAMES))},
        "matrix_bytes": int(xf.nbytes + xs.nbytes + yf.nbytes + ys.nbytes),
        "target_rows_fetched": store.rows_fetched, "target_queries": store.queries,
        "seconds": time.perf_counter() - start, "peak_rss_mb": peak_rss_mb(),
    }
    return xf, yf, xs, ys, quality


def _metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    return {
        "logloss": float(log_loss(labels, scores, labels=[0, 1])),
        "pr_auc": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "positive_score_quantiles": [float(x) for x in np.quantile(scores[labels == 1], [0.01, 0.1, 0.5, 0.9, 0.99])],
        "negative_score_quantiles": [float(x) for x in np.quantile(scores[labels == 0], [0.01, 0.1, 0.5, 0.9, 0.99])],
    }


def train_source_model(source: str, output_dir: Path = PHASE8,
                       dataset_root: Path = DEFAULT_DATASET_ROOT) -> dict:
    """Train three deterministic source-specific variants and persist the best booster."""
    output_dir.mkdir(parents=True, exist_ok=True)
    xf, yf, xs, ys, quality = build_training_matrices(source, dataset_root, output_dir)
    reports: list[dict] = []; selected: tuple[lgb.Booster, dict] | None = None
    start = time.perf_counter()
    for config in MODEL_CONFIGS:
        params = {
            "objective": "binary", "metric": ["binary_logloss", "auc"], "verbosity": -1,
            "learning_rate": config["learning_rate"], "num_leaves": config["num_leaves"],
            "min_data_in_leaf": config["min_data_in_leaf"], "feature_fraction": config["feature_fraction"],
            "bagging_fraction": config["bagging_fraction"], "bagging_freq": 1, "lambda_l2": config["lambda_l2"],
            "seed": SEED, "feature_fraction_seed": SEED, "bagging_seed": SEED,
            "data_random_seed": SEED, "deterministic": True, "force_col_wise": True, "num_threads": THREADS,
        }
        train = lgb.Dataset(xf, label=yf, feature_name=list(FEATURE_NAMES), free_raw_data=False)
        valid = lgb.Dataset(xs, label=ys, reference=train, feature_name=list(FEATURE_NAMES), free_raw_data=False)
        booster = lgb.train(params, train, num_boost_round=2000, valid_sets=[valid], valid_names=["selection"],
                            callbacks=[lgb.early_stopping(100, verbose=False)])
        scores = booster.predict(xs, num_iteration=booster.best_iteration)
        report = {"configuration": config, "best_iteration": int(booster.best_iteration),
                  "selection": _metrics(ys, scores)}
        reports.append(report)
        if selected is None or (report["selection"]["pr_auc"], -report["selection"]["logloss"], -report["best_iteration"]) > (selected[1]["selection"]["pr_auc"], -selected[1]["selection"]["logloss"], -selected[1]["best_iteration"]):
            selected = booster, report
    assert selected is not None
    booster, chosen = selected
    model_path = output_dir / f"model_{source.lower()}.txt"
    booster.save_model(str(model_path), num_iteration=booster.best_iteration)
    report = {
        "phase": "8", "source": source, "feature_count": len(FEATURE_NAMES), "feature_order": list(FEATURE_NAMES),
        "training_rows": int(len(yf)), "selection_rows": int(len(ys)), "training_positive_rate": float(yf.mean()),
        "selection_positive_rate": float(ys.mean()), "num_threads": THREADS, "seed": SEED,
        "scale_pos_weight": None, "model_search": reports, "selected": chosen,
        "feature_importance_gain": {name: float(value) for name, value in zip(FEATURE_NAMES, booster.feature_importance(importance_type="gain"))},
        "feature_quality": quality, "model_path": str(model_path), "model_bytes": model_path.stat().st_size,
        "runtime_seconds": time.perf_counter() - start, "peak_rss_mb": peak_rss_mb(),
    }
    (output_dir / f"training_report_{source.lower()}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def load_models(output_dir: Path = PHASE8) -> dict[str, lgb.Booster]:
    result = {}
    for source in ("S2", "S3"):
        path = output_dir / f"model_{source.lower()}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"missing trained model: {path}")
        result[source] = lgb.Booster(model_file=str(path))
        if tuple(result[source].feature_name()) != FEATURE_NAMES:
            raise ValueError(f"saved {source} model feature order differs from manifest")
    return result


def validate_tune_split() -> list[str]:
    tune = load_id_file(TUNE_IDS); validation = set(load_id_file(VAL_IDS))
    if len(tune) != 100_000 or len(tune) != len(set(tune)) or set(tune) & validation:
        raise ValueError("tune IDs are not exactly 100K unique and disjoint from validation")
    return tune


def score_tune_pool(output_dir: Path = PHASE8, dataset_root: Path = DEFAULT_DATASET_ROOT,
                    benchmark_s1: int | None = None) -> dict:
    """Score the full frozen tune union, or a deterministic prefix benchmark."""
    models = load_models(output_dir)
    ids = validate_tune_split()
    if benchmark_s1 is not None:
        if benchmark_s1 < 1 or benchmark_s1 > len(ids):
            raise ValueError("benchmark_s1 must be inside the tune ID range")
        ids = ids[:benchmark_s1]
    s1 = load_s1_records(dataset_root, ids)
    require_exhausted = benchmark_s1 is None
    start = time.perf_counter(); counts = union_candidate_counts(
        ids, TUNE_V1_CANDIDATES, TUNE_V1_METADATA, TUNE_ADDRESS,
        require_exhausted=require_exhausted)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "benchmark" if benchmark_s1 is not None else "tune_scores"
    paths = {source: output_dir / f"{suffix}_{source.lower()}.tsv.gz" for source in ("S2", "S3")}
    rows = Counter(); store = TargetStore(TARGET_INDEX)
    try:
        with gzip.open(paths["S2"], "wt", encoding="utf-8", newline="") as s2h, \
             gzip.open(paths["S3"], "wt", encoding="utf-8", newline="") as s3h:
            writers = {"S2": csv.DictWriter(s2h, fieldnames=SCORE_HEADER, delimiter="\t", lineterminator="\n"),
                       "S3": csv.DictWriter(s3h, fieldnames=SCORE_HEADER, delimiter="\t", lineterminator="\n")}
            for writer in writers.values(): writer.writeheader()
            groups = iter_union_groups(
                ids, TUNE_V1_CANDIDATES, TUNE_V1_METADATA, TUNE_ADDRESS,
                require_exhausted=require_exhausted)
            for position, (s1_id, pairs) in enumerate(groups, 1):
                target_ids = sorted(pairs)
                targets = store.get_many(target_ids)
                by_source = {source: [pair for pair in pairs.values() if pair["target_source"] == source] for source in ("S2", "S3")}
                for source, group in by_source.items():
                    if not group: continue
                    matrix = np.vstack([extract_features(s1[s1_id], targets[pair["candidate_entity_id"]], pair, counts[s1_id][source]) for pair in group])
                    scores = models[source].predict(matrix)
                    for pair, score in zip(group, scores):
                        writers[source].writerow({"source1_entity_id": s1_id, "candidate_entity_id": pair["candidate_entity_id"],
                                                "score": f"{float(score):.10f}", "from_v1": pair["from_v1"],
                                                "from_address": pair["from_address"], "address_rank": pair["address_rank"],
                                                "address_score": pair["address_score"]})
                        rows[source] += 1
                if position % 10_000 == 0:
                    print(f"Scored {position:,}/{len(ids):,} tune S1; pairs={sum(rows.values()):,}", flush=True)
    finally:
        store.close()
    elapsed = time.perf_counter() - start
    result = {"s1_count": len(ids), "rows": dict(rows), "total_rows": sum(rows.values()),
              "seconds": elapsed, "rows_per_second": sum(rows.values()) / elapsed if elapsed else 0.0,
              "peak_rss_mb": peak_rss_mb(), "target_rows_fetched": store.rows_fetched, "target_queries": store.queries,
              "files": {source: str(path) for source, path in paths.items()},
              "bytes": {source: path.stat().st_size for source, path in paths.items()}, "benchmark": benchmark_s1 is not None}
    return result


def score_preflight(output_dir: Path = PHASE8, dataset_root: Path = DEFAULT_DATASET_ROOT) -> dict:
    """Benchmark a fixed prefix and project full tune time, memory and disk."""
    result = score_tune_pool(output_dir, dataset_root, SCORE_BENCHMARK_S1)
    projected = {"runtime_seconds": result["seconds"] * 100_000 / result["s1_count"],
                 "disk_bytes": sum(result["bytes"].values()) * 100_000 / result["s1_count"],
                 "peak_rss_mb": result["peak_rss_mb"]}
    if projected["disk_bytes"] > 5 * 1024 ** 3:
        raise RuntimeError(f"projected tune score output exceeds 5 GB: {projected['disk_bytes'] / 1024 ** 3:.2f} GiB")
    if projected["peak_rss_mb"] > 12 * 1024:
        raise RuntimeError(f"projected/observed score RAM exceeds 12 GB: {projected['peak_rss_mb']:.1f} MiB")
    result["projection_full_100k"] = projected
    (output_dir / "tune_scoring_preflight.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def finalize_summary(output_dir: Path = PHASE8) -> dict:
    manifest = json.loads((output_dir / "feature_manifest.json").read_text(encoding="utf-8"))
    training = {source: json.loads((output_dir / f"training_report_{source.lower()}.json").read_text(encoding="utf-8")) for source in ("S2", "S3")}
    selection = {source: json.loads((output_dir / f"model_selection_{source.lower()}.json").read_text(encoding="utf-8")) for source in ("S2", "S3")}
    score_paths = {source: output_dir / f"tune_scores_{source.lower()}.tsv.gz" for source in ("S2", "S3")}
    if not all(path.is_file() for path in score_paths.values()):
        raise FileNotFoundError("full tune score files are required before finalizing Phase 8")
    result = {"phase": "8", "feature_manifest": str(output_dir / "feature_manifest.json"), "feature_count": manifest["feature_count"],
              "models": {source: {"path": training[source]["model_path"], "bytes": training[source]["model_bytes"],
                                  "selected": selection[source]["selected"]} for source in ("S2", "S3")},
              "tune_scores": {source: {"path": str(score_paths[source]), "bytes": score_paths[source].stat().st_size} for source in ("S2", "S3")},
              "frozen_artifact_snapshot_after": frozen_snapshot(frozen_paths()),
              "scope": "No entity-level thresholding, singleton policy, target conflicts, validation, or test inference."}
    (output_dir / "phase8_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result



def feature_cache_path(source: str, output_dir: Path = PHASE8) -> Path:
    if source not in ("S2", "S3"):
        raise ValueError("source must be S2 or S3")
    return output_dir / f"training_features_{source.lower()}.npz"


def build_feature_cache(source: str, output_dir: Path = PHASE8,
                        dataset_root: Path = DEFAULT_DATASET_ROOT) -> dict:
    """Materialize label-free feature matrices from frozen Phase 7 pairs once."""
    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(output_dir)
    before = frozen_snapshot(frozen_paths())
    xf, yf, xs, ys, quality = build_training_matrices(source, dataset_root, output_dir)
    path = feature_cache_path(source, output_dir)
    np.savez_compressed(path, x_fit=xf, y_fit=yf, x_selection=xs, y_selection=ys,
                        feature_names=np.asarray(FEATURE_NAMES, dtype="U"))
    after = frozen_snapshot(frozen_paths())
    if before != after:
        raise RuntimeError("a frozen Phase 4-7 artifact changed during feature generation")
    result = {"source": source, "cache_path": str(path), "cache_bytes": path.stat().st_size,
              "feature_count": len(FEATURE_NAMES), "feature_quality": quality,
              "frozen_artifact_snapshot_before": before, "frozen_artifact_snapshot_after": after}
    (output_dir / f"feature_build_report_{source.lower()}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def load_feature_cache(source: str, output_dir: Path = PHASE8) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = feature_cache_path(source, output_dir)
    if not path.is_file():
        raise FileNotFoundError(f"missing feature cache; run features --source {source} first: {path}")
    with np.load(path, allow_pickle=False) as data:
        names = tuple(data["feature_names"].tolist())
        validate_feature_names(names)
        return (data["x_fit"].astype(np.float32, copy=False), data["y_fit"].astype(np.int8, copy=False),
                data["x_selection"].astype(np.float32, copy=False), data["y_selection"].astype(np.int8, copy=False))


def _lgb_params(config: Mapping[str, float | int | str]) -> dict:
    return {
        "objective": "binary", "metric": ["binary_logloss", "auc"], "verbosity": -1,
        "learning_rate": config["learning_rate"], "num_leaves": config["num_leaves"],
        "min_data_in_leaf": config["min_data_in_leaf"], "feature_fraction": config["feature_fraction"],
        "bagging_fraction": config["bagging_fraction"], "bagging_freq": 1, "lambda_l2": config["lambda_l2"],
        "seed": SEED, "feature_fraction_seed": SEED, "bagging_seed": SEED,
        "data_random_seed": SEED, "deterministic": True, "force_col_wise": True, "num_threads": THREADS,
    }


def model_search(source: str, output_dir: Path = PHASE8) -> dict:
    """Select a source-specific configuration only on the internal S1 selection split."""
    xf, yf, xs, ys = load_feature_cache(source, output_dir)
    start = time.perf_counter(); reports = []
    for config in MODEL_CONFIGS:
        train = lgb.Dataset(xf, label=yf, feature_name=list(FEATURE_NAMES), free_raw_data=False)
        valid = lgb.Dataset(xs, label=ys, reference=train, feature_name=list(FEATURE_NAMES), free_raw_data=False)
        booster = lgb.train(_lgb_params(config), train, num_boost_round=2000, valid_sets=[valid], valid_names=["selection"],
                            callbacks=[lgb.early_stopping(100, verbose=False)])
        scores = booster.predict(xs, num_iteration=booster.best_iteration)
        reports.append({"configuration": config, "best_iteration": int(booster.best_iteration),
                        "selection": _metrics(ys, scores)})
    chosen = max(reports, key=lambda report: (report["selection"]["pr_auc"], -report["selection"]["logloss"], -report["best_iteration"]))
    result = {"phase": "8", "source": source, "selection_rule": "highest PR-AUC, then lower logloss, then fewer iterations",
              "feature_order": list(FEATURE_NAMES), "fit_rows": int(len(yf)), "selection_rows": int(len(ys)),
              "fit_positive_rate": float(yf.mean()), "selection_positive_rate": float(ys.mean()),
              "num_threads": THREADS, "seed": SEED, "scale_pos_weight": None,
              "model_search": reports, "selected": chosen, "runtime_seconds": time.perf_counter() - start,
              "peak_rss_mb": peak_rss_mb()}
    path = output_dir / f"model_selection_{source.lower()}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def retrain_selected_model(source: str, output_dir: Path = PHASE8) -> dict:
    """Retrain the selected source model on all Phase 7 sampled pairs without labels outside Phase 7."""
    selection_path = output_dir / f"model_selection_{source.lower()}.json"
    if not selection_path.is_file():
        raise FileNotFoundError(f"missing model selection report; run search --source {source} first")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = selection["selected"]
    config = next(item for item in MODEL_CONFIGS if item["name"] == selected["configuration"]["name"])
    xf, yf, xs, ys = load_feature_cache(source, output_dir)
    x_all = np.concatenate((xf, xs)); y_all = np.concatenate((yf, ys))
    start = time.perf_counter()
    train = lgb.Dataset(x_all, label=y_all, feature_name=list(FEATURE_NAMES), free_raw_data=False)
    booster = lgb.train(_lgb_params(config), train, num_boost_round=int(selected["best_iteration"]))
    model_path = output_dir / f"model_{source.lower()}.txt"
    booster.save_model(str(model_path))
    report = {"phase": "8", "source": source, "retrained_on": "all Phase 7 sampled pairs",
              "selected_configuration": config, "num_boost_round": int(selected["best_iteration"]),
              "training_rows": int(len(y_all)), "positive_rate": float(y_all.mean()), "feature_order": list(FEATURE_NAMES),
              "model_path": str(model_path), "model_bytes": model_path.stat().st_size,
              "feature_importance_gain": {name: float(value) for name, value in zip(FEATURE_NAMES, booster.feature_importance(importance_type="gain"))},
              "runtime_seconds": time.perf_counter() - start, "peak_rss_mb": peak_rss_mb()}
    path = output_dir / f"training_report_{source.lower()}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("features", "search", "retrain"):
        p = sub.add_parser(name)
        p.add_argument("--source", required=True, choices=("S2", "S3"))
        p.add_argument("--output-dir", type=Path, default=PHASE8)
        p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    for name in ("preflight-score", "score-tune", "finalize"):
        p = sub.add_parser(name)
        p.add_argument("--output-dir", type=Path, default=PHASE8)
        p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    args = parser.parse_args(argv)
    if args.command == "features":
        result = build_feature_cache(args.source, args.output_dir, args.dataset_root)
    elif args.command == "search":
        result = model_search(args.source, args.output_dir)
    elif args.command == "retrain":
        before = frozen_snapshot(frozen_paths())
        result = retrain_selected_model(args.source, args.output_dir)
        after = frozen_snapshot(frozen_paths())
        if before != after:
            raise RuntimeError("a frozen Phase 4-7 artifact changed during final retraining")
        result["frozen_artifact_snapshot_before"] = before
        result["frozen_artifact_snapshot_after"] = after
    elif args.command == "preflight-score":
        before = frozen_snapshot(frozen_paths())
        result = score_preflight(args.output_dir, args.dataset_root)
        after = frozen_snapshot(frozen_paths())
        if before != after:
            raise RuntimeError("a frozen Phase 4-7 artifact changed during score preflight")
        result["frozen_artifact_snapshot_before"] = before
        result["frozen_artifact_snapshot_after"] = after
    elif args.command == "score-tune":
        before = frozen_snapshot(frozen_paths())
        result = score_tune_pool(args.output_dir, args.dataset_root)
        after = frozen_snapshot(frozen_paths())
        if before != after:
            raise RuntimeError("a frozen Phase 4-7 artifact changed during tune scoring")
        result["frozen_artifact_snapshot_before"] = before
        result["frozen_artifact_snapshot_after"] = after
        (args.output_dir / "tune_scoring_report.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        result = finalize_summary(args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
