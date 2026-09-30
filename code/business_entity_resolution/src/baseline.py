#!/usr/bin/env python3
"""Phase 5: deterministic, explainable matching over frozen V1 candidates."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import inspect
import json
import resource
import sqlite3
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from rapidfuzz import fuzz

from blocking import METADATA_HEADER, generate_subset, selected_source1
from diagnostics import CANDIDATE_HEADER, DEFAULT_DATASET_ROOT, evaluate_candidates_only
from normalize import (address_tokens, core_name, extract_numeric_tokens,
                       normalize_address, normalize_country, normalize_name, postal_candidates)
from scoring import (PREDICTION_HEADER, count_link_errors, load_ground_truth, load_id_file,
                     load_predictions, score_entity, score_predictions)

BASE = Path(__file__).resolve().parents[1]
ARTIFACTS = BASE / "artifacts"
BLOCKING = ARTIFACTS / "blocking"
BASELINE = ARTIFACTS / "baseline"
SPLITS = ARTIFACTS / "splits"
TUNE_IDS = SPLITS / "tune_s1_ids.txt"
TRAIN_IDS = SPLITS / "train_s1_ids.txt"
VAL_IDS = SPLITS / "val_s1_ids.txt"
INDEX = BLOCKING / "v1_index.sqlite"
TUNE_CANDIDATES = BASELINE / "tune_candidates.tsv.gz"
TUNE_METADATA = BASELINE / "tune_metadata.tsv.gz"
VAL_CANDIDATES = BLOCKING / "v1_validation_candidates.tsv.gz"
VAL_METADATA = BLOCKING / "v1_validation_metadata.tsv.gz"
SCORES = BASELINE / "tune_pair_scores.tsv.gz"
CONFIG = BASELINE / "heuristic_v1.json"
TUNE_PREDICTIONS = BASELINE / "tune_predictions.tsv"
VAL_PREDICTIONS = BASELINE / "v1_validation_predictions.tsv"
SCORE_HEADER = ["source1_entity_id", "candidate_entity_id", "score", "rule_a", "rule_b", "rule_c"]
ERROR_HEADER = ["category", "source1_entity_id", "target_id", "target_source", "source1_name",
                "target_name", "source1_address", "target_address", "score", "components", "routes"]
ROUTE_NAMES = METADATA_HEADER[3:9]


def heuristic_fingerprint() -> str:
    """Fingerprint only fixed pair scoring and thresholding behavior."""
    source = "\n".join(inspect.getsource(value) for value in
                      (make_record, pair_signals, heuristic_score, rule_flags, choose_predictions))
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_tune_ids(train_path: Path = TRAIN_IDS, val_path: Path = VAL_IDS,
                  output: Path = TUNE_IDS, count: int = 100000, seed: int = 20260925) -> dict:
    train = load_id_file(train_path)
    val = set(load_id_file(val_path))
    if count <= 0 or count > len(train) or set(train) & val:
        raise ValueError("Invalid tune size or train/validation overlap")
    ordered = sorted(train)
    indices = np.random.default_rng(seed).permutation(len(ordered))[:count]
    chosen = sorted(ordered[int(i)] for i in indices)
    if len(set(chosen)) != count or set(chosen) & val or not set(chosen) <= set(train):
        raise AssertionError("Tuning split integrity check failed")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(value + "\n" for value in chosen), encoding="utf-8")
    return {"tune_s1_count": count, "development_s1_count": len(train),
            "validation_overlap": 0, "seed": seed, "tune_ids_sha256": file_sha256(output)}


@dataclass(frozen=True)
class Record:
    entity_id: str
    name: str
    address: str
    country: str
    basic: str
    core: str
    core_tokens: frozenset[str]
    address_basic: str
    address_words: frozenset[str]
    numbers: frozenset[str]
    postals: frozenset[str]


def make_record(entity_id: str, name: str, address: str, country: str,
                basic: str | None = None, core: str | None = None) -> Record:
    country = normalize_country(country)
    raw_postals = postal_candidates(address, country)
    # A US ZIP+4 and its five-digit ZIP share the same coarse postal signal.
    postals = frozenset(code[:5] if country == "us" else code for code in raw_postals)
    postal_digits = {digit for code in raw_postals for digit in extract_numeric_tokens(code)}
    numbers = frozenset(extract_numeric_tokens(address)) - postal_digits
    basic_value = normalize_name(name) if basic is None else basic
    core_value = core_name(name) if core is None else core
    return Record(entity_id, name, address, country, basic_value, core_value,
                  frozenset(core_value.split()), normalize_address(address),
                  frozenset(address_tokens(address, standardized=True)), numbers, postals)


class TargetStore:
    """Fetch target rows by ID in SQLite batches and bound normalized-record memory."""
    def __init__(self, path: Path, cache_size: int = 100000):
        self.db = sqlite3.connect("file:" + str(path) + "?mode=ro", uri=True)
        self.cache: OrderedDict[str, Record] = OrderedDict()
        self.cache_size = cache_size
        self.rows_fetched = 0
        self.queries = 0

    def close(self) -> None:
        self.db.close()

    def get_many(self, ids: list[str]) -> dict[str, Record]:
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate candidate IDs for one S1")
        # Keep cached members of this S1 group recent while adding new targets.
        for value in ids:
            if value in self.cache:
                self.cache.move_to_end(value)
        missing = [value for value in ids if value not in self.cache]
        for start in range(0, len(missing), 800):
            batch = missing[start:start + 800]
            placeholders = ",".join("?" for _ in batch)
            sql = ("SELECT entity_id,name,address,country,basic,core FROM target "
                   f"WHERE entity_id IN ({placeholders})")
            self.queries += 1
            for entity_id, name, address, country, basic, core in self.db.execute(sql, batch):
                self.cache[entity_id] = make_record(entity_id, name, address, country, basic, core)
                self.rows_fetched += 1
            if len(self.cache) > self.cache_size:
                for _ in range(len(self.cache) - self.cache_size):
                    self.cache.popitem(last=False)
        if any(value not in self.cache for value in ids):
            raise ValueError("Candidate ID absent from V1 target index")
        for value in ids:
            self.cache.move_to_end(value)
        return {value: self.cache[value] for value in ids}


def _ratio(a: str, b: str) -> float:
    return fuzz.ratio(a, b) / 100 if a and b else 0.0


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def pair_signals(left: Record, right: Record, metadata: list[str]) -> dict[str, float | bool]:
    if left.country != right.country:
        raise ValueError("Candidate country differs from S1 country")
    exact_basic = bool(left.basic and left.basic == right.basic)
    exact_core = bool(left.core and left.core == right.core)
    number_shared = bool(left.numbers & right.numbers)
    number_conflict = bool(left.numbers and right.numbers and not number_shared)
    postal_shared = bool(left.postals & right.postals)
    postal_conflict = bool(left.postals and right.postals and not postal_shared)
    return {"exact_basic": exact_basic, "exact_core": exact_core,
            "basic_ratio": _ratio(left.basic, right.basic),
            "core_ratio": _ratio(left.core, right.core),
            "token_set_ratio": (fuzz.token_set_ratio(left.core, right.core) / 100
                                if left.core and right.core else 0.0),
            "token_jaccard": _jaccard(left.core_tokens, right.core_tokens),
            "address_ratio": _ratio(left.address_basic, right.address_basic),
            "address_jaccard": _jaccard(left.address_words, right.address_words),
            "number_shared": number_shared, "number_conflict": number_conflict,
            "postal_shared": postal_shared, "postal_conflict": postal_conflict,
            "multiple_routes": int(metadata[10]) >= 2,
            "strong_address_route": metadata[8] == "1"}


def heuristic_score(s: dict[str, float | bool]) -> float:
    """Fixed points-based evidence score; only decision thresholds are tuned."""
    score = 75 * (0.50 * s["core_ratio"] + 0.20 * s["basic_ratio"] +
                   0.15 * s["token_set_ratio"] + 0.15 * s["token_jaccard"])
    score += 8 * s["exact_basic"] + 12 * s["exact_core"]
    score += 5 * s["number_shared"] + 5 * s["postal_shared"]
    score += 5 * (s["address_ratio"] >= 0.80) + 3 * (s["address_jaccard"] >= 0.60)
    score += 2 * s["multiple_routes"]
    score -= 10 * s["number_conflict"] + 12 * s["postal_conflict"]
    if (s["strong_address_route"] and s["number_shared"] and
            s["address_ratio"] >= 0.80 and s["address_jaccard"] >= 0.60):
        score = max(score, 78.0)
    return round(min(100.0, max(0.0, score)), 4)


def rule_flags(s: dict[str, float | bool]) -> tuple[bool, bool, bool]:
    support = (s["address_jaccard"] >= 0.50 or s["number_shared"] or s["postal_shared"])
    no_conflict = not (s["number_conflict"] or s["postal_conflict"])
    return bool(s["exact_basic"]), bool(s["exact_core"]), bool(
        (s["exact_basic"] and no_conflict) or (s["exact_core"] and support))


def load_s1_records(dataset_root: Path, ids: list[str]) -> dict[str, Record]:
    return {entity_id: make_record(entity_id, name, address, country)
            for entity_id, name, address, country in
            selected_source1(dataset_root / "train/train_source1.tsv", set(ids))}


def iter_candidate_groups(candidates_path: Path, metadata_path: Path,
                          expected_ids: list[str]) -> Iterator[tuple[str, list[str], list[list[str]]]]:
    """Read both V1 streams in lockstep; reject omissions, duplicates and misalignment."""
    with gzip.open(candidates_path, "rt", encoding="utf-8", newline="") as candidate_file, \
         gzip.open(metadata_path, "rt", encoding="utf-8", newline="") as metadata_file:
        candidates = csv.reader(candidate_file, delimiter="\t")
        metadata = csv.reader(metadata_file, delimiter="\t")
        if next(candidates, None) != CANDIDATE_HEADER or next(metadata, None) != METADATA_HEADER:
            raise ValueError("Unexpected V1 candidate or metadata header")
        for expected in expected_ids:
            row = next(candidates, None)
            if row is None or len(row) != 2 or row[0] != expected:
                raise ValueError("Candidate S1 coverage/order mismatch")
            target_ids = row[1].split(",") if row[1] else []
            if len(target_ids) != len(set(target_ids)):
                raise ValueError("Duplicate candidate ID")
            if target_ids != sorted(target_ids) or any(
                    not (value.startswith("S2-") or value.startswith("S3-")) for value in target_ids):
                raise ValueError("Candidate IDs must be sorted S2/S3 IDs")
            pair_rows = []
            for target_id in target_ids:
                pair = next(metadata, None)
                if (pair is None or len(pair) != len(METADATA_HEADER) or
                        pair[0] != expected or pair[1] != target_id or
                        pair[2] != target_id[:2]):
                    raise ValueError("Candidate metadata is not aligned with candidate list")
                pair_rows.append(pair)
            yield expected, target_ids, pair_rows
        if next(candidates, None) is not None or next(metadata, None) is not None:
            raise ValueError("Extra candidate or metadata rows")


def score_candidate_file(ids: list[str], dataset_root: Path, candidate_path: Path,
                         metadata_path: Path, index_path: Path, output: Path,
                         max_entities: int | None = None) -> dict:
    """Compute pair signals once; stream a compact tuning cache."""
    started = time.perf_counter()
    selected = ids if max_entities is None else ids[:max_entities]
    # Prefix benchmark still validates only a prefix, not whole-file coverage.
    s1 = load_s1_records(dataset_root, selected)
    store = TargetStore(index_path)
    pairs = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with gzip.open(output, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(SCORE_HEADER)
            iterator = iter_candidate_groups(candidate_path, metadata_path, ids)
            for position, (s1_id, target_ids, rows) in enumerate(iterator, 1):
                if position > len(selected):
                    break
                targets = store.get_many(target_ids)
                for target_id, row in zip(target_ids, rows):
                    signals = pair_signals(s1[s1_id], targets[target_id], row)
                    flags = rule_flags(signals)
                    writer.writerow((s1_id, target_id, f"{heuristic_score(signals):.4f}",
                                     *(int(flag) for flag in flags)))
                pairs += len(target_ids)
                if position % 10000 == 0:
                    print(f"Scored {position:,}/{len(selected):,} S1; pairs={pairs:,}", flush=True)
    finally:
        store.close()
    elapsed = time.perf_counter() - started
    return {"s1_entities": len(selected), "pairs_scored": pairs, "seconds": elapsed,
            "pairs_per_second": pairs / elapsed if elapsed else 0.0,
            "target_rows_fetched": store.rows_fetched, "target_queries": store.queries,
            "peak_rss_mb": peak_rss_mb()}


def iter_score_groups(path: Path, expected_ids: list[str]):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        if next(reader, None) != SCORE_HEADER:
            raise ValueError("Unexpected score-cache columns")
        row = next(reader, None)
        for s1_id in expected_ids:
            pairs = []
            while row is not None and row[0] == s1_id:
                if len(row) != 6:
                    raise ValueError("Malformed score-cache row")
                pairs.append((row[1], float(row[2]), row[3] == "1", row[4] == "1", row[5] == "1"))
                row = next(reader, None)
            if row is not None and row[0] < s1_id:
                raise ValueError("Score cache is not sorted")
            yield s1_id, pairs
        if row is not None:
            raise ValueError("Extra score-cache rows")


def choose_predictions(pairs: list[tuple], config: tuple[str, int, int]) -> set[str]:
    kind, s2_threshold, s3_threshold = config
    if kind in ("A", "B", "C"):
        flag = {"A": 2, "B": 3, "C": 4}[kind]
        return {row[0] for row in pairs if row[flag]}
    return {row[0] for row in pairs if row[1] >= (s2_threshold if row[0].startswith("S2-") else s3_threshold)}


def evaluate_configs(score_file: Path, ids: list[str], truth: dict[str, set[str]],
                     configs: list[tuple[str, int, int]]) -> list[dict]:
    metrics = {config: {"sum_f": 0.0, "tp": 0, "fp": 0, "fn": 0,
                        "correct_singletons": 0, "false_merges": 0, "predicted_links": 0}
               for config in configs}
    for s1_id, pairs in iter_score_groups(score_file, ids):
        actual = truth[s1_id]
        for config in configs:
            prediction = choose_predictions(pairs, config)
            current = metrics[config]
            current["sum_f"] += score_entity(actual, prediction)
            counts = count_link_errors(actual, prediction)
            current["tp"] += counts.tp
            current["fp"] += counts.fp
            current["fn"] += counts.fn
            current["predicted_links"] += len(prediction)
            current["correct_singletons"] += int(not actual and not prediction)
            current["false_merges"] += int(not actual and bool(prediction))
    true_singletons = sum(not links for links in truth.values())
    reports = []
    for kind, s2, s3 in configs:
        m = metrics[(kind, s2, s3)]
        reports.append({"configuration": kind, "threshold_s2": s2 if kind == "D" else None,
                        "threshold_s3": s3 if kind == "D" else None,
                        "macro_f0_5": m["sum_f"] / len(ids),
                        "precision_diag": m["tp"] / (m["tp"] + m["fp"]) if m["tp"] + m["fp"] else 0.0,
                        "recall_diag": m["tp"] / (m["tp"] + m["fn"]) if m["tp"] + m["fn"] else 0.0,
                        "singleton_accuracy": m["correct_singletons"] / true_singletons if true_singletons else 0.0,
                        "singleton_false_merges": m["false_merges"],
                        "predicted_links": m["predicted_links"],
                        "mean_predicted_per_s1": m["predicted_links"] / len(ids)})
    return reports


def best_report(reports: list[dict]) -> dict:
    return max(reports, key=lambda r: (r["macro_f0_5"], r["precision_diag"],
                                       -r["singleton_false_merges"],
                                       r["threshold_s2"] == r["threshold_s3"],
                                       r["threshold_s2"], r["threshold_s3"]))


def write_predictions_from_scores(score_file: Path, ids: list[str], config: tuple[str, int, int],
                                  path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(PREDICTION_HEADER)
        for s1_id, pairs in iter_score_groups(score_file, ids):
            prediction = choose_predictions(pairs, config)
            if not prediction <= {row[0] for row in pairs}:
                raise AssertionError("Prediction outside candidates")
            writer.writerow((s1_id, ",".join(sorted(prediction))))
            counts.append(len(prediction))
    return {"total_predicted_links": sum(counts), "mean_predictions": sum(counts) / len(counts),
            "p95_predictions": float(np.percentile(counts, 95)), "max_predictions": max(counts)}


def run_tune(args) -> dict:
    ids = load_id_file(args.tune_ids)
    if CONFIG.exists():
        raise FileExistsError("Frozen tuning configuration already exists: " + str(CONFIG))
    if len(ids) != 100000 or set(ids) & set(load_id_file(args.val_ids)):
        raise ValueError("Tuning IDs must contain exactly 100,000 nonvalidation IDs")
    truth = load_ground_truth(args.dataset_root / "train/train_ground_truth.tsv", ids)
    benchmark_path = BASELINE / "benchmark_scores.tsv.gz"
    benchmark = score_candidate_file(ids, args.dataset_root, args.tune_candidates,
                                     args.tune_metadata, args.index, benchmark_path, max_entities=1000)
    print("Benchmark: " + json.dumps(benchmark, sort_keys=True), flush=True)
    scored = score_candidate_file(ids, args.dataset_root, args.tune_candidates,
                                  args.tune_metadata, args.index, SCORES)
    coarse_values = list(range(55, 101, 5))
    coarse = evaluate_configs(SCORES, ids, truth, [("D", t, t) for t in coarse_values])
    first = best_report(coarse)["threshold_s2"]
    refinement = evaluate_configs(SCORES, ids, truth,
                                  [("D", t, t) for t in range(max(0, first - 4), min(100, first + 4) + 1)])
    second = best_report(refinement)["threshold_s2"]
    local = sorted(set(max(0, min(100, second + offset)) for offset in (-4, -2, 0, 2, 4)))
    pair_grid = evaluate_configs(SCORES, ids, truth, [("D", a, b) for a in local for b in local])
    selected = best_report(coarse + refinement + pair_grid)
    comparison = evaluate_configs(SCORES, ids, truth, [("A", 0, 0), ("B", 0, 0), ("C", 0, 0),
                                                       ("D", selected["threshold_s2"], selected["threshold_s3"])])
    chosen = ("D", selected["threshold_s2"], selected["threshold_s3"])
    prediction_counts = write_predictions_from_scores(SCORES, ids, chosen, TUNE_PREDICTIONS)
    official = score_predictions(truth, load_predictions(TUNE_PREDICTIONS), ids)
    if abs(official.macro_f0_5 - selected["macro_f0_5"]) > 1e-9:
        raise AssertionError("Tune grid and official scorer disagree")
    candidate_metrics = evaluate_candidates_only(truth, args.tune_candidates)
    report = {"version": "heuristic-v1",
              "baseline_source_sha256_at_tuning": file_sha256(Path(__file__)),
              "heuristic_source_sha256": heuristic_fingerprint(),
              "normalize_source_sha256": file_sha256(Path(__file__).with_name("normalize.py")),
              "threshold_s2": chosen[1], "threshold_s3": chosen[2],
              "tune_ids_sha256": file_sha256(args.tune_ids), "tune_ids_count": len(ids),
              "benchmark": benchmark, "scoring": scored, "candidate_metrics": candidate_metrics,
              "coarse": coarse, "refinement": refinement, "source_grid": pair_grid,
              "comparison": comparison, "selected": selected, "prediction_counts": prediction_counts,
              "rules": "Fixed score and A/B/C rules in baseline.py; thresholds selected on tune only"}
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report



def write_error_sample(score_file: Path, ids: list[str], truth: dict[str, set[str]],
                       predictions: dict[str, set[str]], metadata_path: Path,
                       dataset_root: Path, index_path: Path, output: Path,
                       per_category: int = 20) -> dict[str, int]:
    """Select bounded errors by stable hash; never use them to retune validation."""
    categories = ("false_positive", "model_miss", "blocking_miss",
                  "singleton_false_merge", "linked_predicted_singleton")
    samples: dict[str, list[tuple[int, str, str, str]]] = {name: [] for name in categories}
    counts = {name: 0 for name in categories}

    def add(category: str, s1_id: str, target_id: str, score: str) -> None:
        counts[category] += 1
        priority = int.from_bytes(hashlib.blake2b(
            ("phase5:" + category + ":" + s1_id + ":" + target_id).encode(),
            digest_size=8).digest(), "big")
        items = samples[category]
        items.append((priority, s1_id, target_id, score))
        if len(items) > per_category:
            items.remove(max(items))

    for s1_id, pairs in iter_score_groups(score_file, ids):
        actual, predicted = truth[s1_id], predictions[s1_id]
        candidate_ids = {row[0] for row in pairs}
        if not predicted <= candidate_ids:
            raise AssertionError("Predicted pair outside candidate set")
        interesting = actual | predicted
        pair_scores = {row[0]: f"{row[1]:.4f}" for row in pairs if row[0] in interesting}
        for target_id in predicted - actual:
            add("false_positive", s1_id, target_id, pair_scores[target_id])
            if not actual:
                add("singleton_false_merge", s1_id, target_id, pair_scores[target_id])
        for target_id in actual - predicted:
            if target_id in candidate_ids:
                add("model_miss", s1_id, target_id, pair_scores[target_id])
            else:
                add("blocking_miss", s1_id, target_id, "")
            if not predicted:
                add("linked_predicted_singleton", s1_id, target_id,
                    pair_scores.get(target_id, ""))

    chosen = [(category, s1_id, target_id, score)
              for category in categories for _, s1_id, target_id, score in sorted(samples[category])]
    wanted_pairs = {(s1_id, target_id) for _, s1_id, target_id, score in chosen if score}
    metadata: dict[tuple[str, str], list[str]] = {}
    with gzip.open(metadata_path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        if next(reader, None) != METADATA_HEADER:
            raise ValueError("Unexpected metadata header")
        for row in reader:
            pair = (row[0], row[1])
            if pair in wanted_pairs:
                metadata[pair] = row
    if set(metadata) != wanted_pairs:
        raise ValueError("Sampled candidate metadata missing")
    s1_records = load_s1_records(dataset_root, sorted({item[1] for item in chosen}))
    store = TargetStore(index_path, cache_size=1000)
    try:
        targets = store.get_many(sorted({item[2] for item in chosen}))
    finally:
        store.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(ERROR_HEADER)
        for category, s1_id, target_id, score in chosen:
            left, right = s1_records[s1_id], targets[target_id]
            row = metadata.get((s1_id, target_id))
            signals = pair_signals(left, right, row) if row is not None else {}
            routes = ",".join(name for index, name in enumerate(ROUTE_NAMES)
                              if row is not None and row[index + 3] == "1")
            writer.writerow((category, s1_id, target_id, target_id[:2],
                             left.name, right.name, left.address, right.address,
                             score, json.dumps(signals, sort_keys=True), routes))
    return counts


def run_validate(args) -> dict:
    started = time.perf_counter()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if (config.get("version") != "heuristic-v1" or
            config.get("tune_ids_sha256") != file_sha256(args.tune_ids) or
            config.get("heuristic_source_sha256") != heuristic_fingerprint() or
            config.get("normalize_source_sha256") != file_sha256(Path(__file__).with_name("normalize.py"))):
        raise ValueError("Frozen tuning configuration or tuning IDs changed")
    if VAL_PREDICTIONS.exists():
        raise FileExistsError("Validation predictions already exist; refusing to rerun selection")
    ids = load_id_file(args.val_ids)
    truth = load_ground_truth(args.dataset_root / "train/train_ground_truth.tsv", ids)
    score_file = BASELINE / "validation_pair_scores.tsv.gz"
    scored = score_candidate_file(ids, args.dataset_root, args.val_candidates,
                                  args.val_metadata, args.index, score_file)
    chosen = ("D", int(config["threshold_s2"]), int(config["threshold_s3"]))
    prediction_counts = write_predictions_from_scores(score_file, ids, chosen, VAL_PREDICTIONS)
    predictions = load_predictions(VAL_PREDICTIONS)
    official = score_predictions(truth, predictions, ids)
    error_counts = write_error_sample(score_file, ids, truth, predictions, args.val_metadata,
                                      args.dataset_root, args.index,
                                      ARTIFACTS / "diagnostics/phase5_baseline_errors.tsv")
    if (error_counts["false_positive"] != official.fp or
            error_counts["model_miss"] + error_counts["blocking_miss"] != official.fn):
        raise AssertionError("Error categories do not reconcile with official link counts")
    report = {"error_counts": error_counts, "configuration_sha256": file_sha256(args.config), "validation_s1_count": len(ids),
              "macro_f0_5": official.macro_f0_5, "true_links": official.true_links,
              "predicted_links": official.predicted_links, "tp": official.tp,
              "fp": official.fp, "fn": official.fn,
              "precision_diag": official.micro_precision_diagnostic,
              "recall_diag": official.micro_recall_diagnostic,
              "true_singletons": official.true_singletons,
              "predicted_singletons": sum(not links for links in predictions.values()),
              "correct_singletons": official.correctly_predicted_singletons,
              "singleton_accuracy": official.singleton_accuracy,
              "singleton_false_merges": sum(not truth[key] and bool(predictions[key]) for key in truth),
              "scoring": scored, "prediction_counts": prediction_counts,
              "runtime_seconds": time.perf_counter() - started,
              "peak_rss_mb": peak_rss_mb()}
    (BASELINE / "validation_run.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = [
        ("--dataset-root", Path, DEFAULT_DATASET_ROOT),
        ("--train-ids", Path, TRAIN_IDS), ("--val-ids", Path, VAL_IDS),
        ("--tune-ids", Path, TUNE_IDS), ("--index", Path, INDEX),
        ("--tune-candidates", Path, TUNE_CANDIDATES),
        ("--tune-metadata", Path, TUNE_METADATA),
        ("--val-candidates", Path, VAL_CANDIDATES),
        ("--val-metadata", Path, VAL_METADATA), ("--config", Path, CONFIG),
    ]
    for option, value_type, default in common:
        parser.add_argument(option, type=value_type, default=default)
    for name in ("make-tune-ids", "generate-tune-candidates", "tune", "validate"):
        command_parser = sub.add_parser(name)
        for option, value_type, _ in common:
            command_parser.add_argument(option, type=value_type, default=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.command == "make-tune-ids":
            report = make_tune_ids(args.train_ids, args.val_ids, args.tune_ids)
        elif args.command == "generate-tune-candidates":
            report = generate_subset(args.dataset_root, args.tune_ids, args.index,
                                     args.tune_candidates, args.tune_metadata)
            truth = load_ground_truth(args.dataset_root / "train/train_ground_truth.tsv",
                                      load_id_file(args.tune_ids))
            report["candidate_metrics"] = evaluate_candidates_only(truth, args.tune_candidates)
            (BASELINE / "tune_candidate_report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        elif args.command == "tune":
            report = run_tune(args)
        else:
            report = run_validate(args)
    except (OSError, ValueError, csv.Error) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
