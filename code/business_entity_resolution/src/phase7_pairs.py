#!/usr/bin/env python3
"""Build Phase 7 supervised pair data from frozen V1 + Address K10 retrieval.

Candidate generation is label-blind. Ground truth is read only by the
``build-pairs`` stage, where unretrieved links are recorded as blocking misses
and never emitted as negative examples.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import resource
import shutil
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from blocking import METADATA_HEADER, generate_subset, selected_source1
from diagnostics import DEFAULT_DATASET_ROOT, CANDIDATE_HEADER
from phase6d_retrieval import address_retrieve
from scoring import load_ground_truth, load_id_file

BASE = Path(__file__).resolve().parents[1]
ARTIFACTS = BASE / "artifacts"
OUT = ARTIFACTS / "model_data/phase7"
TRAIN_IDS = ARTIFACTS / "splits/train_s1_ids.txt"
TUNE_IDS = ARTIFACTS / "splits/tune_s1_ids.txt"
VAL_IDS = ARTIFACTS / "splits/val_s1_ids.txt"
V1_INDEX = ARTIFACTS / "blocking/v1_index.sqlite"
ADDRESS_INDEX = ARTIFACTS / "retrieval_diagnosis/phase6d/address_postings.sqlite"
GROUND_TRUTH = DEFAULT_DATASET_ROOT / "train/train_ground_truth.tsv"
SEED = 20260925
SUBSET_SIZE = 100_000
MAX_NEGATIVES_PER_S1 = 12
PAIR_COLUMNS = [
    "source1_entity_id", "candidate_entity_id", "target_source", "label",
    "from_v1", "from_address", "address_rank", "address_score", "negative_reason",
]
MISS_COLUMNS = ["source1_entity_id", "true_target_id", "target_source"]


def select_training_subset(
    train_ids: Iterable[str], tune_ids: Iterable[str], val_ids: Iterable[str],
    size: int = SUBSET_SIZE, seed: int = SEED,
) -> list[str]:
    """Select a deterministic, label-independent subset from development IDs."""
    train = list(train_ids)
    tune, val = set(tune_ids), set(val_ids)
    if len(train) != len(set(train)):
        raise ValueError("train S1 ID list contains duplicates")
    if len(tune) != len(set(tune)) or len(val) != len(set(val)):
        raise ValueError("tune/validation S1 ID list contains duplicates")
    if tune & val:
        raise ValueError("tune and validation S1 ID lists overlap")
    if tune - set(train):
        raise ValueError("tune IDs are not all members of the train development IDs")
    eligible = sorted(set(train) - tune - val)
    if size < 1 or len(eligible) < size:
        raise ValueError(f"need {size} eligible train IDs; found {len(eligible)}")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.asarray(eligible, dtype=object), size=size, replace=False)
    result = sorted(map(str, chosen.tolist()))
    if len(result) != size or len(result) != len(set(result)):
        raise AssertionError("selected subset is not unique or has wrong size")
    if set(result) & tune or set(result) & val:
        raise AssertionError("selected subset overlaps tune/validation")
    return result


def sha256_ids(ids: Iterable[str]) -> str:
    """Hash the canonical sorted one-ID-per-line representation."""
    payload = "".join(value + "\n" for value in sorted(ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prepare_subset(
    train_ids_path: Path = TRAIN_IDS, tune_ids_path: Path = TUNE_IDS,
    val_ids_path: Path = VAL_IDS, output_dir: Path = OUT,
    size: int = SUBSET_SIZE, seed: int = SEED,
) -> dict:
    train, tune, val = map(load_id_file, (train_ids_path, tune_ids_path, val_ids_path))
    selected = select_training_subset(train, tune, val, size, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    id_path = output_dir / "train_subset_ids.txt"
    id_path.write_text("".join(entity_id + "\n" for entity_id in selected), encoding="utf-8")
    digest = sha256_ids(selected)
    (output_dir / "train_subset_ids.sha256").write_text(f"{digest}  train_subset_ids.txt\n", encoding="ascii")
    report = {
        "seed": seed, "requested_size": size, "selected_size": len(selected),
        "train_universe_size": len(train), "tune_size": len(tune), "validation_size": len(val),
        "train_ids_unique": len(train) == len(set(train)),
        "tune_overlap": len(set(selected) & set(tune)),
        "validation_overlap": len(set(selected) & set(val)),
        "sha256_sorted_ids_with_trailing_newlines": digest,
        "ids_path": str(id_path),
    }
    if report["tune_overlap"] or report["validation_overlap"]:
        raise AssertionError("training subset overlaps a reserved split")
    (output_dir / "phase7_subset_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def estimate_generation_resources(output_dir: Path = OUT) -> dict:
    """Project the authorized 100K retrieval run from measured V1/6E reports."""
    v1_report_path = ARTIFACTS / "blocking/v1_report.json"
    address_report_path = ARTIFACTS / "retrieval_diagnosis/phase6e/address_top10_100k_retrieval_report.json"
    if not v1_report_path.is_file() or not address_report_path.is_file():
        raise FileNotFoundError("cannot estimate safely without frozen V1 and Phase 6E timing reports")
    v1 = json.loads(v1_report_path.read_text(encoding="utf-8"))
    address = json.loads(address_report_path.read_text(encoding="utf-8"))
    v1_seconds = float(v1["candidate_seconds"]) * SUBSET_SIZE / int(v1["s1_count"])
    # The stored Phase 6E report is a 100K-S1 run of this exact K10 routine.
    address_seconds = float(address["runtime_seconds"]) * SUBSET_SIZE / 100_000
    predicted = v1_seconds + address_seconds
    v1_candidate_bytes = (ARTIFACTS / "blocking/v1_validation_candidates.tsv.gz").stat().st_size
    v1_metadata_bytes = (ARTIFACTS / "blocking/v1_validation_metadata.tsv.gz").stat().st_size
    address_bytes = int(address["bytes"])
    projected_bytes = int((v1_candidate_bytes + v1_metadata_bytes) * SUBSET_SIZE / int(v1["s1_count"]) + address_bytes)
    free_bytes = shutil.disk_usage(output_dir.parent if output_dir.parent.exists() else BASE).free
    return {
        "basis": {"v1_validation_seconds": v1["candidate_seconds"], "v1_validation_s1": v1["s1_count"],
                  "phase6e_address_k10_seconds": address["runtime_seconds"], "phase6e_address_k10_s1": 100000},
        "projected_v1_seconds": v1_seconds, "projected_address_k10_seconds": address_seconds,
        "projected_candidate_generation_seconds": predicted,
        "projected_candidate_generation_minutes": predicted / 60,
        "projected_intermediate_bytes": projected_bytes,
        "available_disk_bytes": free_bytes,
        "projected_peak_rss_mb": max(float(v1["peak_rss_mb"]), float(address["peak_rss_mb"])),
        "wall_time_limit_seconds": 3600,
    }


def generate_candidates(
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    subset_ids_path: Path = OUT / "train_subset_ids.txt",
    output_dir: Path = OUT,
    v1_index: Path = V1_INDEX,
    address_index: Path = ADDRESS_INDEX,
) -> dict:
    """Generate frozen V1 plus unchanged Phase 6D Address K10 pairs."""
    ids = load_id_file(subset_ids_path)
    if len(ids) != SUBSET_SIZE or len(ids) != len(set(ids)):
        raise ValueError(f"expected {SUBSET_SIZE:,} unique subset IDs; found {len(ids):,}")
    for required in (v1_index, address_index):
        if not required.is_file():
            raise FileNotFoundError(f"required frozen retrieval index missing: {required}")
    preflight = estimate_generation_resources(output_dir)
    if preflight["projected_candidate_generation_seconds"] > preflight["wall_time_limit_seconds"]:
        raise RuntimeError(f"projected retrieval exceeds one-hour limit: {preflight['projected_candidate_generation_minutes']:.1f} minutes")
    if preflight["available_disk_bytes"] < preflight["projected_intermediate_bytes"] * 2:
        raise RuntimeError("insufficient disk headroom for projected candidate artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    v1_candidates = output_dir / "v1_candidates.tsv.gz"
    v1_metadata = output_dir / "v1_metadata.tsv.gz"
    address_candidates = output_dir / "address_k10_candidates.tsv.gz"
    started = time.perf_counter()
    v1_report = generate_subset(
        dataset_root, subset_ids_path, v1_index, v1_candidates, v1_metadata,
    )
    selected = set(ids)
    s1_rows = {
        entity_id: {"business_name": name, "business_address": address, "country": country}
        for entity_id, name, address, country in selected_source1(
            dataset_root / "train/train_source1.tsv", selected,
        )
    }
    if set(s1_rows) != selected:
        raise ValueError(f"selected subset is missing {len(selected - s1_rows.keys())} Source 1 rows")
    from phase6d_retrieval import rss_mb
    address_report = address_retrieve(ids, s1_rows, v1_index, address_index, address_candidates, top_k=10)
    report = {
        "configuration": "frozen Phase 4 V1 union Phase 6E Address K10",
        "s1_count": len(ids), "candidate_generation_label_blind": True,
        "preflight_estimate": preflight,
        "v1": {**v1_report, "candidate_bytes": v1_candidates.stat().st_size, "metadata_bytes": v1_metadata.stat().st_size},
        "address_k10": address_report,
        "total_candidate_generation_seconds": time.perf_counter() - started,
        "peak_rss_mb_process_observed": rss_mb(),
        "files": {"v1_candidates": str(v1_candidates), "v1_metadata": str(v1_metadata), "address_candidates": str(address_candidates)},
    }
    (output_dir / "phase7_candidate_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _open_gzip_dict(path: Path) -> tuple[object, csv.DictReader]:
    handle = gzip.open(path, "rt", encoding="utf-8", newline="")
    return handle, csv.DictReader(handle, delimiter="\t")


def _next_group(reader: Iterator[dict[str, str]], current: dict[str, str] | None, s1_id: str) -> tuple[list[dict[str, str]], dict[str, str] | None]:
    rows = []
    while current is not None and current["source1_entity_id"] == s1_id:
        rows.append(current)
        current = next(reader, None)
    if current is not None and current["source1_entity_id"] < s1_id:
        raise ValueError("candidate metadata is not sorted by source1_entity_id")
    return rows, current


def _negative_reason(pair: dict) -> tuple[str, int, int]:
    """Return human-readable sampling category, priority (lower is harder), and address rank."""
    rank = int(pair.get("address_rank") or 0)
    v1 = bool(pair["from_v1"])
    address = bool(pair["from_address"])
    routes = pair.get("routes", {})
    if address and rank <= 3:
        return ("address_top_1_3", 0, rank)
    if routes.get("exact_name"):
        return ("v1_exact_name", 1, rank or 10_000)
    if routes.get("core_name"):
        return ("v1_core_name", 2, rank or 10_000)
    if v1 and address:
        return ("v1_and_address", 3, rank or 10_000)
    if address and rank <= 10:
        return ("address_top_4_10", 4, rank)
    if routes.get("address_numeric_route") or routes.get("postal_route") or routes.get("strong_address_route"):
        return ("v1_address_evidence", 5, rank or 10_000)
    if routes.get("rare_name_token"):
        return ("v1_name_token", 6, rank or 10_000)
    if v1:
        return ("v1_other", 7, rank or 10_000)
    return ("address_ranked", 8, rank or 10_000)


def _stable_pair_hash(seed: int, s1_id: str, target_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{s1_id}\0{target_id}".encode("utf-8")).hexdigest()


def merge_candidate_routes(v1_rows: Iterable[dict[str, str]], address_rows: Iterable[dict[str, str]]) -> dict[str, dict]:
    """Union per-S1 routes while preserving V1 and Address K10 metadata."""
    pair_map: dict[str, dict] = {}
    for row in v1_rows:
        target, source = row["candidate_entity_id"], row["target_source"]
        if target in pair_map:
            raise ValueError(f"duplicate V1 candidate pair: {target}")
        if (target.startswith("S2-") and source != "S2") or (target.startswith("S3-") and source != "S3") or not target.startswith(("S2-", "S3-")):
            raise ValueError(f"V1 target/source prefix mismatch: {target}/{source}")
        pair_map[target] = {
            "candidate_entity_id": target, "target_source": source,
            "from_v1": 1, "from_address": 0, "address_rank": "", "address_score": "",
            "routes": {route: row[route] == "1" for route in METADATA_HEADER[3:-2]},
        }
    v1_ids = set(pair_map)
    for row in address_rows:
        target, source = row["candidate_entity_id"], row["target_source"]
        if (target.startswith("S2-") and source != "S2") or (target.startswith("S3-") and source != "S3") or not target.startswith(("S2-", "S3-")):
            raise ValueError(f"address target/source prefix mismatch: {target}/{source}")
        pair = pair_map.setdefault(target, {
            "candidate_entity_id": target, "target_source": source,
            "from_v1": 0, "from_address": 0, "address_rank": "", "address_score": "", "routes": {},
        })
        if pair["target_source"] != source:
            raise ValueError(f"V1/address source disagreement for {target}")
        if pair["from_address"]:
            raise ValueError(f"duplicate Address K10 candidate pair: {target}")
        pair.update(from_address=1, address_rank=row["rank"], address_score=row["score"])
    if not v1_ids.issubset(pair_map):
        raise AssertionError("V1 candidates must be preserved in the V1 union Address K10 set")
    return pair_map


def select_pair_rows(
    s1_ids: Iterable[str], candidates_by_s1: dict[str, list[dict]],
    truth: dict[str, set[str]], seed: int = SEED,
    max_negatives: int = MAX_NEGATIVES_PER_S1,
) -> tuple[dict[str, list[dict]], dict[str, list[dict]], dict[str, set[str]]]:
    """Keep all retrieved positives and deterministically sample hard negatives."""
    if max_negatives < 1:
        raise ValueError("max_negatives must be positive")
    selected = {"S2": [], "S3": []}
    misses = {"S2": [], "S3": []}
    missed_by_s1: dict[str, set[str]] = {}
    for s1_id in s1_ids:
        candidate_rows = candidates_by_s1.get(s1_id, [])
        seen: set[str] = set()
        positives, negatives = [], []
        for raw in candidate_rows:
            pair = dict(raw)
            target = pair["candidate_entity_id"]
            source = pair["target_source"]
            expected_source = "S2" if target.startswith("S2-") else "S3" if target.startswith("S3-") else ""
            if source not in ("S2", "S3") or source != expected_source:
                raise ValueError(f"target source/prefix mismatch: {target!r}/{source!r}")
            if target in seen:
                raise ValueError(f"duplicate candidate pair: {s1_id}/{target}")
            seen.add(target)
            pair["source1_entity_id"] = s1_id
            pair["label"] = int(target in truth[s1_id])
            pair["negative_reason"] = ""
            if pair["label"]:
                positives.append(pair)
            else:
                reason, priority, rank = _negative_reason(pair)
                pair["negative_reason"] = reason
                pair["_priority"] = priority
                pair["_rank"] = rank
                pair["_stable"] = _stable_pair_hash(seed, s1_id, target)
                negatives.append(pair)

        # Reserve one deterministic negative from the least-supported end of the
        # retrieved set when possible, then fill remaining slots with hard cases.
        easy_pool = [p for p in negatives if p["_priority"] >= 6]
        easy = min(easy_pool, key=lambda p: p["_stable"]) if easy_pool else None
        if easy is not None:
            easy["negative_reason"] = "easy_random"
        ordered = sorted(
            (p for p in negatives if p is not easy),
            key=lambda p: (p["_priority"], p["_rank"], p["_stable"], p["candidate_entity_id"]),
        )
        keep_count = max_negatives - int(easy is not None)
        kept = ordered[:keep_count] + ([easy] if easy is not None else [])
        for pair in positives + kept:
            pair.pop("_priority", None); pair.pop("_rank", None); pair.pop("_stable", None)
            selected[pair["target_source"]].append(pair)
        missed = set(truth[s1_id]) - seen
        if missed:
            missed_by_s1[s1_id] = missed
            for target in sorted(missed):
                source = "S2" if target.startswith("S2-") else "S3"
                misses[source].append({"source1_entity_id": s1_id, "true_target_id": target, "target_source": source})
    for source in selected:
        selected[source].sort(key=lambda p: (p["source1_entity_id"], p["candidate_entity_id"]))
        misses[source].sort(key=lambda p: (p["source1_entity_id"], p["true_target_id"]))
    return selected, misses, missed_by_s1


def _percentile(values: list[int], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p / 100
    lo = int(pos); hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _source_metrics(source: str, s1_ids: list[str], truth: dict[str, set[str]], candidates: dict[str, set[str]], selected: list[dict], misses: list[dict]) -> dict:
    true_by_s1 = {s1: {x for x in truth[s1] if x.startswith(source + "-")} for s1 in s1_ids}
    candidate_by_s1 = {s1: {x for x in candidates.get(s1, set()) if x.startswith(source + "-")} for s1 in s1_ids}
    retrieved_positive = sum(len(true_by_s1[s1] & candidate_by_s1[s1]) for s1 in s1_ids)
    total_positive = sum(map(len, true_by_s1.values()))
    negative_counts = [max(0, len(candidate_by_s1[s1]) - len(true_by_s1[s1] & candidate_by_s1[s1])) for s1 in s1_ids]
    positives_per_s1 = [len(true_by_s1[s1] & candidate_by_s1[s1]) for s1 in s1_ids]
    retained_negatives = sum(int(p["label"] == 0) for p in selected)
    total_negatives = sum(map(len, candidate_by_s1.values())) - retrieved_positive
    reasons = Counter(p["negative_reason"] for p in selected if p["label"] == 0)
    selected_negatives_per_s1 = Counter(p["source1_entity_id"] for p in selected if p["label"] == 0)
    per_s1_selected = [selected_negatives_per_s1[s1] for s1 in s1_ids]
    return {
        "s1_count": len(s1_ids),
        "candidate_pair_count_before_sampling": sum(map(len, candidate_by_s1.values())),
        "positive_pairs_retrieved": retrieved_positive,
        "positive_links_missed_by_blocking": len(misses),
        "total_true_links": total_positive,
        "candidate_recall": retrieved_positive / total_positive if total_positive else 0.0,
        "total_negatives_before_sampling": total_negatives,
        "negatives_retained": retained_negatives,
        "positive_negative_ratio": (retrieved_positive / retained_negatives) if retained_negatives else None,
        "negative_categories_retained": dict(sorted(reasons.items())),
        "negatives_per_s1_before_sampling": {"mean": sum(negative_counts)/len(negative_counts) if negative_counts else 0.0, "median": _percentile(negative_counts, 50), "p95": _percentile(negative_counts, 95)},
        "negatives_retained_per_s1": {"mean": sum(per_s1_selected)/len(per_s1_selected) if per_s1_selected else 0.0, "median": _percentile(per_s1_selected, 50), "p95": _percentile(per_s1_selected, 95)},
        "s1_with_zero_retrieved_positives": sum(v == 0 for v in positives_per_s1),
        "s1_with_multiple_retrieved_positives": sum(v > 1 for v in positives_per_s1),
    }


def build_pair_datasets(
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    subset_ids_path: Path = OUT / "train_subset_ids.txt",
    output_dir: Path = OUT,
    seed: int = SEED,
    max_negatives: int = MAX_NEGATIVES_PER_S1,
) -> dict:
    """Stream V1/address candidates per S1, label, sample and write pair files."""
    started = time.perf_counter()
    ids = load_id_file(subset_ids_path)
    if len(ids) != SUBSET_SIZE or ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError("subset file must contain exactly 100,000 sorted unique S1 IDs")
    v1_path = output_dir / "v1_candidates.tsv.gz"
    meta_path = output_dir / "v1_metadata.tsv.gz"
    address_path = output_dir / "address_k10_candidates.tsv.gz"
    if not all(p.is_file() for p in (v1_path, meta_path, address_path)):
        raise FileNotFoundError("run generate-candidates before build-pairs")
    truth = load_ground_truth(dataset_root / "train/train_ground_truth.tsv", set(ids))
    total_true = Counter()
    for links in truth.values():
        total_true["S2"] += sum(x.startswith("S2-") for x in links)
        total_true["S3"] += sum(x.startswith("S3-") for x in links)

    output_dir.mkdir(parents=True, exist_ok=True)
    pair_paths = {s: output_dir / f"train_pairs_{s.lower()}.tsv.gz" for s in ("S2", "S3")}
    miss_paths = {s: output_dir / f"blocking_misses_{s.lower()}.tsv.gz" for s in ("S2", "S3")}
    total_candidate = Counter(); retrieved_positive = Counter(); retained_rows = Counter()
    retained_negative = Counter(); zero_positive = Counter(); multiple_positive = Counter()
    before_negative_per_s1 = {s: [] for s in ("S2", "S3")}
    retained_negative_per_s1 = {s: [] for s in ("S2", "S3")}
    miss_counts = Counter(); categories = Counter(); address_count_by_source = Counter()
    v1_pair_count = address_pair_count = union_pair_count = 0
    miss_samples = {"S2": [], "S3": []}

    with ExitStack() as stack:
        candidate_handle = stack.enter_context(gzip.open(v1_path, "rt", encoding="utf-8", newline=""))
        candidates_reader = csv.DictReader(candidate_handle, delimiter="\t")
        if candidates_reader.fieldnames != CANDIDATE_HEADER:
            raise ValueError(f"unexpected V1 candidate header: {candidates_reader.fieldnames!r}")
        meta_handle, meta_reader = _open_gzip_dict(meta_path); stack.callback(meta_handle.close)
        if meta_reader.fieldnames != METADATA_HEADER:
            raise ValueError(f"unexpected V1 metadata header: {meta_reader.fieldnames!r}")
        address_handle, address_reader = _open_gzip_dict(address_path); stack.callback(address_handle.close)
        expected_address_header = ["source1_entity_id", "candidate_entity_id", "target_source", "route", "score", "rank"]
        if address_reader.fieldnames != expected_address_header:
            raise ValueError(f"unexpected address candidate header: {address_reader.fieldnames!r}")

        pair_writers = {}; miss_writers = {}
        for source in ("S2", "S3"):
            pair_file = stack.enter_context(gzip.open(pair_paths[source], "wt", encoding="utf-8", newline=""))
            pair_writers[source] = csv.DictWriter(pair_file, fieldnames=PAIR_COLUMNS, delimiter="\t", lineterminator="\n", extrasaction="ignore")
            pair_writers[source].writeheader()
            miss_file = stack.enter_context(gzip.open(miss_paths[source], "wt", encoding="utf-8", newline=""))
            miss_writers[source] = csv.DictWriter(miss_file, fieldnames=MISS_COLUMNS, delimiter="\t", lineterminator="\n")
            miss_writers[source].writeheader()

        meta_current = next(meta_reader, None)
        address_current = next(address_reader, None)
        for position, s1_id in enumerate(ids, 1):
            candidate_row = next(candidates_reader, None)
            if candidate_row is None or candidate_row["source1_entity_id"] != s1_id:
                raise ValueError(f"candidate output misses/reorders S1 row {s1_id}")
            v1_targets = set(candidate_row["candidate_entity_ids"].split(",")) if candidate_row["candidate_entity_ids"] else set()
            meta_rows, meta_current = _next_group(meta_reader, meta_current, s1_id)
            meta_by_target = {}
            for row in meta_rows:
                target = row["candidate_entity_id"]
                if target in meta_by_target:
                    raise ValueError(f"duplicate V1 metadata pair: {s1_id}/{target}")
                meta_by_target[target] = row
            if set(meta_by_target) != v1_targets:
                raise ValueError(f"V1 candidate and metadata pairs disagree for {s1_id}")
            v1_pair_count += len(meta_by_target)

            address_rows, address_current = _next_group(address_reader, address_current, s1_id)
            pair_map = merge_candidate_routes(meta_rows, address_rows)
            union_pair_count += len(pair_map)

            subset, misses, _ = select_pair_rows(
                [s1_id], {s1_id: list(pair_map.values())}, {s1_id: truth[s1_id]}, seed, max_negatives,
            )
            for source in ("S2", "S3"):
                source_pairs = [p for p in pair_map.values() if p["target_source"] == source]
                positives_here = sum(p["candidate_entity_id"] in truth[s1_id] for p in source_pairs)
                negatives_here = len(source_pairs) - positives_here
                rows = subset[source]
                negatives_kept = sum(row["label"] == 0 for row in rows)
                total_candidate[source] += len(source_pairs)
                retrieved_positive[source] += positives_here
                retained_rows[source] += len(rows); retained_negative[source] += negatives_kept
                before_negative_per_s1[source].append(negatives_here)
                retained_negative_per_s1[source].append(negatives_kept)
                zero_positive[source] += positives_here == 0
                multiple_positive[source] += positives_here > 1
                for pair in rows:
                    pair_writers[source].writerow(pair)
                    if pair["label"] == 0:
                        categories[source, pair["negative_reason"]] += 1
                for miss in misses[source]:
                    miss_writers[source].writerow(miss)
                    miss_counts[source] += 1
                    if len(miss_samples[source]) < 5:
                        miss_samples[source].append(miss)
            if position % 10_000 == 0:
                print(f"Labeled and sampled {position:,}/{len(ids):,} S1; candidate pairs={union_pair_count:,}", flush=True)
        if next(candidates_reader, None) is not None or meta_current is not None or address_current is not None:
            raise ValueError("candidate artifacts contain extra S1 rows")

    file_sizes = {}
    for path in [*pair_paths.values(), *miss_paths.values()]:
        file_sizes[str(path)] = path.stat().st_size
    sources = {}
    for source in ("S2", "S3"):
        true_links = total_true[source]
        neg_before = total_candidate[source] - retrieved_positive[source]
        before = before_negative_per_s1[source]
        retained = retained_negative_per_s1[source]
        source_categories = {reason: count for (src, reason), count in categories.items() if src == source}
        sources[source] = {
            "s1_count": len(ids), "candidate_pair_count_before_sampling": total_candidate[source],
            "positive_pairs_retrieved": retrieved_positive[source],
            "positive_links_missed_by_blocking": miss_counts[source], "total_true_links": true_links,
            "candidate_recall": retrieved_positive[source] / true_links if true_links else 0.0,
            "total_negatives_before_sampling": neg_before, "negatives_retained": retained_negative[source],
            "positive_negative_ratio": retrieved_positive[source] / retained_negative[source] if retained_negative[source] else None,
            "negative_categories_retained": dict(sorted(source_categories.items())),
            "negatives_per_s1_before_sampling": {"mean": sum(before)/len(before) if before else 0.0, "median": _percentile(before, 50), "p95": _percentile(before, 95)},
            "negatives_retained_per_s1": {"mean": sum(retained)/len(retained) if retained else 0.0, "median": _percentile(retained, 50), "p95": _percentile(retained, 95)},
            "s1_with_zero_retrieved_positives": zero_positive[source],
            "s1_with_multiple_retrieved_positives": multiple_positive[source],
            "output_rows": retained_rows[source], "output_file": str(pair_paths[source]), "blocking_miss_file": str(miss_paths[source]),
        }
    total_links = sum(total_true.values()); total_retrieved = sum(retrieved_positive.values())
    report = {
        "phase": "7 supervised pair dataset; no model features/training/validation",
        "seed": seed, "max_negatives_per_s1": max_negatives, "s1_count": len(ids),
        "candidate_generation_label_blind": True,
        "candidate_pairs": {"v1": v1_pair_count, "address_k10_raw": address_pair_count,
                            "v1_union_address_k10": union_pair_count, "v1_union_growth": union_pair_count-v1_pair_count,
                            "address_pairs_by_source": dict(address_count_by_source)},
        "ground_truth_links_in_subset": total_links, "retrieved_positive_links": total_retrieved,
        "blocking_misses": sum(miss_counts.values()), "candidate_recall": total_retrieved / total_links if total_links else 0.0,
        "sources": sources,
        "negative_categories_retained": {s: sources[s]["negative_categories_retained"] for s in ("S2", "S3")},
        "files_bytes": file_sizes, "sampling_runtime_seconds": time.perf_counter() - started,
        "sampling_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "blocking_miss_examples_first_five_each_source": miss_samples,
        "artifacts": {"candidate_report": str(output_dir / "phase7_candidate_report.json"),
                      "sampling_report": str(output_dir / "phase7_sampling_report.json"),
                      "subset_ids": str(output_dir / "train_subset_ids.txt"),
                      "subset_sha256": str(output_dir / "train_subset_ids.sha256")},
    }
    prior_path = output_dir / "phase7_candidate_report.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8")) if prior_path.exists() else {}
    prior.update({"candidate_pairs": report["candidate_pairs"], "labeled_candidate_recall": report["candidate_recall"],
                  "retrieved_positive_links": total_retrieved, "blocking_misses": report["blocking_misses"]})
    prior_path.write_text(json.dumps(prior, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "phase7_sampling_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare-subset")
    p.add_argument("--train-ids", type=Path, default=TRAIN_IDS); p.add_argument("--tune-ids", type=Path, default=TUNE_IDS)
    p.add_argument("--val-ids", type=Path, default=VAL_IDS); p.add_argument("--output-dir", type=Path, default=OUT)
    p.add_argument("--size", type=int, default=SUBSET_SIZE); p.add_argument("--seed", type=int, default=SEED)
    p = sub.add_parser("generate-candidates")
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT); p.add_argument("--subset-ids", type=Path, default=OUT / "train_subset_ids.txt")
    p.add_argument("--output-dir", type=Path, default=OUT); p.add_argument("--v1-index", type=Path, default=V1_INDEX)
    p.add_argument("--address-index", type=Path, default=ADDRESS_INDEX)
    p = sub.add_parser("build-pairs")
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT); p.add_argument("--subset-ids", type=Path, default=OUT / "train_subset_ids.txt")
    p.add_argument("--output-dir", type=Path, default=OUT); p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--max-negatives", type=int, default=MAX_NEGATIVES_PER_S1)
    args = parser.parse_args(argv)
    if args.command == "prepare-subset":
        result = prepare_subset(args.train_ids, args.tune_ids, args.val_ids, args.output_dir, args.size, args.seed)
    elif args.command == "generate-candidates":
        result = generate_candidates(args.dataset_root, args.subset_ids, args.output_dir, args.v1_index, args.address_index)
    else:
        result = build_pair_datasets(args.dataset_root, args.subset_ids, args.output_dir, args.seed, args.max_negatives)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
