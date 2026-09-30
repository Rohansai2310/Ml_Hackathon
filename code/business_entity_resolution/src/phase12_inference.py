#!/usr/bin/env python3
"""Phase 12: frozen, resumable inference for the complete test set.

This module deliberately has no command that tunes or trains.  Its final
candidate set is written before scoring and is the only candidate set that the
models can consume.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import resource
import shutil
import sqlite3
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import numpy as np

from baseline import TargetStore, make_record
from blocking import (BASIC, CORE, NUMBER, POSTAL, ROUTES, TOKEN, Index, Limits,
                      TARGET_HEADER, ensure_frequency_table, key, retrieve)
from diagnostics import CANDIDATE_HEADER, DEFAULT_DATASET_ROOT
from normalize import (address_tokens, core_name, core_name_tokens,
                       extract_numeric_tokens, normalize_country, normalize_name,
                       postal_candidates)
from phase6d_retrieval import address_retrieve, build_address_index
from phase7_pairs import merge_candidate_routes
from phase8_model import FEATURE_NAMES, SCORE_HEADER, extract_features, load_models
from phase9_policy import DecisionPolicy


BASE = Path(__file__).resolve().parents[1]
ART = BASE / "artifacts/test_inference/phase12"
OUT = BASE.parent.parent / "output"
TEST = DEFAULT_DATASET_ROOT / "test"
S1 = TEST / "test_source1.tsv"
S2 = TEST / "test_source2.tsv"
S3 = TEST / "test_source3.tsv"
INDEXES = ART / "indexes"
CANDIDATES = ART / "candidates"
SCORES = ART / "scores"
PREDICTIONS = ART / "predictions"
V1_INDEX = INDEXES / "test_v1_index.sqlite"
ADDRESS_INDEX = INDEXES / "test_address_postings.sqlite"
V1_META = INDEXES / "test_v1_index.provenance.json"
ADDRESS_META = INDEXES / "test_address_index.provenance.json"
POLICY_PATH = BASE / "artifacts/model/phase9/decision_policy.json"
MANIFEST_PATH = BASE / "artifacts/model/phase8/feature_manifest.json"
MODEL_PATHS = {s: BASE / f"artifacts/model/phase8/model_{s.lower()}.txt" for s in ("S2", "S3")}
CANONICAL = CANDIDATES / "candidate_pairs_long.tsv.gz"
SHARD_MANIFEST = CANDIDATES / "shard_manifest.json"
SCORE_MANIFEST = SCORES / "score_manifest.json"
CHUNK = 25_000
BENCHMARK_N = 10_000
LONG_HEADER = ["source1_entity_id", "candidate_entity_id", "target_source", "from_v1", "from_address", "address_rank", "address_score", *ROUTES]
SCORE_SHARD_HEADER = SCORE_HEADER
FROZEN = (POLICY_PATH, MANIFEST_PATH, *MODEL_PATHS.values())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_pair_hash(path: Path) -> str:
    """Stable hash of candidate membership, independent of gzip metadata."""
    digest = hashlib.sha256()
    for row in _iter_long(path):
        digest.update(row["source1_entity_id"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(row["candidate_entity_id"].encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def fingerprint(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def rss() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def atomic_write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def frozen_snapshot() -> dict[str, dict[str, object]]:
    return {str(path): fingerprint(path) for path in FROZEN}


def assert_frozen(before: Mapping[str, Mapping[str, object]]) -> None:
    after = frozen_snapshot()
    if dict(before) != after:
        raise RuntimeError("a frozen Phase 4--9 model/policy artifact changed")


def read_s1(path: Path = S1) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != TARGET_HEADER:
            raise ValueError(f"unexpected test Source 1 header: {reader.fieldnames!r}")
        values = list(reader)
    ids = [row["entity_id"] for row in values]
    if len(ids) != len(set(ids)) or any(not value.startswith("S1-") for value in ids):
        raise ValueError("test Source 1 IDs are malformed or duplicated")
    return values


def test_target_paths() -> dict[str, Path]:
    return {"S2": S2, "S3": S3}


def frozen_policy() -> DecisionPolicy:
    payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    value = payload["policy"]
    policy = DecisionPolicy(float(value["s2_threshold"]), float(value["s3_threshold"]),
                            value.get("open_threshold"), value["conflict_policy"], value.get("conflict_margin"))
    if policy != DecisionPolicy(0.93, 0.97, None, "highest", None):
        raise ValueError("Phase 9 policy is not the approved frozen policy")
    return policy


def validate_manifest_and_models() -> dict[str, object]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("feature_count") != 48 or tuple(manifest.get("feature_order", ())) != FEATURE_NAMES:
        raise ValueError("feature manifest does not match frozen 48-feature schema")
    models = load_models(MODEL_PATHS["S2"].parent)
    for source, model in models.items():
        if tuple(model.feature_name()) != FEATURE_NAMES:
            raise ValueError(f"{source} model feature order differs from manifest")
    return {"manifest": fingerprint(MANIFEST_PATH), "models": {s: fingerprint(p) for s, p in MODEL_PATHS.items()}, "policy": asdict(frozen_policy())}


def _count_tsv(path: Path, prefix: str) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        if next(reader, None) != TARGET_HEADER:
            raise ValueError(f"unexpected target header: {path}")
        count = 0
        for row in reader:
            if len(row) != 4 or not row[0].startswith(prefix):
                raise ValueError(f"malformed target row in {path}")
            count += 1
    return count


def inventory() -> dict[str, object]:
    rows = read_s1()
    targets = {source: _count_tsv(path, source + "-") for source, path in test_target_paths().items()}
    existing = []
    for candidate in (V1_INDEX, ADDRESS_INDEX):
        meta = candidate.with_suffix(candidate.suffix + ".provenance.json")
        safe = meta.is_file() and json.loads(meta.read_text(encoding="utf-8")).get("targets") == {s: fingerprint(p) for s, p in test_target_paths().items()}
        existing.append({"path": str(candidate), "exists": candidate.exists(), "provenance": str(meta), "safe_to_reuse": safe})
    free = shutil.disk_usage(BASE / "artifacts").free
    report = {"test_files": {"S1": fingerprint(S1), "S2": fingerprint(S2), "S3": fingerprint(S3)},
              "counts": {"S1": len(rows), **targets}, "existing_test_indexes": existing,
              "frozen": validate_manifest_and_models(), "free_disk_bytes": free,
              "rough_validation_projection": {"candidate_pairs": round(len(rows) * 146.51), "candidate_wide_bytes": round(len(rows) * 146.51 * 14), "score_bytes": round(len(rows) * 146.51 * 12)}}
    atomic_write_json(ART / "inventory.json", report)
    return report


def _index_provenance(path: Path, target_files: Mapping[str, Path]) -> Path:
    return path.with_suffix(path.suffix + ".provenance.json")


def _verify_test_index(path: Path, target_files: Mapping[str, Path]) -> bool:
    meta = _index_provenance(path, target_files)
    if not path.is_file() or not meta.is_file():
        return False
    try:
        payload = json.loads(meta.read_text(encoding="utf-8"))
        return payload.get("targets") == {s: fingerprint(p) for s, p in target_files.items()}
    except (OSError, ValueError):
        return False


def build_v1_index(target_files: Mapping[str, Path], output: Path) -> dict[str, object]:
    """Frozen V1 indexing rules, generalized to explicit target TSV paths."""
    if _verify_test_index(output, target_files):
        return {"reused": True, "index": str(output), "bytes": output.stat().st_size}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".sqlite.tmp")
    temporary.unlink(missing_ok=True)
    start = time.perf_counter(); targets = postings = 0
    db = sqlite3.connect(temporary)
    try:
        db.execute("PRAGMA journal_mode=OFF"); db.execute("PRAGMA synchronous=OFF"); db.execute("PRAGMA temp_store=FILE"); db.execute("PRAGMA cache_size=-131072")
        db.execute("CREATE TABLE target (target INTEGER PRIMARY KEY, entity_id TEXT UNIQUE, source TEXT, country TEXT, name TEXT, address TEXT, basic TEXT, core TEXT)")
        db.execute("CREATE TABLE posting (kind INTEGER, key TEXT, target INTEGER)")
        db.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        for source in ("S2", "S3"):
            batch_t: list[tuple] = []; batch_p: list[tuple] = []
            with target_files[source].open(encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle, delimiter="\t")
                if next(reader, None) != TARGET_HEADER: raise ValueError("unexpected target header")
                for row in reader:
                    if len(row) != 4 or not row[0].startswith(source + "-"): raise ValueError("malformed target row")
                    entity_id, name, address, raw_country = row; country = normalize_country(raw_country)
                    basic, core = normalize_name(name), core_name(name); targets += 1
                    batch_t.append((targets, entity_id, source, country, name, address, basic, core))
                    values = set()
                    if basic: values.add((BASIC, basic))
                    if core: values.add((CORE, core))
                    values.update((TOKEN, token) for token in core_name_tokens(name) if len(token) >= 3)
                    values.update((NUMBER, n) for n in extract_numeric_tokens(address))
                    values.update((POSTAL, p) for p in postal_candidates(address, country))
                    batch_p.extend((kind, key(country, source, value), targets) for kind, value in values)
                    if len(batch_t) >= 10_000:
                        db.executemany("INSERT INTO target VALUES (?,?,?,?,?,?,?,?)", batch_t); db.executemany("INSERT INTO posting VALUES (?,?,?)", batch_p); postings += len(batch_p); batch_t.clear(); batch_p.clear(); db.commit()
            if batch_t: db.executemany("INSERT INTO target VALUES (?,?,?,?,?,?,?,?)", batch_t); db.executemany("INSERT INTO posting VALUES (?,?,?)", batch_p); postings += len(batch_p); db.commit()
        db.execute("CREATE INDEX posting_lookup ON posting(kind,key,target)")
        db.executemany("INSERT INTO metadata VALUES (?,?)", [("version", "v2"), ("complete", "1"), ("target_rows", str(targets)), ("postings", str(postings))]); db.commit()
    except Exception:
        db.close(); temporary.unlink(missing_ok=True); raise
    finally:
        try: db.close()
        except Exception: pass
    temporary.replace(output); frequency = ensure_frequency_table(output)
    payload = {"targets": {s: fingerprint(p) for s, p in target_files.items()}, "index": fingerprint(output), "rules": "frozen blocking.py V1 v2", "frequency": frequency}
    atomic_write_json(_index_provenance(output, target_files), payload)
    return {"reused": False, "index": str(output), "bytes": output.stat().st_size, "targets": targets, "postings": postings, "seconds": time.perf_counter() - start}


def prepare_indices() -> dict[str, object]:
    before = frozen_snapshot(); targets = test_target_paths(); INDEXES.mkdir(parents=True, exist_ok=True)
    v1 = build_v1_index(targets, V1_INDEX)
    if not _verify_test_index(V1_INDEX, targets): raise RuntimeError("test V1 provenance verification failed")
    if _verify_test_index(ADDRESS_INDEX, targets): address = {"reused": True, "index": str(ADDRESS_INDEX), "bytes": ADDRESS_INDEX.stat().st_size}
    else:
        temporary = ADDRESS_INDEX.with_suffix(".sqlite.tmp"); temporary.unlink(missing_ok=True)
        address = build_address_index(V1_INDEX, temporary); temporary.replace(ADDRESS_INDEX)
        atomic_write_json(_index_provenance(ADDRESS_INDEX, targets), {"targets": {s: fingerprint(p) for s, p in targets.items()}, "v1_index": fingerprint(V1_INDEX), "rules": "frozen phase6d Address K10", "index": fingerprint(ADDRESS_INDEX)})
    assert_frozen(before)
    result = {"v1": v1, "address": address, "frozen_unchanged": True, "peak_rss_mb": rss()}
    atomic_write_json(ART / "preflight.json", result); return result


def shards(rows: list[dict[str, str]], size: int = CHUNK) -> list[dict[str, object]]:
    return [{"shard": i, "first_s1": part[0]["entity_id"], "last_s1": part[-1]["entity_id"], "count": len(part)} for i, part in enumerate((rows[n:n + size] for n in range(0, len(rows), size)))]


def _address_rows(rows: list[dict[str, str]], path: Path) -> dict[str, list[dict[str, str]]]:
    mapping: dict[str, list[dict[str, str]]] = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader: mapping[row["source1_entity_id"]].append(row)
    return mapping


def _candidate_shard(spec: Mapping[str, object], rows: list[dict[str, str]], root: Path) -> dict[str, object]:
    shard = int(spec["shard"])
    # Workers receive only their own S1 slice; benchmark calls also pass one slice.
    part = rows if len(rows) == int(spec["count"]) else rows[shard * CHUNK: shard * CHUNK + int(spec["count"])]
    directory = root / "shards"; directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"candidates_{shard:05d}.tsv.gz"; manifest = directory / f"candidates_{shard:05d}.json"
    if output.exists() and manifest.exists():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved.get("sha256") == sha256(output) and saved.get("first_s1") == spec["first_s1"] and saved.get("last_s1") == spec["last_s1"]: return saved
    start = time.perf_counter(); temporary_address = directory / f"address_{shard:05d}.tsv.gz"
    address_retrieve([r["entity_id"] for r in part], {r["entity_id"]: r for r in part}, V1_INDEX, ADDRESS_INDEX, temporary_address, top_k=10)
    address = _address_rows(part, temporary_address); temporary = output.with_suffix(".tsv.gz.tmp")
    v1_pairs = address_pairs = union = 0
    index = Index(V1_INDEX, Limits())
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=LONG_HEADER, delimiter="\t", lineterminator="\n"); writer.writeheader()
            for row in part:
                s1 = row["entity_id"]; masks, shared = retrieve(index, row["business_name"], row["business_address"], row["country"]); targets = index.target_rows(masks)
                v1 = []
                for target_num, mask in masks.items():
                    target, source, _, _ = targets[target_num]
                    v1.append({"candidate_entity_id": target, "target_source": source, **{route: str(int(bool(mask & (1 << n)))) for n, route in enumerate(ROUTES)}, "shared_informative_token_count": str(shared.get(target_num, 0)), "num_blocking_routes": str(mask.bit_count())})
                merged = merge_candidate_routes(v1, address.get(s1, [])); v1_pairs += len(v1); address_pairs += len(address.get(s1, [])); union += len(merged)
                for target, evidence in sorted(merged.items()):
                    writer.writerow({"source1_entity_id": s1, "candidate_entity_id": target, "target_source": evidence["target_source"], "from_v1": evidence["from_v1"], "from_address": evidence["from_address"], "address_rank": evidence["address_rank"], "address_score": evidence["address_score"], **{route: int(evidence.get("routes", {}).get(route, False)) for route in ROUTES}})
    finally:
        index.close(); temporary_address.unlink(missing_ok=True)
    temporary.replace(output)
    result = {**spec, "file": str(output), "bytes": output.stat().st_size, "sha256": sha256(output), "v1_pairs": v1_pairs, "address_pairs": address_pairs, "union_pairs": union, "runtime_seconds": time.perf_counter() - start, "peak_rss_mb": rss(), "status": "complete"}
    atomic_write_json(manifest, result); return result


def enforce_disk_preflight(workers: int) -> dict[str, object]:
    """Use the selected benchmark to reject an unsafe full-materialization run."""
    path = ART / "benchmark" / f"workers_{workers}" / "benchmark.json"
    if not path.is_file():
        raise RuntimeError("run benchmark for the selected worker count before full candidate generation")
    benchmark_report = json.loads(path.read_text(encoding="utf-8"))
    projected_candidate = float(benchmark_report["projection"]["candidate_bytes"])
    # Candidate shards, canonical long form, wide candidate TSV, and score shards coexist.
    # The multiplier deliberately leaves room for SQLite journals and atomic temporary files.
    projected_peak = projected_candidate * 4.5
    free = shutil.disk_usage(BASE / "artifacts").free
    result = {"free_bytes": free, "projected_peak_bytes": projected_peak,
              "limit_bytes": free * .85, "benchmark": str(path)}
    if projected_peak > free * .85:
        raise RuntimeError("projected Phase 12 disk use exceeds 85% of currently free space; free disk or reduce storage safely")
    return result


def generate_candidates(workers: int = 1, shard_size: int = CHUNK) -> dict[str, object]:
    if workers < 1 or workers > 4: raise ValueError("workers must be between 1 and 4")
    if shard_size != CHUNK: raise ValueError("custom shard size is not supported in this frozen runner")
    if not _verify_test_index(V1_INDEX, test_target_paths()) or not _verify_test_index(ADDRESS_INDEX, test_target_paths()): raise RuntimeError("run prepare-indices with verified test indexes first")
    before = frozen_snapshot(); disk_preflight = enforce_disk_preflight(workers); values = read_s1(); specs = shards(values); start = time.perf_counter()
    jobs = [(spec, values[int(spec["shard"]) * CHUNK:int(spec["shard"]) * CHUNK + int(spec["count"])]) for spec in specs]
    results = []
    executor = None
    try:
        if workers == 1:
            completed = (_candidate_shard(spec, part, CANDIDATES) for spec, part in jobs)
        else:
            executor = ProcessPoolExecutor(max_workers=workers)
            futures = [executor.submit(_candidate_shard, spec, part, CANDIDATES) for spec, part in jobs]
            completed = (future.result() for future in as_completed(futures))
        for n, result in enumerate(completed, 1):
            results.append(result)
            elapsed = time.perf_counter() - start; eta = elapsed / n * (len(specs) - n)
            print(f"Shard {n}/{len(specs)} complete; S1={sum(x['count'] for x in results):,}/{len(values):,}; candidates={sum(x['union_pairs'] for x in results):,}; elapsed={elapsed:.1f}s ETA={eta:.1f}s", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    results.sort(key=lambda result: int(result["shard"]))
    payload = {"workers": workers, "shard_size": shard_size, "disk_preflight": disk_preflight, "shards": results, "total_pairs": sum(x["union_pairs"] for x in results), "runtime_seconds": time.perf_counter() - start, "peak_rss_mb": rss()}
    atomic_write_json(SHARD_MANIFEST, payload); assert_frozen(before); return payload


def _iter_long(path: Path) -> Iterator[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != LONG_HEADER: raise ValueError("unexpected canonical candidate header")
        yield from reader


def finalize_candidates() -> dict[str, object]:
    before = frozen_snapshot(); manifest = json.loads(SHARD_MANIFEST.read_text(encoding="utf-8")); rows = read_s1(); saved = manifest["shards"]
    if len(saved) != len(shards(rows)): raise ValueError("candidate shard manifest is incomplete")
    CANDIDATES.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    temporary = CANONICAL.with_suffix(".tsv.gz.tmp"); candidate_out = OUT / "candidate_pairs.tsv"; out_tmp = candidate_out.with_suffix(".tsv.tmp")
    totals = Counter(); sizes = []
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as long_handle, out_tmp.open("w", encoding="utf-8", newline="") as wide_handle:
        lw = csv.DictWriter(long_handle, fieldnames=LONG_HEADER, delimiter="\t", lineterminator="\n"); lw.writeheader(); ww = csv.writer(wide_handle, delimiter="\t", lineterminator="\n"); ww.writerow(CANDIDATE_HEADER)
        for spec, s1 in zip(saved, shards(rows)):
            path = Path(spec["file"])
            if not path.is_file() or sha256(path) != spec["sha256"]: raise ValueError(f"invalid candidate shard {path}")
            grouped: dict[str, list[str]] = {row["entity_id"]: [] for row in rows[int(s1['shard']) * CHUNK:int(s1['shard']) * CHUNK + int(s1['count'])]}
            for row in _iter_long(path):
                lw.writerow(row); grouped[row["source1_entity_id"]].append(row["candidate_entity_id"]); totals[row["target_source"]] += 1
            for s1_id, values in grouped.items():
                if len(values) != len(set(values)): raise ValueError("duplicate candidate pair")
                ww.writerow((s1_id, ",".join(sorted(values)))); sizes.append(len(values))
    temporary.replace(CANONICAL); out_tmp.replace(candidate_out)
    result = {"s1_count": len(rows), "pairs": int(sum(totals.values())), "S2_pairs": totals["S2"], "S3_pairs": totals["S3"], "candidate_stats": {"mean": float(np.mean(sizes)), "median": float(np.percentile(sizes, 50)), "p95": float(np.percentile(sizes, 95)), "p99": float(np.percentile(sizes, 99)), "max": max(sizes), "zero": sum(v == 0 for v in sizes)}, "canonical": fingerprint(CANONICAL), "output": fingerprint(candidate_out)}
    if result["pairs"] != sum(item["union_pairs"] for item in saved): raise AssertionError("final candidate pair count differs from shards")
    atomic_write_json(CANDIDATES / "candidate_summary.json", result); assert_frozen(before); return result


def check_candidates() -> dict[str, object]:
    summary = json.loads((CANDIDATES / "candidate_summary.json").read_text(encoding="utf-8")); rows = read_s1(); count = 0; previous = None
    for row in _iter_long(CANONICAL):
        if row["target_source"] not in ("S2", "S3") or not row["candidate_entity_id"].startswith(row["target_source"] + "-"):
            raise ValueError("candidate target source mismatch")
        pair = (row["source1_entity_id"], row["candidate_entity_id"])
        if previous == pair:
            raise ValueError("duplicate canonical candidate")
        previous = pair; count += 1
    if count != summary["pairs"]: raise ValueError("canonical candidate count mismatch")
    return {"ok": True, "s1_count": len(rows), "pairs": count, "canonical_sha256": sha256(CANONICAL)}


def benchmark(workers: int = 1) -> dict[str, object]:
    if workers not in (1, 2, 4): raise ValueError("benchmark workers must be 1, 2, or 4")
    rows = read_s1()[:BENCHMARK_N]
    # Candidate behavior is exactly shard behavior; this writes only a disposable benchmark directory.
    root = ART / "benchmark" / f"workers_{workers}"; root.mkdir(parents=True, exist_ok=True)
    spec = {"shard": 0, "first_s1": rows[0]["entity_id"], "last_s1": rows[-1]["entity_id"], "count": len(rows)}
    started = time.perf_counter(); result = _candidate_shard(spec, rows, root)
    elapsed = time.perf_counter() - started; projected = len(read_s1()) / len(rows)
    report = {"workers": workers, "sample_s1": len(rows), "candidate": result, "seconds": elapsed, "rows_per_second": len(rows) / elapsed, "projection": {"candidate_seconds": elapsed * projected, "candidate_pairs": result["union_pairs"] * projected, "candidate_bytes": result["bytes"] * projected}, "peak_rss_mb": rss()}
    atomic_write_json(root / "benchmark.json", report); return report


def compare_benchmarks() -> dict[str, object]:
    reports = []
    for workers in (1, 2, 4):
        path = ART / "benchmark" / f"workers_{workers}" / "benchmark.json"
        if path.exists(): reports.append(json.loads(path.read_text(encoding="utf-8")))
    if not reports: raise FileNotFoundError("run benchmark first")
    safe = [r for r in reports if r["peak_rss_mb"] < 20 * 1024]
    recommended = min(safe, key=lambda r: r["projection"]["candidate_seconds"]) if safe else min(reports, key=lambda r: r["peak_rss_mb"])
    result = {"benchmarks": reports, "recommended_workers": recommended["workers"], "reason": "fastest benchmark under 20 GiB RSS"}
    atomic_write_json(ART / "benchmark_comparison.json", result); return result


def score() -> dict[str, object]:
    before = frozen_snapshot(); check_candidates(); validate_manifest_and_models(); rows = read_s1(); s1_by_id = {row["entity_id"]: row for row in rows}; models = load_models(MODEL_PATHS["S2"].parent)
    SCORES.mkdir(parents=True, exist_ok=True); score_paths = {s: SCORES / f"test_scores_{s.lower()}.tsv.gz" for s in ("S2", "S3")}; temporary = {s: p.with_suffix(".tsv.gz.tmp") for s, p in score_paths.items()}; started = time.perf_counter(); counts = Counter(); store = TargetStore(V1_INDEX)
    try:
        handles = {s: gzip.open(temporary[s], "wt", encoding="utf-8", newline="") for s in ("S2", "S3")}; writers = {s: csv.DictWriter(handles[s], fieldnames=SCORE_SHARD_HEADER, delimiter="\t", lineterminator="\n") for s in handles}
        for writer in writers.values(): writer.writeheader()
        current = None; group: list[dict[str, str]] = []
        def consume(s1_id: str, pairs: list[dict[str, str]]) -> None:
            if not pairs: return
            record_row = s1_by_id.get(s1_id)
            if record_row is None:
                raise ValueError(f"canonical candidate references unknown test S1: {s1_id}")
            left = make_record(s1_id, record_row["business_name"], record_row["business_address"], record_row["country"]); targets = store.get_many([p["candidate_entity_id"] for p in pairs])
            for source in ("S2", "S3"):
                selected = [p for p in pairs if p["target_source"] == source]
                if not selected: continue
                matrix = np.vstack([extract_features(left, targets[p["candidate_entity_id"]], p, len(selected)) for p in selected]); values = models[source].predict(matrix, num_threads=1)
                for pair, value in zip(selected, values): writers[source].writerow({"source1_entity_id": s1_id, "candidate_entity_id": pair["candidate_entity_id"], "score": f"{float(value):.10f}", "from_v1": pair["from_v1"], "from_address": pair["from_address"], "address_rank": pair["address_rank"], "address_score": pair["address_score"]}); counts[source] += 1
        for row in _iter_long(CANONICAL):
            if current is None: current = row["source1_entity_id"]
            if row["source1_entity_id"] != current: consume(current, group); current, group = row["source1_entity_id"], []
            group.append(row)
        if current is not None: consume(current, group)
    finally:
        store.close()
        for handle in locals().get("handles", {}).values(): handle.close()
    for source in temporary: temporary[source].replace(score_paths[source])
    result = {"rows": dict(counts), "total": sum(counts.values()), "paths": {s: fingerprint(p) for s, p in score_paths.items()}, "runtime_seconds": time.perf_counter() - started, "peak_rss_mb": rss(), "candidate_sha256": sha256(CANONICAL), "frozen_unchanged": True}
    if result["total"] != json.loads((CANDIDATES / "candidate_summary.json").read_text())["pairs"]: raise AssertionError("not every candidate was scored")
    atomic_write_json(SCORE_MANIFEST, result); assert_frozen(before); return result


def predict() -> dict[str, object]:
    before = frozen_snapshot(); policy = frozen_policy(); rows = read_s1(); score_manifest = json.loads(SCORE_MANIFEST.read_text()); temp = PREDICTIONS / "ownership.sqlite"; PREDICTIONS.mkdir(parents=True, exist_ok=True); temp.unlink(missing_ok=True)
    db = sqlite3.connect(temp); accepted = Counter(); rejected = Counter()
    try:
        db.execute("CREATE TABLE accepted (target TEXT, s1 TEXT, score REAL)")
        for source in ("S2", "S3"):
            threshold = policy.s2_threshold if source == "S2" else policy.s3_threshold
            with gzip.open(SCORES / f"test_scores_{source.lower()}.tsv.gz", "rt", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    value = float(row["score"])
                    if value >= threshold: db.execute("INSERT INTO accepted VALUES (?,?,?)", (row["candidate_entity_id"], row["source1_entity_id"], value)); accepted[source] += 1
                    else: rejected[source] += 1
        db.execute("CREATE INDEX accepted_target ON accepted(target, score DESC, s1)"); db.execute("CREATE TABLE winner AS SELECT target, s1, score FROM (SELECT target,s1,score,ROW_NUMBER() OVER (PARTITION BY target ORDER BY score DESC,s1) AS rn FROM accepted) WHERE rn=1"); db.execute("CREATE INDEX winner_s1 ON winner(s1,target)"); db.commit()
        output = OUT / "matching_results.tsv"; OUT.mkdir(parents=True, exist_ok=True); temporary = output.with_suffix(".tsv.tmp"); sizes=[]
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n"); writer.writerow(["source1_entity_id", "matched_entity_ids"])
            for row in rows:
                values = [v[0] for v in db.execute("SELECT target FROM winner WHERE s1=? ORDER BY target", (row["entity_id"],))]; writer.writerow((row["entity_id"], ",".join(values))); sizes.append(len(values))
        temporary.replace(output)
        winners = db.execute("SELECT count(*) FROM winner").fetchone()[0]; conflicts = db.execute("SELECT count(*) FROM (SELECT target FROM accepted GROUP BY target HAVING count(*)>1)").fetchone()[0]
    finally: db.close(); temp.unlink(missing_ok=True)
    result = {"policy": asdict(policy), "predicted_links": winners, "accepted_before_ownership": dict(accepted), "rejected": dict(rejected), "target_conflicts_resolved": conflicts, "singleton_count": sum(x == 0 for x in sizes), "prediction_stats": {"mean": float(np.mean(sizes)), "median": float(np.percentile(sizes,50)), "p95": float(np.percentile(sizes,95)), "p99": float(np.percentile(sizes,99)), "max": max(sizes)}, "output": fingerprint(OUT / "matching_results.tsv"), "score_manifest": score_manifest["paths"]}
    atomic_write_json(PREDICTIONS / "prediction_summary.json", result); assert_frozen(before); return result


def report() -> dict[str, object]:
    candidate = json.loads((CANDIDATES / "candidate_summary.json").read_text()); score = json.loads(SCORE_MANIFEST.read_text()); prediction = json.loads((PREDICTIONS / "prediction_summary.json").read_text())
    result = {"inputs": inventory(), "retrieval": candidate, "scoring": score, "predictions": prediction, "output": {"candidate_pairs": fingerprint(OUT / "candidate_pairs.tsv"), "matching_results": fingerprint(OUT / "matching_results.tsv")}, "integrity": {"frozen_unchanged": True, "score_rows_equal_candidates": score["total"] == candidate["pairs"], "match_subset_candidates": True}}
    atomic_write_json(ART / "phase12_summary.json", result); return result


def _membership_hash(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        for row in _iter_long(path):
            digest.update(row["source1_entity_id"].encode()); digest.update(b"\0")
            digest.update(row["candidate_entity_id"].encode()); digest.update(b"\n")
    return digest.hexdigest()


def _candidate_jobs(rows: list[dict[str, str]], workers: int, root: Path) -> list[dict[str, object]]:
    # More than one task is required for a meaningful parallel benchmark.
    task_size = max(1, (len(rows) + workers * 2 - 1) // (workers * 2))
    specs = shards(rows, task_size)
    jobs = [(spec, rows[int(spec["shard"]) * task_size:int(spec["shard"]) * task_size + int(spec["count"])]) for spec in specs]
    if workers == 1:
        result = [_candidate_shard(spec, part, root) for spec, part in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            result = [future.result() for future in as_completed([pool.submit(_candidate_shard, spec, part, root) for spec, part in jobs])]
    return sorted(result, key=lambda value: int(value["shard"]))


def benchmark(workers: int = 1) -> dict[str, object]:
    if workers not in (1, 2, 4):
        raise ValueError("benchmark workers must be 1, 2, or 4")
    rows = read_s1()[:BENCHMARK_N]
    root = ART / "benchmark" / f"workers_{workers}"
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter(); results = _candidate_jobs(rows, workers, root); elapsed = time.perf_counter() - started
    total = Counter()
    for item in results:
        for name in ("v1_pairs", "address_pairs", "union_pairs"):
            total[name] += int(item[name])
    multiplier = len(read_s1()) / len(rows)
    bytes_written = sum(int(item["bytes"]) for item in results)
    report = {
        "workers": workers, "sample_s1": len(rows), "shards": results,
        "v1_pairs": total["v1_pairs"], "address_pairs": total["address_pairs"], "union_pairs": total["union_pairs"],
        "rows_per_second": total["union_pairs"] / elapsed if elapsed else 0.0,
        "runtime_seconds": elapsed, "compressed_bytes": bytes_written,
        "peak_rss_mb": rss(), "membership_sha256": _membership_hash([Path(x["file"]) for x in results]),
        "projection": {"candidate_seconds": elapsed * multiplier, "candidate_pairs": total["union_pairs"] * multiplier,
                       "candidate_bytes": bytes_written * multiplier},
    }
    atomic_write_json(root / "benchmark.json", report); return report


def compare_benchmarks() -> dict[str, object]:
    reports = [json.loads((ART / "benchmark" / f"workers_{n}" / "benchmark.json").read_text()) for n in (1,2,4) if (ART / "benchmark" / f"workers_{n}" / "benchmark.json").exists()]
    if not reports: raise FileNotFoundError("run benchmark first")
    hashes = {r["membership_sha256"] for r in reports}
    if len(hashes) != 1: raise RuntimeError("candidate membership differs across worker benchmarks")
    safe = [r for r in reports if float(r["peak_rss_mb"]) < 20 * 1024 and float(r["projection"]["candidate_bytes"]) * 4.5 < shutil.disk_usage(BASE / "artifacts").free * .85]
    selected = min(safe or reports, key=lambda r: float(r["runtime_seconds"]))
    result = {"benchmarks": reports, "recommended_workers": selected["workers"], "membership_equivalent": True,
              "reason": "fastest equivalent benchmark within the 20 GiB RSS and disk safety limits"}
    atomic_write_json(ART / "benchmark_comparison.json", result); return result


def finalize_candidates() -> dict[str, object]:
    before = frozen_snapshot(); manifest = json.loads(SHARD_MANIFEST.read_text()); saved = sorted(manifest["shards"], key=lambda x: int(x["shard"]))
    CANDIDATES.mkdir(parents=True, exist_ok=True); temporary = CANONICAL.with_suffix(".tsv.gz.tmp")
    totals, sizes, previous = Counter(), [], None
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LONG_HEADER, delimiter="\t", lineterminator="\n"); writer.writeheader()
        current = None; count = 0
        for item in saved:
            path = Path(item["file"])
            if not path.is_file() or sha256(path) != item["sha256"]: raise ValueError(f"invalid candidate shard {path}")
            for row in _iter_long(path):
                pair = (row["source1_entity_id"], row["candidate_entity_id"])
                if previous == pair: raise ValueError("duplicate candidate pair")
                previous = pair
                if current is not None and row["source1_entity_id"] != current: sizes.append(count); count = 0
                current = row["source1_entity_id"]; count += 1; totals[row["target_source"]] += 1; writer.writerow(row)
        if current is not None: sizes.append(count)
    temporary.replace(CANONICAL)
    result = {"s1_count": len(read_s1()), "pairs": sum(totals.values()), "S2_pairs": totals["S2"], "S3_pairs": totals["S3"],
              "canonical": fingerprint(CANONICAL), "membership_sha256": semantic_pair_hash(CANONICAL),
              "candidate_stats": {"mean": float(np.mean(sizes)), "median": float(np.percentile(sizes,50)), "p95": float(np.percentile(sizes,95)), "p99": float(np.percentile(sizes,99)), "max": max(sizes), "zero": len(read_s1())-len(sizes)},
              "official_wide_export": "deferred until score manifest proves every canonical pair was scored"}
    if result["pairs"] != sum(int(x["union_pairs"]) for x in saved): raise AssertionError("candidate count differs from shards")
    atomic_write_json(CANDIDATES / "candidate_summary.json", result); assert_frozen(before); return result


def _score_one_shard(item: Mapping[str, object], s1_rows: list[dict[str, str]], candidate_path: Path, output_root: Path) -> dict[str, object]:
    shard = int(item["shard"]); directory = output_root / "shards"; directory.mkdir(parents=True, exist_ok=True)
    outputs = {src: directory / f"scores_{shard:05d}_{src.lower()}.tsv.gz" for src in ("S2","S3")}; manifest = directory / f"scores_{shard:05d}.json"
    candidate_hash = sha256(candidate_path)
    if manifest.exists() and all(path.exists() for path in outputs.values()):
        saved = json.loads(manifest.read_text())
        if saved.get("candidate_sha256") == candidate_hash and all(saved.get("outputs",{}).get(src,{}).get("sha256") == sha256(outputs[src]) for src in outputs): return saved
    models = load_models(MODEL_PATHS["S2"].parent); lookup = {r["entity_id"]:r for r in s1_rows}; counts=Counter(); started=time.perf_counter(); store=TargetStore(V1_INDEX)
    temporary={src:path.with_suffix(".tsv.gz.tmp") for src,path in outputs.items()}
    try:
        handles={src:gzip.open(temporary[src],"wt",encoding="utf-8",newline="") for src in outputs}; writers={src:csv.DictWriter(handles[src],fieldnames=SCORE_SHARD_HEADER,delimiter="\t",lineterminator="\n") for src in outputs}
        for writer in writers.values(): writer.writeheader()
        current=None; group=[]
        def consume(s1_id, pairs):
            if not pairs: return
            leftrow=lookup[s1_id]; left=make_record(s1_id,leftrow["business_name"],leftrow["business_address"],leftrow["country"]); targets=store.get_many([p["candidate_entity_id"] for p in pairs])
            for source in ("S2","S3"):
                selected=[p for p in pairs if p["target_source"]==source]
                if not selected: continue
                matrix=np.vstack([extract_features(left,targets[p["candidate_entity_id"]],p,len(selected)) for p in selected])
                values=models[source].predict(matrix,num_threads=1)
                for pair,value in zip(selected,values): writers[source].writerow({"source1_entity_id":s1_id,"candidate_entity_id":pair["candidate_entity_id"],"score":f"{float(value):.10f}","from_v1":pair["from_v1"],"from_address":pair["from_address"],"address_rank":pair["address_rank"],"address_score":pair["address_score"]}); counts[source]+=1
        for row in _iter_long(candidate_path):
            if current is None: current=row["source1_entity_id"]
            if row["source1_entity_id"] != current: consume(current,group); current,group=row["source1_entity_id"],[]
            group.append(row)
        if current is not None: consume(current,group)
    finally:
        store.close()
        for handle in locals().get("handles",{}).values(): handle.close()
    for source in outputs: temporary[source].replace(outputs[source])
    result={"shard":shard,"candidate_path":str(candidate_path),"candidate_sha256":candidate_hash,"candidate_membership_sha256":semantic_pair_hash(candidate_path),"rows":dict(counts),"total":sum(counts.values()),"models":{s:fingerprint(MODEL_PATHS[s]) for s in outputs},"feature_manifest":fingerprint(MANIFEST_PATH),"outputs":{s:fingerprint(path) for s,path in outputs.items()},"runtime_seconds":time.perf_counter()-started,"peak_rss_mb":rss(),"status":"complete"}
    atomic_write_json(manifest,result); return result


def _score_jobs(items: list[Mapping[str,object]], rows: list[dict[str,str]], workers: int, root: Path) -> list[dict[str,object]]:
    jobs=[]; offset=0
    for item in sorted(items, key=lambda value: int(value["shard"])):
        count=int(item["count"]); part=rows[offset:offset + count]; offset += count
        if len(part) != count:
            raise ValueError("score shard S1 partition does not match its candidate manifest")
        candidate_path=Path(item["file"])
        if not candidate_path.is_file() or sha256(candidate_path) != item["sha256"]:
            raise ValueError("candidate shard changed after canonical candidate freeze")
        jobs.append((item,part,candidate_path,root))
    if workers==1: values=[_score_one_shard(*job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool: values=[f.result() for f in as_completed([pool.submit(_score_one_shard,*job) for job in jobs])]
    return sorted(values,key=lambda x:int(x["shard"]))


def score(workers: int=1) -> dict[str,object]:
    if workers not in (1,2,4): raise ValueError("score workers must be 1, 2, or 4")
    before=frozen_snapshot(); check_candidates(); validate_manifest_and_models(); candidate_manifest=json.loads(SHARD_MANIFEST.read_text()); rows=read_s1(); started=time.perf_counter()
    results=_score_jobs(candidate_manifest["shards"],rows,workers,SCORES)
    total=sum(int(x["total"]) for x in results); expected=json.loads((CANDIDATES/"candidate_summary.json").read_text())["pairs"]
    if total != expected: raise AssertionError("every canonical candidate must be scored exactly once")
    result={"workers":workers,"shards":results,"total":total,"candidate_membership_sha256":semantic_pair_hash(CANONICAL),"runtime_seconds":time.perf_counter()-started,"peak_rss_mb":rss(),"frozen_unchanged":True}
    atomic_write_json(SCORE_MANIFEST,result); assert_frozen(before); return result


def benchmark_score(workers:int=1)->dict[str,object]:
    if workers not in (1,2,4): raise ValueError("score benchmark workers must be 1, 2, or 4")
    source=ART/"benchmark"/"workers_1"/"benchmark.json"
    if not source.exists(): raise RuntimeError("run benchmark --workers 1 first")
    b=json.loads(source.read_text()); items=b["shards"]; rows=read_s1()[:BENCHMARK_N]; started=time.perf_counter(); result=_score_jobs(items,rows,workers,ART/"score_benchmark"/f"workers_{workers}"); elapsed=time.perf_counter()-started
    total=sum(int(x["total"]) for x in result); multiplier=len(read_s1())/len(rows)
    output={"workers":workers,"sample_s1":len(rows),"shards":result,"rows":total,"rows_per_second":total/elapsed if elapsed else 0.,"runtime_seconds":elapsed,"peak_rss_mb":rss(),"projection":{"score_seconds":elapsed*multiplier,"score_rows":total*multiplier,"score_bytes":sum(sum(x["outputs"][s]["bytes"] for s in ("S2","S3")) for x in result)*multiplier}}
    atomic_write_json(ART/"score_benchmark"/f"workers_{workers}"/"benchmark.json",output); return output


def compare_score_benchmarks()->dict[str,object]:
    reports=[json.loads((ART/"score_benchmark"/f"workers_{n}"/"benchmark.json").read_text()) for n in (1,2,4) if (ART/"score_benchmark"/f"workers_{n}"/"benchmark.json").exists()]
    if not reports: raise FileNotFoundError("run benchmark-score first")
    safe=[r for r in reports if float(r["peak_rss_mb"])<20*1024]
    selected=max(safe or reports,key=lambda r:float(r["rows_per_second"]))
    result={"benchmarks":reports,"recommended_workers":selected["workers"],"reason":"highest measured rows/sec under 20 GiB RSS"}; atomic_write_json(ART/"score_benchmark_comparison.json",result); return result


def _iter_score_rows(manifest: Mapping[str,object], source: str):
    for item in sorted(manifest["shards"],key=lambda x:int(x["shard"])):
        with gzip.open(item["outputs"][source]["path"],"rt",encoding="utf-8",newline="") as handle:
            yield from csv.DictReader(handle,delimiter="\t")


def predict() -> dict[str, object]:
    before=frozen_snapshot(); policy=frozen_policy(); rows=read_s1(); score_manifest=json.loads(SCORE_MANIFEST.read_text()); PREDICTIONS.mkdir(parents=True,exist_ok=True); temp=PREDICTIONS/"ownership.sqlite"; temp.unlink(missing_ok=True); db=sqlite3.connect(temp); accepted=Counter(); rejected=Counter()
    try:
        db.execute("CREATE TABLE accepted (target TEXT, s1 TEXT, score REAL)")
        for source,threshold in (("S2",policy.s2_threshold),("S3",policy.s3_threshold)):
            for row in _iter_score_rows(score_manifest,source):
                value=float(row["score"])
                if value>=threshold: db.execute("INSERT INTO accepted VALUES (?,?,?)",(row["candidate_entity_id"],row["source1_entity_id"],value)); accepted[source]+=1
                else: rejected[source]+=1
        db.execute("CREATE INDEX accepted_target ON accepted(target,score DESC,s1)"); db.execute("CREATE TABLE winner AS SELECT target,s1,score FROM (SELECT target,s1,score,ROW_NUMBER() OVER (PARTITION BY target ORDER BY score DESC,s1) rn FROM accepted) WHERE rn=1"); db.execute("CREATE INDEX winner_s1 ON winner(s1,target)"); db.commit()
        OUT.mkdir(parents=True,exist_ok=True); output=OUT/"matching_results.tsv"; temporary=output.with_suffix(".tsv.tmp"); sizes=[]
        with temporary.open("w",encoding="utf-8",newline="") as handle:
            writer=csv.writer(handle,delimiter="\t",lineterminator="\n"); writer.writerow(["source1_entity_id","matched_entity_ids"])
            for row in rows:
                matches=[x[0] for x in db.execute("SELECT target FROM winner WHERE s1=? ORDER BY target",(row["entity_id"],))]; writer.writerow((row["entity_id"],",".join(matches))); sizes.append(len(matches))
        temporary.replace(output); winners=db.execute("SELECT count(*) FROM winner").fetchone()[0]; conflicts=db.execute("SELECT count(*) FROM (SELECT target FROM accepted GROUP BY target HAVING count(*)>1)").fetchone()[0]
    finally: db.close(); temp.unlink(missing_ok=True)
    result={"policy":asdict(policy),"predicted_links":winners,"accepted_before_ownership":dict(accepted),"rejected":dict(rejected),"target_conflicts_resolved":conflicts,"singleton_count":sum(x==0 for x in sizes),"prediction_stats":{"mean":float(np.mean(sizes)),"median":float(np.percentile(sizes,50)),"p95":float(np.percentile(sizes,95)),"p99":float(np.percentile(sizes,99)),"max":max(sizes)},"output":fingerprint(OUT/"matching_results.tsv")}; atomic_write_json(PREDICTIONS/"prediction_summary.json",result); assert_frozen(before); return result


def export_candidates()->dict[str,object]:
    before=frozen_snapshot(); score=json.loads(SCORE_MANIFEST.read_text()); summary=json.loads((CANDIDATES/"candidate_summary.json").read_text())
    if score["total"] != summary["pairs"] or score["candidate_membership_sha256"] != summary["membership_sha256"]: raise RuntimeError("score manifest does not prove exact canonical candidate membership")
    OUT.mkdir(parents=True,exist_ok=True); output=OUT/"candidate_pairs.tsv"; temporary=output.with_suffix(".tsv.tmp"); rows=read_s1(); current=None; values=[]; counts=0
    with temporary.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.writer(handle,delimiter="\t",lineterminator="\n"); writer.writerow(CANDIDATE_HEADER)
        it=iter(_iter_long(CANONICAL)); row=next(it,None)
        for s1 in rows:
            vals=[]
            while row is not None and row["source1_entity_id"]==s1["entity_id"]: vals.append(row["candidate_entity_id"]); counts+=1; row=next(it,None)
            writer.writerow((s1["entity_id"],",".join(sorted(vals))))
        if row is not None: raise ValueError("canonical candidate S1 ordering differs from test source order")
    temporary.replace(output)
    result={"output":fingerprint(output),"pairs":counts,"canonical_membership_sha256":summary["membership_sha256"]}; atomic_write_json(CANDIDATES/"candidate_export.json",result); assert_frozen(before); return result


def report() -> dict[str, object]:
    candidate=json.loads((CANDIDATES/"candidate_summary.json").read_text()); score=json.loads(SCORE_MANIFEST.read_text()); prediction=json.loads((PREDICTIONS/"prediction_summary.json").read_text()); exported=json.loads((CANDIDATES/"candidate_export.json").read_text()) if (CANDIDATES/"candidate_export.json").exists() else None
    result={"inputs":inventory(),"retrieval":candidate,"scoring":score,"predictions":prediction,"candidate_export":exported,"integrity":{"frozen_unchanged":True,"score_rows_equal_candidates":score["total"]==candidate["pairs"],"match_subset_candidates":True,"candidate_export_matches_scored":bool(exported and exported["pairs"]==score["total"] )}}; atomic_write_json(ART/"phase12_summary.json",result); return result


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("command",choices=("inventory","prepare-indices","benchmark","compare-benchmarks","generate-candidates","finalize-candidates","check-candidates","benchmark-score","compare-score-benchmarks","score","predict","export-candidates","report")); parser.add_argument("--workers",type=int,default=1)
    args=parser.parse_args(argv); commands={"inventory":inventory,"prepare-indices":prepare_indices,"benchmark":lambda:benchmark(args.workers),"compare-benchmarks":compare_benchmarks,"generate-candidates":lambda:generate_candidates(args.workers),"finalize-candidates":finalize_candidates,"check-candidates":check_candidates,"benchmark-score":lambda:benchmark_score(args.workers),"compare-score-benchmarks":compare_score_benchmarks,"score":lambda:score(args.workers),"predict":predict,"export-candidates":export_candidates,"report":report}; print(json.dumps(commands[args.command](),indent=2,sort_keys=True,default=str)); return 0


if __name__ == "__main__": raise SystemExit(main())
