#!/usr/bin/env python3
"""Disk-backed character n-gram TF-IDF retrieval for Phase 6."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import resource
import sqlite3
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize as sparse_normalize

from blocking import METADATA_HEADER, selected_source1
from diagnostics import (CANDIDATE_HEADER, DEFAULT_DATASET_ROOT, append_candidate_experiment,
                         compare_candidate_files, evaluate_candidates_only, render_candidate_only_report)
from normalize import normalize_country, normalize_name
from scoring import load_ground_truth, load_id_file, parse_match_list

BASE = Path(__file__).resolve().parents[1]
ARTIFACTS = BASE / "artifacts"
BLOCKING = ARTIFACTS / "blocking"
N_FEATURES = 1 << 20
NGRAM_RANGE = (3, 5)
MAX_DF_RATIO = 0.001
TFIDF_HEADER = ["source1_entity_id", "candidate_entity_id", "target_source", "tfidf_score", "tfidf_rank"]
V2_METADATA_HEADER = [*METADATA_HEADER, "tfidf", "tfidf_score", "tfidf_rank"]


def _vectorizer() -> HashingVectorizer:
    return HashingVectorizer(analyzer="char_wb", ngram_range=NGRAM_RANGE,
                             n_features=N_FEATURES, alternate_sign=False,
                             norm=None, lowercase=False, dtype=np.float32)


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _paths(root: Path, source: str) -> list[Path]:
    return sorted(root.glob(f"{source.lower()}_*.npz"))


def _store(path: Path, matrix: csr_matrix, ids: list[str], countries: list[str]) -> None:
    np.savez_compressed(path, data=matrix.data.astype(np.float32, copy=False),
                        indices=matrix.indices.astype(np.int32, copy=False),
                        indptr=matrix.indptr.astype(np.int32, copy=False),
                        shape=np.asarray(matrix.shape, dtype=np.int64),
                        ids=np.asarray(ids, dtype="U"), countries=np.asarray(countries, dtype="U"))


def _load(path: Path) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as item:
        matrix = csr_matrix((item["data"], item["indices"], item["indptr"]),
                            shape=tuple(int(x) for x in item["shape"]))
        return matrix, item["ids"].copy(), item["countries"].copy()


def build_index(db_path: Path, index_dir: Path, batch_size: int = 10000,
                shard_rows: int = 50000, max_df_ratio: float = MAX_DF_RATIO) -> dict:
    """Fit source-specific IDF and persist sparse, bounded target shards."""
    if batch_size <= 0 or shard_rows <= 0 or not 0 < max_df_ratio <= 1:
        raise ValueError("Invalid index parameters")
    config = {"format": 1, "analyzer": "char_wb", "ngram_range": list(NGRAM_RANGE),
              "n_features": N_FEATURES, "alternate_sign": False,
              "max_df_ratio": max_df_ratio, "idf_scope": "target source",
              "shard_rows": shard_rows}
    manifest = index_dir / "manifest.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if all(saved.get(k) == v for k, v in config.items()) and all(
                _paths(index_dir, s) and (index_dir / f"idf_{s.lower()}.npy").exists()
                for s in ("S2", "S3")):
            return {"reused": True, **saved,
                    "disk_bytes": sum(p.stat().st_size for p in index_dir.iterdir() if p.is_file())}
        raise ValueError(f"Existing TF-IDF index does not match requested settings: {index_dir}")
    index_dir.mkdir(parents=True, exist_ok=True)
    if any(index_dir.iterdir()):
        raise ValueError(f"Index directory is nonempty without a valid manifest: {index_dir}")
    started = time.perf_counter()
    hv = _vectorizer()
    report: dict[str, object] = {"settings": config, "sources": {}}
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        for source in ("S2", "S3"):
            df = np.zeros(N_FEATURES, dtype=np.uint32)
            docs = cursor = 0
            while True:
                rows = list(db.execute("SELECT target,basic FROM target WHERE source=? AND target>? ORDER BY target LIMIT ?",
                                       (source, cursor, batch_size)))
                if not rows:
                    break
                cursor = rows[-1][0]
                texts = [r[1] for r in rows if r[1]]
                if texts:
                    counts = hv.transform(texts).tocsr()
                    df += np.asarray((counts > 0).sum(axis=0)).ravel().astype(np.uint32)
                    docs += len(texts)
            if not docs:
                raise ValueError(f"No nonblank normalized names found for {source}")
            max_df = max(1, math.floor(docs * max_df_ratio))
            idf = (np.log((docs + 1.0) / (df.astype(np.float64) + 1.0)) + 1.0).astype(np.float32)
            idf[df > max_df] = 0.0
            np.save(index_dir / f"idf_{source.lower()}.npy", idf)
            vector_started = time.perf_counter()
            cursor = shard = rows_total = indexed_names = 0
            while True:
                rows = list(db.execute("SELECT target,entity_id,country,basic FROM target WHERE source=? AND target>? ORDER BY target LIMIT ?",
                                       (source, cursor, shard_rows)))
                if not rows:
                    break
                cursor = rows[-1][0]
                ids = [r[1] for r in rows]
                countries = [normalize_country(r[2]) for r in rows]
                texts = [r[3] for r in rows]
                matrix = hv.transform(texts).tocsr()
                if matrix.nnz:
                    matrix.data *= idf[matrix.indices]
                    matrix.eliminate_zeros()
                    sparse_normalize(matrix, norm="l2", axis=1, copy=False)
                _store(index_dir / f"{source.lower()}_{shard:04d}.npz", matrix, ids, countries)
                rows_total += len(rows)
                indexed_names += sum(bool(text) for text in texts)
                shard += 1
            report["sources"][source] = {
                "target_rows": rows_total, "nonempty_names": docs, "indexed_names": indexed_names,
                "shards": shard, "max_df": max_df, "features_pruned_common": int((idf == 0).sum()),
                "vectorization_seconds": time.perf_counter() - vector_started}
        report["total_seconds"] = time.perf_counter() - started
        report["peak_rss_mb"] = _rss_mb()
    finally:
        db.close()
    (index_dir / "manifest.json").write_text(
        json.dumps({**config, **report}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["disk_bytes"] = sum(p.stat().st_size for p in index_dir.iterdir() if p.is_file())
    return report


def _query_matrix(names: list[str], idf: np.ndarray) -> csr_matrix:
    result = _vectorizer().transform(names).tocsr()
    if result.nnz:
        result.data *= idf[result.indices]
        result.eliminate_zeros()
        sparse_normalize(result, norm="l2", axis=1, copy=False)
    return result


def _rank_row(scores: csr_matrix, ids: np.ndarray, countries: np.ndarray,
              country: str, k: int) -> list[tuple[str, float, int]]:
    if scores.nnz == 0:
        return []
    idx, vals = scores.indices, scores.data
    keep = countries[idx] == country
    idx, vals = idx[keep], vals[keep]
    if not len(idx):
        return []
    if len(vals) > k:
        cutoff = np.partition(vals, len(vals) - k)[len(vals) - k]
        keep = vals >= cutoff
        idx, vals = idx[keep], vals[keep]
    ranked = sorted(((str(ids[i]), float(v)) for i, v in zip(idx, vals)),
                    key=lambda pair: (-pair[1], pair[0]))[:k]
    return [(entity_id, score, rank) for rank, (entity_id, score) in enumerate(ranked, 1)]


def retrieve_topk(index_dir: Path, s1_rows: Iterable[tuple[str, str, str, str]],
                  output_path: Path, max_k: int = 50, query_batch_size: int = 16) -> dict:
    """Retrieve separate source-specific top-K lists and write pairs in S1/ID order."""
    if max_k <= 0 or query_batch_size <= 0:
        raise ValueError("max_k and query_batch_size must be positive")
    rows = sorted(s1_rows, key=lambda r: r[0])
    s1_ids = [row[0] for row in rows]
    if len(s1_ids) != len(set(s1_ids)):
        raise ValueError("Duplicate S1 ID")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    vector_seconds = retrieval_seconds = 0.0
    pair_count = 0
    # The target source matrix is loaded once at a time; sparse target shards
    # and query batches bound construction memory, avoiding repeated disk reads.
    for source in ("S2", "S3"):
        target_started = time.perf_counter()
        shards = [_load(path) for path in _paths(index_dir, source)]
        if not shards:
            raise ValueError(f"No {source} matrix shards in {index_dir}")
        matrix = vstack([part[0] for part in shards], format="csr")
        ids = np.concatenate([part[1] for part in shards])
        countries = np.concatenate([part[2] for part in shards])
        if len(ids) != matrix.shape[0] or len(countries) != len(ids):
            raise ValueError(f"Corrupt {source} sparse index")
        del shards
        idf = np.load(index_dir / f"idf_{source.lower()}.npy", mmap_mode="r")
        partitions = {}
        for country in sorted(set(countries.tolist())):
            mask = countries == country
            part_matrix = matrix[mask].tocsr()
            part_ids = ids[mask].copy()
            part_countries = countries[mask].copy()
            partitions[country] = (part_matrix.T.tocsc(), part_ids, part_countries)
        del matrix, ids, countries
        with gzip.open(output_path.with_name(output_path.stem + f"_{source.lower()}.tsv.gz"),
                       "wt", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(TFIDF_HEADER)
            for start in range(0, len(rows), query_batch_size):
                batch = rows[start:start + query_batch_size]
                names = [normalize_name(row[1]) for row in batch]
                country_values = [normalize_country(row[3]) for row in batch]
                tick = time.perf_counter()
                queries = _query_matrix(names, idf)
                vector_seconds += time.perf_counter() - tick
                results_by_local: dict[int, list[tuple[str, float, int]]] = {}
                for country in sorted(set(country_values)):
                    partition = partitions.get(country)
                    if partition is None:
                        continue
                    selected = [i for i, value in enumerate(country_values) if value == country and queries.getrow(i).nnz]
                    if not selected:
                        continue
                    tick = time.perf_counter()
                    scored = (queries[selected] @ partition[0]).tocsr()
                    retrieval_seconds += time.perf_counter() - tick
                    for local, original in enumerate(selected):
                        results_by_local[original] = _rank_row(scored.getrow(local), partition[1], partition[2], country, max_k)
                for local, (s1_id, _, _, _) in enumerate(batch):
                    for entity_id, score, rank in sorted(results_by_local.get(local, []), key=lambda item: item[0]):
                        writer.writerow((s1_id, entity_id, source, f"{score:.8f}", rank))
                        pair_count += 1
                if (start + len(batch)) % 5000 == 0:
                    print(f"{source}: processed {start + len(batch):,}/{len(rows):,} S1", flush=True)
        del partitions
        print(f"{source} target load + scoring seconds: {time.perf_counter() - target_started:.1f}", flush=True)
    # Merge source outputs into one ordered metadata stream.
    s2_path = output_path.with_name(output_path.stem + "_s2.tsv.gz")
    s3_path = output_path.with_name(output_path.stem + "_s3.tsv.gz")
    temp_output = output_path.with_name(output_path.stem + "_merged.tsv.gz")
    with gzip.open(temp_output, "wt", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, delimiter="\t", lineterminator="\n")
        writer.writerow(TFIDF_HEADER)
        handles = [gzip.open(p, "rt", encoding="utf-8", newline="") for p in (s2_path, s3_path)]
        try:
            readers = [csv.reader(h, delimiter="\t") for h in handles]
            next(readers[0]); next(readers[1])
            pending = [next(r, None) for r in readers]
            while any(row is not None for row in pending):
                selected = min((row for row in pending if row is not None), key=lambda r: (r[0], r[1]))
                writer.writerow(selected)
                i = 0 if pending[0] == selected else 1
                pending[i] = next(readers[i], None)
        finally:
            for h in handles:
                h.close()
    s2_path.unlink(); s3_path.unlink()
    temp_output.replace(output_path)
    return {"s1_count": len(rows), "tfidf_pairs": pair_count, "max_k": max_k,
            "vectorization_seconds": vector_seconds, "retrieval_seconds": retrieval_seconds,
            "total_seconds": time.perf_counter() - started,
            "pairs_per_second": pair_count / max(retrieval_seconds, 1e-9),
            "peak_rss_mb": _rss_mb(), "output_bytes": output_path.stat().st_size}


def _read_candidate_row(reader, expected_id: str) -> set[str]:
    row = next(reader, None)
    if row is None or len(row) != 2 or row[0] != expected_id:
        raise ValueError(f"V1 candidate coverage/order mismatch at {expected_id}")
    values = row[1].split(",") if row[1] else []
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate V1 pair for {expected_id}")
    return set(values)


def union_candidates(s1_ids: list[str], v1_candidates: Path, v1_metadata: Path,
                     tfidf_pairs: Path, candidate_output: Path, metadata_output: Path,
                     k_s2: int, k_s3: int) -> dict:
    """Create V2=V1 union source-wise top-K TF-IDF; preserve V1 route flags."""
    candidate_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(v1_candidates, "rt", encoding="utf-8", newline="") as vc, \
         gzip.open(v1_metadata, "rt", encoding="utf-8", newline="") as vh, \
         gzip.open(tfidf_pairs, "rt", encoding="utf-8", newline="") as th, \
         gzip.open(candidate_output, "wt", encoding="utf-8", newline="") as ch, \
         gzip.open(metadata_output, "wt", encoding="utf-8", newline="") as mh:
        candidate_reader = csv.reader(vc, delimiter="\t")
        if next(candidate_reader, None) != CANDIDATE_HEADER:
            raise ValueError("Unexpected V1 candidate header")
        vreader = csv.DictReader(vh, delimiter="\t")
        treader = csv.DictReader(th, delimiter="\t")
        if vreader.fieldnames != METADATA_HEADER or treader.fieldnames != TFIDF_HEADER:
            raise ValueError("Unexpected V1 or TF-IDF metadata header")
        candidate_writer = csv.writer(ch, delimiter="\t", lineterminator="\n")
        metadata_writer = csv.writer(mh, delimiter="\t", lineterminator="\n")
        candidate_writer.writerow(CANDIDATE_HEADER)
        metadata_writer.writerow(V2_METADATA_HEADER)
        vrow, trow = next(vreader, None), next(treader, None)
        v1_pairs = tf_pairs = output_pairs = 0
        for s1_id in s1_ids:
            v1_ids = _read_candidate_row(candidate_reader, s1_id)
            by_id: dict[str, list[str]] = {}
            while vrow is not None and vrow["source1_entity_id"] == s1_id:
                target = vrow["candidate_entity_id"]
                if target in by_id:
                    raise ValueError(f"Duplicate V1 metadata pair: {s1_id} {target}")
                by_id[target] = [vrow[field] for field in METADATA_HEADER[2:]]
                v1_pairs += 1
                vrow = next(vreader, None)
            t_selected = []
            while trow is not None and trow["source1_entity_id"] == s1_id:
                limit = k_s2 if trow["target_source"] == "S2" else k_s3
                if int(trow["tfidf_rank"]) <= limit:
                    t_selected.append(trow)
                trow = next(treader, None)
            if set(by_id) != v1_ids:
                raise ValueError(f"V1 candidate/metadata mismatch for {s1_id}")
            for row in t_selected:
                target = row["candidate_entity_id"]
                tf_pairs += 1
                by_id.setdefault(target, ["0"] * len(METADATA_HEADER[2:]))
            ids = sorted(by_id)
            candidate_writer.writerow((s1_id, ",".join(ids)))
            rank_map = {row["candidate_entity_id"]: row for row in t_selected}
            for target in ids:
                routes = by_id[target]
                vsource = routes[0] if routes[0] in ("S2", "S3") else ""
                trow_for_id = rank_map.get(target)
                source = vsource or (trow_for_id["target_source"] if trow_for_id else "")
                if not source:
                    raise ValueError(f"Missing target source for {target}")
                writer_routes = routes if vsource else [source, *routes[1:]]
                tf = trow_for_id is not None
                metadata_writer.writerow((s1_id, target, *writer_routes, int(tf),
                                          trow_for_id["tfidf_score"] if tf else "",
                                          trow_for_id["tfidf_rank"] if tf else ""))
                output_pairs += 1
        if vrow is not None or trow is not None or next(candidate_reader, None) is not None:
            raise ValueError("Unexpected S1 rows after requested coverage")
    return {"v1_pairs": v1_pairs, "tfidf_pairs_selected": tf_pairs, "v2_pairs": output_pairs,
            "candidate_bytes": candidate_output.stat().st_size,
            "metadata_bytes": metadata_output.stat().st_size}



def validate_pair_artifacts(s1_ids: list[str], candidate_path: Path, metadata_path: Path) -> dict:
    """Check exact S1 coverage, unique pairs, source prefixes, and candidate/metadata equality."""
    with gzip.open(candidate_path, "rt", encoding="utf-8", newline="") as ch, \
         gzip.open(metadata_path, "rt", encoding="utf-8", newline="") as mh:
        candidates = csv.reader(ch, delimiter="\t")
        metadata = csv.DictReader(mh, delimiter="\t")
        if next(candidates, None) != CANDIDATE_HEADER or metadata.fieldnames != V2_METADATA_HEADER:
            raise ValueError("Unexpected V2 artifact schema")
        current = next(metadata, None)
        pairs = 0
        for s1_id in s1_ids:
            row = next(candidates, None)
            if row is None or len(row) != 2 or row[0] != s1_id:
                raise ValueError(f"Candidate S1 coverage mismatch at {s1_id}")
            predicted = parse_match_list(row[1])
            actual = set()
            previous = None
            while current is not None and current["source1_entity_id"] == s1_id:
                target = current["candidate_entity_id"]
                source = current["target_source"]
                if current["tfidf"] not in ("0", "1") or not target.startswith(source + "-"):
                    raise ValueError(f"Invalid target/source metadata pair: {s1_id} {target}")
                if previous is not None and target <= previous:
                    raise ValueError(f"Duplicate or unordered V2 metadata pair: {s1_id} {target}")
                previous = target
                actual.add(target)
                pairs += 1
                current = next(metadata, None)
            if predicted != actual:
                raise ValueError(f"Candidate list/metadata mismatch for {s1_id}")
        if next(candidates, None) is not None or current is not None:
            raise ValueError("Unexpected extra rows in V2 artifacts")
    return {"s1_rows": len(s1_ids), "candidate_pairs": pairs, "unique_pairs": pairs,
            "candidate_metadata_agree": True, "exact_s1_coverage": True}


def verify_v1_subset(v1_path: Path, v2_path: Path, s1_ids: list[str]) -> dict:
    """Assert every Phase 4 V1 candidate remains in the V2 union."""
    with gzip.open(v1_path, "rt", encoding="utf-8", newline="") as v1h, \
         gzip.open(v2_path, "rt", encoding="utf-8", newline="") as v2h:
        old_reader, new_reader = csv.reader(v1h, delimiter="\t"), csv.reader(v2h, delimiter="\t")
        if next(old_reader, None) != CANDIDATE_HEADER or next(new_reader, None) != CANDIDATE_HEADER:
            raise ValueError("Unexpected candidate header in subset check")
        pairs = 0
        for s1_id in s1_ids:
            old = next(old_reader, None)
            new = next(new_reader, None)
            if old is None or new is None or old[0] != s1_id or new[0] != s1_id:
                raise ValueError(f"S1 coverage mismatch in V1 subset check: {s1_id}")
            old_set = set(filter(None, old[1].split(",")))
            new_set = set(filter(None, new[1].split(",")))
            if not old_set <= new_set:
                raise ValueError(f"V1 candidate lost in V2 for {s1_id}")
            pairs += len(old_set)
        if next(old_reader, None) is not None or next(new_reader, None) is not None:
            raise ValueError("Extra candidate S1 rows in subset check")
    return {"v1_pairs_checked": pairs, "v1_subset_of_v2": True}

def run_sweep(dataset_root: Path, ids_path: Path, index_dir: Path,
              v1_candidates: Path, v1_metadata: Path, output_dir: Path,
              ks: tuple[int, ...] = (5, 10, 20, 30, 50)) -> dict:
    """Generate max-K once and evaluate each tune K using the shared diagnostics."""
    ids = load_id_file(ids_path)
    wanted = set(ids)
    rows = sorted(selected_source1(dataset_root / "train/train_source1.tsv", wanted))
    if len(rows) != len(ids):
        raise ValueError("Selected S1 IDs do not match training Source 1")
    output_dir.mkdir(parents=True, exist_ok=True)
    top_path = output_dir / "tune_tfidf_top50.tsv.gz"
    retrieval = retrieve_topk(index_dir, rows, top_path, max(ks))
    truth = load_ground_truth(dataset_root / "train/train_ground_truth.tsv", wanted)
    v1_path = output_dir / "tune_v1_metrics.json"
    from diagnostics import evaluate_candidates_only
    v1_metrics = evaluate_candidates_only(truth, v1_candidates)
    report = {"s1_count": len(ids), "retrieval": retrieval, "v1": v1_metrics, "sweep": []}
    for k in ks:
        cand = output_dir / f".sweep_k{k}.tsv.gz"
        meta = output_dir / f".sweep_k{k}_metadata.tsv.gz"
        union_start = time.perf_counter()
        union = union_candidates(ids, v1_candidates, v1_metadata, top_path, cand, meta, k, k)
        union["artifact_checks"] = validate_pair_artifacts(ids, cand, meta)
        union["union_seconds"] = time.perf_counter() - union_start
        metrics = evaluate_candidates_only(truth, cand)
        row = {"k": k, **metrics, "candidate_pairs_growth": metrics["total_candidate_pairs"] - v1_metrics["total_candidate_pairs"],
               "candidate_pairs_growth_pct": 100 * (metrics["total_candidate_pairs"] / v1_metrics["total_candidate_pairs"] - 1),
               "v1_misses_recovered": v1_metrics["blocking_misses"] - metrics["blocking_misses"],
               "s2_v1_misses_recovered": v1_metrics["s2_blocking_misses"] - metrics["s2_blocking_misses"],
               "s3_v1_misses_recovered": v1_metrics["s3_blocking_misses"] - metrics["s3_blocking_misses"],
               **union}
        row["runtime_seconds"] = retrieval["total_seconds"] + union["union_seconds"]
        row["peak_rss_mb"] = _rss_mb()
        report["sweep"].append(row)
        cand.unlink(); meta.unlink()
    for previous, current in zip(report["sweep"], report["sweep"][1:]):
        current["marginal_recovered_links"] = previous["blocking_misses"] - current["blocking_misses"]
        current["marginal_candidate_pairs"] = current["total_candidate_pairs"] - previous["total_candidate_pairs"]
    report["runtime_seconds"] = retrieval["total_seconds"]
    report["peak_rss_mb"] = _rss_mb()
    v1_path.write_text(json.dumps(v1_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "tune_sweep.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report



def _copy_subset(path: Path, output: Path, wanted: set[str], expected_header: list[str]) -> None:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as src, gzip.open(output, "wt", encoding="utf-8", newline="") as dst:
        reader = csv.reader(src, delimiter="\t")
        header = next(reader, None)
        if header != expected_header:
            raise ValueError(f"Unexpected header in {path}")
        writer = csv.writer(dst, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        for row in reader:
            if row and row[0] in wanted:
                writer.writerow(row)


def benchmark_prefix(dataset_root: Path, tune_ids_path: Path, index_dir: Path,
                     v1_candidates: Path, v1_metadata: Path, output_dir: Path,
                     prefix_count: int = 1000) -> dict:
    """Measure retrieval and union throughput on a deterministic tune-ID prefix."""
    ids = load_id_file(tune_ids_path)[:prefix_count]
    if len(ids) != prefix_count:
        raise ValueError("Tune ID file is smaller than requested benchmark prefix")
    wanted = set(ids)
    rows = sorted(selected_source1(dataset_root / "train/train_source1.tsv", wanted))
    if len(rows) != len(ids):
        raise ValueError("Benchmark S1 rows do not match selected IDs")
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    top = output_dir / "benchmark_tfidf_top50.tsv.gz"
    retrieval = retrieve_topk(index_dir, rows, top, max_k=50)
    retrieval["elapsed_wall_seconds"] = time.perf_counter() - start
    temp_v1 = output_dir / "benchmark_v1_candidates.tsv.gz"
    temp_meta = output_dir / "benchmark_v1_metadata.tsv.gz"
    _copy_subset(v1_candidates, temp_v1, wanted, CANDIDATE_HEADER)
    _copy_subset(v1_metadata, temp_meta, wanted, METADATA_HEADER)
    cand, meta = output_dir / "benchmark_v2_candidates.tsv.gz", output_dir / "benchmark_v2_metadata.tsv.gz"
    union_start = time.perf_counter()
    union = union_candidates(ids, temp_v1, temp_meta, top, cand, meta, 50, 50)
    union["union_seconds"] = time.perf_counter() - union_start
    union["disk_growth_bytes"] = sum(p.stat().st_size for p in output_dir.iterdir() if p.is_file())
    report = {"prefix_count": prefix_count, "retrieval": retrieval, "union": union,
              "peak_rss_mb": _rss_mb()}
    (output_dir / "benchmark_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def write_miss_review(dataset_root: Path, ids: list[str], truth: dict[str, set[str]],
                      v1_candidates: Path, v2_candidates: Path, v2_metadata: Path,
                      db_path: Path, output_path: Path, per_source_category: int = 50) -> dict:
    """Write bounded deterministic samples of TF-IDF recoveries and remaining misses."""
    wanted = set(ids)
    s1 = {row[0]: row[1:] for row in selected_source1(dataset_root / "train/train_source1.tsv", wanted)}
    samples: dict[tuple[str, str], list[tuple[str, str]]] = {}
    recoveries: dict[str, list[tuple[str, str]]] = {"S2": [], "S3": []}
    remaining: dict[str, list[tuple[str, str]]] = {"S2": [], "S3": []}
    with gzip.open(v1_candidates, "rt", encoding="utf-8", newline="") as v1h, \
         gzip.open(v2_candidates, "rt", encoding="utf-8", newline="") as v2h:
        v1r, v2r = csv.reader(v1h, delimiter="\t"), csv.reader(v2h, delimiter="\t")
        if next(v1r, None) != CANDIDATE_HEADER or next(v2r, None) != CANDIDATE_HEADER:
            raise ValueError("Unexpected candidate file header for miss review")
        for s1_id in ids:
            row1, row2 = next(v1r, None), next(v2r, None)
            if row1 is None or row2 is None or row1[0] != s1_id or row2[0] != s1_id:
                raise ValueError(f"Candidate row coverage mismatch during miss review: {s1_id}")
            old = set(filter(None, row1[1].split(",")))
            new = set(filter(None, row2[1].split(",")))
            if not old <= new:
                raise ValueError("V1 candidate not present in V2 during miss review")
            for target in sorted(truth[s1_id] - old):
                source = target[:2]
                if source not in recoveries:
                    continue
                if target in new:
                    if len(recoveries[source]) < per_source_category:
                        recoveries[source].append((s1_id, target))
                elif len(remaining[source]) < per_source_category:
                    remaining[source].append((s1_id, target))
        if next(v1r, None) is not None or next(v2r, None) is not None:
            raise ValueError("Extra candidate rows during miss review")
    chosen = {pair for values in (*recoveries.values(), *remaining.values()) for pair in values}
    tf_meta: dict[tuple[str, str], dict[str, str]] = {}
    with gzip.open(v2_metadata, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != V2_METADATA_HEADER:
            raise ValueError("Unexpected V2 metadata header")
        for row in reader:
            pair = (row["source1_entity_id"], row["candidate_entity_id"])
            if pair in chosen:
                tf_meta[pair] = row
    import sqlite3 as _sqlite3
    db = _sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        targets = {}
        for _, target in sorted(chosen):
            row = db.execute("SELECT entity_id,source,name,address,country FROM target WHERE entity_id=?", (target,)).fetchone()
            if row is None:
                raise ValueError(f"Unknown target ID in miss sample: {target}")
            targets[target] = row
    finally:
        db.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["category", "source1_entity_id", "true_target_id", "target_source",
                         "source1_name", "target_name", "source1_address", "target_address",
                         "source1_country", "target_country", "tfidf", "tfidf_score", "tfidf_rank",
                         *METADATA_HEADER[3:]])
        for category, groups in (("RECOVERED_V1_MISS", recoveries), ("STILL_MISSED_V2", remaining)):
            for source in ("S2", "S3"):
                for s1_id, target in groups[source]:
                    target_row = targets[target]
                    s1_name, s1_address, s1_country = s1[s1_id]
                    metadata = tf_meta.get((s1_id, target), {})
                    route_values = [metadata.get(field, "") for field in METADATA_HEADER[3:]]
                    writer.writerow((category, s1_id, target, source, s1_name, target_row[2],
                                     s1_address, target_row[3], s1_country, target_row[4],
                                     metadata.get("tfidf", ""), metadata.get("tfidf_score", ""),
                                     metadata.get("tfidf_rank", ""), *route_values))
    return {"recovered_sample_s2": len(recoveries["S2"]), "recovered_sample_s3": len(recoveries["S3"]),
            "remaining_sample_s2": len(remaining["S2"]), "remaining_sample_s3": len(remaining["S3"]),
            "sample_path": str(output_path)}


def freeze_tune_configuration(sweep_path: Path, tune_ids_path: Path,
                              v1_candidates: Path, v1_metadata: Path,
                              config_path: Path, candidate_output: Path,
                              metadata_output: Path, k_s2: int, k_s3: int,
                              experiments_path: Path) -> dict:
    """Freeze explicit tune-selected K values and materialize the selected tune union."""
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    ids = load_id_file(tune_ids_path)
    rows = sweep.get("sweep", [])
    by_k = {int(row["k"]): row for row in rows}
    if k_s2 != k_s3:
        raise ValueError("Phase 6 selection uses the shared K sweep; select one K for both sources")
    if k_s2 not in by_k:
        raise ValueError("Selected K value must appear in the completed tune sweep")
    top = sweep_path.parent / "tune_tfidf_top50.tsv.gz"
    union = union_candidates(ids, v1_candidates, v1_metadata, top,
                              candidate_output, metadata_output, k_s2, k_s3)
    union["artifact_checks"] = validate_pair_artifacts(ids, candidate_output, metadata_output)
    union["v1_subset_check"] = verify_v1_subset(v1_candidates, candidate_output, ids)
    chosen = by_k[k_s2] if k_s2 == k_s3 else None
    config = {
        "phase": 6, "name": "tfidf-charwb-v2", "analyzer": "char_wb",
        "ngram_range": list(NGRAM_RANGE), "n_features": N_FEATURES,
        "max_df_ratio": MAX_DF_RATIO, "idf_scope": "target source",
        "country_partition": "normalized exact string, open set",
        "k_s2": k_s2, "k_s3": k_s3,
        "ranking": "cosine score descending, then target ID ascending",
        "union": "all V1 pairs plus top-K TF-IDF pairs independently per source",
        "tune_ids": str(tune_ids_path), "tune_s1_count": len(ids),
        "selection_metrics_s2_k": by_k[k_s2], "selection_metrics_s3_k": by_k[k_s3],
        "selected_union": union,
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    from diagnostics import append_candidate_experiment
    for krow in rows:
        k = int(krow["k"])
        append_candidate_experiment(experiments_path, f"phase6-tfidf-k{k}",
            f"Tune-only char_wb TF-IDF; shared K={k}; sparse top-K plus frozen V1",
            krow, float(krow.get("union_seconds", 0.0)) + float(sweep["retrieval"].get("total_seconds", 0.0)),
            float(sweep.get("peak_rss_mb", 0.0)) or None)
    selected_metrics = by_k[k_s2]
    append_candidate_experiment(experiments_path, "phase6-tfidf-v2-tune-selected",
        f"Frozen selected K values S2={k_s2}, S3={k_s3}; see {config_path.name}",
        selected_metrics, float(sweep["retrieval"].get("total_seconds", 0.0)),
        float(sweep.get("peak_rss_mb", 0.0)) or None)
    return config

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-index")
    build.add_argument("--sqlite", type=Path, default=BLOCKING / "v1_index.sqlite")
    build.add_argument("--index-dir", type=Path, default=ARTIFACTS / "tfidf_v2_index")
    tune = commands.add_parser("tune")
    tune.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    tune.add_argument("--s1-ids", type=Path, default=ARTIFACTS / "splits/tune_s1_ids.txt")
    tune.add_argument("--index-dir", type=Path, default=ARTIFACTS / "tfidf_v2_index")
    tune.add_argument("--v1-candidates", type=Path, default=ARTIFACTS / "baseline/tune_candidates.tsv.gz")
    tune.add_argument("--v1-metadata", type=Path, default=ARTIFACTS / "baseline/tune_metadata.tsv.gz")
    tune.add_argument("--output-dir", type=Path, default=ARTIFACTS / "tfidf_v2_tune")
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--sweep", type=Path, default=ARTIFACTS / "tfidf_v2_tune/tune_sweep.json")
    freeze.add_argument("--s1-ids", type=Path, default=ARTIFACTS / "splits/tune_s1_ids.txt")
    freeze.add_argument("--v1-candidates", type=Path, default=ARTIFACTS / "baseline/tune_candidates.tsv.gz")
    freeze.add_argument("--v1-metadata", type=Path, default=ARTIFACTS / "baseline/tune_metadata.tsv.gz")
    freeze.add_argument("--config", type=Path, default=BLOCKING / "v2_config.json")
    freeze.add_argument("--candidate-output", type=Path, default=ARTIFACTS / "tfidf_v2_tune/tune_v2_candidates.tsv.gz")
    freeze.add_argument("--metadata-output", type=Path, default=ARTIFACTS / "tfidf_v2_tune/tune_v2_metadata.tsv.gz")
    freeze.add_argument("--experiments", type=Path, default=ARTIFACTS / "experiments.csv")
    freeze.add_argument("--k", type=int, required=True)

    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    benchmark.add_argument("--s1-ids", type=Path, default=ARTIFACTS / "splits/tune_s1_ids.txt")
    benchmark.add_argument("--index-dir", type=Path, default=ARTIFACTS / "tfidf_v2_index")
    benchmark.add_argument("--v1-candidates", type=Path, default=ARTIFACTS / "baseline/tune_candidates.tsv.gz")
    benchmark.add_argument("--v1-metadata", type=Path, default=ARTIFACTS / "baseline/tune_metadata.tsv.gz")
    benchmark.add_argument("--output-dir", type=Path, default=ARTIFACTS / "tfidf_v2_benchmark")
    benchmark.add_argument("--prefix-count", type=int, default=1000)

    validate = commands.add_parser("validate")
    validate.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    validate.add_argument("--s1-ids", type=Path, default=ARTIFACTS / "splits/val_s1_ids.txt")
    validate.add_argument("--index-dir", type=Path, default=ARTIFACTS / "tfidf_v2_index")
    validate.add_argument("--v1-candidates", type=Path, default=BLOCKING / "v1_validation_candidates.tsv.gz")
    validate.add_argument("--v1-metadata", type=Path, default=BLOCKING / "v1_validation_metadata.tsv.gz")
    validate.add_argument("--output-dir", type=Path, default=ARTIFACTS / "blocking")
    validate.add_argument("--config", type=Path, default=BLOCKING / "v2_config.json")
    validate.add_argument("--k-s2", type=int)
    validate.add_argument("--k-s3", type=int)
    args = parser.parse_args(argv)
    try:
        started = time.perf_counter()
        if args.command == "build-index":
            result = build_index(args.sqlite, args.index_dir)
            (args.index_dir / "build_report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        elif args.command == "tune":
            result = run_sweep(args.dataset_root, args.s1_ids, args.index_dir, args.v1_candidates,
                               args.v1_metadata, args.output_dir)
        elif args.command == "benchmark":
            result = benchmark_prefix(args.dataset_root, args.s1_ids, args.index_dir,
                                      args.v1_candidates, args.v1_metadata, args.output_dir,
                                      args.prefix_count)
        elif args.command == "freeze":
            result = freeze_tune_configuration(args.sweep, args.s1_ids, args.v1_candidates,
                args.v1_metadata, args.config, args.candidate_output, args.metadata_output,
                args.k, args.k, args.experiments)
        else:
            ids = load_id_file(args.s1_ids)
            frozen = json.loads(args.config.read_text(encoding="utf-8"))
            k_s2 = args.k_s2 if args.k_s2 is not None else int(frozen["k_s2"])
            k_s3 = args.k_s3 if args.k_s3 is not None else int(frozen["k_s3"])
            if k_s2 != int(frozen["k_s2"]) or k_s3 != int(frozen["k_s3"]):
                raise ValueError("Validation K values must match the frozen tune configuration")
            rows = sorted(selected_source1(args.dataset_root / "train/train_source1.tsv", set(ids)))
            tf_path = args.output_dir / "v2_validation_tfidf_top50.tsv.gz"
            retrieval = retrieve_topk(args.index_dir, rows, tf_path, max(k_s2, k_s3))
            candidate_path = args.output_dir / "v2_validation_candidates.tsv.gz"
            metadata_path = args.output_dir / "v2_validation_metadata.tsv.gz"
            union = union_candidates(ids, args.v1_candidates, args.v1_metadata, tf_path,
                                     candidate_path, metadata_path, k_s2, k_s3)
            union["artifact_checks"] = validate_pair_artifacts(ids, candidate_path, metadata_path)
            union["v1_subset_check"] = verify_v1_subset(args.v1_candidates, candidate_path, ids)
            truth = load_ground_truth(args.dataset_root / "train/train_ground_truth.tsv", set(ids))
            metrics = evaluate_candidates_only(truth, candidate_path)
            v1 = evaluate_candidates_only(truth, args.v1_candidates)
            comparison = compare_candidate_files(truth, args.v1_candidates, candidate_path)
            result = {"retrieval": retrieval, "union": union, "v1": v1, "v2": metrics,
                      "candidate_comparison": comparison, "k_s2": k_s2, "k_s3": k_s3}
            diagnostics_dir = ARTIFACTS / "diagnostics"
            diagnostics_dir.mkdir(parents=True, exist_ok=True)
            combined_report = {**metrics, **comparison}
            (diagnostics_dir / "v2_validation_candidates.txt").write_text(render_candidate_only_report(combined_report), encoding="utf-8")
            result["miss_review"] = write_miss_review(args.dataset_root, ids, truth, args.v1_candidates,
                candidate_path, metadata_path, BLOCKING / "v1_index.sqlite",
                diagnostics_dir / "v2_blocking_miss_review.tsv.gz")
            run_seconds = time.perf_counter() - started
            result["total_runtime_seconds"] = run_seconds
            result["peak_rss_mb"] = _rss_mb()
            append_candidate_experiment(ARTIFACTS / "experiments.csv", "phase6-tfidf-v2-validation",
                f"Frozen tune-selected char_wb TF-IDF union; K S2={k_s2}, S3={k_s3}",
                metrics, run_seconds, result["peak_rss_mb"])
            (args.output_dir / "v2_validation_report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, sqlite3.Error, csv.Error) as exc:
        parser.exit(1, f"FAIL: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
