#!/usr/bin/env python3
"""Phase 11.3 contextual reranker.

This module deliberately works after Phase 8 scoring.  It never creates
retrieval candidates or touches Phase 12 score shards.  Phase 7 OOF scores are
the only new base scores it can create, and are produced with S1-grouped folds.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import resource
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import lightgbm as lgb
import numpy as np

from baseline import TargetStore, load_s1_records
from diagnostics import DEFAULT_DATASET_ROOT
from phase8_model import (FEATURE_NAMES, TARGET_INDEX, extract_features,
                          iter_union_groups, union_candidate_counts)
from phase9_policy import (DecisionPolicy, ScoreCache, _apply_conflicts,
                           evaluate_policy, load_cache, read_score_file,
                           save_cache)
from phase112_group_policy import GroupPolicy, predictions_for_group_policy
from scoring import load_ground_truth, load_id_file, score_predictions

BASE = Path(__file__).resolve().parents[1]
ART = BASE / "artifacts"
OUT = ART / "phase113_reranker"
PHASE7, PHASE8, PHASE9 = ART / "model_data/phase7", ART / "model/phase8", ART / "model/phase9"
PHASE112, PHASE12, VAL, SPLITS = ART / "phase112_group_policy", ART / "test_inference/phase12", ART / "validation_evaluation", ART / "splits"
ROOT = BASE.parents[1]
DATASET = DEFAULT_DATASET_ROOT
TEST_S1 = DATASET / "test/test_source1.tsv"
GROUND_TRUTH = DATASET / "train/train_ground_truth.tsv"
TRAIN_IDS, TUNE_IDS, VAL_IDS = PHASE7 / "train_subset_ids.txt", SPLITS / "tune_s1_ids.txt", SPLITS / "val_s1_ids.txt"
TUNE_CACHE = {s: PHASE9 / f"score_cache_{s.lower()}.npz" for s in ("S2", "S3")}
VAL_SCORE = {s: VAL / f"validation_scores_{s.lower()}.tsv.gz" for s in ("S2", "S3")}
POLICY_PATH = PHASE9 / "decision_policy.json"
SEED, FOLDS, THREADS, RESERVE = 20260925, 5, 4, 4 * 1024**3
BASELINE_MACRO, CHAMPION_TUNE, CHAMPION_VALIDATION = .8676284617025911, .871071, .8731357909723747
EXPECTED_TUNE, EXPECTED_VAL = {"S2": 7_227_686, "S3": 7_323_701}, {"S2": 16_098_984, "S3": 16_233_279}
CHAMPION_SHA = "b31fda1d88b53cd6123eb5456181e57d0d71bb2435c581158cc63a5af4035d55"
R3_VALIDATION_MACRO = .9083920932905012
R3_TEST_THRESHOLD = .795
BASE_POLICY = DecisionPolicy(.93, .97, None, "highest", None)
COUNT_BINS = ((1, 25, "1-25"), (26, 100, "26-100"), (101, 500, "101-500"), (501, 10**9, "501+"))
OOF_HEADER = ("source1_entity_id", "candidate_entity_id", "target_source", "base_score", "from_v1", "from_address", "address_rank", "address_score")

FROZEN = (PHASE8 / "model_s2.txt", PHASE8 / "model_s3.txt", PHASE8 / "feature_manifest.json", POLICY_PATH,
          TUNE_CACHE["S2"], TUNE_CACHE["S3"], VAL_SCORE["S2"], VAL_SCORE["S3"],
          PHASE112 / "locked_winner.json", PHASE112 / "validation_confirmation.json",
          PHASE12 / "candidates/candidate_pairs_long.tsv.gz", PHASE12 / "candidates/candidate_summary.json",
          PHASE12 / "scores/score_manifest.json", ROOT / "output/candidate_pairs.tsv",
          ROOT / "output/matching_results.tsv", ROOT / "output/phase112/matching_results.tsv",
          PHASE7 / "train_subset_ids.txt", PHASE7 / "train_pairs_s2.tsv.gz", PHASE7 / "train_pairs_s3.tsv.gz",
          PHASE7 / "v1_candidates.tsv.gz", PHASE7 / "v1_metadata.tsv.gz", PHASE7 / "address_k10_candidates.tsv.gz")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot() -> dict[str, dict[str, object]]:
    out = {}
    for p in FROZEN:
        if not p.exists(): out[str(p)] = {"exists": False}; continue
        st = p.stat(); row: dict[str, object] = {"exists": True, "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}
        if st.st_size < 100_000_000: row["sha256"] = sha256(p)
        out[str(p)] = row
    return out


def assert_unchanged(before: Mapping[str, Mapping[str, object]]) -> None:
    if dict(before) != snapshot(): raise RuntimeError("a frozen Phase 8/9/11.2/12 input changed")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"); tmp.replace(path)


def rss_mb() -> float: return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
def fold_of(s1: str, folds: int = FOLDS) -> int: return int.from_bytes(hashlib.sha256(s1.encode()).digest()[:8], "big") % folds
def stable_hash(*parts: str) -> str: return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()
def count_bin(n: int) -> str:
    for lo, hi, name in COUNT_BINS:
        if lo <= n <= hi: return name
    return "0"
def base_threshold(source: str) -> float: return .93 if source == "S2" else .97
def a_source_threshold(source: str, n: int) -> float: return base_threshold(source) + (-.015 if n <= 100 else .015) + (-.005 if source == "S2" else 0.0)


def require_space(needed: int = 0) -> None:
    free = os.statvfs(OUT.parent).f_bavail * os.statvfs(OUT.parent).f_frsize
    if free - needed < RESERVE: raise RuntimeError(f"insufficient disk: free={free}, reserve={RESERVE}, needed={needed}")


def frozen_policy() -> DecisionPolicy:
    raw = json.loads(POLICY_PATH.read_text())["policy"]
    p = DecisionPolicy(float(raw["s2_threshold"]), float(raw["s3_threshold"]), raw.get("open_threshold"), raw["conflict_policy"], raw.get("conflict_margin"))
    if p != BASE_POLICY: raise RuntimeError("frozen Phase 9 policy is not .93/.97 highest-owner")
    return p


def champion_policy() -> GroupPolicy:
    lock = json.loads((PHASE112 / "locked_winner.json").read_text())
    conf = json.loads((PHASE112 / "validation_confirmation.json").read_text())
    expected = {"kind": "A_source", "params": {"base_low_offset": -.015, "base_crowded_offset": .015, "source_offsets": {"S2": -.005, "S3": 0.0}}}
    if lock.get("experiment_id") != "A_source_02" or lock.get("policy") != expected: raise RuntimeError("Phase 11.2 lock is not A_source_02")
    if conf.get("status") != "CONFIRMED" or abs(float(conf.get("delta_vs_baseline", 0)) - .003130001345525324) > 1e-12: raise RuntimeError("Phase 11.2 validation confirmation mismatch")
    if sha256(ROOT / "output/phase112/matching_results.tsv") != CHAMPION_SHA: raise RuntimeError("Phase 11.2 champion output hash mismatch")
    return GroupPolicy("A_source", expected["params"])


def ids() -> tuple[list[str], list[str], list[str]]:
    train, tune, val = load_id_file(TRAIN_IDS), load_id_file(TUNE_IDS), load_id_file(VAL_IDS)
    if len(train) != 100_000 or len(set(train)) != len(train) or set(train) & set(tune) or set(train) & set(val) or set(tune) & set(val):
        raise RuntimeError("invalid Phase 7/tune/validation S1 separation")
    return train, tune, val


def tune_inputs() -> tuple[ScoreCache, ScoreCache, dict[str, set[str]]]:
    _, tune, _ = ids(); s2, s3 = load_cache(TUNE_CACHE["S2"]), load_cache(TUNE_CACHE["S3"])
    if s2.s1_ids != tuple(tune) or s3.s1_ids != tuple(tune) or len(s2.targets) != EXPECTED_TUNE["S2"] or len(s3.targets) != EXPECTED_TUNE["S3"]: raise RuntimeError("frozen tune score cache mismatch")
    return s2, s3, load_ground_truth(GROUND_TRUTH, tune)


def val_inputs() -> tuple[ScoreCache, ScoreCache, dict[str, set[str]]]:
    _, _, val = ids(); s2, s3 = (read_score_file(VAL_SCORE[s], s, val, EXPECTED_VAL[s]) for s in ("S2", "S3"))
    return s2, s3, load_ground_truth(GROUND_TRUTH, val)


def inventory(output_dir: Path = OUT) -> dict:
    before = snapshot(); train, tune, val = ids(); champion_policy(); frozen_policy()
    oof = [p for p in (output_dir / "oof").glob("*.tsv.gz")] if (output_dir / "oof").exists() else []
    value = {"phase": "11.3", "train_s1": len(train), "tune_s1": len(tune), "validation_s1": len(val),
             "phase7_full_pairs": 14_597_845, "phase7_sampled_pairs": 1_522_861, "oof_reusable": False,
             "oof_files": [str(p) for p in oof], "free_disk_bytes": os.statvfs(OUT.parent).f_bavail * os.statvfs(OUT.parent).f_frsize,
             "disk_reserve_bytes": RESERVE, "frozen_snapshot": before, "champion": "A_source_02", "peak_rss_mb": rss_mb()}
    atomic_json(output_dir / "phase113_inventory.json", value); assert_unchanged(before); return value


def baseline_check(output_dir: Path = OUT) -> dict:
    before = snapshot(); s2, s3, truth = tune_inputs(); frozen = evaluate_policy(s2, s3, truth, BASE_POLICY)
    if frozen["macro_f0_5"] != BASELINE_MACRO: raise RuntimeError("frozen Phase 9 baseline did not reproduce")
    champion = predictions_for_group_policy(s2, s3, champion_policy()); cscore = score_predictions(truth, champion, s2.s1_ids)
    if abs(cscore.macro_f0_5 - CHAMPION_TUNE) > .0005: raise RuntimeError("A_source_02 tune champion did not reproduce")
    value = {"baseline_macro_f0_5": frozen["macro_f0_5"], "champion_tune_macro_f0_5": cscore.macro_f0_5,
             "champion_validation_reference": CHAMPION_VALIDATION, "frozen_snapshot": before, "frozen_artifacts_unchanged": True}
    atomic_json(output_dir / "baseline_reproduction.json", value); assert_unchanged(before); return value


CONTEXT_NAMES = ("base_score", "base_logit", "source_s3", "candidate_count_source", "candidate_count_total", "log1p_count_source", "log1p_count_total", "rank_source", "rank_total", "rank_fraction_source", "top_score_source", "second_score_source", "third_score_source", "top_minus_second_source", "top_minus_current_source", "current_minus_next", "current_minus_previous", "count_ge_050", "count_ge_080", "count_ge_090", "count_ge_093", "count_ge_095", "count_ge_097", "count_ge_099", "score_minus_mean", "score_minus_median", "score_z", "passes_baseline", "passes_a_source", "bin_1_25", "bin_26_100", "bin_101_500", "bin_501_plus", "from_v1", "from_address", "from_both", "address_rank")


def _context(source: str, scores: np.ndarray, candidate_ids: list[str], total_scores: np.ndarray,
             total_ids: list[str], evidence: list[Mapping[str, object]]) -> np.ndarray:
    n, total = len(scores), len(total_scores)
    order = np.lexsort((np.asarray(candidate_ids), -scores)); ranks = np.empty(n, dtype=np.int32); ranks[order] = np.arange(1, n + 1)
    total_order = np.lexsort((np.asarray(total_ids), -total_scores)); total_rank = np.empty(total, dtype=np.int32); total_rank[total_order] = np.arange(1, total + 1)
    total_rank_by_id = {candidate_id: int(total_rank[i]) for i, candidate_id in enumerate(total_ids)}
    top = np.pad(scores[order][:3], (0, max(0, 3 - n)), constant_values=0); mean, med, sd = float(scores.mean()) if n else 0., float(np.median(scores)) if n else 0., float(scores.std()) if n else 0.
    density = [int((scores >= t).sum()) for t in (.50, .80, .90, .93, .95, .97, .99)]; out = np.zeros((n, len(CONTEXT_NAMES)), dtype=np.float32)
    for i, score in enumerate(scores):
        pos = int(ranks[i]) - 1; prev = scores[order[pos - 1]] if pos else score; nxt = scores[order[pos + 1]] if pos + 1 < n else score
        b = count_bin(n); e = evidence[i]; vals = [score, math.log(max(float(score), 1e-6) / max(1 - float(score), 1e-6)), source == "S3", n, total, math.log1p(n), math.log1p(total), ranks[i], total_rank_by_id[candidate_ids[i]], ranks[i] / max(n, 1), top[0], top[1], top[2], top[0] - top[1], top[0] - score, score - nxt, score - prev, *density, score - mean, score - med, (score - mean) / sd if sd > 1e-6 else 0., score >= base_threshold(source), score >= a_source_threshold(source, n), b == "1-25", b == "26-100", b == "101-500", b == "501+", int(e.get("from_v1", 0)), int(e.get("from_address", 0)), int(e.get("from_v1", 0)) and int(e.get("from_address", 0)), float(e.get("address_rank", 0) or 0)]
        out[i] = vals
    return out


def contextual_group(s2_scores: np.ndarray, s3_scores: np.ndarray, s2_ids: list[str], s3_ids: list[str],
                     s2_evidence: list[Mapping[str, object]], s3_evidence: list[Mapping[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    all_scores = np.concatenate((s2_scores, s3_scores)); all_ids = s2_ids + s3_ids
    return (_context("S2", s2_scores, s2_ids, all_scores, all_ids, s2_evidence),
            _context("S3", s3_scores, s3_ids, all_scores, all_ids, s3_evidence))


def sampling_indices(labels: np.ndarray, scores: np.ndarray, ranks: np.ndarray, crowd: bool, s1: str, candidates: list[str]) -> np.ndarray:
    keep = set(np.flatnonzero(labels == 1).tolist()); buckets = ((.90, 1.01, 6), (.75, .90, 4), (.50, .75, 3), (-.01, .50, 2))
    negatives = np.flatnonzero(labels == 0)
    for low, high, cap in buckets:
        rows = [int(i) for i in negatives if low <= scores[i] < high]
        rows.sort(key=lambda i: (-float(scores[i]), int(ranks[i]), stable_hash(str(SEED), s1, candidates[i])))
        keep.update(rows[:cap])
    extra = [int(i) for i in negatives if ranks[i] <= 3 or crowd]
    extra.sort(key=lambda i: (int(ranks[i]), -float(scores[i]), stable_hash(str(SEED), s1, candidates[i])))
    keep.update(extra[:3 if crowd else 1])
    positives = np.flatnonzero(labels == 1).tolist()
    selected_negatives = sorted(keep.difference(positives), key=lambda i: (-float(scores[i]), int(ranks[i]), stable_hash(str(SEED), s1, candidates[i])))[:20]
    return np.asarray(sorted(positives + selected_negatives), dtype=np.int64)


def _phase7_pair_rows(source: str) -> Iterator[dict[str, str]]:
    with gzip.open(PHASE7 / f"train_pairs_{source.lower()}.tsv.gz", "rt", newline="", encoding="utf-8") as h:
        for r in csv.DictReader(h, delimiter="\t"): yield r


def _sample_features(output_dir: Path, source: str, records: Mapping[str, Mapping[str, str]], counts: Mapping[str, Mapping[str, int]]) -> dict[str, np.ndarray]:
    path = output_dir / "oof" / f"phase7_sample_{source.lower()}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as z: return {k: z[k] for k in z.files}
    xs, ys, s1s = [], [], []; store = TargetStore(TARGET_INDEX); batch = []; batch_size = 25_000
    try:
        def consume(rows):
            targets = store.get_many(sorted({r["candidate_entity_id"] for r in rows}))
            feats = np.vstack([extract_features(records[r["source1_entity_id"]], targets[r["candidate_entity_id"]], r, counts[r["source1_entity_id"]][source]) for r in rows]).astype(np.float32)
            xs.append(feats); ys.append(np.asarray([int(r["label"]) for r in rows], np.int8)); s1s.append(np.asarray([r["source1_entity_id"] for r in rows], "S20"))
        for row in _phase7_pair_rows(source):
            batch.append(row)
            if len(batch) >= batch_size: consume(batch); batch=[]
        if batch: consume(batch)
    finally: store.close()
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(".npz.tmp")
    with tmp.open("wb") as handle: np.savez_compressed(handle, features=np.vstack(xs), labels=np.concatenate(ys), s1=np.concatenate(s1s), feature_names=np.asarray(FEATURE_NAMES))
    tmp.replace(path)
    with np.load(path, allow_pickle=False) as z: return {k: z[k] for k in z.files}


def _base_params() -> dict: return {"objective": "binary", "metric": "binary_logloss", "learning_rate": .03, "num_leaves": 95, "min_data_in_leaf": 150, "feature_fraction": .9, "bagging_fraction": .85, "bagging_freq": 1, "lambda_l2": 1., "seed": SEED, "feature_fraction_seed": SEED, "bagging_seed": SEED, "deterministic": True, "force_col_wise": True, "num_threads": THREADS, "verbosity": -1}


def prepare_oof(output_dir: Path = OUT) -> dict:
    before = snapshot(); require_space(2 * 1024**3); train, _, _ = ids(); train = sorted(train)
    records = load_s1_records(DATASET, train)
    counts = union_candidate_counts(train, PHASE7 / "v1_candidates.tsv.gz", PHASE7 / "v1_metadata.tsv.gz", PHASE7 / "address_k10_candidates.tsv.gz")
    data = {s: _sample_features(output_dir, s, records, counts) for s in ("S2", "S3")}
    models: dict[tuple[str, int], lgb.Booster] = {}; model_meta = []
    for source in ("S2", "S3"):
        for fold in range(FOLDS):
            p = output_dir / "oof/models" / f"base_{source.lower()}_fold{fold}.txt"; p.parent.mkdir(parents=True, exist_ok=True)
            if p.exists(): model = lgb.Booster(model_file=str(p))
            else:
                mask = np.asarray([fold_of(x.decode()) != fold for x in data[source]["s1"]])
                model = lgb.train(_base_params(), lgb.Dataset(data[source]["features"][mask], label=data[source]["labels"][mask], feature_name=list(FEATURE_NAMES)), num_boost_round=1485 if source == "S2" else 1495)
                tmp = p.with_suffix(".tmp"); model.save_model(str(tmp)); tmp.replace(p)
            models[source, fold] = model
            model_meta.append({"source": source, "fold": fold, "path": str(p), "sha256": sha256(p), "training_s1_excludes_fold": fold})
    out_dir = output_dir / "oof/shards"; out_dir.mkdir(parents=True, exist_ok=True)
    old_manifest_path = output_dir / "oof/oof_manifest.json"
    old_manifest = json.loads(old_manifest_path.read_text()) if old_manifest_path.exists() else {}
    old_shards = {int(x["fold"]): x for x in old_manifest.get("shards", []) if x.get("status") in ("complete", "reused")}
    paths = {fold: {s: out_dir / f"oof_{s.lower()}_fold{fold}.tsv.gz" for s in ("S2", "S3")} for fold in range(FOLDS)}
    completed = set()
    for fold in range(FOLDS):
        previous = old_shards.get(fold, {}); valid = True
        for source in ("S2", "S3"):
            p = paths[fold][source]; meta = previous.get(source, {})
            if not p.is_file() or not meta.get("sha256") or sha256(p) != meta["sha256"]: valid = False
        if valid: completed.add(fold)
    temps = {fold: {s: paths[fold][s].with_suffix(".tsv.gz.tmp") for s in ("S2", "S3")} for fold in range(FOLDS) if fold not in completed}
    handles = {f: {s: gzip.open(temps[f][s], "wt", newline="", encoding="utf-8") for s in ("S2", "S3")} for f in temps}
    writers = {f: {s: csv.DictWriter(handles[f][s], fieldnames=OOF_HEADER, delimiter="\t", lineterminator="\n") for s in ("S2", "S3")} for f in temps}
    for group in writers.values():
        for writer in group.values(): writer.writeheader()
    row_counts = {f: Counter() for f in temps}; start = time.perf_counter(); store = TargetStore(TARGET_INDEX)
    try:
        # A single deterministic candidate-stream pass dispatches each S1 group to its OOF fold.
        for pos, (s1, pairs) in enumerate(iter_union_groups(train, PHASE7 / "v1_candidates.tsv.gz", PHASE7 / "v1_metadata.tsv.gz", PHASE7 / "address_k10_candidates.tsv.gz"), 1):
            fold = fold_of(s1)
            if fold in completed: continue
            targets = store.get_many(sorted(pairs))
            for source in ("S2", "S3"):
                group = [dict(v) for v in pairs.values() if v["target_source"] == source]
                if not group: continue
                x = np.vstack([extract_features(records[s1], targets[r["candidate_entity_id"]], r, counts[s1][source]) for r in group])
                scores = models[source, fold].predict(x, num_iteration=models[source, fold].current_iteration())
                for row, score in zip(group, scores):
                    writers[fold][source].writerow({**{k: row.get(k, "") for k in OOF_HEADER}, "source1_entity_id": s1, "target_source": source, "base_score": f"{float(score):.9g}"})
                    row_counts[fold][source] += 1
            if pos % 10_000 == 0: print(f"OOF stream: {pos:,}/{len(train):,} S1; generated={sum(sum(c.values()) for c in row_counts.values()):,} rows", flush=True)
    finally:
        store.close()
        for group in handles.values():
            for handle in group.values(): handle.close()
    for fold in temps:
        for source in ("S2", "S3"): temps[fold][source].replace(paths[fold][source])
    manifest = {"folds": FOLDS, "seed": SEED, "models": model_meta, "shards": [], "frozen_snapshot": before,
                "candidate_stream_passes": 1, "runtime_seconds": time.perf_counter()-start}
    for fold in range(FOLDS):
        if fold in completed:
            entry = dict(old_shards[fold]); entry["status"] = "reused"; manifest["shards"].append(entry); continue
        manifest["shards"].append({"fold": fold, "status": "complete", "rows": dict(row_counts[fold]), **{s: {"path": str(paths[fold][s]), "sha256": sha256(paths[fold][s]), "bytes": paths[fold][s].stat().st_size} for s in ("S2", "S3")}})
    manifest["frozen_artifacts_unchanged"] = True; atomic_json(output_dir / "oof/oof_manifest.json", manifest); assert_unchanged(before); return manifest


def _read_groups(path: Path) -> Iterator[tuple[str, list[dict[str, str]]]]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as h:
        current = None; rows: list[dict[str,str]] = []
        for row in csv.DictReader(h, delimiter="\t"):
            if current is None: current = row["source1_entity_id"]
            if row["source1_entity_id"] != current:
                yield current, rows; current, rows = row["source1_entity_id"], []
            rows.append(row)
        if current is not None: yield current, rows


def _aligned_source_groups(path_s2: Path, path_s3: Path) -> Iterator[tuple[str, list[dict[str, str]], list[dict[str, str]]]]:
    """Merge sparse S2/S3 group streams, yielding empty lists for absent sources."""
    it2, it3 = iter(_read_groups(path_s2)), iter(_read_groups(path_s3))
    a, b = next(it2, None), next(it3, None)
    while a is not None or b is not None:
        if b is None or (a is not None and a[0] < b[0]):
            yield a[0], a[1], []; a = next(it2, None)
        elif a is None or b[0] < a[0]:
            yield b[0], [], b[1]; b = next(it3, None)
        else:
            yield a[0], a[1], b[1]; a, b = next(it2, None), next(it3, None)


def prepare_reranker_data(output_dir: Path = OUT) -> dict:
    before = snapshot(); manifest_path = output_dir / "oof/oof_manifest.json"
    if not manifest_path.exists(): raise FileNotFoundError("run prepare-oof first")
    require_space(1024**3); train, _, _ = ids(); truth = load_ground_truth(GROUND_TRUTH, train); values: dict[str, dict[str,list]] = {s: defaultdict(list) for s in ("S2","S3")}; report=Counter()
    for fold in range(FOLDS):
        path2 = output_dir / "oof/shards" / f"oof_s2_fold{fold}.tsv.gz"
        path3 = output_dir / "oof/shards" / f"oof_s3_fold{fold}.tsv.gz"
        for s1, g2, g3 in _aligned_source_groups(path2, path3):
            if fold_of(s1) != fold: raise RuntimeError("OOF shard contains an S1 assigned to another fold")
            sc2=np.asarray([float(r["base_score"]) for r in g2],np.float32); sc3=np.asarray([float(r["base_score"]) for r in g3],np.float32); x2,x3=contextual_group(sc2,sc3,[r["candidate_entity_id"] for r in g2],[r["candidate_entity_id"] for r in g3],g2,g3)
            for source, group, scores, x in (("S2",g2,sc2,x2),("S3",g3,sc3,x3)):
                labels=np.asarray([int(r["candidate_entity_id"] in truth[s1]) for r in group],np.int8); ranks=np.argsort(np.argsort(-scores, kind="stable"),kind="stable") + 1
                chosen=sampling_indices(labels,scores,ranks,len(group)>=101,s1,[r["candidate_entity_id"] for r in group])
                values[source]["features"].append(x[chosen]); values[source]["labels"].append(labels[chosen]); values[source]["s1"].append(np.asarray([s1]*len(chosen),"S20")); report[source+"_rows"] += len(chosen); report[source+"_positives"] += int(labels[chosen].sum())
    result={"feature_names":list(CONTEXT_NAMES),"sources":{},"frozen_snapshot":before}
    for source in ("S2","S3"):
        path=output_dir / "reranker_data" / f"{source.lower()}.npz"; path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,features=np.vstack(values[source]["features"]).astype(np.float32),labels=np.concatenate(values[source]["labels"]),s1=np.concatenate(values[source]["s1"]),feature_names=np.asarray(CONTEXT_NAMES))
        result["sources"][source]={"path":str(path),"rows":report[source+"_rows"],"positives":report[source+"_positives"],"negatives":report[source+"_rows"]-report[source+"_positives"],"bytes":path.stat().st_size}
    atomic_json(output_dir / "reranker_data/manifest.json",result); assert_unchanged(before); return result


def _load_data(output_dir: Path, source: str) -> dict[str,np.ndarray]:
    p=output_dir / "reranker_data" / f"{source.lower()}.npz"
    if not p.exists(): raise FileNotFoundError("run prepare-reranker-data first")
    with np.load(p,allow_pickle=False) as z:
        if tuple(z["feature_names"].tolist()) != CONTEXT_NAMES: raise RuntimeError("reranker feature schema mismatch")
        return {k:z[k] for k in z.files}


RCONFIG={"R1a":(31,.04,200,2.),"R1b":(63,.03,300,3.),"R2":(31,.04,200,2.)}
def _reranker_params(config: str) -> dict:
    leaves,lr,leaf,l2=RCONFIG[config]; return {"objective":"binary","metric":"binary_logloss","num_leaves":leaves,"learning_rate":lr,"min_data_in_leaf":leaf,"lambda_l2":l2,"feature_fraction":1.,"seed":SEED,"deterministic":True,"force_col_wise":True,"num_threads":THREADS,"verbosity":-1}


def _fit(output_dir: Path, variant: str) -> dict:
    before=snapshot(); models={}; details={}
    sources=("S2","S3") if variant != "R2" else ("shared",)
    raw={s:_load_data(output_dir,s) for s in ("S2","S3")}
    for source in sources:
        if source=="shared":
            x=np.vstack((raw["S2"]["features"],raw["S3"]["features"])); x[:,CONTEXT_NAMES.index("source_s3")]=np.r_[np.zeros(len(raw["S2"]["features"])),np.ones(len(raw["S3"]["features"]))]; y=np.r_[raw["S2"]["labels"],raw["S3"]["labels"]]; s1=np.r_[raw["S2"]["s1"],raw["S3"]["s1"]]
        else: x,y,s1=raw[source]["features"],raw[source]["labels"],raw[source]["s1"]
        valid=np.asarray([int.from_bytes(hashlib.sha256(v).digest()[:8],"big") % 5 == 0 for v in s1])
        train=lgb.Dataset(x[~valid],label=y[~valid],feature_name=list(CONTEXT_NAMES)); eva=lgb.Dataset(x[valid],label=y[valid],reference=train,feature_name=list(CONTEXT_NAMES))
        model=lgb.train(_reranker_params(variant),train,num_boost_round=800,valid_sets=[eva],callbacks=[lgb.early_stopping(50,verbose=False)])
        rounds=max(1,model.best_iteration); final=lgb.train(_reranker_params(variant),lgb.Dataset(x,label=y,feature_name=list(CONTEXT_NAMES)),num_boost_round=rounds)
        p=output_dir / "models" / variant / f"reranker_{source.lower()}.txt"; p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix(".tmp"); final.save_model(str(tmp));tmp.replace(p);models[source]=final;details[source]={"path":str(p),"sha256":sha256(p),"rows":len(y),"iterations":rounds}
    result={"variant":variant,"models":details,"feature_names":list(CONTEXT_NAMES),"frozen_snapshot":before};atomic_json(output_dir / "models" / variant / "training.json",result);assert_unchanged(before);return result


def _rerank_cache(base: ScoreCache, other: ScoreCache, model: lgb.Booster, source: str, frontier: bool = False) -> ScoreCache:
    score=[]
    for i in range(len(base.s1_ids)):
        a,b=(base.group(i),other.group(i)); ev=[{"from_v1":int(v),"from_address":int(a_),"address_rank":int(r)} for v,a_,r in zip(base.from_v1[base.offsets[i]:base.offsets[i+1]],base.from_address[base.offsets[i]:base.offsets[i+1]],base.address_rank[base.offsets[i]:base.offsets[i+1]])]
        other_scores=b[1]; base_ids=[x.decode("ascii") for x in a[0]]; other_ids=[x.decode("ascii") for x in b[0]]; x=_context(source,a[1],base_ids,np.concatenate((a[1],other_scores)),base_ids+other_ids,ev); predicted=model.predict(x)
        if frontier:
            preserve_high = (a[1] >= .99) & (len(a[1]) <= 25)
            predicted = np.asarray(predicted, dtype=np.float32)
            predicted[a[1] < .50] = 0.0
            predicted[preserve_high] = 1.0  # A_source_02 accepts these isolated >=.99 pairs.
        score.append(predicted)
    return ScoreCache(base.source,base.s1_ids,base.offsets,base.targets,np.concatenate(score).astype(np.float32),base.from_v1,base.from_address,base.address_rank,base.address_score)


def _predictions(s2: ScoreCache,s3: ScoreCache,t2: float,t3: float,frontier: bool=False) -> dict[str,set[str]]:
    selected={s1:{} for s1 in s2.s1_ids}
    for cache,other,t in ((s2,s3,t2),(s3,s2,t3)):
        for i,s1 in enumerate(cache.s1_ids):
            tar,sc=cache.group(i); n=len(sc); keep=sc>=t
            if frontier:
                base=other.group(i)[1]  # only used to keep allocation local; R3 cache carries reranker values
                # Base score unavailable after replacement; frontier is encoded in reranker cache construction in implementation variants.
            for x,v in zip(tar[keep],sc[keep]): selected[s1][x.decode()]=float(v)
    return _apply_conflicts(selected,BASE_POLICY)


def _metrics(truth: Mapping[str,set[str]],pred:Mapping[str,set[str]],ids_:Iterable[str]) -> dict:
    ids_=list(ids_);r=score_predictions(truth,pred,ids_);source={}
    for s in ("S2","S3"):
        p=sum(len({x for x in pred[i] if x.startswith(s+"-")} & {x for x in truth[i] if x.startswith(s+"-")}) for i in ids_); q=sum(len({x for x in pred[i] if x.startswith(s+"-")} - truth[i]) for i in ids_); f=sum(len({x for x in truth[i] if x.startswith(s+"-")} - pred[i]) for i in ids_); source[s]={"tp":p,"fp":q,"fn":f,"precision":p/(p+q) if p+q else 0.,"recall":p/(p+f) if p+f else 0.}
    return {**asdict(r),"source_metrics":source,"predicted_singletons":sum(not pred[x] for x in ids_),"singleton_false_merges":sum(not truth[x] and bool(pred[x]) for x in ids_)}


def _threshold_pairs() -> list[tuple[float,float]]: return [(.80,.80),(.85,.85),(.90,.90),(.93,.93),(.95,.95),(.97,.97),(.98,.98),(.99,.99),(.90,.95),(.93,.97)]


def _evaluate_variant(output_dir:Path, variant:str, split:str="tune", locked_thresholds: tuple[float,float] | None=None) -> dict:
    if split == "validation" and locked_thresholds is None: raise RuntimeError("validation evaluation requires tune-locked thresholds")
    before=snapshot(); b2,b3,truth=tune_inputs() if split=="tune" else val_inputs(); model_variant="R1a" if variant=="R3" else variant
    meta=json.loads((output_dir/"models"/model_variant/"training.json").read_text()); models={k:lgb.Booster(model_file=v["path"]) for k,v in meta["models"].items()}; m2=models.get("S2") or models["shared"];m3=models.get("S3") or models["shared"]
    frontier=variant=="R3";s2,s3=_rerank_cache(b2,b3,m2,"S2",frontier),_rerank_cache(b3,b2,m3,"S3",frontier)
    if split == "validation":
        pairs=[locked_thresholds]
    else: pairs=_threshold_pairs()
    best=None
    for a,b in pairs:
        pred=_predictions(s2,s3,a,b); row=_metrics(truth,pred,s2.s1_ids); row.update({"s2_threshold":a,"s3_threshold":b})
        if best is None or (row["macro_f0_5"],row["micro_precision_diagnostic"])>(best["macro_f0_5"],best["micro_precision_diagnostic"]):best=row
    if split=="tune":
        a0,b0=best["s2_threshold"],best["s3_threshold"]
        for a in sorted({max(0,min(1,a0+d)) for d in (-.005,0,.005)}):
            for b in sorted({max(0,min(1,b0+d)) for d in (-.005,0,.005)}):
                pred=_predictions(s2,s3,a,b);row=_metrics(truth,pred,s2.s1_ids);row.update({"s2_threshold":a,"s3_threshold":b})
                if (row["macro_f0_5"],row["micro_precision_diagnostic"])>(best["macro_f0_5"],best["micro_precision_diagnostic"]):best=row
    best.update({"variant":variant,"split":split,"delta_vs_champion":best["macro_f0_5"]-(CHAMPION_TUNE if split=="tune" else CHAMPION_VALIDATION),"frozen_artifacts_unchanged":True})
    atomic_json(output_dir/"models"/variant/f"{split}_result.json",best);assert_unchanged(before);return best


def experiment_r1(output_dir:Path=OUT)->dict:
    return {v:_evaluate_variant(output_dir,v) for v in ("R1a","R1b") for _ in (_fit(output_dir,v),)}
def experiment_r2(output_dir:Path=OUT)->dict: _fit(output_dir,"R2");return _evaluate_variant(output_dir,"R2")
def experiment_r3(output_dir:Path=OUT)->dict:
    # R3 shares R1a's conservative model; its stored result documents the predeclared frontier.
    if not (output_dir/"models/R1a/training.json").exists(): raise FileNotFoundError("run experiment-r1 first")
    value=_evaluate_variant(output_dir,"R3");value["frontier"]={"reject_base_below":.50,"preserve_isolated_base_at_least":.99,"isolated_max_group":25};atomic_json(output_dir/"models/R3/tune_result.json",value);return value


def compare_tune(output_dir:Path=OUT)->dict:
    before=snapshot(); rows=[]
    for v in ("R1a","R1b","R2","R3"):
        p=output_dir/"models"/v/"tune_result.json"
        if p.exists():rows.append(json.loads(p.read_text()))
    if not rows:raise RuntimeError("no tune reranker results")
    rows.sort(key=lambda x:(x["macro_f0_5"],x["micro_precision_diagnostic"]),reverse=True);atomic_json(output_dir/"tune_comparison.json",{"baseline_champion":CHAMPION_TUNE,"rows":rows,"frozen_snapshot":before});assert_unchanged(before);return {"winner":rows[0]["variant"],"rows":rows}


def lock_winner(output_dir:Path=OUT)->dict:
    before=snapshot(); path=output_dir/"locked_winner.json"
    if path.exists() or (output_dir/"no_winner.json").exists():raise RuntimeError("winner decision already locked")
    result=compare_tune(output_dir)["rows"][0]; delta=float(result["delta_vs_champion"])
    # The lower gain branch needs slice/FP/singleton evidence; only the strong
    # aggregate gate is automated until those diagnostics are written.
    qualifies=delta>=.0015
    model_variant="R1a" if result["variant"]=="R3" else result["variant"]
    train_meta=json.loads((output_dir/"models"/model_variant/"training.json").read_text())
    oof_manifest=output_dir/"oof/oof_manifest.json"
    value={"status":"LOCKED" if qualifies else "NO_WINNER","winner":result["variant"] if qualifies else None,
           "tune_metrics":result,"thresholds":{"S2":result["s2_threshold"],"S3":result["s3_threshold"]},
           "model_hashes":{k:v["sha256"] for k,v in train_meta["models"].items()},
           "feature_names":list(CONTEXT_NAMES),"oof_manifest_sha256":sha256(oof_manifest),
           "code_sha256":sha256(Path(__file__)),"frozen_snapshot":before,"validation_used":False}
    destination=path if qualifies else output_dir/"no_winner.json";atomic_json(destination,value);assert_unchanged(before);return value


def confirm_validation(output_dir:Path=OUT)->dict:
    before=snapshot(); lock=output_dir/"locked_winner.json"; done=output_dir/"validation_confirmation.json"
    if not lock.exists() or done.exists():raise RuntimeError("validation requires one locked winner and no prior confirmation")
    value=json.loads(lock.read_text()); winner=value["winner"]; thresholds=json.loads(lock.read_text())["thresholds"]; result=_evaluate_variant(output_dir, winner, "validation", (float(thresholds["S2"]),float(thresholds["S3"]))); delta=result["macro_f0_5"]-CHAMPION_VALIDATION; status="CONFIRMED" if delta>=0 else "MIXED" if delta>=-.001 else "FAILED_TO_GENERALIZE";result.update({"status":status,"locked_winner":winner,"delta_vs_champion_validation":delta});atomic_json(done,result);assert_unchanged(before);return result


def report(output_dir:Path=OUT)->dict:
    before=snapshot(); lock=output_dir/"locked_winner.json"; confirm=output_dir/"validation_confirmation.json"; data={"phase":"11.3","champion":"A_source_02","candidate_regeneration_required":"NO","phase8_test_rescoring_required":"NO","locked_winner":json.loads(lock.read_text()) if lock.exists() else None,"validation_confirmation":json.loads(confirm.read_text()) if confirm.exists() else None,"frozen_artifacts_unchanged":True}
    lines=["# Phase 11.3 Contextual Reranker", "", f"Champion comparator: A_source_02 ({CHAMPION_TUNE:.6f} tune).", "", "No Phase 12 candidates, base scores, or submissions were changed."]
    output_dir.mkdir(parents=True,exist_ok=True);(output_dir/"phase113_report.md").write_text("\n".join(lines)+"\n");atomic_json(output_dir/"phase113_summary.json",data);assert_unchanged(before);return data



def _test_s1_ids() -> list[str]:
    """Public test IDs only.  This runner never opens a test-label file."""
    with TEST_S1.open("rt", encoding="utf-8", newline="") as h:
        reader = csv.DictReader(h, delimiter="\t")
        if reader.fieldnames is None or "entity_id" not in reader.fieldnames:
            raise RuntimeError("unexpected test Source 1 schema")
        out = [row["entity_id"] for row in reader]
    if len(out) != 1_732_544 or len(set(out)) != len(out) or any(not x.startswith("S1-") for x in out):
        raise RuntimeError("invalid test S1 coverage")
    return out


def _shard_snapshot(manifest: Mapping[str, object]) -> dict[str, tuple[int, int]]:
    """The score-manifest hashes authenticate shard contents; snapshot prevents writes here."""
    found = {}
    for shard in manifest["shards"]:
        for source in ("S2", "S3"):
            row = shard["outputs"][source]
            path = Path(row["path"])
            if not path.is_file() or path.stat().st_size != int(row["bytes"]):
                raise RuntimeError(f"missing or altered Phase 12 {source} shard: {path}")
            stat = path.stat()
            found[str(path)] = (stat.st_size, stat.st_mtime_ns)
    return found


def _test_audit(output_dir: Path) -> tuple[dict, list[str], dict[str, tuple[int, int]]]:
    """All mandatory fail-fast checks. Called before any test score file is opened."""
    lock_path, confirmation_path = output_dir / "locked_winner.json", output_dir / "validation_confirmation.json"
    if not lock_path.exists() or not confirmation_path.exists():
        raise RuntimeError("missing R3 lock or validation confirmation")
    lock, confirmation = json.loads(lock_path.read_text()), json.loads(confirmation_path.read_text())
    if lock.get("status") != "LOCKED" or lock.get("winner") != "R3" or lock.get("thresholds") != {"S2": R3_TEST_THRESHOLD, "S3": R3_TEST_THRESHOLD}:
        raise RuntimeError("R3 lock/threshold mismatch")
    if confirmation.get("status") != "CONFIRMED" or abs(float(confirmation.get("macro_f0_5", -1)) - R3_VALIDATION_MACRO) > 1e-12:
        raise RuntimeError("R3 validation confirmation mismatch")
    meta = json.loads((output_dir / "models/R1a/training.json").read_text())
    if {source: value["sha256"] for source, value in meta["models"].items()} != lock.get("model_hashes"):
        raise RuntimeError("R3 model hashes differ from lock")
    forbidden = ("label", "truth", "negative_reason", "entity_id")
    if tuple(lock.get("feature_names", ())) != CONTEXT_NAMES or any(word in " ".join(CONTEXT_NAMES) for word in forbidden):
        raise RuntimeError("reranker feature schema contains prohibited leakage fields")
    oof_path = output_dir / "oof/oof_manifest.json"
    oof = json.loads(oof_path.read_text())
    if sha256(oof_path) != lock.get("oof_manifest_sha256") or oof.get("folds") != 5 or len(oof.get("models", [])) != 10 or any(x.get("fold") != x.get("training_s1_excludes_fold") for x in oof["models"]):
        raise RuntimeError("grouped OOF provenance mismatch")
    train, tune, validation = ids(); test_ids = _test_s1_ids()
    if set(train) & set(tune) or set(train) & set(validation) or set(train) & set(test_ids) or set(tune) & set(test_ids) or set(validation) & set(test_ids):
        raise RuntimeError("S1 split leakage detected")
    if sha256(ROOT / "output/phase112/matching_results.tsv") != CHAMPION_SHA:
        raise RuntimeError("Phase 11.2 champion submission changed")
    manifest = json.loads((PHASE12 / "scores/score_manifest.json").read_text())
    candidates = json.loads((PHASE12 / "candidates/candidate_summary.json").read_text())
    if int(manifest.get("total", -1)) != 309_551_055 or manifest.get("candidate_membership_sha256") != candidates.get("membership_sha256") or int(candidates.get("pairs", -1)) != 309_551_055:
        raise RuntimeError("Phase 12 score/candidate manifests do not match")
    locked = lock.get("frozen_snapshot", {})
    for path in (PHASE12 / "scores/score_manifest.json", PHASE12 / "candidates/candidate_pairs_long.tsv.gz", ROOT / "output/candidate_pairs.tsv", ROOT / "output/matching_results.tsv", ROOT / "output/phase112/matching_results.tsv"):
        if locked.get(str(path)) != snapshot().get(str(path)):
            raise RuntimeError(f"locked frozen artifact changed: {path}")
    return manifest, test_ids, _shard_snapshot(manifest)


def _ordered_groups(path2: Path, path3: Path, positions: Mapping[str, int]):
    """Merge sparse score sources according to immutable test-file order."""
    it2, it3 = iter(_read_groups(path2)), iter(_read_groups(path3)); a, b = next(it2, None), next(it3, None)
    while a is not None or b is not None:
        if b is None or (a is not None and positions[a[0]] < positions[b[0]]):
            yield a[0], a[1], []; a = next(it2, None)
        elif a is None or positions[b[0]] < positions[a[0]]:
            yield b[0], [], b[1]; b = next(it3, None)
        else:
            yield a[0], a[1], b[1]; a, b = next(it2, None), next(it3, None)


def _r3_frontier(model: lgb.Booster, scores: np.ndarray, context: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(model.predict(context), dtype=np.float32)
    values[scores < .50] = 0.0
    values[(scores >= .99) & (len(scores) <= 25)] = 1.0
    return values >= R3_TEST_THRESHOLD, values


def predict_test(output_dir: Path = OUT) -> dict:
    """Generate Submission #3 from existing Phase 12 scores only."""
    started, before = time.perf_counter(), snapshot()
    manifest, s1_ids, shard_before = _test_audit(output_dir)
    positions = {s1: index for index, s1 in enumerate(s1_ids)}
    meta = json.loads((output_dir / "models/R1a/training.json").read_text())
    models = {source: lgb.Booster(model_file=value["path"]) for source, value in meta["models"].items()}
    state = output_dir / "test_prediction"; state.mkdir(parents=True, exist_ok=True)
    database_path = state / "ownership.sqlite"
    db = sqlite3.connect(database_path)
    db.execute("PRAGMA journal_mode=WAL"); db.execute("PRAGMA synchronous=NORMAL")
    db.execute("CREATE TABLE IF NOT EXISTS accepted (target TEXT NOT NULL, s1 TEXT NOT NULL, score REAL NOT NULL, source TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS completed (shard INTEGER PRIMARY KEY, rows INTEGER NOT NULL)")
    done = {x[0] for x in db.execute("SELECT shard FROM completed")}
    processed = 0
    try:
        for number, item in enumerate(sorted(manifest["shards"], key=lambda x: int(x["shard"])), 1):
            shard = int(item["shard"])
            if shard in done:
                processed += int(item["total"]); print(f"R3 test: shard {number}/70 reused; pairs={processed:,}", flush=True); continue
            p2, p3 = Path(item["outputs"]["S2"]["path"]), Path(item["outputs"]["S3"]["path"])
            rows = 0
            for s1, g2, g3 in _ordered_groups(p2, p3, positions):
                if s1 not in positions: raise RuntimeError(f"unknown test S1 in score shard: {s1}")
                s2 = np.asarray([float(x["score"]) for x in g2], dtype=np.float32); s3 = np.asarray([float(x["score"]) for x in g3], dtype=np.float32)
                e2 = [{"from_v1":int(x["from_v1"]),"from_address":int(x["from_address"]),"address_rank":int(x["address_rank"] or 0)} for x in g2]
                e3 = [{"from_v1":int(x["from_v1"]),"from_address":int(x["from_address"]),"address_rank":int(x["address_rank"] or 0)} for x in g3]
                x2, x3 = contextual_group(s2, s3, [x["candidate_entity_id"] for x in g2], [x["candidate_entity_id"] for x in g3], e2, e3)
                for source, group, values, context in (("S2", g2, s2, x2), ("S3", g3, s3, x3)):
                    if not group: continue
                    keep, reranked = _r3_frontier(models[source], values, context)
                    data = [(row["candidate_entity_id"], s1, float(score), source) for row, score, selected in zip(group, reranked, keep) if selected]
                    if data: db.executemany("INSERT INTO accepted VALUES (?,?,?,?)", data)
                    rows += len(group)
            db.execute("INSERT INTO completed VALUES (?,?)", (shard, rows)); db.commit(); processed += rows
            elapsed = time.perf_counter() - started; rate = processed / elapsed if elapsed else 0.; eta = (int(manifest["total"]) - processed) / rate if rate else 0.
            print(f"R3 test: shard {number}/70 complete; pairs={processed:,}/{int(manifest['total']):,}; elapsed={elapsed:.1f}s ETA={eta:.1f}s", flush=True)
        db.execute("CREATE INDEX IF NOT EXISTS accepted_target ON accepted(target, score DESC, s1)")
        db.execute("DROP TABLE IF EXISTS winner")
        db.execute("CREATE TABLE winner AS SELECT target,s1,score,source FROM (SELECT target,s1,score,source,ROW_NUMBER() OVER (PARTITION BY target ORDER BY score DESC,s1) rn FROM accepted) WHERE rn=1")
        db.execute("CREATE INDEX winner_s1 ON winner(s1,target)"); db.commit()
        target = ROOT / "output/phase113"; target.mkdir(parents=True, exist_ok=True)
        output, temporary = target / "matching_results.tsv", target / "matching_results.tsv.tmp"; sizes = []
        with temporary.open("wt", encoding="utf-8", newline="") as h:
            writer = csv.writer(h, delimiter="\t", lineterminator="\n"); writer.writerow(("source1_entity_id", "matched_entity_ids"))
            for s1 in s1_ids:
                matches = [x[0] for x in db.execute("SELECT target FROM winner WHERE s1=? ORDER BY target", (s1,))]
                if len(matches) != len(set(matches)) or any(not x.startswith(("S2-", "S3-")) for x in matches): raise RuntimeError("invalid R3 match output")
                writer.writerow((s1, ",".join(matches))); sizes.append(len(matches))
        temporary.replace(output)
        links = db.execute("SELECT count(*) FROM winner").fetchone()[0]
        conflicts = db.execute("SELECT count(*) FROM (SELECT target FROM accepted GROUP BY target HAVING count(*)>1)").fetchone()[0]
        by_source = {source: db.execute("SELECT count(*) FROM winner WHERE source=?", (source,)).fetchone()[0] for source in ("S2", "S3")}
    finally:
        db.close()
    if _shard_snapshot(manifest) != shard_before: raise RuntimeError("Phase 12 score shards changed during prediction")
    assert_unchanged(before)
    lines = sum(1 for _ in output.open("rt", encoding="utf-8"))
    if len(sizes) != 1_732_544 or lines != 1_732_545 or links != sum(sizes): raise RuntimeError("R3 output coverage/count validation failed")
    result = {"status":"READY", "output":str(output), "sha256":sha256(output), "s1_rows":len(sizes), "predicted_links":int(links), "predicted_singletons":sum(x == 0 for x in sizes), "accepted_by_source":by_source, "ownership_conflicts_resolved":int(conflicts), "prediction_stats":{"mean":float(np.mean(sizes)),"median":float(np.percentile(sizes,50)),"p95":float(np.percentile(sizes,95)),"p99":float(np.percentile(sizes,99)),"max":int(max(sizes))}, "runtime_seconds":time.perf_counter()-started, "peak_rss_mb":rss_mb(), "leakage_audit":"PASS", "candidate_set_unchanged":True, "base_scores_unchanged":True, "frozen_artifacts_unchanged":True}
    atomic_json(state / "prediction_summary.json", result)
    return result


def main(argv: list[str] | None=None)->int:
    p=argparse.ArgumentParser();p.add_argument("command",choices=("inventory","baseline-check","prepare-oof","prepare-reranker-data","experiment-r1","experiment-r2","experiment-r3","compare-tune","lock-winner","confirm-validation","predict-test","report"));args=p.parse_args(argv)
    fn={"inventory":inventory,"baseline-check":baseline_check,"prepare-oof":prepare_oof,"prepare-reranker-data":prepare_reranker_data,"experiment-r1":experiment_r1,"experiment-r2":experiment_r2,"experiment-r3":experiment_r3,"compare-tune":compare_tune,"lock-winner":lock_winner,"confirm-validation":confirm_validation,"predict-test":predict_test,"report":report}[args.command]
    print(json.dumps(fn(),indent=2,sort_keys=True,default=str));return 0

if __name__=="__main__":raise SystemExit(main())
