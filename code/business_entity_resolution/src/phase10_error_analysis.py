#!/usr/bin/env python3
"""Read-only Phase 10 error analysis for the frozen entity-resolution pipeline.

This module never regenerates candidates, retrains models, or changes the frozen
Phase 8/9/12 inputs.  It consumes cached tune/validation scores and writes compact
aggregate diagnostics plus bounded, deterministic examples under artifacts/phase10_error_analysis.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import resource
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from baseline import TargetStore, load_s1_records
from diagnostics import DEFAULT_DATASET_ROOT
from phase8_model import FEATURE_NAMES, extract_features, iter_union_groups
from phase9_policy import (DecisionPolicy, ScoreCache, _selected_scores,
                           load_cache, predictions_for_policy, read_score_file)
from scoring import (count_link_errors, load_ground_truth, load_id_file,
                     precision_recall_diagnostics, score_entity,
                     score_predictions)

BASE = Path(__file__).resolve().parents[1]
ART = BASE / "artifacts"
OUT = ART / "phase10_error_analysis"
DATA = DEFAULT_DATASET_ROOT
GROUND_TRUTH = DATA / "train/train_ground_truth.tsv"
PHASE8 = ART / "model/phase8"
PHASE9 = ART / "model/phase9"
VALIDATION = ART / "validation_evaluation"
TARGET_INDEX = ART / "blocking/v1_index.sqlite"
POLICY_PATH = PHASE9 / "decision_policy.json"

SPLITS = {
    "tune": {
        "ids": ART / "splits/tune_s1_ids.txt",
        "cache_s2": PHASE9 / "score_cache_s2.npz",
        "cache_s3": PHASE9 / "score_cache_s3.npz",
        "v1_candidates": ART / "baseline/tune_candidates.tsv.gz",
        "v1_metadata": ART / "baseline/tune_metadata.tsv.gz",
        "address": ART / "retrieval_diagnosis/phase6e/address_top10_100k.tsv.gz",
        "predictions": PHASE9 / "tune_predictions.tsv.gz",
    },
    "validation": {
        "ids": ART / "splits/val_s1_ids.txt",
        "scores_s2": VALIDATION / "validation_scores_s2.tsv.gz",
        "scores_s3": VALIDATION / "validation_scores_s3.tsv.gz",
        "score_report": VALIDATION / "scoring_report.json",
        "v1_candidates": ART / "blocking/v1_validation_candidates.tsv.gz",
        "v1_metadata": ART / "blocking/v1_validation_metadata.tsv.gz",
        "address": VALIDATION / "address_top10_validation.tsv.gz",
        "predictions": VALIDATION / "validation_predictions.tsv.gz",
    },
}

IMMEDIATE_BELOW = {"S2": "0.90-0.93", "S3": "0.95-0.97"}

SLICE_BUCKETS = {
    "name_condition": ("exact_or_very_high", "medium", "weak", "missing"),
    "address_condition": ("exact_or_very_high", "medium", "weak", "missing"),
}

SCORE_BINS = {
    "S2": ((0.0, 0.50, "<0.50"), (0.50, 0.80, "0.50-0.80"), (0.80, 0.90, "0.80-0.90"),
           (0.90, 0.93, "0.90-0.93"), (0.93, 0.97, "0.93-0.97"),
           (0.97, 0.99, "0.97-0.99"), (0.99, math.inf, ">=0.99")),
    "S3": ((0.0, 0.50, "<0.50"), (0.50, 0.80, "0.50-0.80"), (0.80, 0.90, "0.80-0.90"),
           (0.90, 0.95, "0.90-0.95"), (0.95, 0.97, "0.95-0.97"),
           (0.97, 0.99, "0.97-0.99"), (0.99, math.inf, ">=0.99")),
}
FROZEN_FILES = (
    PHASE8 / "model_s2.txt", PHASE8 / "model_s3.txt", PHASE8 / "feature_manifest.json",
    POLICY_PATH, TARGET_INDEX,
    ART / "test_inference/phase12/candidates/candidate_pairs_long.tsv.gz",
    ART / "test_inference/phase12/scores/score_manifest.json",
    BASE.parents[1] / "output/matching_results.tsv", BASE.parents[1] / "output/candidate_pairs.tsv",
)


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def file_state(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def snapshot_frozen() -> dict[str, dict[str, int | str]]:
    """Cheap immutable-input guard; avoids hashing multi-gigabyte frozen outputs."""
    return {str(path): file_state(path) for path in FROZEN_FILES if path.exists()}


def assert_unchanged(before: Mapping[str, Mapping[str, object]]) -> None:
    after = snapshot_frozen()
    if dict(before) != after:
        raise RuntimeError("a frozen Phase 8/9/12 input changed during Phase 10 analysis")


def load_policy(path: Path = POLICY_PATH) -> DecisionPolicy:
    raw = json.loads(path.read_text(encoding="utf-8"))["policy"]
    policy = DecisionPolicy(float(raw["s2_threshold"]), float(raw["s3_threshold"]),
                            raw.get("open_threshold"), str(raw["conflict_policy"]),
                            raw.get("conflict_margin"))
    if asdict(policy) != {"s2_threshold": 0.93, "s3_threshold": 0.97,
                          "open_threshold": None, "conflict_policy": "highest",
                          "conflict_margin": None}:
        raise ValueError("Phase 10 only supports the frozen Phase 9 policy")
    return policy


def _cache_for(split: str) -> tuple[list[str], ScoreCache, ScoreCache]:
    paths = SPLITS[split]
    ids = load_id_file(paths["ids"])
    if split == "tune":
        return ids, load_cache(paths["cache_s2"]), load_cache(paths["cache_s3"])
    report = json.loads(paths["score_report"].read_text(encoding="utf-8"))
    return (ids,
            read_score_file(paths["scores_s2"], "S2", ids, int(report["rows"]["S2"])),
            read_score_file(paths["scores_s3"], "S3", ids, int(report["rows"]["S3"])))


def _decode(values: np.ndarray) -> list[str]:
    return [x.decode("ascii") for x in values]


def _score_bin(source: str, score: float) -> str:
    for lo, hi, label in SCORE_BINS[source]:
        if lo <= score < hi:
            return label
    raise AssertionError(score)


def address_rank_distribution(address_ranks: Mapping[tuple[str, str], int],
                              truth: Mapping[str, set[str]]) -> list[dict]:
    """Return deterministic rank and cumulative AddressK10 truth-hit evidence."""
    rank_order=("1", "2-3", "4-5", "6-10")
    rows=[]
    for source in ("S2", "S3"):
        total_truth=sum(1 for links in truth.values() for target in links if target.startswith(source+"-"))
        total_address_hits=sum(address_ranks.get((source, rank), 0) for rank in rank_order)
        cumulative=0
        for rank in rank_order:
            hits=address_ranks.get((source, rank), 0); cumulative += hits
            cutoff={"1":"<=1","2-3":"<=3","4-5":"<=5","6-10":"<=10"}[rank]
            rows.append({"source":source,"rank_bin":rank,"truth_hits":hits,"cumulative_cutoff":cutoff,"cumulative_truth_hits":cumulative,
                         "share_of_address_ranked_truth_hits":hits/total_address_hits if total_address_hits else 0.0,
                         "cumulative_share_of_address_ranked_truth_hits":cumulative/total_address_hits if total_address_hits else 0.0,
                         "share_of_all_source_truth_links":hits/total_truth if total_truth else 0.0,
                         "cumulative_share_of_all_source_truth_links":cumulative/total_truth if total_truth else 0.0})
    return rows


def _candidate_bin(count: int) -> str:
    if count == 0: return "0"
    if count <= 25: return "1-25"
    if count <= 100: return "26-100"
    if count <= 500: return "101-500"
    return "501+"


def _f05_bin(value: float) -> str:
    if value == 1.0: return "1.00"
    if value >= .75: return "0.75-<1"
    if value >= .50: return "0.50-<0.75"
    if value > 0: return ">0-<0.50"
    return "0"


def _source(target: str) -> str:
    return target[:2]


def _score_lookup(cache: ScoreCache, index: int) -> dict[str, float]:
    targets, scores = cache.group(index)
    return dict(zip(_decode(targets), map(float, scores)))


def classify_truth_link(target: str, scores: Mapping[str, float], prediction: set[str], policy: DecisionPolicy) -> str:
    """Classify a positive link at its first failed frozen-pipeline stage."""
    score = scores.get(target)
    if score is None:
        return "A_blocking_miss"
    threshold = policy.s2_threshold if target.startswith("S2-") else policy.s3_threshold
    if score < threshold:
        return "B1_below_threshold"
    if target not in prediction:
        return "B2_ownership_conflict"
    return "TP"


def classify_fp(target: str, score: float, threshold: float, candidate_count: int,
                contested: bool) -> str:
    """Evidence-backed, mutually ordered final-FP labels."""
    if score >= .99:
        return "C1_high_confidence"
    if score < threshold + .02:
        return "C2_near_threshold"
    if contested:
        return "C3_target_conflict_related"
    if candidate_count > 500:
        return "C4_crowded_candidate_set"
    return "C5_other"


def ownership_summary(preselected: Mapping[str, Mapping[str, float]], predictions: Mapping[str, set[str]],
                      truth: Mapping[str, set[str]], ids: Iterable[str]) -> tuple[dict, set[tuple[str, str]], set[str]]:
    by_target: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for s1, values in preselected.items():
        for target, score in values.items(): by_target[target].append((s1, score))
    rows = Counter(); removed_truth: set[tuple[str, str]] = set(); gaps = []
    links_removed = fp_removed = tp_removed = 0
    for target, contenders in by_target.items():
        if len(contenders) < 2: continue
        contenders.sort(key=lambda x: (-x[1], x[0]))
        winner, winner_score = contenders[0]
        gap = winner_score - contenders[1][1]
        gaps.append(gap)
        rows["conflicted_targets"] += 1
        rows["tiny_gap_<0.01"] += int(gap < .01)
        rows["large_gap_>=0.05"] += int(gap >= .05)
        winner_true = target in truth[winner]
        loser_true = any(target in truth[s1] for s1, _ in contenders[1:])
        if winner_true and loser_true: rows["winner_and_loser_true"] += 1
        elif winner_true: rows["winner_correct_loser_incorrect"] += 1
        elif loser_true: rows["winner_incorrect_loser_correct"] += 1
        else: rows["all_contenders_incorrect"] += 1
        for s1, _ in contenders[1:]:
            if target not in predictions[s1]:
                links_removed += 1
                if target in truth[s1]: tp_removed += 1; removed_truth.add((s1, target))
                else: fp_removed += 1
    contested_targets = {target for target, contenders in by_target.items() if len(contenders) > 1}
    return ({**rows, "links_removed": links_removed, "tp_removed": tp_removed,
             "fp_removed": fp_removed, "score_gap_median": float(np.median(gaps)) if gaps else 0.0,
             "score_gap_p95": float(np.percentile(gaps,95)) if gaps else 0.0}, removed_truth, contested_targets)


def _rows(path: Path, rows: Iterable[Mapping[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _add_metric(bucket: dict, truth: bool, predicted: bool, retrieved: bool = True) -> None:
    bucket["support"] += int(truth)
    bucket["retrieved_truth"] += int(truth and retrieved)
    bucket["tp"] += int(truth and predicted)
    bucket["fn"] += int(truth and not predicted)
    bucket["fp"] += int((not truth) and predicted)


def _finalize_metrics(table: Mapping[tuple[str, str], Counter]) -> list[dict]:
    out=[]
    for (dimension, bucket), x in sorted(table.items()):
        support, retrieved, tp, fn, fp = (x[k] for k in ("support","retrieved_truth","tp","fn","fp"))
        # Explicit zeroes keep required diagnostic buckets visible and machine-readable.
        out.append({"dimension":dimension,"bucket":bucket,
                    "support":support,"retrieved_truth":retrieved,"tp":tp,"fn":fn,"fp":fp,
                    "retrieval_recall": retrieved/support if support else "",
                    "model_policy_recall_given_retrieved": tp/retrieved if retrieved else "",
                    "precision_diagnostic": tp/(tp+fp) if tp+fp else ""})
    return out


def _agreement_bucket(*, missing: bool, exact: bool, ratio: float) -> str:
    """Display the frozen 0--1 RapidFuzz features as documented 70/95% buckets."""
    if missing:
        return "missing"
    if exact or ratio >= 0.95:
        return "exact_or_very_high"
    if ratio >= 0.70:
        return "medium"
    return "weak"


def _slice_values(features: Mapping[str, float], source_count: int) -> list[tuple[str, str]]:
    # Phase 8 divides RapidFuzz's native 0--100 score by 100.  These are the
    # original intended 70/95% diagnostic cutoffs expressed on that 0--1 scale.
    name = _agreement_bucket(
        missing=bool(features["name_missing_left"] or features["name_missing_right"]),
        exact=bool(features["name_basic_exact"] or features["name_core_exact"]),
        ratio=float(features["name_ratio"]),
    )
    addr = _agreement_bucket(
        missing=bool(features["address_missing_left"] or features["address_missing_right"]),
        exact=bool(features["address_exact"]), ratio=float(features["address_ratio"]),
    )
    postal = "shared" if features["postal_shared"] else ("conflict" if features["postal_conflict"] else "missing")
    country = "same" if features["country_equal"] else ("missing" if features["country_missing_left"] or features["country_missing_right"] else "conflict")
    complete = "both_available" if not features["name_missing_left"] and not features["address_missing_left"] and not features["name_missing_right"] and not features["address_missing_right"] else ("name_missing" if features["name_missing_left"] or features["name_missing_right"] else "address_missing")
    return [("name_condition",name),("address_condition",addr),("numeric", "shared" if features["numeric_shared"] else ("conflict" if features["numeric_conflict"] else "neither")),
            ("postal",postal),("country",country),("completeness",complete),("candidate_ambiguity",_candidate_bin(source_count))]


def analyze_split(split: str, output_dir: Path = OUT, include_examples: bool = True) -> dict:
    """Run a bounded, read-only diagnostic pass over one labeled split."""
    if split not in SPLITS: raise ValueError(split)
    started=time.perf_counter(); before=snapshot_frozen(); output_dir.mkdir(parents=True, exist_ok=True)
    ids, s2, s3 = _cache_for(split); policy=load_policy(); truth=load_ground_truth(GROUND_TRUTH, ids)
    if tuple(ids) != s2.s1_ids or tuple(ids) != s3.s1_ids: raise ValueError("score cache S1 coverage mismatch")
    pre=_selected_scores(s2,s3,policy); predictions=predictions_for_policy(s2,s3,policy)
    official=score_predictions(truth,predictions,ids)
    ownership, removed_truth, contested_targets=ownership_summary(pre,predictions,truth,ids)
    failures=Counter(); by_source=Counter(); score_bins=defaultdict(Counter); per_s1=Counter(); candidate_table=defaultdict(Counter)
    fp_categories=Counter(); examples=defaultdict(list); interest=defaultdict(set); singleton=Counter()
    for i,s1 in enumerate(ids):
        scores={**_score_lookup(s2,i),**_score_lookup(s3,i)}
        candidate_count=len(scores); actual=truth[s1]; pred=predictions[s1]
        tp=actual & pred; fp=pred-actual; fn=actual-pred
        if not actual:
            singleton["true_singletons"] += 1
            singleton["correct_singletons"] += int(not pred)
            singleton["false_singleton_merges"] += int(bool(pred))
        elif not pred:
            singleton["non_singleton_predicted_empty"] += 1
        if actual==pred: per_s1["perfect_prediction"]+=1
        elif fp and fn: per_s1["both_fp_and_fn"]+=1
        elif fp: per_s1["fp_only"]+=1
        elif fn: per_s1["fn_only"]+=1
        if actual and not tp: per_s1["totally_missed_truth"]+=1
        if not actual and pred: per_s1["false_merge_true_singleton"]+=1
        cb=_candidate_bin(candidate_count)
        c=candidate_table[("candidate_count",cb)]
        c["s1_count"]+=1; c["candidate_count_sum"]+=candidate_count; c["f05_sum"]+=score_entity(actual,pred)
        c["tp"]+=len(tp); c["fp"]+=len(fp); c["fn"]+=len(fn)
        for target in actual:
            kind=classify_truth_link(target,scores,pred,policy); failures[kind]+=1; by_source[(_source(target),kind)]+=1
            interest[s1].add(target)
            score=scores.get(target)
            if kind=="A_blocking_miss": examples["blocking_miss"].append((s1,target,"",kind))
            elif kind=="B1_below_threshold":
                label="near_threshold_fn" if score is not None and score >= (policy.s2_threshold if target.startswith("S2-") else policy.s3_threshold)-.03 else "low_score_truth"
                examples[label].append((s1,target,score,kind))
                if target.startswith("S3-"): examples["s3_false_negative"].append((s1,target,score,kind))
            elif kind=="B2_ownership_conflict": examples["ownership_truth_removed"].append((s1,target,score,kind))
            if score is not None: score_bins[(_source(target),_score_bin(_source(target),score))]["truth_positive"]+=1
        for source, cache in (("S2",s2),("S3",s3)):
            targets, values=cache.group(i)
            for target_b, score_f in zip(targets,values):
                target=target_b.decode("ascii"); score=float(score_f); b=score_bins[(source,_score_bin(source,score))]
                if target in actual: continue
                b["retrieved_negative"]+=1
        for target in fp:
            interest[s1].add(target); score=scores[target]; src=_source(target)
            # Target is conflict-related when another selected S1 competed for it.
            label=classify_fp(target,score,policy.s2_threshold if src=="S2" else policy.s3_threshold,candidate_count,target in contested_targets)
            fp_categories[label]+=1; score_bins[(src,_score_bin(src,score))]["final_fp"]+=1
            if label=="C1_high_confidence": examples["high_confidence_fp"].append((s1,target,score,label))
            if not actual: examples["singleton_false_merge"].append((s1,target,score,label))
        for target in pred: interest[s1].add(target)
    # score-bin rates
    score_rows=[]
    for (source,bucket),x in sorted(score_bins.items()):
        pos=x["truth_positive"]; neg=x["retrieved_negative"]; fp=x["final_fp"]
        score_rows.append({"source":source,"score_bin":bucket,"truth_positives":pos,"retrieved_negatives":neg,"final_false_positives":fp,
                           "below_threshold_truth_fn":pos if bucket == IMMEDIATE_BELOW[source] else 0,
                           "precision_like": pos/(pos+neg) if pos+neg else ""})
    # route and interpretable feature pass: only positives plus final predictions, never all candidates.
    routes=Counter(); address_ranks=Counter(); slices=defaultdict(Counter)
    # Preserve only the first deterministic examples from each category; all large pair data stays streamed.
    example_lookup=defaultdict(list); example_rows=[]
    for category, values in examples.items():
        for s1_id, target, score, reason in values[:40]:
            example_lookup[(s1_id, target)].append((category, score, reason))
    records=load_s1_records(DATA,ids); store=TargetStore(TARGET_INDEX)
    try:
        paths=SPLITS[split]
        for pos,(s1,pairs) in enumerate(iter_union_groups(ids,paths["v1_candidates"],paths["v1_metadata"],paths["address"] )):
            if pos % 25000 == 0 and pos: print(f"Phase 10 {split}: route/features {pos:,}/{len(ids):,}",flush=True)
            left=records[s1]; actual=truth[s1]; pred=predictions[s1]
            csource=Counter(row["target_source"] for row in pairs.values())
            scores={**_score_lookup(s2,pos),**_score_lookup(s3,pos)}
            requested=sorted(interest[s1] | actual)
            targets=store.get_many(requested) if requested else {}
            for target in actual:
                pair=pairs.get(target)
                route="neither" if pair is None else ("both" if pair["from_v1"] and pair["from_address"] else ("v1_only" if pair["from_v1"] else "address_only"))
                routes[(_source(target),route,"truth")]+=1
                if pair and pair["from_address"]:
                    rank=int(float(pair["address_rank"])); address_ranks[(_source(target), "1" if rank==1 else "2-3" if rank<=3 else "4-5" if rank<=5 else "6-10")]+=1
            for target in requested:
                right=targets.get(target)
                if right is None: raise ValueError(f"missing target in frozen index: {target}")
                pair=pairs.get(target)
                evidence=pair or {"from_v1":0,"from_address":0,"address_rank":"","address_score":""}
                evidence={**evidence,"candidate_count_source":csource[_source(target)]}
                feats=dict(zip(FEATURE_NAMES, extract_features(left,right,evidence,csource[_source(target)])))
                actual_link=target in actual; predicted=target in pred; retrieved=pair is not None
                for dim,bucket in _slice_values(feats,csource[_source(target)]): _add_metric(slices[(dim,bucket)],actual_link,predicted,retrieved)
                example_categories = list(example_lookup.get((s1,target), ()))
                if include_examples and actual_link and retrieved and not predicted and feats["address_ratio"]>=0.95 and feats["name_ratio"]<0.70:
                    example_categories.append(("strong_address_weak_name_fn", scores.get(target, ""), "B1_or_B2"))
                for category, score, reason in example_categories:
                    if sum(row["category"] == category for row in example_rows) >= 40: continue
                    example_rows.append({"category":category,"source1_entity_id":s1,"candidate_entity_id":target,"target_source":_source(target),"score":score,"reason":reason,
                        "from_v1":evidence["from_v1"],"from_address":evidence["from_address"],"address_rank":evidence["address_rank"],"address_score":evidence["address_score"],
                        "source1_name":left.name,"target_name":right.name,"source1_address":left.address,"target_address":right.address,
                        "source1_name_normalized":left.basic,"target_name_normalized":right.basic,"source1_address_normalized":left.address_basic,"target_address_normalized":right.address_basic,
                        "name_ratio":feats["name_ratio"],"address_ratio":feats["address_ratio"],"numeric_shared":feats["numeric_shared"],"numeric_conflict":feats["numeric_conflict"],"postal_shared":feats["postal_shared"],"postal_conflict":feats["postal_conflict"]})
    finally: store.close()
    # Keep required display buckets explicit even when a split has zero support.
    for dimension, buckets in SLICE_BUCKETS.items():
        for bucket in buckets:
            slices[(dimension, bucket)] += Counter()
    # outputs
    split_dir=output_dir / split; split_dir.mkdir(parents=True,exist_ok=True)
    score_fields=["source","score_bin","truth_positives","retrieved_negatives","final_false_positives","below_threshold_truth_fn","precision_like"]
    _rows(split_dir/"score_bins.tsv",score_rows,score_fields)  # legacy combined compatibility output
    for source in ("S2", "S3"):
        _rows(split_dir/f"score_bins_{source.lower()}.tsv", [row for row in score_rows if row["source"] == source], score_fields)
    route_rows=[{"source":s,"route":r,"kind":k,"links":v} for (s,r,k),v in sorted(routes.items())]
    _rows(split_dir/"retrieval_routes.tsv",route_rows,["source","route","kind","links"])
    address_rank_rows=address_rank_distribution(address_ranks, truth)
    _rows(split_dir/"address_ranks.tsv",address_rank_rows,["source","rank_bin","truth_hits","cumulative_cutoff","cumulative_truth_hits","share_of_address_ranked_truth_hits","cumulative_share_of_address_ranked_truth_hits","share_of_all_source_truth_links","cumulative_share_of_all_source_truth_links"])
    _rows(split_dir/"address_rank_distribution.tsv",address_rank_rows,["source","rank_bin","truth_hits","cumulative_cutoff","cumulative_truth_hits","share_of_address_ranked_truth_hits","cumulative_share_of_address_ranked_truth_hits","share_of_all_source_truth_links","cumulative_share_of_all_source_truth_links"])
    _rows(split_dir/"slice_performance.tsv",_finalize_metrics(slices),["dimension","bucket","support","retrieved_truth","tp","fn","fp","retrieval_recall","model_policy_recall_given_retrieved","precision_diagnostic"])
    candidate_rows=[]
    for (_,bucket),x in sorted(candidate_table.items()):
        n=x["s1_count"]; tp=x["tp"];fp=x["fp"];fn=x["fn"]
        candidate_rows.append({"candidate_count_bin":bucket,"s1_count":n,"mean_candidates":x["candidate_count_sum"]/n,"macro_f0_5":x["f05_sum"]/n,"tp":tp,"fp":fp,"fn":fn,"precision":tp/(tp+fp) if tp+fp else "","recall":tp/(tp+fn) if tp+fn else ""})
    _rows(split_dir/"candidate_count_analysis.tsv",candidate_rows,["candidate_count_bin","s1_count","mean_candidates","macro_f0_5","tp","fp","fn","precision","recall"])
    per_f05=Counter(); per_f05_loss=Counter()
    for s1 in ids:
        value=score_entity(truth[s1],predictions[s1]); bucket=_f05_bin(value); per_f05[bucket] += 1; per_f05_loss[bucket] += 1.0-value
    _rows(split_dir/"per_s1_f05_distribution.tsv",[{"f05_bin":k,"s1_count":v,"share":v/len(ids),"macro_loss_contribution":per_f05_loss[k]/len(ids)} for k,v in sorted(per_f05.items())],["f05_bin","s1_count","share","macro_loss_contribution"])
    conflict_free={s:set(v) for s,v in pre.items()}; unconstrained=score_predictions(truth,conflict_free,ids)
    own_row={**ownership,"macro_f0_5_with_conflict":official.macro_f0_5,"macro_f0_5_without_conflict":unconstrained.macro_f0_5,"macro_delta":official.macro_f0_5-unconstrained.macro_f0_5}
    _rows(split_dir/"ownership_conflicts.tsv",[own_row],list(own_row))
    _rows(split_dir/"loss_accounting.tsv",[
        {"scope":"all","tp":official.tp,"fp":official.fp,"fn":official.fn,"macro_f0_5":official.macro_f0_5,
         "blocking_miss":failures["A_blocking_miss"],"below_threshold":failures["B1_below_threshold"],"ownership_conflict":failures["B2_ownership_conflict"],"other_policy":failures["B3_other"]},
        *[{"scope":src, "tp":"", "fp":"", "fn":"", "macro_f0_5":"", "blocking_miss":by_source[(src,"A_blocking_miss")],"below_threshold":by_source[(src,"B1_below_threshold")],"ownership_conflict":by_source[(src,"B2_ownership_conflict")],"other_policy":by_source[(src,"B3_other")]} for src in ("S2","S3")]],
          ["scope","tp","fp","fn","macro_f0_5","blocking_miss","below_threshold","ownership_conflict","other_policy"])
    _rows(split_dir/"per_s1_error_categories.tsv",[{"category":k,"s1_count":v,"share":v/len(ids)} for k,v in sorted(per_s1.items())],["category","s1_count","share"])
    singleton_rows=[{"category":k,"s1_count":v,"share":v/len(ids)} for k,v in sorted(singleton.items())]
    _rows(split_dir/"singleton_analysis.tsv",singleton_rows,["category","s1_count","share"])
    source_gap=[]
    for src in ("S2","S3"):
        total=sum(1 for links in truth.values() for target in links if target.startswith(src+"-"))
        block=by_source[(src,"A_blocking_miss")]; below=by_source[(src,"B1_below_threshold")]; own=by_source[(src,"B2_ownership_conflict")]
        tp=sum(count_link_errors(truth[s1],predictions[s1],src+"-").tp for s1 in ids); fp=sum(count_link_errors(truth[s1],predictions[s1],src+"-").fp for s1 in ids); fn=total-tp
        source_gap.append({"source":src,"true_links":total,"retrieval_recall":(total-block)/total if total else 0,"model_policy_recall_given_retrieved":tp/(total-block) if total-block else 0,"tp":tp,"fp":fp,"fn":fn,"blocking_misses":block,"below_threshold":below,"ownership_conflicts":own})
    _rows(split_dir/"s2_s3_gap.tsv",source_gap,["source","true_links","retrieval_recall","model_policy_recall_given_retrieved","tp","fp","fn","blocking_misses","below_threshold","ownership_conflicts"])
    _rows(split_dir/"false_positive_categories.tsv",[{"category":k,"links":v,"share_of_fp":v/official.fp if official.fp else 0} for k,v in sorted(fp_categories.items())],["category","links","share_of_fp"])
    if include_examples:
        fields=["category","source1_entity_id","candidate_entity_id","target_source","score","reason","from_v1","from_address","address_rank","address_score","source1_name","target_name","source1_address","target_address","source1_name_normalized","target_name_normalized","source1_address_normalized","target_address_normalized","name_ratio","address_ratio","numeric_shared","numeric_conflict","postal_shared","postal_conflict"]
        _rows(split_dir/"examples.tsv",example_rows,fields)
    result={"split":split,"s1_count":len(ids),"metrics":asdict(official),"failure_stages":dict(failures),"failure_stages_by_source":{"%s:%s"%k:v for k,v in by_source.items()},"false_positive_categories":dict(fp_categories),"ownership":own_row,"runtime_seconds":time.perf_counter()-started,"peak_rss_mb":rss_mb(),"artifacts":str(split_dir)}
    (split_dir/"summary.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    assert_unchanged(before); return result


def inventory(output_dir: Path = OUT) -> dict:
    before=snapshot_frozen(); policy=load_policy(); entries=[]
    for split,paths in SPLITS.items():
        for key,path in paths.items():
            if isinstance(path,Path): entries.append({"split":split,"artifact":key,"path":str(path),"exists":path.exists(),"bytes":path.stat().st_size if path.exists() else 0})
    result={"phase":10,"policy":asdict(policy),"artifacts":entries,
            "reusable":{"tune":"Phase 9 compact S2/S3 score caches + frozen route streams","validation":"cached frozen score files + V1 metadata + Address K10 route ranks","phase7":"labeled sampled pairs and feature matrices for training-only context"},
            "not_rebuilt":["Phase 12 candidates","Phase 12 scores","retrieval","LightGBM models"],"frozen_snapshot":before}
    output_dir.mkdir(parents=True,exist_ok=True); (output_dir/"inventory.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    assert_unchanged(before); return result


def compare(output_dir: Path = OUT) -> dict:
    """Classify aggregate findings only; no validation-driven selection is made."""
    summaries={s:json.loads((output_dir/s/"summary.json").read_text(encoding="utf-8")) for s in SPLITS}
    rows=[]
    # TP is an outcome, not a failure stage. Restrict comparison shares to the
    # mutually exclusive A/B loss taxonomy so every share is relative to FN.
    keys=sorted(key for key in (
        set(summaries["tune"]["failure_stages"])
        | set(summaries["validation"]["failure_stages"])
    ) if key != "TP")
    for key in keys:
        t=summaries["tune"]["failure_stages"].get(key,0); v=summaries["validation"]["failure_stages"].get(key,0)
        tr=t/max(1,summaries["tune"]["metrics"]["fn"]); vr=v/max(1,summaries["validation"]["metrics"]["fn"])
        classification="ROBUST" if abs(tr-vr)<=.05 else ("TUNE_ONLY" if tr>vr else "VALIDATION_ONLY")
        rows.append({"finding":key,"tune_count":t,"validation_count":v,"tune_share_fn":tr,"validation_share_fn":vr,"classification":classification})
    _rows(output_dir/"tune_validation_comparison.tsv",rows,["finding","tune_count","validation_count","tune_share_fn","validation_share_fn","classification"])
    result={"comparison":rows,"scope":"diagnostic comparison only; validation did not select a policy"}
    (output_dir/"comparison.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"); return result


def report(output_dir: Path = OUT) -> dict:
    tune=json.loads((output_dir/"tune/summary.json").read_text()); val=json.loads((output_dir/"validation/summary.json").read_text())
    tune_gap=list(csv.DictReader((output_dir/"tune/s2_s3_gap.tsv").open(encoding="utf-8"),delimiter="\t"))
    val_gap=list(csv.DictReader((output_dir/"validation/s2_s3_gap.tsv").open(encoding="utf-8"),delimiter="\t"))
    gap={"tune":{r["source"]:r for r in tune_gap},"validation":{r["source"]:r for r in val_gap}}
    lines=["# Phase 10 Error Analysis", "", "## Frozen inputs", "", "- Policy: S2 ≥ 0.93, S3 ≥ 0.97; highest-score target ownership.", "- This report is diagnostic only. No model, threshold, retrieval, score, or Phase 12 artifact was changed.", "", "## Name and address diagnostic buckets", "", "- Phase 8 RapidFuzz ratio features are stored on a 0–1 scale because native 0–100 values are divided by 100.", "- `exact_or_very_high`: existing exact feature or ratio ≥0.95; `medium`: 0.70–<0.95; `weak`: <0.70; `missing`: either normalized field is blank.", "", "## Top sources of remaining error", ""]
    for key,value in sorted(tune["failure_stages"].items(),key=lambda x:-x[1]): lines.append(f"- Tune `{key}`: {value:,} links.")
    rank_lines=[]
    for split in ("tune", "validation"):
        rank_rows=list(csv.DictReader((output_dir/split/"address_rank_distribution.tsv").open(encoding="utf-8"),delimiter="\t"))
        for source in ("S2", "S3"):
            values={r["cumulative_cutoff"]: r for r in rank_rows if r["source"] == source}
            if values:
                rank_lines.append(
                    f"- {split.title()} {source}: cumulative AddressK10 true-hit share `<=1` "
                    f"{float(values['<=1']['cumulative_share_of_address_ranked_truth_hits']):.2%}, "
                    f"`<=3` {float(values['<=3']['cumulative_share_of_address_ranked_truth_hits']):.2%}, "
                    f"`<=5` {float(values['<=5']['cumulative_share_of_address_ranked_truth_hits']):.2%}, "
                    f"`<=10` {float(values['<=10']['cumulative_share_of_address_ranked_truth_hits']):.2%}. "
                    "The same rows record shares of all source truth links.")
    lines += ["", "## Score-bin interpretation", "", "- Immediate pre-threshold FN bins are S2 0.90–<0.93 and S3 0.95–<0.97. These are diagnostics only; no threshold search was run.", "- Per-source files record retrieved truth positives, retrieved negatives, final FPs, and precision-like support.", "", "## AddressK10 rank distributions", "", *rank_lines, "", "## Validation reproduction", "", "See `tune_validation_comparison.tsv`; only ROBUST findings should motivate Phase 11 experiments.", "", "## What not to change", "", "- Do not change the frozen Phase 9 thresholds or ownership policy from validation diagnostics.", "- Do not reuse hidden-test outcomes for design choices.", "", "## Phase 11 opportunity matrix", "", "| Issue | Stage | Candidate experiment | Test reuse | Compute | Risk |", "|---|---|---|---|---|---|"]
    opportunities=[
      ("Retrieved truths below threshold", "model/feature", "Review hard-negative sampling and source-specific features", "existing candidates; rescore required", "feature+scoring", "medium"),
      ("Blocking misses", "retrieval", "Investigate only if robust route analysis shows address rank saturation", "no", "retrieval rerun", "high"),
      ("Ownership losses", "policy", "Tune-only ownership-margin experiment", "existing scores", "prediction only", "medium"),
      ("S3 conditional recall gap", "model", "S3-specific feature and hard-negative training experiment", "existing candidates", "feature+scoring", "medium"),
    ]
    for row in opportunities: lines.append("| "+" | ".join(row)+" |")
    s3_t, s3_v = gap["tune"]["S3"], gap["validation"]["S3"]
    opportunity_rows=[]
    for issue,stage,experiment,reuse,compute,risk in opportunities:
        if issue == "Blocking misses":
            tune_evidence=tune["failure_stages"].get("A_blocking_miss",""); validation_evidence=val["failure_stages"].get("A_blocking_miss",""); mechanism="retrieval miss"
        elif issue == "Retrieved truths below threshold":
            tune_evidence=tune["failure_stages"].get("B1_below_threshold",""); validation_evidence=val["failure_stages"].get("B1_below_threshold",""); mechanism="retrieved truth score below frozen threshold"
        elif issue == "Ownership losses":
            tune_evidence=tune["failure_stages"].get("B2_ownership_conflict",""); validation_evidence=val["failure_stages"].get("B2_ownership_conflict",""); mechanism="highest-score ownership removes retrieved truth"
        else:
            tune_evidence=f"blocking={s3_t['blocking_misses']}; below_threshold={s3_t['below_threshold']}; conditional_recall={float(s3_t['model_policy_recall_given_retrieved']):.6f}; S2={float(gap['tune']['S2']['model_policy_recall_given_retrieved']):.6f}"
            validation_evidence=f"blocking={s3_v['blocking_misses']}; below_threshold={s3_v['below_threshold']}; conditional_recall={float(s3_v['model_policy_recall_given_retrieved']):.6f}; S2={float(gap['validation']['S2']['model_policy_recall_given_retrieved']):.6f}"
            mechanism="S3 retrieved truths are more often below the frozen S3 threshold than S2"
        opportunity_rows.append({"issue":issue,"stage":stage,"tune_evidence":tune_evidence,"validation_confirmation":validation_evidence,"likely_mechanism":mechanism,"candidate_phase11_experiment":experiment,"existing_phase12_candidates_reusable":"yes" if reuse != "no" else "no","existing_phase12_scores_reusable":"yes" if compute == "prediction only" else "no","expected_compute_class":compute,"overfitting_risk":risk,"priority":"HIGH" if issue in ("Blocking misses","Retrieved truths below threshold") else "MEDIUM"})
    _rows(output_dir/"phase11_opportunities.tsv",opportunity_rows,["issue","stage","tune_evidence","validation_confirmation","likely_mechanism","candidate_phase11_experiment","existing_phase12_candidates_reusable","existing_phase12_scores_reusable","expected_compute_class","overfitting_risk","priority"])
    (output_dir/"phase10_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    result={"phase":10,"tune_macro_f0_5":tune["metrics"]["macro_f0_5"],"validation_macro_f0_5":val["metrics"]["macro_f0_5"],"report":str(output_dir/"phase10_report.md"),"opportunities":str(output_dir/"phase11_opportunities.tsv")}
    (output_dir/"phase10_summary.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"); return result


def main(argv: list[str] | None=None) -> int:
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="command",required=True)
    for name in ("inventory","analyze-tune","analyze-validation","compare","report"):
        q=sub.add_parser(name); q.add_argument("--output-dir",type=Path,default=OUT)
    args=p.parse_args(argv)
    if args.command=="inventory": result=inventory(args.output_dir)
    elif args.command=="analyze-tune": result=analyze_split("tune",args.output_dir)
    elif args.command=="analyze-validation": result=analyze_split("validation",args.output_dir)
    elif args.command=="compare": result=compare(args.output_dir)
    else: result=report(args.output_dir)
    print(json.dumps(result,indent=2,sort_keys=True)); return 0

if __name__=="__main__": raise SystemExit(main())
