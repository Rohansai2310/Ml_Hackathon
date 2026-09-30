#!/usr/bin/env python3
"""Phase 11: controlled tune-selected model/feature experiments.

This runner is deliberately retrieval-free.  It trains only from frozen Phase 7
pairs, scores the frozen V1 union Address K10 candidates, and keeps validation
behind a one-shot locked-winner gate.
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
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping

import lightgbm as lgb
import numpy as np

from baseline import TargetStore, load_s1_records
from diagnostics import DEFAULT_DATASET_ROOT
from phase8_model import (
    FEATURE_NAMES, MODEL_CONFIGS, PAIR_REQUIRED_COLUMNS, SCORE_HEADER,
    TARGET_INDEX, TUNE_ADDRESS, TUNE_V1_CANDIDATES, TUNE_V1_METADATA,
    extract_features, internal_s1_split, iter_union_groups, union_candidate_counts,
)
from phase9_policy import DecisionPolicy, ScoreCache, evaluate_policy, load_cache, read_score_file
from scoring import load_ground_truth, load_id_file, score_entity, score_predictions

BASE = Path(__file__).resolve().parents[1]
ART = BASE / "artifacts"
OUT = ART / "phase11_experiments"
PHASE7 = ART / "model_data/phase7"
PHASE8 = ART / "model/phase8"
PHASE9 = ART / "model/phase9"
PHASE12 = ART / "test_inference/phase12"
VALIDATION = ART / "validation_evaluation"
SPLITS = ART / "splits"
TRAIN_IDS = PHASE7 / "train_subset_ids.txt"
TUNE_IDS = SPLITS / "tune_s1_ids.txt"
VAL_IDS = SPLITS / "val_s1_ids.txt"
GROUND_TRUTH = DEFAULT_DATASET_ROOT / "train/train_ground_truth.tsv"
SEED = 20260925
THREADS = 4
BASELINE_MACRO = 0.8676284617025911
SIGNIFICANCE_GATE = 0.0015
EXPECTED_TUNE_ROWS = {"S2": 7_227_686, "S3": 7_323_701}
BASE_SCORE = {"S2": PHASE8 / "tune_scores_s2.tsv.gz", "S3": PHASE8 / "tune_scores_s3.tsv.gz"}
BASE_CACHE = {"S2": PHASE9 / "score_cache_s2.npz", "S3": PHASE9 / "score_cache_s3.npz"}
VALIDATION_SCORE = {"S2": VALIDATION / "validation_scores_s2.tsv.gz", "S3": VALIDATION / "validation_scores_s3.tsv.gz"}
VALIDATION_V1 = ART / "blocking/v1_validation_candidates.tsv.gz"
VALIDATION_META = ART / "blocking/v1_validation_metadata.tsv.gz"
VALIDATION_ADDRESS = VALIDATION / "address_top10_validation.tsv.gz"
POLICY_PATH = PHASE9 / "decision_policy.json"

FROZEN_PATHS = (
    PHASE8 / "model_s2.txt", PHASE8 / "model_s3.txt", PHASE8 / "feature_manifest.json",
    POLICY_PATH, TARGET_INDEX,
    PHASE12 / "candidates/candidate_pairs_long.tsv.gz", PHASE12 / "scores/score_manifest.json",
    BASE.parents[1] / "output/candidate_pairs.tsv", BASE.parents[1] / "output/matching_results.tsv",
)

EXTRA_FEATURE_NAMES = (
    "weak_or_missing_name_and_strong_address",
    "numeric_conflict_and_strong_address",
    "numeric_conflict_and_strong_name",
    "numeric_conflict_and_postal_shared",
    "best_name_or_address_ratio",
    "candidate_count_101_plus",
    "candidate_count_501_plus",
)
PHASE11_FEATURE_NAMES = FEATURE_NAMES + EXTRA_FEATURE_NAMES


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_state(path: Path) -> dict[str, object]:
    st = path.stat()
    return {"path": str(path), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": sha256(path) if st.st_size < 100_000_000 else "skipped_large"}


def frozen_snapshot() -> dict[str, dict[str, object]]:
    return {str(path): file_state(path) for path in FROZEN_PATHS if path.exists()}


def assert_unchanged(before: Mapping[str, Mapping[str, object]]) -> None:
    if dict(before) != frozen_snapshot():
        raise RuntimeError("a frozen Phase 8/9/12 artifact changed during Phase 11")


def frozen_policy() -> DecisionPolicy:
    raw = json.loads(POLICY_PATH.read_text(encoding="utf-8"))["policy"]
    policy = DecisionPolicy(float(raw["s2_threshold"]), float(raw["s3_threshold"]), raw.get("open_threshold"), raw["conflict_policy"], raw.get("conflict_margin"))
    expected = DecisionPolicy(.93, .97, None, "highest", None)
    if policy != expected:
        raise ValueError("Phase 11 requires frozen .93/.97 highest-owner policy")
    return policy


def _ids() -> tuple[list[str], list[str], list[str]]:
    train, tune, val = load_id_file(TRAIN_IDS), load_id_file(TUNE_IDS), load_id_file(VAL_IDS)
    if len(train) != 100_000 or len(set(train)) != 100_000 or set(train) & set(tune) or set(train) & set(val) or set(tune) & set(val):
        raise ValueError("Phase 7 training subset leaks into tune/validation or has invalid size")
    if len(tune) != 100_000:
        raise ValueError("expected exactly 100K tune IDs")
    return train, tune, val


def inventory(output_dir: Path = OUT) -> dict:
    before = frozen_snapshot(); train, tune, val = _ids(); policy = frozen_policy()
    manifest = json.loads((PHASE8 / "feature_manifest.json").read_text())
    if manifest.get("feature_count") != 48 or tuple(manifest.get("feature_order", ())) != FEATURE_NAMES:
        raise ValueError("Phase 8 manifest mismatch")
    models = {source: lgb.Booster(model_file=str(PHASE8 / f"model_{source.lower()}.txt")) for source in ("S2", "S3")}
    for source, model in models.items():
        if tuple(model.feature_name()) != FEATURE_NAMES:
            raise ValueError(f"frozen {source} model schema mismatch")
    caches = {s: load_cache(BASE_CACHE[s]) for s in ("S2", "S3")}
    if any(len(caches[s].targets) != EXPECTED_TUNE_ROWS[s] or caches[s].s1_ids != tuple(tune) for s in caches):
        raise ValueError("frozen tune score cache mismatch")
    free = os.statvfs(OUT.parent if OUT.parent.exists() else ART).f_bavail * os.statvfs(OUT.parent if OUT.parent.exists() else ART).f_frsize
    result = {"phase": 11, "train_s1": len(train), "tune_s1": len(tune), "validation_s1": len(val),
              "train_tune_overlap": len(set(train) & set(tune)), "train_validation_overlap": len(set(train) & set(val)),
              "policy": asdict(policy), "baseline_rows": EXPECTED_TUNE_ROWS,
              "free_disk_bytes": free, "frozen_snapshot": before,
              "expected_tune_score_bytes_per_both_source_variant": sum(p.stat().st_size for p in BASE_SCORE.values()),
              "frozen_artifacts_unchanged": True}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase11_inventory.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    assert_unchanged(before); return result


def baseline_check(output_dir: Path = OUT) -> dict:
    before = frozen_snapshot(); _, tune, _ = _ids(); policy = frozen_policy(); truth = load_ground_truth(GROUND_TRUTH, tune)
    s2, s3 = load_cache(BASE_CACHE["S2"]), load_cache(BASE_CACHE["S3"])
    result = evaluate_policy(s2, s3, truth, policy)
    if result["macro_f0_5"] != BASELINE_MACRO:
        raise RuntimeError(f"baseline macro mismatch: {result['macro_f0_5']} != {BASELINE_MACRO}")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"baseline_macro_f0_5": result["macro_f0_5"], "exact": True, "policy": asdict(policy),
              "tp": result["tp"], "fp": result["fp"], "fn": result["fn"],
              "rows": {"S2": len(s2.targets), "S3": len(s3.targets)}, "frozen_snapshot": before,
              "frozen_artifacts_unchanged": True}
    (output_dir / "baseline_reproduction.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    assert_unchanged(before); return report


def _pair_rows(source: str):
    path = PHASE7 / f"train_pairs_{source.lower()}.tsv.gz"
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != PAIR_REQUIRED_COLUMNS:
            raise ValueError("unexpected Phase 7 pair schema")
        for row in reader:
            if row["target_source"] != source:
                raise ValueError("mixed source training pairs")
            yield row


def _stable_hash(*values: str) -> int:
    return int.from_bytes(hashlib.sha256("\x1f".join(values).encode()).digest()[:8], "big")


def extra_features(base: np.ndarray) -> np.ndarray:
    """Seven Phase 11 interactions, derived only from frozen Phase 8 features."""
    ix = {name: i for i, name in enumerate(FEATURE_NAMES)}
    nratio, aratio = base[:, ix["name_ratio"]], base[:, ix["address_ratio"]]
    nmissing = (base[:, ix["name_missing_left"]] > 0) | (base[:, ix["name_missing_right"]] > 0)
    strong_address = aratio >= .95
    strong_name = nratio >= .95
    numeric_conflict = base[:, ix["numeric_conflict"]] > 0
    postal_shared = base[:, ix["postal_shared"]] > 0
    count = base[:, ix["candidate_count_source"]]
    values = np.column_stack((
        ((nratio < .70) | nmissing) & strong_address,
        numeric_conflict & strong_address,
        numeric_conflict & strong_name,
        numeric_conflict & postal_shared,
        np.maximum(nratio, aratio),
        count >= 101,
        count >= 501,
    )).astype(np.float32)
    return values


def feature_matrix(base: np.ndarray, feature_set: str) -> tuple[np.ndarray, tuple[str, ...]]:
    if feature_set == "base": return base, FEATURE_NAMES
    if feature_set == "phase11": return np.hstack((base, extra_features(base))), PHASE11_FEATURE_NAMES
    raise ValueError(f"unknown feature set: {feature_set}")


def _training_path(output_dir: Path, source: str) -> Path:
    return output_dir / "training_sampling" / f"prepared_{source.lower()}.npz"


def prepare_training(output_dir: Path = OUT, dataset_root: Path = DEFAULT_DATASET_ROOT) -> dict:
    """Build reusable Phase 7-only base feature arrays; never reads tune/validation labels."""
    before = frozen_snapshot(); train, _, _ = _ids(); records = load_s1_records(dataset_root, train)
    counts = union_candidate_counts(train, PHASE7 / "v1_candidates.tsv.gz", PHASE7 / "v1_metadata.tsv.gz", PHASE7 / "address_k10_candidates.tsv.gz")
    fit, _ = internal_s1_split(train); output_dir.mkdir(parents=True, exist_ok=True); (output_dir / "training_sampling").mkdir(exist_ok=True)
    result = {"sources": {}, "seed": SEED, "feature_count": len(FEATURE_NAMES)}
    for source in ("S2", "S3"):
        features=[]; labels=[]; s1s=[]; candidates=[]; negatives=[]; ccounts=[]
        store=TargetStore(TARGET_INDEX)
        try:
            batch=[]; position=0
            for row in _pair_rows(source):
                batch.append(row)
                if len(batch) < 25_000:
                    continue
                targets=store.get_many(sorted({r["candidate_entity_id"] for r in batch}))
                for item in batch:
                    target=targets.get(item["candidate_entity_id"])
                    if target is None: raise ValueError("missing Phase 7 target")
                    features.append(extract_features(records[item["source1_entity_id"]], target, item, counts[item["source1_entity_id"]][source]))
                    labels.append(int(item["label"])); s1s.append(item["source1_entity_id"]); candidates.append(item["candidate_entity_id"])
                    negatives.append(item["negative_reason"]); ccounts.append(counts[item["source1_entity_id"]][source])
                position += len(batch); batch=[]
                if position % 100_000 < 25_000: print(f"Prepared {source} {position:,} training rows", flush=True)
            if batch:
                targets=store.get_many(sorted({r["candidate_entity_id"] for r in batch}))
                for item in batch:
                    target=targets.get(item["candidate_entity_id"])
                    if target is None: raise ValueError("missing Phase 7 target")
                    features.append(extract_features(records[item["source1_entity_id"]], target, item, counts[item["source1_entity_id"]][source]))
                    labels.append(int(item["label"])); s1s.append(item["source1_entity_id"]); candidates.append(item["candidate_entity_id"])
                    negatives.append(item["negative_reason"]); ccounts.append(counts[item["source1_entity_id"]][source])
        finally: store.close()
        array=np.vstack(features).astype(np.float32); path=_training_path(output_dir,source)
        np.savez_compressed(path, features=array, labels=np.asarray(labels,dtype=np.int8), s1=np.asarray(s1s,dtype="S20"),
                            candidate=np.asarray(candidates,dtype="S20"), negative_reason=np.asarray(negatives,dtype="U32"),
                            candidate_count=np.asarray(ccounts,dtype=np.int32), fit=np.asarray([x in fit for x in s1s],dtype=np.bool_), feature_names=np.asarray(FEATURE_NAMES))
        result["sources"][source]={"rows":len(labels),"positives":int(sum(labels)),"negatives":int(len(labels)-sum(labels)),"path":str(path),"bytes":path.stat().st_size}
    assert_unchanged(before); (output_dir / "training_sampling/preparation.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    return result


def _load_training(output_dir: Path, source: str) -> dict[str, np.ndarray]:
    path = _training_path(output_dir, source)
    if not path.is_file(): raise FileNotFoundError(f"run prepare-training first: {path}")
    z=np.load(path, allow_pickle=False)
    if tuple(z["feature_names"].tolist()) != FEATURE_NAMES: raise ValueError("prepared feature schema mismatch")
    return {key:z[key] for key in z.files}


def _reason_priority(reason: str) -> int:
    # Higher values are harder Phase 7 categories.
    return {"v1_and_address":8,"v1_exact_name":7,"v1_core_name":6,"address_top_1_3":5,"address_top_4_10":4,"v1_address_evidence":3,"v1_name_token":2,"easy_random":1}.get(reason,0)


def sample_mask(data: Mapping[str, np.ndarray], variant: str) -> np.ndarray:
    labels=data["labels"]; s1=data["s1"]; cand=data["candidate"]; counts=data["candidate_count"]; reasons=data["negative_reason"]; x=data["features"]
    selected=np.asarray(labels == 1, dtype=bool); groups=defaultdict(list)
    for i in np.flatnonzero(labels == 0): groups[str(s1[i])].append(int(i))
    ix={name:i for i,name in enumerate(FEATURE_NAMES)}
    for entity, indices in groups.items():
        if variant == "A1":
            total=int(counts[indices[0]]); limit=6 if total <=100 else 10 if total <=500 else 14
            ordered=sorted(indices,key=lambda i:(-_reason_priority(str(reasons[i])),_stable_hash(str(SEED),entity,str(cand[i]))))
        elif variant == "A2":
            def hardness(i: int):
                row=x[i]; rank=row[ix["address_rank"]]; both=row[ix["from_v1_and_address"]]
                evidence=max(row[ix["name_ratio"]],row[ix["address_ratio"]])
                agreement=row[ix["postal_shared"]]+row[ix["numeric_shared"]]
                return (-both, rank if rank else 99.0, -evidence, -agreement, _stable_hash(str(SEED),entity,str(cand[i])))
            limit=12; ordered=sorted(indices,key=hardness)
        elif variant == "base":
            selected[indices]=True; continue
        else: raise ValueError(variant)
        selected[ordered[:limit]]=True
    return selected


def _params() -> dict:
    config=next(c for c in MODEL_CONFIGS if c["name"]=="wide_95")
    return {"objective":"binary","metric":["binary_logloss","auc"],"verbosity":-1,"learning_rate":config["learning_rate"],"num_leaves":config["num_leaves"],"min_data_in_leaf":config["min_data_in_leaf"],"feature_fraction":config["feature_fraction"],"bagging_fraction":config["bagging_fraction"],"bagging_freq":1,"lambda_l2":config["lambda_l2"],"seed":SEED,"feature_fraction_seed":SEED,"bagging_seed":SEED,"data_random_seed":SEED,"deterministic":True,"force_col_wise":True,"num_threads":THREADS}


def train_variant(variant: str, source: str, sampling: str, feature_set: str, output_dir: Path = OUT) -> dict:
    before=frozen_snapshot(); data=_load_training(output_dir,source); mask=sample_mask(data,sampling); x,names=feature_matrix(data["features"][mask],feature_set); y=data["labels"][mask]; fit=data["fit"][mask]
    # Preserve the Phase 8 final-model style: train on all selected Phase 7 rows at frozen selected rounds.
    rounds=1485 if source=="S2" else 1495
    train=lgb.Dataset(x,label=y,feature_name=list(names),free_raw_data=False)
    booster=lgb.train(_params(),train,num_boost_round=rounds)
    directory=output_dir/"models"/variant; directory.mkdir(parents=True,exist_ok=True)
    model_path=directory/f"model_{source.lower()}.txt"; booster.save_model(str(model_path))
    manifest={"variant":variant,"source":source,"sampling":sampling,"feature_set":feature_set,"feature_order":list(names),"training_rows":int(mask.sum()),"positives":int(y.sum()),"negatives":int(len(y)-y.sum()),"fit_s1":int(len(set(data['s1'][mask & data['fit']].tolist()))),"all_s1":int(len(set(data['s1'][mask].tolist()))),"rounds":rounds,"params":_params(),"model_path":str(model_path),"model_sha256":sha256(model_path),"runtime_seconds":None,"frozen_snapshot":before}
    (directory/f"training_{source.lower()}.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    assert_unchanged(before); return manifest


def _model_for(variant: str, source: str, output_dir: Path) -> tuple[lgb.Booster, tuple[str,...], Path]:
    if variant == "baseline":
        p=PHASE8/f"model_{source.lower()}.txt"; return lgb.Booster(model_file=str(p)), FEATURE_NAMES, p
    p=output_dir/"models"/variant/f"model_{source.lower()}.txt"
    if not p.is_file(): raise FileNotFoundError(f"missing trained {variant} {source} model")
    model=lgb.Booster(model_file=str(p)); return model, tuple(model.feature_name()), p


def _score_path(output_dir: Path, variant: str, split: str, source: str) -> Path:
    return output_dir/"models"/variant/"scores"/f"{split}_{source.lower()}.tsv.gz"


def _score_split(variant: str, split: str, changed_sources: set[str], output_dir: Path, dataset_root: Path = DEFAULT_DATASET_ROOT) -> dict:
    """Score changed sources only from frozen candidate streams; no retrieval is invoked."""
    before=frozen_snapshot(); policy=frozen_policy()
    if split=="tune":
        _, ids, _=_ids(); v1,meta,address=TUNE_V1_CANDIDATES,TUNE_V1_METADATA,TUNE_ADDRESS; expected=EXPECTED_TUNE_ROWS
    elif split=="validation":
        _,_,ids=_ids(); v1,meta,address=VALIDATION_V1,VALIDATION_META,VALIDATION_ADDRESS
        raw=json.loads((VALIDATION/"scoring_report.json").read_text()); expected={s:int(raw["rows"][s]) for s in ("S2","S3")}
    else: raise ValueError(split)
    models={s:_model_for(variant,s,output_dir) for s in changed_sources}
    records=load_s1_records(dataset_root,ids); counts=union_candidate_counts(ids,v1,meta,address); paths={s:_score_path(output_dir,variant,split,s) for s in changed_sources}
    for path in paths.values(): path.parent.mkdir(parents=True,exist_ok=True)
    rows=Counter(); start=time.perf_counter(); store=TargetStore(TARGET_INDEX)
    try:
        with __import__('contextlib').ExitStack() as stack:
            writers={}
            for source,path in paths.items():
                h=stack.enter_context(gzip.open(path,"wt",encoding="utf-8",newline="")); writers[source]=csv.DictWriter(h,fieldnames=SCORE_HEADER,delimiter="\t",lineterminator="\n"); writers[source].writeheader()
            for pos,(s1,pairs) in enumerate(iter_union_groups(ids,v1,meta,address),1):
                by_source={s:[p for p in pairs.values() if p["target_source"]==s] for s in changed_sources}
                wanted=sorted(p["candidate_entity_id"] for groups in by_source.values() for p in groups); targets=store.get_many(wanted) if wanted else {}
                for source,group in by_source.items():
                    if not group: continue
                    model,names,_=models[source]; base=np.vstack([extract_features(records[s1],targets[p["candidate_entity_id"]],p,counts[s1][source]) for p in group]); matrix,_=feature_matrix(base,"phase11" if names==PHASE11_FEATURE_NAMES else "base")
                    scores=model.predict(matrix)
                    for pair,score in zip(group,scores):
                        writers[source].writerow({"source1_entity_id":s1,"candidate_entity_id":pair["candidate_entity_id"],"score":f"{float(score):.10f}","from_v1":pair["from_v1"],"from_address":pair["from_address"],"address_rank":pair["address_rank"],"address_score":pair["address_score"]}); rows[source]+=1
                if pos % 10_000==0: print(f"Phase 11 {variant} {split}: scored {pos:,}/{len(ids):,} S1; pairs={sum(rows.values()):,}",flush=True)
    finally: store.close()
    if any(rows[s] != expected[s] for s in changed_sources): raise ValueError("experimental score row count mismatch")
    result={"variant":variant,"split":split,"changed_sources":sorted(changed_sources),"rows":dict(rows),"paths":{s:str(p) for s,p in paths.items()},"bytes":{s:p.stat().st_size for s,p in paths.items()},"seconds":time.perf_counter()-start,"peak_rss_mb":rss_mb(),"candidate_membership":"frozen V1 union AddressK10"}
    (output_dir/"models"/variant/f"score_{split}.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    assert_unchanged(before); return result


def _score_cache(variant: str, split: str, source: str, output_dir: Path, ids: list[str], expected_rows: int) -> ScoreCache:
    changed=_score_path(output_dir,variant,split,source)
    if changed.is_file(): return read_score_file(changed,source,ids,expected_rows)
    if split=="tune": return load_cache(BASE_CACHE[source])
    return read_score_file(VALIDATION_SCORE[source],source,ids,expected_rows)


def _source_metrics(s2: ScoreCache,s3: ScoreCache,truth: Mapping[str,set[str]],policy: DecisionPolicy,ids:list[str]) -> dict:
    result={}
    for source,cache in (("S2",s2),("S3",s3)):
        tp=fp=fn=0
        threshold=policy.s2_threshold if source=="S2" else policy.s3_threshold
        for i,s1 in enumerate(ids):
            targets,scores=cache.group(i); predicted={t.decode() for t,v in zip(targets,scores) if float(v)>=threshold}; actual={x for x in truth[s1] if x.startswith(source+'-')}; tp+=len(actual&predicted); fp+=len(predicted-actual); fn+=len(actual-predicted)
        result[source]={"tp":tp,"fp":fp,"fn":fn,"precision":tp/(tp+fp) if tp+fp else 0.0,"recall":tp/(tp+fn) if tp+fn else 0.0}
    return result


def evaluate_variant(variant: str, output_dir: Path = OUT) -> dict:
    before=frozen_snapshot(); _,ids,_=_ids(); truth=load_ground_truth(GROUND_TRUTH,ids); policy=frozen_policy()
    s2=_score_cache(variant,"tune","S2",output_dir,ids,EXPECTED_TUNE_ROWS["S2"]); s3=_score_cache(variant,"tune","S3",output_dir,ids,EXPECTED_TUNE_ROWS["S3"])
    evaluation=evaluate_policy(s2,s3,truth,policy); source=_source_metrics(s2,s3,truth,policy,ids)
    # Compact diagnostics based only on score caches and exact truth; no candidate replay.
    below=Counter(); immediate=Counter(); low=Counter(); candidate_bins=defaultdict(lambda:Counter())
    for i,s1 in enumerate(ids):
        pred=evaluation["predictions"][s1]; actual=truth[s1]; n=len(s2.group(i)[0])+len(s3.group(i)[0]); b="1-25" if n<=25 else "26-100" if n<=100 else "101-500" if n<=500 else "501+"
        candidate_bins[b]["s1"]+=1; candidate_bins[b]["f05"]+=score_entity(actual,pred)
        for cache,src,threshold,nearlo in ((s2,"S2",.93,.90),(s3,"S3",.97,.95)):
            scores={t.decode():float(v) for t,v in zip(*cache.group(i))}
            for target in actual:
                if target.startswith(src+'-') and target in scores and scores[target] < threshold:
                    below[src]+=1; low[src]+=int(scores[target] < .5); immediate[src]+=int(nearlo <= scores[target] < threshold)
    result={"variant":variant,"macro_f0_5":evaluation["macro_f0_5"],"delta_vs_baseline":evaluation["macro_f0_5"]-BASELINE_MACRO,"tp":evaluation["tp"],"fp":evaluation["fp"],"fn":evaluation["fn"],"precision":evaluation["micro_precision_diagnostic"],"recall":evaluation["micro_recall_diagnostic"],"predicted_links":evaluation["predicted_links"],"singleton_false_merges":evaluation["singleton_false_merges"],"source_metrics":source,"below_threshold":{s:{"total":below[s],"lt_050":low[s],"immediate":immediate[s]} for s in ("S2","S3")},"candidate_count_bins":[{"bin":b,"s1_count":x["s1"],"macro_f0_5":x["f05"]/x["s1"]} for b,x in sorted(candidate_bins.items())],"policy":asdict(policy),"frozen_artifacts_unchanged":True}
    (output_dir/"models"/variant/"tune_result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    assert_unchanged(before); return result


def _completed_tune_score(variant: str, sources: set[str], output_dir: Path) -> bool:
    report = output_dir / "models" / variant / "score_tune.json"
    return report.is_file() and all(_score_path(output_dir, variant, "tune", source).is_file() for source in sources)


def experiment_a(output_dir: Path = OUT, dataset_root: Path = DEFAULT_DATASET_ROOT) -> dict:
    """Run or safely resume A1/A2. Completed score artifacts are never recomputed."""
    before=frozen_snapshot(); results={}
    for variant in ("A1","A2"):
        if not _completed_tune_score(variant, {"S2", "S3"}, output_dir):
            for source in ("S2","S3"):
                model = output_dir / "models" / variant / f"model_{source.lower()}.txt"
                if not model.is_file():
                    train_variant(variant,source,variant,"base",output_dir)
            _score_split(variant,"tune",{"S2","S3"},output_dir,dataset_root)
        print(f"Phase 11 {variant}: evaluating completed tune scores", flush=True)
        results[variant]=evaluate_variant(variant,output_dir)
    assert_unchanged(before); return results


def experiment_b(output_dir: Path = OUT, dataset_root: Path = DEFAULT_DATASET_ROOT) -> dict:
    before=frozen_snapshot(); variant="B1"
    manifest={"phase":11,"feature_count":len(PHASE11_FEATURE_NAMES),"feature_order":list(PHASE11_FEATURE_NAMES),"base_feature_manifest":str(PHASE8/"feature_manifest.json"),"new_features":list(EXTRA_FEATURE_NAMES),"description":"Seven deterministic interactions derived from frozen Phase 8 features."}
    (output_dir/"models"/variant).mkdir(parents=True,exist_ok=True); (output_dir/"models"/variant/"feature_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    for source in ("S2","S3"): train_variant(variant,source,"base","phase11",output_dir)
    _score_split(variant,"tune",{"S2","S3"},output_dir,dataset_root); result=evaluate_variant(variant,output_dir)
    assert_unchanged(before); return result


def experiment_c(output_dir: Path = OUT, dataset_root: Path = DEFAULT_DATASET_ROOT) -> dict:
    """Evaluate S3-only substitutions and conditionally train one C1 combined S3 model."""
    before=frozen_snapshot(); _, ids, _ = _ids(); truth=load_ground_truth(GROUND_TRUTH, ids); policy=frozen_policy()
    available=[]
    for variant in ("A1","A2","B1"):
        p=output_dir/"models"/variant/"tune_result.json"
        if p.is_file(): available.append(json.loads(p.read_text()))
    if len(available)<3: raise FileNotFoundError("run experiment-a and experiment-b first")
    baseline_s2=load_cache(BASE_CACHE["S2"]); substitutions=[]
    for variant in ("A1","A2","B1"):
        s3=_score_cache(variant,"tune","S3",output_dir,ids,EXPECTED_TUNE_ROWS["S3"])
        e=evaluate_policy(baseline_s2,s3,truth,policy)
        substitutions.append({"variant":f"{variant}_S3_only","base_variant":variant,"macro_f0_5":e["macro_f0_5"],"delta_vs_baseline":e["macro_f0_5"]-BASELINE_MACRO,"tp":e["tp"],"fp":e["fp"],"fn":e["fn"],"precision":e["micro_precision_diagnostic"],"recall":e["micro_recall_diagnostic"],"singleton_false_merges":e["singleton_false_merges"],"predicted_links":e["predicted_links"]})
    strongest_a=max((x for x in available if x["variant"] in ("A1","A2")),key=lambda x:x["macro_f0_5"])
    b1=next(x for x in available if x["variant"]=="B1")
    baseline=json.loads((output_dir/"baseline_reproduction.json").read_text())
    gate=(strongest_a["delta_vs_baseline"]>=SIGNIFICANCE_GATE and b1["delta_vs_baseline"]>=SIGNIFICANCE_GATE and strongest_a["fp"]<=1.02*baseline["fp"] and b1["fp"]<=1.02*baseline["fp"])
    result={"s3_only_substitutions":substitutions,"combined_gate":gate}
    if gate:
        train_variant("C1","S3",strongest_a["variant"],"phase11",output_dir)
        _score_split("C1","tune",{"S3"},output_dir,dataset_root)
        result["combined_c1"]=evaluate_variant("C1",output_dir)
    else:
        result["combined_not_run_reason"]="A and B did not both meet tune gain/FP gate"
    (output_dir/"models/C1_gate.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    assert_unchanged(before); return result


def compare_tune(output_dir: Path = OUT) -> list[dict]:
    before=frozen_snapshot(); rows=[]
    for path in sorted((output_dir/"models").glob("*/tune_result.json")):
        raw=json.loads(path.read_text()); rows.append(raw)
    base=baseline_check(output_dir); rows.append({"variant":"baseline","macro_f0_5":BASELINE_MACRO,"delta_vs_baseline":0.0})
    rows.sort(key=lambda r:(-r["macro_f0_5"],-r.get("precision",0),r.get("singleton_false_merges",10**18),r.get("predicted_links",10**18),r["variant"]))
    fields=["variant","macro_f0_5","delta_vs_baseline","tp","fp","fn","precision","recall","singleton_false_merges","predicted_links"]
    with (output_dir/"tune_results.tsv").open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=fields,delimiter="\t",lineterminator="\n",extrasaction="ignore");w.writeheader();w.writerows(rows)
    slice_rows=[]
    for row in rows:
        for item in row.get("candidate_count_bins", []):
            slice_rows.append({"variant":row["variant"],"slice":"candidate_count","bucket":item["bin"],"s1_count":item["s1_count"],"macro_f0_5":item["macro_f0_5"],"value":""})
        for source, item in row.get("below_threshold", {}).items():
            for name, value in item.items():
                slice_rows.append({"variant":row["variant"],"slice":f"{source}_below_threshold","bucket":name,"s1_count":"","macro_f0_5":"","value":value})
    with (output_dir/"tune_slice_results.tsv").open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=["variant","slice","bucket","s1_count","macro_f0_5","value"],delimiter="\t",lineterminator="\n");w.writeheader();w.writerows(slice_rows)
    assert_unchanged(before); return rows


def lock_winner(output_dir: Path = OUT) -> dict:
    before=frozen_snapshot()
    if (output_dir/"locked_winner.json").exists() or (output_dir/"no_winner.json").exists(): raise RuntimeError("winner decision is already immutable")
    rows=compare_tune(output_dir); candidates=[r for r in rows if r["variant"]!="baseline" and r["delta_vs_baseline"]>=SIGNIFICANCE_GATE]
    if not candidates:
        payload={"status":"NO_WINNER","reason":f"no non-baseline variant reached +{SIGNIFICANCE_GATE:.4f} tune macro gate","baseline_macro_f0_5":BASELINE_MACRO,"frozen_snapshot":before}; (output_dir/"no_winner.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); assert_unchanged(before); return payload
    winner=candidates[0]; variant=winner["variant"]; models={}
    for source in ("S2","S3"):
        p=output_dir/"models"/variant/f"model_{source.lower()}.txt"
        if p.is_file(): models[source]={"path":str(p),"sha256":sha256(p)}
        else: models[source]={"path":str(PHASE8/f"model_{source.lower()}.txt"),"sha256":sha256(PHASE8/f"model_{source.lower()}.txt")}
    payload={"status":"LOCKED","variant":variant,"models":models,"tune_metrics":winner,"policy":asdict(frozen_policy()),"selection_rationale":f"highest tune macro meeting +{SIGNIFICANCE_GATE:.4f} gate","frozen_snapshot":before}
    (output_dir/"locked_winner.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); assert_unchanged(before); return payload


def confirm_validation(output_dir: Path = OUT, dataset_root: Path = DEFAULT_DATASET_ROOT) -> dict:
    before=frozen_snapshot(); lock=output_dir/"locked_winner.json"; done=output_dir/"validation_confirmation.json"
    if done.exists(): raise RuntimeError("validation confirmation is one-shot and already exists")
    if not lock.is_file(): raise FileNotFoundError("lock a tune winner before validation")
    winner=json.loads(lock.read_text()); variant=winner["variant"]; _,_,ids=_ids(); raw=json.loads((VALIDATION/"scoring_report.json").read_text()); expected={s:int(raw["rows"][s]) for s in ("S2","S3")}
    changed={s for s in ("S2","S3") if (output_dir/"models"/variant/f"model_{s.lower()}.txt").is_file()}
    _score_split(variant,"validation",changed,output_dir,dataset_root)
    truth=load_ground_truth(GROUND_TRUTH,ids); policy=frozen_policy(); s2=_score_cache(variant,"validation","S2",output_dir,ids,expected["S2"]);s3=_score_cache(variant,"validation","S3",output_dir,ids,expected["S3"]); e=evaluate_policy(s2,s3,truth,policy); delta=e["macro_f0_5"]-0.8700057896268494
    status="CONFIRMED" if delta>=0 else "MIXED" if delta>=-.0015 else "FAILED_TO_GENERALIZE"
    result={"variant":variant,"status":status,"macro_f0_5":e["macro_f0_5"],"delta_vs_frozen_validation":delta,"tp":e["tp"],"fp":e["fp"],"fn":e["fn"],"source_metrics":_source_metrics(s2,s3,truth,policy,ids),"policy":asdict(policy),"changed_sources":sorted(changed),"frozen_artifacts_unchanged":True}
    done.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); assert_unchanged(before); return result


def report(output_dir: Path = OUT) -> dict:
    lines=["# Phase 11 Controlled Experiments","","Phase 12 candidates were not regenerated. Changed models require future rescoring only.",""]
    if (output_dir/"tune_results.tsv").is_file(): lines += ["## Tune results","",(output_dir/"tune_results.tsv").read_text()]
    if (output_dir/"models/C1_gate.json").is_file(): lines += ["## S3-only experiment results","",(output_dir/"models/C1_gate.json").read_text()]
    if (output_dir/"locked_winner.json").is_file(): lines += ["## Locked winner","",(output_dir/"locked_winner.json").read_text()]
    if (output_dir/"validation_confirmation.json").is_file(): lines += ["## One-shot validation confirmation","",(output_dir/"validation_confirmation.json").read_text()]
    elif (output_dir/"no_winner.json").is_file(): lines += ["## Result","",(output_dir/"no_winner.json").read_text()]
    path=output_dir/"phase11_report.md"; path.write_text("\n".join(lines)+"\n")
    return {"report":str(path),"phase":11}


def main(argv: list[str] | None=None) -> int:
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="command",required=True)
    for name in ("inventory","baseline-check","prepare-training","experiment-a","experiment-b","experiment-c","evaluate-tune","compare-tune","lock-winner","confirm-validation","report"):
        q=sub.add_parser(name);q.add_argument("--output-dir",type=Path,default=OUT);q.add_argument("--dataset-root",type=Path,default=DEFAULT_DATASET_ROOT)
    a=p.parse_args(argv)
    dispatch={"inventory":inventory,"baseline-check":baseline_check,"prepare-training":prepare_training,"experiment-a":experiment_a,"experiment-b":experiment_b,"experiment-c":experiment_c,"compare-tune":compare_tune,"lock-winner":lock_winner,"confirm-validation":confirm_validation,"report":report}
    if a.command == "evaluate-tune":
        raise SystemExit("use experiment-a/experiment-b to evaluate completed variants safely")
    result=dispatch[a.command](a.output_dir,a.dataset_root) if a.command in ("prepare-training","experiment-a","experiment-b","experiment-c","confirm-validation") else dispatch[a.command](a.output_dir)
    print(json.dumps(result,indent=2,sort_keys=True,default=str));return 0
if __name__=="__main__": raise SystemExit(main())
