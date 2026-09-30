#!/usr/bin/env python3
"""Tune-only diagnostics and bounded lexical retrieval prototypes for Phase 6B/6C.

The script never reads validation or test records. It uses the existing V1
SQLite inverted token postings and frozen Phase 6 tune artifacts.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sqlite3
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rapidfuzz.fuzz import ratio

from blocking import BASIC, TOKEN, key
from diagnostics import DEFAULT_DATASET_ROOT, CANDIDATE_HEADER
from normalize import (address_tokens, core_name, core_name_tokens,
                       extract_numeric_tokens, normalize_address,
                       normalize_country, normalize_name, postal_candidates)
from scoring import load_ground_truth, load_id_file

BASE = Path(__file__).resolve().parents[1]
ART = BASE / "artifacts"
TRAIN = DEFAULT_DATASET_ROOT / "train"
INDEX = ART / "blocking/v1_index.sqlite"
TUNE_IDS = ART / "splits/tune_s1_ids.txt"
V1_CANDIDATES = ART / "baseline/tune_candidates.tsv.gz"
TFIDF_PAIRS = ART / "tfidf_v2_tune/tune_tfidf_top50.tsv.gz"
OUT = ART / "retrieval_diagnosis"
INFO = TOKEN
TOKEN_DF_CAP = 20_000
POSTING_FETCH_CAP = 1_000
QUERY_TERMS = 4


def _percentile(values: list[int], pct: int) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.int64), pct)) if values else 0.0


def _read_s1(dataset_root: Path, ids: list[str]) -> dict[str, dict[str, str]]:
    wanted = set(ids)
    result = {}
    with (dataset_root / "train/train_source1.tsv").open(encoding="utf-8", newline="") as h:
        for row in csv.DictReader(h, delimiter="\t"):
            if row["entity_id"] in wanted:
                result[row["entity_id"]] = row
    if len(result) != len(wanted):
        raise ValueError(f"Missing selected S1 records: {len(wanted)-len(result)}")
    return result


def _read_candidate_subset(path: Path, ids: list[str]) -> dict[str, set[str]]:
    wanted = set(ids)
    result = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as h:
        r = csv.reader(h, delimiter="\t")
        if next(r, None) != CANDIDATE_HEADER:
            raise ValueError(f"Unexpected candidate header in {path}")
        for row in r:
            if row[0] in wanted:
                result[row[0]] = set(filter(None, row[1].split(",")))
    if result.keys() != wanted:
        raise ValueError(f"Candidate S1 coverage mismatch: {len(wanted-result.keys())} missing")
    return result


def _read_tfidf_subset(path: Path, ids: list[str]) -> dict[str, set[str]]:
    wanted = set(ids)
    result = {s1: set() for s1 in ids}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as h:
        r = csv.DictReader(h, delimiter="\t")
        required = {"source1_entity_id", "candidate_entity_id", "target_source", "tfidf_score", "tfidf_rank"}
        if set(r.fieldnames or ()) != required:
            raise ValueError(f"Unexpected TF-IDF columns in {path}: {r.fieldnames}")
        for row in r:
            if row["source1_entity_id"] in wanted:
                result[row["source1_entity_id"]].add(row["candidate_entity_id"])
    return result


def _candidate_stats(truth: dict[str, set[str]], candidates: dict[str, set[str]]) -> dict:
    counts = [len(candidates[s1]) for s1 in sorted(truth)]
    true_total = sum(map(len, truth.values()))
    recovered = sum(len(truth[s1] & candidates[s1]) for s1 in truth)
    by_source = {}
    for prefix, label in (("S2-", "S2"), ("S3-", "S3")):
        n = sum(t.startswith(prefix) for values in truth.values() for t in values)
        tp = sum(t.startswith(prefix) for s1 in truth for t in truth[s1] & candidates[s1])
        by_source[label] = {"true_links": n, "recovered": tp, "recall": tp / n if n else 0.0}
    return {"s1_entities": len(truth), "true_links": true_total, "recovered_links": recovered,
            "blocking_misses": true_total - recovered, "candidate_recall": recovered / true_total if true_total else 0.0,
            "source": by_source, "candidate_pairs": sum(counts), "mean": float(np.mean(counts)) if counts else 0.0,
            "median": _percentile(counts, 50), "p90": _percentile(counts, 90), "p95": _percentile(counts, 95),
            "p99": _percentile(counts, 99), "max": max(counts, default=0),
            "zero_candidate_pct": 100 * sum(n == 0 for n in counts) / len(counts) if counts else 0.0}


def audit(dataset_root: Path, index_path: Path, out_dir: Path) -> dict:
    """Join full training truth to S1/target country and audit target reuse."""
    start = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(prefix="er_structural_audit_", suffix=".sqlite", delete=False)
    tmp.close()
    db = sqlite3.connect(tmp.name)
    db.execute("PRAGMA journal_mode=OFF"); db.execute("PRAGMA synchronous=OFF")
    db.execute("PRAGMA temp_store=FILE"); db.execute("PRAGMA cache_size=-262144")
    db.execute("CREATE TABLE s1(id TEXT PRIMARY KEY,country TEXT)")
    db.execute("CREATE TABLE links(s1 TEXT,target TEXT)")
    labels = Counter()
    for source, filename in (("S1", "train_source1.tsv"), ("S2", "train_source2.tsv"), ("S3", "train_source3.tsv")):
        with (dataset_root / "train" / filename).open(encoding="utf-8", newline="") as h:
            rows = csv.DictReader(h, delimiter="\t")
            batch = []
            for row in rows:
                ctry = normalize_country(row["country"])
                labels[f"{source}:{ctry or '<blank>'}"] += 1
                if source == "S1":
                    batch.append((row["entity_id"], ctry))
                    if len(batch) == 20_000:
                        db.executemany("INSERT INTO s1 VALUES (?,?)", batch); batch.clear()
            if source == "S1" and batch:
                db.executemany("INSERT INTO s1 VALUES (?,?)", batch)
    db.commit()
    link_count = 0
    with (dataset_root / "train/train_ground_truth.tsv").open(encoding="utf-8", newline="") as h:
        batch = []
        for row in csv.DictReader(h, delimiter="\t"):
            batch.extend((row["source1_entity_id"], target) for target in row["matched_entity_ids"].split(",") if target)
            link_count += len(row["matched_entity_ids"].split(",")) if row["matched_entity_ids"] else 0
            if len(batch) >= 50_000:
                db.executemany("INSERT INTO links VALUES (?,?)", batch); batch.clear()
        if batch:
            db.executemany("INSERT INTO links VALUES (?,?)", batch)
    db.commit(); db.execute("CREATE INDEX links_target_s1 ON links(target,s1)"); db.commit()
    uri = f"file:{index_path.resolve()}?mode=ro"
    db.execute(f"ATTACH DATABASE '{uri}' AS ix")
    relations = defaultdict(Counter)
    q = """SELECT coalesce(t.source,substr(l.target,1,2)),
      CASE WHEN s.id IS NULL OR t.entity_id IS NULL OR coalesce(s.country,'')='' OR coalesce(t.country,'')=''
           THEN 'either_or_both_missing'
           WHEN lower(s.country)=lower(t.country) THEN 'equal_nonblank' ELSE 'unequal_nonblank' END,
      count(*) FROM links l LEFT JOIN s1 s ON s.id=l.s1
      LEFT JOIN ix.target t ON t.entity_id=l.target GROUP BY 1,2 ORDER BY 1,2"""
    for source, relation, count in db.execute(q):
        relations[source][relation] = count
    uniqueness, examples = {}, {}
    for source in ("S2", "S3"):
        q = """SELECT l.target,count(DISTINCT l.s1) n FROM links l
          JOIN ix.target t ON t.entity_id=l.target WHERE t.source=?
          GROUP BY l.target HAVING n>1 ORDER BY n DESC,l.target"""
        rows = db.execute(q, (source,)).fetchall()
        uniqueness[source] = {"target_ids_linked_to_multiple_s1": len(rows),
                              "maximum_distinct_s1_per_target": max((r[1] for r in rows), default=1)}
        examples[source] = [{"target_id": r[0], "distinct_s1": r[1]} for r in rows[:10]]
    result = {"country_relationships_by_target_source": {s: dict(c) for s,c in relations.items()},
              "observed_open_set_countries_by_source": {s: dict(sorted((k.split(':',1)[1],v) for k,v in labels.items() if k.startswith(s+':'))) for s in ("S1","S2","S3")},
              "target_uniqueness_by_source": uniqueness, "duplicate_examples": examples,
              "ground_truth_links": link_count, "runtime_seconds": round(time.perf_counter()-start,2),
              "country_rule": "normalized open-set equality; no country whitelist"}
    (out_dir / "structural_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    db.close(); Path(tmp.name).unlink(missing_ok=True)
    return result


def _indicators(s1: dict[str,str], target: dict[str,str]) -> dict[str,bool]:
    a, b = normalize_name(s1["business_name"]), normalize_name(target["name"])
    ca, cb = core_name(s1["business_name"]), core_name(target["name"])
    ta, tb = set(core_name_tokens(s1["business_name"])), set(core_name_tokens(target["name"]))
    union = ta | tb
    jac = len(ta & tb) / len(union) if union else 0.0
    nums_a, nums_b = set(extract_numeric_tokens(s1["business_address"])), set(extract_numeric_tokens(target["address"]))
    post_a = set(postal_candidates(s1["business_address"], s1["country"]))
    post_b = set(postal_candidates(target["address"], target["country"]))
    addr_a, addr_b = set(address_tokens(s1["business_address"])), set(address_tokens(target["address"]))
    addr_union = addr_a | addr_b
    addr_jac = len(addr_a & addr_b) / len(addr_union) if addr_union else 0.0
    latin_a = any("LATIN" in __import__('unicodedata').name(c,"" ) for c in a)
    other_a = any(c.isalpha() and "LATIN" not in __import__('unicodedata').name(c,"") for c in a)
    latin_b = any("LATIN" in __import__('unicodedata').name(c,"") for c in b)
    other_b = any(c.isalpha() and "LATIN" not in __import__('unicodedata').name(c,"") for c in b)
    return {
        "blank_s1_name": not a, "blank_target_name": not b,
        "exact_basic_name": bool(a and a == b), "exact_core_name": bool(ca and ca == cb),
        "strong_token_overlap_jaccard_ge_0_5": jac >= 0.5,
        "weak_or_no_token_overlap_jaccard_lt_0_2": jac < 0.2,
        "word_order_only_same_tokens": bool(ta and sorted(core_name_tokens(s1["business_name"])) == sorted(core_name_tokens(target["name"])) and ca != cb),
        "spacing_or_concatenation_only": bool(a and b and a.replace(" ", "") == b.replace(" ", "") and a != b),
        "heavy_character_difference_ratio_lt_60": bool(a and b and ratio(a,b) < 60),
        "different_script_possible_transliteration": bool((latin_a and other_b) or (latin_b and other_a)),
        "short_or_generic_name_le_one_core_token": min(len(ta),len(tb)) <= 1,
        "blank_s1_address": not normalize_address(s1["business_address"]),
        "blank_target_address": not normalize_address(target["address"]),
        "useful_address_token_overlap_jaccard_ge_0_5": addr_jac >= 0.5,
        "shared_house_or_numeric_tokens": bool(nums_a & nums_b),
        "conflicting_numeric_tokens": bool(nums_a and nums_b and not nums_a & nums_b),
        "shared_postal_candidate": bool(post_a & post_b),
        "conflicting_postal_candidates": bool(post_a and post_b and not post_a & post_b),
    }


def diagnose_misses(dataset_root: Path, tune_ids_path: Path, v1_path: Path,
                    tfidf_path: Path, index_path: Path, out_dir: Path,
                    sample_size: int = 500) -> dict:
    ids = load_id_file(tune_ids_path); wanted = set(ids); out_dir.mkdir(parents=True,exist_ok=True)
    truth = load_ground_truth(dataset_root / "train/train_ground_truth.tsv", wanted)
    v1 = _read_candidate_subset(v1_path, ids)
    tfidf = _read_tfidf_subset(tfidf_path, ids)
    remaining, recovered = [], []
    for s1 in ids:
        for target in truth[s1] - v1[s1]:
            pair = (s1,target)
            if target in tfidf[s1]: recovered.append(pair)
            else: remaining.append(pair)
    s1rows = _read_s1(dataset_root, ids)
    all_targets = sorted({t for _,t in remaining+recovered})
    target_rows = {}
    db = sqlite3.connect(f"file:{index_path.resolve()}?mode=ro", uri=True)
    for i in range(0,len(all_targets),800):
        batch=all_targets[i:i+800]
        sql="SELECT entity_id,source,name,address,country FROM target WHERE entity_id IN ("+",".join("?" for _ in batch)+")"
        target_rows.update({r[0]:{"entity_id":r[0],"source":r[1],"name":r[2] or "","address":r[3] or "","country":r[4] or ""} for r in db.execute(sql,batch)})
    db.close()
    def summarize(pairs):
        flags=Counter(); by_source=Counter()
        for s1,target in pairs:
            if target not in target_rows: continue
            by_source[target_rows[target]["source"]]+=1
            flags.update(k for k,v in _indicators(s1rows[s1],target_rows[target]).items() if v)
        return {"links_by_source":dict(by_source),"overlapping_indicator_counts":dict(sorted(flags.items()))}
    seed=20260925
    def chosen(pairs,n):
        return sorted(pairs,key=lambda p: hashlib.sha256(f"{seed}:{p[0]}:{p[1]}".encode()).digest())[:n]
    sample=chosen(remaining,sample_size); recovered_sample=chosen(recovered,min(100,sample_size))
    path=out_dir/"tune_k50_miss_examples.tsv"
    headers=["kind","source1_entity_id","target_entity_id","target_source","s1_name","target_name","s1_address","target_address","country_s1","country_target","normalized_name_s1","normalized_name_target","core_name_s1","core_name_target","tfidf_score","tfidf_rank","qualitative_flags"]
    # Pull scores/ranks only for bounded recovered examples.
    selected=set(recovered_sample); tfmeta={}
    if selected:
        with gzip.open(tfidf_path,"rt",encoding="utf-8",newline="") as h:
            for row in csv.DictReader(h,delimiter="\t"):
                keypair=(row["source1_entity_id"],row["candidate_entity_id"])
                if keypair in selected: tfmeta[keypair]=(row["tfidf_score"],row["tfidf_rank"])
    with path.open("w",encoding="utf-8",newline="") as h:
        w=csv.writer(h,delimiter="\t",lineterminator="\n"); w.writerow(headers)
        for kind,pairs in (("tfidf_recovered_v1_miss",recovered_sample),("still_missed_at_k50",sample)):
            for s1,target in pairs:
                t=target_rows.get(target)
                if t is None: continue
                flags=_indicators(s1rows[s1],t)
                score,rank=tfmeta.get((s1,target),("",""))
                w.writerow((kind,s1,target,t["source"],s1rows[s1]["business_name"],t["name"],s1rows[s1]["business_address"],t["address"],normalize_country(s1rows[s1]["country"]),normalize_country(t["country"]),normalize_name(s1rows[s1]["business_name"]),normalize_name(t["name"]),core_name(s1rows[s1]["business_name"]),core_name(t["name"]),score,rank,";".join(k for k,v in flags.items() if v)))
    result={"tune_s1_ids":len(ids),"true_links":sum(map(len,truth.values())),"v1_misses":len(remaining)+len(recovered),
            "v1_misses_recovered_by_tfidf_k50":len(recovered),"remaining_v1_misses_after_k50":len(remaining),
            "recovered_comparison":summarize(recovered),"remaining_miss_comparison":summarize(remaining),
            "categories_are_overlapping_heuristics_not_labels":True,"sample_size_remaining":len(sample),
            "sample_size_recovered":len(recovered_sample),"sample_file":str(path),"sample_seed":seed}
    (out_dir/"tune_k50_miss_summary.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return result


def prototype_a(dataset_root: Path, tune_ids_path: Path, gt_path: Path,
                index_path: Path, v1_path: Path, tfidf_path: Path,
                out_dir: Path, sample_size: int=5000, seed: int=20260925,
                top_k: int=50, posting_cap: int=POSTING_FETCH_CAP) -> dict:
    """Retrieve BM25-like ranked name-token postings from the existing index."""
    started=time.perf_counter(); out_dir.mkdir(parents=True,exist_ok=True)
    all_ids=load_id_file(tune_ids_path)
    rng=np.random.default_rng(seed); ids=sorted(rng.choice(np.asarray(all_ids),size=min(sample_size,len(all_ids)),replace=False).tolist())
    wanted=set(ids); rows=_read_s1(dataset_root,ids); truth=load_ground_truth(gt_path,wanted)
    v1=_read_candidate_subset(v1_path,ids); tfidf=_read_tfidf_subset(tfidf_path,ids)
    base={s1:set(v1[s1]) for s1 in ids}
    target_db=sqlite3.connect(f"file:{index_path.resolve()}?mode=ro",uri=True)
    target_db.execute("PRAGMA cache_size=-262144")
    retrieved=defaultdict(dict); query_seconds=0.0; postings_read=0; skipped_common=0; scored_terms=0
    target_n={"S2":5_034_616,"S3":5_285_603}
    for s1 in ids:
        row=rows[s1]; ctry=normalize_country(row["country"])
        toks=sorted(set(t for t in core_name_tokens(row["business_name"]) if len(t)>=3))
        for source in ("S2","S3"):
            choices=[]
            for token in toks:
                k=key(ctry,source,token)
                found=target_db.execute("SELECT n FROM frequency WHERE kind=? AND key=?",(TOKEN,k)).fetchone()
                if not found: continue
                df=int(found[0])
                if df>TOKEN_DF_CAP:
                    skipped_common+=1; continue
                idf=math.log(1.0+(target_n[source]+0.5)/(df+0.5))
                choices.append((df,token,idf,k))
            choices.sort(key=lambda v:(v[0],v[1]))
            choices=choices[:QUERY_TERMS]
            scores=defaultdict(float)
            tick=time.perf_counter()
            for df,token,idf,k in choices:
                scored_terms+=1
                cur=target_db.execute("SELECT target FROM posting INDEXED BY posting_lookup WHERE kind=? AND key=? ORDER BY target LIMIT ?",(TOKEN,k,posting_cap))
                for (tid,) in cur:
                    scores[tid]+=idf
                    postings_read+=1
            query_seconds+=time.perf_counter()-tick
            if scores:
                ranked_ids=sorted(scores,key=lambda tid:(-scores[tid],tid))[:top_k]
                # ID conversion is one bounded query rather than one lookup per candidate.
                target_map={r[0]:r[1] for i in range(0,len(ranked_ids),800) for r in target_db.execute("SELECT target,entity_id FROM target WHERE target IN ("+",".join("?" for _ in ranked_ids[i:i+800])+")",ranked_ids[i:i+800])}
                for rank,tid in enumerate(ranked_ids,1):
                    entity=target_map.get(tid)
                    if entity:
                        retrieved[s1][entity]=(source,float(scores[tid]),rank)
    target_db.close()
    union={s1:base[s1]|set(retrieved[s1]) for s1 in ids}
    metrics=_candidate_stats(truth,union); base_metrics=_candidate_stats(truth,base)
    char_subset={s1:base[s1]|tfidf[s1] for s1 in ids}; char_metrics=_candidate_stats(truth,char_subset)
    added=sum(len(union[s1]-base[s1]) for s1 in ids); recovered=metrics["recovered_links"]-base_metrics["recovered_links"]
    out_pairs=out_dir/"prototype_a_added_candidates.tsv.gz"
    with gzip.open(out_pairs,"wt",encoding="utf-8",newline="") as h:
        w=csv.writer(h,delimiter="\t",lineterminator="\n"); w.writerow(["source1_entity_id","candidate_entity_id","target_source","retrieval_route","bm25_like_score","rank"])
        for s1 in ids:
            for entity,(source,score,rank) in sorted(retrieved[s1].items(),key=lambda x:(x[1][0],x[1][2],x[0])):
                if entity not in base[s1]: w.writerow((s1,entity,source,"bm25_name_token",f"{score:.6f}",rank))
    result={"prototype":"A_inverted_lexical_bm25_like_name_tokens","sample_size":len(ids),"seed":seed,"top_k_per_source":top_k,
            "max_target_document_frequency":TOKEN_DF_CAP,"max_postings_fetched_per_term":posting_cap,"query_terms_per_source":QUERY_TERMS,
            "v1":base_metrics,"existing_tfidf_k50_on_same_sample":char_metrics,"prototype_union":metrics,
            "additional_candidate_pairs":added,"additional_true_links_recovered":recovered,
            "extra_candidates_per_new_true_link":added/recovered if recovered else None,
            "posting_rows_examined":postings_read,"token_term_lookups":scored_terms,"common_tokens_skipped":skipped_common,
            "posting_lookup_seconds":round(query_seconds,3),"total_runtime_seconds":round(time.perf_counter()-started,3),
            "throughput_s1_per_second":len(ids)/max(time.perf_counter()-started,1e-9),
            "output_file":str(out_pairs),"output_bytes":out_pairs.stat().st_size,
            "index_bytes":index_path.stat().st_size}
    (out_dir/"prototype_a_summary.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return result



def prototype_char(dataset_root: Path, tune_ids_path: Path, gt_path: Path,
                   index_path: Path, v1_path: Path, tfidf_path: Path,
                   out_dir: Path, sample_size: int=5000, seed: int=20260925,
                   max_df_ratio: float=0.005, top_k: int=50) -> dict:
    """Rebuild only an isolated sparse index and benchmark tune S1s."""
    from tfidf_retrieval import build_index, retrieve_topk
    started=time.perf_counter(); out_dir.mkdir(parents=True,exist_ok=True)
    all_ids=load_id_file(tune_ids_path); rng=np.random.default_rng(seed)
    ids=sorted(rng.choice(np.asarray(all_ids),size=min(sample_size,len(all_ids)),replace=False).tolist())
    wanted=set(ids); rows=sorted((r[0],r[1],r[2],r[3]) for r in
        __import__("blocking").selected_source1(dataset_root/"train/train_source1.tsv",wanted))
    if len(rows)!=len(ids): raise ValueError("Tune sample does not match training S1 records")
    index_dir=out_dir/f"tfidf_index_df{int(round(max_df_ratio*1000)):03d}"
    index_report=build_index(index_path,index_dir,max_df_ratio=max_df_ratio)
    retrieval=retrieve_topk(index_dir,rows,out_dir/"prototype_char_topk.tsv.gz",max_k=top_k)
    truth=load_ground_truth(gt_path,wanted); v1=_read_candidate_subset(v1_path,ids)
    tf={s:set() for s in ids}
    with gzip.open(out_dir/"prototype_char_topk.tsv.gz","rt",encoding="utf-8",newline="") as h:
        for row in csv.DictReader(h,delimiter="\t"): tf[row["source1_entity_id"]].add(row["candidate_entity_id"])
    base_metrics=_candidate_stats(truth,v1); union={s:v1[s]|tf[s] for s in ids}; metrics=_candidate_stats(truth,union)
    added=metrics["candidate_pairs"]-base_metrics["candidate_pairs"]; recovered=metrics["recovered_links"]-base_metrics["recovered_links"]
    result={"prototype":"char_wb_3_5_sparse_cosine","sample_size":len(ids),"seed":seed,"max_df_ratio":max_df_ratio,
            "index":index_report,"retrieval":retrieval,"v1":base_metrics,"v2_union":metrics,
            "additional_true_links":recovered,"additional_candidate_pairs":added,
            "extra_candidates_per_new_true_link":added/recovered if recovered else None,
            "total_runtime_seconds":time.perf_counter()-started,"index_bytes":sum(x.stat().st_size for x in index_dir.iterdir() if x.is_file()),
            "output_bytes":(out_dir/"prototype_char_topk.tsv.gz").stat().st_size}
    (out_dir/f"prototype_char_df{int(max_df_ratio*10000):04d}_summary.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return result

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="cmd",required=True)
    for name in ("audit","misses","prototype-a","prototype-char"):
        s=sub.add_parser(name)
        s.add_argument("--dataset-root",type=Path,default=DEFAULT_DATASET_ROOT)
        s.add_argument("--tune-ids",type=Path,default=TUNE_IDS)
        s.add_argument("--index",type=Path,default=INDEX)
        s.add_argument("--v1-candidates",type=Path,default=V1_CANDIDATES)
        s.add_argument("--tfidf-pairs",type=Path,default=TFIDF_PAIRS)
        s.add_argument("--output-dir",type=Path,default=OUT)
    sub.choices["prototype-a"].add_argument("--sample-size",type=int,default=5000)
    sub.choices["prototype-a"].add_argument("--seed",type=int,default=20260925)
    sub.choices["prototype-a"].add_argument("--top-k",type=int,default=50)
    sub.choices["prototype-a"].add_argument("--posting-cap",type=int,default=POSTING_FETCH_CAP)
    sub.choices["misses"].add_argument("--sample-size",type=int,default=500)
    sub.choices["prototype-char"].add_argument("--sample-size",type=int,default=5000)
    sub.choices["prototype-char"].add_argument("--seed",type=int,default=20260925)
    sub.choices["prototype-char"].add_argument("--top-k",type=int,default=50)
    sub.choices["prototype-char"].add_argument("--max-df-ratio",type=float,default=0.005)
    a=p.parse_args(argv)
    if a.cmd=="audit": result=audit(a.dataset_root,a.index,a.output_dir)
    elif a.cmd=="misses": result=diagnose_misses(a.dataset_root,a.tune_ids,a.v1_candidates,a.tfidf_pairs,a.index,a.output_dir,a.sample_size)
    elif a.cmd=="prototype-char": result=prototype_char(a.dataset_root,a.tune_ids,a.dataset_root/"train/train_ground_truth.tsv",a.index,a.v1_candidates,a.tfidf_pairs,a.output_dir,a.sample_size,a.seed,a.max_df_ratio,a.top_k)
    else: result=prototype_a(a.dataset_root,a.tune_ids,a.dataset_root/"train/train_ground_truth.tsv",a.index,a.v1_candidates,a.tfidf_pairs,a.output_dir,a.sample_size,a.seed,a.top_k,a.posting_cap)
    print(json.dumps(result,indent=2,sort_keys=True)); return 0


if __name__=="__main__":
    raise SystemExit(main())
