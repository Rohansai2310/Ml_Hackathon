#!/usr/bin/env python3
"""Bounded tune-only SQLite FTS5 BM25 candidate-retrieval prototype."""
from __future__ import annotations
import argparse,csv,gzip,json,resource,sqlite3,time
from pathlib import Path
import numpy as np
from diagnostics import DEFAULT_DATASET_ROOT
from normalize import core_name_tokens,normalize_address,normalize_country
from scoring import load_id_file,load_ground_truth
from retrieval_diagnosis import _read_candidate_subset,_read_s1,_read_tfidf_subset,_candidate_stats,OUT,V1_CANDIDATES,TFIDF_PAIRS,TUNE_IDS,INDEX

def build(source_db:Path,fts_path:Path,batch_size:int=5000)->dict:
    if fts_path.exists(): raise FileExistsError(f"Refusing to overwrite {fts_path}")
    fts_path.parent.mkdir(parents=True,exist_ok=True); started=time.perf_counter()
    src=sqlite3.connect(f"file:{source_db.resolve()}?mode=ro",uri=True); dst=sqlite3.connect(fts_path)
    dst.execute("PRAGMA journal_mode=OFF"); dst.execute("PRAGMA synchronous=OFF"); dst.execute("PRAGMA temp_store=FILE")
    counts={}
    try:
        for source in ("S2","S3"):
            table="fts_"+source.lower()
            dst.execute(f"CREATE VIRTUAL TABLE {table} USING fts5(entity_id UNINDEXED,country,name,address,tokenize='unicode61 remove_diacritics 2')")
            cur=src.execute("SELECT entity_id,country,core,address FROM target WHERE source=? ORDER BY target",(source,)); n=0
            while True:
                records=cur.fetchmany(batch_size)
                if not records: break
                dst.executemany(f"INSERT INTO {table} VALUES (?,?,?,?)",records); n+=len(records)
            dst.commit(); counts[source]=n
        dst.execute("CREATE VIRTUAL TABLE vocab_s2 USING fts5vocab(fts_s2,'col')")
        vocab_count=dst.execute("SELECT count(*) FROM vocab_s2").fetchone()[0]; dst.commit()
    finally: src.close(); dst.close()
    return {"target_rows_by_source":counts,"s2_term_column_rows":vocab_count,"build_seconds":time.perf_counter()-started,"index_bytes":fts_path.stat().st_size,"peak_rss_mb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,"country_policy":"open-set exact country filter; separate S2/S3 FTS indexes"}

def quote(text:str)->str: return '"'+text.replace('"','""')+'"'
def make_query(country:str,field:str,tokens:list[str])->str:
    if not country or not tokens:return ""
    return f"country : {quote(country)} AND {field} : ("+" OR ".join(quote(t) for t in tokens)+")"

def run(dataset_root:Path,tune_ids:Path,gt_path:Path,v1_path:Path,tfidf_path:Path,fts_path:Path,out_dir:Path,sample_size:int=5000,seed:int=20260925,top_k:int=50)->dict:
    started=time.perf_counter(); out_dir.mkdir(parents=True,exist_ok=True)
    all_ids=load_id_file(tune_ids); rng=np.random.default_rng(seed); ids=sorted(rng.choice(np.asarray(all_ids),size=min(sample_size,len(all_ids)),replace=False).tolist())
    wanted=set(ids); records=_read_s1(dataset_root,ids); truth=load_ground_truth(gt_path,wanted); v1=_read_candidate_subset(v1_path,ids); tfidf=_read_tfidf_subset(tfidf_path,ids)
    base={s:set(v1[s]) for s in ids}; fdb=sqlite3.connect(f"file:{fts_path.resolve()}?mode=ro",uri=True); fdb.execute("PRAGMA cache_size=-131072")
    results={s:{"name":{},"address":{}} for s in ids}; calls=0; retrieval_started=time.perf_counter()
    for s1 in ids:
        row=records[s1]; country=normalize_country(row["country"])
        name_tokens=list(dict.fromkeys(t for t in core_name_tokens(row["business_name"]) if len(t)>=2))[:8]
        address_tokens=list(dict.fromkeys(t for t in normalize_address(row["business_address"]).split() if len(t)>=3 or (t.isdigit() and len(t)>=4)))[:8]
        for source in ("S2","S3"):
            table="fts_"+source.lower()
            for route,field,tokens,weights in (("name","name",name_tokens,(0,0,1,0)),("address","address",address_tokens,(0,0,0,1))):
                match=make_query(country,field,tokens)
                if not match: continue
                sql=f"SELECT entity_id,bm25({table},?,?,?,?) score FROM {table} WHERE {table} MATCH ? ORDER BY score,entity_id LIMIT ?"
                for rank,(entity,score) in enumerate(fdb.execute(sql,(*weights,match,top_k)),1):
                    results[s1][route][entity]=(source,float(score),rank)
                calls+=1
    fdb.close(); retrieval_seconds=time.perf_counter()-retrieval_started
    baseline=_candidate_stats(truth,base); sweep=[]
    for k in (10,20,30,50):
        for label,routes in (("name_only",("name",)),("address_only",("address",)),("name_plus_address",("name","address"))):
            cand={s:base[s].union(*(set(t for t,(_,_,rank) in results[s][route].items() if rank<=k) for route in routes)) for s in ids}
            m=_candidate_stats(truth,cand); add_pairs=m["candidate_pairs"]-baseline["candidate_pairs"]; add_links=m["recovered_links"]-baseline["recovered_links"]
            sweep.append({"k_per_route":k,"configuration":label,"candidate_recall":m["candidate_recall"],"s2_recall":m["source"]["S2"]["recall"],"s3_recall":m["source"]["S3"]["recall"],"v1_misses_recovered":add_links,"remaining_misses":m["blocking_misses"],"candidate_pairs":m["candidate_pairs"],"additional_candidate_pairs":add_pairs,"mean":m["mean"],"median":m["median"],"p90":m["p90"],"p95":m["p95"],"p99":m["p99"],"max":m["max"],"extra_candidates_per_new_true_link":add_pairs/add_links if add_links else None})
    meta=out_dir/"prototype_fts5_pairs.tsv.gz"
    with gzip.open(meta,"wt",encoding="utf-8",newline="") as h:
        w=csv.writer(h,delimiter="\t",lineterminator="\n"); w.writerow(["source1_entity_id","candidate_entity_id","target_source","name_bm25","name_rank","address_bm25","address_rank","already_in_v1"])
        for s1 in ids:
            targets=set(results[s1]["name"])|set(results[s1]["address"])
            for target in sorted(targets):
                n=results[s1]["name"].get(target); a=results[s1]["address"].get(target); source=(n or a)[0]
                w.writerow((s1,target,source,f"{n[1]:.8f}" if n else "",n[2] if n else "",f"{a[1]:.8f}" if a else "",a[2] if a else "",int(target in base[s1])))
    tfunion={s:base[s]|tfidf[s] for s in ids}
    report={"prototype":"SQLite_FTS5_BM25_name_address","s1_count":len(ids),"sample_seed":seed,"top_k_per_source_per_route":top_k,"v1":baseline,"existing_char_tfidf_k50_same_sample":_candidate_stats(truth,tfunion),"query_count":calls,"retrieval_seconds":retrieval_seconds,"queries_per_second":calls/max(retrieval_seconds,1e-9),"total_runtime_seconds":time.perf_counter()-started,"peak_rss_mb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,"index_bytes":fts_path.stat().st_size,"metadata_file":str(meta),"metadata_bytes":meta.stat().st_size,"sweep":sweep,"open_set_country_filter":True}
    (out_dir/"prototype_fts5_summary.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); return report

def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="cmd",required=True)
    b=sub.add_parser("build-index"); b.add_argument("--source-sqlite",type=Path,default=INDEX); b.add_argument("--fts-index",type=Path,default=OUT/"fts_target_index.sqlite")
    r=sub.add_parser("retrieve-sample")
    for name,default in (("--dataset-root",DEFAULT_DATASET_ROOT),("--tune-ids",TUNE_IDS),("--ground-truth",DEFAULT_DATASET_ROOT/"train/train_ground_truth.tsv"),("--v1-candidates",V1_CANDIDATES),("--tfidf-pairs",TFIDF_PAIRS),("--fts-index",OUT/"fts_target_index.sqlite"),("--output-dir",OUT)):
        r.add_argument(name,type=Path,default=default)
    r.add_argument("--sample-size",type=int,default=5000); r.add_argument("--seed",type=int,default=20260925); r.add_argument("--top-k",type=int,default=50)
    a=p.parse_args(); result=build(a.source_sqlite,a.fts_index) if a.cmd=="build-index" else run(a.dataset_root,a.tune_ids,a.ground_truth,a.v1_candidates,a.tfidf_pairs,a.fts_index,a.output_dir,a.sample_size,a.seed,a.top_k)
    print(json.dumps(result,indent=2,sort_keys=True))
if __name__=="__main__": main()
