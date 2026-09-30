#!/usr/bin/env python3
"""Disk-backed multi-route candidate generation for validation Source 1 records."""
from __future__ import annotations
import argparse
import csv
import gzip
import json
import resource
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from normalize import address_tokens, core_name, core_name_tokens, extract_numeric_tokens, normalize_country, normalize_name, postal_candidates
from scoring import load_ground_truth, load_id_file
from diagnostics import DEFAULT_DATASET_ROOT, DEFAULT_S1_IDS, CANDIDATE_HEADER

BASE = Path(__file__).resolve().parents[1]
ROUTES = ("exact_name", "core_name", "rare_name_token", "address_numeric_route", "postal_route", "strong_address_route")
BASIC, CORE, TOKEN, NUMBER, POSTAL = range(5)
TARGET_HEADER = ["entity_id", "business_name", "business_address", "country"]
METADATA_HEADER = ["source1_entity_id", "candidate_entity_id", "target_source", *ROUTES, "shared_informative_token_count", "num_blocking_routes"]
MISS_HEADER = ["source1_entity_id", "true_target_id", "target_source", "source1_name", "target_name", "source1_address", "target_address", "source1_basic_name", "target_basic_name", "source1_core_name", "target_core_name", "source1_tokens", "target_tokens", "source1_country", "target_country"]

@dataclass(frozen=True)
class Limits:
    exact: int = 500
    core: int = 300
    rare_token: int = 200
    informative_token: int = 3000
    address_token: int = 20000
    address_number: int = 20000
    postal: int = 6000
    max_query_tokens: int = 4
    strong_number: int = 1000

def key(country: str, source: str, value: str) -> str:
    return country + "\x1f" + source + "\x1f" + value

def build_index(dataset_root: Path, db_path: Path, rebuild: bool = False) -> dict:
    """Index target names and address signals in SQLite using bounded batches."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists() and not rebuild:
        with sqlite3.connect(db_path) as db:
            metadata = dict(db.execute("SELECT key,value FROM metadata"))
            if metadata.get("complete") != "1" or metadata.get("version") != "v2":
                raise ValueError("Incomplete or stale index; use --rebuild-index")
            return {"reused": True, "target_rows": int(metadata["target_rows"]), "postings": int(metadata["postings"]), "seconds": 0.0, "index_bytes": db_path.stat().st_size}
    if db_path.exists():
        db_path.unlink()
    start = time.perf_counter()
    db = sqlite3.connect(db_path)
    try:
        db.execute("PRAGMA journal_mode=OFF")
        db.execute("PRAGMA synchronous=OFF")
        db.execute("PRAGMA temp_store=FILE")
        db.execute("PRAGMA cache_size=-131072")
        db.execute("CREATE TABLE target (target INTEGER PRIMARY KEY, entity_id TEXT UNIQUE, source TEXT, country TEXT, name TEXT, address TEXT, basic TEXT, core TEXT)")
        db.execute("CREATE TABLE posting (kind INTEGER, key TEXT, target INTEGER)")
        db.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        targets = postings = 0
        for source in ("S2", "S3"):
            path = dataset_root / "train" / ("train_source" + source[1] + ".tsv")
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle, delimiter="\t")
                if next(reader, None) != TARGET_HEADER:
                    raise ValueError("Unexpected target columns: " + str(path))
                target_batch, post_batch = [], []
                for row_number, row in enumerate(reader, 2):
                    if len(row) != 4 or not row[0].startswith(source + "-"):
                        raise ValueError("Malformed target row: " + str(path) + ":" + str(row_number))
                    entity_id, name, address, raw_country = row
                    country = normalize_country(raw_country)
                    basic, core = normalize_name(name), core_name(name)
                    targets += 1
                    target_batch.append((targets, entity_id, source, country, name, address, basic, core))
                    keys = set()
                    if basic:
                        keys.add((BASIC, basic))
                    if core:
                        keys.add((CORE, core))
                    keys.update((TOKEN, token) for token in core_name_tokens(name) if len(token) >= 3)
                    keys.update((NUMBER, number) for number in extract_numeric_tokens(address))
                    keys.update((POSTAL, postal) for postal in postal_candidates(address, country))
                    post_batch.extend((kind, key(country, source, value), targets) for kind, value in keys)
                    if len(target_batch) == 10000:
                        with db:
                            db.executemany("INSERT INTO target VALUES (?,?,?,?,?,?,?,?)", target_batch)
                            db.executemany("INSERT INTO posting VALUES (?,?,?)", post_batch)
                        postings += len(post_batch)
                        target_batch.clear(); post_batch.clear()
                if target_batch:
                    with db:
                        db.executemany("INSERT INTO target VALUES (?,?,?,?,?,?,?,?)", target_batch)
                        db.executemany("INSERT INTO posting VALUES (?,?,?)", post_batch)
                    postings += len(post_batch)
            print("Indexed " + source + ": " + format(targets, ",") + " targets, " + format(postings, ",") + " postings", flush=True)
        print("Creating posting index", flush=True)
        db.execute("CREATE INDEX posting_lookup ON posting(kind,key,target)")
        db.executemany("INSERT INTO metadata VALUES (?,?)", [("version","v2"),("complete","1"),("target_rows",str(targets)),("postings",str(postings))])
        db.commit()
        return {"reused": False, "target_rows": targets, "postings": postings, "seconds": time.perf_counter()-start, "index_bytes": db_path.stat().st_size}
    finally:
        db.close()


def ensure_frequency_table(db_path: Path) -> dict:
    """Materialize posting sizes once, so common-key checks avoid large fetches."""
    started=time.perf_counter()
    with sqlite3.connect(db_path) as db:
        exists=db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='frequency'").fetchone()
        if not exists:
            print("Building posting-frequency table",flush=True)
            db.execute("CREATE TABLE frequency AS SELECT kind,key,COUNT(*) AS n FROM posting GROUP BY kind,key")
            db.execute("CREATE UNIQUE INDEX frequency_lookup ON frequency(kind,key)")
            db.commit()
        keys=db.execute("SELECT COUNT(*) FROM frequency").fetchone()[0]
    return {"keys":keys,"seconds":time.perf_counter()-started,"reused":bool(exists)}


def index_statistics(db_path: Path, limits: Limits = Limits()) -> dict:
    """Measure posting sizes by target source and index kind from stored counts."""
    labels={BASIC:"exact_name",CORE:"core_name",TOKEN:"name_token",NUMBER:"address_number",POSTAL:"postal"}
    caps={BASIC:limits.exact,CORE:limits.core,TOKEN:limits.address_token,NUMBER:limits.address_number,POSTAL:limits.postal}
    summary={}
    with sqlite3.connect("file:"+str(db_path)+"?mode=ro",uri=True) as db:
        for kind,encoded,n in db.execute("SELECT kind,key,n FROM frequency ORDER BY kind,key"):
            parts=encoded.split("\x1f",2)
            if len(parts)!=3:
                raise ValueError("Malformed posting key")
            source=parts[1]
            label=source+"_"+labels[kind]
            item=summary.setdefault(label,{"unique_keys":0,"indexed_postings":0,"max_postings":0,"keys_over_cap":0,"posting_size_histogram":Counter()})
            item["unique_keys"]+=1
            item["indexed_postings"]+=n
            item["max_postings"]=max(item["max_postings"],n)
            item["keys_over_cap"]+=int(n>caps[kind])
            item["posting_size_histogram"][n]+=1
    for item in summary.values():
        histogram=item.pop("posting_size_histogram")
        n=item["unique_keys"]
        offsets=(0.5*(n-1),0.95*(n-1),0.99*(n-1))
        result=[]
        cumulative=0
        ordered=sorted(histogram.items())
        for offset in offsets:
            target=int(offset)
            cumulative=0
            for size,count in ordered:
                cumulative+=count
                if cumulative>target:
                    result.append(size)
                    break
        item["median_postings"],item["p95_postings"],item["p99_postings"]=result
    return summary

class Index:
    def __init__(self, path: Path, limits: Limits = Limits()):
        self.db = sqlite3.connect("file:" + str(path) + "?mode=ro", uri=True)
        self.limits = limits
        self.skipped = Counter()
        self.has_frequency = bool(self.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='frequency'").fetchone())
    def close(self):
        self.db.close()
    def frequency(self, kind: int, country: str, source: str, value: str) -> int:
        if not value:
            return 0
        encoded=key(country,source,value)
        if self.has_frequency:
            row=self.db.execute("SELECT n FROM frequency WHERE kind=? AND key=?",(kind,encoded)).fetchone()
            return row[0] if row else 0
        return self.db.execute("SELECT COUNT(*) FROM posting WHERE kind=? AND key=?",(kind,encoded)).fetchone()[0]
    def posting(self, kind: int, country: str, source: str, value: str, cap: int, label: str) -> set[int]:
        count=self.frequency(kind,country,source,value)
        if count>cap:
            self.skipped[label]+=1
            return set()
        if not count:
            return set()
        return {r[0] for r in self.db.execute("SELECT target FROM posting WHERE kind=? AND key=?",(kind,key(country,source,value)))}
    def intersect(self, kind_a: int, country: str, source: str, value_a: str, count_a: int, kind_b: int, value_b: str, count_b: int) -> set[int]:
        """Intersect posting lists in SQLite, fetching only resulting target rows."""
        if not count_a or not count_b:
            return set()
        if count_a<=count_b:
            first=(kind_a,key(country,source,value_a));second=(kind_b,key(country,source,value_b))
        else:
            first=(kind_b,key(country,source,value_b));second=(kind_a,key(country,source,value_a))
        sql=("SELECT a.target FROM posting a INDEXED BY posting_lookup "
             "JOIN posting b INDEXED BY posting_lookup ON b.kind=? AND b.key=? AND b.target=a.target "
             "WHERE a.kind=? AND a.key=?")
        return {row[0] for row in self.db.execute(sql,(second[0],second[1],first[0],first[1]))}
    def target_rows(self, rows) -> dict[int,tuple[str,str,str,str]]:
        result = {}
        rows = list(rows)
        for start in range(0,len(rows),900):
            batch = rows[start:start+900]
            sql = "SELECT target,entity_id,source,name,address FROM target WHERE target IN (" + ",".join("?" for _ in batch) + ")"
            result.update((r[0],(r[1],r[2],r[3],r[4])) for r in self.db.execute(sql,batch))
        return result
    def target_by_id(self, entity_id: str):
        return self.db.execute("SELECT entity_id,source,name,address,country FROM target WHERE entity_id=?",(entity_id,)).fetchone()

def retrieve(index: Index, name: str, address: str, country: str) -> tuple[dict[int,int],dict[int,int]]:
    """Union all routes and preserve a bit for each route per target row."""
    country = normalize_country(country)
    basic, core = normalize_name(name), core_name(name)
    tokens = tuple(dict.fromkeys(t for t in core_name_tokens(name) if len(t)>=3))
    numbers = tuple(dict.fromkeys(extract_numeric_tokens(address)))
    postals = tuple(dict.fromkeys(postal_candidates(address,country)))
    strong_tokens = {token for token in address_tokens(address, standardized=True) if len(token)>=4 and not token.isdigit()}
    masks: dict[int,int] = {}
    shared: dict[int,int] = Counter()
    def add(rows, stage):
        bit = 1 << stage
        for row in rows:
            masks[row] = masks.get(row,0) | bit
    for source in ("S2","S3"):
        add(index.posting(BASIC,country,source,basic,index.limits.exact,"exact_name"),0)
        if len(core)>=3:
            add(index.posting(CORE,country,source,core,index.limits.core,"core_name"),1)
        token_lists=[]
        for token in tokens:
            count=index.frequency(TOKEN,country,source,token)
            if count>index.limits.address_token:
                index.skipped["common_token"]+=1
            elif count:
                token_lists.append((count,token))
        selected=sorted(token_lists,key=lambda item:(item[0],item[1]))[:index.limits.max_query_tokens]
        for count,token in selected:
            if count<=index.limits.informative_token:
                rows=index.posting(TOKEN,country,source,token,index.limits.informative_token,"common_token")
                if count<=index.limits.rare_token:
                    add(rows,2)
                for row in rows:
                    shared[row]+=1
        add((row for row,count in shared.items() if count>=2),2)
        for number in numbers:
            n=index.frequency(NUMBER,country,source,number)
            if n>index.limits.address_number:
                index.skipped["common_number"]+=1
                continue
            for count,token in selected:
                add(index.intersect(TOKEN,country,source,token,count,NUMBER,number,n),3)
        for postal in postals:
            n=index.frequency(POSTAL,country,source,postal)
            if n>index.limits.postal:
                index.skipped["common_postal"]+=1
                continue
            for count,token in selected:
                add(index.intersect(TOKEN,country,source,token,count,POSTAL,postal,n),4)
        # A number plus two address words (one length >= 5) is strong enough
        # to retrieve some cross-script aliases without blocking on a number alone.
        if len(strong_tokens)>=2:
            address_cache={}
            for number in numbers:
                if len(number)<3:
                    continue
                rows=index.posting(NUMBER,country,source,number,index.limits.strong_number,"common_strong_number")
                for row,(_,_,_,target_address) in index.target_rows(rows).items():
                    if row not in address_cache:
                        address_cache[row]={token for token in address_tokens(target_address, standardized=True) if len(token)>=4 and not token.isdigit()}
                    overlap=strong_tokens & address_cache[row]
                    if len(overlap)>=2 and any(len(token)>=5 for token in overlap):
                        add((row,),5)
    return masks,dict(shared)


def percentile(ordered: list[int], p: int) -> float:
    if not ordered:
        return 0.0
    position = (len(ordered)-1)*p/100
    lo = int(position)
    hi = min(lo+1,len(ordered)-1)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(position-lo)

def distribution(values: list[int]) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {"mean":0.0,"median":0.0,"p90":0.0,"p95":0.0,"p99":0.0,"max":0,"zero_pct":0.0,"over_1000_pct":0.0}
    return {"mean":sum(ordered)/len(ordered),"median":percentile(ordered,50),"p90":percentile(ordered,90),"p95":percentile(ordered,95),"p99":percentile(ordered,99),"max":ordered[-1],"zero_pct":100*ordered.count(0)/len(ordered),"over_1000_pct":100*sum(x>1000 for x in ordered)/len(ordered)}

def selected_source1(path: Path, wanted: set[str]):
    seen = set()
    with path.open(encoding="utf-8",newline="") as handle:
        reader = csv.reader(handle,delimiter="\t")
        if next(reader,None) != TARGET_HEADER:
            raise ValueError("Unexpected Source 1 columns: " + str(path))
        for row in reader:
            if len(row)!=4:
                raise ValueError("Malformed Source 1 row")
            if row[0] in wanted:
                if row[0] in seen:
                    raise ValueError("Duplicate S1 ID: " + row[0])
                seen.add(row[0])
                yield tuple(row)
    if seen != wanted:
        raise ValueError("Missing Source 1 rows: " + str(len(wanted-seen)))


def generate_subset(dataset_root: Path, ids_path: Path, db_path: Path,
                    candidates_path: Path, metadata_path: Path,
                    limits: Limits = Limits()) -> dict:
    """Reuse the frozen V1 index/routes for any S1 subset without validation side effects."""
    started = time.perf_counter()
    ids = load_id_file(ids_path)
    if not ids:
        raise ValueError("Cannot generate candidates for an empty S1 subset")
    s1_rows = sorted(selected_source1(dataset_root / "train/train_source1.tsv", set(ids)))
    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    index = Index(db_path, limits)
    total_pairs = 0
    try:
        with gzip.open(candidates_path, "wt", encoding="utf-8", newline="") as candidate_file, \
             gzip.open(metadata_path, "wt", encoding="utf-8", newline="") as metadata_file:
            candidate_writer = csv.writer(candidate_file, delimiter="\t", lineterminator="\n")
            metadata_writer = csv.writer(metadata_file, delimiter="\t", lineterminator="\n")
            candidate_writer.writerow(CANDIDATE_HEADER)
            metadata_writer.writerow(METADATA_HEADER)
            for position, (s1_id, name, address, country) in enumerate(s1_rows, 1):
                masks, shared = retrieve(index, name, address, country)
                targets = index.target_rows(masks)
                pair_rows = sorted(((targets[row][0], row, mask) for row, mask in masks.items()),
                                   key=lambda item: item[0])
                candidate_writer.writerow((s1_id, ",".join(item[0] for item in pair_rows)))
                for target_id, row, mask in pair_rows:
                    metadata_writer.writerow((s1_id, target_id, targets[row][1],
                                              *(int(bool(mask & (1 << stage))) for stage in range(len(ROUTES))),
                                              shared.get(row, 0), mask.bit_count()))
                total_pairs += len(pair_rows)
                if position % 10000 == 0:
                    print(f"Processed {position:,}/{len(s1_rows):,} S1; candidates={total_pairs:,}", flush=True)
    finally:
        index.close()
    return {"s1_count": len(s1_rows), "total_candidate_pairs": total_pairs,
            "candidate_seconds": time.perf_counter() - started,
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "skipped_queries": dict(index.skipped), "candidate_file": str(candidates_path),
            "metadata_file": str(metadata_path)}


def run_validation(dataset_root: Path, ids_path: Path, db_path: Path, output_dir: Path, limits: Limits = Limits(), max_miss_details: int = 1000, diagnostics_dir: Path | None = None) -> dict:
    """Stream validation results, ablations, metadata, and inspectable misses."""
    start=time.perf_counter()
    ids=load_id_file(ids_path)
    wanted=set(ids)
    truth=load_ground_truth(dataset_root/"train/train_ground_truth.tsv",wanted)
    s1_rows=sorted(selected_source1(dataset_root/"train/train_source1.tsv",wanted))
    index=Index(db_path,limits)
    output_dir.mkdir(parents=True,exist_ok=True)
    candidates_path=output_dir/"v1_validation_candidates.tsv.gz"
    metadata_path=output_dir/"v1_validation_metadata.tsv.gz"
    diagnostics_dir=diagnostics_dir or BASE/"artifacts/diagnostics"
    miss_path=diagnostics_dir/"blocking_misses_v1.tsv"
    miss_ids_path=diagnostics_dir/"blocking_miss_ids_v1.tsv.gz"
    miss_path.parent.mkdir(parents=True,exist_ok=True)
    stage_totals=[0]*len(ROUTES)
    recovered=[Counter() for _ in ROUTES]
    distributions=[[] for _ in ROUTES]
    missed=Counter()
    true_links=Counter()
    details=0
    try:
        with gzip.open(candidates_path,"wt",encoding="utf-8",newline="") as cand_file, gzip.open(metadata_path,"wt",encoding="utf-8",newline="") as meta_file, gzip.open(miss_ids_path,"wt",encoding="utf-8",newline="") as miss_ids_file, miss_path.open("w",encoding="utf-8",newline="") as miss_file:
            cand_writer=csv.writer(cand_file,delimiter="\t",lineterminator="\n")
            meta_writer=csv.writer(meta_file,delimiter="\t",lineterminator="\n")
            miss_ids_writer=csv.writer(miss_ids_file,delimiter="\t",lineterminator="\n")
            miss_writer=csv.writer(miss_file,delimiter="\t",lineterminator="\n")
            cand_writer.writerow(CANDIDATE_HEADER)
            meta_writer.writerow(METADATA_HEADER)
            miss_ids_writer.writerow(["source1_entity_id","true_target_id","target_source"])
            miss_writer.writerow(MISS_HEADER)
            for position,(s1_id,name,address,country) in enumerate(s1_rows,1):
                masks,shared=retrieve(index,name,address,country)
                targets=index.target_rows(masks)
                pair_rows=sorted(((targets[row][0],row,mask) for row,mask in masks.items()),key=lambda x:x[0])
                cand_writer.writerow((s1_id,",".join(item[0] for item in pair_rows)))
                mask_by_id={target_id:mask for target_id,_,mask in pair_rows}
                for target_id,row,mask in pair_rows:
                    meta_writer.writerow((s1_id,target_id,targets[row][1],*(int(bool(mask & (1<<i))) for i in range(len(ROUTES))),shared.get(row,0),mask.bit_count()))
                for true_id in sorted(truth[s1_id]):
                    source=true_id[:2]
                    true_links[source]+=1
                    mask=mask_by_id.get(true_id,0)
                    if not mask:
                        missed[source]+=1
                        miss_ids_writer.writerow((s1_id,true_id,source))
                        if details<max_miss_details:
                            target=index.target_by_id(true_id)
                            if target:
                                miss_writer.writerow((s1_id,true_id,source,name,target[2],address,target[3],normalize_name(name),normalize_name(target[2]),core_name(name),core_name(target[2])," ".join(core_name_tokens(name))," ".join(core_name_tokens(target[2])),normalize_country(country),target[4]))
                                details+=1
                    for stage in range(len(ROUTES)):
                        if mask & ((1<<(stage+1))-1):
                            recovered[stage][source]+=1
                for stage in range(len(ROUTES)):
                    count=sum(bool(mask & ((1<<(stage+1))-1)) for mask in masks.values())
                    stage_totals[stage]+=count
                    distributions[stage].append(count)
                if position%10000==0:
                    print("Processed " + format(position,",") + "/" + format(len(s1_rows),",") + " S1; candidates=" + format(stage_totals[-1],","),flush=True)
        ablation=[]
        for stage,route in enumerate(ROUTES):
            found=recovered[stage]
            ablation.append({"configuration":"+".join(ROUTES[:stage+1]),"candidate_recall":sum(found.values())/sum(true_links.values()),"s2_recall":found["S2"]/true_links["S2"],"s3_recall":found["S3"]/true_links["S3"],"total_candidate_pairs":stage_totals[stage],**distribution(distributions[stage])})
        return {"s1_count":len(s1_rows),"true_links":dict(true_links),"missed":dict(missed),"ablation":ablation,"candidate_seconds":time.perf_counter()-start,"peak_rss_mb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,"skipped_queries":dict(index.skipped),"candidate_file":str(candidates_path),"metadata_file":str(metadata_path),"miss_ids_file":str(miss_ids_path),"miss_sample_file":str(miss_path)}
    finally:
        index.close()


def route_only_ablation(metadata_path: Path, truth: dict[str,set[str]]) -> list[dict]:
    """Measure each route alone from streamed pair metadata, including zero-hit S1s."""
    counts: dict[str,list[int]]={}
    recovered=[Counter() for _ in ROUTES]
    with gzip.open(metadata_path,"rt",encoding="utf-8",newline="") as handle:
        reader=csv.DictReader(handle,delimiter="\t")
        if reader.fieldnames!=METADATA_HEADER:
            raise ValueError("Unexpected blocking metadata columns")
        for row in reader:
            s1_id=row["source1_entity_id"]
            if s1_id not in truth:
                raise ValueError("Unexpected S1 in blocking metadata: "+s1_id)
            count=counts.setdefault(s1_id,[0]*len(ROUTES))
            for stage,route in enumerate(ROUTES):
                if row[route]=="1":
                    count[stage]+=1
                    if row["candidate_entity_id"] in truth[s1_id]:
                        recovered[stage][row["target_source"]]+=1
    total_true=sum(len(links) for links in truth.values())
    s2_true=sum(value.startswith("S2-") for links in truth.values() for value in links)
    s3_true=total_true-s2_true
    result=[]
    for stage,route in enumerate(ROUTES):
        dist=distribution([counts.get(s1_id,[0]*len(ROUTES))[stage] for s1_id in truth])
        found=recovered[stage]
        result.append({"configuration":route+" alone","candidate_recall":sum(found.values())/total_true if total_true else 0.0,"s2_recall":found["S2"]/s2_true if s2_true else 0.0,"s3_recall":found["S3"]/s3_true if s3_true else 0.0,"total_candidate_pairs":sum(count[stage] for count in counts.values()),**dist})
    return result

def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root",type=Path,default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--s1-ids",type=Path,default=DEFAULT_S1_IDS)
    parser.add_argument("--output-dir",type=Path,default=BASE/"artifacts/blocking")
    parser.add_argument("--rebuild-index",action="store_true")
    parser.add_argument("--max-miss-details",type=int,default=1000)
    args=parser.parse_args(argv)
    start=time.perf_counter()
    db_path=args.output_dir/"v1_index.sqlite"
    build=build_index(args.dataset_root,db_path,args.rebuild_index)
    frequencies=ensure_frequency_table(db_path)
    report=run_validation(args.dataset_root,args.s1_ids,db_path,args.output_dir,max_miss_details=args.max_miss_details)
    report["frequencies"]=frequencies
    report["index_statistics"]=index_statistics(db_path)
    ids=load_id_file(args.s1_ids)
    truth=load_ground_truth(args.dataset_root/"train/train_ground_truth.tsv",set(ids))
    report["route_only_ablation"]=route_only_ablation(Path(report["metadata_file"]),truth)
    report["index"]=build
    report["total_seconds"]=time.perf_counter()-start
    report_path=args.output_dir/"v1_report.json"
    report_path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
