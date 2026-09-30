#!/usr/bin/env python3
"""Bounded Phase 6D tune-only address and sparse top-N retrieval prototypes."""
from __future__ import annotations
import argparse, csv, gzip, json, math, resource, sqlite3, time, importlib.metadata, heapq, tempfile, contextlib, itertools
from collections import defaultdict
from pathlib import Path
import numpy as np
from sparse_dot_topn import sp_matmul_topn
from diagnostics import DEFAULT_DATASET_ROOT
from normalize import (address_tokens, extract_numeric_tokens, normalize_country, normalize_name,
                       postal_candidates)
from scoring import load_ground_truth, load_id_file
from retrieval_diagnosis import (_candidate_stats, _read_candidate_subset, _read_s1,
                                 OUT, V1_CANDIDATES, INDEX)
from tfidf_retrieval import _load, _paths, _query_matrix

BASE = Path(__file__).resolve().parents[1]
ART = BASE / 'artifacts'
TUNE_IDS = ART / 'splits/tune_s1_ids.txt'
GT = DEFAULT_DATASET_ROOT / 'train/train_ground_truth.tsv'
TFIDF01 = ART / 'tfidf_v2_tune/tune_tfidf_top50.tsv.gz'
TFIDF05 = OUT / 'tfidf_df005_5k/prototype_b_tfidf_top50.tsv.gz'
ADDRESS_CAP = 20_000
ADDRESS_QUERY_TERMS = 4
ADDRESS_POSTING_CAP = 20_000
ADDRESS_RERANK_CAP = 100
ADDRESS_K = 20
CHAR_K = 50
SAMPLE_N = 5_000
SEED = 20260925
PHASE_OUT = OUT / 'phase6d'
PAIR_HEADER = ['source1_entity_id','candidate_entity_id','target_source','route','score','rank']


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def sparse_topn_row(query, target_transpose, ids, k: int, oversample: int = 32):
    """Exact top-k by score, then ID. Recompute only cutoff-tie rows exactly."""
    if k < 1 or oversample < 0:
        raise ValueError('k must be positive and oversample nonnegative')
    if query.shape[0] != 1 or target_transpose.shape[1] != len(ids):
        raise ValueError('sparse row/index shape mismatch')
    if query.nnz == 0 or len(ids) == 0:
        return []
    n_keep = min(len(ids), k + oversample)
    top = sp_matmul_topn(query.tocsr(), target_transpose.tocsr(), top_n=n_keep,
                         sort=False, n_threads=1)
    row = top.getrow(0)
    indices, values = row.indices, row.data
    if not len(values):
        return []
    # A tie at rank K could have been truncated arbitrarily by the native top-N
    # kernel. Recompute that one sparse row with SciPy, never a dense product.
    if len(values) >= k:
        boundary = np.partition(values, len(values)-k)[len(values)-k]
        if int(np.count_nonzero(values == boundary)) > 1:
            exact = (query @ target_transpose).tocsr().getrow(0)
            indices, values = exact.indices, exact.data
    ranked = sorted(((str(ids[int(i)]), float(v)) for i, v in zip(indices, values)),
                    key=lambda x: (-x[1], x[0]))
    return [(entity, score, rank) for rank,(entity,score) in enumerate(ranked[:k],1)]


def sort_pair_file(path:Path, chunk_rows:int=50_000) -> dict:
    """Sort a compressed pair stream by S1/target with bounded memory."""
    started=time.perf_counter(); count=0; chunks=[]
    with tempfile.TemporaryDirectory(prefix='phase6d_sort_',dir=path.parent) as tmp:
        with gzip.open(path,'rt',encoding='utf-8',newline='') as src:
            reader=csv.reader(src,delimiter='\t'); header=next(reader)
            if header!=['source1_entity_id','candidate_entity_id','target_source','tfidf_score','tfidf_rank']:
                raise ValueError('Unexpected sparse candidate header')
            while True:
                batch=list(itertools.islice(reader,chunk_rows))
                if not batch: break
                batch.sort(key=lambda r:(r[0],r[1])); count+=len(batch)
                out=Path(tmp)/f'chunk_{len(chunks):04d}.tsv'
                with out.open('w',encoding='utf-8',newline='') as h:
                    csv.writer(h,delimiter='\t',lineterminator='\n').writerows(batch)
                chunks.append(out)
        merged=Path(tmp)/'sorted.tsv.gz'
        with contextlib.ExitStack() as stack, gzip.open(merged,'wt',encoding='utf-8',newline='') as h:
            writer=csv.writer(h,delimiter='\t',lineterminator='\n'); writer.writerow(header)
            readers=[csv.reader(stack.enter_context(p.open(encoding='utf-8',newline='')),delimiter='\t') for p in chunks]
            writer.writerows(heapq.merge(*readers,key=lambda r:(r[0],r[1])))
        merged.replace(path)
    return {'pairs':count,'sort_seconds':time.perf_counter()-started,'chunk_rows':chunk_rows}


def retrieve_char_fast(index_dir:Path, sample_ids, s1_rows, output:Path, max_k=CHAR_K, batch_size=64):
    """Run bounded, country-partitioned char TF-IDF with native sparse top-N."""
    if batch_size < 1: raise ValueError('batch_size must be positive')
    from scipy.sparse import vstack
    started=time.perf_counter(); load_seconds=vector_seconds=retrieve_seconds=0.0; pairs=0
    output.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(output,'wt',encoding='utf-8',newline='') as h:
        w=csv.writer(h,delimiter='\t',lineterminator='\n'); w.writerow(['source1_entity_id','candidate_entity_id','target_source','tfidf_score','tfidf_rank'])
        for source in ('S2','S3'):
            tick=time.perf_counter(); shards=[_load(p) for p in _paths(index_dir,source)]
            if not shards: raise ValueError(f'missing {source} shards in {index_dir}')
            matrix=vstack([q[0] for q in shards],format='csr'); target_ids=np.concatenate([q[1] for q in shards]); countries=np.concatenate([q[2] for q in shards]); del shards
            partitions={}
            for country in sorted(set(countries.tolist())):
                mask=countries==country; part=matrix[mask].tocsr(); ids=target_ids[mask].copy()
                partitions[country]=(part.T.tocsr(),ids)
            del matrix,target_ids,countries
            idf=np.load(index_dir/f'idf_{source.lower()}.npy',mmap_mode='r'); load_seconds+=time.perf_counter()-tick
            for start in range(0,len(sample_ids),batch_size):
                batch_ids=sample_ids[start:start+batch_size]
                names=[normalize_name(s1_rows[s]['business_name']) for s in batch_ids]
                countries1=[normalize_country(s1_rows[s]['country']) for s in batch_ids]
                tick=time.perf_counter(); queries=_query_matrix(names,idf); vector_seconds+=time.perf_counter()-tick
                for country in sorted(set(countries1)):
                    part=partitions.get(country)
                    if part is None: continue
                    local=[i for i,c in enumerate(countries1) if c==country and queries.getrow(i).nnz]
                    if not local: continue
                    tick=time.perf_counter()
                    block=queries[local].tocsr()
                    top=sp_matmul_topn(block,part[0],top_n=min(part[0].shape[1],max_k+32),sort=False,n_threads=1)
                    retrieve_seconds+=time.perf_counter()-tick
                    for j,i in enumerate(local):
                        result=top.getrow(j); indices,values=result.indices,result.data
                        if not len(values): continue
                        if len(values)>=max_k:
                            boundary=np.partition(values,len(values)-max_k)[len(values)-max_k]
                            if int(np.count_nonzero(values==boundary))>1:
                                exact=(block.getrow(j) @ part[0]).tocsr().getrow(0)
                                indices,values=exact.indices,exact.data
                        ranked=sorted(((str(part[1][int(ix)]),float(v)) for ix,v in zip(indices,values)),key=lambda x:(-x[1],x[0]))[:max_k]
                        for rank,(entity,score) in enumerate(ranked,1):
                            w.writerow((batch_ids[i],entity,source,f'{score:.8f}',rank)); pairs+=1
            del partitions
    sort_report=sort_pair_file(output)
    return {'source_specific_pairs':pairs,'sort_seconds':sort_report['sort_seconds'],'load_seconds':load_seconds,'vectorization_seconds':vector_seconds,'retrieval_seconds':retrieve_seconds,'total_seconds':time.perf_counter()-started,'peak_rss_mb':rss_mb(),'bytes':output.stat().st_size,'batch_size':batch_size,'top_k':max_k,'threads':1,'output':str(output)}


def verify_sparse_topn() -> dict:
    """Compare native top-N with exact sparse multiplication on tie/varied rows."""
    from scipy.sparse import csr_matrix
    a = csr_matrix(np.asarray([[1,0,1,0],[0,1,0,1],[0,0,0,0]], dtype=np.float32))
    b = csr_matrix(np.asarray([[1,0,0],[0,1,0],[1,0,0],[0,1,0]], dtype=np.float32))
    ids = ['T-C','T-A','T-B']
    expected = []
    exact = (a @ b).tocsr()
    for r in range(a.shape[0]):
        rr=exact.getrow(r)
        expected.append([(ids[i],float(v)) for i,v in sorted(zip(rr.indices,rr.data),key=lambda z:(-z[1],ids[z[0]]))[:2]])
    got = [sparse_topn_row(a.getrow(r),b.tocsc(),ids,2,oversample=1) for r in range(a.shape[0])]
    normalized = [[(i,v) for i,v,_ in row] for row in got]
    if normalized != expected:
        raise AssertionError(f'sparse-dot-topn mismatch: {normalized!r} != {expected!r}')
    return {'version':'1.2.0','fixture_rows':3,'top_k':2,'tie_policy':'score descending then target ID ascending','exact_match':True,'n_threads':1}


def _ids_sample(path: Path, sample_size=SAMPLE_N, seed=SEED) -> list[str]:
    values=load_id_file(path)
    if len(values)<sample_size: raise ValueError('tune ID file smaller than requested sample')
    rng=np.random.default_rng(seed)
    return sorted(rng.choice(np.asarray(values),size=sample_size,replace=False).tolist())


def _iter_targets(index_path: Path, batch_size=20_000):
    db=sqlite3.connect(f'file:{index_path.resolve()}?mode=ro',uri=True)
    try:
        for source in ('S2','S3'):
            last=0
            while True:
                rows=db.execute('SELECT target,entity_id,source,country,address FROM target WHERE source=? AND target>? ORDER BY target LIMIT ?', (source,last,batch_size)).fetchall()
                if not rows: break
                last=rows[-1][0]
                yield from rows
    finally: db.close()


def build_address_index(index_path: Path, output: Path, max_bytes=8_000_000_000) -> dict:
    """Build source/country address word postings; skip terms above DF cap."""
    if output.exists():
        db=sqlite3.connect(f'file:{output.resolve()}?mode=ro',uri=True)
        try:
            meta=dict(db.execute('select key,value from meta'))
            if int(meta.get('format','0')) != 1: raise ValueError('unknown address index format')
            return {'reused':True,'index_bytes':output.stat().st_size,**{k:v for k,v in meta.items() if k not in ('format',)}}
        finally: db.close()
    output.parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(output)
    db.execute('PRAGMA journal_mode=OFF'); db.execute('PRAGMA synchronous=OFF'); db.execute('PRAGMA temp_store=FILE'); db.execute('PRAGMA cache_size=-131072')
    db.executescript('CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT); CREATE TABLE df(source TEXT,country TEXT,token TEXT,n INTEGER,PRIMARY KEY(source,country,token)); CREATE TABLE posting(source TEXT,country TEXT,token TEXT,target INTEGER);')
    started=time.perf_counter(); n_docs=0; batch=[]; last_progress=0
    try:
        for target,entity,source,country,address in _iter_targets(index_path):
            c=normalize_country(country)
            toks=set(t for t in address_tokens(address,standardized=True) if len(t)>=4 and not t.isdigit())
            batch.extend((source,c,t) for t in toks)
            n_docs+=1
            if len(batch)>=100_000:
                db.executemany('INSERT INTO df VALUES (?,?,?,1) ON CONFLICT(source,country,token) DO UPDATE SET n=n+1',batch); batch.clear()
            if n_docs-last_progress>=250_000:
                db.commit(); last_progress=n_docs
                print(f'address df pass: {n_docs:,} targets; {output.stat().st_size:,} bytes',flush=True)
        if batch: db.executemany('INSERT INTO df VALUES (?,?,?,1) ON CONFLICT(source,country,token) DO UPDATE SET n=n+1',batch)
        db.commit()
        counts={s:db.execute('SELECT count(*) FROM df WHERE source=? AND n<=?',(s,ADDRESS_CAP)).fetchone()[0] for s in ('S2','S3')}
        total_terms=sum(counts.values())
        # Second pass inserts only informative, frequency-bounded terms.
        batch=[]; done=0
        for target,entity,source,country,address in _iter_targets(index_path):
            c=normalize_country(country)
            toks=set(t for t in address_tokens(address,standardized=True) if len(t)>=4 and not t.isdigit())
            batch.extend((source,c,t,target) for t in toks)
            done+=1
            if len(batch)>=100_000:
                db.executemany('INSERT INTO posting VALUES (?,?,?,?)',batch); batch.clear()
            if done%250_000==0:
                if batch: db.executemany('INSERT INTO posting VALUES (?,?,?,?)',batch); batch.clear()
                db.commit(); db.execute('DELETE FROM posting WHERE rowid IN (SELECT p.rowid FROM posting p JOIN df d USING(source,country,token) WHERE d.n>?)',(ADDRESS_CAP,)); db.commit()
                if output.stat().st_size>max_bytes: raise RuntimeError('address index exceeded configured 8GB disk budget; stopping')
                print(f'address postings pass: {done:,}/{n_docs:,}; {output.stat().st_size:,} bytes',flush=True)
        if batch: db.executemany('INSERT INTO posting VALUES (?,?,?,?)',batch)
        db.execute('DELETE FROM posting WHERE rowid IN (SELECT p.rowid FROM posting p JOIN df d USING(source,country,token) WHERE d.n>?)',(ADDRESS_CAP,))
        db.execute('CREATE INDEX posting_lookup ON posting(source,country,token,target)'); db.commit()
        if output.stat().st_size>max_bytes: raise RuntimeError('address index exceeded configured 8GB disk budget; stopping')
        meta={'format':'1','source_targets':str(n_docs),'max_df':str(ADDRESS_CAP),'query_terms':str(ADDRESS_QUERY_TERMS),'posting_rows':str(db.execute('select count(*) from posting').fetchone()[0]),'build_seconds':f'{time.perf_counter()-started:.3f}','skipped_df_terms':str(db.execute('select count(*) from df where n>?',(ADDRESS_CAP,)).fetchone()[0])}
        db.executemany('INSERT INTO meta VALUES (?,?)',meta.items()); db.commit()
        return {'reused':False,'index_bytes':output.stat().st_size,**meta,'retained_terms':total_terms}
    except Exception:
        db.close(); output.unlink(missing_ok=True); raise
    finally:
        try: db.close()
        except Exception: pass


def select_address_terms(token_frequencies, max_df=ADDRESS_CAP, max_terms=ADDRESS_QUERY_TERMS):
    """Select rare, informative address words under a target-frequency cap."""
    return sorted(((int(df),str(token)) for token,df in token_frequencies.items()
                   if int(df)>0 and int(df)<=max_df and len(str(token))>=4 and not str(token).isdigit()),
                  key=lambda item:(item[0],item[1]))[:max_terms]


def address_candidate_score(shared_tokens, token_dfs, numbers_agree=False, postal_agree=False, target_count=10_320_219):
    """IDF-weighted address overlap with numbers/postal as supporting signals."""
    score=sum(math.log1p(target_count/(int(token_dfs[t])+1)) for t in set(shared_tokens))
    return score + .8*bool(numbers_agree) + 1.0*bool(postal_agree)


def rank_address_candidates(candidate_rows, top_k=ADDRESS_K):
    """Rows are (target ID, entity ID, score); ties resolve by target entity ID."""
    return sorted(candidate_rows,key=lambda row:(-float(row[2]),str(row[1])))[:top_k]


def address_retrieve(sample_ids, s1_rows, v1db_path:Path, address_index:Path, out_path:Path, top_k=ADDRESS_K):
    """Retrieve per-source/country top address-supported target pairs."""
    adb=sqlite3.connect(f'file:{address_index.resolve()}?mode=ro',uri=True); adb.execute('PRAGMA cache_size=-131072')
    tdb=sqlite3.connect(f'file:{v1db_path.resolve()}?mode=ro',uri=True); tdb.execute('PRAGMA cache_size=-131072')
    out_path.parent.mkdir(parents=True,exist_ok=True); started=time.perf_counter(); rows_written=0; queries=0; max_query_postings=0
    with gzip.open(out_path,'wt',encoding='utf-8',newline='') as h:
        w=csv.writer(h,delimiter='\t',lineterminator='\n'); w.writerow(PAIR_HEADER)
        for s1 in sample_ids:
            row=s1_rows[s1]; country=normalize_country(row['country'])
            words=set(t for t in address_tokens(row['business_address'],standardized=True) if len(t)>=4 and not t.isdigit())
            nums=set(extract_numeric_tokens(row['business_address'])); posts=set(postal_candidates(row['business_address'],country))
            if not words: continue
            for source in ('S2','S3'):
                frequencies={}
                for token in words:
                    f=adb.execute('SELECT n FROM df WHERE source=? AND country=? AND token=?',(source,country,token)).fetchone()
                    if f: frequencies[token]=int(f[0])
                stats=select_address_terms(frequencies)
                if not stats: continue
                queries+=1
                for df,token in stats:
                    max_query_postings=max(max_query_postings,df)
                placeholders=','.join('?' for _ in stats)
                params=[source,country,*[t for _,t in stats]]
                candidate_rows=adb.execute(f'''SELECT p.target,p.token,d.n
                    FROM posting p JOIN df d USING(source,country,token)
                    WHERE p.source=? AND p.country=? AND p.token IN ({placeholders})''',params).fetchall()
                scores=defaultdict(float); shared=defaultdict(set)
                for tid,token,df in candidate_rows:
                    scores[int(tid)]+=math.log1p(10_320_219/(int(df)+1)); shared[int(tid)].add(token)
                candidate_rows=sorted(((tid,score,len(shared[tid])) for tid,score in scores.items()),key=lambda x:(-x[1],x[0]))[:ADDRESS_RERANK_CAP]
                if not candidate_rows: continue
                target_ids=[int(r[0]) for r in candidate_rows]
                details={}
                for off in range(0,len(target_ids),800):
                    chunk=target_ids[off:off+800]
                    marks=','.join('?' for _ in chunk)
                    details.update((int(t), (entity,addr,ctry,src)) for t,entity,addr,ctry,src in tdb.execute(f'SELECT target,entity_id,address,country,source FROM target WHERE target IN ({marks})',chunk))
                ranked=[]
                for tid,score,overlap in candidate_rows:
                    det=details.get(int(tid))
                    if not det: continue
                    entity,addr,tcountry,tsource=det
                    if tsource!=source or normalize_country(tcountry)!=country: continue
                    tnums=set(extract_numeric_tokens(addr)); tposts=set(postal_candidates(addr,tcountry))
                    # Word postings are mandatory; number/postal evidence only reranks.
                    adjusted=address_candidate_score(shared[int(tid)],{token:df for df,token in stats if token in shared[int(tid)]},bool(nums & tnums),bool(posts & tposts))
                    ranked.append((int(tid),entity,adjusted))
                for rank,(tid,entity,score) in enumerate(rank_address_candidates(ranked,top_k),1):
                    w.writerow((s1,entity,source,'address_token',f'{score:.8f}',rank)); rows_written+=1
    adb.close(); tdb.close()
    return {'pairs':rows_written,'queries':queries,'runtime_seconds':time.perf_counter()-started,'peak_rss_mb':rss_mb(),'max_single_token_postings':max_query_postings,'path':str(out_path),'bytes':out_path.stat().st_size}


def _read_pair_file(path:Path, max_rank:int|None=None):
    values=defaultdict(set); scores={}
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding='utf-8',newline='') as h:
        r=csv.DictReader(h,delimiter='\t')
        for row in r:
            s1=row.get('source1_entity_id'); target=row.get('candidate_entity_id')
            if not s1 or not target: continue
            if max_rank is not None and int(row.get('rank') or row.get('tfidf_rank') or 0)>max_rank: continue
            values[s1].add(target); scores[(s1,target)]=row
    return values,scores


def _metrics_for_routes(ids,v1,truth,routes):
    cand={s:set(v1[s]) for s in ids}
    for route in routes:
        pairs=route.get('pairs',{})
        for s in ids: cand[s].update(pairs.get(s,set()))
    return _candidate_stats(truth,cand),cand


def run_sample(dataset_root:Path, tune_ids:Path, gt_path:Path, v1_path:Path,
               target_index:Path, char01:Path, char05:Path, out_dir:Path,
               sample_size=5000,seed=SEED, build_index=True, max_sample_runtime=3500):
    started=time.perf_counter(); ids=_ids_sample(tune_ids,sample_size,seed); wanted=set(ids)
    records=_read_s1(dataset_root,ids); truth=load_ground_truth(gt_path,wanted); v1=_read_candidate_subset(v1_path,ids)
    base=_candidate_stats(truth,v1)
    token_path=OUT/'prototype_a_full_postings/prototype_a_added_candidates.tsv.gz'
    token_pairs,_=_read_pair_file(token_path,max_rank=20)
    address_db=out_dir/'address_postings.sqlite'
    if not build_index and not address_db.exists():
        raise FileNotFoundError(f'--skip-index-build requires {address_db}')
    index_report=build_address_index(target_index,address_db)
    address_file=out_dir/'address_top20.tsv.gz'
    address_report=address_retrieve(ids,records,target_index,address_db,address_file,ADDRESS_K)
    if time.perf_counter()-started>max_sample_runtime: raise RuntimeError('5K runtime guard reached after address retrieval')
    address_pairs,address_meta=_read_pair_file(address_file)
    fast01=out_dir/'char01_fast_top50.tsv.gz'; fast05=out_dir/'char05_fast_top50.tsv.gz'
    fast01_report=retrieve_char_fast(ART/'tfidf_v2_index',ids,records,fast01,CHAR_K)
    if time.perf_counter()-started>max_sample_runtime: raise RuntimeError('5K runtime guard reached after 0.1% retrieval')
    fast05_report=retrieve_char_fast(OUT/'tfidf_index_df005',ids,records,fast05,CHAR_K)
    if time.perf_counter()-started>max_sample_runtime: raise RuntimeError('5K runtime guard reached after 0.5% retrieval')
    char_files={'char_01':fast01,'char_05':fast05}
    route_data=[{'name':'token_k20','pairs':token_pairs}]
    route_data.append({'name':'address_k20','pairs':address_pairs})
    for name,path in char_files.items():
        pairs,_=_read_pair_file(path); route_data.append({'name':name,'pairs':pairs})
    configs={'V1':[],'V1+tokenK20':['token_k20'],'V1+addressK20':['address_k20'],'V1+char01K50':['char_01'],'V1+char05K50':['char_05'],'V1+tokenK20+addressK20':['token_k20','address_k20'],'V1+tokenK20+addressK20+char01K50':['token_k20','address_k20','char_01'],'V1+tokenK20+addressK20+char05K50':['token_k20','address_k20','char_05']}
    stats={}; unions={}
    for label,names in configs.items():
        use=[r for r in route_data if r['name'] in names]
        metrics,cands=_metrics_for_routes(ids,v1,truth,use); unions[label]=cands
        stats[label]={**metrics,'growth_pct':100*(metrics['candidate_pairs']-base['candidate_pairs'])/base['candidate_pairs'] if base['candidate_pairs'] else 0,'added_pairs':metrics['candidate_pairs']-base['candidate_pairs'],'runtime_seconds':None,'peak_rss_mb':None}
    recovered={}
    for route in route_data:
        pairs=route['pairs']; hit={(s,t) for s in ids for t in truth[s]&pairs.get(s,set())-v1[s]}
        recovered[route['name']]=hit
    overlap={}
    for i,a in enumerate(recovered):
        for b in list(recovered)[i+1:]: overlap[f'{a}&{b}']=len(recovered[a]&recovered[b])
    route_counts={n:len(v) for n,v in recovered.items()}
    exclusive={n:len(v-set().union(*(other for other_name,other in recovered.items() if other_name!=n))) for n,v in recovered.items()}
    sample_runtime=time.perf_counter()-started
    if sample_runtime>max_sample_runtime: raise RuntimeError(f'sample run took {sample_runtime:.1f}s; abort before any expanded run')
    report={'phase':'6D tune-sample only','sample_size':len(ids),'seed':seed,'tune_sample_ids_sha256':__import__('hashlib').sha256(('\n'.join(ids)+'\n').encode()).hexdigest(),'version':importlib.metadata.version('sparse-dot-topn'),'base_v1':base,'address_index':index_report,'address_retrieval':address_report,'optimized_char_01':fast01_report,'optimized_char_05':fast05_report,'configurations':stats,'recovered_true_links_per_route':route_counts,'exclusive_true_links_per_route':exclusive,'pairwise_route_recovery_intersections':overlap,'address_max_target_df':ADDRESS_CAP,'address_query_terms':ADDRESS_QUERY_TERMS,'address_k':ADDRESS_K,'char_k':CHAR_K,'sparse_topn_threads':1,'sample_total_runtime_seconds':sample_runtime,'peak_rss_mb':rss_mb(),'disk_bytes':{'address_index':address_db.stat().st_size,'address_pairs':address_file.stat().st_size,'char01_index':sum(p.stat().st_size for p in (ART/'tfidf_v2_index').glob('*')),'char05_index':sum(p.stat().st_size for p in (OUT/'tfidf_index_df005').glob('*'))}}
    report['sample_runtime_projection_seconds']={'tune_100k':sample_runtime*100000/len(ids),'validation_220683':sample_runtime*220683/len(ids),'test_1732545':sample_runtime*1732545/len(ids)}
    report=enrich_report(report,ids,char01,char05,out_dir)
    out_dir.mkdir(parents=True,exist_ok=True); (out_dir/'phase6d_summary.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return report


def compare_reference_pairs(ids, reference:Path, fast:Path) -> int:
    """Require exact candidate, rank, score, and order equality on selected S1s."""
    wanted=set(ids); count=0
    with gzip.open(reference,'rt',encoding='utf-8',newline='') as a, gzip.open(fast,'rt',encoding='utf-8',newline='') as b:
        old=csv.reader(a,delimiter='\t'); new=csv.reader(b,delimiter='\t')
        if next(old)!=next(new): raise ValueError('TF-IDF candidate header mismatch')
        selected=(row for row in old if row[0] in wanted)
        for before,after in itertools.zip_longest(selected,new):
            if before!=after: raise AssertionError(f'optimized retrieval differs at row {count}: {before!r} != {after!r}')
            count+=1
    return count


def enrich_report(report, ids, reference01:Path=TFIDF01, reference05:Path=TFIDF05,
                  out_dir:Path=PHASE_OUT):
    """Add reproducible equivalence, resource, and fixed-load projections."""
    report['reference_equivalence']={
       'char01_exact_rows':compare_reference_pairs(ids,reference01,out_dir/'char01_fast_top50.tsv.gz'),
       'char05_exact_rows':compare_reference_pairs(ids,reference05,out_dir/'char05_fast_top50.tsv.gz'),
       'pair_rank_score_order_disagreements':0}
    report['quality_gate_over_90_pct']=max(v['candidate_recall'] for v in report['configurations'].values())>0.9
    report['token_route_rank_cap']=20
    token_seconds=67.233  # Prior measured K50 posting run; K20 derived from its ranks.
    address=float(report['address_retrieval']['runtime_seconds'])
    char01=report['optimized_char_01']; char05=report['optimized_char_05']
    def extra_sort(tag):
        if 'sort_seconds' in report['optimized_char_'+tag]: return 0.0
        return float(report.get('postprocess_sort',{}).get('char'+tag,{}).get('sort_seconds',0.0))
    extra01,extra05=extra_sort('01'),extra_sort('05')
    projected={}
    for label,n in [('tune_100k',100000),('validation_220683',220683),('test_1732545',1732545)]:
        factor=n/len(ids)
        projected[label]={
          'address_only_seconds':address*factor,
          'token_k20_plus_address_seconds':(token_seconds+address)*factor,
          'token_k20_plus_address_plus_char05_seconds':(token_seconds+address+char05['total_seconds']-char05['load_seconds']+extra05)*factor+char05['load_seconds'],
          'char01_fast_seconds':(char01['total_seconds']-char01['load_seconds']+extra01)*factor+char01['load_seconds'],
          'char05_fast_seconds':(char05['total_seconds']-char05['load_seconds']+extra05)*factor+char05['load_seconds']}
    report['route_runtime_projections']=projected
    report['projection_note']='Linear projections from measured 5K single-thread route times; character index load counted once. Token K20 uses the prior 67.233s K50 posting run as a conservative proxy. Excludes one-time address-index build, candidate union/write overhead, and hardware contention.'
    route_times={'token_k20':token_seconds,'address_k20':address,'char_01':char01['total_seconds']+extra01,'char_05':char05['total_seconds']+extra05}
    configs={'V1':[],'V1+tokenK20':['token_k20'],'V1+addressK20':['address_k20'],
       'V1+char01K50':['char_01'],'V1+char05K50':['char_05'],
       'V1+tokenK20+addressK20':['token_k20','address_k20'],
       'V1+tokenK20+addressK20+char01K50':['token_k20','address_k20','char_01'],
       'V1+tokenK20+addressK20+char05K50':['token_k20','address_k20','char_05']}
    for name,routes in configs.items():
        report['configurations'][name]['route_runtime_sum_seconds_estimate']=sum(route_times[r] for r in routes)
        peaks=[report['address_retrieval']['peak_rss_mb'] if r=='address_k20' else char01['peak_rss_mb'] if r=='char_01' else char05['peak_rss_mb'] if r=='char_05' else None for r in routes]
        report['configurations'][name]['max_measured_route_rss_mb']=max((p for p in peaks if p is not None),default=None)
    report['configuration_resource_note']='Route times are measured separately, then summed as a cost estimate; token-route RSS was not measured. RSS values are maxima observed for included address/character routes, not a concurrent union measurement.'
    return report


def reevaluate_existing(dataset_root:Path=DEFAULT_DATASET_ROOT, tune_ids:Path=TUNE_IDS,
                        gt_path:Path=GT, v1_path:Path=V1_CANDIDATES,
                        out_dir:Path=PHASE_OUT, sample_size:int=SAMPLE_N, seed:int=SEED):
    """Recompute union statistics from completed 5K artifacts; no retrieval."""
    ids=_ids_sample(tune_ids,sample_size,seed); truth=load_ground_truth(gt_path,set(ids)); v1=_read_candidate_subset(v1_path,ids)
    base=_candidate_stats(truth,v1)
    routes={}
    for name,path,k in (
        ('token_k20',OUT/'prototype_a_full_postings/prototype_a_added_candidates.tsv.gz',20),
        ('address_k20',out_dir/'address_top20.tsv.gz',20),
        ('char_01',out_dir/'char01_fast_top50.tsv.gz',50),
        ('char_05',out_dir/'char05_fast_top50.tsv.gz',50)):
        routes[name]=_read_pair_file(path,max_rank=k)[0]
    configs={'V1':[],'V1+tokenK20':['token_k20'],'V1+addressK20':['address_k20'],
       'V1+char01K50':['char_01'],'V1+char05K50':['char_05'],
       'V1+tokenK20+addressK20':['token_k20','address_k20'],
       'V1+tokenK20+addressK20+char01K50':['token_k20','address_k20','char_01'],
       'V1+tokenK20+addressK20+char05K50':['token_k20','address_k20','char_05']}
    stats={}
    for label,names in configs.items():
        m,_=_metrics_for_routes(ids,v1,truth,[{'pairs':routes[n]} for n in names])
        stats[label]={**m,'growth_pct':100*(m['candidate_pairs']-base['candidate_pairs'])/base['candidate_pairs'],
                      'added_pairs':m['candidate_pairs']-base['candidate_pairs']}
    recovered={n:{(s,t) for s in ids for t in (truth[s]&pairs.get(s,set()))-v1[s]} for n,pairs in routes.items()}
    overlap={f'{a}&{b}':len(recovered[a]&recovered[b]) for i,a in enumerate(recovered) for b in list(recovered)[i+1:]}
    exclusive={n:len(values-set().union(*(other for other_name,other in recovered.items() if other_name!=n))) for n,values in recovered.items()}
    summary_path=out_dir/'phase6d_summary.json'
    report=json.loads(summary_path.read_text())
    report['base_v1']=base; report['configurations']=stats
    report['recovered_true_links_per_route']={n:len(v) for n,v in recovered.items()}
    report['exclusive_true_links_per_route']=exclusive
    report['pairwise_route_recovery_intersections']=overlap
    report.pop('unique_true_links_recovered_per_route',None)
    report['quality_gate_over_90_pct']=max(v['candidate_recall'] for v in stats.values())>0.9
    report=enrich_report(report,ids,TFIDF01,TFIDF05,out_dir)
    summary_path.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return report


def sweep_address_k(tune_ids:Path=TUNE_IDS, gt_path:Path=GT, v1_path:Path=V1_CANDIDATES,
                    out_dir:Path=PHASE_OUT, sample_size:int=SAMPLE_N, seed:int=SEED):
    """Derive K=5/10/15/20 from completed 5K address ranks; no retrieval."""
    ids=_ids_sample(tune_ids,sample_size,seed); truth=load_ground_truth(gt_path,set(ids)); v1=_read_candidate_subset(v1_path,ids)
    token=_read_pair_file(OUT/'prototype_a_full_postings/prototype_a_added_candidates.tsv.gz',max_rank=20)[0]
    rows=[]; baseline=_candidate_stats(truth,v1)
    for k in (5,10,15,20):
        address=_read_pair_file(out_dir/'address_top20.tsv.gz',max_rank=k)[0]
        for label,routes in (('address',[address]),('token_k20_plus_address',[token,address])):
            m,_=_metrics_for_routes(ids,v1,truth,[{'pairs':pairs} for pairs in routes])
            rows.append({'address_k':k,'configuration':label,'candidate_recall':m['candidate_recall'],
                's2_recall':m['source']['S2']['recall'],'s3_recall':m['source']['S3']['recall'],
                'v1_misses_recovered':m['recovered_links']-baseline['recovered_links'],
                'remaining_misses':m['blocking_misses'],'candidate_pairs':m['candidate_pairs'],
                'growth_pct':100*(m['candidate_pairs']-baseline['candidate_pairs'])/baseline['candidate_pairs'],
                'mean':m['mean'],'median':m['median'],'p95':m['p95'],'p99':m['p99'],'max':m['max']})
    output={'sample_size':len(ids),'seed':seed,'source':'completed address top-20 artifact only','sweep':rows}
    (out_dir/'address_k_sweep.json').write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='cmd',required=True)
    sub.add_parser('check-sparse-topn')
    sub.add_parser('evaluate-existing')
    sub.add_parser('sweep-address-k')
    b=sub.add_parser('build-address-index')
    b.add_argument('--target-index',type=Path,default=INDEX)
    b.add_argument('--output-dir',type=Path,default=PHASE_OUT)
    r=sub.add_parser('run-sample')
    r.add_argument('--dataset-root',type=Path,default=DEFAULT_DATASET_ROOT); r.add_argument('--tune-ids',type=Path,default=TUNE_IDS); r.add_argument('--ground-truth',type=Path,default=GT); r.add_argument('--v1-candidates',type=Path,default=V1_CANDIDATES); r.add_argument('--target-index',type=Path,default=INDEX); r.add_argument('--char01',type=Path,default=TFIDF01); r.add_argument('--char05',type=Path,default=TFIDF05); r.add_argument('--output-dir',type=Path,default=PHASE_OUT); r.add_argument('--sample-size',type=int,default=SAMPLE_N); r.add_argument('--seed',type=int,default=SEED); r.add_argument('--skip-index-build',action='store_true'); r.add_argument('--max-sample-runtime',type=int,default=3500)
    a=p.parse_args()
    if a.cmd=='check-sparse-topn': out=verify_sparse_topn()
    elif a.cmd=='evaluate-existing': out=reevaluate_existing()
    elif a.cmd=='sweep-address-k': out=sweep_address_k()
    elif a.cmd=='build-address-index': out=build_address_index(a.target_index,a.output_dir/'address_postings.sqlite')
    else: out=run_sample(a.dataset_root,a.tune_ids,a.ground_truth,a.v1_candidates,a.target_index,a.char01,a.char05,a.output_dir,a.sample_size,a.seed,not a.skip_index_build,a.max_sample_runtime)
    print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__': main()
