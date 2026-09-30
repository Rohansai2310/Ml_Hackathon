"""Focused tests for bounded Phase 6D retrieval primitives."""
import csv, sqlite3, sys, tempfile, unittest
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from phase6d_retrieval import (address_candidate_score, build_address_index,
    rank_address_candidates, select_address_terms, sparse_topn_row, verify_sparse_topn)

class Phase6DRetrievalTests(unittest.TestCase):
    def test_sparse_topn_matches_exact_and_stable_ties(self):
        self.assertTrue(verify_sparse_topn()['exact_match'])
        q=csr_matrix(np.array([[1,0,1]],dtype=np.float32))
        targets=csr_matrix(np.array([[1,1],[0,0],[1,1]],dtype=np.float32))
        ranked=sparse_topn_row(q,targets,['S2-Z','S2-A'],1,oversample=1)
        self.assertEqual(ranked[0][0],'S2-A')

    def test_frequency_cap_and_rare_terms(self):
        selected=select_address_terms({'road':30000,'industrial':12,'market':18000,'st':1,'12345':2},max_df=20000,max_terms=4)
        self.assertEqual(selected,[(12,'industrial'),(18000,'market')])

    def test_number_and_postal_are_supporting_scores_only(self):
        base=address_candidate_score({'industrial'},{'industrial':10})
        boosted=address_candidate_score({'industrial'},{'industrial':10},True,True)
        self.assertGreater(boosted,base)
        self.assertEqual(address_candidate_score(set(),{}),0)

    def test_deterministic_address_ranking_and_metadata(self):
        rows=[(3,'S2-Z',2.0),(1,'S2-A',2.0),(2,'S2-M',1.0)]
        self.assertEqual(rank_address_candidates(rows,2),[(1,'S2-A',2.0),(3,'S2-Z',2.0)])

    def test_address_index_separates_source_and_open_set_country(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); source=root/'source.sqlite'; out=root/'address.sqlite'
            db=sqlite3.connect(source)
            db.execute('CREATE TABLE target(target INTEGER PRIMARY KEY,entity_id TEXT,source TEXT,country TEXT,address TEXT)')
            db.executemany('INSERT INTO target VALUES(?,?,?,?,?)',[
                (1,'S2-A','S2','Mars','7 Industrial Park'),
                (2,'S3-A','S3','Mars','9 Industrial Park'),
                (3,'S2-B','S2','Venus','2 Industrial Park')])
            db.commit(); db.close()
            report=build_address_index(source,out)
            self.assertEqual(report['source_targets'],'3')
            db=sqlite3.connect(out)
            mars=db.execute("SELECT source,country,token,target FROM posting WHERE token='industrial' ORDER BY source,target").fetchall()
            db.close()
            self.assertEqual(mars,[('S2','mars','industrial',1),('S2','venus','industrial',3),('S3','mars','industrial',2)])

if __name__=='__main__': unittest.main()

class Phase6DIntegrationTests(unittest.TestCase):
    def test_batched_sparse_matches_exact_and_compressed_output(self):
        from normalize import normalize_name
        from tfidf_retrieval import build_index, retrieve_topk
        from phase6d_retrieval import retrieve_char_fast
        import gzip
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); dbpath=root/'targets.sqlite'; index=root/'index'
            db=sqlite3.connect(dbpath)
            db.execute('CREATE TABLE target(target INTEGER PRIMARY KEY,entity_id TEXT,source TEXT,country TEXT,name TEXT,address TEXT,basic TEXT,core TEXT)')
            db.executemany('INSERT INTO target VALUES(?,?,?,?,?,?,?,?)',[
                (1,'S2-A','S2','Mars','Mason Supply','','mason supply','mason supply'),
                (2,'S2-B','S2','Mars','Mason Supplies','','mason supplies','mason supplies'),
                (3,'S3-A','S3','Mars','Mason Supply','','mason supply','mason supply'),
                (4,'S3-B','S3','Venus','Mason Supply','','mason supply','mason supply')])
            db.commit(); db.close(); build_index(dbpath,index,batch_size=2,shard_rows=2,max_df_ratio=1.0)
            s1={'S1-A':{'business_name':'Mason Supply','country':'Mars'},'S1-B':{'business_name':'Mason Supplies','country':'Mars'},'S1-C':{'business_name':'','country':'Mars'}}
            ids=sorted(s1)
            fast1=root/'fast1.tsv.gz'; fast2=root/'fast2.tsv.gz'; exact=root/'exact.tsv.gz'
            retrieve_char_fast(index,ids,s1,fast1,max_k=2,batch_size=1)
            retrieve_char_fast(index,ids,s1,fast2,max_k=2,batch_size=3)
            retrieve_topk(index,[(i,s1[i]['business_name'],'',s1[i]['country']) for i in ids],exact,max_k=2,query_batch_size=2)
            def read(path):
                with gzip.open(path,'rt',encoding='utf-8',newline='') as h: return list(csv.reader(h,delimiter='\t'))
            self.assertEqual(read(fast1),read(fast2))
            self.assertEqual(read(fast1),read(exact))
            self.assertTrue(any(r[1]=='S2-A' for r in read(fast1)[1:]))
            self.assertFalse(any(r[1]=='S3-B' for r in read(fast1)[1:]))

    def test_address_retrieval_metadata_and_country_source(self):
        from phase6d_retrieval import address_retrieve
        import gzip
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); source=root/'source.sqlite'; idx=root/'addr.sqlite'; out=root/'pairs.tsv.gz'
            db=sqlite3.connect(source)
            db.execute('CREATE TABLE target(target INTEGER PRIMARY KEY,entity_id TEXT,source TEXT,country TEXT,address TEXT)')
            db.executemany('INSERT INTO target VALUES(?,?,?,?,?)',[
                (1,'S2-A','S2','Mars','7 Industrial Park'),
                (2,'S3-A','S3','Mars','7 Industrial Park'),
                (3,'S2-B','S2','Venus','7 Industrial Park'),
                (4,'S2-C','S2','Mars','7 Market Road')])
            db.commit(); db.close(); build_address_index(source,idx)
            s1={'S1-A':{'country':'Mars','business_address':'7 Industrial Park','business_name':'X'},
                'S1-B':{'country':'Mars','business_address':'7','business_name':'X'}}
            info=address_retrieve(['S1-A','S1-B'],s1,source,idx,out,top_k=2)
            with gzip.open(out,'rt',encoding='utf-8',newline='') as h: rows=list(csv.DictReader(h,delimiter='\t'))
            self.assertEqual(info['pairs'],len(rows))
            self.assertEqual({r['candidate_entity_id'] for r in rows},{'S2-A','S3-A'})
            self.assertTrue(all(r['route']=='address_token' and r['rank']=='1' for r in rows))
            self.assertEqual({r['target_source'] for r in rows},{'S2','S3'})

    def test_rank_cap_and_union_recovery_counts(self):
        from phase6d_retrieval import _read_pair_file,_metrics_for_routes
        import gzip
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'ranked.tsv.gz'
            with gzip.open(p,'wt',encoding='utf-8',newline='') as h:
                w=csv.writer(h,delimiter='\t'); w.writerow(['source1_entity_id','candidate_entity_id','target_source','retrieval_route','bm25_like_score','rank'])
                w.writerow(['S1-A','S2-X','S2','token','2','20']); w.writerow(['S1-A','S3-Y','S3','token','1','21'])
            pairs,_=_read_pair_file(p,max_rank=20)
            self.assertEqual(pairs['S1-A'],{'S2-X'})
            metrics,candidates=_metrics_for_routes(['S1-A'],{'S1-A':set()},{'S1-A':{'S2-X','S3-Y'}},[{'pairs':pairs}])
            self.assertEqual(metrics['recovered_links'],1)
            self.assertEqual(metrics['candidate_pairs'],1)
            self.assertEqual(candidates['S1-A'],{'S2-X'})
