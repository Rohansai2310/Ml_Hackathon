"""Small, deterministic fixtures for disk-backed multi-pass blocking."""
import csv
import contextlib
import io
import gzip
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from blocking import Index, Limits, ROUTES, build_index, ensure_frequency_table, generate_subset, index_statistics, retrieve, route_only_ablation, run_validation
from diagnostics import evaluate_candidates_only, main as diagnostics_main
from scoring import load_ground_truth

class BlockingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        train=self.root/"train"
        train.mkdir()
        self.write(train/"train_source1.tsv",["entity_id","business_name","business_address","country"],[
            ["S1-A","Acme LLC","","US"],
            ["S1-B","Rare Widgets","5 Cedar Rd","Atlantis"],
            ["S1-C","Orchid Market","12 Main St, 02108","US"],
            ["S1-D","","","US"],
        ])
        self.write(train/"train_source2.tsv",["entity_id","business_name","business_address","country"],[
            ["S2-A","Acme Inc","","US"],
            ["S2-B","Rare Supply","9 Cedar Rd","Atlantis"],
            ["S2-C","Orchid Supplies","12 Other Ave, 02108","US"],
            ["S2-D","","","US"],
            ["S2-E","Orchid Hardware","92 Other Ave, 90210","US"],
            ["S2-F","सेवन एनर्जी प्राइवेट लिमिटेड","","India"],
        ])
        self.write(train/"train_source3.tsv",["entity_id","business_name","business_address","country"],[
            ["S3-A","ACME LLC","","US"],
            ["S3-C","Orchid Repair","12 Main Rd","US"],
            ["S3-D","Acme Incorporated","","Atlantis"],
            ["S3-E","Mystery Alias","1234 Maple Creek Road","US"],
        ])
        self.write(train/"train_ground_truth.tsv",["source1_entity_id","matched_entity_ids"],[
            ["S1-A","S2-A,S3-A"],["S1-B","S2-B"],["S1-C","S2-C"],["S1-D",""]
        ])
        self.ids=self.root/"ids.txt"
        self.ids.write_text("S1-A\nS1-B\nS1-C\nS1-D\n",encoding="utf-8")
        self.db_path=self.root/"index.sqlite"
        self.build=build_index(self.root,self.db_path)
        self.index=Index(self.db_path)
    def tearDown(self):
        self.index.close()
        self.temp.cleanup()
    @staticmethod
    def write(path,header,rows):
        with path.open("w",encoding="utf-8",newline="") as handle:
            writer=csv.writer(handle,delimiter="\t",lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)
    def named_masks(self,name,address,country,limits=None):
        index=self.index if limits is None else Index(self.db_path,limits)
        try:
            masks,shared=retrieve(index,name,address,country)
            rows=index.target_rows(masks)
            return {row[0]:(masks[k],shared.get(k,0)) for k,row in rows.items()}
        finally:
            if limits is not None:
                index.close()
    def test_exact_name_core_union_and_source_identity(self):
        hits=self.named_masks("Acme LLC","","US")
        self.assertTrue(hits["S3-A"][0]&1)
        self.assertTrue(hits["S2-A"][0]&2)
        self.assertEqual(len([k for k in hits if k=="S3-A"]),1)
        self.assertEqual(hits["S3-A"][0].bit_count(),len([r for i,r in enumerate(ROUTES) if hits["S3-A"][0]&(1<<i)]))
        self.assertNotIn("S3-D",hits)
    def test_rare_token_and_open_country(self):
        hits=self.named_masks("Rare Widgets","5 Cedar Rd","Atlantis")
        self.assertTrue(hits["S2-B"][0]&(1<<2))
    def test_indic_same_script_name_retrieval(self):
        hits=self.named_masks("सेवन एनर्जी प्राइवेट लिमिटेड","","India")
        self.assertTrue(hits["S2-F"][0]&1)
    def test_empty_name_does_not_make_block(self):
        self.assertEqual(self.named_masks("","","US"),{})
    def test_common_token_skip(self):
        limits=Limits(exact=1,core=1,rare_token=1,informative_token=1,address_token=1)
        hits=self.named_masks("Orchid Place","","US",limits)
        self.assertNotIn("S2-C",hits)
    def test_address_routes_need_name_evidence(self):
        limits=Limits(rare_token=0,informative_token=0)
        hits=self.named_masks("Orchid Market","12 Main St, 02108","US",limits)
        self.assertTrue(hits["S2-C"][0]&(1<<3))
        self.assertTrue(hits["S2-C"][0]&(1<<4))
        self.assertNotIn("S2-E",hits)
        self.assertEqual(self.named_masks("Unrelated Market","12 Main St, 02108","US",limits).get("S2-C"),None)
    def test_missing_address_keeps_name_routes(self):
        self.assertIn("S3-A",self.named_masks("Acme LLC","","US"))
    def test_index_reuse(self):
        self.assertTrue(build_index(self.root,self.db_path)["reused"])
        self.assertEqual(self.build["target_rows"],10)
    def test_generated_output_diagnostics_and_ablation(self):
        output=self.root/"output"
        report=run_validation(self.root,self.ids,self.db_path,output,diagnostics_dir=output)
        self.assertEqual(report["s1_count"],4)
        self.assertEqual(report["true_links"],{"S2":3,"S3":1})
        self.assertEqual(len(report["ablation"]),len(ROUTES))
        with gzip.open(report["candidate_file"],"rt",encoding="utf-8") as handle:
            lines=handle.read().splitlines()
        self.assertEqual(len(lines),5)
        self.assertEqual(lines[-1],"S1-D\t")
        truth=load_ground_truth(self.root/"train/train_ground_truth.tsv",["S1-A","S1-B","S1-C","S1-D"])
        metrics=evaluate_candidates_only(truth,Path(report["candidate_file"]))
        self.assertEqual(metrics["s1_entities"],4)
        self.assertEqual(metrics["true_links"],4)
        self.assertEqual(metrics["recovered_links"],4)
        self.assertEqual(metrics["blocking_misses"],0)
        self.assertEqual(metrics["total_candidate_pairs"],report["ablation"][-1]["total_candidate_pairs"])
        self.assertEqual(metrics["zero_candidate_pct"],25.0)
        self.assertEqual(metrics["median_candidates"],1.5)
        self.assertAlmostEqual(metrics["p95_candidates"],2.85)
        self.assertEqual(metrics["max_candidates"],3)
        with gzip.open(report["metadata_file"],"rt",encoding="utf-8") as handle:
            rows=list(csv.DictReader(handle,delimiter="\t"))
        self.assertTrue(any(r["candidate_entity_id"]=="S3-A" and r["exact_name"]=="1" and r["core_name"]=="1" for r in rows))
    def test_generate_subset_reuses_index_without_validation_artifacts(self):
        output = self.root / "tune"
        output.mkdir()
        subset_ids = output / "ids.txt"
        subset_ids.write_text("S1-A\nS1-D\n", encoding="utf-8")
        candidates = output / "tune_candidates.tsv.gz"
        metadata = output / "tune_metadata.tsv.gz"
        report = generate_subset(self.root, subset_ids, self.db_path, candidates, metadata)
        self.assertEqual(report["s1_count"], 2)
        with gzip.open(candidates, "rt", encoding="utf-8") as handle:
            rows = list(csv.reader(handle, delimiter="\t"))
        self.assertEqual(rows[0], ["source1_entity_id", "candidate_entity_ids"])
        self.assertEqual(rows[-1], ["S1-D", ""])
        self.assertFalse((output / "blocking_misses_v1.tsv").exists())

    def test_union_recovers_when_exact_route_fails(self):
        hits=self.named_masks("Acme LLC","","US")
        self.assertFalse(hits["S2-A"][0]&1)
        self.assertTrue(hits["S2-A"][0]&2)

    def test_exact_case_punctuation_route(self):
        hits=self.named_masks("... ACME LLC!","","US")
        self.assertTrue(hits["S3-A"][0]&1)
    def test_core_route_legal_variants(self):
        hits=self.named_masks("Acme Limited","","US")
        self.assertTrue(hits["S2-A"][0]&2)
        self.assertTrue(hits["S3-A"][0]&2)
    def test_country_partition_applies_to_all_routes(self):
        self.assertNotIn("S3-D",self.named_masks("Acme Incorporated","","US"))
        self.assertIn("S3-D",self.named_masks("Acme Incorporated","","Atlantis"))
    def test_number_alone_never_retrieves(self):
        self.assertNotIn("S2-C",self.named_masks("Unrelated","12 Main St","US"))
    def test_postal_alone_never_retrieves(self):
        self.assertNotIn("S2-C",self.named_masks("Unrelated","02108","US"))
    def test_duplicate_route_metadata_count(self):
        output=self.root/"output"
        report=run_validation(self.root,self.ids,self.db_path,output,diagnostics_dir=output)
        with gzip.open(report["metadata_file"],"rt",encoding="utf-8") as handle:
            rows=list(csv.DictReader(handle,delimiter="\t"))
        pair=next(r for r in rows if r["source1_entity_id"]=="S1-A" and r["candidate_entity_id"]=="S3-A")
        self.assertEqual(int(pair["num_blocking_routes"]),sum(int(pair[route]) for route in ROUTES))
        self.assertEqual(len([r for r in rows if r["source1_entity_id"]=="S1-A" and r["candidate_entity_id"]=="S3-A"]),1)
    def test_candidate_only_cli_logs_blank_prediction_metrics(self):
        output=self.root/"output"
        report=run_validation(self.root,self.ids,self.db_path,output,diagnostics_dir=output)
        summary=output/"summary.txt"
        experiments=output/"experiments.csv"
        args=["--ground-truth",str(self.root/"train/train_ground_truth.tsv"),"--s1-ids",str(self.ids),"--candidates",report["candidate_file"],"--candidates-only","--output",str(summary),"--experiments-file",str(experiments),"--experiment-name","blocking-fixture","--runtime-seconds","1.25"]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(diagnostics_main(args),0)
        self.assertIn("candidate_recall: 1.00000000",summary.read_text())
        with experiments.open(newline="",encoding="utf-8") as handle:
            row=next(csv.DictReader(handle))
        self.assertEqual(row["macro_f0_5"],"")
        self.assertEqual(row["candidate_recall"],"1.0")
        self.assertEqual(row["runtime_seconds"],"1.25")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(diagnostics_main(args),1)

    def test_cumulative_route_recall_and_growth(self):
        output=self.root/"output"
        report=run_validation(self.root,self.ids,self.db_path,output,diagnostics_dir=output)
        stages=report["ablation"]
        self.assertEqual(stages[0]["candidate_recall"],0.25)
        self.assertEqual(stages[1]["candidate_recall"],0.5)
        self.assertEqual(stages[-1]["candidate_recall"],1.0)
        self.assertEqual(stages[-1]["s2_recall"],1.0)
        self.assertEqual(stages[-1]["s3_recall"],1.0)
        self.assertEqual(stages[-1]["total_candidate_pairs"],6)
    def test_output_rows_are_deterministic(self):
        output=self.root/"output"
        first=run_validation(self.root,self.ids,self.db_path,output,diagnostics_dir=output)
        with gzip.open(first["candidate_file"],"rt",encoding="utf-8") as handle:
            first_text=handle.read()
        run_validation(self.root,self.ids,self.db_path,output,diagnostics_dir=output)
        with gzip.open(first["candidate_file"],"rt",encoding="utf-8") as handle:
            self.assertEqual(handle.read(),first_text)

    def test_frequency_table_and_index_statistics(self):
        table=ensure_frequency_table(self.db_path)
        self.assertGreater(table["keys"],0)
        stats=index_statistics(self.db_path,Limits(address_token=1))
        self.assertGreater(stats["S2_name_token"]["indexed_postings"],0)
        self.assertGreater(stats["S2_name_token"]["keys_over_cap"],0)
        indexed=Index(self.db_path,Limits(address_token=1))
        try:
            self.assertTrue(indexed.has_frequency)
            self.assertEqual(indexed.posting(2,"us","S2","orchid",1,"common_token"),set())
            self.assertEqual(indexed.skipped["common_token"],1)
        finally:
            indexed.close()

    def test_individual_route_ablation(self):
        output=self.root/"output"
        report=run_validation(self.root,self.ids,self.db_path,output,diagnostics_dir=output)
        truth=load_ground_truth(self.root/"train/train_ground_truth.tsv",["S1-A","S1-B","S1-C","S1-D"])
        routes=route_only_ablation(Path(report["metadata_file"]),truth)
        self.assertEqual(len(routes),len(ROUTES))
        self.assertEqual(routes[0]["candidate_recall"],0.25)
        self.assertEqual(routes[1]["candidate_recall"],0.5)
        self.assertEqual(routes[0]["total_candidate_pairs"],1)
        self.assertEqual(routes[0]["zero_pct"],75.0)

    def test_strong_address_route_uses_number_and_two_words(self):
        hits=self.named_masks("Wholly Different","1234 Maple Creek Rd","US")
        self.assertTrue(hits["S3-E"][0]&(1<<5))
        self.assertFalse(hits["S3-E"][0]&((1<<5)-1))
        self.assertNotIn("S3-E",self.named_masks("Wholly Different","9234 Maple Creek Rd","US"))
        self.assertNotIn("S3-E",self.named_masks("Wholly Different","1234 Maple Ave","US"))

if __name__=="__main__":
    unittest.main()
