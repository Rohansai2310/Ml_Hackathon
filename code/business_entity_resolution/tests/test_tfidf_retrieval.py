"""Small deterministic tests for the Phase 6 sparse TF-IDF retriever."""
import csv
import gzip
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from blocking import METADATA_HEADER
from diagnostics import CANDIDATE_HEADER, evaluate_candidates_only
from normalize import normalize_country, normalize_name
from tfidf_retrieval import (TFIDF_HEADER, V2_METADATA_HEADER, build_index,
                             retrieve_topk, union_candidates, validate_pair_artifacts, verify_v1_subset)


class TfidfRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = self.root / "targets.sqlite"
        db = sqlite3.connect(self.db)
        try:
            db.execute("CREATE TABLE target (target INTEGER PRIMARY KEY, entity_id TEXT, source TEXT, country TEXT, name TEXT, address TEXT, basic TEXT, core TEXT)")
            records = [
                (1, "S2-A", "S2", "Mars", "Mason Supply Company", "", normalize_name("Mason Supply Company"), "mason supply"),
                (2, "S2-B", "S2", "mars", "Mason-Supply Company", "", normalize_name("Mason-Supply Company"), "mason supply"),
                (3, "S2-C", "S2", "Mars", "Different Hardware", "", normalize_name("Different Hardware"), "different hardware"),
                (4, "S2-D", "S2", "Mars", "", "", "", ""),
                (5, "S3-A", "S3", "Mars", "Mason Supplies Ltd", "", normalize_name("Mason Supplies Ltd"), "mason supplies"),
                (6, "S3-B", "S3", "Elsewhere", "Mason Supply Company", "", normalize_name("Mason Supply Company"), "mason supply"),
            ]
            db.executemany("INSERT INTO target VALUES (?,?,?,?,?,?,?,?)", records)
            db.commit()
        finally:
            db.close()
        self.index_dir = self.root / "index"
        build_index(self.db, self.index_dir, batch_size=2, shard_rows=2, max_df_ratio=1.0)

    def tearDown(self):
        self.temp.cleanup()

    def read_tsv(self, path):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", newline="") as handle:
            return list(csv.reader(handle, delimiter="\t"))

    def test_typo_character_variation_and_source_separation(self):
        rows = [("S1-A", "Mason Suplpy Company", "", " MARS ")]
        out = self.root / "top.tsv.gz"
        retrieve_topk(self.index_dir, rows, out, max_k=2)
        values = self.read_tsv(out)
        self.assertEqual(values[0], TFIDF_HEADER)
        self.assertTrue(any(r[1] == "S2-A" and r[2] == "S2" for r in values[1:]))
        self.assertTrue(any(r[1] == "S3-A" and r[2] == "S3" for r in values[1:]))
        self.assertFalse(any(r[1] == "S3-B" for r in values[1:]))

    def test_blank_names_produce_no_pairs_and_unicode_safe(self):
        rows = [("S1-A", "", "", "Mars"), ("S1-B", "मेसन सप्लाई", "", "India")]
        out = self.root / "blank.tsv.gz"
        retrieve_topk(self.index_dir, rows, out, max_k=3)
        self.assertEqual(self.read_tsv(out), [TFIDF_HEADER])

    def test_top_k_rank_score_and_ties_are_deterministic(self):
        rows = [("S1-A", "Mason Supply Company", "", "Mars")]
        out1, out2 = self.root / "one.tsv.gz", self.root / "two.tsv.gz"
        retrieve_topk(self.index_dir, rows, out1, max_k=2, query_batch_size=1)
        retrieve_topk(self.index_dir, rows, out2, max_k=2, query_batch_size=4)
        a, b = self.read_tsv(out1), self.read_tsv(out2)
        self.assertEqual(a, b)
        self.assertEqual(len(a) - 1, 3)
        for source in ("S2", "S3"):
            source_rows = [row for row in a[1:] if row[2] == source]
            self.assertEqual([int(row[4]) for row in source_rows], [1, 2] if source == "S2" else [1])
            self.assertEqual(source_rows, sorted(source_rows, key=lambda row: row[1]))
            self.assertTrue(all(0 <= float(row[3]) <= 1.00001 for row in source_rows))



    def test_batch_size_equivalence_for_multiple_queries(self):
        rows = [("S1-A", "Mason Supply Company", "", "Mars"),
                ("S1-B", "Different Hardware", "", "Mars"),
                ("S1-C", "", "", "Mars")]
        out1, out2 = self.root / "batch1.tsv.gz", self.root / "batch3.tsv.gz"
        retrieve_topk(self.index_dir, rows, out1, max_k=3, query_batch_size=1)
        retrieve_topk(self.index_dir, rows, out2, max_k=3, query_batch_size=3)
        self.assertEqual(self.read_tsv(out1), self.read_tsv(out2))
    def test_country_normalization_is_open_set(self):
        self.assertEqual(normalize_country("  Mars "), "mars")
        rows = [("S1-A", "Mason Supply Company", "", "MARS")]
        out = self.root / "country.tsv.gz"
        retrieve_topk(self.index_dir, rows, out, max_k=10)
        found = self.read_tsv(out)[1:]
        self.assertIn("S2-A", {r[1] for r in found})
        self.assertNotIn("S3-B", {r[1] for r in found})

    def test_v1_candidates_are_preserved_and_metadata_unioned(self):
        s1_ids = ["S1-A", "S1-B"]
        v1c, v1m, tf = self.root / "v1.tsv.gz", self.root / "v1meta.tsv.gz", self.root / "tf.tsv.gz"
        with gzip.open(v1c, "wt", encoding="utf-8", newline="") as handle:
            w = csv.writer(handle, delimiter="\t", lineterminator="\n")
            w.writerow(CANDIDATE_HEADER); w.writerow(("S1-A", "S2-A")); w.writerow(("S1-B", ""))
        with gzip.open(v1m, "wt", encoding="utf-8", newline="") as handle:
            w = csv.writer(handle, delimiter="\t", lineterminator="\n")
            w.writerow(METADATA_HEADER)
            w.writerow(("S1-A", "S2-A", "S2", "1", "0", "0", "0", "0", "0", "0", "1"))
        with gzip.open(tf, "wt", encoding="utf-8", newline="") as handle:
            w = csv.writer(handle, delimiter="\t", lineterminator="\n")
            w.writerow(TFIDF_HEADER)
            w.writerow(("S1-A", "S2-A", "S2", "0.91", "2"))
            w.writerow(("S1-A", "S3-X", "S3", "0.88", "1"))
            w.writerow(("S1-B", "S2-Y", "S2", "0.77", "1"))
        c_out, m_out = self.root / "v2.tsv.gz", self.root / "v2meta.tsv.gz"
        union_candidates(s1_ids, v1c, v1m, tf, c_out, m_out, 2, 2)
        candidate_rows = self.read_tsv(c_out)
        self.assertEqual(candidate_rows, [CANDIDATE_HEADER, ["S1-A", "S2-A,S3-X"], ["S1-B", "S2-Y"]])
        metadata_rows = self.read_tsv(m_out)
        self.assertEqual(metadata_rows[0], V2_METADATA_HEADER)
        self.assertEqual(len({(r[0], r[1]) for r in metadata_rows[1:]}), 3)
        by_pair = {(r[0], r[1]): r for r in metadata_rows[1:]}
        self.assertEqual(by_pair[("S1-A", "S2-A")][3], "1")
        self.assertEqual(by_pair[("S1-A", "S2-A")][-3:], ["1", "0.91", "2"])
        self.assertEqual(by_pair[("S1-B", "S2-Y")][-3:], ["1", "0.77", "1"])
        metrics = evaluate_candidates_only({"S1-A": {"S2-A", "S3-X"}, "S1-B": {"S2-Y"}}, c_out)
        self.assertEqual(metrics["candidate_recall"], 1.0)
        checks = validate_pair_artifacts(s1_ids, c_out, m_out)
        self.assertEqual(checks["candidate_pairs"], 3)
        self.assertTrue(checks["candidate_metadata_agree"])
        subset = verify_v1_subset(v1c, c_out, s1_ids)
        self.assertTrue(subset["v1_subset_of_v2"])
        self.assertEqual(subset["v1_pairs_checked"], 1)

    def test_union_rejects_missing_v1_candidate_metadata(self):
        s1_ids = ["S1-A"]
        v1c, v1m, tf = self.root / "bad.tsv.gz", self.root / "badmeta.tsv.gz", self.root / "empty.tsv.gz"
        with gzip.open(v1c, "wt", encoding="utf-8", newline="") as handle:
            w = csv.writer(handle, delimiter="\t"); w.writerow(CANDIDATE_HEADER); w.writerow(("S1-A", "S2-A"))
        with gzip.open(v1m, "wt", encoding="utf-8", newline="") as handle:
            w = csv.writer(handle, delimiter="\t"); w.writerow(METADATA_HEADER)
        with gzip.open(tf, "wt", encoding="utf-8", newline="") as handle:
            w = csv.writer(handle, delimiter="\t"); w.writerow(TFIDF_HEADER)
        with self.assertRaises(ValueError):
            union_candidates(s1_ids, v1c, v1m, tf, self.root/"o.tsv.gz", self.root/"m.tsv.gz", 5, 5)


if __name__ == "__main__":
    unittest.main()
