"""Deterministic Phase 5 heuristic and output contract tests."""
import csv
import gzip
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baseline import (TargetStore, choose_predictions, evaluate_configs, heuristic_score,
                      make_tune_ids, write_error_sample,
                      iter_candidate_groups, make_record, pair_signals, rule_flags,
                      write_predictions_from_scores, SCORE_HEADER)
from blocking import METADATA_HEADER
from diagnostics import CANDIDATE_HEADER, build_diagnostics, stream_candidate_diagnostics
from scoring import load_predictions, score_predictions


def metadata(s1="S1-A", target="S2-A", *, exact=0, core=0, strong=0, routes=1):
    return [s1, target, target[:2], str(exact), str(core), "0", "0", "0", str(strong), "0", str(routes)]


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.left = make_record("S1-A", "Acme Services LLC", "12 Cedar Road, 02108", "US")
        self.same = make_record("S2-A", "ACME Services LLC", "12 Cedar Road, 02108", "US")
        self.legal = make_record("S3-B", "Acme Services Inc", "12 Cedar Road, 02108", "US")

    def test_exact_basic_name(self):
        s = pair_signals(self.left, self.same, metadata(exact=1, core=1, routes=2))
        self.assertTrue(s["exact_basic"])
        self.assertEqual(rule_flags(s)[:2], (True, True))

    def test_exact_core_legal_variant(self):
        s = pair_signals(self.left, self.legal, metadata(target="S3-B", core=1))
        self.assertFalse(s["exact_basic"])
        self.assertTrue(s["exact_core"])
        self.assertTrue(rule_flags(s)[2])

    def test_strong_name_and_address_score(self):
        good = pair_signals(self.left, self.same, metadata(exact=1, core=1, routes=2))
        weak = pair_signals(self.left, make_record("S2-X", "Other Business", "", "US"), metadata(target="S2-X"))
        self.assertGreater(heuristic_score(good), heuristic_score(weak))

    def test_missing_address_does_not_reject(self):
        s = pair_signals(make_record("S1-A", "Acme Services LLC", "", "US"),
                         make_record("S2-A", "Acme Services LLC", "", "US"), metadata())
        self.assertEqual(s["address_ratio"], 0)
        self.assertGreater(heuristic_score(s), 80)

    def test_numeric_conflict_reduces_score(self):
        agreeing = pair_signals(self.left, self.same, metadata())
        conflict = pair_signals(self.left, make_record("S2-A", "Acme Services LLC", "92 Cedar Road, 02108", "US"), metadata())
        self.assertTrue(conflict["number_conflict"])
        self.assertLess(heuristic_score(conflict), heuristic_score(agreeing))

    def test_postal_agreement_and_conflict(self):
        same = pair_signals(self.left, self.same, metadata())
        other = pair_signals(self.left, make_record("S2-A", "Acme Services LLC", "12 Cedar Road, 90210", "US"), metadata())
        self.assertTrue(same["postal_shared"])
        self.assertTrue(other["postal_conflict"])
        self.assertLess(heuristic_score(other), heuristic_score(same))

    def test_zip_plus_four_agrees_with_five_digit_zip(self):
        extended = make_record("S2-A", "Acme Services LLC", "12 Cedar Road, 02108-1234", "US")
        signals = pair_signals(self.left, extended, metadata())
        self.assertTrue(signals["postal_shared"])
        self.assertFalse(signals["postal_conflict"])

    def test_token_and_character_similarity(self):
        partial = pair_signals(self.left, make_record("S2-A", "Acme Services Group", "", "US"), metadata())
        self.assertGreater(partial["core_ratio"], 0)
        self.assertGreater(partial["token_jaccard"], 0)
        self.assertGreater(partial["token_set_ratio"], 0)

    def test_score_determinism_and_bounds(self):
        s = pair_signals(self.left, self.same, metadata())
        self.assertEqual(heuristic_score(s), heuristic_score(s))
        self.assertGreaterEqual(heuristic_score(s), 0)
        self.assertLessEqual(heuristic_score(s), 100)

    def test_open_set_country(self):
        a = make_record("S1-A", "Neptune", "5 Ocean Road", "Atlantis")
        b = make_record("S2-A", "Neptune", "5 Ocean Road", "Atlantis")
        self.assertGreater(heuristic_score(pair_signals(a, b, metadata())), 0)

    def test_country_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "country"):
            pair_signals(self.left, make_record("S2-A", "Acme", "", "India"), metadata())

    def test_blank_fields_are_safe(self):
        s = pair_signals(make_record("S1-A", "", "", "US"),
                         make_record("S2-A", "", "", "US"), metadata())
        self.assertEqual(heuristic_score(s), 0)
        self.assertEqual(rule_flags(s), (False, False, False))

    def test_zero_one_and_many_predictions_without_argmax(self):
        pairs = [("S2-A", 90.0, True, True, True), ("S3-B", 85.0, False, True, True)]
        self.assertEqual(choose_predictions(pairs, ("D", 95, 95)), set())
        self.assertEqual(choose_predictions(pairs, ("D", 88, 88)), {"S2-A"})
        self.assertEqual(choose_predictions(pairs, ("D", 80, 80)), {"S2-A", "S3-B"})

    def test_source_specific_thresholds(self):
        pairs = [("S2-A", 80.0, False, False, False), ("S3-B", 80.0, False, False, False)]
        self.assertEqual(choose_predictions(pairs, ("D", 81, 79)), {"S3-B"})

    def test_three_fixed_ablation_rules(self):
        pairs = [("S2-A", 60.0, True, True, False), ("S3-B", 60.0, False, True, True)]
        self.assertEqual(choose_predictions(pairs, ("A", 0, 0)), {"S2-A"})
        self.assertEqual(choose_predictions(pairs, ("B", 0, 0)), {"S2-A", "S3-B"})
        self.assertEqual(choose_predictions(pairs, ("C", 0, 0)), {"S3-B"})

    def test_candidate_stream_checks_coverage_and_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_path, meta_path = root / "c.gz", root / "m.gz"
            with gzip.open(candidate_path, "wt", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t")
                writer.writerow(CANDIDATE_HEADER)
                writer.writerow(["S1-A", "S2-A,S3-B"])
                writer.writerow(["S1-B", ""])
            with gzip.open(meta_path, "wt", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t")
                writer.writerow(METADATA_HEADER)
                writer.writerow(metadata(target="S2-A"))
                writer.writerow(metadata(target="S3-B"))
            rows = list(iter_candidate_groups(candidate_path, meta_path, ["S1-A", "S1-B"]))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0][1], ["S2-A", "S3-B"])
            self.assertEqual(rows[1][1], [])
            with self.assertRaisesRegex(ValueError, "Extra candidate"):
                list(iter_candidate_groups(candidate_path, meta_path, ["S1-A"]))

    def test_duplicate_candidate_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            c, m = root / "c.gz", root / "m.gz"
            with gzip.open(c, "wt", newline="") as f:
                f.write("source1_entity_id\tcandidate_entity_ids\nS1-A\tS2-A,S2-A\n")
            with gzip.open(m, "wt", newline="") as f:
                f.write("\t".join(METADATA_HEADER) + "\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                list(iter_candidate_groups(c, m, ["S1-A"]))

    def test_prediction_coverage_subset_and_macro(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scores, predictions = root / "scores.gz", root / "pred.tsv"
            with gzip.open(scores, "wt", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(SCORE_HEADER)
                writer.writerow(["S1-A", "S2-A", "92", "1", "1", "1"])
                writer.writerow(["S1-A", "S3-B", "80", "0", "1", "1"])
            ids = ["S1-A", "S1-B"]
            truth = {"S1-A": {"S2-A", "S3-B"}, "S1-B": set()}
            reports = evaluate_configs(scores, ids, truth, [("D", 85, 85)])
            write_predictions_from_scores(scores, ids, ("D", 85, 85), predictions)
            loaded = load_predictions(predictions)
            self.assertEqual(set(loaded), set(ids))
            self.assertEqual(loaded["S1-B"], set())
            self.assertEqual(loaded["S1-A"], {"S2-A"})
            self.assertEqual(reports[0]["macro_f0_5"], score_predictions(truth, loaded).macro_f0_5)

    def test_streaming_candidate_diagnostics_gzip(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "candidates.tsv.gz"
            with gzip.open(path, "wt", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t")
                writer.writerow(CANDIDATE_HEADER)
                writer.writerow(["S1-A", "S2-A,S3-B"])
                writer.writerow(["S1-B", ""])
            truth = {"S1-A": {"S2-A", "S3-X"}, "S1-B": set()}
            predicted = {"S1-A": {"S2-A"}, "S1-B": set()}
            result = stream_candidate_diagnostics(truth, predicted, path)
            self.assertEqual(result.blocking_misses, 1)
            self.assertEqual(result.model_misses, 0)
            self.assertEqual(result.candidate_recall, 0.5)
            self.assertEqual(build_diagnostics(truth, predicted, result).candidates, result)
            with self.assertRaisesRegex(ValueError, "outside candidates"):
                stream_candidate_diagnostics(truth, {"S1-A": {"S3-X"}, "S1-B": set()}, path)

    def test_target_fetch_batches_and_missing_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            db = Path(temporary) / "index.sqlite"
            connection = sqlite3.connect(db)
            connection.execute("CREATE TABLE target (target INTEGER PRIMARY KEY, entity_id TEXT UNIQUE, source TEXT, country TEXT, name TEXT, address TEXT, basic TEXT, core TEXT)")
            connection.execute("INSERT INTO target VALUES (1,'S2-A','S2','us','Acme','12 Road','acme','acme')")
            connection.execute("INSERT INTO target VALUES (2,'S2-B','S2','us','Beta','14 Road','beta','beta')")
            connection.execute("INSERT INTO target VALUES (3,'S2-C','S2','us','Cedar','16 Road','cedar','cedar')")
            connection.commit(); connection.close()
            store = TargetStore(db, cache_size=2)
            try:
                self.assertEqual(store.get_many(["S2-A"])["S2-A"].basic, "acme")
                self.assertEqual(store.get_many(["S2-A"])["S2-A"].basic, "acme")
                self.assertEqual(store.queries, 1)
                store.get_many(["S2-B"])
                self.assertEqual(set(store.get_many(["S2-A", "S2-C"])), {"S2-A", "S2-C"})
                with self.assertRaisesRegex(ValueError, "Duplicate"):
                    store.get_many(["S2-A", "S2-A"])
                with self.assertRaisesRegex(ValueError, "absent"):
                    store.get_many(["S2-Z"])
            finally:
                store.close()


    def test_tune_split_is_deterministic_and_nonoverlapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, val, output = root / "train.txt", root / "val.txt", root / "tune.txt"
            train.write_text("S1-A\nS1-B\nS1-C\nS1-D\n")
            val.write_text("S1-X\n")
            first = make_tune_ids(train, val, output, count=2, seed=20260925)
            text = output.read_text()
            second = make_tune_ids(train, val, output, count=2, seed=20260925)
            self.assertEqual(text, output.read_text())
            self.assertEqual(first, second)
            self.assertEqual(first["validation_overlap"], 0)
            self.assertEqual(len(text.splitlines()), 2)

    def test_error_categories_and_bounded_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train"
            train.mkdir()
            with (train / "train_source1.tsv").open("w", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(["entity_id", "business_name", "business_address", "country"])
                writer.writerows([["S1-A", "Acme", "12 Road", "US"],
                                  ["S1-B", "Beta", "14 Road", "US"]])
            db = root / "index.sqlite"
            connection = sqlite3.connect(db)
            connection.execute("CREATE TABLE target (target INTEGER PRIMARY KEY, entity_id TEXT UNIQUE, source TEXT, country TEXT, name TEXT, address TEXT, basic TEXT, core TEXT)")
            for i, target in enumerate(["S2-A", "S2-W", "S3-X"], 1):
                connection.execute("INSERT INTO target VALUES (?,?,?,?,?,?,?,?)",
                                   (i, target, target[:2], "us", target, "12 Road", target.lower(), target.lower()))
            connection.commit(); connection.close()
            scores, meta, output = root / "scores.gz", root / "metadata.gz", root / "errors.tsv"
            with gzip.open(scores, "wt", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(SCORE_HEADER)
                writer.writerow(["S1-A", "S2-A", "40", "0", "0", "0"])
                writer.writerow(["S1-A", "S2-W", "90", "0", "0", "0"])
                writer.writerow(["S1-B", "S3-X", "90", "0", "0", "0"])
            with gzip.open(meta, "wt", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(METADATA_HEADER)
                writer.writerow(metadata(target="S2-A"))
                writer.writerow(metadata(target="S2-W"))
                writer.writerow(metadata(s1="S1-B", target="S3-X"))
            truth = {"S1-A": {"S2-A", "S3-MISSING"}, "S1-B": set()}
            # Include the blocked target in SQLite for inspectable output.
            connection = sqlite3.connect(db)
            connection.execute("INSERT INTO target VALUES (4,'S3-MISSING','S3','us','Missing','', 'missing','missing')")
            connection.commit(); connection.close()
            predictions = {"S1-A": {"S2-W"}, "S1-B": {"S3-X"}}
            counts = write_error_sample(scores, ["S1-A", "S1-B"], truth, predictions,
                                        meta, root, db, output, per_category=2)
            self.assertEqual(counts["false_positive"], 2)
            self.assertEqual(counts["model_miss"], 1)
            self.assertEqual(counts["blocking_miss"], 1)
            self.assertEqual(counts["singleton_false_merge"], 1)
            with output.open(newline="") as f:
                self.assertEqual(len(list(csv.DictReader(f, delimiter="\t"))), 5)

if __name__ == "__main__":
    unittest.main()
