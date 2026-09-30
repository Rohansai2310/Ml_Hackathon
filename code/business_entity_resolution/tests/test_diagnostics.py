import contextlib
import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diagnostics import (
    EXPERIMENT_COLUMNS,
    append_experiment,
    build_diagnostics,
    main,
    render_report,
)
from scoring import parse_match_list, score_predictions


TRUTH = {
    "S1-A": {"S2-X"},
    "S1-B": set(),
    "S1-C": set(),
    "S1-D": {"S2-A", "S3-B"},
    "S1-E": {"S3-Q"},
}
PREDICTIONS = {
    "S1-A": {"S2-X"},
    "S1-B": set(),
    "S1-C": {"S3-Z"},
    "S1-D": {"S2-A"},
    "S1-E": {"S2-W"},
}
CANDIDATES = {
    "S1-A": {"S2-X"},
    "S1-B": set(),
    "S1-C": {"S3-Z"},
    "S1-D": {"S2-A", "S3-B"},
    "S1-E": {"S2-W"},
}


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.result = build_diagnostics(TRUTH, PREDICTIONS)

    def test_overall_counts_and_macro_reuse_scorer(self):
        score = self.result.score
        self.assertEqual(score.entities_evaluated, 5)
        self.assertEqual(score.true_links, 4)
        self.assertEqual(score.predicted_links, 4)
        self.assertEqual((score.tp, score.fp, score.fn), (2, 2, 2))
        self.assertAlmostEqual(score.macro_f0_5, 17 / 30)
        self.assertEqual(score.macro_f0_5, score_predictions(TRUTH, PREDICTIONS).macro_f0_5)

    def test_singleton_diagnostics(self):
        score = self.result.score
        self.assertEqual(score.true_singletons, 2)
        self.assertEqual(self.result.predicted_singletons, 1)
        self.assertEqual(score.correctly_predicted_singletons, 1)
        self.assertEqual(self.result.singleton_false_merges, 1)
        self.assertEqual(score.singleton_accuracy, 0.5)

    def test_s2_diagnostics(self):
        s2 = self.result.source2
        self.assertEqual((s2.true_links, s2.predicted_links), (2, 3))
        self.assertEqual((s2.tp, s2.fp, s2.fn), (2, 1, 0))
        self.assertEqual((s2.precision_diagnostic, s2.recall_diagnostic), (2 / 3, 1.0))

    def test_s3_diagnostics(self):
        s3 = self.result.source3
        self.assertEqual((s3.true_links, s3.predicted_links), (2, 1))
        self.assertEqual((s3.tp, s3.fp, s3.fn), (0, 1, 2))
        self.assertEqual((s3.precision_diagnostic, s3.recall_diagnostic), (0.0, 0.0))

    def test_entity_quality_counts_and_report(self):
        score = self.result.score
        self.assertEqual(score.perfect_match_sets, 2)
        self.assertEqual(score.partial_match_sets, 1)
        self.assertEqual(score.completely_missed_nonempty_truth, 1)
        self.assertEqual(self.result.false_positive_entities, 2)
        report = render_report(self.result)
        self.assertEqual(report, render_report(self.result))
        self.assertIn("Not available yet (no candidate file supplied).", report)
        self.assertIn("Macro entity-level F0.5 (primary challenge metric): 0.56666667", report)
        self.assertIn("FALSE_POSITIVE / FALSE_MERGE predicted links outside truth: 2", report)
        self.assertIn("SINGLETON_FALSE_MERGE count: 1", report)

    def test_empty_match_list_parsing(self):
        self.assertEqual(parse_match_list(""), set())

    def test_candidate_recall_misses_and_distribution(self):
        result = build_diagnostics(TRUTH, PREDICTIONS, CANDIDATES).candidates
        self.assertEqual(result.candidate_recall, 3 / 4)
        self.assertEqual(result.s2_candidate_recall, 1.0)
        self.assertEqual(result.s3_candidate_recall, 0.5)
        self.assertEqual(result.blocking_misses, 1)
        self.assertEqual(result.model_misses, 1)
        self.assertEqual(result.mean_candidates, 1.0)
        self.assertEqual(result.median_candidates, 1.0)
        self.assertAlmostEqual(result.p90_candidates, 1.6)
        self.assertAlmostEqual(result.p95_candidates, 1.8)
        self.assertAlmostEqual(result.p99_candidates, 1.96)
        self.assertEqual(result.max_candidates, 2)

    def test_candidate_coverage_required(self):
        with self.assertRaisesRegex(ValueError, "Candidate S1 coverage"):
            build_diagnostics(TRUTH, PREDICTIONS, {"S1-A": set()})

    def test_experiment_csv_append_duplicate_and_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "experiments.csv"
            append_experiment(path, "fixture", "first", self.result, 1.25, timestamp="2026-09-25T00:00:00+00:00")
            append_experiment(path, "second", "second row", self.result, 2.5, timestamp="2026-09-25T00:00:01+00:00")
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, EXPERIMENT_COLUMNS)
                rows = list(reader)
            self.assertEqual([row["experiment_name"] for row in rows], ["fixture", "second"])
            self.assertEqual(rows[0]["candidate_recall"], "")
            with self.assertRaisesRegex(ValueError, "already exists"):
                append_experiment(path, "fixture", "duplicate", self.result, 3.0)
            append_experiment(path, "fixture", "allowed", self.result, 3.0, allow_duplicate=True)
            with path.open(encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 3)

    def test_cli_smoke_saves_report_and_logs_experiment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            (temp / "gt.tsv").write_text(
                "source1_entity_id\tmatched_entity_ids\nS1-A\tS2-X\nS1-B\t\n", encoding="utf-8"
            )
            (temp / "pred.tsv").write_text(
                "source1_entity_id\tmatched_entity_ids\nS1-A\tS2-X\nS1-B\t\n", encoding="utf-8"
            )
            (temp / "ids.txt").write_text("S1-A\nS1-B\n", encoding="utf-8")
            (temp / "candidates.tsv").write_text(
                "source1_entity_id" + chr(9) + "candidate_entity_ids" + chr(10) + "S1-A" + chr(9) + "S2-X" + chr(10) + "S1-B" + chr(9) + chr(10),
                encoding="utf-8"
            )
            output, experiments = temp / "summary.txt", temp / "experiments.csv"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main([
                    "--ground-truth", str(temp / "gt.tsv"),
                    "--predictions", str(temp / "pred.tsv"),
                    "--s1-ids", str(temp / "ids.txt"),
                    "--candidates", str(temp / "candidates.tsv"),
                    "--output", str(output),
                    "--experiment-name", "cli-smoke",
                    "--experiments-file", str(experiments),
                ])
            self.assertEqual(status, 0)
            self.assertEqual(output.read_text(encoding="utf-8"), stdout.getvalue())
            self.assertIn("Overall candidate recall / pair completeness: 1.00000000", stdout.getvalue())
            self.assertTrue(experiments.is_file())



    def test_streamed_v1_v2_candidate_comparison(self):
        import gzip
        from diagnostics import compare_candidate_files
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            v1, v2 = root / "v1.tsv.gz", root / "v2.tsv.gz"
            rows1 = [("S1-A", "S2-X"), ("S1-B", ""), ("S1-C", ""),
                     ("S1-D", "S2-A"), ("S1-E", "")]
            rows2 = [("S1-A", "S2-X"), ("S1-B", ""), ("S1-C", "S3-Z"),
                     ("S1-D", "S2-A,S3-B"), ("S1-E", "S2-W,S3-Q")]
            for path, rows in ((v1, rows1), (v2, rows2)):
                with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                    writer.writerow(["source1_entity_id", "candidate_entity_ids"])
                    writer.writerows(rows)
            metrics = compare_candidate_files(TRUTH, v1, v2)
            self.assertEqual(metrics["v1_blocking_misses"], 2)
            self.assertEqual(metrics["v1_misses_recovered"], 2)
            self.assertEqual(metrics["v2_blocking_misses"], 0)
            self.assertEqual(metrics["s3_v1_misses_recovered"], 2)
            self.assertEqual(metrics["candidate_pair_growth"], 4)

if __name__ == "__main__":
    unittest.main()
