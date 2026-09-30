import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scoring import parse_match_list, score_entity, score_predictions


class ScoringTests(unittest.TestCase):
    def test_perfect_single_match(self):
        self.assertEqual(score_entity({"S2-A"}, {"S2-A"}), 1.0)

    def test_perfect_multiple_matches(self):
        self.assertEqual(score_entity({"S2-A", "S3-B"}, {"S2-A", "S3-B"}), 1.0)

    def test_one_correct_one_false_positive(self):
        self.assertAlmostEqual(score_entity({"S2-A"}, {"S2-A", "S3-X"}), 5 / 9)

    def test_one_of_two_true_matches(self):
        self.assertAlmostEqual(score_entity({"S2-A", "S3-B"}, {"S2-A"}), 5 / 6)

    def test_correct_singleton(self):
        self.assertEqual(score_entity(set(), set()), 1.0)

    def test_false_singleton_merge(self):
        self.assertEqual(score_entity(set(), {"S2-X"}), 0.0)

    def test_completely_missed_linked_entity(self):
        self.assertEqual(score_entity({"S2-A"}, set()), 0.0)

    def test_macro_is_average_of_entity_scores(self):
        truth = {"S1-A": {"S2-A"}, "S1-B": set()}
        predictions = {"S1-A": {"S2-A", "S3-X"}, "S1-B": {"S2-X"}}
        report = score_predictions(truth, predictions)
        self.assertAlmostEqual(report.macro_f0_5, (5 / 9 + 0.0) / 2)
        pooled = 1.25 * (1 / 3) * 1.0 / (0.25 * (1 / 3) + 1.0)
        self.assertAlmostEqual(pooled, 5 / 13)
        self.assertNotAlmostEqual(report.macro_f0_5, pooled)

    def test_match_list_parsing(self):
        self.assertEqual(parse_match_list("S2-A,S3-B"), {"S2-A", "S3-B"})
        self.assertEqual(parse_match_list(""), set())

    def test_duplicate_prediction_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_match_list("S2-A,S2-A")

    def test_missing_prediction_entity_rejected(self):
        with self.assertRaisesRegex(ValueError, "coverage"):
            score_predictions({"S1-A": set()}, {})


if __name__ == "__main__":
    unittest.main()
