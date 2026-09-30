from __future__ import annotations

import csv
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase9_policy import (
    DecisionPolicy,
    SCORE_HEADER,
    ScoreCache,
    evaluate_policy,
    load_cache,
    predictions_for_policy,
    read_score_file,
    save_cache,
    write_predictions,
)
from scoring import score_predictions


def cache(source: str, ids: list[str], groups: list[list[tuple[str, float]]]) -> ScoreCache:
    offsets = [0]
    targets: list[bytes] = []
    scores: list[float] = []
    for group in groups:
        for target, score in group:
            targets.append(target.encode("ascii")); scores.append(score)
        offsets.append(len(targets))
    count = len(targets)
    return ScoreCache(
        source, tuple(ids), np.asarray(offsets, dtype=np.int64), np.asarray(targets, dtype="S20"),
        np.asarray(scores, dtype=np.float32), np.zeros(count, dtype=np.uint8),
        np.zeros(count, dtype=np.uint8), np.zeros(count, dtype=np.float32), np.zeros(count, dtype=np.float32),
    )


class Phase9PolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ids = ["S1-A", "S1-B", "S1-C", "S1-D"]
        self.s2 = cache("S2", self.ids, [
            [("S2-A", 0.90), ("S2-X", 0.80)], [], [("S2-C", 0.70)], [("S2-D", 0.95)],
        ])
        self.s3 = cache("S3", self.ids, [
            [], [("S3-B", 0.85)], [("S3-C", 0.92)], [("S3-D", 0.10)],
        ])
        self.truth = {"S1-A": {"S2-A"}, "S1-B": set(), "S1-C": {"S2-C", "S3-C"}, "S1-D": {"S2-D"}}

    def test_threshold_boundaries_and_no_top_one_forcing(self) -> None:
        predictions = predictions_for_policy(self.s2, self.s3, DecisionPolicy(0.80, 0.85))
        self.assertEqual(predictions["S1-A"], {"S2-A", "S2-X"})
        self.assertEqual(predictions["S1-B"], {"S3-B"})
        self.assertEqual(predictions["S1-C"], {"S3-C"})
        self.assertEqual(predictions["S1-D"], {"S2-D"})
        self.assertEqual(predictions_for_policy(self.s2, self.s3, DecisionPolicy(0.99, 0.99))["S1-A"], set())

    def test_exact_scorer_singletons_multiple_matches_and_diagnostics(self) -> None:
        policy = DecisionPolicy(0.70, 0.90)
        result = evaluate_policy(self.s2, self.s3, self.truth, policy)
        expected = score_predictions(self.truth, result["predictions"], self.ids)
        self.assertEqual(result["macro_f0_5"], expected.macro_f0_5)
        self.assertEqual(result["true_singletons"], 1)
        self.assertEqual(result["singleton_false_merges"], 0)
        self.assertEqual(result["blocking_misses"], 0)
        self.assertEqual(result["decision_misses"], 0)
        self.assertEqual(result["predictions"]["S1-C"], {"S2-C", "S3-C"})

    def test_false_singleton_merge_and_empty_prediction(self) -> None:
        result = evaluate_policy(self.s2, self.s3, self.truth, DecisionPolicy(0.99, 0.80))
        self.assertEqual(result["predictions"]["S1-B"], {"S3-B"})
        self.assertEqual(result["singleton_false_merges"], 1)
        self.assertEqual(result["predictions"]["S1-A"], set())

    def test_open_policy_gates_all_links_for_closed_entity(self) -> None:
        predictions = predictions_for_policy(self.s2, self.s3, DecisionPolicy(0.70, 0.80, open_threshold=0.93))
        self.assertEqual(predictions["S1-C"], set())
        self.assertEqual(predictions["S1-D"], {"S2-D"})

    def test_conflict_resolution_is_deterministic(self) -> None:
        ids = ["S1-A", "S1-B"]
        s2 = cache("S2", ids, [[("S2-X", 0.90)], [("S2-X", 0.90)]])
        s3 = cache("S3", ids, [[], []])
        highest = predictions_for_policy(s2, s3, DecisionPolicy(0.5, 0.5, conflict_policy="highest"))
        self.assertEqual(highest, {"S1-A": {"S2-X"}, "S1-B": set()})
        retained = predictions_for_policy(s2, s3, DecisionPolicy(0.5, 0.5, conflict_policy="margin", conflict_margin=0.01))
        self.assertEqual(retained, {"S1-A": {"S2-X"}, "S1-B": {"S2-X"}})

    def test_cache_round_trip_and_prediction_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "cache.npz"
            save_cache(self.s2, path)
            loaded = load_cache(path)
            self.assertEqual(loaded.s1_ids, self.s2.s1_ids)
            np.testing.assert_array_equal(loaded.scores, self.s2.scores)
            prediction_path = root / "predictions.tsv.gz"
            predictions = predictions_for_policy(self.s2, self.s3, DecisionPolicy(0.7, 0.9))
            write_predictions(prediction_path, predictions, self.ids)
            with gzip.open(prediction_path, "rt", newline="") as handle:
                rows = list(csv.reader(handle, delimiter="\t"))
            self.assertEqual(rows[0], ["source1_entity_id", "matched_entity_ids"])
            self.assertEqual([row[0] for row in rows[1:]], self.ids)

    def test_score_file_integrity_duplicate_probability_and_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scores.tsv.gz"
            with gzip.open(path, "wt", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=SCORE_HEADER, delimiter="\t")
                writer.writeheader()
                writer.writerow({"source1_entity_id": "S1-A", "candidate_entity_id": "S2-X", "score": "0.5", "from_v1": "1", "from_address": "0", "address_rank": "", "address_score": ""})
            loaded = read_score_file(path, "S2", ["S1-A"], 1)
            self.assertEqual(loaded.group(0)[0][0].decode(), "S2-X")
            with gzip.open(path, "wt", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=SCORE_HEADER, delimiter="\t"); writer.writeheader()
                for _ in range(2):
                    writer.writerow({"source1_entity_id": "S1-A", "candidate_entity_id": "S2-X", "score": "0.5", "from_v1": "1", "from_address": "0", "address_rank": "", "address_score": ""})
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_score_file(path, "S2", ["S1-A"], 2)
            with gzip.open(path, "wt", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=SCORE_HEADER, delimiter="\t"); writer.writeheader()
                writer.writerow({"source1_entity_id": "S1-A", "candidate_entity_id": "S3-X", "score": "1.1", "from_v1": "1", "from_address": "0", "address_rank": "", "address_score": ""})
            with self.assertRaises(ValueError):
                read_score_file(path, "S2", ["S1-A"], 1)

    def test_policy_json_round_trip(self) -> None:
        policy = DecisionPolicy(0.93, 0.95, 0.90, "margin", 0.02)
        restored = DecisionPolicy(**json.loads(json.dumps(policy.__dict__)))
        self.assertEqual(policy, restored)


if __name__ == "__main__":
    unittest.main()
