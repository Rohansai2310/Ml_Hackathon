from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase112_group_policy import (
    BASE_POLICY, BASELINE_MACRO, GroupPolicy, _bin_name, _degree_key, _evaluate,
    _crossfit_policy, _predict_c, _stable_fold, gap_grid, predictions_for_group_policy,
    shared_offset_grid, confirm_validation, apply_test, _compare_prediction_files,
    CONFIRMED_VALIDATION_MACRO, BASELINE_SUBMISSION_SHA256,
)
from phase9_policy import ScoreCache
from scoring import score_predictions


def cache(source: str, ids: list[str], groups: list[list[tuple[str, float]]]) -> ScoreCache:
    offsets = [0]
    targets, scores = [], []
    for group in groups:
        for target, score in group:
            targets.append(target.encode("ascii")); scores.append(score)
        offsets.append(len(targets))
    n = len(targets)
    return ScoreCache(source, tuple(ids), np.asarray(offsets, dtype=np.int64),
                      np.asarray(targets, dtype="S20"), np.asarray(scores, dtype=np.float32),
                      np.zeros(n, dtype=np.uint8), np.zeros(n, dtype=np.uint8),
                      np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32))


class Phase112GroupPolicyTests(unittest.TestCase):
    def setUp(self):
        self.ids = ["S1-A", "S1-B", "S1-C"]
        self.s2 = cache("S2", self.ids, [
            [("S2-A", .93), ("S2-X", .94)], [("S2-B", .92)], []])
        self.s3 = cache("S3", self.ids, [
            [("S3-A", .97)], [], [("S3-C", .98), ("S3-X", .97)]])
        self.truth = {"S1-A": {"S2-A", "S3-A"}, "S1-B": set(), "S1-C": {"S3-C"}}

    def test_count_bins_are_deterministic_and_source_local(self):
        self.assertEqual([_bin_name(x) for x in (0, 1, 25, 26, 100, 101, 500, 501)],
                         ["0", "1-25", "1-25", "26-100", "26-100", "101-500", "101-500", "501+"])
        self.assertEqual(len(shared_offset_grid()), 20)
        self.assertEqual(len(gap_grid()), 18)
        self.assertEqual(_degree_key(25, 101), (0, 2))

    def test_A_policy_threshold_boundary_and_no_forced_top_one(self):
        policy = GroupPolicy("A", {"low_offset": 0.0, "crowded_offset": 0.0})
        preds = predictions_for_group_policy(self.s2, self.s3, policy)
        self.assertEqual(preds["S1-A"], {"S2-A", "S2-X", "S3-A"})
        self.assertEqual(preds["S1-B"], set())
        self.assertEqual(preds["S1-C"], {"S3-C", "S3-X"})
        relaxed = GroupPolicy("A", {"low_offset": -.015, "crowded_offset": 0.0})
        self.assertIn("S2-B", predictions_for_group_policy(self.s2, self.s3, relaxed)["S1-B"])

    def test_gap_rule_uses_score_gap_and_keeps_multiple_candidates(self):
        policy = GroupPolicy("B_top_gap", {"relax": .01, "minimum_gap": .01})
        preds = predictions_for_group_policy(self.s2, self.s3, policy)
        self.assertIn("S2-X", preds["S1-A"])
        # S1-C's top S3 candidate passes; another independently passing link remains allowed.
        self.assertEqual(preds["S1-C"], {"S3-C", "S3-X"})

    def test_target_ownership_is_highest_score_with_s1_tie_break(self):
        ids = ["S1-Z", "S1-A"]
        s2 = cache("S2", ids, [[("S2-SHARED", .99)], [("S2-SHARED", .99)]])
        s3 = cache("S3", ids, [[], []])
        result = predictions_for_group_policy(s2, s3, GroupPolicy("A", {"low_offset": 0, "crowded_offset": 0}))
        self.assertEqual(result, {"S1-Z": set(), "S1-A": {"S2-SHARED"}})

    def test_evaluation_matches_official_scorer_and_singleton_semantics(self):
        policy = GroupPolicy("A", {"low_offset": 0, "crowded_offset": 0})
        result, _ = _evaluate(self.s2, self.s3, self.truth, policy)
        official = score_predictions(self.truth, result["predictions"], self.ids)
        self.assertEqual(result["macro_f0_5"], official.macro_f0_5)
        self.assertEqual(result["predicted_links"], official.predicted_links)
        self.assertEqual(result["predicted_singletons"], 1)
        self.assertEqual(result["singleton_false_merges"], 0)

    def test_fold_assignment_is_stable(self):
        self.assertEqual(_stable_fold("S1-abc"), _stable_fold("S1-abc"))
        self.assertIn(_stable_fold("S1-abc"), range(5))

    def test_crossfit_is_s1_grouped_and_deterministic(self):
        ids = [f"S1-{i}" for i in range(20)]
        s2 = cache("S2", ids, [[(f"S2-{i}", .2 + .03 * (i % 10))] for i in range(20)])
        s3 = cache("S3", ids, [[(f"S3-{i}", .3 + .02 * (i % 10))] for i in range(20)])
        truth = {s1: ({f"S2-{i}", f"S3-{i}"} if i % 3 == 0 else set()) for i, s1 in enumerate(ids)}
        with tempfile.TemporaryDirectory() as tmp:
            first, info = _crossfit_policy(s2, s3, truth, Path(tmp))
            again, info2 = _crossfit_policy(s2, s3, truth, Path(tmp))
        self.assertEqual(first, again)
        self.assertEqual(info["folds"], 5)
        self.assertEqual(info["oof_mean_proxy"], info2["oof_mean_proxy"])
        self.assertEqual(set(first), set(ids))

    def test_c_policy_can_return_multiple_links_and_reuses_saved_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cal.json"
            path.write_text(json.dumps({
                "models": {"S2": {"x": [0, 1], "y": [.99, .99]}, "S3": {"x": [0, 1], "y": [.99, .99]}},
                "expected_truth_links_by_source_count_bins": {"0,0": 1.0},
                "fallback_expected_truth_links": 1.0,
            }))
            predictions = _predict_c(self.s2, self.s3, path)
        self.assertEqual(predictions["S1-A"], {"S2-A", "S2-X", "S3-A"})
        self.assertEqual(predictions["S1-B"], {"S2-B"})
        self.assertEqual(predictions["S1-C"], {"S3-C", "S3-X"})

    def test_test_prediction_requires_confirmed_validation(self):
        from phase112_group_policy import apply_test
        with tempfile.TemporaryDirectory() as tmp:
            with patch("phase112_group_policy.frozen_snapshot", return_value={}):
                with self.assertRaisesRegex(RuntimeError, "requires a locked tune winner"):
                    apply_test(Path(tmp))

    def test_test_prediction_rejects_unconfirmed_or_wrong_validation_score(self):
        from phase112_group_policy import apply_test
        policy = {"kind": "A_source", "params": {"base_crowded_offset": .015, "base_low_offset": -.015,
                  "source_offsets": {"S2": -.005, "S3": 0.0}}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "locked_winner.json").write_text(json.dumps({"experiment_id": "A_source_02", "policy": policy,
                "frozen_snapshot": {}}))
            (root / "validation_confirmation.json").write_text(json.dumps({"status": "CONFIRMED",
                "validation_macro_f0_5": CONFIRMED_VALIDATION_MACRO - .0001, "policy": policy}))
            with patch("phase112_group_policy.frozen_snapshot", return_value={}):
                with self.assertRaisesRegex(RuntimeError, "exact one-shot CONFIRMED"):
                    apply_test(root)

    def test_test_prediction_rejects_snapshot_drift_and_baseline_hash(self):
        policy = {"kind": "A_source", "params": {"base_crowded_offset": .015, "base_low_offset": -.015,
                  "source_offsets": {"S2": -.005, "S3": 0.0}}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "locked_winner.json").write_text(json.dumps({"experiment_id": "A_source_02", "policy": policy,
                "frozen_snapshot": {"unexpected": {"exists": True}}}))
            (root / "validation_confirmation.json").write_text(json.dumps({"status": "CONFIRMED",
                "validation_macro_f0_5": CONFIRMED_VALIDATION_MACRO, "policy": policy}))
            with patch("phase112_group_policy.frozen_snapshot", return_value={}):
                with self.assertRaisesRegex(RuntimeError, "do not match the locked winner snapshot"):
                    apply_test(root)
            snapshot = {"fixture": {"exists": True}}
            (root / "locked_winner.json").write_text(json.dumps({"experiment_id": "A_source_02", "policy": policy,
                "frozen_snapshot": snapshot}))
            with patch("phase112_group_policy.frozen_snapshot", return_value=snapshot), \
                 patch("phase112_group_policy._sha256", return_value="0" * 64):
                with self.assertRaisesRegex(RuntimeError, "Submission #1 SHA256"):
                    apply_test(root)
        self.assertEqual(BASELINE_SUBMISSION_SHA256, "c7bed709aa2058e1fa08467096e1c2afca0f6d820baab0749bdd88b73ffa2da8")

    def test_prediction_comparison_counts_changes_and_rejects_duplicate_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp) / "old.tsv"; new = Path(tmp) / "new.tsv"
            old.write_text("source1_entity_id\tmatched_entity_ids\nS1-A\tS2-X\nS1-B\t\nS1-C\tS3-Z\n")
            new.write_text("source1_entity_id\tmatched_entity_ids\nS1-A\tS2-X,S2-Y\nS1-B\tS3-B\nS1-C\t\n")
            result = _compare_prediction_files(old, new, expected_rows=3)
            self.assertEqual(result, {"s1_rows": 3, "identical_predictions": 0, "changed_predictions": 3,
                "links_added": 2, "links_removed": 1, "singleton_to_non_singleton": 1,
                "non_singleton_to_singleton": 1})
            new.write_text("source1_entity_id\tmatched_entity_ids\nS1-A\tS2-X,S2-X\nS1-B\tS3-B\nS1-C\t\n")
            with self.assertRaisesRegex(ValueError, "duplicate matched ID"):
                _compare_prediction_files(old, new, expected_rows=3)

    def test_validation_requires_locked_winner_and_is_one_shot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "locked_winner.json").write_text(json.dumps({
                "experiment_id": "A_shared_01",
                "policy": {"kind": "A", "params": {"low_offset": 0, "crowded_offset": 0}},
                "tune_metrics": {"delta_vs_baseline": .002},
            }))
            with patch("phase112_group_policy.frozen_snapshot", return_value={}), \
                 patch("phase112_group_policy._assert_snapshot"), \
                 patch("phase112_group_policy._val_inputs", return_value=(self.s2, self.s3, self.truth)):
                result = confirm_validation(root)
                self.assertIn(result["status"], {"CONFIRMED", "MIXED", "FAILED_TO_GENERALIZE"})
                with self.assertRaisesRegex(RuntimeError, "already been run"):
                    confirm_validation(root)

    def test_runner_does_not_access_hidden_labels_or_retrieve_candidates(self):
        text = (Path(__file__).resolve().parents[1] / "src" / "phase112_group_policy.py").read_text()
        self.assertNotIn("test_ground_truth", text)
        self.assertNotIn("address_retrieve", text)
        self.assertNotIn("generate-candidates", text)
        self.assertEqual(BASE_POLICY.s2_threshold, .93)
        self.assertEqual(BASE_POLICY.s3_threshold, .97)
        self.assertEqual(BASE_POLICY.conflict_policy, "highest")
        self.assertEqual(BASELINE_MACRO, 0.8676284617025911)


if __name__ == "__main__":
    unittest.main()
