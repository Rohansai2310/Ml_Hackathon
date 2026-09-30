from __future__ import annotations

import csv
import gzip
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blocking import METADATA_HEADER
from diagnostics import CANDIDATE_HEADER
from phase8_model import ADDRESS_HEADER
from phase9_policy import DecisionPolicy, ScoreCache, evaluate_policy
from scoring import score_predictions
import validation_evaluation as validation
from validation_evaluation import (
    assert_frozen_unchanged,
    candidate_metrics,
    frozen_policy,
    frozen_snapshot,
    format_summary,
    validate_scored_membership,
)


def make_cache(source: str, ids: list[str], rows: list[list[tuple[str, float]]]) -> ScoreCache:
    offsets = [0]
    targets: list[bytes] = []
    scores: list[float] = []
    for group in rows:
        for target, score in group:
            targets.append(target.encode("ascii"))
            scores.append(score)
        offsets.append(len(targets))
    count = len(targets)
    return ScoreCache(source, tuple(ids), np.asarray(offsets, dtype=np.int64),
                      np.asarray(targets, dtype="S20"), np.asarray(scores, dtype=np.float32),
                      np.zeros(count, dtype=np.uint8), np.zeros(count, dtype=np.uint8),
                      np.zeros(count, dtype=np.float32), np.zeros(count, dtype=np.float32))


class ValidationEvaluationTests(unittest.TestCase):
    def test_frozen_phase9_policy_is_read_and_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "policy.json"
            payload = {
                "policy": {"s2_threshold": 0.93, "s3_threshold": 0.97,
                           "open_threshold": None, "conflict_policy": "highest",
                           "conflict_margin": None},
                "feature_manifest": str(validation.PHASE8 / "feature_manifest.json"),
                "models": {source: str(validation.PHASE8 / f"model_{source.lower()}.txt")
                           for source in ("S2", "S3")},
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            policy, original = frozen_policy(path)
            self.assertEqual(policy, DecisionPolicy(0.93, 0.97, None, "highest", None))
            self.assertEqual(original, payload)
            payload["policy"]["s2_threshold"] = 0.92
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                frozen_policy(path)

    def test_candidate_metrics_report_overall_and_source_recall(self) -> None:
        ids = ["S1-A", "S1-B", "S1-C"]
        s2 = make_cache("S2", ids, [[("S2-A", 0.9)], [], [("S2-C", 0.7)]])
        s3 = make_cache("S3", ids, [[], [("S3-B", 0.8)], []])
        truth = {"S1-A": {"S2-A", "S3-X"}, "S1-B": {"S3-B"}, "S1-C": {"S2-X"}}
        result = candidate_metrics(ids, truth, s2, s3)
        self.assertEqual(result["overall"]["true_links"], 4)
        self.assertEqual(result["overall"]["recovered"], 2)
        self.assertEqual(result["overall"]["blocking_misses"], 2)
        self.assertEqual(result["by_source"]["S2"]["candidate_pairs"], 2)
        self.assertEqual(result["by_source"]["S2"]["recovered"], 1)
        self.assertEqual(result["by_source"]["S3"]["recovered"], 1)
        self.assertEqual(result["candidate_count_per_s1"]["zero_candidate_s1"], 0)

    def test_existing_v1_rows_are_preserved_in_candidate_union(self) -> None:
        ids = ["S1-A", "S1-B"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidates = root / "v1.tsv.gz"
            metadata = root / "v1_meta.tsv.gz"
            address = root / "address.tsv.gz"
            with gzip.open(candidates, "wt", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t")
                writer.writerow(CANDIDATE_HEADER)
                writer.writerow(["S1-A", "S2-A,S3-A"])
                writer.writerow(["S1-B", ""])
            with gzip.open(metadata, "wt", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t")
                writer.writerow(METADATA_HEADER)
                for target, source in (("S2-A", "S2"), ("S3-A", "S3")):
                    writer.writerow(["S1-A", target, source, *(["1"] + ["0"] * 5), "1", "1"])
            with gzip.open(address, "wt", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t")
                writer.writerow(ADDRESS_HEADER)
                writer.writerow(["S1-A", "S2-A", "S2", "address_token", "3.5", "1"])
                writer.writerow(["S1-A", "S2-X", "S2", "address_token", "4.5", "2"])
            from phase8_model import iter_union_groups
            groups = list(iter_union_groups(ids, candidates, metadata, address))
            self.assertEqual(set(groups[0][1]), {"S2-A", "S3-A", "S2-X"})
            self.assertEqual(groups[0][1]["S2-A"]["from_v1"], 1)
            self.assertEqual(groups[0][1]["S2-A"]["from_address"], 1)
            self.assertEqual(groups[0][1]["S2-X"]["from_address"], 1)
            self.assertEqual(groups[1][1], {})

    def test_score_candidate_membership_and_s2_s3_separation(self) -> None:
        ids = ["S1-A", "S1-B"]
        s2 = make_cache("S2", ids, [[("S2-A", 0.99)], []])
        s3 = make_cache("S3", ids, [[], [("S3-B", 0.99)]])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidates, metadata, address = root / "c.tsv.gz", root / "m.tsv.gz", root / "a.tsv.gz"
            with gzip.open(candidates, "wt", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t"); writer.writerow(CANDIDATE_HEADER)
                writer.writerow(["S1-A", "S2-A"]); writer.writerow(["S1-B", "S3-B"])
            with gzip.open(metadata, "wt", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t"); writer.writerow(METADATA_HEADER)
                writer.writerow(["S1-A", "S2-A", "S2", *(["1"] + ["0"] * 5), "1", "1"])
                writer.writerow(["S1-B", "S3-B", "S3", *(["1"] + ["0"] * 5), "1", "1"])
            with gzip.open(address, "wt", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t"); writer.writerow(ADDRESS_HEADER)
            validate_scored_membership(ids, address, s2, s3, candidates, metadata)

    def test_phase9_decision_policy_and_exact_scorer_reuse(self) -> None:
        ids = ["S1-A", "S1-B", "S1-C"]
        s2 = make_cache("S2", ids, [[("S2-A", 0.99)], [], [("S2-X", 0.98)]])
        s3 = make_cache("S3", ids, [[], [("S3-B", 0.99)], []])
        truth = {"S1-A": {"S2-A"}, "S1-B": set(), "S1-C": {"S2-X"}}
        result = evaluate_policy(s2, s3, truth, DecisionPolicy(0.93, 0.97, conflict_policy="highest"))
        exact = score_predictions(truth, result["predictions"], ids)
        self.assertEqual(result["macro_f0_5"], exact.macro_f0_5)
        self.assertEqual(result["predictions"], {"S1-A": {"S2-A"}, "S1-B": {"S3-B"}, "S1-C": {"S2-X"}})
        self.assertEqual(result["entities_evaluated"], len(ids))
        self.assertEqual(result["singleton_false_merges"], 1)

    def test_frozen_snapshot_guard_and_highest_target_ownership(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frozen.bin"
            path.write_bytes(b"fixed")
            snapshot = frozen_snapshot([path])
            assert_frozen_unchanged(snapshot)
            path.write_bytes(b"changed")
            with self.assertRaises(RuntimeError):
                assert_frozen_unchanged(snapshot)
        ids = ["S1-A", "S1-B"]
        s2 = make_cache("S2", ids, [[("S2-SHARED", 0.95)], [("S2-SHARED", 0.94)]])
        s3 = make_cache("S3", ids, [[], []])
        result = evaluate_policy(s2, s3, {"S1-A": {"S2-SHARED"}, "S1-B": set()},
                                 DecisionPolicy(0.93, 0.97, conflict_policy="highest"))
        self.assertEqual(result["predictions"], {"S1-A": {"S2-SHARED"}, "S1-B": set()})
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 0)

    def test_evaluate_stage_writes_complete_report_from_tiny_fixture(self) -> None:
        ids = ["S1-A", "S1-B"]
        truth = {"S1-A": {"S2-A"}, "S1-B": set()}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with gzip.open(output / validation.ADDRESS_CANDIDATES.name, "wt", newline="") as handle:
                csv.writer(handle, delimiter="\t").writerow(ADDRESS_HEADER)
            for source, s1_id, target in (("S2", "S1-A", "S2-A"), ("S3", "S1-B", "S3-B")):
                with gzip.open(output / f"validation_scores_{source.lower()}.tsv.gz", "wt", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=validation.SCORE_HEADER, delimiter="\t")
                    writer.writeheader()
                    writer.writerow({"source1_entity_id": s1_id, "candidate_entity_id": target,
                                     "score": "0.99", "from_v1": "1", "from_address": "0",
                                     "address_rank": "", "address_score": ""})
            (output / "scoring_report.json").write_text(json.dumps({
                "rows": {"S2": 1, "S3": 1}, "runtime_seconds": 2.0, "peak_rss_mb": 100.0,
            }), encoding="utf-8")
            (output / "address_generation_report.json").write_text(json.dumps({
                "runtime_seconds": 3.0, "peak_rss_mb": 90.0,
            }), encoding="utf-8")
            with mock.patch.object(validation, "validation_ids", return_value=ids), \
                 mock.patch.object(validation, "frozen_snapshot", return_value={}), \
                 mock.patch.object(validation, "assert_frozen_unchanged"), \
                 mock.patch.object(validation, "validate_scored_membership"), \
                 mock.patch.object(validation, "load_ground_truth", return_value=truth), \
                 mock.patch.object(validation, "frozen_policy", return_value=(
                     DecisionPolicy(0.93, 0.97, conflict_policy="highest"), {})):
                report = validation.evaluate_validation(output)
            self.assertEqual(report["candidate_pairs"], 2)
            self.assertEqual(report["prediction_metrics"]["macro_f0_5"],
                             score_predictions(truth, {"S1-A": {"S2-A"}, "S1-B": {"S3-B"}}, ids).macro_f0_5)
            self.assertEqual(report["prediction_metrics"]["S2"]["tp"], 1)
            self.assertEqual(report["prediction_metrics"]["S3"]["fp"], 1)
            self.assertEqual(report["singleton_metrics"]["false_merges"], 1)
            self.assertIn("S2: TP=1", (output / "validation_summary.txt").read_text())
            self.assertTrue((output / "validation_predictions.tsv.gz").is_file())

    def test_score_mismatch_is_rejected_and_summary_is_stable(self) -> None:
        ids = ["S1-A"]
        s2 = make_cache("S2", ids, [[("S2-WRONG", 0.99)]])
        s3 = make_cache("S3", ids, [[]])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            c, m, a = root / "c.tsv.gz", root / "m.tsv.gz", root / "a.tsv.gz"
            with gzip.open(c, "wt", newline="") as f:
                w=csv.writer(f, delimiter="\t"); w.writerow(CANDIDATE_HEADER); w.writerow(["S1-A", "S2-A"])
            with gzip.open(m, "wt", newline="") as f:
                w=csv.writer(f, delimiter="\t"); w.writerow(METADATA_HEADER); w.writerow(["S1-A", "S2-A", "S2", *(["1"]+["0"]*5), "1", "1"])
            with gzip.open(a, "wt", newline="") as f:
                csv.writer(f, delimiter="\t").writerow(ADDRESS_HEADER)
            with self.assertRaises(ValueError):
                validate_scored_membership(ids, a, s2, s3, c, m)
        sample_summary = {
            "validation_s1_count": 1, "candidate_pairs": 1,
            "candidate_metrics": {"overall": {"candidate_recall": 1.0},
                                  "by_source": {"S2": {"candidate_recall": 1.0}, "S3": {"candidate_recall": 0.0}}},
            "prediction_metrics": {"macro_f0_5": 1.0, "tp": 1, "fp": 0, "fn": 0,
                                   "micro_precision_diagnostic": 1.0, "micro_recall_diagnostic": 1.0,
                                   "predicted_mean": 1.0, "predicted_p95": 1.0, "predicted_p99": 1.0,
                                   "predicted_max": 1, "predicted_cardinality_distribution": {"0": 0, "1": 1}, "S2": {"tp": 1, "fp": 0, "fn": 0,
                                   "precision_diagnostic": 1.0, "recall_diagnostic": 1.0},
                                   "S3": {"tp": 0, "fp": 0, "fn": 0, "precision_diagnostic": 0.0,
                                   "recall_diagnostic": 0.0}},
            "blocking_misses": 0, "model_decision_misses": 0,
            "singleton_metrics": {"true": 0, "predicted": 0, "correct": 0, "false_merges": 0,
                                  "accuracy": 0.0},
            "runtime_seconds_evaluation": 1.0, "runtime_seconds_scoring": 2.0,
            "runtime_seconds_address_generation": 3.0, "peak_rss_mb": 100.0,
            "evaluation_peak_rss_mb": 100.0, "disk_bytes_total": 10,
        }
        report_text = format_summary(sample_summary)
        self.assertIn("Macro entity-level F0.5: 1.00000000", report_text)
        self.assertIn("S2: TP=1", report_text)


if __name__ == "__main__":
    unittest.main()
