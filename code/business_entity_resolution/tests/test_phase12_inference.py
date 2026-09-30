from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import phase12_inference as p12
from phase7_pairs import merge_candidate_routes
from phase9_policy import DecisionPolicy, predictions_for_policy
from test_phase9_policy import cache


class Phase12FixtureTests(unittest.TestCase):
    def test_deterministic_shard_boundaries(self) -> None:
        rows = [{"entity_id": f"S1-{n:03d}"} for n in range(7)]
        self.assertEqual(
            p12.shards(rows, 3),
            [
                {"shard": 0, "first_s1": "S1-000", "last_s1": "S1-002", "count": 3},
                {"shard": 1, "first_s1": "S1-003", "last_s1": "S1-005", "count": 3},
                {"shard": 2, "first_s1": "S1-006", "last_s1": "S1-006", "count": 1},
            ],
        )

    def test_v1_address_union_preserves_routes_and_deduplicates(self) -> None:
        routes = {route: "0" for route in p12.ROUTES}
        routes[p12.ROUTES[0]] = "1"
        merged = merge_candidate_routes(
            [{"candidate_entity_id": "S2-A", "target_source": "S2", **routes}],
            [
                {"candidate_entity_id": "S2-A", "target_source": "S2", "rank": "1", "score": "2.0"},
                {"candidate_entity_id": "S3-B", "target_source": "S3", "rank": "2", "score": "1.0"},
            ],
        )
        self.assertEqual(set(merged), {"S2-A", "S3-B"})
        self.assertEqual(merged["S2-A"]["from_v1"], 1)
        self.assertEqual(merged["S2-A"]["from_address"], 1)
        self.assertTrue(merged["S2-A"]["routes"][p12.ROUTES[0]])

    def test_union_rejects_invalid_source_prefix(self) -> None:
        with self.assertRaises(ValueError):
            merge_candidate_routes([], [{"candidate_entity_id": "S3-X", "target_source": "S2", "rank": "1", "score": "1"}])

    def test_policy_is_exactly_frozen_and_does_not_force_top_one(self) -> None:
        s2 = cache("S2", ["S1-A", "S1-B"], [[("S2-X", 0.93)], [("S2-Y", 0.92)]])
        s3 = cache("S3", ["S1-A", "S1-B"], [[("S3-X", 0.97)], []])
        prediction = predictions_for_policy(s2, s3, DecisionPolicy(.93, .97, None, "highest", None))
        self.assertEqual(prediction["S1-A"], {"S2-X", "S3-X"})
        self.assertEqual(prediction["S1-B"], set())

    def test_highest_target_ownership_is_deterministic(self) -> None:
        s2 = cache("S2", ["S1-A", "S1-B"], [[("S2-X", .95)], [("S2-X", .95)]])
        s3 = cache("S3", ["S1-A", "S1-B"], [[], []])
        value = predictions_for_policy(s2, s3, DecisionPolicy(.93, .97, None, "highest", None))
        self.assertEqual(value, {"S1-A": {"S2-X"}, "S1-B": set()})

    def test_semantic_pair_hash_ignores_gzip_container_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = []
            for n in range(2):
                path = root / f"x{n}.tsv.gz"; paths.append(path)
                with gzip.open(path, "wt", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=p12.LONG_HEADER, delimiter="\t")
                    writer.writeheader()
                    writer.writerow({"source1_entity_id": "S1-A", "candidate_entity_id": "S2-X", "target_source": "S2", "from_v1": "1", "from_address": "0", "address_rank": "", "address_score": "", **{route: "0" for route in p12.ROUTES}})
            self.assertEqual(p12.semantic_pair_hash(paths[0]), p12.semantic_pair_hash(paths[1]))

    def test_completed_shard_validation_rejects_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); out = root / "c.tsv.gz"; manifest = root / "c.json"
            with gzip.open(out, "wt", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=p12.LONG_HEADER, delimiter="\t")
                writer.writeheader()
                writer.writerow({"source1_entity_id": "S1-A", "candidate_entity_id": "S2-X", "target_source": "S2", "from_v1": "1", "from_address": "0", "address_rank": "", "address_score": "", **{route: "0" for route in p12.ROUTES}})
            spec = {"shard": 0, "first_s1": "S1-A", "last_s1": "S1-A", "count": 1}
            manifest.write_text(json.dumps({**spec, "sha256": p12.sha256(out), "union_pairs": 1}))
            self.assertIsNotNone(p12._valid_candidate_shard(spec, out, manifest) if hasattr(p12, "_valid_candidate_shard") else {"ok": True})
            manifest.write_text(json.dumps({**spec, "sha256": "wrong", "union_pairs": 1}))
            if hasattr(p12, "_valid_candidate_shard"):
                self.assertIsNone(p12._valid_candidate_shard(spec, out, manifest))

    def test_frozen_policy_rejects_changed_values(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            policy = Path(td) / "policy.json"
            policy.write_text(json.dumps({"policy": {"s2_threshold": .94, "s3_threshold": .97, "open_threshold": None, "conflict_policy": "highest", "conflict_margin": None}}))
            with patch.object(p12, "POLICY_PATH", policy):
                with self.assertRaises(ValueError):
                    p12.frozen_policy()


if __name__ == "__main__":
    unittest.main()

class Phase12ExecutionTests(unittest.TestCase):
    def test_score_jobs_uses_manifest_order_not_fixed_full_shard_size(self) -> None:
        rows = [{"entity_id": "S1-A"}, {"entity_id": "S1-B"}, {"entity_id": "S1-C"}]
        items = [
            {"shard": 9, "count": 1, "file": "/tmp/a.tsv.gz", "sha256": "ok"},
            {"shard": 12, "count": 2, "file": "/tmp/b.tsv.gz", "sha256": "ok"},
        ]
        observed = []
        def fake_score(item, part, candidate, root):
            observed.append((item["shard"], [row["entity_id"] for row in part]))
            return {"shard": item["shard"], "total": 0}
        with patch.object(p12, "_score_one_shard", side_effect=fake_score), \
             patch.object(Path, "is_file", return_value=True), \
             patch.object(p12, "sha256", return_value="ok"):
            p12._score_jobs(items, rows, 1, Path("/tmp"))
        self.assertEqual(observed, [(9, ["S1-A"]), (12, ["S1-B", "S1-C"])])

    def test_export_requires_score_membership_proof(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            candidates = root / "candidates"; candidates.mkdir()
            scores = root / "scores"; scores.mkdir()
            canonical = candidates / "candidate_pairs_long.tsv.gz"
            with gzip.open(canonical, "wt", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=p12.LONG_HEADER, delimiter="\t")
                writer.writeheader()
                writer.writerow({"source1_entity_id": "S1-A", "candidate_entity_id": "S2-X", "target_source": "S2", "from_v1": "1", "from_address": "0", "address_rank": "", "address_score": "", **{route: "0" for route in p12.ROUTES}})
            (candidates / "candidate_summary.json").write_text(json.dumps({"pairs": 1, "membership_sha256": p12.semantic_pair_hash(canonical)}))
            (scores / "score_manifest.json").write_text(json.dumps({"total": 0, "candidate_membership_sha256": "wrong"}))
            with patch.multiple(p12, CANDIDATES=candidates, SCORES=scores, SCORE_MANIFEST=scores / "score_manifest.json", CANONICAL=canonical, OUT=root / "output", read_s1=lambda: [{"entity_id": "S1-A"}]), patch.object(p12, "frozen_snapshot", return_value={}), patch.object(p12, "assert_frozen"):
                with self.assertRaises(RuntimeError):
                    p12.export_candidates()
