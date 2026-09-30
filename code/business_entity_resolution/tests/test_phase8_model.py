from __future__ import annotations

import csv
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from baseline import make_record
from blocking import METADATA_HEADER
from diagnostics import CANDIDATE_HEADER
from phase8_model import (
    ADDRESS_HEADER,
    FEATURE_NAMES,
    SCORE_HEADER,
    extract_features,
    feature_manifest,
    internal_s1_split,
    finalize_summary,
    iter_union_groups,
    validate_feature_names,
)


class Phase8FeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.left = make_record("S1-A", "Acme & Sons Pvt Ltd", "12 Cedar Rd, 02108", "US")
        self.right = make_record("S2-A", "ACME and Sons Limited", "12 Cedar Road, 02108", "US")
        self.evidence = {"from_v1": "1", "from_address": "1", "address_rank": "1", "address_score": "4.25"}

    def test_features_are_deterministic_finite_and_schema_ordered(self) -> None:
        first = extract_features(self.left, self.right, self.evidence, 7)
        second = extract_features(self.left, self.right, self.evidence, 7)
        self.assertEqual(first.dtype, np.float32)
        self.assertEqual(len(first), len(FEATURE_NAMES))
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_array_equal(first, second)

    def test_numeric_postal_and_retrieval_features(self) -> None:
        values = dict(zip(FEATURE_NAMES, extract_features(self.left, self.right, self.evidence, 7)))
        self.assertEqual(values["name_core_exact"], 1.0)
        self.assertEqual(values["numeric_shared"], 1.0)
        self.assertEqual(values["numeric_conflict"], 0.0)
        self.assertEqual(values["postal_shared"], 1.0)
        self.assertEqual(values["from_v1_and_address"], 1.0)
        self.assertEqual(values["address_top_candidate"], 1.0)
        self.assertEqual(values["candidate_count_source"], 7.0)

    def test_conflicts_and_missing_values_are_safe(self) -> None:
        other = make_record("S3-X", "Other", "92 River Ave, 90210", "Atlantis")
        values = dict(zip(FEATURE_NAMES, extract_features(
            make_record("S1-B", "", "", ""), other,
            {"from_v1": 0, "from_address": 0, "address_rank": "", "address_score": ""}, 0,
        )))
        self.assertEqual(values["name_missing_left"], 1.0)
        self.assertEqual(values["address_missing_left"], 1.0)
        self.assertEqual(values["address_rank_missing"], 1.0)
        self.assertTrue(all(np.isfinite(v) for v in values.values()))

    def test_manifest_and_leakage_guard(self) -> None:
        manifest = feature_manifest()
        self.assertEqual(tuple(manifest["feature_order"]), FEATURE_NAMES)
        self.assertNotIn("label", FEATURE_NAMES)
        self.assertNotIn("negative_reason", FEATURE_NAMES)
        self.assertNotIn("source1_entity_id", FEATURE_NAMES)
        with self.assertRaises(ValueError):
            validate_feature_names(tuple(FEATURE_NAMES) + ("label",))

    def test_internal_split_is_deterministic_and_complete(self) -> None:
        ids = [f"S1-{i:04}" for i in range(25)]
        fit_a, select_a = internal_s1_split(ids)
        fit_b, select_b = internal_s1_split(ids)
        self.assertEqual((fit_a, select_a), (fit_b, select_b))
        self.assertFalse(fit_a & select_a)
        self.assertEqual(fit_a | select_a, set(ids))
        self.assertEqual(len(fit_a), 20)

    def test_model_reload_preserves_scores_and_feature_order(self) -> None:
        matrix = np.vstack([
            extract_features(self.left, self.right, self.evidence, 3),
            extract_features(self.left, make_record("S2-Z", "Different", "92 Road", "US"), self.evidence, 3),
            extract_features(self.right, self.left, self.evidence, 3),
            extract_features(self.right, make_record("S2-Y", "Else", "4 Street", "US"), self.evidence, 3),
        ])
        labels = np.array([1, 0, 1, 0])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.txt"
            train = lgb.Dataset(matrix, label=labels, feature_name=list(FEATURE_NAMES))
            model = lgb.train({"objective": "binary", "verbosity": -1, "seed": 2, "num_threads": 1}, train, num_boost_round=3)
            before = model.predict(matrix)
            model.save_model(str(path))
            restored = lgb.Booster(model_file=str(path))
            np.testing.assert_allclose(before, restored.predict(matrix), rtol=0, atol=0)
            self.assertEqual(tuple(restored.feature_name()), FEATURE_NAMES)

    def test_frozen_union_merges_routes_and_keeps_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            v1, metadata, address = root / "v1.tsv.gz", root / "metadata.tsv.gz", root / "address.tsv.gz"
            with gzip.open(v1, "wt", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t"); writer.writerow(CANDIDATE_HEADER)
                writer.writerow(["S1-A", "S2-A"]); writer.writerow(["S1-B", ""])
            with gzip.open(metadata, "wt", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t"); writer.writerow(METADATA_HEADER)
                writer.writerow(["S1-A", "S2-A", "S2", "1", "0", "0", "0", "0", "0", "0", "1"])
            with gzip.open(address, "wt", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t"); writer.writerow(ADDRESS_HEADER)
                writer.writerow(["S1-A", "S2-A", "S2", "address", "5.2", "2"])
                writer.writerow(["S1-A", "S3-B", "S3", "address", "4.1", "1"])
            groups = list(iter_union_groups(["S1-A", "S1-B"], v1, metadata, address))
            self.assertEqual(list(groups[0][1]), ["S2-A", "S3-B"])
            self.assertEqual(groups[0][1]["S2-A"]["from_v1"], 1)
            self.assertEqual(groups[0][1]["S2-A"]["from_address"], 1)
            self.assertEqual(groups[0][1]["S3-B"]["address_rank"], "1")
            self.assertEqual(groups[1][1], {})
            prefix = list(iter_union_groups(
                ["S1-A"], v1, metadata, address, require_exhausted=False))
            self.assertEqual(list(prefix[0][1]), ["S2-A", "S3-B"])
            with self.assertRaisesRegex(ValueError, "extra S1 rows"):
                list(iter_union_groups(["S1-A"], v1, metadata, address))

    def test_finalize_uses_model_selection_report_after_retraining(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "feature_manifest.json").write_text(json.dumps({"feature_count": len(FEATURE_NAMES)}))
            selected = {"configuration": {"name": "wide_95"}, "best_iteration": 12}
            for source in ("s2", "s3"):
                (root / f"training_report_{source}.json").write_text(json.dumps({
                    "model_path": str(root / f"model_{source}.txt"), "model_bytes": 10,
                    "selected_configuration": {"name": "wide_95"},
                }))
                (root / f"model_selection_{source}.json").write_text(json.dumps({"selected": selected}))
                with gzip.open(root / f"tune_scores_{source}.tsv.gz", "wt") as handle:
                    handle.write("source1_entity_id\tcandidate_entity_id\tscore\n")
            from unittest.mock import patch
            with patch("phase8_model.frozen_snapshot", return_value={}):
                summary = finalize_summary(root)
            self.assertEqual(summary["models"]["S2"]["selected"], selected)
            self.assertEqual(summary["models"]["S3"]["selected"], selected)


if __name__ == "__main__":
    unittest.main()
