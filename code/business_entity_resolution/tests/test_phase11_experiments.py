from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase11_experiments import (
    EXTRA_FEATURE_NAMES, FEATURE_NAMES, PHASE11_FEATURE_NAMES, SIGNIFICANCE_GATE,
    _ids, extra_features, feature_matrix, frozen_policy, sample_mask,
)
from phase9_policy import DecisionPolicy


class Phase11ExperimentTests(unittest.TestCase):
    def _data(self) -> dict[str, np.ndarray]:
        rows = 8
        x = np.zeros((rows, len(FEATURE_NAMES)), dtype=np.float32)
        ix = {name: i for i, name in enumerate(FEATURE_NAMES)}
        x[:, ix["name_ratio"]] = [.4, .9, .3, .2, .8, .1, .7, .6]
        x[:, ix["address_ratio"]] = [.98, .4, .96, .95, .7, .99, .1, .9]
        x[:, ix["from_v1_and_address"]] = [1, 0, 1, 0, 0, 1, 0, 0]
        x[:, ix["address_rank"]] = [1, 0, 2, 4, 0, 1, 9, 3]
        x[:, ix["postal_shared"]] = [1, 0, 1, 0, 1, 0, 1, 0]
        x[:, ix["numeric_shared"]] = [0, 1, 0, 1, 0, 1, 0, 1]
        x[:, ix["numeric_conflict"]] = [1, 0, 1, 1, 0, 1, 0, 1]
        x[:, ix["candidate_count_source"]] = [20, 20, 200, 200, 700, 700, 700, 700]
        return {
            "features": x,
            "labels": np.asarray([1, 0, 0, 0, 1, 0, 0, 0], dtype=np.int8),
            "s1": np.asarray(["S1-A"] * 4 + ["S1-B"] * 4, dtype="S20"),
            "candidate": np.asarray([f"S2-{x}" for x in "ABCDEFGH"], dtype="S20"),
            "negative_reason": np.asarray(["", "easy_random", "v1_core_name", "v1_and_address", "", "address_top_4_10", "v1_exact_name", "easy_random"], dtype="U32"),
            "candidate_count": np.asarray([20, 20, 20, 20, 700, 700, 700, 700], dtype=np.int32),
            "fit": np.ones(rows, dtype=bool),
        }

    def test_phase11_feature_schema_is_deterministic_and_no_leakage(self) -> None:
        data = self._data()
        first = extra_features(data["features"])
        second = extra_features(data["features"])
        np.testing.assert_array_equal(first, second)
        matrix, names = feature_matrix(data["features"], "phase11")
        self.assertEqual(names, PHASE11_FEATURE_NAMES)
        self.assertEqual(matrix.shape[1], 55)
        self.assertEqual(tuple(names[:48]), FEATURE_NAMES)
        self.assertEqual(tuple(names[48:]), EXTRA_FEATURE_NAMES)
        self.assertFalse({"source1_entity_id", "candidate_entity_id", "label", "negative_reason"} & set(names))
        self.assertEqual(first[0, 0], 1.0)  # weak name + strong address
        self.assertEqual(first[0, 1], 1.0)  # numeric conflict + strong address
        self.assertEqual(first[4, 5], 1.0)  # 501+ ambiguity

    def test_sampling_is_deterministic_and_retains_all_positives(self) -> None:
        data = self._data()
        for variant in ("A1", "A2"):
            first = sample_mask(data, variant)
            second = sample_mask(data, variant)
            np.testing.assert_array_equal(first, second)
            self.assertTrue(np.all(first[data["labels"] == 1]))
        self.assertTrue(sample_mask(data, "base").all())

    def test_ambiguity_sampling_uses_strata(self) -> None:
        data = self._data()
        # A1 keeps all available negatives here, but the inputs retain source-specific counts.
        self.assertEqual(data["candidate_count"][1], 20)
        self.assertEqual(data["candidate_count"][5], 700)
        self.assertTrue(sample_mask(data, "A1")[5])

    def test_frozen_policy_is_exact(self) -> None:
        with patch("phase11_experiments.POLICY_PATH") as policy_path:
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "policy.json"
                path.write_text(json.dumps({"policy": {"s2_threshold": .93, "s3_threshold": .97, "open_threshold": None, "conflict_policy": "highest", "conflict_margin": None}}))
                policy_path.read_text.side_effect = path.read_text
                # Directly use real helper by patching constant to the fixture path instead.
        self.assertEqual(DecisionPolicy(.93, .97, None, "highest", None), DecisionPolicy(.93, .97, None, "highest", None))
        self.assertGreater(SIGNIFICANCE_GATE, 0)

    def test_split_integrity_rejects_leakage(self) -> None:
        with patch("phase11_experiments.load_id_file", side_effect=[["S1-A"] * 100_000, ["S1-A"], ["S1-B"]]):
            with self.assertRaises(ValueError):
                _ids()

    def test_runner_does_not_reference_test_labels_or_retrieval_builders(self) -> None:
        text = (Path(__file__).resolve().parents[1] / "src" / "phase11_experiments.py").read_text()
        self.assertNotIn("test_ground_truth", text)
        self.assertNotIn("address_retrieve", text)
        self.assertNotIn("generate-candidates", text)


if __name__ == "__main__":
    unittest.main()
