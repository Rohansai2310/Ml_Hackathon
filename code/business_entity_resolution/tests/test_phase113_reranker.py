import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import phase113_reranker as r
from phase9_policy import ScoreCache


def cache(source, s1, targets, scores):
    offsets = np.asarray([0, len(targets)], dtype=np.int64)
    return ScoreCache(source, (s1,), offsets, np.asarray(targets, dtype="S20"),
                      np.asarray(scores, dtype=np.float32), np.ones(len(targets), dtype=np.uint8),
                      np.zeros(len(targets), dtype=np.uint8), np.zeros(len(targets), dtype=np.int16),
                      np.zeros(len(targets), dtype=np.float32))


def test_fold_assignment_is_deterministic_and_group_local():
    ids = [f"S1-{i}" for i in range(100)]
    first = [r.fold_of(x) for x in ids]
    assert first == [r.fold_of(x) for x in ids]
    assert set(first) <= set(range(5))
    for held_out in range(5):
        training = {x for x in ids if r.fold_of(x) != held_out}
        scored = {x for x in ids if r.fold_of(x) == held_out}
        assert not training & scored


def test_contextual_features_have_rank_gap_density_and_no_identity_feature():
    evidence = [{"from_v1": 1, "from_address": 1, "address_rank": 1},
                {"from_v1": 0, "from_address": 1, "address_rank": 2}]
    x2, x3 = r.contextual_group(np.asarray([.95, .80], np.float32), np.asarray([.97], np.float32), ["S2-z", "S2-a"], ["S3-y"], evidence, evidence[:1])
    ix = {v: i for i, v in enumerate(r.CONTEXT_NAMES)}
    assert x2.shape[1] == len(r.CONTEXT_NAMES)
    assert x2[0, ix["rank_source"]] == 1
    assert x2[1, ix["rank_source"]] == 2
    assert x2[0, ix["top_minus_second_source"]] == pytest.approx(.15, abs=1e-6)
    assert x2[0, ix["count_ge_090"]] == 1
    assert x3[0, ix["source_s3"]] == 1
    assert "source1_entity_id" not in r.CONTEXT_NAMES
    assert "candidate_entity_id" not in r.CONTEXT_NAMES


def test_a_source_policy_context_matches_locked_offsets():
    assert r.a_source_threshold("S2", 100) == pytest.approx(.91)
    assert r.a_source_threshold("S3", 100) == pytest.approx(.955)
    assert r.a_source_threshold("S2", 101) == pytest.approx(.94)
    assert r.a_source_threshold("S3", 101) == pytest.approx(.985)


def test_sampling_keeps_all_positives_is_deterministic_and_bounded():
    labels = np.asarray([1, 0, 0, 0, 1, 0, 0], dtype=np.int8)
    scores = np.asarray([.98, .96, .82, .60, .20, .40, .91], dtype=np.float32)
    ranks = np.arange(1, 8)
    candidates = [f"S2-{i}" for i in range(7)]
    one = r.sampling_indices(labels, scores, ranks, True, "S1-x", candidates)
    two = r.sampling_indices(labels, scores, ranks, True, "S1-x", candidates)
    assert np.array_equal(one, two)
    assert {0, 4} <= set(one.tolist())
    assert len(one) <= int(labels.sum()) + 20


def test_highest_target_ownership_is_preserved():
    s2 = cache("S2", "S1-a", ["S2-x"], [.95])
    s3 = cache("S3", "S1-a", ["S3-y"], [.98])
    prediction = r._predictions(s2, s3, .9, .9)
    assert prediction["S1-a"] == {"S2-x", "S3-y"}


def test_reranker_feature_schema_excludes_labels_and_negative_reason():
    forbidden = ("label", "truth", "negative_reason", "candidate_entity_id", "source1_entity_id")
    assert all(token not in r.CONTEXT_NAMES for token in forbidden)


def test_validation_gate_rejects_missing_lock(tmp_path):
    with pytest.raises(RuntimeError, match="locked winner"):
        r.confirm_validation(tmp_path)


def test_count_bins_are_deterministic():
    assert [r.count_bin(x) for x in (1, 25, 26, 100, 101, 500, 501)] == ["1-25", "1-25", "26-100", "26-100", "101-500", "101-500", "501+"]


def test_sampling_never_truncates_positives_even_above_negative_cap():
    labels = np.asarray([1] * 25 + [0] * 40, dtype=np.int8)
    scores = np.linspace(.99, .01, len(labels), dtype=np.float32)
    chosen = r.sampling_indices(labels, scores, np.arange(1, len(labels) + 1), True, "S1-many", [f"S2-{i:03}" for i in range(len(labels))])
    assert set(range(25)) <= set(chosen.tolist())
    assert len(chosen) <= 25 + 20


def test_validation_threshold_grid_requires_locked_pair():
    with pytest.raises(RuntimeError, match="tune-locked thresholds"):
        r._evaluate_variant(Path("/not-used"), "R1a", split="validation")


def test_sparse_source_group_alignment_inserts_empty_source_groups(tmp_path):
    import csv
    import gzip
    from pathlib import Path

    s2, s3 = tmp_path / "s2.tsv.gz", tmp_path / "s3.tsv.gz"
    header = ["source1_entity_id", "candidate_entity_id", "target_source", "base_score", "from_v1", "from_address", "address_rank", "address_score"]
    for path, rows in ((s2, [("S1-a", "S2-x", "S2", ".9", "1", "0", "", "")]),
                       (s3, [("S1-b", "S3-y", "S3", ".8", "0", "1", "1", ".7")])):
        with gzip.open(path, "wt", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(header)
            writer.writerows(rows)
    got = list(r._aligned_source_groups(s2, s3))
    assert got == [("S1-a", [dict(zip(header, ["S1-a", "S2-x", "S2", ".9", "1", "0", "", ""]))], []),
                   ("S1-b", [], [dict(zip(header, ["S1-b", "S3-y", "S3", ".8", "0", "1", "1", ".7"]))])]
