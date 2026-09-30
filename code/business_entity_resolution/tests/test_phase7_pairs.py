from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from phase7_pairs import merge_candidate_routes, select_pair_rows, select_training_subset, sha256_ids


def test_training_subset_is_deterministic_and_excludes_reserved_ids():
    train = [f"S1-{i:05}" for i in range(1000)]
    tune = train[100:150]
    validation = train[900:]
    first = select_training_subset(train, tune, validation, size=100, seed=20260925)
    second = select_training_subset(train, tune, validation, size=100, seed=20260925)
    assert first == second
    assert len(first) == len(set(first)) == 100
    assert not (set(first) & set(tune))
    assert not (set(first) & set(validation))
    assert sha256_ids(first) == sha256_ids(second)


def test_training_subset_rejects_overlap_and_duplicates():
    with pytest.raises(ValueError, match="overlap"):
        select_training_subset(["S1-a", "S1-b", "S1-c"], ["S1-a"], ["S1-a"], 1)
    with pytest.raises(ValueError, match="duplicates"):
        select_training_subset(["S1-a", "S1-a", "S1-b"], [], [], 1)


def _pair(target, source, *, v1=0, address=0, rank="", exact=False, core=False, token=False):
    return {
        "candidate_entity_id": target,
        "target_source": source,
        "from_v1": v1,
        "from_address": address,
        "address_rank": rank,
        "address_score": "7.25" if address else "",
        "routes": {"exact_name": exact, "core_name": core, "rare_name_token": token},
    }


def test_labels_keep_all_positives_and_never_emit_blocking_misses_as_negatives():
    ids = ["S1-a", "S1-b"]
    candidates = {
        "S1-a": [_pair("S2-1", "S2", v1=1), _pair("S2-2", "S2", address=1, rank="1"), _pair("S3-1", "S3", v1=1)],
        "S1-b": [_pair("S3-2", "S3", address=1, rank="2")],
    }
    truth = {"S1-a": {"S2-1", "S3-1", "S3-missed"}, "S1-b": {"S3-2"}}
    selected, misses, _ = select_pair_rows(ids, candidates, truth, max_negatives=12)
    positives = {row["candidate_entity_id"] for rows in selected.values() for row in rows if row["label"] == 1}
    negatives = {row["candidate_entity_id"] for rows in selected.values() for row in rows if row["label"] == 0}
    assert positives == {"S2-1", "S3-1", "S3-2"}
    assert "S3-missed" not in negatives
    assert misses["S3"] == [{"source1_entity_id": "S1-a", "true_target_id": "S3-missed", "target_source": "S3"}]
    assert all(row["negative_reason"] for rows in selected.values() for row in rows if row["label"] == 0)


def test_hard_negative_cap_prioritizes_address_and_keeps_a_deterministic_easy_sample():
    candidates = []
    for i in range(20):
        candidates.append(_pair(f"S2-{i:02}", "S2", address=1, rank=str(i + 1)))
    candidates += [_pair("S2-easy-a", "S2", v1=1, token=True), _pair("S2-easy-b", "S2", v1=1, token=True)]
    selected1, _, _ = select_pair_rows(["S1-a"], {"S1-a": candidates}, {"S1-a": set()}, seed=12, max_negatives=8)
    selected2, _, _ = select_pair_rows(["S1-a"], {"S1-a": candidates}, {"S1-a": set()}, seed=12, max_negatives=8)
    rows1 = selected1["S2"]
    rows2 = selected2["S2"]
    assert len(rows1) == 8
    assert [(r["candidate_entity_id"], r["negative_reason"]) for r in rows1] == [(r["candidate_entity_id"], r["negative_reason"]) for r in rows2]
    assert sum(r["negative_reason"] == "easy_random" for r in rows1) == 1
    assert sum(int(r["address_rank"] or 99) <= 3 for r in rows1) == 3


def test_pair_source_separation_duplicate_rejection_and_route_categories():
    candidates = {"S1-a": [
        _pair("S2-1", "S2", v1=1, exact=True),
        _pair("S3-1", "S3", address=1, rank="1"),
    ]}
    truth = {"S1-a": {"S2-1"}}
    selected, misses, _ = select_pair_rows(["S1-a"], candidates, truth)
    assert len(selected["S2"]) == 1 and selected["S2"][0]["label"] == 1
    assert len(selected["S3"]) == 1 and selected["S3"][0]["label"] == 0
    assert selected["S3"][0]["negative_reason"] == "address_top_1_3"
    assert misses == {"S2": [], "S3": []}
    with pytest.raises(ValueError, match="duplicate candidate pair"):
        select_pair_rows(["S1-a"], {"S1-a": candidates["S1-a"] * 2}, truth)
    with pytest.raises(ValueError, match="prefix mismatch"):
        select_pair_rows(["S1-a"], {"S1-a": [_pair("S3-9", "S2")]}, truth)


def test_v1_union_address_preserves_v1_and_merges_metadata():
    v1 = [{
        "candidate_entity_id": "S2-v1", "target_source": "S2",
        "exact_name": "1", "core_name": "0", "rare_name_token": "0",
        "address_numeric_route": "0", "postal_route": "0", "strong_address_route": "0",
        "shared_informative_token_count": "0", "num_blocking_routes": "1",
    }]
    address = [
        {"candidate_entity_id": "S2-v1", "target_source": "S2", "rank": "2", "score": "6.75"},
        {"candidate_entity_id": "S3-address", "target_source": "S3", "rank": "1", "score": "9.25"},
    ]
    merged = merge_candidate_routes(v1, address)
    assert set(merged) == {"S2-v1", "S3-address"}
    assert merged["S2-v1"]["from_v1"] == merged["S2-v1"]["from_address"] == 1
    assert merged["S2-v1"]["routes"]["exact_name"] is True
    assert merged["S2-v1"]["address_rank"] == "2"
    assert merged["S2-v1"]["address_score"] == "6.75"
    assert merged["S3-address"]["from_v1"] == 0 and merged["S3-address"]["from_address"] == 1
