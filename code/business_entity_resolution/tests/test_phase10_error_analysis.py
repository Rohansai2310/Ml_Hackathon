from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src/phase10_error_analysis.py"
sys.path.insert(0, str(SRC.parent))
import phase10_error_analysis as phase10


def test_truth_failure_stage_boundaries():
    policy = phase10.DecisionPolicy(.93, .97, None, "highest", None)
    assert phase10.classify_truth_link("S2-1", {}, set(), policy) == "A_blocking_miss"
    assert phase10.classify_truth_link("S2-1", {"S2-1": .929999}, set(), policy) == "B1_below_threshold"
    assert phase10.classify_truth_link("S3-1", {"S3-1": .969999}, set(), policy) == "B1_below_threshold"
    assert phase10.classify_truth_link("S2-1", {"S2-1": .93}, set(), policy) == "B2_ownership_conflict"
    assert phase10.classify_truth_link("S3-1", {"S3-1": .97}, {"S3-1"}, policy) == "TP"


def test_false_positive_categories_are_deterministic():
    assert phase10.classify_fp("S2-1", .99, .93, 1, False) == "C1_high_confidence"
    assert phase10.classify_fp("S3-1", .971, .97, 1, False) == "C2_near_threshold"
    assert phase10.classify_fp("S2-1", .96, .93, 1, True) == "C3_target_conflict_related"
    assert phase10.classify_fp("S2-1", .96, .93, 501, False) == "C4_crowded_candidate_set"
    assert phase10.classify_fp("S2-1", .96, .93, 1, False) == "C5_other"


def test_ownership_conflict_highest_owner_and_truth_accounting():
    pre = {"S1-a": {"S2-x": .99}, "S1-b": {"S2-x": .98}, "S1-c": {"S2-y": .97}}
    predictions = {"S1-a": {"S2-x"}, "S1-b": set(), "S1-c": {"S2-y"}}
    truth = {"S1-a": set(), "S1-b": {"S2-x"}, "S1-c": {"S2-y"}}
    summary, removed, contested = phase10.ownership_summary(pre, predictions, truth, truth)
    assert summary["conflicted_targets"] == 1
    assert summary["winner_incorrect_loser_correct"] == 1
    assert summary["tp_removed"] == 1
    assert removed == {("S1-b", "S2-x")}
    assert contested == {"S2-x"}


def test_bins_and_per_entity_f05_use_official_semantics():
    assert phase10._candidate_bin(0) == "0"
    assert phase10._candidate_bin(100) == "26-100"
    assert phase10._candidate_bin(501) == "501+"
    assert phase10._score_bin("S2", .93) == "0.93-0.97"
    assert phase10._score_bin("S3", .97) == "0.97-0.99"
    # The singleton rule is delegated to scoring.score_entity in all production aggregation.
    assert phase10.score_entity(set(), set()) == 1.0
    assert phase10._f05_bin(phase10.score_entity({"S2-x"}, {"S2-y"})) == "0"


def test_finalize_metrics_and_slice_metrics():
    table = {("numeric", "shared"): phase10.Counter({"support": 2, "retrieved_truth": 2, "tp": 1, "fn": 1, "fp": 1})}
    row = phase10._finalize_metrics(table)[0]
    assert row["retrieval_recall"] == 1.0
    assert row["model_policy_recall_given_retrieved"] == .5
    assert row["precision_diagnostic"] == .5


def test_snapshot_detects_frozen_input_mutation(tmp_path, monkeypatch):
    p = tmp_path / "frozen.txt"; p.write_text("one")
    monkeypatch.setattr(phase10, "FROZEN_FILES", (p,))
    before = phase10.snapshot_frozen()
    phase10.assert_unchanged(before)
    p.write_text("two changed")
    with pytest.raises(RuntimeError, match="frozen"):
        phase10.assert_unchanged(before)


def test_phase10_never_mentions_hidden_test_labels_or_retrieval_regeneration():
    source = SRC.read_text(encoding="utf-8")
    assert "test_ground_truth" not in source
    assert "generate-candidates" not in source
    assert "address_retrieve(" not in source


def test_phase101_agreement_buckets_use_phase8_zero_one_scale():
    assert phase10._agreement_bucket(missing=True, exact=True, ratio=1.0) == "missing"
    assert phase10._agreement_bucket(missing=False, exact=True, ratio=.1) == "exact_or_very_high"
    assert phase10._agreement_bucket(missing=False, exact=False, ratio=.95) == "exact_or_very_high"
    assert phase10._agreement_bucket(missing=False, exact=False, ratio=.70) == "medium"
    assert phase10._agreement_bucket(missing=False, exact=False, ratio=.699999) == "weak"
    assert phase10.IMMEDIATE_BELOW == {"S2": "0.90-0.93", "S3": "0.95-0.97"}


def test_phase101_address_rank_cumulatives_are_source_separated():
    truth = {"S1-a": {"S2-1", "S2-2", "S2-3", "S3-1"}, "S1-b": {"S3-2"}}
    ranks = {("S2", "1"): 1, ("S2", "2-3"): 1, ("S2", "4-5"): 1,
             ("S3", "1"): 1, ("S3", "6-10"): 1}
    rows = phase10.address_rank_distribution(ranks, truth)
    s2 = [row for row in rows if row["source"] == "S2"]
    s3 = [row for row in rows if row["source"] == "S3"]
    assert [row["cumulative_truth_hits"] for row in s2] == [1, 2, 3, 3]
    assert s2[-1]["cumulative_share_of_address_ranked_truth_hits"] == 1.0
    assert s3[-1]["cumulative_truth_hits"] == 2
    assert s3[-1]["cumulative_share_of_all_source_truth_links"] == 1.0


def test_phase101_required_name_and_address_buckets_are_declared():
    assert phase10.SLICE_BUCKETS["name_condition"] == ("exact_or_very_high", "medium", "weak", "missing")
    assert phase10.SLICE_BUCKETS["address_condition"] == ("exact_or_very_high", "medium", "weak", "missing")
