#!/usr/bin/env python3
"""Validate training ground-truth coverage and ID-list integrity."""

from __future__ import annotations

import argparse
import csv
import sqlite3
import tempfile
from pathlib import Path


EXPECTED_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def _valid_id(value: str, prefix: str) -> bool:
    return value.startswith(prefix) and bool(value[len(prefix):]) and not any(
        char.isspace() for char in value
    )


def validate_ground_truth(dataset_root: Path) -> dict[str, int]:
    """Check the ground truth against train_source1 using a disk-backed index."""
    train_dir = dataset_root / "train"
    source_path = train_dir / "train_source1.tsv"
    truth_path = train_dir / "train_ground_truth.tsv"
    with tempfile.TemporaryDirectory(prefix="ber_gt_check_") as temp_dir:
        db = sqlite3.connect(Path(temp_dir) / "ids.sqlite")
        db.execute("CREATE TABLE source1 (id TEXT PRIMARY KEY)")
        db.execute("CREATE TABLE truth (id TEXT PRIMARY KEY)")
        source_rows = 0
        with source_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            header = next(reader, None)
            if header != ["entity_id", "business_name", "business_address", "country"]:
                raise ValueError(f"Unexpected Source 1 columns: {header!r}")
            for row_num, row in enumerate(reader, start=2):
                source_rows += 1
                entity_id = row[0] if row else ""
                if not _valid_id(entity_id, "S1-"):
                    raise ValueError(f"Malformed S1 ID at {source_path}:{row_num}: {entity_id!r}")
                try:
                    db.execute("INSERT INTO source1 VALUES (?)", (entity_id,))
                except sqlite3.IntegrityError as exc:
                    raise ValueError(f"Duplicate S1 ID in source file: {entity_id}") from exc
                if source_rows % 100_000 == 0:
                    db.commit()
        db.commit()

        truth_rows = 0
        link_count = 0
        singleton_rows = 0
        with truth_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            header = next(reader, None)
            if header != EXPECTED_COLUMNS:
                raise ValueError(f"Unexpected ground-truth columns: {header!r}; expected {EXPECTED_COLUMNS!r}")
            for row_num, row in enumerate(reader, start=2):
                if len(row) != 2:
                    raise ValueError(f"Malformed ground-truth row at {truth_path}:{row_num}")
                s1_id, raw_ids = row
                truth_rows += 1
                if not _valid_id(s1_id, "S1-"):
                    raise ValueError(f"Malformed/blank source1_entity_id at {truth_path}:{row_num}: {s1_id!r}")
                try:
                    db.execute("INSERT INTO truth VALUES (?)", (s1_id,))
                except sqlite3.IntegrityError as exc:
                    raise ValueError(f"Duplicate source1_entity_id in ground truth: {s1_id}") from exc
                if db.execute("SELECT 1 FROM source1 WHERE id = ?", (s1_id,)).fetchone() is None:
                    raise ValueError(f"Ground truth references unknown training S1 ID: {s1_id}")
                if raw_ids == "":
                    singleton_rows += 1
                else:
                    ids = raw_ids.split(",")
                    if any(not target_id for target_id in ids):
                        raise ValueError(f"Blank ID inside match list at {truth_path}:{row_num}")
                    if len(ids) != len(set(ids)):
                        raise ValueError(f"Duplicate ID inside match list at {truth_path}:{row_num} ({s1_id})")
                    for target_id in ids:
                        if not (target_id.startswith("S2-") or target_id.startswith("S3-")) or not _valid_id(target_id, target_id[:3]):
                            raise ValueError(f"Invalid matched ID at {truth_path}:{row_num}: {target_id!r}")
                        link_count += 1
                if truth_rows % 100_000 == 0:
                    db.commit()
        db.commit()

        missing = db.execute(
            "SELECT COUNT(*) FROM source1 s LEFT JOIN truth t ON t.id=s.id WHERE t.id IS NULL"
        ).fetchone()[0]
        extras = db.execute(
            "SELECT COUNT(*) FROM truth t LEFT JOIN source1 s ON s.id=t.id WHERE s.id IS NULL"
        ).fetchone()[0]
        db.close()
        if source_rows != truth_rows or missing or extras:
            raise ValueError(
                "Ground-truth S1 coverage mismatch: "
                f"source1_rows={source_rows}, ground_truth_rows={truth_rows}, "
                f"missing_ground_truth={missing}, unknown_ground_truth={extras}"
            )
        return {
            "source1_rows": source_rows,
            "ground_truth_rows": truth_rows,
            "true_links": link_count,
            "singletons": singleton_rows,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path,
        default=Path(__file__).resolve().parents[3]
        / "6ab10eb3b23ba_student_resource/student_resource/dataset",
    )
    args = parser.parse_args()
    try:
        result = validate_ground_truth(args.dataset_root)
    except (OSError, ValueError, csv.Error) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(
        "PASS ground truth: "
        f"S1 rows={result['source1_rows']:,}; "
        f"GT rows={result['ground_truth_rows']:,}; "
        f"links={result['true_links']:,}; "
        f"singletons={result['singletons']:,}; coverage exact; IDs/prefixes valid"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
