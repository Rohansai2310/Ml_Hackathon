#!/usr/bin/env python3
"""Create a deterministic source-1 entity-level train/validation split."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from check_ground_truth import validate_ground_truth


def load_s1_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = ["entity_id", "business_name", "business_address", "country"]
        if reader.fieldnames != expected:
            raise ValueError(f"Unexpected Source 1 columns in {path}: {reader.fieldnames!r}")
        for row in reader:
            ids.append(row["entity_id"])
    if len(ids) != len(set(ids)):
        raise ValueError(f"Source 1 contains duplicate entity IDs: {path}")
    if any(not value.startswith("S1-") or len(value) <= 3 for value in ids):
        raise ValueError(f"Source 1 contains blank or malformed S1 IDs: {path}")
    return ids


def create_split(ids: list[str], train_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("--train-fraction must be strictly between 0 and 1")
    ordered = sorted(ids)
    permutation = np.random.default_rng(seed).permutation(len(ordered))
    train_size = int(train_fraction * len(ordered))
    train_ids = sorted(ordered[int(i)] for i in permutation[:train_size])
    val_ids = sorted(ordered[int(i)] for i in permutation[train_size:])
    if len(train_ids) != len(set(train_ids)) or len(val_ids) != len(set(val_ids)):
        raise AssertionError("Split contains duplicate S1 IDs")
    if set(train_ids).intersection(val_ids):
        raise AssertionError("Train/validation split overlaps")
    if sorted(train_ids + val_ids) != ordered:
        raise AssertionError("Train/validation union differs from the S1 universe")
    return train_ids, val_ids


def write_ids(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.writelines(entity_id + "\n" for entity_id in ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path,
        default=Path(__file__).resolve().parents[3]
        / "6ab10eb3b23ba_student_resource/student_resource/dataset",
    )
    parser.add_argument("--train-fraction", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts/splits",
    )
    args = parser.parse_args()
    integrity = validate_ground_truth(args.dataset_root)
    ids = load_s1_ids(args.dataset_root / "train" / "train_source1.tsv")
    if integrity["source1_rows"] != len(ids):
        raise ValueError("Ground-truth integrity checker and Source 1 row count disagree")
    train_ids, val_ids = create_split(ids, args.train_fraction, args.seed)
    write_ids(args.output_dir / "train_s1_ids.txt", train_ids)
    write_ids(args.output_dir / "val_s1_ids.txt", val_ids)
    print(
        f"PASS split: train={len(train_ids):,}; validation={len(val_ids):,}; "
        f"union={len(ids):,}; overlap=0; seed={args.seed}; "
        f"train_fraction={args.train_fraction:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
