#!/usr/bin/env python3
"""Stream-check the challenge TSV schemas, row counts, and source ID prefixes."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SOURCE_FILES = {
    "train_source1.tsv": "S1-",
    "train_source2.tsv": "S2-",
    "train_source3.tsv": "S3-",
    "test_source1.tsv": "S1-",
    "test_source2.tsv": "S2-",
    "test_source3.tsv": "S3-",
}
EXPECTED_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
STRING_DTYPES = {column: "string" for column in EXPECTED_COLUMNS}


def check_file(path: Path, prefix: str, chunksize: int) -> tuple[int, list[str]]:
    rows = 0
    errors: list[str] = []
    observed_columns: list[str] | None = None

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=STRING_DTYPES,
        keep_default_na=False,
        chunksize=chunksize,
    ):
        if observed_columns is None:
            observed_columns = list(chunk.columns)
            if observed_columns != EXPECTED_COLUMNS:
                errors.append(
                    f"columns {observed_columns!r}; expected {EXPECTED_COLUMNS!r}"
                )
        rows += len(chunk)
        bad = ~chunk["entity_id"].str.startswith(prefix)
        if bad.any():
            examples = chunk.loc[bad, "entity_id"].head(5).tolist()
            errors.append(f"IDs without {prefix!r} prefix (examples: {examples})")
            break

    if observed_columns is None:
        errors.append("file has no header or data rows")
    return rows, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "6ab10eb3b23ba_student_resource/student_resource/dataset",
        help="Directory containing train/ and test/ TSV folders",
    )
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()

    failures = 0
    for filename, prefix in SOURCE_FILES.items():
        split = "train" if filename.startswith("train_") else "test"
        path = args.dataset_root / split / filename
        if not path.is_file():
            print(f"FAIL {path}: file not found")
            failures += 1
            continue
        try:
            rows, errors = check_file(path, prefix, args.chunksize)
        except Exception as exc:  # surface parse/schema failures in the summary
            print(f"FAIL {path}: {type(exc).__name__}: {exc}")
            failures += 1
            continue
        if errors:
            print(f"FAIL {path}: rows={rows}; " + "; ".join(errors))
            failures += 1
        else:
            print(
                f"PASS {path}: rows={rows}; columns={EXPECTED_COLUMNS}; "
                "dtypes=string; ID prefix=" + prefix
            )

    print(f"Checked {len(SOURCE_FILES)} source files; failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
