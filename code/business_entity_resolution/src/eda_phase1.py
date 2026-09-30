#!/usr/bin/env python3
"""Phase 1 EDA for the local business entity resolution challenge data."""

from __future__ import annotations

import argparse
import csv
import io
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd


COLUMNS = ["entity_id", "business_name", "business_address", "country"]
DTYPES = {column: "string" for column in COLUMNS}
SOURCE_FILES = (
    ("train", "S1", "train_source1.tsv"),
    ("train", "S2", "train_source2.tsv"),
    ("train", "S3", "train_source3.tsv"),
    ("test", "S1", "test_source1.tsv"),
    ("test", "S2", "test_source2.tsv"),
    ("test", "S3", "test_source3.tsv"),
)

US_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine",
    "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi",
    "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey",
    "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma",
    "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming", "District of Columbia",
]
US_CODES = "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC".split()
INDIA_STATES_UTS = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa",
    "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala",
    "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland",
    "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal", "Andaman and Nicobar Islands",
    "Chandigarh", "Dadra and Nagar Haveli", "Daman and Diu", "Delhi",
    "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry",
]
INDIA_CODES = "AP AR AS BR CG GA GJ HR HP JH KA KL MP MH MN ML MZ NL OD PB RJ SK TN TS TR UP UK WB AN CH DD DL JK LA LD PY".split()


def alternatives(values: Iterable[str]) -> str:
    return "|".join(re.escape(value) for value in sorted(values, key=len, reverse=True))


US_STATE_NAME = re.compile(
    r"(?<![A-Za-z])(?:" + alternatives(US_STATES) + r")(?![A-Za-z])", re.IGNORECASE
)
INDIA_STATE_NAME = re.compile(
    r"(?<![A-Za-z])(?:" + alternatives(INDIA_STATES_UTS) + r")(?![A-Za-z])",
    re.IGNORECASE,
)
US_STATE_CODE = re.compile(
    r"(?:^|[,\s])(?:" + alternatives(US_CODES) + r")\s*$", re.IGNORECASE
)
INDIA_STATE_CODE = re.compile(
    r"(?:^|[,\s])(?:" + alternatives(INDIA_CODES) + r")\s*$", re.IGNORECASE
)
POSTAL_PATTERNS = {
    "US": re.compile(r"(?<!\d)\d{5}(?:-\d{4})?(?!\d)"),
    "India": re.compile(r"(?<!\d)[1-9]\d{5}(?!\d)"),
    "France": re.compile(r"(?<!\d)\d{5}(?!\d)"),
}
LEGAL_SUFFIX = re.compile(
    r"(?:\bincorporated|\binc\.?|\bcorporation|\bcorp\.?|\bl\.?l\.?c\.?|"
    r"\bl\.?l\.?p\.?|\bl\.?p\.?|\bp\.?c\.?|\bprivate\s+limited|"
    r"\bpvt\.?\s*(?:ltd\.?|limited)|\blimited|\bltd\.?|\bcompany|\bco\.?|"
    r"\b(?:sas|sarl|eurl|snc|gmbh|srl))\W*$",
    re.IGNORECASE,
)


def find_sample(dataset_root: Path, size: int, seed: int) -> tuple[list[tuple[str, str]], Counter[int], int, Counter[str]]:
    if size < 2 or size % 2:
        raise ValueError("--sample-size must be an even number of at least 2")
    per_source = size // 2
    rng = random.Random(seed)
    samples: dict[str, list[tuple[str, str]]] = {"S2-": [], "S3-": []}
    seen = Counter()
    cardinality: Counter[int] = Counter()
    source1_rows = 0
    gt_path = dataset_root / "train" / "train_ground_truth.tsv"
    with gt_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if header != ["source1_entity_id", "matched_entity_ids"]:
            raise ValueError(f"Unexpected ground-truth header in {gt_path}: {header}")
        for source1_id, joined_ids in reader:
            source1_rows += 1
            target_ids = joined_ids.split(",") if joined_ids else []
            cardinality[len(target_ids)] += 1
            for target_id in target_ids:
                prefix = "S2-" if target_id.startswith("S2-") else "S3-"
                if not target_id.startswith(("S2-", "S3-")):
                    raise ValueError(f"Unexpected target ID prefix: {target_id}")
                seen[prefix] += 1
                bucket = samples[prefix]
                pair = (source1_id, target_id)
                if len(bucket) < per_source:
                    bucket.append(pair)
                else:
                    replacement = rng.randrange(seen[prefix])
                    if replacement < per_source:
                        bucket[replacement] = pair
    if any(len(bucket) != per_source for bucket in samples.values()):
        raise ValueError("Not enough labeled links to fill the requested sample")
    if sum(cardinality.values()) != source1_rows:
        raise ValueError("Cardinality counts do not reconcile with ground-truth rows")
    return samples["S2-"] + samples["S3-"], cardinality, source1_rows, seen


def read_selected_records(
    path: Path, wanted: set[str], output: dict[str, dict[str, str]]
) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"Unexpected source header in {path}: {reader.fieldnames}")
        for row in reader:
            entity_id = row["entity_id"]
            if entity_id in wanted:
                output[entity_id] = row


def print_sample(
    dataset_root: Path,
    pairs: list[tuple[str, str]],
    seed: int,
    sample_output: Path,
) -> None:
    wanted_s1 = {source1 for source1, _ in pairs}
    wanted_targets = {target for _, target in pairs}
    records: dict[str, dict[str, str]] = {}
    train_root = dataset_root / "train"
    read_selected_records(train_root / "train_source1.tsv", wanted_s1, records)
    read_selected_records(train_root / "train_source2.tsv", wanted_targets, records)
    read_selected_records(train_root / "train_source3.tsv", wanted_targets, records)
    if any(source1 not in records or target not in records for source1, target in pairs):
        raise ValueError("A sampled labeled record was not found in its source TSV")

    header = [
        "target_source", "source1_entity_id", "source1_name", "source1_address",
        "source1_country", "matched_entity_id", "matched_name", "matched_address",
        "matched_country",
    ]
    rows = []
    for source1_id, target_id in pairs:
        source1, target = records[source1_id], records[target_id]
        rows.append(
            [
                target_id[:2], source1_id, source1["business_name"],
                source1["business_address"], source1["country"], target_id,
                target["business_name"], target["business_address"], target["country"],
            ]
        )

    sample_output.parent.mkdir(parents=True, exist_ok=True)
    with sample_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nTRUE-MATCH SAMPLE: {len(pairs)} pairs; {len(pairs)//2} per target source; seed={seed}")
    print(f"Saved sample TSV: {sample_output}")
    output = io.StringIO()
    writer = csv.writer(output, delimiter="\t", lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    print(output.getvalue(), end="")


def collect_source_stats(dataset_root: Path, chunk_size: int) -> dict:
    results: dict[str, dict[str, Counter]] = {}
    for split, source, filename in SOURCE_FILES:
        key = f"{split}_{source}"
        by_country: dict[str, Counter] = defaultdict(Counter)
        path = dataset_root / split / filename
        for frame in pd.read_csv(
            path,
            sep="\t",
            dtype=DTYPES,
            keep_default_na=False,
            chunksize=chunk_size,
        ):
            if list(frame.columns) != COLUMNS:
                raise ValueError(f"Unexpected source columns in {path}: {list(frame.columns)}")
            for country, group in frame.groupby("country", sort=False):
                stats = by_country[country]
                names = group["business_name"]
                addresses = group["business_address"]
                nonblank = addresses.ne("")
                suffix = names.str.contains(LEGAL_SUFFIX)
                stats["rows"] += len(group)
                stats["missing_name"] += int(names.eq("").sum())
                stats["blank_address"] += int((~nonblank).sum())
                stats["legal_suffix"] += int(suffix.sum())
                postal_pattern = POSTAL_PATTERNS.get(country)
                if postal_pattern is not None:
                    postal_present = addresses[nonblank].str.contains(postal_pattern)
                    stats["postal_denominator"] += int(nonblank.sum())
                    stats["postal_present"] += int(postal_present.sum())
                if country in ("US", "India"):
                    if country == "US":
                        state_flags = addresses.str.contains(US_STATE_NAME) | addresses.str.contains(
                            US_STATE_CODE
                        )
                    else:
                        state_flags = (
                            addresses.str.contains(INDIA_STATE_NAME)
                            | addresses.str.contains(INDIA_STATE_CODE)
                            | addresses.str.contains("తెలంగాణ", regex=False)
                            | addresses.str.contains("தமிழ்நாடு", regex=False)
                        )
                    state_present = int(state_flags[nonblank].sum())
                    stats["state_denominator"] += int(nonblank.sum())
                    stats["state_present"] += state_present
        results[key] = by_country
        print(f"Scanned {path}", flush=True)
    return results


def percent(numerator: int, denominator: int) -> str:
    return f"{100 * numerator / denominator:.2f}%" if denominator else "n/a"


def print_report(
    cardinality: Counter[int],
    source1_rows: int,
    stats: dict,
    link_counts: Counter[str],
) -> None:
    train_countries = {country for source in ("S1", "S2", "S3")
                       for country in stats[f"train_{source}"]}
    test_countries = {country for source in ("S1", "S2", "S3")
                      for country in stats[f"test_{source}"]}
    if "France" in train_countries or "France" not in test_countries:
        raise ValueError("Expected France in test only")
    print("\nCOUNTRY CHECK")
    print(f"Train countries: {sorted(train_countries)}; France in train: False")
    print(f"Test countries: {sorted(test_countries)}")
    print("\nMATCH CARDINALITY")
    for label, count in (
        ("0", cardinality[0]),
        ("1", cardinality[1]),
        (">1", sum(value for size, value in cardinality.items() if size > 1)),
    ):
        print(f"{label}\t{count}\t{percent(count, source1_rows)}")
    print(f"TOTAL\t{source1_rows}\t100.00%")
    print("\nLABELED LINK COUNTS BY TARGET SOURCE")
    print("S2 links\t" + format(link_counts["S2-"], ","))
    print("S3 links\t" + format(link_counts["S3-"], ","))

    print("\nCOUNTRY AND TEXT INDICATORS BY SPLIT / SOURCE / COUNTRY")
    print(
        "split\tsource\tcountry\trows\tmissing_name\tblank_address\tpostal_present_nonblank\tpostal_absent_nonblank\t"
        "state_absent_nonblank\tlegal_suffix_at_name_end"
    )
    for key, countries in stats.items():
        split, source = key.split("_")
        for country, values in sorted(countries.items()):
            blank = values["blank_address"]
            rows = values["rows"]
            missing_names = values["missing_name"]
            postal_absent = values["postal_denominator"] - values["postal_present"]
            if country in ("US", "India"):
                state_absent = values["state_denominator"] - values["state_present"]
                state_text = (
                    f"{state_absent}/{values['state_denominator']} "
                    f"({percent(state_absent, values['state_denominator'])})"
                )
            else:
                state_text = "N/A"
            print(
                f"{split}\t{source}\t{country}\t{rows}\t"
                f"{missing_names}/{rows} ({percent(missing_names, rows)})\t"
                f"{blank}/{rows} ({percent(blank, rows)})\t"
                f"{values['postal_present']}/{values['postal_denominator']} "
                f"({percent(values['postal_present'], values['postal_denominator'])})\t"
                f"{postal_absent}/{values['postal_denominator']} "
                f"({percent(postal_absent, values['postal_denominator'])})\t"
                f"{state_text}\t{values['legal_suffix']}/{rows} "
                f"({percent(values['legal_suffix'], rows)})"
            )

    train_targets = sum(
        values["rows"]
        for source in ("S2", "S3")
        for values in stats[f"train_{source}"].values()
    )
    test_targets = sum(
        values["rows"]
        for source in ("S2", "S3")
        for values in stats[f"test_{source}"].values()
    )
    train_pairs = source1_rows * train_targets
    test_s1 = sum(values["rows"] for values in stats["test_S1"].values())
    test_pairs = test_s1 * test_targets
    print("\nSOURCE ROW COUNTS")
    for split, source, _ in SOURCE_FILES:
        total = sum(values["rows"] for values in stats[f"{split}_{source}"].values())
        print(f"{split} {source}\t{total:,}")
    print("\nPAIRWISE SCALE")
    print(f"train S2+S3 records\t{train_targets:,}")
    print(f"train S1 x (S2+S3)\t{train_pairs:,} comparisons")
    print(f"test S2+S3 records\t{test_targets:,}")
    print(f"test S1 x (S2+S3)\t{test_pairs:,} comparisons")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "6ab10eb3b23ba_student_resource/student_resource/dataset",
    )
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument(
        "--sample-output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "artifacts/eda/true_link_samples.tsv",
        help="Where to save the deterministic side-by-side true-link sample",
    )
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()

    pairs, cardinality, source1_rows, link_counts = find_sample(
        args.dataset_root, args.sample_size, args.seed
    )
    print_sample(args.dataset_root, pairs, args.seed, args.sample_output)
    stats = collect_source_stats(args.dataset_root, args.chunksize)
    print_report(cardinality, source1_rows, stats, link_counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
