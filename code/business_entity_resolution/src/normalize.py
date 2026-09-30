#!/usr/bin/env python3
"""Conservative, deterministic text representations for entity resolution.

These utilities normalize representations only; they do not decide whether
records match. Postal matches are shape-based text heuristics, not validation.
"""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from pathlib import Path
from typing import Iterable


TextValue = str | None
SAMPLE_DEFAULT = Path(__file__).resolve().parents[1] / "artifacts/eda/true_link_samples.tsv"
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "＇": "'"})
_PUNCTUATION_RE = re.compile(r"[\W_]+", flags=re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")
_NUMERIC_TOKEN_RE = re.compile(r"\d+")

# Normalize dotted forms of common legal initialisms before punctuation becomes
# token separators. The original input remains available to callers unchanged.
_NAME_INITIALISM_PATTERNS = (
    (re.compile(r"(?<!\w)l\s*\.\s*l\s*\.\s*c\s*\.?(?!\w)"), "llc"),
    (re.compile(r"(?<!\w)l\s*\.\s*l\s*\.\s*p\s*\.?(?!\w)"), "llp"),
    (re.compile(r"(?<!\w)l\s*\.\s*p\s*\.?(?!\w)"), "lp"),
    (re.compile(r"(?<!\w)p\s*\.\s*c\s*\.?(?!\w)"), "pc"),
)

# Longest terminal forms are checked first; every item is a normalized token
# sequence. "private" and "pvt" alone are included because those variants
# occur in the inspected true-link sample (e.g. Private Limited / Private).
_LEGAL_SUFFIXES = tuple(sorted((
    ("limited", "liability", "company"),
    ("private", "limited"),
    ("pvt", "limited"),
    ("pvt", "ltd"),
    ("incorporated",), ("corporation",), ("company",),
    ("limited",), ("private",), ("pvt",),
    ("inc",), ("corp",), ("llc",), ("l", "l", "c"),
    ("llp",), ("l", "l", "p"), ("lp",), ("l", "p"),
    ("pc",), ("p", "c"), ("ltd",), ("co",),
    ("sas",), ("sarl",), ("eurl",), ("snc",),
    ("gmbh",), ("srl",),
), key=len, reverse=True))

# Observed address variants from the Phase 1 positive-pair sample. Expansion is
# restricted to the final token of a comma-delimited component; "st" is omitted
# because it can mean Street or Saint (among other things).
_ADDRESS_SUFFIX_EXPANSIONS = {
    "rd": "road",
    "ave": "avenue",
    "ct": "court",
    "dr": "drive",
    "ln": "lane",
    "trl": "trail",
    "cir": "circle",
}
_POSTAL_PATTERNS = {
    "us": re.compile(r"(?<!\d)\d{5}(?:-\d{4})?(?!\d)"),
    "india": re.compile(r"(?<!\d)[1-9]\d{5}(?!\d)"),
    "france": re.compile(r"(?<!\d)\d{5}(?!\d)"),
}
_SAMPLE_COLUMNS = {
    "target_source", "source1_entity_id", "source1_name", "source1_address",
    "source1_country", "matched_entity_id", "matched_name", "matched_address",
    "matched_country",
}


def _value(value: TextValue) -> str:
    """Map None and whitespace-only input to the common empty-string value."""
    if value is None:
        return ""
    return str(value)


def normalize_text(value: TextValue) -> str:
    """Apply Unicode NFKC, casefolding, and whitespace collapse only."""
    text = unicodedata.normalize("NFKC", _value(value)).casefold()
    return _WHITESPACE_RE.sub(" ", text).strip()


def _punctuation_to_spaces(text: str) -> str:
    # Python regex \w omits combining marks used in Indic scripts. Keep
    # marks attached to their letters; use the fast regex path for ASCII.
    if text.isascii():
        cleaned = _PUNCTUATION_RE.sub(" ", text)
    else:
        cleaned = "".join(
            char if (char.isalnum() or unicodedata.category(char).startswith("M")
                     or char in "\u200c\u200d") else " "
            for char in text
        )
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


def normalize_name(value: TextValue) -> str:
    """Return a basic name form, preserving legal suffix tokens and accents."""
    text = unicodedata.normalize("NFKC", _value(value)).casefold().translate(_APOSTROPHES)
    text = text.replace("&", " and ").replace("'", "")
    for pattern, replacement in _NAME_INITIALISM_PATTERNS:
        text = pattern.sub(f" {replacement} ", text)
    return _punctuation_to_spaces(text)


def name_tokens(value: TextValue, *, core: bool = False) -> tuple[str, ...]:
    """Return ordered tokens; repetitions and token order are retained."""
    text = core_name(value) if core else normalize_name(value)
    return tuple(text.split()) if text else ()


def core_name(value: TextValue) -> str:
    """Return the normalized name with recognized legal suffixes removed at end."""
    tokens = list(name_tokens(value))
    changed = True
    while tokens and changed:
        changed = False
        for suffix in _LEGAL_SUFFIXES:
            width = len(suffix)
            if width <= len(tokens) and tuple(tokens[-width:]) == suffix:
                del tokens[-width:]
                changed = True
                break
    return " ".join(tokens)


def core_name_tokens(value: TextValue) -> tuple[str, ...]:
    """Return ordered tokens after terminal legal suffix removal."""
    return name_tokens(value, core=True)


def strip_latin_accents(value: TextValue) -> str:
    """Strip diacritics from Latin letters while retaining marks in other scripts."""
    decomposed = unicodedata.normalize("NFKD", _value(value))
    output: list[str] = []
    last_base_is_latin = False
    for char in decomposed:
        category = unicodedata.category(char)
        if category.startswith("M"):
            if not last_base_is_latin:
                output.append(char)
            continue
        output.append(char)
        name = unicodedata.name(char, "")
        last_base_is_latin = name.startswith("LATIN ")
    return unicodedata.normalize("NFKC", "".join(output))


def name_search_text(value: TextValue) -> str:
    """Return an optional accent-stripped form of the basic normalized name."""
    return normalize_text(strip_latin_accents(normalize_name(value)))


def normalize_address(value: TextValue) -> str:
    """Return a punctuation/whitespace-normalized address without dropping digits."""
    text = unicodedata.normalize("NFKC", _value(value)).casefold().translate(_APOSTROPHES)
    text = text.replace("'", "")
    return _punctuation_to_spaces(text)


def standardize_address(value: TextValue) -> str:
    """Expand a short observed street type only at a component's final token.

    This is a separate representation. The basic address remains available and
    ambiguous "st" is intentionally not expanded.
    """
    text = unicodedata.normalize("NFKC", _value(value)).casefold()
    parts: list[str] = []
    for component in text.split(","):
        tokens = normalize_address(component).split()
        if len(tokens) > 1 and tokens[-1] in _ADDRESS_SUFFIX_EXPANSIONS:
            tokens[-1] = _ADDRESS_SUFFIX_EXPANSIONS[tokens[-1]]
        if tokens:
            parts.extend(tokens)
    return " ".join(parts)


def address_tokens(value: TextValue, *, standardized: bool = False) -> tuple[str, ...]:
    """Return ordered address tokens from the requested representation."""
    text = standardize_address(value) if standardized else normalize_address(value)
    return tuple(text.split()) if text else ()


def extract_numeric_tokens(value: TextValue) -> tuple[str, ...]:
    """Extract digit runs in order, preserving leading zeros and repetitions."""
    text = unicodedata.normalize("NFKC", _value(value))
    return tuple(_NUMERIC_TOKEN_RE.findall(text))


def normalize_country(value: TextValue) -> str:
    """Normalize country text without a closed vocabulary or alias mapping."""
    return normalize_text(value)


def postal_candidates(value: TextValue, country: TextValue) -> tuple[str, ...]:
    """Extract country-shaped postal candidates heuristically; not a postal parser.

    US yields five-digit ZIP-like tokens (including ZIP+4); India yields six-digit
    PIN-like tokens beginning 1-9; France yields five-digit postcode-like tokens.
    Unknown country labels safely produce an empty tuple.
    """
    pattern = _POSTAL_PATTERNS.get(normalize_country(country))
    if pattern is None:
        return ()
    text = unicodedata.normalize("NFKC", _value(value))
    return tuple(pattern.findall(text))


def postal_candidate(value: TextValue, country: TextValue) -> str | None:
    """Return the first heuristic postal candidate, if any."""
    candidates = postal_candidates(value, country)
    return candidates[0] if candidates else None


def _sample_summary(sample_path: Path, show_examples: int = 8) -> str:
    """Normalize the fixed true-link sample and summarize simple exact signals."""
    if show_examples < 0:
        raise ValueError("--show-examples must be non-negative")
    with sample_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or set(reader.fieldnames) != _SAMPLE_COLUMNS:
            raise ValueError(f"Unexpected true-link sample columns: {reader.fieldnames!r}")
        rows = list(reader)
    basic_equal = core_equal = numeric_conflicts = postal_conflicts = 0
    for row in rows:
        s1_name = normalize_name(row["source1_name"])
        matched_name = normalize_name(row["matched_name"])
        basic_equal += s1_name == matched_name
        core_equal += core_name(row["source1_name"]) == core_name(row["matched_name"])
        nums_a = set(extract_numeric_tokens(row["source1_address"]))
        nums_b = set(extract_numeric_tokens(row["matched_address"]))
        numeric_conflicts += bool(nums_a and nums_b and nums_a.isdisjoint(nums_b))
        postal_a = set(postal_candidates(row["source1_address"], row["source1_country"]))
        postal_b = set(postal_candidates(row["matched_address"], row["matched_country"]))
        postal_conflicts += bool(postal_a and postal_b and postal_a.isdisjoint(postal_b))

    lines = [
        f"TRUE-LINK NORMALIZATION SANITY SAMPLE ({len(rows)} pairs)",
        f"Exact basic-name pairs: {basic_equal}/{len(rows)}",
        f"Exact core-name pairs: {core_equal}/{len(rows)}",
        f"Conflicting numeric signals (both present, disjoint): {numeric_conflicts}/{len(rows)}",
        f"Conflicting postal candidates (both present, disjoint): {postal_conflicts}/{len(rows)}",
        "Descriptive only; this is not validation scoring or a match-quality estimate.",
    ]
    for index, row in enumerate(rows[:show_examples], start=1):
        country = row["source1_country"]
        lines.extend([
            "",
            f"Example {index} [{row['target_source']}]",
            f"S1 name: {row['source1_name']!r}",
            f"   basic: {normalize_name(row['source1_name'])!r}",
            f"    core: {core_name(row['source1_name'])!r}",
            f"Target name: {row['matched_name']!r}",
            f"      basic: {normalize_name(row['matched_name'])!r}",
            f"       core: {core_name(row['matched_name'])!r}",
            f"S1 address: {row['source1_address']!r}",
            f"    basic: {normalize_address(row['source1_address'])!r}",
            f"    nums: {extract_numeric_tokens(row['source1_address'])!r}; postal candidates: {postal_candidates(row['source1_address'], country)!r}",
            f"Target address: {row['matched_address']!r}",
            f"       basic: {normalize_address(row['matched_address'])!r}",
            f"       nums: {extract_numeric_tokens(row['matched_address'])!r}; postal candidates: {postal_candidates(row['matched_address'], row['matched_country'])!r}",
        ])
    return "\n".join(lines) + "\n"


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanity-sample", action="store_true", help="Print normalization examples and summary for the saved 30-pair sample")
    parser.add_argument("--sample-path", type=Path, default=SAMPLE_DEFAULT)
    parser.add_argument("--show-examples", type=int, default=8)
    args = parser.parse_args(argv)
    if not args.sanity_sample:
        parser.error("pass --sanity-sample to run the qualitative sample check")
    try:
        print(_sample_summary(args.sample_path, args.show_examples), end="")
    except (OSError, ValueError, csv.Error) as exc:
        print(f"FAIL: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
