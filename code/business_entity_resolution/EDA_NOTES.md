# Phase 1 EDA Notes

Source: supplied dataset/train and dataset/test TSVs only. Counts and text
signals below are reproducible with src/eda_phase1.py (seed 20260925). Its
default run prints 30 labeled links side by side, saves them to
code/business_entity_resolution/artifacts/eda/true_link_samples.tsv, then scans
the source files in 200,000-row chunks.

## Findings

- **Match cardinality (2,206,821 S1 training rows):** 123,247 (5.58%) have no
  linked records, 119,157 (5.40%) have one, and 1,964,417 (89.02%) have more
  than one. A single-best-match or one-to-one assignment would contradict most
  training labels.
- **Countries:** training contains US and India only. Test also contains France:
  259,452 S1, 703,378 S2, and 731,615 S3 rows.
- **Scale:** training has 10,320,219 S2+S3 records; test has 9,969,589. Comparing
  every S1 against every target record would mean about 22.8 trillion training
  comparisons and 17.3 trillion test comparisons. Blocking is required.
- **Address availability:** S1 addresses are filled throughout train and test.
  In train, blank address rates are 2.87–3.68% for S2 and 3.07–3.50% for S3
  (US/India range). Test blanks range from 2.28–3.06% for S2 and 2.46–2.94%
  for S3 across countries.
- **Postal signal:** among nonblank addresses, a recognizable postal-shaped
  sequence is absent in about 89% of US addresses, over 99.98% of India
  addresses, and 99.45–99.59% of France addresses. Postal codes are therefore
  sparse evidence, especially for India and France.
- **State/UT signal:** among nonblank US addresses, recognizable state tokens
  are absent in roughly 0.5–13.2% depending on source/split. For India the
  measured absence ranges from 1.6% in S1 to about 19–30.5% in S2/S3. Differences
  across sources suggest field completeness and formatting vary materially.
- **Legal-form signal:** a recognized suffix at the end of the name appears in
  roughly 39–55% of US names, 41–84% of India names, and 33–55% of France names,
  varying by source. The remaining names often lack a suffix or use a form the
  conservative suffix list does not recognize.

## Country distribution and heuristic rates

Counts are records per source. Postal/state absence is measured among nonblank
addresses; legal suffix is measured over all names.

| Split | Source | US | India | France |
|---|---:|---:|---:|---:|
| Train | S1 | 1,323,633 | 883,188 | — |
| Train | S2 | 3,016,817 | 2,017,799 | — |
| Train | S3 | 3,170,056 | 2,115,547 | — |
| Test | S1 | 663,106 | 809,986 | 259,452 |
| Test | S2 | 1,871,330 | 2,312,565 | 703,378 |
| Test | S3 | 1,945,701 | 2,405,000 | 731,615 |

| Split | Source | Country | Blank address | Postal absent | State/UT absent | Legal suffix |
|---|---|---|---:|---:|---:|---:|
| Train | S1 | US | 0.00% | 89.05% | 13.20% | 55.26% |
| Train | S1 | India | 0.00% | 99.98% | 1.56% | 84.22% |
| Train | S2 | US | 3.68% | 89.15% | 12.94% | 39.83% |
| Train | S2 | India | 2.87% | 99.98% | 19.22% | 41.20% |
| Train | S3 | US | 3.50% | 89.14% | 0.65% | 39.11% |
| Train | S3 | India | 3.07% | 99.99% | 30.13% | 46.60% |
| Test | S1 | US | 0.00% | 89.05% | 13.22% | 55.21% |
| Test | S1 | India | 0.00% | 99.98% | 1.57% | 84.19% |
| Test | S1 | France | 0.00% | 99.59% | N/A | 55.03% |
| Test | S2 | US | 2.94% | 89.00% | 12.87% | 41.50% |
| Test | S2 | India | 2.28% | 99.99% | 19.18% | 42.98% |
| Test | S2 | France | 3.06% | 99.47% | N/A | 32.96% |
| Test | S3 | US | 2.84% | 88.97% | 0.53% | 40.74% |
| Test | S3 | India | 2.46% | 99.99% | 30.47% | 48.77% |
| Test | S3 | France | 2.94% | 99.45% | N/A | 32.71% |

## What the labeled examples show

The 30-pair sample is balanced between S2 and S3 and includes both countries
present in training. It shows several concrete patterns:

- Case, punctuation, legal-form and abbreviation changes: “Private Limited” /
  “Private”, “LLC” / “L.L.C.”, “Road” / “Rd”, and inserted punctuation.
- Name corruption beyond simple spelling edits: a sampled S2 link changes
  “Scholarship Fellowship” to “SCHOLARSHIP SCHOLARSHIP FECLHOWMSIP”; another
  changes “Optimal Xsolla” to “Optimal Xsola”.
- Address numbers may be padded or altered (“5064” / “005064”, “501” / “501A”),
  components can move, and some true linked records have blank addresses.
- Indian state names may be transliterated into regional scripts (for example,
  Telugu “తెలంగాణ” alongside “Telangana”) or shortened to codes such as “RJ”,
  “WB”, and “DL”.
- Some linked names are only loosely similar (“Dermatology Medicine” /
  “Dermatology Partners”; “CC Dynamic” / “CC Center”), so address evidence and
  hard negatives matter. Other records use a website or trading-name string in
  place of a conventional business name.

## Measurement notes

Postal-code detection uses US five-digit ZIP (optionally ZIP+4), India six-digit
PIN, and France five-digit postcode shapes. State detection searches known US
state names/codes and India state/UT names/codes, plus the Telugu and Tamil
variants observed in the sample. State rates are not applicable to France.
Legal-form detection is a conservative list of US/India/French forms anchored at
the end of the name. Address blankness is counted separately; postal and state
absence rates use nonblank addresses as their denominator.

These are string heuristics, not parsed address fields: numeric house numbers
can look like postcodes, two-letter codes can be ambiguous, and unlisted or
unrecognized-script state names can be missed. Use the figures to describe the
data and prioritize validation, not as ground-truth component labels.

## Phase 1 exit checks

The deterministic sample is also saved as
code/business_entity_resolution/artifacts/eda/true_link_samples.tsv (30 data rows,
15 per target source), so the evidence is retained beyond terminal output.

| Split | S1 records | S2 records | S3 records |
|---|---:|---:|---:|
| Train | 2,206,821 | 5,034,616 | 5,285,603 |
| Test | 1,732,544 | 4,887,273 | 5,082,316 |

The training ground truth contains 3,693,619 S2 links and 3,944,746 S3 links;
S3 has 251,127 more labeled links. S3 is also larger than S2 in both splits.
S2 and S3 address blankness is broadly similar; their recognizable state/UT
signals differ more, particularly for India (state/UT absent in about 19% of S2
versus 30% of S3 nonblank addresses). The same detector reports much lower US
state-token absence in S3 than S2. These source differences should be treated as
real profile signals, subject to the heuristic limitations below.

No business names are blank in any of the six source files (0 rows in each).
Address blankness is nonzero only in S2/S3. Postal-shaped signals are present
in about 10.85–11.03% of nonblank US addresses, 0.01–0.02% of nonblank India
addresses, and 0.41–0.55% of nonblank France addresses.

Observed legal-form variations include Inc./Incorporated, Corp./Corporation,
LLC/L.L.C., LLP, Limited/Ltd., Private Limited, and Pvt./Pvt Ltd. The sample
contains direct pairs such as LLC versus L.L.C., Private Limited versus Private,
and Limited versus Ltd.
