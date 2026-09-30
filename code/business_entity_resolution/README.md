# Business Entity Resolution

> **Current status:** model selection, validation, and hidden-test inference are complete. For current submission paths, validation commands, Phase 12 rerun order, and the GitHub-vs-local artifact inventory, start with the repository [README](../../README.md) and [ARTIFACTS.md](ARTIFACTS.md). This file keeps the detailed phase-by-phase development record.

## Current output files

- `output/candidate_pairs.tsv` — frozen candidate set; required in the challenge package.
- `output/matching_results.tsv` — original prediction.
- `output/phase112/matching_results.tsv` — Phase 11.2A policy-only prediction.

All are local-only and ignored by Git. The candidate file is shared by both predictions because Phase 11.2A changed only the policy.

Project scaffold for the three-source business entity resolution challenge.

## Setup

From the workspace root:

```bash
python3 -m venv venv
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -r code/business_entity_resolution/requirements.txt
```

The requirements file is pinned to the versions verified in this environment.

## Check the supplied data

```bash
venv/bin/python code/business_entity_resolution/src/check_data.py
```

The checker reads the six Source 1/2/3 TSV files in chunks, with an explicit tab
separator and string dtypes. By default it uses the dataset extracted at
`6ab10eb3b23ba_student_resource/student_resource/dataset`. To use another copy:

```bash
venv/bin/python code/business_entity_resolution/src/check_data.py \
  --dataset-root /path/to/student_resource/dataset
```

The checker reports row counts, schemas, and source-prefix errors. It does not
load the ground-truth file or perform matching.

## Phase 1 EDA

Run the reproducible sample and full chunked profile:

```bash
venv/bin/python code/business_entity_resolution/src/eda_phase1.py
```

The default run prints 30 true links (15 to each target source), saves the side-by-side sample to code/business_entity_resolution/artifacts/eda/true_link_samples.tsv, and reports cardinality, links by target source, row counts, missingness, country, text-signal, and pairwise-scale summaries. See code/business_entity_resolution/EDA_NOTES.md for the recorded findings.

## Validate submission files

The challenge validator is at
`6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py`.
Run it from the extracted `student_resource` directory, passing matching and
candidate TSV paths plus `--test-dir dataset/test`. The optional `--check-ids`
mode can use substantial memory on the full test set.

## Phase 2: integrity, split, and validation scoring

Check the training ground truth against the complete Source 1 training universe. This uses a temporary disk-backed SQLite index for the multi-million-row coverage check:

```bash
venv/bin/python code/business_entity_resolution/src/check_ground_truth.py
```

Create the fixed S1-level 90/10 split (all rows for an S1 entity stay together):

```bash
venv/bin/python code/business_entity_resolution/src/split_data.py \
  --dataset-root 6ab10eb3b23ba_student_resource/student_resource/dataset \
  --train-fraction 0.90 --seed 20260925 \
  --output-dir code/business_entity_resolution/artifacts/splits
```

The script checks ground-truth coverage before writing sorted IDs to the split artifacts. Re-running with the same arguments reproduces the same files.

After generating validation predictions with exactly one row for every validation S1 ID, score them against the full training ground truth (the scorer filters ground truth to the validation IDs):

```bash
venv/bin/python code/business_entity_resolution/src/scoring.py \
  --ground-truth 6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_ground_truth.tsv \
  --predictions /path/to/validation_predictions.tsv \
  --s1-ids code/business_entity_resolution/artifacts/splits/val_s1_ids.txt
```

The primary reported score is macro entity-level F0.5, including singletons. Micro precision and recall are labeled diagnostics. Run deterministic unit tests with:

```bash
venv/bin/python -m unittest discover -s code/business_entity_resolution/tests -v
```

## Phase 2.5: central diagnostics

Run diagnostics for validation predictions (one row for every validation S1 ID):

    venv/bin/python code/business_entity_resolution/src/diagnostics.py --ground-truth 6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_ground_truth.tsv --predictions /path/to/validation_predictions.tsv --s1-ids code/business_entity_resolution/artifacts/splits/val_s1_ids.txt

The report is printed and saved to code/business_entity_resolution/artifacts/diagnostics/summary.txt. Macro entity-level F0.5 is the official optimization metric and is supplied by src/scoring.py. Overall micro precision/recall, singleton accuracy, entity error counts, and S2/S3-specific precision/recall are diagnostics; they do not replace the macro score.

When a final candidate TSV exists, add --candidates /path/to/candidate_pairs.tsv to report overall and per-source candidate recall, blocking misses, model misses, and candidate-count distribution. Until then the report explicitly says blocking diagnostics are unavailable. This option analyzes an existing file only; it does not generate candidates.

To append a named run to artifacts/experiments.csv, add --experiment-name baseline-v1 --notes "description". Existing names are rejected; --allow-duplicate-experiment permits an intentional repeat. Timestamps are UTC metadata. Candidate metrics remain blank when no candidate file is supplied, and peak memory is left blank because this phase does not measure it.

## Phase 3: conservative text normalization

The src/normalize.py module provides deterministic text representations for
later blocking and feature work; normalization does not perform matching.
Original source values remain untouched. Names expose a basic Unicode
NFKC/casefolded form that retains Indic combining marks, ordered tokens, a core form with recognized legal
suffixes removed only at the end, and a separate Latin accent-stripped search
form. The suffix set covers observed and common terminal forms such as
Ltd/Limited, Pvt/Private, Inc, Corp, LLC/LLP/LP/PC, Company, SAS, SARL, EURL,
SNC, GmbH, and SRL. Basic names retain the legal form, and words like India
and USA remain in the core form.

Addresses expose a punctuation-normalized form, ordered tokens, extracted digit
runs (including leading zeros), and a separate standardized form that expands
only observed terminal component abbreviations (rd, ave, ct, dr, ln, trl, cir).
Ambiguous st is left unchanged. Postal candidates are shape heuristics only:
US ZIP-like, India PIN-like, and France postcode-like strings; they do not
verify postal codes. Country normalization is open-set text cleanup, so
unfamiliar country labels are retained safely. None and blank text produce
empty strings/tuples or no postal candidate.

Run all phase tests:

    venv/bin/python -m unittest discover -s code/business_entity_resolution/tests -v

Print a qualitative report for the saved sample (eight examples by default):

    venv/bin/python code/business_entity_resolution/src/normalize.py --sanity-sample

Use --sample-path /path/to/true_link_samples.tsv to inspect another sample or
--show-examples 30 to print every saved pair. Exact-name and signal counts are
descriptive sample summaries only, not validation scores.


## Phase 4: validation candidate generation

Blocking retrieves plausible Source 2 and Source 3 records for each validation
Source 1 entity. Exhaustive search would require about 22.8 trillion training
comparisons. This stage measures true-link coverage; it does not decide final
matches or calculate challenge F0.5.

Run the disk-backed V1 index and validation generator from the workspace root:

    venv/bin/python code/business_entity_resolution/src/blocking.py

The generator uses the existing name, core-name, token, country, numeric, and
postal utilities. It unions exact basic name, exact core name, informative name
tokens, name-token plus address number, name-token plus postal candidate, and
strong address plus number routes. The last route requires a shared number of at
least three digits and two shared address words, with one at least five letters. The token route uses target posting frequencies rather than a fixed
vocabulary. The selected caps allow up to 20,000 postings for address-supported
name tokens and numbers, and up to 1,000 number postings for the strong address
route; a seeded 4,000-S1 threshold pilot is recorded in
artifacts/blocking/v1_threshold_pilot.json. Empty keys and overly common postings are skipped and counted in
the V1 report. Country strings remain open-set. Address disagreement never
removes a name-based candidate.

The first run builds artifacts/blocking/v1_index.sqlite from train S2 and S3;
later runs reuse it. Pass --rebuild-index after changing the source files or
blocking index format. Candidate pairs and per-pair route metadata are written
as compressed TSVs in artifacts/blocking. The report and route ablation are in
artifacts/blocking/v1_report.json. A bounded, inspectable miss sample is in
artifacts/diagnostics/blocking_misses_v1.tsv; all missed IDs are in the
compressed blocking_miss_ids_v1.tsv.gz file. Large derived artifacts are
ignored by Git. Index construction and validation can take substantial time
and disk space on the full data; peak resident memory is measured in the V1
report.

Score candidate coverage using diagnostics without match predictions:

    venv/bin/python code/business_entity_resolution/src/diagnostics.py --candidates-only --candidates code/business_entity_resolution/artifacts/blocking/v1_validation_candidates.tsv.gz --output code/business_entity_resolution/artifacts/diagnostics/blocking_v1_summary.txt

The diagnostics CLI streams the compressed candidate file and checks exact
validation S1 coverage. Candidate recall, S2/S3 recall, blocking misses, and
candidate-count distribution are the Phase 4 measures. To record a run in
artifacts/experiments.csv, add a unique --experiment-name and optional
--notes, --runtime-seconds, and --peak-memory-mb values taken from the V1
report. Prediction-related experiment fields stay blank.


The completed V1 validation run evaluated 220,683 S1 entities and 764,074 true
links. It recovered 621,634 links (81.36% candidate recall: S2 80.36%, S3
82.30%) while generating 28,558,762 candidate pairs, or 129.41 per S1 on
average. P95 was 398 and 0.54% of S1 entities had no candidates. The
candidate-only diagnostics report independently reconciled these totals.
The measured index build took 426.72 seconds; the final validation run reused
that index and took 2,049.73 seconds overall, including 1,946.17 seconds for
candidate generation. Peak resident memory in the final run was 307.51 MiB.
The SQLite index occupies about 8.21 GB. These are measurements on this
machine, not promised runtimes elsewhere.

V1 misses 142,440 validation links. Inspection shows script/transliteration
changes, domain-style or unrelated names, short aliases, and missing or changed
address signals. A later fuzzy retrieval phase is expected to address more of
these cases; this phase stops at candidate generation. See
artifacts/diagnostics/blocking_miss_review_v1.md for eight inspected examples.


## Phase 5: heuristic matching baseline

Phase 5 is the first end-to-end matcher, not the final model. It turns Phase 4
candidates into zero, one, or multiple predicted links per S1. A correctly
predicted empty list represents a singleton. The fixed heuristic combines
basic/core name character similarity, core token similarity, exact names,
address similarity, numeric and postal support/conflict, and blocking-route
evidence. These signals use the existing normalization utilities and RapidFuzz.
No external business data, ML model, or new retrieval method is used.

Create a label-independent 100,000-S1 tuning subset from the development IDs,
then generate its candidates with the unchanged V1 index and six routes:

    venv/bin/python code/business_entity_resolution/src/baseline.py make-tune-ids
    venv/bin/python code/business_entity_resolution/src/baseline.py generate-tune-candidates

Benchmark and score tuning pairs; compare exact-name, exact-core, high-precision
rule, and weighted-heuristic baselines; search shared and S2/S3 thresholds using
only the tuning subset:

    venv/bin/python code/business_entity_resolution/src/baseline.py tune

The selected configuration is frozen in artifacts/baseline/heuristic_v1.json.
The command refuses to replace an existing frozen configuration. After tuning,
run the held-out validation matcher once using the existing Phase 4 candidate
and metadata files:

    venv/bin/python code/business_entity_resolution/src/baseline.py validate

Score and diagnose the fixed predictions. The diagnostics command streams the
compressed candidate file and checks that every prediction was a candidate:

    venv/bin/python code/business_entity_resolution/src/scoring.py --ground-truth 6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_ground_truth.tsv --predictions code/business_entity_resolution/artifacts/baseline/v1_validation_predictions.tsv --s1-ids code/business_entity_resolution/artifacts/splits/val_s1_ids.txt
    venv/bin/python code/business_entity_resolution/src/diagnostics.py --predictions code/business_entity_resolution/artifacts/baseline/v1_validation_predictions.tsv --candidates code/business_entity_resolution/artifacts/blocking/v1_validation_candidates.tsv.gz --output code/business_entity_resolution/artifacts/diagnostics/phase5_baseline_summary.txt --experiment-name phase5-heuristic-baseline --notes "Frozen tune thresholds in artifacts/baseline/heuristic_v1.json" --runtime-seconds 1205.46015618 --peak-memory-mb 1105.1484375

The values above come from artifacts/baseline/validation_run.json. The
experiment name is unique and can be appended once; use
--allow-duplicate-experiment only for an intentional repeated log row. The
deterministic, bounded error sample is saved as
artifacts/diagnostics/phase5_baseline_errors.tsv. Large pair-level caches and
candidate files are reproducible local artifacts and are ignored by Git.

The 220,683-ID validation split is held out from Phase 5 heuristic selection.
Phase 4 blocking was previously developed on that split, so its candidate
recall is context rather than a newly independent retrieval measurement.
Candidate recall measures retrieval; macro entity-level F0.5 measures final
match quality. Do not retune the heuristic after seeing the validation score.

The CLI refuses to replace the frozen configuration and refuses to overwrite
validation predictions. To redo tuning deliberately, first archive or remove
artifacts/baseline/heuristic_v1.json and tune_predictions.tsv. Validation is a
one-run held-out evaluation for this configuration.

Run all tests with:

    venv/bin/python -m unittest discover -s code/business_entity_resolution/tests -v

All 80 tests passed after Phase 5 implementation.

### Phase 5 measured baseline (2026-09-25)

The fixed score uses weights 37.5% core-name character ratio, 15% basic-name
character ratio, 11.25% token-set ratio, and 11.25% core-token Jaccard, plus
small exact-name, address, number, postal, and route bonuses; number/postal
conflicts subtract points. A strong-address candidate can receive a score floor
of 78 when its address is close and its nonpostal number agrees. Thresholds
were selected only on the deterministic 100,000-ID tune subset. The best result
used S2=90 and S3=90. The search tested shared thresholds 55–100 by five,
refined 86–94 by one, then compared a 5×5 source-specific grid from
86/88/90/92/94 for each source.

Tuning found macro entity-level F0.5 **0.51493559**, precision diagnostic
0.36196, recall diagnostic 0.37091, singleton accuracy 0.69283, and 1,706
singleton false merges. The exact-name, exact-core, high-precision-rule, and
weighted heuristic comparison scores were 0.32520, 0.39582, 0.50542, and
0.51494. Tune candidate recall was 81.16% over 12,840,689 candidates (mean
128.41 per S1), close to the 81.36% Phase 4 validation candidate recall.

On the held-out 220,683 validation S1 entities, frozen thresholds produced
macro entity-level F0.5 **0.51628906**, TP=284,680, FP=507,869, FN=479,394,
precision diagnostic 0.35920 and recall diagnostic 0.37258. The system emitted
792,549 links (mean 3.59, P95 14, maximum 394). Singleton diagnostics: 12,386
true singletons, 42,786 predicted singletons, 8,590 correctly predicted
singletons, 69.35% singleton accuracy, and 3,796 false merges. S2 precision /
recall diagnostics were 0.36278 / 0.38644; S3 were 0.35565 / 0.35958.

The existing V1 candidates had 81.36% candidate recall; the matcher recovered
284,680 true links overall, missed 142,440 true links at blocking, and missed
336,954 true links that were present among candidates. The bounded error sample
shows false positives often come from repeated or generic exact business names
at unrelated addresses. Model misses include domain-form names, typos, reordered
names, and candidates scored below threshold when addresses conflict or provide
little support. Blocking misses include cross-script variants, missing/omitted
address signals, and name token rearrangements. These examples are qualitative
and do not estimate prevalence.

Tune candidate generation took 911.36 seconds at 107.41 MiB peak RSS. Tune pair
scoring processed 12,840,689 pairs in 519.13 seconds (24,735 pairs/second,
697.15 MiB peak RSS). The held-out pass scored 28,558,762 pairs in 1,143.03
seconds (24,985 pairs/second), completed in 1,205.46 seconds total, and peaked
at 1,105.15 MiB RSS. These are machine-specific measurements. Phase 4 blocking
was developed using this validation split, so candidate recall is not an
independent retrieval estimate. Validation results were not used to change the
Phase 5 thresholds.


## Phase 6: character n-gram candidate retrieval

Phase 6 extends the frozen six-route V1 candidate set with source-separated character n-gram TF-IDF retrieval. It uses the Phase 3 normalized name, `char_wb` n-grams from length 3 through 5, a deterministic 2^20-feature hashing space, source-specific IDF, sparse target shards, and normalized exact-country partitioning that accepts unseen country strings. N-grams appearing in more than 0.1% of the target source are pruned to control common-feature work. Retrieval ranks cosine scores descending and breaks ties by target ID ascending. Empty names produce no TF-IDF candidates. V2 is the union of every V1 pair and each source’s selected top-K pairs; it does not decide final matches.

Reproduction commands (run from the repository root):

```bash
cd code/business_entity_resolution
../../venv/bin/python src/tfidf_retrieval.py build-index
../../venv/bin/python src/tfidf_retrieval.py benchmark --prefix-count 1000
../../venv/bin/python src/tfidf_retrieval.py tune
# Review artifacts/tfidf_v2_tune/tune_sweep.json, then freeze the chosen shared K.
../../venv/bin/python src/tfidf_retrieval.py freeze --k <selected-K>
../../venv/bin/python src/tfidf_retrieval.py validate
```

The tune sweep evaluates K=5, 10, 20, 30, and 50 using only `tune_s1_ids.txt`. Validation reads the frozen K values from `artifacts/blocking/v2_config.json`; it does not select or retune them. `artifacts/experiments.csv` retains the prior Phase 4/5 rows and adds the Phase 6 sweep and selected-configuration rows. Reproducible sparse indexes and candidate caches are ignored by Git.

### Phase 6 tune-only result and Phase 6B/6C diagnosis (2026-09-26)

No V2 configuration was frozen, and no validation or test retrieval was run.
The completed 100,000-ID tune sweep is in
`artifacts/tfidf_v2_tune/tune_sweep.json`; its source-specific top-50 output
and metrics were left unchanged.

On tune labels, V1 candidate recall was 81.1602% (S2 80.1330%, S3 82.1208%),
with 12,840,689 pairs and 65,125 misses. Character TF-IDF at K=20 reached
81.8701%, recovered 2,454 V1 misses, and grew the pair set 13.38%. K=50
reached 82.3876% (S2 81.4386%, S3 83.2752%), recovered 4,243 misses, and
grew the pair set 39.49%. Mean/P95/P99/max candidate counts were
128.41/397/573/3,096 for V1, 145.59/405/576/3,096 at K20, and
179.11/423/588/3,096 at K50. The 100K retrieval took 8,826 seconds with
2,694 MiB peak RSS; the sparse index is about 907 MB. These tune results do
not establish validation or prediction performance.

The structural audit compared country strings using open-set normalized
equality. All 7,638,365 links had equal nonblank country labels (S2:
3,693,619; S3: 3,944,746); none had unequal or missing country. No S2 or S3
target ID was linked to multiple S1 entities. The observed training labels
were US and India, but the implementation must continue to accept unseen
country strings. Counts are in
`artifacts/retrieval_diagnosis/structural_audit.json`.

The K50 miss review found no blank names in remaining misses. Overlapping
heuristic flags included: 20,616 different-script pairs (33.9%), 28,732 with
normalized character ratio below 60 (47.2%), 36,568 with weak/no name-token
overlap (60.1%), 38,275 with address-token Jaccard at least 0.5 (62.9%),
37,310 with a shared numeric token (61.3%), and 8,948 with conflicting
numeric tokens (14.7%). Postal candidates were sparse (174 shared; 153
conflicting). These are heuristic, overlapping flags—not ground-truth error
classes. A seeded sample of 500 remaining misses and 100 TF-IDF-recovered
links is saved in
`artifacts/retrieval_diagnosis/tune_k50_miss_examples.tsv`. Examples include
domain-style names such as “NOF Elite LLC” / “NOFELITE.COM”, word order
changes, aliases, and cross-script Hindi/Latin names. Many missed pairs still
have useful address evidence, which name-only character retrieval cannot use.

A fast inverted-token prototype used the existing V1 SQLite postings, separate
S2/S3 and country keys, IDF-weighted shared core-name tokens, a maximum target
document frequency of 20,000, and deterministic top-K ranking. On a
label-independent 5,000-ID tune sample, V1 recall was 81.2944%; the prototype
reached 83.0488% at K20 (299 V1 misses recovered; 107,227 additional pairs)
and 83.6355% at K50 (399 recovered; 297,458 additional pairs). At K20 the
candidate count distribution was mean/median/P90/P95/P99/max =
152.89/108/320/411/589/3,096. At K50 it was
190.94/143/335/437/631/3,096. Retrieval took 67 seconds end to end,
examining 108.5 million postings, and wrote 2.84 MB of added-pair metadata.
The earlier 1,000-posting truncation variant was weaker; the reported sweep
uses all postings for selected tokens up to the 20,000-frequency safeguard.
This simple name-only route is fast, but its recall lift is modest.

The 0.5% character-DF prototype was run on the same 5,000 tune IDs. It reached
84.5450% candidate recall (S2 83.3698%, S3 85.6429%), recovering 554 V1 misses
for 345,612 additional pairs (about 624 added pairs per recovered link).
It produced 1,002,843 total pairs, with mean/median/P90/P95/P99/max =
200.57/152/351/456/629/3,096. The separate index build took 369 seconds and
used 1.70 GB; retrieval took 762 seconds (about 652 returned pairs/second)
and peaked at 3,994 MiB RSS. This improves sample recall over the existing
0.1% cutoff but is slower and increases candidate counts. It is not a
production-ready replacement.

SQLite FTS5 is available; a separate FTS index over both target sources and
their names/addresses built in 79 seconds, used 1.97 GB, and peaked at
143 MiB. The 5K BM25 query benchmark was stopped after 53 minutes under the
one-hour job limit, before producing candidate metrics. It is therefore not a
viable current query implementation. `sparse-dot-topn` 1.2.0 is installed
locally in the project venv, along with psutil 7.2.2. SciPy 1.18.1,
scikit-learn 1.9.1, RapidFuzz 3.14.6, LightGBM 4.7.0, and SQLite FTS5 are
available. These package versions are pinned in `requirements.txt`.

Approximate linear projections from measured throughput (not guarantees):
existing 0.1% TF-IDF would take about 2.45 h for 100K, 5.4 h for validation,
and 42.5 h for 1.73M test S1s. The 0.5% prototype projects to about 4.4 h,
9.8 h, and 76.7 h respectively. The bounded inverted-token script processed
5K S1s in 67 seconds, corresponding conservatively to about 22 min, 50 min,
and 6.5 h for those sizes; this end-to-end measurement includes scanning
ground truth and the V1 candidates. FTS5 query time exceeded 53 minutes for
the unfinished 5K sample, so its full-size projection is already impractical.
No full tune replacement, validation, or test retrieval was started.

Reproduce the completed audit and tune-only prototypes (from this project
directory; no command below reads validation or test IDs):

```bash
PYTHONPATH=src ../../venv/bin/python src/retrieval_diagnosis.py audit
PYTHONPATH=src ../../venv/bin/python src/retrieval_diagnosis.py misses --sample-size 500
PYTHONPATH=src ../../venv/bin/python src/retrieval_diagnosis.py prototype-a --sample-size 5000 --seed 20260925 --top-k 50 --posting-cap 20000 --output-dir artifacts/retrieval_diagnosis/prototype_a_full_postings
PYTHONPATH=src ../../venv/bin/python src/retrieval_diagnosis.py prototype-char --sample-size 5000 --seed 20260925 --max-df-ratio 0.005
```

The Phase 6B/6C recommendation is to prototype an address-word inverted
route joined with rare name-token support, then revisit character retrieval
only with a compiled sparse top-N method such as `sparse_dot_topn`. The package is installed locally in `venv` and pinned in `requirements.txt`. Phase 6D tested it on a bounded tune sample, as reported below.

### Phase 6D: address retrieval and sparse top-N (5K tune sample)

The Phase 6D prototype uses the same seed-20260925 sample of 5,000 tune S1 IDs as Phase 6B/6C. It reads training target addresses from the frozen V1 SQLite index and creates a separate, disposable address-word posting index. Tokens are selected by target-side frequency within source and open-set country; tokens appearing in more than 20,000 target records are skipped. Each query uses up to four rare address words and returns at most 20 candidates from each target source. Numeric and postal agreement rerank word-supported candidates; neither is a retrieval key by itself. No labels enter index construction or retrieval.

The address index covers 10,320,219 target records and retains 26,101,606 postings. It took 1,147.7 seconds to build and occupies 2.13 GB. The address K20 run returned 197,786 pairs for the 5K sample in 116.2 seconds. FTS5 was not used. The separate index can be reused and the frozen V1 index was not modified.

`sparse-dot-topn==1.2.0` performs character TF-IDF top-50 multiplication in batches of 64 S1 records with one worker. Its score, rank, target ID, and output order matched the prior exact sparse method on all 378,862 pairs with the 0.1% DF index and all 496,177 pairs with the 0.5% DF index. The 0.1% run took 66.1 seconds plus 2.7 seconds to sort its compressed output. The 0.5% run took 403.9 seconds plus 3.4 seconds to sort, compared with 797.3 seconds for the prior 5K method. Peak sample RSS was 4,451 MiB, below the 16 GB limit.

Tune-sample candidate coverage (17,043 true links; V1 has 657,231 pairs):

| V1 union with | Recall | S2 | S3 | V1 misses recovered | Candidate pairs | Growth vs V1 | P95 / P99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| No new route | 81.29% | 80.27% | 82.25% | 0 | 657,231 | 0% | 398 / 576 |
| Name-token K20 | 83.05% | 82.05% | 83.99% | 299 | 764,458 | 16.31% | 411 / 589 |
| Address K20 | 93.43% | 93.39% | 93.46% | 2,068 | 840,442 | 27.88% | 433 / 609 |
| Character 0.1% K50 | 82.56% | 81.41% | 83.63% | 216 | 911,653 | 38.71% | 423 / 584 |
| Character 0.5% K50 | 84.54% | 83.37% | 85.64% | 554 | 1,002,843 | 52.59% | 456 / 629 |
| Name-token K20 + address K20 | 94.16% | 94.16% | 94.17% | 2,193 | 947,031 | 44.09% | 449 / 624 |
| Name-token K20 + address K20 + character 0.1% | 94.41% | 94.31% | 94.51% | 2,236 | 1,197,056 | 82.14% | 480 / 646 |
| Name-token K20 + address K20 + character 0.5% | 94.82% | 94.72% | 94.93% | 2,306 | 1,268,936 | 93.07% | 505 / 700 |

The last union leaves 882 of the sample's 3,188 V1 misses. Address K20 recovers 2,068 V1 misses, of which 1,632 are exclusive to that route across the four measured routes. The corresponding exclusive counts are 33 for name-token K20, 10 for character 0.1%, and 80 for character 0.5%. Address overlaps with name-token on 174 recovered links and with character 0.5% on 349. These counts describe retrieval coverage, not matching quality.

Deriving smaller address K values from the saved top-20 ranks shows a useful cost curve: address K5 reaches 91.84% recall with 694,716 candidate pairs (+5.70% over V1); K10 reaches 92.81% with 742,754 (+13.01%); K15 reaches 93.25% with 791,503 (+20.43%); K20 reaches 93.43% with 840,442 (+27.88%). The shared-sample quality gate of 90% is exceeded even at K5. The index and query runtime at K5 have not been measured separately; these are rank-derived metrics.

The current single-thread implementation remains too expensive to approve for full test retrieval. Linear estimates, excluding the one-time 19.1-minute address-index build, are about 0.65/1.43/11.19 hours for address alone at 100K tune / 220,683 validation / 1,732,545 test S1s; 1.02/2.25/17.66 hours for name-token plus address; and 3.18/7.02/55.04 hours when adding character 0.5%. Character index load is counted once in those estimates. They are machine-specific projections, not measured large runs. The next experiment should optimize the address query at K5 on this same 5K tune sample, then remeasure throughput before any larger run. No full 100K replacement, validation, or test retrieval was run in Phase 6D.

The structural audit found that each linked training S2/S3 target belongs to only one S1. A later matching phase can consider target competition as a feature or constraint; this prototype does not apply a global assignment rule. The observed property should be rechecked where it matters and should not be treated as a hardcoded country or match assumption.

Reproduce Phase 6D from `code/business_entity_resolution`:

```bash
PYTHONPATH=src ../../venv/bin/python src/phase6d_retrieval.py check-sparse-topn
PYTHONPATH=src ../../venv/bin/python src/phase6d_retrieval.py build-address-index
PYTHONPATH=src ../../venv/bin/python src/phase6d_retrieval.py run-sample --sample-size 5000 --seed 20260925 --skip-index-build --max-sample-runtime 3500
PYTHONPATH=src ../../venv/bin/python src/phase6d_retrieval.py evaluate-existing
PYTHONPATH=src ../../venv/bin/python src/phase6d_retrieval.py sweep-address-k
../../venv/bin/python -m unittest discover -s tests
```

The measured reports are `artifacts/retrieval_diagnosis/phase6d/phase6d_summary.json` and `address_k_sweep.json`. The large address index and compressed pair files are reproducible, disposable, and ignored by Git.

## Phase 7: supervised pair data

Phase 7 creates a label-independent 100,000-S1 subset from the development
split, generates candidates with the frozen Phase 4 V1 routes plus the Phase 6E
Address K10 route, then labels and samples pairs against training ground truth.
All retrieved positives are retained. True links absent from retrieval are
written as blocking misses and are never treated as negative examples. S2 and S3
pair pools are separate; each S1 retains up to 12 deterministic, prioritized
negatives, including an easy-negative sample when available.

Run the stages in order from the repository root:

```bash
venv/bin/python code/business_entity_resolution/src/phase7_pairs.py prepare-subset
venv/bin/python code/business_entity_resolution/src/phase7_pairs.py generate-candidates
venv/bin/python code/business_entity_resolution/src/phase7_pairs.py build-pairs
```

The frozen `artifacts/blocking/v1_index.sqlite` and
`artifacts/retrieval_diagnosis/phase6d/address_postings.sqlite` are reused
read-only. Candidate artifacts, split IDs and reports are written under
`artifacts/model_data/phase7/`. The ID subset has a companion SHA256 file.
The candidate report records retrieval time and RSS; the sampling report
records per-source recall, blocking misses, negative categories, pair counts,
file sizes and per-S1 negative-count summaries. This phase does not calculate
model features, train a model, or read validation labels.

Measured Phase 7 results for this fixed training subset:

- The subset contains 100,000 unique S1 IDs selected with seed `20260925`; the SHA256 is in `train_subset_ids.sha256`. It has zero overlap with tune and validation IDs.
- V1 generated 12,888,365 pairs. Address K10 generated 1,985,994 pairs, giving a 14,597,845-pair union (+13.26% over V1).
- Candidate recall on these training IDs is 93.19% for S2 (156,472/167,898 links), 93.15% for S3 (167,002/179,283), and 93.17% overall (323,474/347,181). These are training-subset retrieval statistics, not validation results.
- Blocking misses total 23,707: 11,426 S2 and 12,281 S3. They are recorded separately and are not negative examples.
- The sampled datasets retain 583,055 S2 and 616,332 S3 negatives, alongside every retrieved positive. Address rank buckets, V1 exact/core/token evidence, other address-supported routes, combined V1/address evidence, and deterministic easy negatives are counted in the sampling report.
- Candidate generation took 50m30s (V1 15m28s; Address K10 35m00s) at 531 MiB peak RSS. Labeling and sampling took 69s at 242 MiB peak RSS. All Phase 7 artifacts occupy about 186 MB.
- The compressed pair datasets contain 739,527 S2 rows and 783,334 S3 rows. The negative cap is 12 per S1 across both sources when enough negatives are available.

Phase 8 pair-dataset inputs:

- `artifacts/model_data/phase7/train_pairs_s2.tsv.gz`
- `artifacts/model_data/phase7/train_pairs_s3.tsv.gz`
- `artifacts/model_data/phase7/phase7_sampling_report.json`


## Phase 8: pair features and separate LightGBM models

Phase 8 uses only the frozen Phase 7 S2/S3 pair pools to build one ordered,
label-free feature schema. It trains separate S1→S2 and S1→S3 LightGBM pair
classifiers. The model features contain normalized name/address similarity,
numeric/postal/country evidence, frozen V1 and Address K10 retrieval evidence,
and source-local candidate counts. They exclude entity IDs, labels, and
`negative_reason`.

The 100,000-ID tune candidate union is read from the frozen V1 and Address K10
artifacts and scored without filtering. Phase 8 does not choose entity-level
thresholds, singleton policy, target conflicts, validation, test predictions,
or a submission.

Run the manual stages from the workspace root in this order:

```bash
venv/bin/python code/business_entity_resolution/src/phase8_model.py features --source S2
venv/bin/python code/business_entity_resolution/src/phase8_model.py features --source S3
venv/bin/python code/business_entity_resolution/src/phase8_model.py search --source S2
venv/bin/python code/business_entity_resolution/src/phase8_model.py search --source S3
venv/bin/python code/business_entity_resolution/src/phase8_model.py retrain --source S2
venv/bin/python code/business_entity_resolution/src/phase8_model.py retrain --source S3
venv/bin/python code/business_entity_resolution/src/phase8_model.py preflight-score
venv/bin/python code/business_entity_resolution/src/phase8_model.py score-tune
venv/bin/python code/business_entity_resolution/src/phase8_model.py finalize
venv/bin/python -m pytest -q code/business_entity_resolution/tests
```

Outputs are under `artifacts/model/phase8/`: the feature manifest and compressed
feature caches, per-source internal-selection reports, final `model_s2.txt` and
`model_s3.txt`, complete unfiltered tune score files, a scoring preflight, and
the Phase 8 summary. Every command snapshots frozen Phase 4–7 inputs and fails
if one changed.
