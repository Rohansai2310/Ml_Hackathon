# Business Entity Resolution Challenge

This repository contains the ML Challenge 2026 entity-resolution project. The complete phase-by-phase development notes are in [`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md). The concise local-vs-GitHub artifact inventory is in [`code/business_entity_resolution/ARTIFACTS.md`](code/business_entity_resolution/ARTIFACTS.md).

## Repository map

```text
.
├── 6ab10eb3b23ba_student_resource/   Supplied challenge docs, validator, local dataset
├── code/business_entity_resolution/
│   ├── src/                          Pipeline entry points, named by phase
│   ├── tests/                        Fixture-based tests
│   ├── artifacts/                    Reports, models, indexes, score caches
│   ├── README.md                     Full phase workflow
│   └── ARTIFACTS.md                  What is on GitHub versus local-only
├── output/                           Local submission files; intentionally ignored by Git
├── venv/                             Local Python environment; ignored by Git
└── .gitignore
```

## Current deliverables

- `output/candidate_pairs.tsv`: canonical Phase 12 candidate set.
- `output/matching_results.tsv`: original prediction file (Submission #1).
- `output/phase112/matching_results.tsv`: policy-only Phase 11.2A prediction (Submission #2 candidate).
- Phase 11.2A tune and one-shot validation confirmation are recorded locally under `code/business_entity_resolution/artifacts/phase112_group_policy/`.

These outputs are local and are not pushed to GitHub. `candidate_pairs.tsv` is still required alongside the selected `matching_results.tsv` in the challenge submission package. Test labels do not exist locally.

## Environment and tests

Run from the repository root:

```bash
venv/bin/python -m pytest -q code/business_entity_resolution/tests
```

If the environment needs to be recreated:

```bash
python3 -m venv venv
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -r code/business_entity_resolution/requirements.txt
```

The extracted challenge dataset is expected at:
`6ab10eb3b23ba_student_resource/student_resource/dataset/`.
It is intentionally ignored by Git.

## Validate a submission

The supplied validator checks the output schema, coverage, prefixes, and (with `--check-ids`) whether IDs exist in the test sources. Run it against either prediction file:

```bash
venv/bin/python \
  6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir 6ab10eb3b23ba_student_resource/student_resource/dataset/test \
  --check-ids
```

For the Phase 11.2A prediction, change the matching path to `output/phase112/matching_results.tsv`. This validator is a structural check; it cannot measure hidden-test F0.5.

## Re-running completed test inference

Phase 12 is already complete. Its stages are recorded here for reproducibility, but they are expensive and write large local artifacts. Do not rerun them just to validate the existing submissions.

```bash
venv/bin/python code/business_entity_resolution/src/phase12_inference.py inventory
venv/bin/python code/business_entity_resolution/src/phase12_inference.py prepare-indices
venv/bin/python code/business_entity_resolution/src/phase12_inference.py benchmark --workers 1
venv/bin/python code/business_entity_resolution/src/phase12_inference.py benchmark --workers 2
venv/bin/python code/business_entity_resolution/src/phase12_inference.py benchmark --workers 4
venv/bin/python code/business_entity_resolution/src/phase12_inference.py compare-benchmarks
venv/bin/python code/business_entity_resolution/src/phase12_inference.py generate-candidates --workers 4
venv/bin/python code/business_entity_resolution/src/phase12_inference.py finalize-candidates
venv/bin/python code/business_entity_resolution/src/phase12_inference.py check-candidates
venv/bin/python code/business_entity_resolution/src/phase12_inference.py benchmark-score --workers 1
venv/bin/python code/business_entity_resolution/src/phase12_inference.py benchmark-score --workers 2
venv/bin/python code/business_entity_resolution/src/phase12_inference.py benchmark-score --workers 4
venv/bin/python code/business_entity_resolution/src/phase12_inference.py compare-score-benchmarks
venv/bin/python code/business_entity_resolution/src/phase12_inference.py score --workers 4
venv/bin/python code/business_entity_resolution/src/phase12_inference.py predict
venv/bin/python code/business_entity_resolution/src/phase12_inference.py export-candidates
venv/bin/python code/business_entity_resolution/src/phase12_inference.py report
```

The pipeline resumes completed candidate and score shards. `predict` writes `output/matching_results.tsv`; `export-candidates` writes `output/candidate_pairs.tsv`. Back up any file you want to preserve before deliberately rerunning those final stages.
