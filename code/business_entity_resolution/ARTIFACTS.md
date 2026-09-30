# Artifact and Storage Guide

This project intentionally keeps large inputs and generated run products on the workstation. GitHub contains source, tests, selected frozen models/manifests, and compact reports; it is not a backup of the complete local run.

## Tracked on GitHub (`main`)

These are committed and available from the repository:

- All pipeline source and fixture tests under `src/` and `tests/`.
- Challenge documentation and the supplied submission validator (the dataset itself is excluded).
- Selected compact Phase 4–7 reports, route diagnostics, and split ID lists.
- Phase 8 model files, feature manifest, model-selection reports, and selected compact training/tune artifacts.
- Phase 9 `decision_policy.json`, summary, and threshold sweep.
- Compact validation reports and other small reports enumerated by `git ls-files`.

The exact committed file list is authoritative:

```bash
git ls-files code/business_entity_resolution/artifacts
```

## Local-only files (ignored by Git)

The following are present locally but intentionally not pushed:

| Local path | Current approximate size | What it contains |
|---|---:|---|
| `6ab10eb3b23ba_student_resource/student_resource/dataset/` | 2.4 GB | Challenge train/test TSVs and training truth |
| `venv/` | 394 MB | Python virtual environment |
| `code/business_entity_resolution/artifacts/test_inference/phase12/indexes/` | 9.8 GB | Test-specific V1 and Address retrieval indexes |
| `code/business_entity_resolution/artifacts/blocking/` | 8.0 GB | Frozen V1 blocking index and validation candidate files |
| `code/business_entity_resolution/artifacts/test_inference/phase12/candidates/` | 4.1 GB | Canonical candidate pairs, generation shards, and manifests |
| `code/business_entity_resolution/artifacts/test_inference/phase12/scores/` | 3.5 GB | Phase 12 score shards and manifest |
| `code/business_entity_resolution/artifacts/retrieval_diagnosis/` | 2.1 GB | Retrieval prototype indexes and diagnostics |
| `code/business_entity_resolution/artifacts/validation_evaluation/` | 409 MB | Validation retrieval and score files/reports |
| `code/business_entity_resolution/artifacts/phase11_experiments/` | 630 MB | Local experimental models, scores, and manifests |
| `code/business_entity_resolution/artifacts/phase10_error_analysis/` | <1 MB | Tune/validation diagnostics and compact examples |
| `code/business_entity_resolution/artifacts/phase112_group_policy/` | <1 MB | Locked policy, confirmation, comparisons, and test policy summary |
| `output/` | 3.9 GB | Candidate export and prediction TSVs |

Sizes are a snapshot from 2026-09-27 and may change. Check current use with:

```bash
du -h -d 2 code/business_entity_resolution/artifacts | sort -h | tail -n 30
du -sh output 6ab10eb3b23ba_student_resource/student_resource/dataset venv .git
```

## Artifact status by phase folder

Counts below mean **tracked files / local-only files currently present**, not total files ever produced. Local-only files include ignored generated output. Snapshot taken 2026-09-27.

| Artifact folder | GitHub / local-only file counts | Notes |
|---|---:|---|
| `baseline/` | 5 / 11 | Compact reports tracked; generated pair scores and logs local |
| `blocking/` | 3 / 4 | Reports tracked; SQLite index and validation candidates local |
| `diagnostics/` | 6 / 3 | Small diagnostic reports tracked; generated miss lists local |
| `eda/` | 1 / 0 | EDA sample tracked |
| `model/` | 22 / 3 | Phase 8 models/manifests and selected files tracked; Phase 9 score caches local |
| `model_data/` | 5 / 7 | Reports/subset IDs tracked; large pair pools local |
| `retrieval_diagnosis/` | 13 / 12 | Compact reports tracked; indexes and candidate files local |
| `splits/` | 3 / 0 | Split IDs tracked |
| `validation_evaluation/` | 4 / 4 | Compact reports tracked; large validation scores/predictions local |
| `phase10_error_analysis/` | 0 / 40 | All local and ignored |
| `phase11_experiments/` | 0 / 35 | All local and ignored |
| `phase112_group_policy/` | 0 / 59 | All local and ignored |
| `test_inference/` | 0 / 406 | All Phase 12 indexes, candidate shards, score shards, and summaries local |
| `tfidf_v2_benchmark/` | 0 / 7 | Local benchmark outputs |
| `tfidf_v2_tune/` | 0 / 3 | Local tune outputs |

`output/` is ignored entirely. Therefore the three files listed below are on this machine only, not in GitHub:

- `output/candidate_pairs.tsv` — candidate file to include in the challenge submission package.
- `output/matching_results.tsv` — original prediction file.
- `output/phase112/matching_results.tsv` — Phase 11.2A policy-only prediction.

## Check tracking before committing

```bash
git status --short
git check-ignore -v output/candidate_pairs.tsv
# Show any artifact file that Git would stage (tracked files only):
git ls-files code/business_entity_resolution/artifacts
```

Do not use `git add -f` for datasets, score shards, indexes, candidate outputs, or submission TSVs. The local `.git` object database is large because of repository history; it is not a generated artifact folder, and this guide does not rewrite or clean Git history.
