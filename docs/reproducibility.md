# Reproducibility

## Setup

Install dependencies with `pip install -r requirements.txt` or use `environment.yml`.

## Package Versions

Dependency files are included, but exact locked hashes are not provided. This is a partial reproducibility item.

## Configs and Seeds

Configuration files are stored under `configs/`. Random seeds and selected model metadata are recorded in configs and result tables.

## Data Splits

Validation and held-out test manifests are stored under `outputs/manifests/`. The final result uses the locked 60-episode HIGH_RISK_B held-out test set.

## Execution Order

Use the numbered scripts under `scripts/`. Training scripts are separate from evaluation and reporting scripts. Do not run training unless the local raw data and model artifacts are available.

## Excluded Artifacts

Raw data and trained model binaries are intentionally not committed. `outputs/models/MODEL_ARTIFACTS.md` documents this.

## Non-Training Checks

Useful checks include `python -m compileall -q src scripts`, Markdown link validation, CSV schema validation, JSON config parsing, and a secret scan.

## Skipped Tests

Model-dependent tests may skip in the public repository because raw data and model binaries are excluded.

