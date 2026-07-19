# Security and File Size Audit

Scope: source code, documentation, configuration-like files, notebooks, scripts, and lightweight CSV/JSON/Markdown files in the local project.

## Secret Scan Summary
- No API keys, GitHub tokens, passwords, SSH private keys, cookies, or email addresses were confirmed in the scanned candidate source/documentation/config files.
- All initial matches were false positives from ordinary variable names or dataset references.
- Raw datasets and large generated data artifacts remain excluded regardless of secret-scan status.

| Path | Finding | Action |
|---|---|---|
| src\train_ppo_recovered_redesign.py | false positive: local variable named token | safe to keep |
| docs\project_progress_and_redesign.md | false positive: Kaggle dataset reference | safe to keep |
| src\freshretail_data_processing.py | false positive: date token parsing terms | safe to keep |
| src\perishability_scenarios.py | false positive: category token parsing terms | safe to keep |

## Files Above 50 MB
| Path | Size MB | Public GitHub decision |
|---|---:|---|
| data\operational\raw\freshretail\raw\freshretail_train.parquet | 1696.3 | Exclude from normal Git |
| data\operational\raw\freshretail\raw\freshretail_eval.parquet | 132.38 | Exclude from normal Git |

## Required Exclusions
- `data/operational/raw/freshretail/raw/freshretail_train.parquet` and `freshretail_eval.parquet` are raw dataset files and should not be committed.
- Legacy Kaggle raw/processed files under `archive/prototype_kaggle/data/` should not be committed to the public repository.
- Caches, notebooks checkpoints, IDE folders, temporary files, logs, and redundant checkpoints should stay ignored.

## Git LFS Recommendation
- Git LFS is optional for this project if final model binaries are excluded and documented by hash.
- If selected DQN/PPO model binaries must be archived, use Git LFS for `*.zip`, `*.pkl`, `*.joblib`, `*.pt`, and `*.pth` only after confirming they are scientifically necessary.

## Machine-Specific Path Exclusions

A follow-up path scan found absolute local Windows paths only in the legacy `archive/prototype_kaggle/` tree and `docs/machine_specific_path_audit.csv`. These files are excluded from public Git candidates through `.gitignore` and should not be staged for the public repository.
