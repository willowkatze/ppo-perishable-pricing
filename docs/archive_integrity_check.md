# Archive Integrity Check

Date: 2026-07-20

## Commands Run
- `python --version`
- `python -m pytest tests`
- key artifact existence checks
- non-training secret/path scan
- artifact manifest and SHA-256 checksum generation

## Results
- Python version observed: 3.13.9
- Pytest result: 33 passed in 58.41 seconds
- No PPO, DQN, planning, validation, or held-out test rerun was performed.
- README, LICENSE, CITATION.cff, environment.yml, archive audits, result docs, artifact manifest, checksum manifest, and model artifact documentation exist.
- Raw FreshRetailNet parquet files exceed 50 MB and are excluded from public Git.
- Legacy `archive/prototype_kaggle/` and `docs/machine_specific_path_audit.csv` contain machine-specific paths and are excluded from public Git candidates.
- No confirmed secrets were found in candidate source/documentation/config files.

## Git Limitation
The local project directory currently does not contain `.git`, and `git` was not available in the PowerShell PATH used by Codex. Branch creation, commits, push, and pull request creation remain manual follow-up steps unless Git/GitHub Desktop is installed and the project is initialized or copied into a clone of `willowkatze/ppo-perishable-pricing`.

## Clone Archive Check

After copying into the GitHub clone and excluding raw data/model binaries, `python -m pytest tests` collected 33 tests and skipped 33 tests. This is expected for the public archive because local data and fitted model artifacts are excluded. Full integration tests require restoring the documented local artifacts.
