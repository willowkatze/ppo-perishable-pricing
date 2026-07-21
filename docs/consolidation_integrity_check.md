# Consolidation Integrity Check

Generated after project-results consolidation on 2026-07-21.

## Non-Training Checks Run

- `python -m pytest tests`: 33 collected, 33 skipped because raw data and model artifacts are intentionally excluded from GitHub.
- `python -m compileall -q src scripts`: passed with no syntax errors.
- Markdown relative-link validation: 40 Markdown files checked, 0 missing links.
- Git-tracked binary/raw-data scan: no tracked `.zip`, `.pkl`, `.pt`, `.pth`, `.parquet`, or raw-data CSV files found.
- Secret scan: no credential-like secrets found. Matches on the word `token` were ordinary source-code variable names.
- Claim scan: primary files consistently state that always_0pct is strongest overall and balanced_recovered_ppo is strongest learned.

## Required Final Claims

- Strongest overall policy: always_0pct.
- Strongest learned policy: balanced_recovered_ppo.
- Final DQN test result: negative versus always_0pct.
- Secondary baseline ladder: post-hoc descriptive.
- No raw data in repository.
- No model binaries in repository.

## Repository Structure Check

- Main entry files exist: `README.md`, `START_HERE.md`, `results/README.md`.
- Primary documents exist: `docs/01_research_design.md` through `docs/07_limitations.md`, plus `docs/reproducibility.md`.
- Curated result tables exist under `results/key_tables/`.
- Curated figures exist under `results/key_figures/`.
- Intermediate tables and diagnostic figures were moved under `archive/`.
- Existing implementation modules remain in `src/` to avoid import-path churn.
- Numbered wrappers exist under `scripts/`.

## Integrity Outcome

CONSOLIDATION_INTEGRITY_CHECK_PASSED for the concise project archive. Full experiment reruns still require local data and model artifacts that are intentionally excluded from GitHub.

