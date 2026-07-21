# Reproducibility

## Purpose
Explain review and rerun process.

## Inputs
Local data/model artifacts excluded from GitHub.

## Method
Use numbered scripts; training is explicit and optional.

## Implementation
pip install -r requirements.txt; python -m pytest tests.

## Main Results
Final summary and key figures can be inspected without rerunning experiments.

## Interpretation
Archive supports review and partial reproduction.

## Limitations
Skipped tests mean missing excluded artifacts, not full reproduction.

## Related Files
scripts/README.md; pyproject.toml

