"""Prepare the FreshRetailNet modeling subset.

Purpose:
    Inspect locally downloaded FreshRetailNet parquet files and create the
    reproducible student-laptop subset used by later demand recovery and RL
    modules.
Implementation module:
    src.freshretail_data_processing
Required local inputs:
    data/operational/raw/freshretail/raw/freshretail_train.parquet
    data/operational/raw/freshretail/raw/freshretail_eval.parquet
Config files:
    None. This step writes discovered schema and subset metadata instead of
    reading an external config file.
Main outputs:
    data/freshretail/processed/freshretail_modeling_subset.parquet
    data/freshretail/processed/freshretail_pilot_subset.parquet
    outputs/tables/freshretail_*
    outputs/figures/freshretail_processing/
Trains a model:
    No.
Requires excluded local artifacts:
    Yes. Raw FreshRetailNet files are not stored in Git.
Example command:
    python scripts/01_prepare_freshretail_data.py
Runtime category:
    Moderate.
Overwrite behavior:
    Existing subset tables and processing diagnostics may be overwritten.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from src.freshretail_data_processing import main


if __name__ == "__main__":
    main()
