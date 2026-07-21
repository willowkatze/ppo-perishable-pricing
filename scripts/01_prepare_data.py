"""preprocessing wrapper.

Expected inputs: local FreshRetailNet parquet files.
Produced outputs: processed subset tables.
Config path: configs/evaluation.
Example command: python scripts/01_prepare_data.py
Execution note: does not train a model.
"""

from runpy import run_module


if __name__ == "__main__":
    run_module("src.freshretail_data_processing", run_name="__main__")
