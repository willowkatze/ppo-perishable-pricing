"""final reporting wrapper.

Expected inputs: locked test outputs and compatible artifacts.
Produced outputs: baseline ladder tables and figures.
Config path: configs/evaluation.
Example command: python scripts/07_generate_report_outputs.py
Execution note: evaluates existing artifacts.
"""

from runpy import run_module


if __name__ == "__main__":
    run_module("src.final_test_baseline_ladder", run_name="__main__")
