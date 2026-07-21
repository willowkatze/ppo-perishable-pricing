"""locked model evaluation wrapper.

Expected inputs: locked test manifest and trained artifacts.
Produced outputs: held-out evaluation tables.
Config path: configs/evaluation.
Example command: python scripts/06_evaluate_models.py
Execution note: evaluates existing artifacts.
"""

from runpy import run_module


if __name__ == "__main__":
    run_module("src.final_dqn_ensemble_test_evaluation", run_name="__main__")
