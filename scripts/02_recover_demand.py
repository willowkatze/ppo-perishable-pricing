"""demand recovery wrapper.

Expected inputs: processed FreshRetailNet subset.
Produced outputs: latent demand recovery outputs.
Config path: configs/evaluation.
Example command: python scripts/02_recover_demand.py
Execution note: does not train a model.
"""

from runpy import run_module


if __name__ == "__main__":
    run_module("src.latent_demand_recovery", run_name="__main__")
