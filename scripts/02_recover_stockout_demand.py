"""Recover latent demand for stockout-affected FreshRetailNet observations.

Purpose:
    Fit the stockout-aware ExtraTrees recovery workflow and save the recovered
    demand table used by the environment.
Implementation module:
    src.latent_demand_recovery
Required local inputs:
    data/freshretail/processed/freshretail_modeling_subset.parquet
Config files:
    None. Model choices and validation rules are explicit constants in the
    implementation module.
Main outputs:
    data/freshretail/processed/freshretail_demand_recovered.parquet
    outputs/models/freshretail_latent_demand_model.joblib
    outputs/tables/latent_demand_*
    outputs/figures/latent_demand_recovery/
Trains a model:
    Yes, a supervised demand-recovery model; not an RL policy.
Requires excluded local artifacts:
    Yes. The processed FreshRetailNet subset is local.
Example command:
    python scripts/02_recover_stockout_demand.py
Runtime category:
    Moderate.
Overwrite behavior:
    Existing recovered-demand outputs may be overwritten.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from src.latent_demand_recovery import main


if __name__ == "__main__":
    main()
