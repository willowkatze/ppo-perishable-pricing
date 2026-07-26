# Data and Demand Recovery

## Dataset

FreshRetailNet-50K was used as the operational data source for a smaller, split-contained modeling subset. The public dataset is described at a high level as approximately 50,000 store-product series covering approximately three months (approximately 90 days), from 898 stores, 865 perishable SKUs, and 18 major cities, with separate train and evaluation parquet files. Raw files are not committed; expected local paths are listed in `data/raw/README.md`.

## Modeling Subset

The final modeling subset contains 29,100 rows from 300 complete store-product sequences, covering 232 stores and 177 SKUs. Every selected complete sequence contains exactly 97 daily observations from 2024-03-28 through 2024-07-02. The subset is the input to demand recovery and downstream response modeling; it is not the full public dataset.

## Stockout Censoring

Sales are censored when available inventory is insufficient. A recorded zero or low sales value can therefore mean low demand, or demand that could not be fulfilled. A markdown policy that treats all observed sales as demand may understate the opportunity to sell and may misrepresent the relationship between markdown and demand.

The pipeline identifies observations compatible with stockout censoring using the available stock, sales, inventory, and time-series fields. The censoring label is used to define a recovery target. This does not reveal the true unobserved demand; it identifies rows where observed sales are an incomplete signal.

## Recovery Model

`src/latent_demand_recovery.py` fits an ExtraTrees recovery model using the prepared modeling subset. The recovered target is constrained so recovered demand is never below observed sales. The fitted model and validation outputs are written locally under `outputs/models/` and `outputs/tables/` when the recovery script is run.

The recovery step also produces segment summaries, robustness checks, and a promotion-stockout interaction table. These outputs make it possible to inspect how the adjustment varies across markdown and operational segments before the pricing environment is used.

## Recovery Results

The selected recovery summary reports:

- Mean observed sales: `1.0763`.
- Mean recovered demand: `1.2196`.
- Aggregate recovered-demand increase: `13.317%`.
- Share of rows adjusted: `40.196%`.
- Recovered demand never below observed sales by construction.

The increase is a model estimate, not a measurement of true latent demand. It indicates how the downstream experiment changes when possible stockout censoring is represented.

## Interpretation and Limits

Demand recovery supplies an alternative calibration signal for the pricing environment. It should be interpreted as a proxy that is useful for controlled sensitivity analysis. It cannot establish what customers would have bought if inventory had been available, and it does not by itself identify a causal markdown effect.

The raw source data is not in the repository, the subset is much smaller than the public dataset, and the recovery model depends on the available operational features and split design. Generalization beyond the documented store-product subset is not established. Markdown response remains observational and model-implied in later stages.

## Related Files

- Data source and local paths: `data/raw/README.md`
- Subset preparation: `src/freshretail_data_processing.py`
- Recovery implementation: `src/latent_demand_recovery.py`
- Recovery summary: `results/tables/demand_recovery_summary.csv`
- Observed versus recovered demand: `results/figures/01_observed_vs_recovered_demand.png`
- Adjustment distribution: `results/figures/02_recovered_demand_adjustment_distribution.png`
- Local detailed outputs: `outputs/tables/latent_demand_*` and `outputs/models/freshretail_latent_demand_model.joblib`
