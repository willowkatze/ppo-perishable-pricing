# 02 Data And Demand Recovery

## Purpose
Document FreshRetailNet-informed inputs and demand recovery.

## Inputs
results/key_tables/demand_recovery_summary.csv and archived processing tables.

## Method
Recover latent demand to reduce stockout censoring bias.

## Implementation
src/freshretail_data_processing.py; src/latent_demand_recovery.py; src/discount_response.py.

## Main Results
Demand-recovery summary is retained as a key table; detailed robustness tables are archived.

## Interpretation
Recovered demand supports pricing calibration.

## Limitations
Raw data is excluded; environment remains semi-synthetic.

## Related Files
results/key_figures/01_observed_vs_recovered_demand.png

