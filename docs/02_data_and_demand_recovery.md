# Data and Demand Recovery

## Data

The project uses FreshRetailNet-informed operational time series. Raw data is not committed to the repository. Public files retain processed summaries, source-code entry points, and configuration records.

## Modeling Subset

The modeling subset is documented through archived data-processing tables and figures. The final report should use `results/tables/demand_recovery_summary.csv` as the main table for this part of the project.

## Stockout Censoring

Observed sales can be censored during stockouts because demand above available inventory is not observed. This matters because markdown policies trained on observed sales alone can learn from a biased demand signal.

## Recovery Method

The project uses a stockout-aware recovered-demand target before downstream pricing experiments. The implementation is in `src/latent_demand_recovery.py`, supported by data preparation in `src/freshretail_data_processing.py`.

## Recovery Results

The main demand-recovery evidence is `results/tables/demand_recovery_summary.csv` and the figures `results/figures/01_observed_vs_recovered_demand.png` and `results/figures/02_recovered_demand_adjustment_distribution.png`.

## Data Limitations

Raw data and large generated artifacts are excluded from GitHub. The recovered-demand step improves the modeling signal but does not remove all uncertainty about unobserved demand.

