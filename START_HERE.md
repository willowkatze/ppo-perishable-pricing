# Start Here

## Project in one paragraph
This repository studies dynamic markdown decisions for perishable fresh-retail inventory using FreshRetailNet-informed operational time series, stockout-aware demand recovery, and a semi-synthetic perishability and financial environment. PPO is the main experimental and diagnostic model; DQN is a value-based follow-up comparison for the six-action markdown space. The final held-out result is conservative: balanced recovered PPO is the strongest learned policy, but the strongest overall policy is always_0pct.

## Research question
Can learned markdown policies improve decision quality in high-risk perishable-inventory states, and do they outperform strong fixed operational baselines on locked held-out episodes?

## Dataset
The project uses FreshRetailNet-informed operational time series with stockout-aware demand recovery. The perishability and financial environment is semi-synthetic. Raw data is not stored in this repository.

## What was implemented
Preprocessing, latent demand recovery, discount-response estimation, a perishability pricing environment, PPO training and redesign, limited-horizon planning diagnostics, DQN follow-up experiments, and locked held-out evaluation.

## Main models
PPO is the primary experimental model for calibration sensitivity, action-collapse diagnosis, and scenario-balanced redesign. DQN is a value-based follow-up model suitable for the discrete six-action setting.

## Main held-out findings
| Policy | Role | Mean normalized profit | Gain vs always_0pct | Interpretation |
|---|---|---:|---:|---|
| always_0pct | strongest overall policy | 0.4479993306199982 | 0.000000 | Best held-out profit. |
| balanced_recovered_ppo | strongest learned policy | 0.4359085234475656 | -0.012090807172432544 | Beats weaker baselines and learned alternatives, not always_0pct. |
| locked_dqn_ensemble | value-based follow-up | 0.42572130058655044 | -0.022278030033447582 | Did not generalize beyond always_0pct. |


## Strongest overall policy
always_0pct was the strongest overall policy on the locked held-out HIGH_RISK_B test population.

## Strongest learned policy
balanced_recovered_ppo was the strongest learned policy. It beat all evaluated positive fixed-markdown policies, random pricing, two simple rule-based strategies, original PPO variants, and the DQN ensemble. It did not beat always_0pct.

## What the project demonstrates
The project demonstrates meaningful policy learning and model improvement, but not overall superiority over the strongest operational baseline.

## Recommended reading order
1. [README.md](README.md)
2. [docs/01_research_design.md](docs/01_research_design.md)
3. [docs/04_ppo_analysis.md](docs/04_ppo_analysis.md)
4. [docs/05_dqn_comparison.md](docs/05_dqn_comparison.md)
5. [docs/06_final_results.md](docs/06_final_results.md)
6. [results/final_summary.csv](results/final_summary.csv)
7. [results/key_figures/](results/key_figures/)

## Reproduction entry points
Use the numbered wrappers in [scripts/README.md](scripts/README.md). Training scripts are explicit and optional.
