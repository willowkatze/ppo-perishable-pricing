# Results

This folder contains the selected tables and figures used to interpret the project. The files are report artifacts copied from the larger local `outputs/` and `archive/` history; they do not contain raw data or trained model binaries.

## Essential Results

- `tables/final_heldout_baseline_ladder.csv`: answers which of the 13 evaluated policies has the highest mean normalized profit on the locked held-out test set.
- `figures/heldout_policy_comparison.png`: visualizes held-out profit for all learned and baseline policies.
- `tables/ppo_original_vs_balanced_comparison.csv`: shows how balanced PPO checkpoints changed validation profit, action behavior, and decision diagnostics.
- `figures/04_ppo_action_collapse_diagnostic.png`: shows the action-collapse pattern that motivated the balanced PPO redesign.

## Supporting Results

### Tables

- `tables/demand_recovery_summary.csv`: quantifies the change from observed sales to recovered demand.
- `tables/dqn_heldout_test_summary.csv`: records the locked DQN ensemble held-out result and paired comparison against `always_0pct`.
- `tables/final_profit_waste_pareto_frontier.csv`: identifies policies that are non-dominated across normalized profit and waste rate.

### Figures

- `figures/01_observed_vs_recovered_demand.png`: compares observed sales with recovered demand.
- `figures/02_recovered_demand_adjustment_distribution.png`: shows the distribution of recovery adjustments.
- `figures/03_original_vs_balanced_ppo_error_decomposition.png`: compares original and balanced PPO decision-error components.
- `figures/06_learned_policies_vs_fixed_markdown.png`: compares learned policies with fixed markdown policies.
- `figures/07_profit_waste_pareto_frontier.png`: plots the profit-waste tradeoff and its frontier.

The complete local execution history remains under `outputs/` and `archive/`; those directories are useful for provenance but are not required for the short results reading path.
