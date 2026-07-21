# Key Results

This folder keeps the compact result set used by the final report. Full supporting material is retained under `archive/`.

## Main Tables

- `final_summary.csv`: one-screen summary of the final conclusion.
- `key_tables/demand_recovery_summary.csv`: data and recovered-demand summary.
- `key_tables/ppo_original_vs_balanced_comparison.csv`: PPO redesign comparison.
- `key_tables/final_heldout_baseline_ladder.csv`: final held-out baseline comparison across all 13 policies: no-markdown, five positive fixed-markdown baselines, random_uniform, two rule-based baselines, original PPO variants, balanced recovered PPO, and locked DQN.
- `key_tables/dqn_heldout_test_summary.csv`: locked DQN held-out result.
- `key_tables/final_profit_waste_pareto_frontier.csv`: profit-waste comparison.

## Main Figures

- `01_observed_vs_recovered_demand.png`: observed versus recovered demand.
- `02_recovered_demand_adjustment_distribution.png`: recovered-demand adjustment scale.
- `03_original_vs_balanced_ppo_error_decomposition.png`: PPO redesign error decomposition.
- `04_ppo_action_collapse_diagnostic.png`: zero-action collapse diagnostic.
- `heldout_policy_comparison.png`: held-out profit across all 13 policies on the locked 60-episode HIGH_RISK_B test set.
- `06_learned_policies_vs_fixed_markdown.png`: learned policies versus fixed markdowns.
- `07_profit_waste_pareto_frontier.png`: profit-waste frontier.


