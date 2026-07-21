# Final Content Selection

## Main Report Tables

| item | why included | question answered | exact source | use |
|---|---|---|---|---|
| Data and demand-recovery summary | Shows the data-processing basis for later modeling. | How was stockout-censored demand handled? | results/key_tables/demand_recovery_summary.csv | report and presentation |
| PPO results | Summarizes original versus balanced PPO behavior. | Did balanced training improve learned policy behavior? | results/key_tables/ppo_original_vs_balanced_comparison.csv | report and presentation |
| Final baseline comparison | Main held-out comparison across baselines and learned policies. | Which policy performed best on held-out test? | results/key_tables/final_heldout_baseline_ladder.csv | report and presentation |
| DQN held-out result | Documents the value-based follow-up result. | Did DQN generalize on held-out test? | results/key_tables/dqn_heldout_test_summary.csv | report |
| Profit-waste comparison | Shows the tradeoff behind the final interpretation. | What is the profit-waste frontier? | results/key_tables/final_profit_waste_pareto_frontier.csv | report |

## Main Report Figures

| item | why included | question answered | exact source | use |
|---|---|---|---|---|
| Observed versus recovered demand | Explains why demand recovery is needed. | How did recovered demand differ from observed demand? | results/key_figures/01_observed_vs_recovered_demand.png | report and presentation |
| Demand adjustment distribution | Shows recovery scale. | How large were the adjustments? | results/key_figures/02_recovered_demand_adjustment_distribution.png | report |
| PPO behavior comparison | Shows redesign effect. | How did balanced PPO change decision errors? | results/key_figures/03_original_vs_balanced_ppo_error_decomposition.png | report and presentation |
| PPO action-collapse diagnosis | Explains the original PPO limitation. | Did PPO collapse to no markdown? | results/key_figures/04_ppo_action_collapse_diagnostic.png | report |
| Held-out baseline comparison | Main final result. | Which policy had highest held-out profit? | results/key_figures/05_final_heldout_baseline_ladder.png | report and presentation |
| Learned versus fixed markdowns | Supports the secondary claim. | Which weaker baselines did learned policies beat? | results/key_figures/06_learned_policies_vs_fixed_markdown.png | report |
| Profit-waste frontier | Shows tradeoff and limitation. | What was gained or lost in profit-waste terms? | results/key_figures/07_profit_waste_pareto_frontier.png | report and presentation |

The presentation should use at most five figures: demand recovery, PPO behavior comparison, held-out baseline comparison, profit-waste frontier, and one workflow slide created in the presentation itself.
