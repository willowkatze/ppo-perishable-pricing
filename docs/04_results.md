# Results

## What Was Evaluated

The final comparison uses the identical locked 60-episode HIGH_RISK_B held-out test set. The full ladder has 13 policies: always 0%, 5%, 10%, 20%, 30%, and 40% markdown; random uniform; two rule-based policies; original observed PPO; original recovered PPO; balanced recovered PPO; and the locked DQN ensemble.

The primary metric is mean normalized accounting profit. Paired gains are computed episode by episode against `always_0pct`, with episode-level bootstrap intervals in the locked evaluation records.

## PPO Before and After Balancing

The original recovered PPO diagnostics showed an unnecessary-markdown rate of `0.625` and action entropy of `0.3268` in the comparison used to motivate redesign. After scenario-balanced training, the selected 20k balanced checkpoint had unnecessary markdown `0.1948` and action entropy `0.3904`. This indicates improved decision behavior under the validation diagnostics, but it does not imply superior held-out profit.

The balanced PPO redesign is therefore a behavioral improvement experiment, not a guaranteed financial improvement. The final held-out balanced PPO result is `0.435909`, below `always_0pct`.

## Held-Out Policy Ladder

| Policy | Mean normalized profit | Gain vs always_0pct | Waste rate | Interpretation |
|---|---:|---:|---:|---|
| `always_0pct` | 0.447999 | 0.000000 | 0.001905 | Strongest overall policy. |
| `always_5pct` | 0.398558 | -0.049441 | 0.001442 | Fixed markdown below no markdown. |
| `always_10pct` | 0.348732 | -0.099267 | 0.001334 | Fixed markdown below no markdown. |
| `always_20pct` | 0.250000 | -0.197999 | 0.000000 | Fixed markdown below no markdown. |
| `always_30pct` | 0.150000 | -0.297999 | 0.000000 | Fixed markdown below no markdown. |
| `always_40pct` | 0.050000 | -0.397999 | 0.000000 | Fixed markdown below no markdown. |
| `random_uniform` | 0.240322 | -0.207677 | 0.000610 | Random baseline below no markdown. |
| `expiry_rule_10pct` | 0.348732 | -0.099267 | 0.001334 | Rule baseline below no markdown. |
| `inventory_coverage_rule_10pct` | 0.392620 | -0.055379 | 0.001489 | Rule baseline below no markdown. |
| `original_observed_ppo` | 0.307872 | -0.140127 | 0.000405 | Original learned policy below balanced PPO. |
| `original_recovered_ppo` | 0.310856 | -0.137143 | 0.000405 | Original learned policy below balanced PPO. |
| `balanced_recovered_ppo` | 0.435909 | -0.012091 | 0.001643 | Strongest learned policy, below no markdown. |
| `locked_dqn_ensemble` | 0.425721 | -0.022278 | 0.001701 | Learned value-based comparison below no markdown. |

All values in this table are from `results/tables/final_heldout_baseline_ladder.csv`; each policy has 60 held-out episodes.

## DQN Result

The locked DQN ensemble achieved mean normalized profit `0.425721`. Its paired difference versus `always_0pct` was `-0.022278`, with bootstrap 95% CI `[-0.027486, -0.017319]`. The interval is below zero, so the primary held-out conclusion is that the locked DQN ensemble did not outperform `always_0pct`.

## Interpretation

The strongest overall policy is `always_0pct`. The strongest learned policy is `balanced_recovered_ppo`. Balanced PPO beat every evaluated positive fixed-markdown policy, `random_uniform`, both simple rules, the original PPO variants, and the locked DQN ensemble, but it did not beat `always_0pct`.

The full ladder is a post-hoc descriptive extension of the primary evaluation. It should not be used to replace the primary conclusion or to claim that a learned policy is superior to the strongest baseline.

## Profit and Waste

The selected profit-waste table and figure show that policies can occupy different points on the profit-waste tradeoff. A lower waste rate alone is not enough to establish a better policy because markdown can also reduce normalized accounting profit. The relevant files are `results/tables/final_profit_waste_pareto_frontier.csv` and `results/figures/07_profit_waste_pareto_frontier.png`.

## Related Files

- Main table: `results/tables/final_heldout_baseline_ladder.csv`
- Main figure: `results/figures/heldout_policy_comparison.png`
- PPO diagnostics: `results/tables/ppo_original_vs_balanced_comparison.csv`, `results/figures/03_original_vs_balanced_ppo_error_decomposition.png`, `results/figures/04_ppo_action_collapse_diagnostic.png`
- DQN summary: `results/tables/dqn_heldout_test_summary.csv`
- Evaluation implementation: `src/final_test_baseline_ladder.py`, `src/final_dqn_ensemble_test_evaluation.py`
