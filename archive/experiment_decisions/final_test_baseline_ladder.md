# Final Test Baseline Ladder

This is a post-hoc secondary descriptive analysis on the already locked 60-episode HIGH_RISK_B held-out test population. It does not retrain PPO or DQN, does not modify the locked DQN ensemble, and does not replace the primary conclusion.

Primary conclusion preserved: the locked DQN ensemble did not outperform `always_0pct` on held-out test episodes.

## Strongest Policies

- Strongest overall policy: `always_0pct`, mean normalized profit 0.447999.
- Strongest learned policy: `balanced_recovered_ppo`, mean normalized profit 0.435909.
- Locked DQN ensemble mean normalized profit: 0.425721.

## Secondary Findings

The locked DQN ensemble beat several weaker fixed, random, and rule-based baselines, including `always_5pct`, `always_10pct`, `always_20pct`, `always_30pct`, `always_40pct`, `random_uniform`, `expiry_rule_10pct`, and `inventory_coverage_rule_10pct`. It did not beat `always_0pct`.

The balanced recovered PPO was the strongest learned policy and also remained below `always_0pct`.

## Defensible Report Claim

A defensible claim is: under the locked HIGH_RISK_B held-out test, the no-markdown policy remained the strongest overall policy. Learned policies showed state-dependent behavior and outperformed several weaker markdown baselines, but the final locked DQN ensemble did not beat the primary no-markdown baseline.

## Outputs

- `outputs/tables/final_test_baseline_ladder.csv`
- `outputs/tables/final_test_model_vs_baseline_matrix.csv`
- `outputs/tables/final_test_baseline_pairwise_comparisons.csv`
- `outputs/figures/final_test_baseline_ladder/`
