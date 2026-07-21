# 06 Final Results

## Purpose
State the final locked held-out result without omitting weaker or stronger baselines.

## Inputs
The final result uses `results/final_summary.csv`, `results/key_tables/final_heldout_baseline_ladder.csv`, `results/key_tables/dqn_heldout_test_summary.csv`, and `results/key_tables/final_profit_waste_pareto_frontier.csv`.

## Method
All final policies were evaluated on the same locked 60-episode HIGH_RISK_B held-out test set. The full baseline ladder includes always_0pct, five positive fixed-markdown baselines, random_uniform, two simple rule-based policies, two original PPO variants, balanced recovered PPO, and the locked DQN ensemble.

## Implementation
The relevant source code is `src/final_test_baseline_ladder.py` and `src/final_dqn_ensemble_test_evaluation.py`. The locked test manifest is `outputs/manifests/high_risk_b_final_test_manifest.csv`.

## Main Results
always_0pct remained the strongest overall policy with 0.447999 mean normalized profit. Balanced recovered PPO was the strongest learned policy on the held-out test set. It outperformed all evaluated positive fixed-markdown policies, random_uniform, two simple rule-based policies, the original PPO variants, and the locked DQN ensemble. However, always_0pct remained the strongest overall policy. The locked DQN ensemble achieved 0.425721 mean normalized profit and a paired gain of -0.022278 versus always_0pct.

## Interpretation
The project supports a precise result: balanced PPO improved over the evaluated learned and positive-markdown alternatives, but no learned policy beat the no-markdown baseline on the locked held-out test set.

## Limitations
The secondary baseline ladder is post-hoc descriptive. It should not be shortened to "PPO beat the baselines" because balanced PPO did not beat always_0pct.

## Related Files
`results/key_tables/final_heldout_baseline_ladder.csv`, `results/key_figures/heldout_policy_comparison.png`, `results/key_tables/dqn_heldout_test_summary.csv`, `results/key_tables/final_profit_waste_pareto_frontier.csv`.
