# 04 PPO Analysis

## Purpose
Summarize PPO as the main course-aligned reinforcement learning model in the project.

## Inputs
The PPO analysis uses `configs/ppo/`, `results/key_tables/ppo_original_vs_balanced_comparison.csv`, and the locked held-out comparison in `results/key_tables/final_heldout_baseline_ladder.csv`.

## Method
The project first evaluated original observed-demand and recovered-demand PPO policies, then diagnosed action collapse and trained a scenario-balanced recovered PPO variant. The held-out baseline ladder compares the resulting PPO policies with fixed-markdown, random, rule-based, and DQN alternatives.

## Implementation
Relevant code includes `src/train_ppo_operational.py`, `src/train_ppo_recovered_redesign.py`, `src/evaluate_ppo_checkpoints.py`, and `src/final_test_baseline_ladder.py`.

## Main Results
Balanced recovered PPO was the strongest learned policy on the held-out test set. It outperformed all evaluated positive fixed-markdown policies, random_uniform, two simple rule-based policies, the original PPO variants, and the locked DQN ensemble. However, always_0pct remained the strongest overall policy.

## Interpretation
Balanced training improved the learned PPO policy relative to the original PPO variants and weaker baselines. It did not prove that dynamic markdowns were better than the no-markdown baseline on the locked held-out test set.

## Limitations
The strongest learned policy still had lower held-out mean normalized profit than always_0pct. The baseline ladder is a post-hoc descriptive comparison on the locked 60-episode HIGH_RISK_B test set.

## Related Files
`results/key_tables/ppo_original_vs_balanced_comparison.csv`, `results/key_tables/final_heldout_baseline_ladder.csv`, `results/key_figures/03_original_vs_balanced_ppo_error_decomposition.png`, `results/key_figures/heldout_policy_comparison.png`.
