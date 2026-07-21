# 04 Ppo Analysis

## Purpose
Summarize PPO as primary experimental model.

## Inputs
configs/ppo and results/key_tables/ppo_original_vs_balanced_comparison.csv.

## Method
Diagnose original PPO, action collapse, and scenario-balanced redesign.

## Implementation
src/train_ppo_operational.py; src/train_ppo_recovered_redesign.py; src/evaluate_ppo_checkpoints.py.

## Main Results
Balanced recovered PPO became strongest learned held-out policy at 0.435909.

## Interpretation
PPO improved learned behavior but did not beat always_0pct.

## Limitations
Positive learned behavior remains weaker than no markdown.

## Related Files
results/key_figures/03_original_vs_balanced_ppo_error_decomposition.png

