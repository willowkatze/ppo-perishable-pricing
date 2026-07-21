# 05 Dqn Comparison

## Purpose
Summarize DQN as value-based follow-up.

## Inputs
configs/dqn; DQN validation and held-out summary tables.

## Method
Train/evaluate DQN for discrete six-action setting.

## Implementation
src/train_dqn_high_risk_b.py; src/complete_standard_dqn_multiseed.py; src/final_dqn_ensemble_test_evaluation.py.

## Main Results
DQN 0.425721 vs always_0pct 0.447999; paired gain -0.022278; CI [-0.027486, -0.017319].

## Interpretation
DQN had positive validation signal but failed held-out generalization.

## Limitations
DQN is not the final best model.

## Related Files
results/key_tables/dqn_heldout_test_summary.csv

