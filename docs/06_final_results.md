# 06 Final Results

## Purpose
State final held-out result and secondary ladder.

## Inputs
results/final_summary.csv and results/key_tables.

## Method
Evaluate locked policies on identical HIGH_RISK_B held-out episodes.

## Implementation
src/final_dqn_ensemble_test_evaluation.py; src/final_test_baseline_ladder.py.

## Main Results
always_0pct strongest overall; balanced_recovered_ppo strongest learned; DQN negative.

## Interpretation
Defensible claim: learned policies improve over weaker baselines, not always_0pct.

## Limitations
Secondary ladder is post-hoc descriptive.

## Related Files
results/key_figures/05_final_heldout_baseline_ladder.png

