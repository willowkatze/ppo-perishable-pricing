# 01 Research Design

## Purpose
Define a controlled RL study for dynamic markdown decisions.

## Inputs
FreshRetailNet-informed series, recovered demand, HIGH_RISK_B manifests.

## Method
PPO is primary; DQN is follow-up.

## Implementation
src/freshretail_data_processing.py; src/pricing_env_operational.py; src/train_ppo_operational.py; src/train_dqn_high_risk_b.py.

## Main Results
always_0pct 0.447999; balanced recovered PPO 0.435909; DQN 0.425721.

## Interpretation
Negative result against strongest baseline with learned-policy improvement over weaker baselines.

## Limitations
Semi-synthetic environment; raw data and model binaries excluded.

## Related Files
START_HERE.md; results/final_summary.csv

