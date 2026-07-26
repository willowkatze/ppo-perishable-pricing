# Start Here

## Project Summary

This project studies dynamic markdown decisions for perishable fresh-retail inventory. It combines FreshRetailNet-50K-informed data preparation, stockout-aware demand recovery, a semi-synthetic perishability environment, PPO experiments, a DQN comparison, and a locked held-out policy ladder.

The central question is whether a learned markdown policy can improve normalized accounting profit while managing waste on high-risk perishable inventory. The project is a controlled reinforcement-learning experiment in a semi-synthetic environment calibrated with historical retail data; it is not a deployed retail pricing system.

## Main Conclusion

On the locked 60-episode HIGH_RISK_B held-out test set, `always_0pct` was the strongest overall policy with mean normalized profit `0.447999`. `balanced_recovered_ppo` was the strongest learned policy with `0.435909`, but it did not beat `always_0pct`. The locked DQN ensemble scored `0.425721`, with paired gain `-0.022278` and bootstrap CI `[-0.027486, -0.017319]` versus `always_0pct`.

The balanced PPO redesign improved decision behavior relative to the original recovered PPO diagnostics, but the final evidence does not support the claim that learned markdowns outperform the no-markdown baseline.

## Important Files

| Question | File |
|---|---|
| Project overview | [README.md](README.md) |
| Demand recovery | [docs/02_data_and_demand_recovery.md](docs/02_data_and_demand_recovery.md) and [src/latent_demand_recovery.py](src/latent_demand_recovery.py) |
| Environment and reward | [docs/03_environment_and_methods.md](docs/03_environment_and_methods.md) and [src/pricing_env_operational.py](src/pricing_env_operational.py) |
| Original PPO | [src/train_ppo_operational.py](src/train_ppo_operational.py) |
| Balanced PPO | [src/train_ppo_recovered_redesign.py](src/train_ppo_recovered_redesign.py) |
| DQN | [src/train_dqn_high_risk_b.py](src/train_dqn_high_risk_b.py) |
| Final held-out evaluation | [src/final_test_baseline_ladder.py](src/final_test_baseline_ladder.py) |
| Main result table | [results/tables/final_heldout_baseline_ladder.csv](results/tables/final_heldout_baseline_ladder.csv) |
| Main result figure | [results/figures/heldout_policy_comparison.png](results/figures/heldout_policy_comparison.png) |
| Reproducibility | [docs/reproducibility.md](docs/reproducibility.md) |

## Reading Paths

### A. Five-Minute Overview

1. [README.md](README.md)
2. [results/figures/heldout_policy_comparison.png](results/figures/heldout_policy_comparison.png)
3. [results/tables/final_heldout_baseline_ladder.csv](results/tables/final_heldout_baseline_ladder.csv)

### B. Research and Results

1. [docs/01_project_and_research_design.md](docs/01_project_and_research_design.md)
2. [docs/02_data_and_demand_recovery.md](docs/02_data_and_demand_recovery.md)
3. [docs/03_environment_and_methods.md](docs/03_environment_and_methods.md)
4. [docs/04_results.md](docs/04_results.md)
5. [docs/05_conclusions_and_limitations.md](docs/05_conclusions_and_limitations.md)

### C. Source-Code Review

1. [scripts/README.md](scripts/README.md)
2. [docs/reproducibility.md](docs/reproducibility.md)
3. Core source files listed under `scripts/README.md`, beginning with data preparation, demand recovery, the environment, and the two learning implementations.
