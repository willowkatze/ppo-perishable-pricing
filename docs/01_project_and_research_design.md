# Project and Research Design

## Problem

The project evaluates dynamic markdown policies for perishable fresh-retail inventory. It creates a controlled pipeline from operational time series to recovered demand, a perishability environment, learned policies, and a paired held-out comparison against fixed and rule-based alternatives.

The final evaluation population is the locked 60-episode HIGH_RISK_B test set. The policy ladder contains original PPO, balanced recovered PPO, a locked DQN ensemble, fixed markdown policies, random actions, and two simple rules.

## Project Scope

Observed sales are not always demand. When inventory is unavailable, sales are censored and a pricing policy trained directly on sales can learn from a biased target. Perishable inventory adds a second issue: markdowns may increase sell-through but can reduce unit margin, so a useful policy must be assessed with both sales and accounting consequences.

## Research Questions

1. Does stockout-aware demand recovery change the signal used by pricing models?
2. Can PPO learn state-dependent markdown behavior after accounting for finite shelf life and action-collapse risk?
3. Does a balanced training design improve decision behavior without changing the environment, reward, or action space?
4. Do learned policies outperform fixed, random, rule-based, and no-markdown baselines on a locked held-out population?

## Workflow

The implementation follows this sequence:

1. `src/freshretail_data_processing.py` selects split-contained, complete store-product sequences.
2. `src/latent_demand_recovery.py` estimates demand under possible stockout censoring.
3. `src/discount_response.py` estimates observed and recovered markdown-response artifacts.
4. `src/perishability_scenarios.py` defines controlled shelf-life, inventory, and cost scenarios.
5. `src/pricing_env_operational.py` simulates inventory aging, FEFO issuing, demand response, waste, and accounting.
6. `src/train_ppo_operational.py` trains the original PPO matrix.
7. `src/train_ppo_recovered_redesign.py` trains the balanced recovered-PPO variants.
8. `src/train_dqn_high_risk_b.py` trains the HIGH_RISK_B DQN candidates.
9. `src/final_test_baseline_ladder.py` evaluates the locked policies and baselines on identical held-out episodes.

The experiment separates training, validation, and held-out test populations. Model selection and diagnostic decisions use training or validation artifacts; the held-out test is reserved for the final locked comparison.

## Main Contribution

The final held-out result is negative for the primary learned-policy claim. `always_0pct` achieved mean normalized profit `0.447999`, while the strongest learned policy, `balanced_recovered_ppo`, achieved `0.435909`. The locked DQN ensemble achieved `0.425721`. Thus the learned policies were useful comparisons but did not beat the no-markdown baseline.

## Interpretation

The project compares learned and non-learned policies while keeping the strongest baseline visible. The balanced PPO experiment tests whether action imbalance was limiting learning; it does not assume that a positive markdown is always valuable. The results support a conservative financial policy under the specified semi-synthetic assumptions, not a general claim that markdowns never help.

## Limitations

The environment is semi-synthetic, the recovery target is estimated, markdown response is model-implied from observational data, and the cost and shelf-life values are controlled assumptions. The held-out population is a locked HIGH_RISK_B subset rather than a complete retailer deployment population. Raw data and trained binaries are local artifacts and are not included in Git.

## Prior Work Context

The design draws on four established areas: stockout-censored demand estimation, demand recovery under inventory constraints, markdown or dynamic pricing for perishables, and reinforcement learning for inventory or pricing control. FreshRetailNet-50K supplies the fresh-retail time-series setting. This project uses those areas as methodological context and does not claim to reproduce the dataset paper or any one prior algorithm.

## Related Files

- Overview and navigation: `README.md`, `START_HERE.md`
- Data and recovery: `docs/02_data_and_demand_recovery.md`, `src/freshretail_data_processing.py`, `src/latent_demand_recovery.py`
- Environment and methods: `docs/03_environment_and_methods.md`, `src/pricing_env_operational.py`
- Results: `docs/04_results.md`, `results/tables/final_heldout_baseline_ladder.csv`
- Reproduction details: `docs/reproducibility.md`, `scripts/README.md`
