# Environment and Methods

## What Was Done

The project uses a semi-synthetic pricing environment to combine recovered demand with explicit perishability and accounting mechanics. It is semi-synthetic because demand-response artifacts are estimated from operational data, while shelf life, inventory age composition, initial inventory, cost ratios, and realized waste are controlled scenario assumptions.

The environment is implemented in `src/pricing_env_operational.py`. Its main scenario and reward records are preserved in `configs/environment/`.

## State Construction

State variables are grouped into five categories:

- Inventory: normalized total inventory and inventory coverage.
- Perishability: remaining-life buckets, weighted remaining life, fraction expiring today, and fraction expiring within two days.
- Demand: predicted zero-markdown demand, lagged simulated targets, rolling means, and simulated volatility.
- Time and history: episode time, day of week, weekend indicator, previous markdown, and previous stockout.
- Context: calibration mode, store encoding, and product encoding.

The state is designed to expose the inventory and demand conditions relevant to markdown decisions without using future held-out outcomes.

## Inventory and FEFO Logic

Inventory ages each period. The first-expiring-first-out rule issues sales from the oldest inventory bucket before aging the remaining units. Physical waste is the inventory in the oldest bucket that expires after sales. Episodes stop when inventory resolves or the configured horizon is reached.

Shelf life is scenario-assumed rather than observed. The master scenario record uses short-life values of 2, 3, or 5 days, medium-life values of 5, 7, or 10 days, and long-life values of 10, 14, or 21 days. Initial inventory is based on expected demand and coverage, bounded by observed demand and inventory quantiles.

## Actions and Demand Response

The action space has six markdown levels:

`0%`, `5%`, `10%`, `20%`, `30%`, and `40%`.

The response models are separate for observed and recovered calibration. By default, promotion context is derived from the selected markdown action, and predictions are clipped to configured non-negative ranges. Demand is deterministic by default in the recorded environment configuration; residual-noise bootstrap is not the default validated mode.

## Reward and Accounting

The financial reward is the raw financial step divided by initial inventory:

`normalized financial reward = raw financial step / initial inventory`

The accounting identity is:

`profit = revenue - procurement cost - disposal cost + terminal salvage`

The default terminal salvage value is zero. Procurement and disposal costs are normalized scenario assumptions. Waste is reported both as physical expired units and as waste rate. Normalized accounting profit is the primary comparison metric in the final held-out ladder.

## Learned Policies

### Original PPO

`src/train_ppo_operational.py` trains the original observed- and recovered-calibration PPO agents under the operational environment. The recorded experiment uses the PPO hyperparameters in `configs/ppo/ppo_operational_experiment_config.json`.

### Recovered PPO and Balanced PPO

The recovered PPO uses the recovered demand calibration. The balanced redesign changes only training-scenario exposure and, for the stable variant, the learning rate and entropy coefficient. It keeps the environment, reward, accounting, action space, validation manifest, and test population unchanged. `src/train_ppo_recovered_redesign.py` records checkpoint diagnostics such as unnecessary markdown, oracle regret, action entropy, and state dependence.

### DQN

`src/train_dqn_high_risk_b.py` trains a value-based comparison on the HIGH_RISK_B population. The locked ensemble averages Q-values from three STANDARD_DQN seed checkpoints and chooses the action with the highest mean Q-value. The ensemble is an additional comparison, not a replacement for the PPO experiment.

## Baselines

The final ladder includes `always_0pct`, fixed 5%, 10%, 20%, 30%, and 40% markdown policies, `random_uniform`, `expiry_rule_10pct`, and `inventory_coverage_rule_10pct`. The rules apply 10% only when their existing risk condition is met; otherwise they apply zero markdown. These policies are evaluated on the same held-out episodes as the learned policies.

## Data Split and Evaluation Design

Training uses the training split. Model and checkpoint selection use validation manifests, including the locked 22-episode HIGH_RISK_B validation population where applicable. The final test comparison uses the locked 60-episode HIGH_RISK_B manifest and is run only after model selection records are fixed. No test outcome is used to tune a model or rule threshold.

## Configurations and Related Files

- Environment: `configs/environment/perishability_master_config.json`, `configs/environment/perishability_accounting_specification.json`, `configs/environment/perishability_financial_reward_specification.json`, `configs/environment/pricing_env_operational_config.json`
- PPO: `configs/ppo/ppo_operational_experiment_config.json`
- Balanced PPO selection: `configs/ppo/ppo_redesign_selected_model.json`
- DQN lock: `configs/dqn/final_locked_dqn_ensemble.json`
- Held-out protocol: `configs/evaluation/final_test_evaluation_protocol.json`
- Environment implementation: `src/pricing_env_operational.py`
- Scenario definitions: `src/perishability_scenarios.py`
- Final comparison: `src/final_test_baseline_ladder.py`
