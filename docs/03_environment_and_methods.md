# Environment and Methods

## Environment

The markdown environment is semi-synthetic. It uses recovered demand, inventory state, remaining shelf life, markdown actions, and accounting logic to compare pricing policies under controlled assumptions.

## State, Actions, and Accounting

The state includes demand and perishability-related variables. The action space contains six markdown actions. Rewards and accounting focus on normalized profit, revenue, waste, sell-through, and related diagnostics.

## Perishability Assumptions

Inventory ages over time and remaining shelf life affects waste risk. These assumptions are documented in `configs/environment/` and implemented in `src/pricing_env_operational.py`.

## PPO and Balanced PPO

PPO is the main course-aligned reinforcement learning method. The project compares original observed-demand PPO, original recovered-demand PPO, and balanced recovered PPO. Balanced training was introduced after PPO action-collapse diagnostics.

## DQN

DQN is retained as an additional value-based comparison for the six-action space. It had a positive validation signal but did not beat always_0pct on the held-out test set.

## Baselines

The final held-out comparison includes always_0pct, five positive fixed-markdown policies, random_uniform, two simple rule-based policies, PPO variants, balanced recovered PPO, and the locked DQN ensemble.

## Train, Validation, and Test Design

Training, validation, and held-out test artifacts are separated. The main final result uses the locked 60-episode HIGH_RISK_B held-out test set. The baseline ladder is a post-hoc descriptive analysis and does not replace the primary held-out conclusion.

