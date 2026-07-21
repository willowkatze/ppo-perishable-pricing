# Results

## PPO Before Balancing

The original PPO variants did not outperform the no-markdown baseline. The recovered-demand version changed the learning signal but did not solve the final held-out performance gap.

## PPO Collapse Diagnosis

PPO diagnostics showed that policy behavior could collapse toward simple action patterns. This motivated the balanced recovered PPO experiment.

## Balanced PPO Redesign

Balanced recovered PPO was the strongest learned policy on the held-out test set. It outperformed all evaluated positive fixed-markdown policies, random_uniform, two simple rule-based policies, the original PPO variants, and the locked DQN ensemble. However, always_0pct remained the strongest overall policy.

## DQN Results

The locked DQN ensemble achieved 0.425721 mean normalized profit on the held-out test set. Its paired gain versus always_0pct was -0.022278, with bootstrap CI [-0.027486, -0.017319]. DQN did not beat always_0pct and was below balanced recovered PPO.

## Full Baseline Ladder

The full ladder contains 13 policies and is reported in `results/tables/final_heldout_baseline_ladder.csv`. The main figure is `results/figures/heldout_policy_comparison.png`.

## Held-Out Strongest Policies

The strongest overall held-out policy was always_0pct. The strongest learned held-out policy was balanced_recovered_ppo. No learned policy beat always_0pct on the held-out test set.

## Profit-Waste Results

The profit-waste comparison is retained in `results/tables/final_profit_waste_pareto_frontier.csv` and `results/figures/07_profit_waste_pareto_frontier.png`. It supports the conclusion that learned markdown behavior can reduce some weaker-baseline losses but did not improve over the no-markdown policy.

