# Final Model Decision-Task Audit

## Purpose

This audit implements the instructor's strict success criterion: a final model must beat a strong, pre-specified baseline on a paired held-out task. This script does not use the test split and does not train any model.

## Locked High-Risk Population

Locked high-risk population: `HIGH_RISK_B`.

Definition: inventory coverage >= 1.5, fraction expiring within two days >= 0.50, and projected sell-through proxy <= 0.75, evaluated on the six pre-existing meaningful profit-waste scenarios under recovered calibration.

Selection basis: operational interpretability, inventory pressure, expiry pressure, low sell-through proxy, and existing scenario design. It was not selected based on PPO wins.

Population proxy size: 7 episodes/states proxy, 0.146 share of the representative meaningful-scenario validation population.

## Locked Baseline

Locked primary baseline for the high-risk population: `always_0pct`.

Selection rule: choose the better of the best fixed validation policy and the best rule validation policy by mean normalized validation profit; ties prefer lower waste.

## Dynamic Value Result

Phase gate status: `DYNAMIC_VALUE_REQUIRES_FULL_PAIRED_PLANNING_EVALUATION`.

The current audit uses existing validation outputs. It does not claim a final model has been confirmed. A full one-time test evaluation remains forbidden until all task definitions, baselines, algorithms, and model-selection rules are locked.

## Planning Benchmark

The planning output is labelled `limited_horizon_planning_policy` only as a diagnostic unless a full paired episode-level planning rollout is implemented. It is not a perfect oracle and is not guaranteed optimal.

## Report-Ready Language

If dynamic policy wins: "A pre-specified dynamic markdown policy outperformed the locked strong baseline on paired validation in the high-risk population, justifying final model locking before one-time test evaluation."

If only planning wins: "A limited-horizon planning benchmark indicates dynamic value exists, but learned policies have not yet captured it; additional training can be justified only within the locked high-risk task."

If only waste-constrained value exists: "Dynamic markdown is not financially superior under the unconstrained objective, but may be justified for waste-constrained operations if it improves profit at the same waste constraint."

If no dynamic policy wins: "Within the current economic assumptions and pre-specified high-risk task, strong fixed/rule baselines are sufficient; no defensible dynamic final model is currently supported."
