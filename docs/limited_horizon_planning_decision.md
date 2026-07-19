# Limited-Horizon Planning Decision

## Locked Decision Task

Population: `HIGH_RISK_B`.
Split: validation only.
Calibration: recovered-demand calibration.
Locked baseline: `always_0pct`.

## Planning Method

Policy name: `limited_horizon_planning_policy`.
Planning horizon: 3 decision steps.
Branching factor: 6 markdown actions.
Maximum candidate sequences per decision: 216.
Continuation assumption: deterministic finite lookahead using the existing environment and reward/accounting mechanics.
Terminal value handling: no learned terminal value beyond simulated rewards within the bounded horizon.
Optimality: not a perfect oracle and not guaranteed globally optimal.

## Primary Result

Status: `DYNAMIC_VALUE_MODEST`.
PPO retraining status: `PPO_RETRAINING_JUSTIFIED`.

Mean paired normalized-profit gain: 0.010792.
95% bootstrap CI: [-0.025571, 0.039094].
Win/tie/loss: 0.818 / 0.091 / 0.091.
Waste-rate difference: -0.138921.
Sell-through difference: 0.138921.

Best scenario: `core_003` with mean gain 0.039615.
Worst scenario: `core_009` with mean gain -0.218162.

Runtime: 684.8 seconds.

## Interpretation

If status is `DYNAMIC_VALUE_CONFIRMED`, the locked high-risk task contains realizable dynamic value under the current economic assumptions, and further RL training can be justified without using the test split.

If status is `DYNAMIC_VALUE_MODEST`, dynamic value may exist but uncertainty remains; further RL work should be cautious and validation-locked.

If status is `NO_DEFENSIBLE_DYNAMIC_MODEL`, the current environment and locked task do not support a defensible claim that dynamic markdown beats the strong baseline.

## Limitations

This is a bounded limited-horizon planner, not a perfect oracle. It uses validation only and must not be tuned against the test split.
