# PPO Experiment History

This file consolidates earlier experiment notes and keeps the order of major decisions without repeating the main result tables.

## Source: limited_horizon_planning_decision.md

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

## PPO Limitations

This is a bounded limited-horizon planner, not a perfect oracle. It uses validation only and must not be tuned against the test split.



## Source: planning_and_distillation_results.md

锘? Planning and Distillation Results

Limited-horizon planning on locked validation episodes found modest positive dynamic value in HIGH_RISK_B states. The planning evaluation justified a focused learned-policy attempt but did not itself constitute a deployable learned policy.

Planning distillation generated labels for high-risk states and supported later DQN experiments. Distillation and planning were validation-locked and did not use the held-out test split for tuning.



## Source: ppo_redesign_reporting_corrections.md

# PPO Redesign Reporting Corrections

## Audit Outcome

The controlled PPO redesign reporting audit found two reporting issues and corrected them without retraining any PPO model and without using the test split.

## Negative Metric Interpretation

The negative values were not treated as impossible true regret. The audited implementation computes `rolling_oracle_value - selected_action_value`, but the rolling oracle is a finite-horizon diagnostic with `greedy_financial` continuation and is not guaranteed to upper-bound the PPO-selected action. Therefore the metric is classified as `LIMITED_ORACLE_VALUE_GAP` and renamed to `diagnostic_oracle_value_gap`.

Signed interpretation: positive values mean the limited diagnostic oracle value exceeds the policy action value; negative values mean the policy action value exceeded this limited diagnostic benchmark under the recorded action-value protocol.

## Canonical Selected Artifact

Canonical selected model ID: `BALANCED__checkpoint_20000`.

The previous duplicate label occurred because both the explicit 20,000-step checkpoint and `final_model.zip` were labelled `BALANCED__checkpoint_20000`. Their hashes differ, so the final model is retained as a distinct evaluated record named `BALANCED__final_model_20000`. Only `BALANCED__checkpoint_20000` is marked `selected_best_policy=True`.

## Final Redesign Status

Final status: `PPO_TRAINING_REDESIGN_SUCCESSFUL`.

The redesign is reportable as a training-state balancing and decision-precision improvement experiment because validation profit remains within 0.005 of the original recovered PPO checkpoint, unnecessary markdown is materially lower, state dependence is preserved, and no complete action collapse occurs. It still does not prove that PPO outperforms the always-zero baseline.

## Exact Report Wording

"Scenario-balanced PPO training corrected a strong training-state imbalance and produced a recovered-financial policy that retained near-identical validation profit relative to the original recovered PPO checkpoint while reducing unnecessary markdown and preserving state-dependent action behavior. The oracle comparison should be reported as a signed limited-horizon diagnostic value gap, not as true nonnegative regret, because the diagnostic oracle is not a guaranteed upper bound. The result supports improved policy precision under the diagnostic protocol, but not replacement of the always-zero financial baseline."

## Changed Files

- `outputs/tables/ppo_training_redesign_results.csv`
- `outputs/tables/ppo_multimetric_checkpoint_selection.csv`
- `outputs/tables/ppo_training_redesign_report.json`
- `outputs/tables/ppo_redesign_duplicate_model_audit.csv`
- `outputs/tables/ppo_oracle_regret_definition_audit.csv`
- `outputs/tables/ppo_oracle_pairing_audit.csv`
- `outputs/configs/ppo_redesign_selected_model.json`
- `outputs/configs/ppo_redesign_selected_model_hashes.json`
- affected figures under `outputs/figures/ppo_training_redesign/`

Updated figures:
- `outputs\figures\ppo_training_redesign\diagnostic_oracle_value_gap_by_timestep.png`
- `outputs\figures\ppo_training_redesign\unnecessary_markdown_by_timestep.png`
- `outputs\figures\ppo_training_redesign\selected_action_distributions.png`
- `outputs\figures\ppo_training_redesign\selected_model_comparison_after_audit.png`
- `outputs\figures\ppo_training_redesign\multimetric_checkpoint_selection_summary.png`

## Confirmation

No PPO training was run. No training artifacts, environment mechanics, rewards, scenarios, demand models, response models, selected checkpoint files, or VecNormalize files were modified.



## Source: ppo_results.md

锘? PPO Results

PPO was useful as a benchmark and diagnostic tool, but did not become the final successful model. Financial PPO variants exposed action-collapse risks, sensitivity to recovered versus observed demand calibration, and the need for paired validation against strong conservative baselines.

Scenario-balanced PPO tested whether sparse positive-markdown opportunities were underrepresented. It produced diagnostic value, but did not support a final claim that PPO beat the no-markdown baseline.


