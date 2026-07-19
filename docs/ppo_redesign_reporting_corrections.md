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
