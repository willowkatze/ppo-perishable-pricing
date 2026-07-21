# Planning and DQN History

This file consolidates earlier experiment notes and keeps the order of major decisions without repeating the main result tables.

## Source: dqn_final_statistical_decision.md

# DQN Final Statistical Decision

Final validation-only status: `DQN_POSITIVE_SIGNAL_NOT_STATISTICALLY_CONFIRMED`.

The original pooled analysis used seed-episode rows and is classified as
`PSEUDOREPLICATION_RISK` because
66 rows come from 22 unique validation episodes evaluated across three seeds.

Corrected clustered inference:

```json
[
  {
    "analysis": "EPISODE_MEAN_ACROSS_SEEDS",
    "sampling_unit": "episode_id",
    "unique_episodes": 22,
    "seed_count": 3,
    "mean_gain": 0.011421570610163885,
    "median_gain": 0.0012842227089391665,
    "ci_low": -0.0067716918642228945,
    "ci_high": 0.02966195141588394,
    "win_share": 0.5454545454545454,
    "tie_share": 0.0,
    "loss_share": 0.45454545454545453,
    "worst_decile_gain": -0.04917765830282653,
    "maximum_loss": -0.057763297171083895,
    "bootstrap_resamples": 10000
  },
  {
    "analysis": "CLUSTER_BOOTSTRAP",
    "sampling_unit": "episode_cluster_with_all_seed_rows",
    "unique_episodes": 22,
    "seed_count": 3,
    "mean_gain": 0.011421570610163884,
    "median_gain": 0.00173711580581,
    "ci_low": -0.0065982773478293144,
    "ci_high": 0.02973197828705219,
    "win_share": 0.5454545454545454,
    "tie_share": 0.0,
    "loss_share": 0.45454545454545453,
    "worst_decile_gain": -0.04917765830282653,
    "maximum_loss": -0.057763297171083895,
    "bootstrap_resamples": 10000
  }
]
```

Equal-weight Q ensemble validation:

```json
{
  "policy_id": "STANDARD_DQN_EQUAL_WEIGHT_Q_ENSEMBLE",
  "mean_normalized_profit": 0.1680285963578003,
  "raw_accounting_profit": NaN,
  "revenue": 1.8920765322505786,
  "waste_rate": 0.15341012655972258,
  "sell_through": 0.8465898734402775,
  "average_markdown": NaN,
  "action_entropy": 1.1726840204816535,
  "zero_action_share": 0.40540540540540543,
  "positive_action_share": 0.5945945945945946,
  "dominant_action_share": 0.40540540540540543,
  "state_dependence": "STATE_DEPENDENT",
  "collapse_flag": false,
  "accounting_valid": true,
  "runtime_seconds": 1.5931281000375748,
  "paired_episode_count": 22,
  "paired_mean_gain_vs_always_0pct": 0.014947527803537659,
  "paired_median_gain": 0.0036454153652369925,
  "episode_cluster_ci_low": -0.0015693822263161874,
  "episode_cluster_ci_high": 0.031844517484376104,
  "win_share": 0.5454545454545454,
  "tie_share": 0.09090909090909091,
  "loss_share": 0.36363636363636365,
  "waste_rate_difference": -0.12936980815050336,
  "sell_through_difference": 0.1293698081505034,
  "worst_decile_gain": -0.038022375642763104,
  "maximum_episode_loss": -0.04447743451773417,
  "action_0_share": 0.40540540540540543,
  "action_1_share": 0.24324324324324326,
  "action_2_share": 0.02702702702702703,
  "action_3_share": 0.0,
  "action_4_share": 0.0,
  "action_5_share": 0.32432432432432434
}
```

No test split was used. A final test evaluation is justified only if the
unchanged validation-lock eligibility criteria are satisfied.



## Source: dqn_results.md

锘? DQN Results

DQN was trained in a focused HIGH_RISK_B task using multiple random seeds. Validation suggested a positive but uncertain signal. A statistical-integrity audit found pseudoreplication risk in pooled seed-episode rows, so inference was corrected using episode-clustered summaries.

The final validation candidate was an equal-weight Q-value ensemble of selected DQN checkpoints. It was locked before the held-out test evaluation.



## Source: final_candidate_model_selection.md

# Final Candidate Model Selection

## Locked Validation Task

Population: `HIGH_RISK_B`.
Calibration: recovered calibration.
Locked baseline: `always_0pct`.
Test split used: false.

## Expanded Planning Result

Paired validation episodes: 22.
Mean paired planning gain: 0.010792.
Bootstrap 95% CI: [-0.025571, 0.039094].
Win/tie/loss: 0.818 / 0.091 / 0.091.
Waste-rate difference: -0.138921.

## Candidate Model Status

Final validation-only status: `PLANNING_VALUE_EXISTS_BUT_NOT_LEARNED`.

This script locks the expanded validation task and planning result. It does not
silently train long-running PPO/DQN jobs. If the expanded planning result remains
positive, separate explicit trainers can be created for GATED_PPO, GATED_DQN,
and PLANNING_DISTILLED_POLICY using the locked manifest.

## Report-Ready Wording

The project identified a validation-locked high-risk decision population and
evaluated a limited-horizon planning benchmark against the locked always-zero
baseline without using the test split. If learned candidates are not yet trained,
the correct claim is that planning value exists or does not exist on validation,
not that a final learned model beats the baseline.



## Source: final_dqn_ensemble_test_decision.md

# Final DQN Ensemble Test Decision

Final status: `FINAL_TEST_EVALUATION_REQUIRES_REVISION`.

Locked model: `STANDARD_DQN_EQUAL_WEIGHT_Q_ENSEMBLE`.

Locking timestamp: `2026-07-19T11:44:07.883726+00:00`.

Locked task: HIGH_RISK_B, recovered calibration, existing financial reward,
six markdown actions, locked baseline `always_0pct`.

Test sample size: 60 eligible held-out test episodes.

Test result:

```json
{
  "ensemble_mean_normalized_profit": 0.42572130058655044,
  "baseline_mean_normalized_profit": 0.4479993306199982,
  "paired_mean_gain": -0.022278030033447582,
  "paired_median_gain": -0.017320922859513505,
  "paired_gain_std": 0.0204290063420295,
  "bootstrap_ci_low": -0.027486237775400803,
  "bootstrap_ci_high": -0.01731852298262823,
  "win_share": 0.0,
  "tie_share": 0.23333333333333334,
  "loss_share": 0.7666666666666667,
  "waste_rate_difference": -0.0002043833435954387,
  "sell_through_difference": 0.00020438334359543602,
  "average_markdown_difference": NaN,
  "worst_decile_gain": -0.0500000000000001,
  "maximum_episode_loss": -0.0500000000000001,
  "action_entropy": 0.6930670630541902,
  "zero_action_share": 0.5063291139240507,
  "positive_action_share": 0.4936708860759494,
  "accounting_valid_rate": 1.0,
  "numerical_valid": false,
  "test_episode_count": 60,
  "action_0_share": 0.5063291139240507,
  "action_1_share": 0.4936708860759494,
  "action_2_share": 0.0,
  "action_3_share": 0.0,
  "action_4_share": 0.0,
  "action_5_share": 0.0
}
```

Exact final claim:

The final claim is determined by the locked primary endpoint, paired normalized
profit gain versus always_0pct, with uncertainty and robustness reported
separately. No post-test tuning or model development was performed.

Limitations:

- Test evidence is conditional on the locked HIGH_RISK_B population.
- The ensemble uses normalized accounting assumptions from the existing
  environment, not retailer-deployment accounting.
- The strict win-share threshold is reported as secondary robustness evidence.

Confirmation: no test split was accessed before candidate and protocol locking,
and no post-test tuning was performed by this script.



## Source: planning_distillation_decision.md

# HIGH_RISK_B Planning Distillation

Final validation status: `PLANNING_VALUE_EXISTS_BUT_DISTILLATION_FAILED`.

The script trained compact supervised policies using planning labels generated
only from the training split. The locked validation manifest and always-zero
baseline were not changed, and the test split was not used.

Best distilled validation metrics:

```json
{
  "policy_id": "planning_distilled_policy",
  "mean_normalized_profit": 0.13310723882526332,
  "paired_gain_vs_always_0pct": -0.019973829728999283,
  "paired_median_gain": 0.009831790741232506,
  "bootstrap_ci_low": -0.058678115933868256,
  "bootstrap_ci_high": 0.01633797172620917,
  "win_share": 0.5454545454545454,
  "tie_share": 0.09090909090909091,
  "loss_share": 0.36363636363636365,
  "waste_rate_difference": -0.13552999230251492,
  "sell_through_difference": 0.13552999230251492,
  "average_markdown": NaN,
  "worst_decile_paired_profit_difference": -0.18420380830886662,
  "maximum_episode_loss": -0.21816159738756535,
  "action_entropy": 0.4083724251616484,
  "planning_gain_capture_ratio": -1.8507229904243858,
  "runtime_per_episode": 0.15497630909191107,
  "accounting_valid": true,
  "eligible_for_final_lock": false
}
```

Grouped training metrics are saved in
`outputs/tables/planning_distillation_training_metrics.csv`.

The final locked candidate config is saved in
`outputs/configs/final_high_risk_b_candidate.json`.


