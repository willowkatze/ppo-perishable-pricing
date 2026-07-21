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
