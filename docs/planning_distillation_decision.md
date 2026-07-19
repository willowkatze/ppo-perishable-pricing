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
