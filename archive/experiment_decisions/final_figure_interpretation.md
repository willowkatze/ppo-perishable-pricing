# Final Figure Interpretation

## PPO Validation Return Over Training
- Data source: `final_ppo_training_curve_data.csv`
- x-axis meaning: timesteps
- y-axis meaning: mean validation return
- Visible pattern: Checkpoint 22288 is marked; later recovered checkpoints show conservatism/collapse.
- Valid interpretation: Training improved early but longer training did not monotonically improve decision quality.
- Invalid interpretation: This does not prove test-set generalization.
- Suggested report caption: Validation return over training with validation-only model selection.
- Suggested 20-second presentation explanation: Use this to explain why checkpoint selection matters.

## Recovered Action Distribution Over Training
- Data source: `final_ppo_training_curve_data.csv`
- x-axis meaning: checkpoint timestep
- y-axis meaning: action share
- Visible pattern: Recovered policy becomes mostly zero markdown later.
- Valid interpretation: Recovered calibration pushed behavior toward conservatism.
- Invalid interpretation: Do not interpret deterministic action shares as stochastic policy entropy.
- Suggested report caption: Recovered policy action distribution over checkpoints.
- Suggested 20-second presentation explanation: Show the collapse narrative.

## PPO vs Baselines
- Data source: `ppo_financial_paired_validation_summary.csv`
- x-axis meaning: policy
- y-axis meaning: normalized profit
- Visible pattern: Always-zero is highest; recovered PPO beats observed PPO.
- Valid interpretation: Financial PPO is not the best average validation policy.
- Invalid interpretation: Raw profit is not scale-comparable here.
- Suggested report caption: Paired validation normalized-profit comparison.
- Suggested 20-second presentation explanation: Use this as the main performance result.

## Financial Markdown Opportunity Map
- Data source: `financial_markdown_oracle_diagnostic.csv`
- x-axis meaning: inventory coverage
- y-axis meaning: fraction expiring soon
- Visible pattern: Most states prefer 0%; sparse high-risk states prefer positive markdown.
- Valid interpretation: The environment is not degenerate.
- Invalid interpretation: Do not claim markdown is broadly optimal.
- Suggested report caption: Oracle map of state-dependent markdown opportunities.
- Suggested 20-second presentation explanation: Explain why average zero baseline can win while opportunities exist.

## PPO vs Oracle Error Decomposition
- Data source: `financial_markdown_ppo_regret.csv`
- x-axis meaning: policy
- y-axis meaning: decision share
- Visible pattern: Main error is unnecessary markdown.
- Valid interpretation: PPO loses value through over-markdown relative to oracle.
- Invalid interpretation: Do not claim PPO never markdowns.
- Suggested report caption: Decision errors versus diagnostic oracle.
- Suggested 20-second presentation explanation: Use this to connect RL underperformance to decision quality.

## Empirical Profit-Waste Trade-off
- Data source: `sustainability_tradeoff_vs_always_zero.csv`
- x-axis meaning: waste rate
- y-axis meaning: normalized profit
- Visible pattern: Waste reduction is possible at measurable profit cost.
- Valid interpretation: Sustainability is a constrained trade-off, not a weighted PPO success.
- Invalid interpretation: Do not call this a full Pareto frontier.
- Suggested report caption: Empirical profit-waste trade-off over finite policies.
- Suggested 20-second presentation explanation: Use this for the innovation/sustainability angle.

## Required Scientific Conclusions
1. PPO training initially improved validation performance.
2. Training longer did not monotonically improve policy quality.
3. Recovered PPO became increasingly conservative and later collapsed to deterministic zero markdown.
4. Checkpoint_22288 retained state dependence and was selected using validation only.
5. Recovered PPO outperformed observed PPO but not always-zero.
6. Zero markdown is optimal in most states, but not all states.
7. Positive markdown is financially valuable in approximately 16.7% of audited recovered-calibration states.
8. PPO mainly loses value through unnecessary markdown.
9. Recovered calibration lowers regret relative to observed calibration.
10. Sustainability results show a measurable profit-waste trade-off.
