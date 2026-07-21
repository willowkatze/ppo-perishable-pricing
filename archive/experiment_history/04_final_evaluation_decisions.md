# Final Evaluation Decisions

This file consolidates earlier experiment notes and keeps the order of major decisions without repeating the main result tables.

## Source: final_figure_interpretation.md

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



## Source: final_model_decision_task_interpretation.md

# Final Model Decision-Task Audit

## Purpose

This audit implements the course requirement's strict success criterion: a final model must beat a strong, pre-specified baseline on a paired held-out task. This script does not use the test split and does not train any model.

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



## Source: final_test_baseline_ladder.md

锘? Final Test Baseline Ladder

This is a post-hoc secondary descriptive analysis on the already locked 60-episode HIGH_RISK_B held-out test population. It does not retrain PPO or DQN, does not modify the locked DQN ensemble, and does not replace the primary conclusion.

Primary conclusion preserved: the locked DQN ensemble did not outperform `always_0pct` on held-out test episodes.

## Strongest Policies

- Strongest overall policy: `always_0pct`, mean normalized profit 0.447999.
- Strongest learned policy: `balanced_recovered_ppo`, mean normalized profit 0.435909.
- Locked DQN ensemble mean normalized profit: 0.425721.

## Secondary Findings

The locked DQN ensemble beat several weaker fixed, random, and rule-based baselines, including `always_5pct`, `always_10pct`, `always_20pct`, `always_30pct`, `always_40pct`, `random_uniform`, `expiry_rule_10pct`, and `inventory_coverage_rule_10pct`. It did not beat `always_0pct`.

The balanced recovered PPO was the strongest learned policy and also remained below `always_0pct`.

## Defensible Report Claim

A defensible claim is: under the locked HIGH_RISK_B held-out test, the no-markdown policy remained the strongest overall policy. Learned policies showed state-dependent behavior and outperformed several weaker markdown baselines, but the final locked DQN ensemble did not beat the primary no-markdown baseline.

## Outputs

- `outputs/tables/final_test_baseline_ladder.csv`
- `outputs/tables/final_test_model_vs_baseline_matrix.csv`
- `outputs/tables/final_test_baseline_pairwise_comparisons.csv`
- `outputs/figures/final_test_baseline_ladder/`



## Source: final_test_results.md

锘? Final Held-Out Test Results

The locked final candidate was `STANDARD_DQN_EQUAL_WEIGHT_Q_ENSEMBLE`. The locked baseline was `always_0pct`.

Held-out test population: 60 HIGH_RISK_B episodes. Pairing: 60/60 valid.

| Metric | Value |
|---|---:|
| DQN ensemble mean normalized profit | 0.425721 |
| always_0pct mean normalized profit | 0.447999 |
| paired mean gain | -0.022278 |
| paired median gain | -0.017321 |
| bootstrap 95% CI | [-0.027486, -0.017319] |
| win / tie / loss | 0.000 / 0.233 / 0.767 |
| waste-rate difference | -0.000204 |
| sell-through difference | +0.000204 |

Conclusion: the final locked DQN ensemble did not outperform the no-markdown baseline on held-out test episodes. Positive validation signals did not generalize.



## Source: project_progress_and_redesign.md

锘? Project Progress And Research Redesign

## 1. Original Research Objective

The original objective was PPO-based dynamic markdown optimization for perishable inventory. The first implementation used a financial reward, with the longer-term goal of extending toward a physical-waste sustainability objective.

## 2. Completed Work

### `data_processing.py`
Purpose: clean and validate the Kaggle perishable inventory dataset.

Main outputs: cleaned CSV, data-quality summaries, logical consistency checks, profit-formula audit.

Major findings: date parsing required explicit whitelisting; reported profit matched a formula including waste cost; general data-quality flags were low after correction.

Limitations: dataset provenance was not verified; many fields are formula-derived.

### `demand_model.py`
Purpose: diagnose demand targets and train a frozen Ridge demand model.

Main outputs: demand-model metrics, coefficients, demand-by-discount tables, confounding checks, demand figures, Ridge model artifact.

Major findings: discount appeared associated with demand, but discount was confounded with expiry/spoilage conditions.

Limitations: the demand model is observational and not causal.

### `counterfactual_demand_audit.py`
Purpose: audit whether the frozen Ridge model supports counterfactual markdown decisions.

Main outputs: counterfactual prediction tables, support analysis, scenario summaries, audit figures.

Major findings: single-period contribution often favored 0% markdown; deep markdowns could create negative unit margins.

Limitations: counterfactual support was limited and observational.

### `pricing_env.py`
Purpose: implement a multi-period Gymnasium environment using the frozen demand model.

Main outputs: environment validation episodes, fixed-policy sanity checks, waste-cost calibration, environment config.

Major findings: multi-period expiration creates a meaningful markdown-vs-waste tradeoff even when one-period contribution favors no markdown.

Limitations: environment dynamics are simulated from a likely educational dataset.

### `baselines.py`
Purpose: evaluate fixed, random, expiry-based, inventory-pressure, and margin-aware baseline policies.

Main outputs: baseline episode results, summaries, paired comparisons, reward-scale diagnostics, figures.

Major findings: AlwaysZero maximized mean financial profit; rule policies reduced waste at a financial cost; high-value categories dominated raw reward scale.

Limitations: baselines inherit the dataset and model limitations.

### `train_ppo.py`
Purpose: train a financial PPO baseline with value-normalized reward scaling.

Main outputs: PPO models for seeds 42, 123, and 2026; VecNormalize files; training evaluations; pilot results; action-distribution diagnostics; seed-stability tables; figures.

Major findings: financial PPO learned a mostly-zero policy with occasional mild markdowns, reducing waste relative to AlwaysZero while preserving high profit.

Limitations: it is a financial prototype, not a sustainability model or real deployment claim.

## 3. Dataset Audit Findings

The Kaggle dataset audit found:

- 100,000 rows
- deterministic accounting identities
- weak panel structure
- median sequence length 1
- maximum sequence length 3
- only about 4% of rows in multi-period sequences
- inventory continuity rate 0
- sequential-looking identifiers
- no verified provenance, owner, license, collection method, or generation method in local files

The final audit conclusion was `LIKELY_SYNTHETIC_OR_EDUCATIONAL_DATA`.

## 4. Reason For Research Redesign

The old dataset remains useful for pipeline prototyping but is insufficient as the main empirical foundation. The formal research design should use operational time-series data. The project is being redesigned rather than abandoned.

## 5. Reusable Engineering Assets

Reusable components include:

- Gymnasium environment architecture
- inventory transition logic
- baseline policy interface
- paired evaluation framework
- deterministic seed management
- PPO training loop
- VecNormalize handling
- action-distribution diagnostics
- reward scaling
- profit-waste evaluation tables and figures

## 6. Proposed New Research Direction

Proposed revised topic:

**Stockout-Aware and Sustainability-Aware Dynamic Markdown Optimization for Perishable Retail Products**

Possible research question:

**How does stockout-aware demand calibration affect dynamic markdown policies and the trade-off between financial performance and physical waste?**

## 7. Proposed New Data Foundation

The next pipeline may use FreshRetailNet-50K or another operational retail time-series dataset for observed sales, inventory, stockout information, promotions, discount variation, temporal patterns, and category heterogeneity.

Shelf life, expiration, cost structure, and physical waste may still require transparent semi-synthetic business assumptions.

## 8. Proposed Future Modules

- `src/freshretail_data_processing.py`: inspect operational data, validate schemas, and create a representative subset.
- `src/latent_demand_recovery.py`: recover latent demand under stockouts.
- `src/discount_response.py`: estimate discount response using stockout-aware demand.
- `src/perishability_scenarios.py`: create transparent shelf-life and waste assumptions.
- `src/pricing_env_operational.py`: adapt the pricing environment to operational time-series inputs.
- `src/baselines_operational.py`: evaluate fixed and rule-based policies on operational-data-informed simulations.
- `src/train_financial_ppo.py`: train financial PPO on the redesigned environment.
- `src/train_sustainable_ppo.py`: train sustainability-aware PPO.
- `src/evaluate_policies.py`: compare all policies using profit, waste, stockout, and action-distribution metrics.

## 9. Proposed Experimental Design

Demand calibration comparison:

- raw observed sales
- stockout-aware recovered demand

Reward comparison:

- financial reward
- sustainability-aware reward

Experiment m

