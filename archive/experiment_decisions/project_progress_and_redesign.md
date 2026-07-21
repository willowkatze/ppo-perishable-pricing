# Project Progress And Research Redesign

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

Experiment matrix:

1. raw-sales demand + financial reward
2. recovered-demand + financial reward
3. raw-sales demand + sustainability reward
4. recovered-demand + sustainability reward

Core output metrics:

- accounting profit
- waste rate
- sell-through rate
- stockout rate
- average markdown
- markdown timing
- policy action distribution
- profit-waste Pareto frontier

## 10. Remaining Risks

- operational dataset may not provide shelf life or waste
- perishability may require transparent assumptions
- stockout-aware demand recovery may introduce model uncertainty
- full dataset may be too large for a student laptop
- project scope must remain manageable
- the final result will still be a semi-synthetic simulation, not a live retailer deployment
