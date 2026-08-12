# From Stockout-Censored Demand to Sequential Markdown Pricing

This repository studies a signal-to-policy question in perishable retail:

> **Does improving a stockout-censored demand signal through demand recovery actually propagate into better sequential pricing behavior and economic value?**

The central chain is:

**Stockout censoring → Demand-signal recovery → Pricing policy → Policy behavior → Economic value**

Observed sales can understate demand when available inventory limits what can be sold. This creates an upstream information-quality problem for any downstream pricing system. The project therefore constructs a stockout-aware recovered-demand proxy, carries observed and recovered demand representations through parallel markdown-response calibrations, and evaluates whether the changed signal improves sequential pricing decisions.

PPO and DQN are experimental tools used to study this transmission. The project does not propose a new PPO algorithm, DQN algorithm, latent-demand-recovery algorithm, or causal price-elasticity estimator.

## Why This Project

Censored-demand research often emphasizes estimation or forecasting, while reinforcement-learning pricing studies often take the demand environment as given. This repository connects those layers and asks what happens downstream when the demand representation is corrected or changed:

- Does recovery materially change the measured demand signal?
- Does the changed signal alter learned markdown behavior?
- If behavior improves, does that improvement create held-out economic value?

The contribution is a controlled downstream evaluation of recovered demand, not simply an RL pricing implementation. The empirical distinction matters: **better information does not automatically produce a better policy, and a better-behaved policy does not automatically create higher economic value.**

## Main Contribution and Positioning

The project connects three adjacent methodological layers:

| Research layer | Typical focus | Role in this project |
|---|---|---|
| Censored-demand / FreshRetailNet-style work | Censoring → recovery → forecasting or estimation | Identify stockout censoring and construct a recovered-demand proxy. |
| Perishable-pricing DRL | Given demand environment → learn pricing or inventory policy | Use standard PPO and DQN as downstream policy learners. |
| This project | Censoring → recovery → response calibration → policy behavior → economic value | Evaluate the downstream decision value of changing the censored demand signal. |

No claim is made that no prior work has combined these elements. The distinguishing research question is whether an upstream signal correction survives the full path to downstream economic outcomes.

## Experimental Design: Signal → Policy → Value

### Layer 1 — Signal and Measurement

1. Select complete, split-contained FreshRetailNet store-product sequences.
2. Identify observations that may be censored by stockouts.
3. Fit and validate a stockout-aware recovery model using artificial-censoring checks.
4. Construct `recovered_demand` as a model-based proxy, not observed latent-demand ground truth.
5. Fit parallel observed-demand and recovered-demand markdown-response models.

### Layer 2 — Policy and Learning

1. Compare Original Observed PPO with Original Recovered PPO as a signal/calibration intervention.
2. Diagnose behavioral concentration and limited policy selectivity on fixed validation episodes.
3. Train Scenario-Balanced Recovered PPO while keeping recovered calibration fixed.
4. Use HIGH_RISK_B limited-horizon planning as a diagnostic for dynamic opportunities in interpretable high-risk states.
5. Use DQN as a value-based algorithm-family robustness comparison.

### Layer 3 — Economic Value

1. Lock model candidates, preprocessing records, baseline definitions, manifests, and the evaluation protocol before held-out testing.
2. Evaluate policies on the same 60 paired HIGH_RISK_B held-out episodes.
3. Compare learned policies with a strong no-markdown baseline, fixed markdowns, random actions, and simple rules.
4. Separate behavioral improvement from economic superiority.

## Data and Demand Recovery

FreshRetailNet-50K provides the operational fresh-retail setting. Raw parquet files are not committed; expected local inputs are documented in [data/raw/README.md](data/raw/README.md).

The final modeling subset contains 29,100 rows from 300 complete store-product sequences, covering 232 stores and 177 SKUs. Each sequence contains 97 daily observations from 2024-03-28 through 2024-07-02. Splits are chronological so future records do not enter model fitting.

The recovery pipeline identifies possible stockout censoring, fits candidate demand models on likely uncensored training observations, and uses artificial censoring to evaluate recovery where a pseudo-ground-truth remains observable. The selected pipeline reports:

- aggregate demand increase from observed to recovered: **13.317%**;
- share of rows adjusted: **40.196%**.

These values show that recovery materially changes the demand representation. They do not prove that `recovered_demand` is true latent demand or that a changed signal will improve pricing profit.

Parallel markdown-response models are fitted for observed and recovered demand. Their counterfactual action responses are observational and model-implied; they are not causally identified price elasticities.

## Semi-Synthetic Pricing Environment

The common Gymnasium environment combines operational demand context with explicit perishability and accounting mechanics:

- a fixed 41-dimensional state covering inventory, perishability, demand, time/history, and context;
- six markdown actions: `0%`, `5%`, `10%`, `20%`, `30%`, and `40%`;
- FEFO inventory issuing and inventory aging;
- deterministic demand in the validated main environment;
- physical expiration waste;
- procurement and disposal accounting;
- normalized financial reward and normalized accounting profit.

Shelf life, inventory age composition, procurement/disposal costs, expiration, and related operational mechanisms include scenario assumptions because FreshRetailNet does not provide every field required for sequential perishability simulation. The environment is a controlled comparison setting, not a directly deployable retailer system.

## Controlled Comparisons

### Experiment A — Signal Intervention

**Original Observed PPO vs Original Recovered PPO**

Approximately held constant:

- PPO algorithm and network family;
- six-action markdown space;
- financial reward;
- inventory mechanics;
- FEFO and accounting framework.

Changed:

- observed-demand calibration pipeline;
- recovered-demand calibration pipeline.

Held-out normalized profit:

| Policy | Mean normalized profit |
|---|---:|
| `original_observed_ppo` | 0.307872 |
| `original_recovered_ppo` | 0.310856 |

The recovered pipeline is only slightly higher. This does not support a strong claim that demand recovery alone created meaningful economic improvement.

This is a comparison between two calibration pipelines, not a pure causal estimate of the ExtraTrees recovery step. The demand representation/history and the corresponding markdown-response model both change with calibration mode.

### Experiment B — Policy-Learning Intervention

**Original Recovered PPO vs Scenario-Balanced Recovered PPO**

Held constant:

- recovered-demand calibration;
- environment mechanics and accounting;
- financial reward;
- action space;
- validation distribution.

Changed:

- training-scenario exposure.

Scenario-balanced training increases exposure to validated high-inventory, short-life, low-sell-through, and meaningful trade-off scenarios. It is a training-exposure redesign, not a new PPO algorithm. The selected model is `BALANCED__checkpoint_20000`; it does not use the BALANCED_STABLE hyperparameter changes.

The redesign improved behavioral selectivity, including reducing unnecessary markdown, while remaining state-dependent. This behavioral improvement did not automatically create superior held-out economic value.

Because the project does not include Balanced Observed PPO, it does not form a complete 2×2 recovery × balanced-training design. Balanced Recovered PPO should therefore not be compared directly with Original Observed PPO as though that isolated the effect of demand recovery.

## Roles of PPO, Planning, and DQN

### Original and Balanced PPO

Original PPO provides the controlled observed-versus-recovered calibration comparison. Balanced PPO tests whether limited exposure to sparse positive-markdown opportunities remains a downstream learning bottleneck. Diagnostics describe the original recovered policy cautiously as behaviorally concentrated or mostly-zero but state-dependent; the repository does not rely on an unsupported claim of strict action collapse.

### HIGH_RISK_B Limited-Horizon Planning

Planning is a diagnostic experiment, not the final policy. HIGH_RISK_B is defined independently of PPO wins using operationally interpretable inventory, expiration, and projected sell-through conditions.

On 22 paired validation episodes, horizon-3 planning produced:

- mean normalized-profit gain vs `always_0pct`: **+0.010792**;
- bootstrap 95% CI: **[-0.025571, 0.039094]**;
- Win / Tie / Loss: **81.8% / 9.1% / 9.1%**.

The interval crosses zero, so this is classified as a **modest dynamic-value signal**, not statistical superiority. It motivates further policy-learning analysis but does not overturn the locked held-out conclusion.

### DQN Robustness Comparison

DQN is a value-based algorithm-family robustness comparison. The six-action discrete space makes Q-value learning a natural contrast with PPO's policy-gradient representation. Environment, recovered calibration, reward, accounting, and action mapping remain fixed.

The final locked DQN policy is a three-seed equal-weight Q ensemble: each member evaluates the same six actions using its own saved observation normalization, Q-values are averaged across members, and the action with the highest mean Q-value is selected.

## Main Findings

### Locked Held-Out Test

All policies in the final ladder use the same 60 HIGH_RISK_B held-out episodes.

| Policy | Role | Mean normalized profit | Gain vs `always_0pct` |
|---|---|---:|---:|
| `always_0pct` | Strong no-markdown baseline | 0.447999 | 0.000000 |
| `balanced_recovered_ppo` | Strongest learned policy | 0.435909 | -0.012091 |
| `locked_dqn_ensemble` | Value-based robustness comparison | 0.425721 | -0.022278 |
| `original_recovered_ppo` | Original recovered-calibration PPO | 0.310856 | -0.137143 |
| `original_observed_ppo` | Original observed-calibration PPO | 0.307872 | -0.140127 |

`always_0pct` is the strongest overall policy. Balanced Recovered PPO is the strongest learned policy, but it does not exceed the no-markdown baseline. The locked DQN ensemble also remains below `always_0pct`; its paired gain is `-0.022278` with bootstrap 95% CI `[-0.027486, -0.017319]`.

The complete 13-policy comparison is available in [results/tables/final_heldout_baseline_ladder.csv](results/tables/final_heldout_baseline_ladder.csv), with the main visual in [results/figures/heldout_policy_comparison.png](results/figures/heldout_policy_comparison.png).

## What the Results Mean

The evidence separates three stages that should not be treated as interchangeable:

1. **Signal quality:** stockout-aware recovery materially changed the demand representation.
2. **Policy behavior:** the original observed-versus-recovered PPO comparison showed only a small economic difference, while balanced training later improved policy selectivity.
3. **Economic value:** improved behavior still did not produce higher held-out profit than the conservative no-markdown policy.

The central empirical takeaway is therefore:

> **Better information does not automatically produce a better policy, and a better-behaved policy does not automatically create higher economic value.**

This conclusion is conditional on the selected data, recovered-demand proxy, response models, HIGH_RISK_B population, and semi-synthetic environment assumptions. It is not a general claim that RL or markdowns cannot work in perishable retail.

## Source Code and Repository Structure

[`src/`](src/) is the central source-code directory and the recommended starting point for implementation review. It contains the complete Python pipeline together with detailed Chinese reading aids:

- [`src/技术文档.md`](src/技术文档.md): pipeline, module responsibilities, inputs, outputs, implementation logic, and interpretation boundaries;
- [`src/实验记录_.docx`](src/实验记录_.docx): experiment history organized by development stage, with file/function locations;
- [`src/环境设置.docx`](src/环境设置.docx): detailed specification of state, actions, FEFO, transitions, reward, accounting, and training wrappers.

For code inspection, begin with `src/技术文档.md`, then read `src/pricing_env_operational.py` and follow the guide's recommended order.

| Directory | Purpose | Important contents |
|---|---|---|
| `src/` | **Primary source-code directory.** | Data/recovery modules, environment, PPO/DQN, evaluation, reporting, Chinese comments, and detailed technical records. |
| `scripts/` | Numbered command-line entry points. | Data preparation, recovery, training, locked evaluation, and result-building wrappers. |
| `configs/` | Version-controlled final specifications and locked records. | Environment, PPO, DQN, manifests, and held-out protocol metadata. |
| `results/` | Selected final report artifacts. | Canonical tables and figures supporting the final findings. |
| `outputs/` | Generated working artifacts. | Local intermediates, diagnostics, model files, manifests, and runtime records. |
| `docs/` | Concise research documentation. | Research design, data, methods, results, conclusions, and reproducibility. |
| `tests/` | Lightweight static and mechanical checks. | Environment and pipeline checks; artifact-dependent tests may skip. |
| `archive/` | Historical traceability. | Earlier experiments, audits, and intermediate artifacts not required for the main code path. |
| `data/` | Local data location. | Raw/processed placeholders; raw FreshRetailNet files are not committed. |

`outputs/` contains generated intermediate and diagnostic artifacts, while `results/` contains selected final report artifacts. `configs/` contains version-controlled specifications and immutable records; local runtime-generated copies may also appear under `outputs/configs/`.

## Reproducibility and Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Raw FreshRetailNet files and large trained model artifacts must be supplied locally before full data preparation, training, or artifact-dependent evaluation. Exact commands and input/output paths are documented in [scripts/README.md](scripts/README.md). [START_HERE.md](START_HERE.md) provides compact reading paths.

Important records include:

- environment: `configs/environment/pricing_env_operational_config.json`;
- original PPO: `configs/ppo/ppo_operational_experiment_config.json`;
- balanced PPO selection: `configs/ppo/ppo_redesign_selected_model.json`;
- DQN lock: `configs/dqn/final_locked_dqn_ensemble.json`;
- held-out protocol: `configs/evaluation/final_test_evaluation_protocol.json`.

Inspecting the archived results does not require rerunning compute-intensive training scripts.

## Limitations

### Demand Recovery

`recovered_demand` is a model-based latent-demand proxy, not observed ground truth. Artificial censoring provides a validation device but cannot reveal true unmet demand in real stockout records.

### Markdown Response

Observed and recovered markdown-response models produce observational, model-implied responses. They do not identify causal price elasticity, and historical action support is uneven.

### Semi-Synthetic Environment

Shelf life, inventory age profile, procurement/disposal costs, expiration, physical waste, and related operating mechanisms contain scenario assumptions. Economic conclusions are conditional on simulator fidelity and these assumptions.

### Experimental Design

The available PPO comparisons are Original Observed PPO, Original Recovered PPO, and Balanced Recovered PPO. Balanced Observed PPO is missing, so the project can study the original calibration comparison and the balanced-training intervention conditional on recovered calibration, but cannot fully identify a recovery × balanced-training interaction.

### Reproducibility

Repository traceability is strong, but raw FreshRetailNet files and large trained model binaries are not committed. Full rerunning therefore requires external or local artifacts.

