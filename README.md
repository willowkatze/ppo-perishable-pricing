# PPO Perishable Pricing

This repository studies dynamic markdown decisions for perishable fresh-retail inventory. The workflow uses FreshRetailNet-50K-informed time series, stockout-aware demand recovery, a semi-synthetic perishability environment, PPO, a DQN comparison, and a locked held-out baseline ladder.

The project asks whether learned markdown policies improve normalized accounting profit on high-risk inventory after waste, stockout, and finite-shelf-life effects are represented. It is a controlled reinforcement-learning experiment using a semi-synthetic environment calibrated with historical retail data, not a deployed pricing system.

## Source Code

[`src/`](src/) is the central and most important source-code directory in this repository. It contains the complete Python implementation of the experimental pipeline: data preparation, stockout-aware demand recovery, markdown-response calibration, perishability scenarios, the Gymnasium pricing environment, PPO and DQN training, locked evaluation, and final reporting.

The Python modules include concise Chinese comments and module-level descriptions to make the implementation easier to inspect. The same folder also contains three detailed reading aids:

- [`src/技术文档.md`](src/技术文档.md): the main technical guide, including the pipeline, module responsibilities, inputs, outputs, implementation logic, and interpretation boundaries.
- [`src/实验记录_.docx`](src/实验记录_.docx): the experiment log organized by development stage, with direct references to the relevant files and functions.
- [`src/环境设置.docx`](src/环境设置.docx): the detailed environment specification covering the 41-dimensional state, six markdown actions, FEFO inventory flow, reward/accounting, and PPO/DQN environment wrappers.

For source-code review, begin with [`src/技术文档.md`](src/技术文档.md), then inspect [`src/pricing_env_operational.py`](src/pricing_env_operational.py) and follow the recommended reading order in the guide.

## Data and Problem

FreshRetailNet-50K provides the operational fresh-retail setting. Raw parquet files are not committed; the expected local files are described in [data/raw/README.md](data/raw/README.md). The project selects a modeling subset, identifies stockout-censored observations, and estimates a recovered demand signal before pricing experiments.

The final modeling subset contains 29,100 rows from 300 complete store-product sequences, covering 232 stores and 177 SKUs. Each complete sequence has exactly 97 daily observations from 2024-03-28 through 2024-07-02. The recovery summary reports a 13.317% aggregate increase from observed sales to recovered demand, with 40.196% of rows adjusted. These estimates are model-based proxies, not direct observations of latent demand.

## Method Pipeline

1. Prepare a split-contained FreshRetailNet modeling subset.
2. Recover demand affected by stockout censoring with an ExtraTrees model.
3. Fit observed and recovered discount-response artifacts.
4. Simulate inventory aging, FEFO issuing, markdown actions, demand response, waste, and accounting in the pricing environment.
5. Train and diagnose original PPO policies.
6. Train balanced recovered PPO after observing action-collapse diagnostics.
7. Train and lock a multi-seed DQN ensemble as a value-based comparison.
8. Evaluate learned policies and fixed, random, and rule-based baselines on the identical 60-episode HIGH_RISK_B held-out test set.

## Main Models

- Original PPO uses observed or recovered calibration under the operational environment.
- Balanced recovered PPO changes training-scenario exposure while preserving the environment, reward, action space, and validation protocol.
- The locked DQN ensemble averages Q-values from three STANDARD_DQN seeds and selects the highest-valued action.
- Baselines include no markdown, fixed markdown levels, uniform random actions, and two simple risk rules.

## Compact Final Result

| Policy | Mean normalized profit | Gain vs always_0pct |
|---|---:|---:|
| `always_0pct` | 0.447999 | 0.000000 |
| `balanced_recovered_ppo` | 0.435909 | -0.012091 |
| `locked_dqn_ensemble` | 0.425721 | -0.022278 |

`always_0pct` is the strongest overall held-out policy. `balanced_recovered_ppo` is the strongest learned policy, but no learned policy beat `always_0pct`. The complete 13-policy comparison is in [results/tables/final_heldout_baseline_ladder.csv](results/tables/final_heldout_baseline_ladder.csv), with the main visual comparison in [results/figures/heldout_policy_comparison.png](results/figures/heldout_policy_comparison.png).

## Repository Structure

| Directory | Purpose | Important contents |
|---|---|---|
| `scripts/` | Numbered command-line entry points for the main workflow. | Data preparation, demand recovery, PPO/DQN training wrappers, locked DQN evaluation, final baseline ladder. |
| `src/` | **Primary source-code directory and recommended starting point for code inspection.** | Complete Python implementation, Chinese code comments, `技术文档.md`, `实验记录_.docx`, and `环境设置.docx`. |
| `configs/` | Version-controlled final specifications, manifests, and locked metadata. | Environment, PPO, DQN, and held-out evaluation records retained in Git. |
| `results/` | Selected final report artifacts. | Main tables and figures used to interpret the final findings. |
| `outputs/` | Generated intermediate and diagnostic artifacts. | Local training, validation, manifests, model artifacts, and runtime configuration records. |
| `docs/` | Main project explanation. | Research design, data, methods, results, conclusions, and reproducibility. |
| `tests/` | Lightweight code checks. | Environment and PPO pipeline tests; model-dependent tests may skip without local artifacts. |
| `archive/` | Supporting history and traceability. | Experiment history, audits, intermediate tables, and archived source modules; not required for the main code path. |
| `data/` | Local data location. | Raw and processed data placeholders; raw files are not committed. |

`configs/` and `outputs/configs/` are deliberately different. `configs/` contains version-controlled final records included in Git. `outputs/configs/` contains local runtime-generated copies, checksums, and execution records. Similarly, `outputs/` is the working area for generated intermediate diagnostics, while `results/` contains selected final tables and figures.

## Setup Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Raw FreshRetailNet files and trained model binaries must be supplied locally before data preparation or training. The numbered commands and their exact inputs are documented in [scripts/README.md](scripts/README.md). Start with [START_HERE.md](START_HERE.md) for the three reading paths.

## Key Files

- [src/技术文档.md](src/技术文档.md)
- [src/实验记录_.docx](src/实验记录_.docx)
- [src/环境设置.docx](src/环境设置.docx)
- [docs/02_data_and_demand_recovery.md](docs/02_data_and_demand_recovery.md)
- [docs/03_environment_and_methods.md](docs/03_environment_and_methods.md)
- [docs/04_results.md](docs/04_results.md)
- [scripts/README.md](scripts/README.md)
- [src/latent_demand_recovery.py](src/latent_demand_recovery.py)
- [src/pricing_env_operational.py](src/pricing_env_operational.py)
- [src/train_ppo_operational.py](src/train_ppo_operational.py)
- [src/train_ppo_recovered_redesign.py](src/train_ppo_recovered_redesign.py)
- [src/train_dqn_high_risk_b.py](src/train_dqn_high_risk_b.py)
- [src/final_test_baseline_ladder.py](src/final_test_baseline_ladder.py)
- [results/tables/final_heldout_baseline_ladder.csv](results/tables/final_heldout_baseline_ladder.csv)
- [results/figures/heldout_policy_comparison.png](results/figures/heldout_policy_comparison.png)

## Project Limitations

The environment is semi-synthetic. Raw data and trained model binaries are excluded from GitHub. Full reproduction therefore requires local data and model artifacts. No learned policy beat `always_0pct` on the locked held-out test set.

