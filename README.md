# PPO Perishable Pricing

This repository contains a reinforcement-learning course project on markdown decisions for perishable fresh-retail inventory. The project uses FreshRetailNet-informed demand recovery, a semi-synthetic perishability environment, PPO experiments, a DQN comparison, and a locked held-out baseline ladder.

## Research Question

Can learned markdown policies improve decision quality for high-risk perishable inventory, and do they outperform fixed, random, rule-based, and no-markdown baselines on a locked held-out test set?

## Method Pipeline

1. Prepare FreshRetailNet-informed time series.
2. Recover demand affected by stockout censoring.
3. Build a semi-synthetic perishability pricing environment.
4. Train and diagnose PPO policies.
5. Train balanced recovered PPO after collapse diagnostics.
6. Compare DQN as an additional value-based method.
7. Evaluate all final policies on the locked 60-episode HIGH_RISK_B held-out test set.

## Main Results

| Policy | Role | Mean normalized profit | Gain vs always_0pct | Interpretation |
|---|---|---:|---:|---|
| always_0pct | strongest overall | 0.447999 | 0.000000 | Best held-out policy. |
| balanced_recovered_ppo | strongest learned | 0.435909 | -0.012091 | Beat positive fixed-markdown, random, rule-based, original PPO, and DQN alternatives; did not beat always_0pct. |
| locked_dqn_ensemble | learned comparison | 0.425721 | -0.022278 | Did not beat always_0pct; CI [-0.027486, -0.017319]. |

The full 13-policy comparison is in `results/tables/final_heldout_baseline_ladder.csv` and `results/figures/heldout_policy_comparison.png`.

## Repository Structure

| Directory | Purpose | Important contents |
|---|---|---|
| `scripts/` | Numbered command-line entry points for the main workflow. | Data preparation, demand recovery, PPO/DQN training wrappers, locked DQN evaluation, final baseline ladder. |
| `src/` | Active implementation modules. | FreshRetailNet processing, latent-demand recovery, discount response, scenario generation, Gymnasium environment, PPO/DQN training, locked evaluation, final report figures. |
| `configs/` | Small repository-level configuration records. | Lightweight config files retained in Git; many locked runtime configs are generated under `outputs/configs/` and archived as records. |
| `results/` | Selected final report artifacts. | Five main tables and seven main figures used by the written report. |
| `outputs/` | Generated intermediate and diagnostic artifacts. | Local training, validation, diagnostic, manifest, model, and figure outputs; many large artifacts are excluded from Git. |
| `docs/` | Main project explanation. | Research design, data, methods, results, conclusions, reproducibility. |
| `tests/` | Lightweight code checks. | Environment and PPO pipeline tests; model-dependent tests may be skipped without local artifacts. |
| `archive/` | Supporting history and traceability. | Consolidated experiment history, audits, intermediate tables, and archived source modules; not required for the main reading path. |
| `data/` | Local raw and processed data location. | FreshRetailNet raw files and processed parquet files; raw data is not committed. |

`outputs/` and `results/` have different roles. `outputs/` is the working area for generated intermediate diagnostics and model-run artifacts. `results/` contains only the selected final tables and figures intended for the report.

## Setup Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Main Documents

- [Project and research design](docs/01_project_and_research_design.md)
- [Data and demand recovery](docs/02_data_and_demand_recovery.md)
- [Environment and methods](docs/03_environment_and_methods.md)
- [Results](docs/04_results.md)
- [Conclusions and limitations](docs/05_conclusions_and_limitations.md)
- [Reproducibility](docs/reproducibility.md)

## Project Limitations

The environment is semi-synthetic. Raw data and trained model binaries are excluded from GitHub. No learned policy beat always_0pct on the locked held-out test set.
