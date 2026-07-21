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

- `START_HERE.md`: short project entry.
- `docs/`: six main documentation files.
- `src/`: implementation modules.
- `scripts/`: numbered execution wrappers.
- `configs/`: saved experiment and environment configs.
- `results/`: final report tables and figures.
- `archive/`: supporting outputs, audits, and experiment history.

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

