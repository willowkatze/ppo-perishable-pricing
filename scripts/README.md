# Script Guide

Run commands from the repository root. The numbered scripts are narrow entry points; the implementation and exact paths are in the linked `src/` modules. Training commands are optional for reading the archived results because raw data and model binaries are intentionally excluded from Git.

## Execution Order

| Step | Command | Purpose | Underlying module | Trains model? | Requires local data/model? | Main outputs |
|---:|---|---|---|---|---|---|
| 1 | `python scripts/01_prepare_freshretail_data.py` | Prepare the FreshRetailNet modeling subset. | `src.freshretail_data_processing` | No | Raw FreshRetailNet parquet files | `data/freshretail/processed/freshretail_modeling_subset.parquet`, `outputs/tables/freshretail_*` |
| 2 | `python scripts/02_recover_stockout_demand.py` | Estimate demand hidden by stockout censoring. | `src.latent_demand_recovery` | Fits a supervised recovery model | Processed FreshRetailNet subset | `data/freshretail/processed/freshretail_demand_recovered.parquet`, `outputs/tables/latent_demand_*` |
| 3 | `python scripts/03_train_original_ppo.py --stage smoke --timesteps 2000` | Train original observed/recovered PPO agents. | `src.train_ppo_operational` | Yes, PPO | Recovered demand, environment artifacts, discount-response models | `outputs/models/ppo_operational/`, `outputs/tables/ppo_*` |
| 4 | `python scripts/04_train_balanced_ppo.py --timesteps 20000` | Train the balanced recovered PPO redesign. | `src.train_ppo_recovered_redesign` | Yes, PPO | Recovered demand, locked validation manifest, prior diagnostics | `outputs/models/ppo_training_redesign/`, `outputs/tables/ppo_training_*` |
| 5 | `python scripts/05_train_dqn.py --timesteps 5000 --seeds 42` | Train HIGH_RISK_B DQN candidates. | `src.train_dqn_high_risk_b` | Yes, DQN | HIGH_RISK_B labels and validation manifest | `outputs/models/dqn_high_risk_b/`, `outputs/tables/dqn_*` |
| 6 | `python scripts/06_evaluate_locked_dqn_ensemble.py` | Evaluate the locked DQN ensemble on held-out episodes. | `src.final_dqn_ensemble_test_evaluation` | No | Locked DQN model artifacts | `outputs/tables/final_dqn_ensemble_test_*` |
| 7 | `python scripts/07_run_final_baseline_ladder.py` | Compare the locked learned policies and nine fixed/random/rule baselines. | `src.final_test_baseline_ladder` | No | Locked PPO/DQN artifacts and final test manifest | `outputs/tables/final_test_baseline_ladder.csv`, `outputs/figures/final_test_baseline_ladder/` |

Steps 1-5 prepare or train models. Steps 6-7 are evaluation/reporting entry points and do not retrain models. The selected public artifacts under `results/` are sufficient for reading the final conclusion without running these commands.

## Core Implementation

Read these files first, in this order:

1. `src/freshretail_data_processing.py`
2. `src/latent_demand_recovery.py`
3. `src/discount_response.py`
4. `src/perishability_scenarios.py`
5. `src/pricing_env_operational.py`
6. `src/train_ppo_operational.py`
7. `src/train_ppo_recovered_redesign.py`
8. `src/train_dqn_high_risk_b.py`
9. `src/final_dqn_ensemble_test_evaluation.py`
10. `src/final_test_baseline_ladder.py`

## Optional Diagnostics and History

These files explain diagnostics or earlier branches of the analysis but are not required to understand the main implementation path:

- `src/evaluate_ppo_operational.py`
- `src/dqn_statistical_audit_and_ensemble.py`
- `src/final_report_figures.py`
- `src/high_risk_b_planning_distillation.py`
- `archive/source_history/`

## Training Versus Evaluation

Data preparation and demand recovery create the local inputs. The PPO and DQN scripts train models and may be compute-intensive. The locked evaluation scripts read existing model artifacts and fixed manifests. The final baseline ladder is a post-hoc comparison and does not replace the primary DQN or PPO conclusions.

## Configs

The repository uses one consistent distinction:

- `configs/` contains version-controlled final configurations, specifications, manifests, checksums, and locked metadata included in Git.
- `outputs/configs/` contains local runtime-generated copies, execution records, artifact hashes, and environment snapshots. These files are produced during local runs and are not required to inspect the selected result tables.

Exact records for the main path are:

- Environment: `configs/environment/perishability_master_config.json`, `configs/environment/perishability_accounting_specification.json`, `configs/environment/perishability_financial_reward_specification.json`, and `configs/environment/pricing_env_operational_config.json`.
- Original PPO: `configs/ppo/ppo_operational_experiment_config.json` and `configs/ppo/ppo_input_artifact_hashes.json`.
- Balanced PPO: `configs/ppo/ppo_redesign_selected_model.json`, `configs/ppo/ppo_redesign_selected_model_hashes.json`, and `configs/ppo/ppo_validation_episode_manifest.csv`.
- DQN: `configs/dqn/final_locked_dqn_ensemble.json` and `configs/dqn/final_locked_dqn_ensemble_hashes.json`.
- Held-out evaluation: `configs/evaluation/final_test_evaluation_protocol.json`, `configs/evaluation/artifact_manifest.csv`, and `configs/evaluation/checksum_manifest.sha256`.

The implementation modules write local runtime records under `outputs/configs/`; the committed `configs/` records preserve the final specifications and locks.

## Local Artifacts

Raw FreshRetailNet data, trained PPO/DQN binaries, VecNormalize files, and large intermediate outputs are excluded from Git. The archived `results/` files are enough to inspect the reported conclusions, but not enough to rerun training from scratch. `data/raw/README.md` describes the expected local raw files.

## Important Warning

Running training scripts can be compute-intensive and is not needed to inspect the archived results. Use the code and selected results first; rerun training only when local data and model-artifact requirements are available.
