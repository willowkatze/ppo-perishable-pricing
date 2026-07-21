# Script Guide

Run commands from the repository root. Training commands are optional for code inspection because raw data and model binaries are intentionally excluded from Git.

## Execution Order

| Step | Command | Purpose | Underlying module | Trains model? | Requires local data/model? | Main outputs |
|---:|---|---|---|---|---|---|
| 1 | `python scripts/01_prepare_freshretail_data.py` | Prepare FreshRetailNet subset. | `src.freshretail_data_processing` | No | Raw FreshRetailNet parquet files | `data/freshretail/processed/freshretail_modeling_subset.parquet`, `outputs/tables/freshretail_*` |
| 2 | `python scripts/02_recover_stockout_demand.py` | Recover demand affected by stockouts. | `src.latent_demand_recovery` | Supervised recovery model only | Processed FreshRetailNet subset | `data/freshretail/processed/freshretail_demand_recovered.parquet`, `outputs/tables/latent_demand_*` |
| 3 | `python scripts/03_train_original_ppo.py --stage smoke --timesteps 2000` | Train original PPO policies. | `src.train_ppo_operational` | Yes, PPO | Recovered demand, environment artifacts, discount-response models | `outputs/models/ppo_operational/`, `outputs/tables/ppo_*` |
| 4 | `python scripts/04_train_balanced_ppo.py --timesteps 20000` | Train balanced recovered PPO redesign. | `src.train_ppo_recovered_redesign` | Yes, PPO | Locked validation manifest and prior diagnostics | `outputs/models/ppo_training_redesign/`, `outputs/tables/ppo_training_*` |
| 5 | `python scripts/05_train_dqn.py --timesteps 5000 --seeds 42` | Train HIGH_RISK_B DQN candidates. | `src.train_dqn_high_risk_b` | Yes, DQN | HIGH_RISK_B labels and validation manifest | `outputs/models/dqn_high_risk_b/`, `outputs/tables/dqn_*` |
| 6 | `python scripts/06_evaluate_locked_dqn_ensemble.py` | Evaluate locked DQN ensemble on held-out episodes. | `src.final_dqn_ensemble_test_evaluation` | No | Locked DQN model artifacts | `outputs/tables/final_dqn_ensemble_test_*` |
| 7 | `python scripts/07_run_final_baseline_ladder.py` | Run final 13-policy baseline ladder. | `src.final_test_baseline_ladder` | No | Locked PPO/DQN artifacts and final test manifest | `outputs/tables/final_test_baseline_ladder.csv`, `outputs/figures/final_test_baseline_ladder/` |

## Recommended Code Reading Order

1. `src/freshretail_data_processing.py`
2. `src/latent_demand_recovery.py`
3. `src/discount_response.py`
4. `src/perishability_scenarios.py`
5. `src/pricing_env_operational.py`
6. `src/train_ppo_operational.py`
7. `src/train_ppo_recovered_redesign.py`
8. `src/high_risk_b_planning_distillation.py`
9. `src/train_dqn_high_risk_b.py`
10. `src/dqn_statistical_audit_and_ensemble.py`
11. `src/evaluate_ppo_operational.py`
12. `src/final_dqn_ensemble_test_evaluation.py`
13. `src/final_test_baseline_ladder.py`
14. `src/final_report_figures.py`

## Training Versus Evaluation

Preprocessing and demand recovery prepare the local data. PPO and DQN scripts train models and may take substantial time. Locked evaluation scripts read existing artifacts and should not change training decisions. Reporting scripts read existing tables and figures to create selected outputs.

## Configs

- Required runtime records are mainly generated under `outputs/configs/`, including environment, PPO, DQN, manifest, and hash records.
- PPO training writes `outputs/configs/ppo_operational_experiment_config.json` and `outputs/configs/ppo_input_artifact_hashes.json`.
- DQN training writes `outputs/configs/dqn_locked_candidate.json` and related hash records.
- Held-out evaluation writes final protocol/hash records under `outputs/configs/`.
- Repository-level files under `configs/` are retained as lightweight records; locked runtime records should not be deleted or edited to change conclusions.

## Local Artifacts

Raw FreshRetailNet data, trained PPO/DQN binaries, vec-normalization files, and large intermediate outputs are excluded from Git. The archived `results/` files are enough to inspect the reported conclusions, but not enough to rerun training from scratch.

## Important Warning

Running training scripts can be compute-intensive and is not needed to inspect the archived results. Use the code and selected results first; rerun training only when local data and model-artifact requirements are available.
