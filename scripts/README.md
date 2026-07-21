# Numbered Script Entry Points

Run scripts from the repository root. They wrap existing `src/` modules and do not duplicate implementation code.

1. `python scripts/01_prepare_data.py` prepares local FreshRetailNet-derived data.
2. `python scripts/02_recover_demand.py` runs stockout-aware demand recovery.
3. `python scripts/03_train_ppo.py` trains original PPO. Optional and compute-intensive.
4. `python scripts/04_train_balanced_ppo.py` trains balanced PPO. Optional and compute-intensive.
5. `python scripts/05_train_dqn.py` trains DQN. Optional and compute-intensive.
6. `python scripts/06_evaluate_models.py` evaluates locked model artifacts.
7. `python scripts/07_generate_report_outputs.py` regenerates final report outputs without retraining by design.

Teacher review usually requires only `START_HERE.md`, `README.md`, `docs/`, and `results/`.
