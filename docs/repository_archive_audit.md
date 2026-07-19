# Repository Archive Audit

Project root: `ppo_perishable_pricing_project`.

Git status note: local project directory does not currently contain `.git`, and `git` is not available in the PowerShell PATH used by Codex. Remote branch creation and push therefore require Git/GitHub Desktop on the user machine or a separate clone step.

Total files scanned: 725
Total size: 1953.73 MB
Candidate commit files: 624
Excluded files: 39
Manual-review files: 68

## Largest Files
| Path | Size MB | Decision | Reason/Purpose |
|---|---:|---|---|
| data\operational\raw\freshretail\raw\freshretail_train.parquet | 1696.304 | NO - raw dataset | raw dataset, exclude from public repo |
| data\operational\raw\freshretail\raw\freshretail_eval.parquet | 132.378 | NO - raw dataset | raw dataset, exclude from public repo |
| archive\prototype_kaggle\data\processed\perishable_goods_clean.csv | 24.453 | NO - legacy dataset copy | legacy/prototype archive, mostly exclude except documentation if needed |
| archive\prototype_kaggle\data\raw\kaggle_source\perishable_goods_management.csv | 20.118 | NO - legacy dataset copy | legacy/prototype archive, mostly exclude except documentation if needed |
| outputs\tables\latent_demand_validation_predictions.csv | 10.494 | YES/MAYBE - candidate after secret review | result table |
| outputs\models\freshretail_latent_demand_model.joblib | 5.496 | YES/MAYBE - candidate after secret review | model artifact or metadata |
| archive\prototype_kaggle\outputs\tables\ppo_financial_training_episodes.csv | 5.074 | YES/MAYBE - candidate after secret review | legacy/prototype archive, mostly exclude except documentation if needed |
| archive\prototype_kaggle\outputs\tables\ppo_financial_action_distribution.csv | 4.208 | YES/MAYBE - candidate after secret review | legacy/prototype archive, mostly exclude except documentation if needed |
| archive\prototype_kaggle\outputs\tables\baseline_episode_results.csv | 1.913 | YES/MAYBE - candidate after secret review | legacy/prototype archive, mostly exclude except documentation if needed |
| data\freshretail\processed\freshretail_demand_recovered.parquet | 1.599 | NO - derived dataset | processed data, exclude unless tiny documented sample |
| outputs\models\planning_distillation\best_planning_distilled_model.pkl | 0.98 | YES/MAYBE - candidate after secret review | model artifact or metadata |
| archive\prototype_kaggle\outputs\tables\counterfactual_discount_predictions.csv | 0.778 | YES/MAYBE - candidate after secret review | legacy/prototype archive, mostly exclude except documentation if needed |
| outputs\models\discount_response_recovered.joblib | 0.651 | YES/MAYBE - candidate after secret review | model artifact or metadata |
| outputs\models\discount_response_observed.joblib | 0.651 | YES/MAYBE - candidate after secret review | model artifact or metadata |
| outputs\models\ppo_operational\observed_financial\seed_42\checkpoints\pilot_observed_financial_22288_steps.zip | 0.545 | MAYBE - essential model only, preferably LFS or external artifact manifest | model artifact or metadata |
| outputs\models\ppo_operational\observed_financial\seed_42\checkpoints\pilot_observed_financial_27288_steps.zip | 0.545 | MAYBE - essential model only, preferably LFS or external artifact manifest | model artifact or metadata |
| outputs\models\ppo_training_redesign\balanced\seed_123\final_model.zip | 0.545 | MAYBE - essential model only, preferably LFS or external artifact manifest | model artifact or metadata |
| outputs\models\ppo_operational\recovered_financial\seed_42\final_model.zip | 0.545 | MAYBE - essential model only, preferably LFS or external artifact manifest | model artifact or metadata |
| outputs\models\ppo_training_redesign\balanced\seed_123\checkpoints\redesign_balanced_10000_steps.zip | 0.545 | MAYBE - essential model only, preferably LFS or external artifact manifest | model artifact or metadata |
| outputs\models\ppo_operational\observed_financial\seed_42\best_model.zip | 0.545 | MAYBE - essential model only, preferably LFS or external artifact manifest | model artifact or metadata |
| outputs\models\ppo_operational\observed_financial\seed_42\final_model.zip | 0.545 | MAYBE - essential model only, preferably LFS or external artifact manifest | model artifact or metadata |
| outputs\models\ppo_operational\observed_financial\seed_42\checkpoints\pilot_observed_financial_17288_steps.zip | 0.545 | MAYBE - essential model only, preferably LFS or external artifact manifest | model artifact or metadata |
| outputs\models\ppo_training_redesign\balanced\seed_123\checkpoints\redesign_balanced_15000_steps.zip | 0.545 | MAYBE - essential model only, preferably LFS or external artifact manifest | model artifact or metadata |
| outputs\models\ppo_operational\recovered_financial\seed_42\best_model.zip | 0.545 | MAYBE - essential model only, preferably LFS or external artifact manifest | model artifact or metadata |
| outputs\models\ppo_operational\recovered_financial\seed_42\checkpoints\pilot_recovered_financial_32288_steps.zip | 0.545 | MAYBE - essential model only, preferably LFS or external artifact manifest | model artifact or metadata |

## Archive Policy
- Exclude raw FreshRetailNet parquet files and legacy Kaggle raw/processed copies from public GitHub.
- Exclude caches, temporary files, local IDE metadata, checkpoints not required for scientific claims, and regenerated intermediate datasets.
- Commit source code, tests, documentation, selected configs, selected manifests, scientifically important tables, and report-ready figures.
- Essential binary models should either use Git LFS or be represented by hashes and a model artifact manifest.

## Full Inventory CSV
A machine-readable inventory was generated at `outputs/configs/repository_file_inventory.csv`.
