"""Train the original operational PPO policies.

Purpose:
    Run the original PPO training design for observed/recovered calibration
    agents. Re-running it is not needed to inspect the archived final results.
Implementation module:
    src.train_ppo_operational
Required local inputs:
    data/freshretail/processed/freshretail_demand_recovered.parquet
    outputs/configs/pricing_env_operational_config.json
    outputs/configs/perishability_master_config.json
    outputs/models/discount_response_observed.joblib
    outputs/models/discount_response_recovered.joblib
    outputs/models/discount_response_config.json
Config files written or updated:
    outputs/configs/ppo_operational_experiment_config.json
    outputs/configs/ppo_input_artifact_hashes.json
Main outputs:
    outputs/models/ppo_operational/
    outputs/tables/ppo_training_diagnostics.csv
    outputs/tables/ppo_training_curves.csv
    outputs/tables/ppo_episode_level_evaluation.csv
Trains a model:
    Yes, PPO.
Requires excluded local artifacts:
    Yes. Trained model binaries and local data are not stored in Git.
Example command:
    python scripts/03_train_original_ppo.py --stage smoke --timesteps 2000
Runtime category:
    Compute-intensive for pilot/main stages.
Overwrite behavior:
    PPO checkpoints, monitor logs, and training summaries may be overwritten.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from src.train_ppo_operational import main


if __name__ == "__main__":
    main()
