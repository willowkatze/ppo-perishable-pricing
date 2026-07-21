"""Train the balanced recovered-calibration PPO redesign.

Purpose:
    Run the scenario-balanced PPO redesign experiment for recovered_financial
    only. This tests whether sparse positive-markdown opportunities were
    underrepresented during original training.
Implementation module:
    src.train_ppo_recovered_redesign
Required local inputs:
    outputs/tables/sustainability_scenario_audit.csv
    outputs/configs/ppo_validation_episode_manifest.csv
    outputs/tables/financial_markdown_oracle_diagnostic.csv
    outputs/tables/financial_markdown_multistep_action_values.csv
    outputs/tables/financial_markdown_ppo_regret.csv
Config files written or updated:
    outputs/models/ppo_training_redesign/*/training_metadata.json
    outputs/tables/ppo_redesign_sampling_weights.csv
Main outputs:
    outputs/models/ppo_training_redesign/
    outputs/tables/ppo_training_design_audit.csv
    outputs/tables/ppo_training_state_balance_audit.csv
    outputs/tables/ppo_multimetric_checkpoint_selection.csv
Trains a model:
    Yes, PPO.
Requires excluded local artifacts:
    Yes. Existing environment artifacts and model outputs are local.
Example command:
    python scripts/04_train_balanced_ppo.py --timesteps 20000
Runtime category:
    Compute-intensive.
Overwrite behavior:
    Balanced PPO checkpoints and redesign diagnostics may be overwritten.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from src.train_ppo_recovered_redesign import main


if __name__ == "__main__":
    main()
