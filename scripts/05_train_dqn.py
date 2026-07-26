"""Train the HIGH_RISK_B DQN comparison model.

Purpose:
    Train DQN candidates on the locked HIGH_RISK_B training setup and write
    validation summaries for later ensemble selection.
Implementation module:
    src.train_dqn_high_risk_b
Required local inputs:
    outputs/tables/high_risk_b_planning_training_labels.csv
    outputs/manifests/high_risk_b_expanded_validation_manifest.csv
    outputs/tables/limited_horizon_planning_episode_results.csv
Config files written or updated:
    outputs/configs/dqn_locked_candidate.json
    outputs/configs/dqn_locked_candidate_hashes.json
Main outputs:
    outputs/models/dqn_high_risk_b/
    outputs/tables/dqn_checkpoint_validation_episode_results.csv
    outputs/tables/dqn_checkpoint_validation_summary.csv
    outputs/tables/dqn_final_candidate_comparison.csv
    outputs/figures/dqn_final_candidate/
Trains a model:
    Yes, DQN.
Requires excluded local artifacts:
    Yes. Trained model binaries and local manifests are not stored in Git.
Example command:
    python scripts/05_train_dqn.py --timesteps 5000 --seeds 42
Runtime category:
    Compute-intensive for full multi-seed runs.
Overwrite behavior:
    DQN checkpoints, validation summaries, and candidate configs may be overwritten.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from src.train_dqn_high_risk_b import main


if __name__ == "__main__":
    main()
