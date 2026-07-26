"""Evaluate the locked DQN ensemble on the held-out HIGH_RISK_B test set.

Purpose:
    Run the one-time locked DQN ensemble held-out test evaluation. This script
    does not train or select models.
Implementation module:
    src.final_dqn_ensemble_test_evaluation
Required local inputs:
    outputs/configs/dqn_equal_weight_ensemble_candidate.json
    outputs/models/dqn_high_risk_b/STANDARD_DQN/
    outputs/manifests/high_risk_b_final_test_manifest.csv or local validation
    artifacts from which the locked manifest is reconstructed
Config files written or updated:
    outputs/configs/final_dqn_ensemble_test_protocol.json
    outputs/configs/final_dqn_ensemble_test_hashes.json
Main outputs:
    outputs/tables/final_dqn_ensemble_test_episode_results.csv
    outputs/tables/final_dqn_ensemble_test_summary.csv
    outputs/tables/final_dqn_ensemble_test_pairing_audit.csv
    outputs/figures/final_dqn_ensemble_test/
Trains a model:
    No.
Requires excluded local artifacts:
    Yes. Locked DQN model binaries are not stored in Git.
Example command:
    python scripts/06_evaluate_locked_dqn_ensemble.py
Runtime category:
    Moderate.
Overwrite behavior:
    Held-out DQN test tables and figures may be overwritten.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from src.final_dqn_ensemble_test_evaluation import main


if __name__ == "__main__":
    main()
