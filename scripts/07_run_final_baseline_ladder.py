"""Run the final held-out baseline-ladder comparison.

Purpose:
    Evaluate the fixed, random, rule-based, PPO, balanced PPO, and locked DQN
    policies on the same 60 HIGH_RISK_B held-out test episodes. This is a
    post-hoc descriptive comparison with always_0pct retained as the primary
    reference.
Implementation module:
    src.final_test_baseline_ladder
Required local inputs:
    outputs/manifests/high_risk_b_final_test_manifest.csv
    outputs/tables/final_dqn_ensemble_test_episode_results.csv
    outputs/models/ppo_operational/
    outputs/models/ppo_training_redesign/
    outputs/models/dqn_high_risk_b/
Config files:
    Uses locked model metadata under outputs/configs/ and model directories.
Main outputs:
    outputs/tables/final_test_baseline_ladder.csv
    outputs/tables/final_test_model_vs_baseline_matrix.csv
    outputs/tables/final_test_baseline_pairwise_comparisons.csv
    outputs/figures/final_test_baseline_ladder/
Trains a model:
    No.
Requires excluded local artifacts:
    Yes. Locked model binaries are not stored in Git.
Example command:
    python scripts/07_run_final_baseline_ladder.py
Runtime category:
    Moderate.
Overwrite behavior:
    Final baseline-ladder tables and figures may be overwritten.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from src.final_test_baseline_ladder import main


if __name__ == "__main__":
    main()
