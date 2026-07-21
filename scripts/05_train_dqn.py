"""DQN training wrapper.

Expected inputs: HIGH_RISK_B artifacts and DQN configs.
Produced outputs: DQN checkpoints and validation summaries.
Config path: configs/dqn.
Example command: python scripts/05_train_dqn.py
Execution note: trains DQN when executed.
"""

from runpy import run_module


if __name__ == "__main__":
    run_module("src.train_dqn_high_risk_b", run_name="__main__")
