"""balanced PPO training wrapper.

Expected inputs: validated scenarios and PPO configs.
Produced outputs: balanced PPO checkpoints and redesign outputs.
Config path: configs/ppo.
Example command: python scripts/04_train_balanced_ppo.py
Execution note: trains PPO when executed.
"""

from runpy import run_module


if __name__ == "__main__":
    run_module("src.train_ppo_recovered_redesign", run_name="__main__")
