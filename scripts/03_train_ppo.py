"""PPO training wrapper.

Expected inputs: environment configs and local artifacts.
Produced outputs: PPO checkpoints and validation outputs.
Config path: configs/ppo.
Example command: python scripts/03_train_ppo.py
Execution note: trains PPO when executed.
"""

from runpy import run_module


if __name__ == "__main__":
    run_module("src.train_ppo_operational", run_name="__main__")
