# Reproducibility

Use the repository as an archived research workflow. Raw FreshRetailNet files must be downloaded separately and placed in the expected data directory. Do not commit raw data.

Important entry points include data processing, demand recovery, environment validation, PPO evaluation, planning evaluation, DQN training/evaluation, statistical audits, and final held-out test evaluation under `src/`.

The final held-out test should not be rerun for model selection. If rerun for verification, it must use the locked protocol and must not be followed by post-test tuning.

Reproducibility metadata are stored under `outputs/configs/`, including file inventory and checksum manifests where available.
