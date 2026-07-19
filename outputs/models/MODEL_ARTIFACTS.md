# Model Artifacts

Model binaries are excluded from ordinary public Git commits by default. This protects the repository from redundant checkpoints and keeps the public archive lightweight.

## Final Locked Learned Candidate

The final validation-selected candidate was:

`STANDARD_DQN_EQUAL_WEIGHT_Q_ENSEMBLE`

Components:

- `STANDARD_DQN__seed_42__20000`
- `STANDARD_DQN__seed_123__15000`
- `STANDARD_DQN__seed_456__25000`

The final held-out test did not confirm this ensemble as better than `always_0pct`.

## Artifact Policy

- Raw datasets are excluded.
- Redundant PPO and DQN checkpoints are excluded.
- If model binaries are required, store them with Git LFS or an external artifact release.
- Use `outputs/configs/checksum_manifest.sha256` to verify local artifacts.
- Keep locked configs, protocol JSON files, and model-selection metadata under version control.

## Expected Local Artifact Families

- `outputs/models/dqn_high_risk_b/STANDARD_DQN/seed_42/`
- `outputs/models/dqn_high_risk_b/STANDARD_DQN/seed_123/`
- `outputs/models/dqn_high_risk_b/STANDARD_DQN/seed_456/`
- `outputs/models/planning_distillation/`
- `outputs/models/ppo_operational/`

These paths are intentionally relative and do not encode machine-specific usernames.
