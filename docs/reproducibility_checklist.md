# Reproducibility Checklist

| item | status | evidence | issue | correction |
|---|---|---|---|---|
| Pinned dependencies | PARTIAL | requirements.txt; environment.yml | Versions are listed but not fully locked with hashes. | Pin exact versions before final release if required. |
| Saved configs | PASS | configs/ | JSON configs are committed. | None. |
| Random seeds recorded | PASS | configs/ and result tables | Seeds are present in experiment configs/results. | None. |
| Data splits documented | PASS | outputs/manifests/; configs/evaluation/ | Validation and held-out manifests are committed. | None. |
| Training scripts separate from notebooks | PASS | scripts/03_train_ppo.py; scripts/04_train_balanced_ppo.py; scripts/05_train_dqn.py | Training wrappers are separate. | None. |
| Evaluation scripts do not train automatically | PASS | scripts/06_evaluate_models.py; scripts/07_generate_report_outputs.py | Evaluation/reporting wrappers are separate. | None. |
| README commands valid | PASS | README.md | Setup commands are simple and non-training. | None. |
| Relative paths | PARTIAL | src/ and docs/ | Some local artifact paths may exist in archived diagnostics, not main docs. | Keep main docs path-relative. |
| No raw data committed | PASS | git-tracked scan | No raw data files are tracked. | None. |
| No secrets committed | PASS | secret scan | Only ordinary variable names matched. | None. |
| Model artifacts documented | PASS | outputs/models/MODEL_ARTIFACTS.md | Model binaries are excluded and documented. | None. |
| Tag/release submitted version | PARTIAL | GitHub branch/PR | Tag/release not created by this audit. | Create release after final merge. |
| Key outputs identify script/config | PARTIAL | docs/report_evidence_map.md | Some archived diagnostics lack explicit source metadata. | Use selected main outputs in report. |
