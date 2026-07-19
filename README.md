# Stockout-Aware Dynamic Markdown Decisions for Perishable Retail

**Demand Recovery, PPO Diagnostics, Planning, and DQN Evaluation**

This repository archives a completed reinforcement-learning course project on dynamic markdown decisions for perishable retail products. The project moved from an initial PPO-only prototype to a broader research workflow covering FreshRetailNet-informed demand recovery, a semi-synthetic perishable inventory environment, strong baselines, PPO diagnostics, limited-horizon planning, planning distillation, multi-seed DQN, equal-weight DQN ensembling, statistical-integrity checks, and one locked held-out test evaluation.

The final locked DQN ensemble did **not** outperform the no-markdown baseline on held-out test episodes. Positive validation signals did not generalize, and the project does not claim a successful baseline-beating learned markdown policy.

## 1. Project Overview

The project studies whether stockout-aware demand recovery can support better markdown decisions for perishable inventory. It uses FreshRetailNet-50K as the operational sales and stockout source, then constructs a semi-synthetic markdown environment because the public data do not include complete batch-level expiration, waste, replenishment, and cost information.

## 2. Research Motivation

Fresh retail sales are censored during stockouts. A naive policy can mistake stockout-limited sales for low demand. The project asks whether recovering latent demand and targeting high-risk perishable states can improve markdown decisions under a transparent accounting model.

## 3. Research Questions

1. Can stockout-aware demand recovery change the demand signal used by markdown policies?
2. Do PPO policies learn useful state-dependent markdown behavior or collapse to simple actions?
3. Can limited-horizon planning identify positive-markdown opportunities in locked high-risk states?
4. Can a learned DQN policy or ensemble transfer validation gains to a held-out test split?

## 4. Dataset and Data Limitations

The operational dataset is FreshRetailNet-50K from Dingdong-Inc on Hugging Face. Raw data are not included in this repository. Users must download the dataset separately according to its license and terms.

The source data provide detailed sales and stockout information, but do not provide complete batch-level expiration, realized waste, replenishment decisions, or all financial cost components needed for direct markdown-control evaluation. For that reason, perishability, waste, and financial accounting are modeled as semi-synthetic layers calibrated from available fields and documented assumptions.

## 5. Stockout-Aware Demand Recovery

The demand-recovery pipeline estimates latent demand for stockout-affected observations and compares observed-sales and recovered-demand calibration paths. Recovered demand is not ground-truth demand; it is a model-based estimate used for controlled downstream experiments.

## 6. Semi-Synthetic Perishable Inventory Environment

The environment simulates inventory aging, sell-through, markdown actions, revenue, accounting profit, and waste under locked assumptions. The environment is designed for comparative policy evaluation, not as a direct operational simulator of Dingdong's internal systems.

## 7. State Space

State variables include inventory risk, expiry risk, predicted demand, stockout-aware demand signals, episode timing, and scenario descriptors. High-risk task definitions are locked before final validation and test phases.

## 8. Six Discrete Markdown Actions

Policies choose among six discrete markdown actions. Action 0 corresponds to no markdown. Positive actions correspond to increasing markdown levels. The action space is fixed across PPO, planning, distillation, and DQN experiments.

## 9. Reward and Accounting Assumptions

The primary financial objective uses normalized accounting profit under the semi-synthetic environment. Waste and sell-through are reported as secondary operational metrics. The reward and accounting assumptions are locked before final model comparison and test evaluation.

## 10. Baselines

Baselines include always_0pct, fixed positive markdown policies, expiry-threshold rules, inventory-coverage rules, random policies, and planning/oracle diagnostics. The no-markdown baseline is intentionally retained as a strong conservative benchmark.

## 11. PPO Experiments

PPO was used as an important calibration-sensitivity benchmark, training-pathology diagnostic, and controlled scenario-balanced sampling experiment. PPO was not treated as the final successful model.

## 12. Scenario-Balanced PPO Redesign

Scenario-balanced PPO tested whether sparse positive-markdown opportunities were underrepresented during training. The redesign reduced some collapse symptoms but did not justify claiming a robust baseline-beating PPO policy.

## 13. Limited-Horizon Planning

Limited-horizon planning was evaluated on locked validation episodes to quantify whether dynamic markdown value existed under the environment assumptions. Planning showed modest positive validation value and supported further learned-policy experiments.

## 14. Planning Distillation

Planning labels were generated for high-risk states and used for focused learned-policy development. This phase was diagnostic and validation-locked; it did not use the held-out test split for tuning.

## 15. Multi-Seed DQN

DQN was trained across multiple seeds on the locked high-risk task. Seed-level validation results were audited for pseudo-replication risk and corrected with episode-clustered inference.

## 16. Equal-Weight Q-Value Ensemble

The final candidate before test was an equal-weight Q-value ensemble using selected DQN checkpoints. The ensemble was selected from validation evidence before the held-out test evaluation.

## 17. Validation Protocol

Validation used fixed paired episode manifests. Every policy was compared on identical episodes to avoid distributional advantages. The test split was not used during validation or model selection.

## 18. Statistical-Integrity Correction

An audit found that pooled seed-episode rows could overstate evidence by treating correlated rows as independent. The corrected inference used episode-level clustering and ensemble evaluation.

## 19. Final Held-Out Test

The final locked held-out test used 60 HIGH_RISK_B episodes with 60/60 valid pairings.

| Metric | DQN ensemble vs always_0pct |
|---|---:|
| DQN ensemble mean normalized profit | 0.425721 |
| always_0pct mean normalized profit | 0.447999 |
| paired mean gain | -0.022278 |
| paired median gain | -0.017321 |
| bootstrap 95% CI | [-0.027486, -0.017319] |
| win / tie / loss | 0.000 / 0.233 / 0.767 |
| waste-rate difference | -0.000204 |
| sell-through difference | +0.000204 |

## 20. Main Findings

The project found useful diagnostics and positive validation signals in targeted high-risk states, but the final locked DQN ensemble failed to beat always_0pct on held-out test episodes. The strongest scientific conclusion is about the difficulty of transferring validation markdown opportunities into robust learned policy value.

## 21. What Can Be Claimed

- Stockout-aware recovery changes the downstream demand signal.
- PPO revealed important collapse and calibration-sensitivity behavior.
- Limited-horizon planning found modest validation opportunities in locked high-risk states.
- Multi-seed DQN and ensembling produced validation signals that required strict test confirmation.
- The final held-out test did not confirm the learned ensemble as better than no markdown.

## 22. What Cannot Be Claimed

- The project cannot claim that PPO or DQN beat the held-out no-markdown baseline.
- The project cannot claim recovered demand is ground-truth demand.
- The project cannot claim dynamic markdown is ineffective in general.
- The project cannot claim the semi-synthetic environment reproduces the retailer's full internal operations.

## 23. Repository Structure

```text
src/                 source modules for data, environment, PPO, planning, DQN, audits, and reporting
tests/               lightweight non-training checks
outputs/tables/      key result tables and audit outputs
outputs/figures/     report-ready figures
outputs/configs/     locked protocols, manifests, and reproducibility metadata
outputs/models/      model metadata and artifact manifest; large binaries excluded by default
docs/                methodology, result interpretation, limitations, and archive audits
data/                placeholder directories only; raw data are excluded
archive/             local legacy work; data and large artifacts excluded from public Git
```

## 24. Setup

Create a Python environment and install the requirements:

```bash
pip install -r requirements.txt
```

Some experiments used reinforcement-learning libraries such as Stable-Baselines3 and PyTorch. Exact local package versions should be recorded in `environment.yml` or `requirements-lock.txt` for reproduction.

## 25. Reproduction Instructions

The repository is archived for inspection and selective reproduction. Do not rerun full training unless necessary.

Typical non-training checks:

```bash
python -m pytest tests
python src/final_report_figures.py --help
```

Full experimental reruns are computationally expensive and should follow the documented timeline in `docs/experiment_timeline.md`.

## 26. Data-Access Instructions

Download FreshRetailNet-50K separately from Hugging Face:

`Dingdong-Inc/FreshRetailNet-50K`

Place raw files under the expected local data directory described in `docs/data_and_assumptions.md`. Raw parquet files are intentionally excluded from GitHub.

## 27. Limitations

The main limitations are semi-synthetic perishability/accounting assumptions, model-based recovered demand, limited final test population, validation-to-test generalization failure, and dependence on locked scenario definitions.

## 28. Future Work

Future work should use richer batch-level expiration and replenishment data, externally validate cost assumptions, test broader task populations, and compare learned policies against operational decision rules under realistic constraints.

## 29. Citation

See `CITATION.cff` for this project citation metadata. FreshRetailNet-50K should be cited separately according to the dataset authors' instructions.

## 30. License

See `LICENSE`. Dataset licensing is separate from this repository's code and documentation license.
