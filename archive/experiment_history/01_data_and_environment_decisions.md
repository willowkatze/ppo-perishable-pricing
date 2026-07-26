# Data and Environment Decisions

This file consolidates earlier experiment notes and keeps the order of major decisions without repeating the main result tables.

## Source: data_and_assumptions.md

锘? Data and Assumptions

The project uses FreshRetailNet-50K as the source of operational sales, promotion, and stockout information. Raw data are excluded from the repository and must be downloaded separately from Hugging Face.

Recovered demand is a model-based estimate, not ground truth. Perishability, waste, batch aging, and financial costs are semi-synthetic because the source data do not contain complete batch-level expiration, realized waste, replenishment, disposal, and cost accounting fields.

Public GitHub archive rule: keep source code, documentation, selected tables, report-ready figures, configs, manifests, and hashes; exclude raw datasets and large generated data artifacts.



## Source: experiment_timeline.md

锘? Experiment Timeline

1. Original dataset assessment found the initial Kaggle-style data insufficient for the final research design.
2. The project transitioned to FreshRetailNet-informed operational data.
3. Stockout-aware latent demand recovery was developed.
4. Observed-sales and recovered-demand PPO variants were compared.
5. PPO action collapse and calibration sensitivity were diagnosed.
6. Scenario-balanced PPO redesign tested whether rare positive-markdown opportunities were underrepresented.
7. Strong baseline policies were evaluated.
8. HIGH_RISK_B was locked as the final high-risk task population.
9. Limited-horizon planning evaluated dynamic markdown value on paired validation episodes.
10. Planning distillation created focused training labels for high-risk states.
11. Three-seed DQN training was performed.
12. Pseudoreplication in pooled seed-episode inference was corrected.
13. An equal-weight DQN Q-value ensemble was selected from validation evidence.
14. A locked held-out test was run once.
15. The final conclusion was negative: validation gains did not generalize to held-out test episodes.



## Source: limitations.md

锘? Limitations

- The environment is semi-synthetic and depends on explicit perishability and accounting assumptions.
- Recovered demand is estimated, not observed ground truth.
- The final held-out test population is limited to locked HIGH_RISK_B episodes.
- Validation gains did not generalize to held-out test performance.
- Raw FreshRetailNet data and large generated data artifacts are excluded from the public repository.
- Model binaries may require external artifact storage or Git LFS depending on final archival policy.



## Source: research_design.md

锘? Research Design

This project evaluates stockout-aware dynamic markdown decisions for perishable retail under a semi-synthetic environment.

The design follows a locked, staged workflow: data provenance audit, FreshRetailNet-informed demand recovery, perishability scenario construction, environment validation, baseline evaluation, PPO diagnostics, scenario-balanced PPO redesign, limited-horizon planning, planning distillation, DQN training, multi-seed statistical audit, equal-weight ensemble selection, and one held-out test evaluation.

The final decision criterion is not whether a learned policy looks complex, but whether it improves paired normalized accounting profit against a locked no-markdown baseline on held-out episodes without leakage or post-test tuning.


