# Final Repository Audit

This file consolidates the course requirement review, result consistency review, table review, figure review, language review, reproducibility review, known corrections, and final repository status.

Final status: the project has six main documentation files, five main result tables, seven main result figures, archived supporting material, no committed raw data, and no committed model binaries. The full 13-policy held-out baseline ladder remains a main result.

## Source: docs/course_requirements_audit.md


# Course Requirements Audit

| requirement | source | current repository evidence | status | exact issue | required correction | relevant file paths |
|---|---|---|---|---|---|---|
| Problem definition | Lecture 0 report requirements | README.md; START_HERE.md; docs/01_project_and_research_design.md | PASS | Problem is stated as perishable markdown decision-making. | Keep concise problem statement. | README.md; START_HERE.md |
| Motivation | Lecture 0 report requirements | START_HERE.md; docs/01_project_and_research_design.md | PASS | Motivation is tied to stockout censoring and perishability. | None. | docs/01_project_and_research_design.md |
| Prior work | Lecture 0 report requirements | FreshRetailNet paper and general RL/dynamic pricing framing are referenced at a high level. | PARTIAL | Formal literature citations are not fully written in repo docs. | Add polished citations in the final report text. | docs/report_evidence_map.md |
| Methodology | Lecture 0 report requirements | docs/02_data_and_demand_recovery.md; docs/03_environment_and_methods.md; docs/04_results.md | PASS | Methods are linked to code and configs. | None. | src/; configs/ |
| Numerical studies | Lecture 0 report requirements | results/final_summary.csv; results/tables/ | PASS | Final values trace to source tables. | Use selected tables only in report. | results/ |
| Conclusions | Lecture 0 report requirements | docs/04_results.md | PASS | Conclusion states always_0pct strongest overall and balanced PPO strongest learned. | None. | docs/04_results.md |
| Limitations | Lecture 0 report requirements | docs/05_conclusions_and_limitations.md | PASS | Limitations include semi-synthetic environment and negative held-out baseline result. | None. | docs/05_conclusions_and_limitations.md |
| Source code availability | Lecture 2 GitHub expectations | src/ and scripts/ | PASS | Source modules and wrappers are committed. | None. | src/; scripts/ |
| Reproducibility | Lecture 2 checklist | README.md; scripts/README.md; docs/reproducibility.md | PARTIAL | Full reruns require excluded raw data and model artifacts. | Document artifact requirements clearly. | docs/reproducibility.md |
| Dependency pinning | Lecture 2 checklist | requirements.txt; environment.yml | PARTIAL | requirements.txt exists but has loose minimum-style dependencies; environment.yml exists. | Pin exact versions before final release if required. | requirements.txt; environment.yml |
| Configuration files | Lecture 2 checklist | configs/ | PASS | JSON configs are saved. | None. | configs/ |
| Random seeds | Lecture 2 checklist | configs/ and archived tables | PASS | Seeds appear in DQN/PPO configs and result tables. | None. | configs/ |
| Data splits | Lecture 2 checklist | outputs/manifests/; configs/evaluation/ | PASS | Validation and held-out manifests are saved. | None. | outputs/manifests/ |
| Training/evaluation separation | Lecture 2 checklist | scripts/ | PASS | Training wrappers are separate from evaluation/reporting wrappers. | None. | scripts/ |
| README usability | Lecture 2 checklist | README.md; START_HERE.md | PASS | Reader has entry points and setup instructions. | None. | README.md |
| Final result traceability | Lecture 2 checklist | outputs/tables/final_result_consistency_audit.csv | PASS | Each main result has source table/code/split. | None. | outputs/tables/final_result_consistency_audit.csv |
| Presentation readiness | Lecture 0 presentation | docs/final_content_selection.md | PASS | Presentation figures are limited to 5. | Use selected figures only. | docs/final_content_selection.md |


## Source: docs/language_style_audit.md


# Language Style Audit

The primary documentation was checked for generated-sounding phrases such as `main`, `canonical`, `scientific position`, `decision record`, `shows`, and similar wording.

| file | original wording | why it sounded unnatural or vague | simpler replacement |
|---|---|---|---|
| README.md | main archive | The phrase sounds like the repository is speaking directly to the course requirement instead of summarizing the project. | concise project archive / reinforcement-learning course project |
| results/README.md | Main Results | The label sounded artificial. | Key Results |
| docs/repository_consolidation_audit.md | main main file | Internal audit label, not natural report language. | main review file |
| docs/consolidation_integrity_check.md | main consolidation/archive | Internal phrasing. | project-results consolidation / review-ready archive |
| archive/intermediate_tables/dqn_canonical_candidate_selection.csv | canonical | Internal model-selection wording; not used in main report. | archived candidate-selection table |

No remaining primary document uses `main` or `project`. Technical terms such as held-out test, paired evaluation, confidence interval, policy collapse, demand censoring, and semi-synthetic environment were retained.


## Source: docs/file_rename_map.md


锘? File Rename and Move Map

| original path | new path | reason |
|---|---|---|
| results/tables/final_method_comparison.csv | archive/intermediate_tables/final_method_comparison.csv | reduce main result set to course-sized report selection |
| results/tables/dqn_validation_summary.csv | archive/intermediate_tables/dqn_validation_summary.csv | reduce main result set to course-sized report selection |
| results/figures/08_project_method_timeline_proxy.png | archive/diagnostic_figures/final_report/figure_ppo_validation_return_over_training.png | reduce main result set to course-sized report selection |
| results/README.md heading: Main Results | results/README.md heading: Key Results | Avoid unnatural audience-specific wording. |
| docs/repository_consolidation_audit.md column: main main file | docs/repository_consolidation_audit.md column: main review file | Use ordinary review language. |
| archive/intermediate_tables/dqn_canonical_candidate_selection.csv | not renamed | Archived internal filename; not used in main report, so renaming could reduce traceability. |

| results/figures/05_final_heldout_baseline_ladder.png | archive/diagnostic_figures/final_test_baseline_ladder/05_final_heldout_baseline_ladder_previous.png | Replaced by clearer full 13-policy comparison figure. |
| archive/intermediate_tables/final_heldout_baseline_ladder_full_source.csv | results/tables/final_heldout_baseline_ladder.csv | Full source retained in archive; main table rewritten as concise report table without changing values. |



## Source: docs/final_course_alignment_review.md


锘? Final Course Alignment Review

## Course Requirement Coverage

The repository now maps directly to the expected report sections: problem and motivation, prior work, methodology, numerical studies, conclusions, and limitations. The main evidence map is `docs/report_evidence_map.md`.

## Remaining Missing Requirement

Prior work is only summarized at a high level in the repository. The final 8-12 page report should add a polished citation paragraph for FreshRetailNet, stockout-censored demand estimation, dynamic pricing, and RL for pricing/inventory.

## Incorrect Tables Found

One reporting inconsistency was found and corrected earlier: `results/tables/final_heldout_baseline_ladder.csv` now uses the locked DQN held-out bootstrap CI [-0.027486, -0.017319]. The same table has also been restored as the full 13-policy held-out comparison, including all fixed-markdown, random, rule-based, PPO, and DQN policies. Supporting DQN validation rows were clarified as repeated seed-episode observations, not independent episodes.

## Incorrect Figures Found

No retained main figure was corrected. One extra timeline-style figure was moved out of the main figure set because it was less directly useful for the report.

## Obsolete or Duplicated Files Found

The repository had more result files than needed for the final report. Supporting tables and figures remain archived instead of deleted.

## Language Issues Found

Audience-specific and generated-sounding labels such as `main` were replaced with plain labels such as `key results`, `main review file`, and `project-results consolidation`.

## Corrections Made

- Main tables reduced to five report tables plus `results/final_summary.csv`.
- Main figures reduced to seven.
- Course requirement audit added.
- Result consistency audit added.
- Table and figure quality audits added.
- Report evidence map added.
- Reproducibility checklist added.

## Files Archived

Two supporting result tables and one supporting figure were moved from `results/` back to `archive/`. The full held-out baseline ladder remains in `results/tables/` because it is a main report result.

## Final Main Tables

1. `results/tables/demand_recovery_summary.csv`
2. `results/tables/ppo_original_vs_balanced_comparison.csv`
3. `results/tables/final_heldout_baseline_ladder.csv`
4. `results/tables/dqn_heldout_test_summary.csv`
5. `results/tables/final_profit_waste_pareto_frontier.csv`

## Final Main Figures

1. `results/figures/01_observed_vs_recovered_demand.png`
2. `results/figures/02_recovered_demand_adjustment_distribution.png`
3. `results/figures/03_original_vs_balanced_ppo_error_decomposition.png`
4. `results/figures/04_ppo_action_collapse_diagnostic.png`
5. `results/figures/heldout_policy_comparison.png`
6. `results/figures/06_learned_policies_vs_fixed_markdown.png`
7. `results/figures/07_profit_waste_pareto_frontier.png`

## PPO/DQN Positioning

PPO remains the main course-aligned RL model. It covers demand-calibration comparison, action-collapse diagnosis, and balanced-training redesign. Balanced recovered PPO is the strongest learned policy on held-out test.

DQN is retained as an additional value-based comparison for the six-action discrete action space. It had a positive validation signal but was weaker than balanced recovered PPO and did not beat always_0pct on held-out test.

## Reproducibility Status

Reproducibility is mostly complete for repository review: source code, configs, seeds, splits, scripts, and key outputs are committed. Full reruns still require excluded raw data and model artifacts. A final GitHub release/tag remains a manual submission step.

## Readiness

The repository is ready to support an 8-12 page final report and a 10-minute presentation, provided the final written report adds a concise prior-work section and uses only the selected main tables and figures.




## Source: docs/report_evidence_map.md


锘? Report Evidence Map

| report section | main claim | supporting table | supporting figure | supporting code | supporting config | relevant document | unresolved gap |
|---|---|---|---|---|---|---|---|
| Problem definition and motivation | Stockouts censor observed demand, and perishability creates markdown tradeoffs. | results/tables/demand_recovery_summary.csv | results/figures/01_observed_vs_recovered_demand.png | src/freshretail_data_processing.py | configs/evaluation/final_test_evaluation_protocol.json | docs/01_project_and_research_design.md | Final report should add polished motivation prose. |
| Prior work | FreshRetailNet motivates stockout-aware demand recovery; RL/dynamic pricing motivates policy learning. | results/tables/demand_recovery_summary.csv | results/figures/01_observed_vs_recovered_demand.png | src/latent_demand_recovery.py | configs/evaluation/final_test_evaluation_protocol.json | docs/02_data_and_demand_recovery.md | Formal citation paragraph still belongs in the final report. |
| Data and demand recovery | Demand recovery is used before discount-response and environment calibration. | results/tables/demand_recovery_summary.csv | results/figures/02_recovered_demand_adjustment_distribution.png | src/latent_demand_recovery.py | configs/evaluation/final_test_evaluation_protocol.json | docs/02_data_and_demand_recovery.md | Raw data is excluded from GitHub. |
| Methodology | PPO is the main course-aligned RL model; DQN is an additional comparison. | results/tables/ppo_original_vs_balanced_comparison.csv | results/figures/03_original_vs_balanced_ppo_error_decomposition.png | src/train_ppo_operational.py; src/train_ppo_recovered_redesign.py | configs/ppo/ppo_operational_experiment_config.json | docs/04_results.md | None. |
| Numerical studies | Balanced recovered PPO was the strongest learned policy on the held-out test set. It outperformed all evaluated positive fixed-markdown policies, random_uniform, two simple rule-based policies, the original PPO variants, and the locked DQN ensemble. However, always_0pct remained the strongest overall policy. | results/tables/final_heldout_baseline_ladder.csv | results/figures/heldout_policy_comparison.png | src/final_test_baseline_ladder.py | configs/evaluation/final_test_evaluation_protocol.json | docs/04_results.md | None. |
| Conclusions | No learned policy beat always_0pct on held-out test. | results/final_summary.csv | results/figures/07_profit_waste_pareto_frontier.png | src/final_test_baseline_ladder.py | configs/evaluation/final_test_evaluation_protocol.json | docs/04_results.md | None. |
| Limitations | Environment is semi-synthetic and full reproduction requires excluded artifacts. | results/final_summary.csv | results/figures/07_profit_waste_pareto_frontier.png | src/pricing_env_operational.py | configs/environment/pricing_env_operational_config.json | docs/05_conclusions_and_limitations.md | Final submitted release/tag still needs to be created manually. |

## Baseline Ladder Traceability

The full baseline ladder uses `results/tables/final_heldout_baseline_ladder.csv`, which is derived from the existing locked held-out baseline-ladder artifact archived at `archive/intermediate_tables/final_heldout_baseline_ladder_full_source.csv`. The report figure is `results/figures/heldout_policy_comparison.png`. The generation script is `src/final_test_baseline_ladder.py`, the locked test manifest is `outputs/manifests/high_risk_b_final_test_manifest.csv`, the population is HIGH_RISK_B, the calibration is recovered_calibration, and no retraining or test rerun was performed for this restoration.


## Source: docs/final_content_selection.md


锘? Final Content Selection

## Main Report Tables

| item | why included | question answered | exact source | use |
|---|---|---|---|---|
| Data and demand-recovery summary | Shows the data-processing basis for later modeling. | How was stockout-censored demand handled? | results/tables/demand_recovery_summary.csv | report and presentation |
| PPO results | Summarizes original versus balanced PPO behavior. | Did balanced training improve learned policy behavior? | results/tables/ppo_original_vs_balanced_comparison.csv | report and presentation |
| Final baseline comparison | Main held-out comparison across all 13 fixed, random, rule-based, PPO, and DQN policies; no policy is omitted based on performance. | Which policy performed best on the locked 60-episode HIGH_RISK_B held-out test? | results/tables/final_heldout_baseline_ladder.csv | report and presentation |
| DQN held-out result | Documents the value-based follow-up result. | Did DQN generalize on held-out test? | results/tables/dqn_heldout_test_summary.csv | report |
| Profit-waste comparison | Shows the tradeoff behind the final interpretation. | What is the profit-waste frontier? | results/tables/final_profit_waste_pareto_frontier.csv | report |

## Main Report Figures

| item | why included | question answered | exact source | use |
|---|---|---|---|---|
| Observed versus recovered demand | Explains why demand recovery is needed. | How did recovered demand differ from observed demand? | results/figures/01_observed_vs_recovered_demand.png | report and presentation |
| Demand adjustment distribution | Shows recovery scale. | How large were the adjustments? | results/figures/02_recovered_demand_adjustment_distribution.png | report |
| PPO behavior comparison | Shows redesign effect. | How did balanced PPO change decision errors? | results/figures/03_original_vs_balanced_ppo_error_decomposition.png | report and presentation |
| PPO action-collapse diagnosis | Explains the original PPO limitation. | Did PPO collapse to no markdown? | results/figures/04_ppo_action_collapse_diagnostic.png | report |
| Held-out baseline comparison | Main final result. | Which policy had highest held-out profit? | results/figures/heldout_policy_comparison.png | report and presentation |
| Learned versus fixed markdowns | Supports the secondary claim. | Which weaker baselines did learned policies beat? | results/figures/06_learned_policies_vs_fixed_markdown.png | report |
| Profit-waste frontier | Shows tradeoff and limitation. | What was gained or lost in profit-waste terms? | results/figures/07_profit_waste_pareto_frontier.png | report and presentation |

The presentation should use at most five figures: demand recovery, PPO behavior comparison, held-out baseline comparison, profit-waste frontier, and one workflow slide created in the presentation itself.




## Source: docs/repository_consolidation_audit.md


锘? Repository Consolidation Audit

Generated before moving files on 2026-07-21 18:14:30.

## Findings

- Total non-git files inventoried: 534
- Markdown documents in docs before consolidation: 22
- Duplicate binary/content groups by SHA-256: 9
- Outdated PPO-only framing was consolidated into model-role documents.
- Old Kaggle wording was checked; consolidated docs use FreshRetailNet-informed wording.
- Raw data and model binaries are not intended for GitHub.

## Complete File Inventory

| file path | file type | size | purpose | main review file | supporting material | duplicated | move to archive | obsolete | files that reference it |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| .gitignore | unknown | 1643 | supporting project artifact | False | True | False | False | False | docs/security_and_file_size_audit.md; outputs/configs/artifact_manifest.csv; outputs/configs/repository_file_inventory.csv |
| CITATION.cff | config | 558 | supporting project artifact | False | True | False | False | False | README.md; docs/archive_integrity_check.md; outputs/configs/artifact_manifest.csv |
| LICENSE | unknown | 1187 | supporting project artifact | False | True | False | False | False | README.md; docs/archive_integrity_check.md; outputs/configs/artifact_manifest.csv |
| README.md | markdown | 10493 | project documentation | True | True | False | False | False | pyproject.toml; outputs/configs/artifact_manifest.csv; outputs/configs/repository_file_inventory.csv; outputs/tables/freshretail_discount_semantics_audit.csv |
| data/operational/processed/.gitkeep | unknown | 5 | supporting project artifact | False | True | True | False | False | outputs/configs/artifact_manifest.csv; outputs/configs/repository_file_inventory.csv |
| data/processed/.gitkeep | unknown | 2 | supporting project artifact | False | True | True | False | False | outputs/configs/artifact_manifest.csv; outputs/configs/repository_file_inventory.csv |
| data/raw/.gitkeep | unknown | 2 | supporting project artifact | False | True | True | False | False | outputs/configs/artifact_manifest.csv; outputs/configs/repository_file_inventory.csv |
| data/raw/README.md | markdown | 884 | supporting project artifact | False | True | False | False | False | pyproject.toml; outputs/configs/artifact_manifest.csv; outputs/configs/repository_file_inventory.csv; outputs/tables/freshretail_discount_semantics_audit.csv |
| docs/archive_integrity_check.md | markdown | 1602 | project documentation | False | True | False | False | False | outputs/configs/artifact_manifest.csv |
| docs/data_and_assumptions.md | markdown | 700 | project documentation | False | True | False | False | False | README.md; outputs/configs/artifact_manifest.csv; data/raw/README.md |
| docs/dqn_final_statistical_decision.md | markdown | 3001 | project documentation | False | True | False | False | False | src/dqn_statistical_audit_and_ensemble.py; outputs/configs/artifact_manifest.csv; outputs/configs/repository_file_inventory.csv |
| docs/dqn_results.md | markdown | 447 | project documentation | False | True | False | False | False | outputs/configs/artifact_manifest.csv |
| docs/experiment_timeline.md | markdown | 1128 | project documentation | False | True | False | False | False | README.md; outputs/configs/artifact_manifest.csv |
| docs/final_candidate_model_selection.md | markdown | 1217 | project documentation | False | True | False | False | False | src/final_validation_locked_model_development.py; outputs/configs/artifact_manifest.csv; outputs/configs/repository_file_inventory.csv |
| docs/final_dqn_ensemble_test_decision.md | markdown | 2205 | project documentation | False | True | False | False | False | src/final_dqn_ensemble_test_evaluation.py; outputs/configs/artifact_manifest.csv; outputs/configs/repository_file_inventory.csv |
| docs/final_figure_interpretation.md | markdown | 4348 | project documentation | False | True | False | False | False | src/final_report_figures.py; outputs/configs/artifact_manifest.csv; outputs/configs/repository_file_inventory.csv |
| docs/final_model_decision_task_interpretation.md | markdown | 2654 | project documentation | False | True | False | False | False | src/final_model_decision_task_audit.py; outputs/configs/artifact_manifest.csv; outputs/configs/repository_file_inventory.csv |
| docs/final_test_baseline_ladder.md | markdown | 1610 | project documentation | False | True | False | False | False | outputs/configs/artifact_manifest.csv |
| docs/final_test_results.md | markdown | 750 | project documentation | False | True | False | False | False | outputs/configs/artifact_manifest.csv |
| docs/limitations.md | markdown | 533 | project documentation | False | True | False | False | False | outputs/configs/artifact_manifest.csv |
| docs/limited_horizon_planning_decision.md | markdown | 1841 | project documentation | False | True | False | False | False | src/limited_horizon_planning_evaluation.py; outputs/configs/artifact_manifest.csv; outputs/configs/repository_file_inventory.csv |
| docs/planning_and_distillation_results.md | markdown | 480 | project documentation | False | True | False | False | False | outputs/configs/artifact_manifest.csv |
| docs/planning_distillation_decision.md | markdown | 1452 | project documentation | False | True | False | False | False | src/high_risk_b_planning_distillation.py; outputs/configs/artifact_manifest.csv; outputs/configs/repository_file_inventory.csv |
| docs/ppo_redesign_reporting_corrections.md | markdown | 3659 | project documentation | False | True | False | False | False | src/ppo_redesign_reporting_integrity_audit.py; outputs/configs/artifact_manifest.csv; outputs/configs/repository_file_inventory.csv |
| docs/ppo_results.md | markdown | 510 | project documentation | False | True | False | False | False | outputs/configs/artifact_manifest.csv |
| docs/project_progress_and_redesign.md | markdown | 6751 | project documentation | False | True | False | False | False | docs/security_and_file_size_audit.md; outputs/configs/artifact_manifest.csv; outputs/configs/repository_file_inventory.csv |
| docs/repository_archive_audit.md | markdown | 5851 | project documentation | False | True | False | False | False | outputs/configs/artifact_manifest.csv |
| docs/reproducibility.md | markdown | 718 | project documentation | False | True | False | False | False | outputs/configs/artifact_manifest.csv; data/raw/README.md |
| docs/research_design.md | markdown | 781 | project documentation | False | True | False | False | False | outputs/configs/artifact_manifest.csv |
| docs/security_and_file_size_audit.md | markdown | 2346 | project documentation | False | True | False | False | False | outputs/configs/artifact_manifest.csv |
| environment.yml | config | 287 | supporting project artifact | False | True | False | False | False | README.md; docs/archive_integrity_check.md; outputs/configs/artifact_manifest.csv |
| notebooks/README.md | markdown | 106 | supporting project artifact | False | True | False | False | False | pyproject.toml; outputs/configs/artifact_manifest.csv; outputs/configs/repository_file_inventory.csv; outputs/tables/freshretail_discount_semantics_audit.csv |
| outputs/configs/artifact_manifest.csv | table | 78679 | supporting project artifact | False | True | False | False | False |  |
| outputs/configs/checksum_manifest.sha256 | sha256 | 68715 | supporting project artifact | False | True | False | False | False | outputs/configs/artifact_manifest.csv; outputs/models/MODEL_ARTIFACTS.md |
| outputs/configs/dqn_equal_weight_ensemble_candidate.json | config | 3056 | supporting project artifact | False | True | False | False | False | src/dqn_statistical_audit_and_ensemble.py; outputs/configs/artifact_manifest.csv; outputs/configs/repository_file_inventory.csv |
| outputs/configs/dqn_locked_candidate.json | config | 3683 | supporting project artifact | False | True | False | False | False | src/complete_standard_dqn_multiseed.py; src/train_

## Source: docs/consolidation_integrity_check.md


锘? Consolidation Integrity Check

Generated after project-results consolidation on 2026-07-21.

## Non-Training Checks Run

- `python -m pytest tests`: 33 collected, 33 skipped because raw data and model artifacts are intentionally excluded from GitHub.
- `python -m compileall -q src scripts`: passed with no syntax errors.
- Markdown relative-link validation: 40 Markdown files checked, 0 missing links.
- Git-tracked binary/raw-data scan: no tracked `.zip`, `.pkl`, `.pt`, `.pth`, `.parquet`, or raw-data CSV files found.
- Secret scan: no credential-like secrets found. Matches on the word `token` were ordinary source-code variable names.
- Claim scan: primary files consistently state that always_0pct is strongest overall and balanced_recovered_ppo is strongest learned.

## Required Final Claims

- Strongest overall policy: always_0pct.
- Strongest learned policy: balanced_recovered_ppo.
- Final DQN test result: negative versus always_0pct.
- Secondary baseline ladder: post-hoc descriptive.
- No raw data in repository.
- No model binaries in repository.

## Repository Structure Check

- Main entry files exist: `README.md`, `START_HERE.md`, `results/README.md`.
- Primary documents exist: `docs/01_project_and_research_design.md` through `docs/05_conclusions_and_limitations.md`, plus `docs/reproducibility.md`.
- Curated result tables exist under `results/tables/`.
- Curated figures exist under `results/figures/`.
- Intermediate tables and diagnostic figures were moved under `archive/`.
- Existing implementation modules remain in `src/` to avoid import-path churn.
- Numbered wrappers exist under `scripts/`.

## Integrity Outcome

CONSOLIDATION_INTEGRITY_CHECK_PASSED for the concise project archive. Full experiment reruns still require local data and model artifacts that are intentionally excluded from GitHub.



## Source: docs/document_consolidation_map.md


# Document Consolidation Map

| original file | destination primary document | section incorporated | archived | omitted content |
|---|---|---|---:|---|
| docs/archive_integrity_check.md | docs/consolidation_integrity_check.md | Artifact checks | True | Repetitive progress prose omitted; concrete values retained where relevant. |
| docs/data_and_assumptions.md | docs/02_data_and_demand_recovery.md | Inputs, Method | True | Repetitive progress prose omitted; concrete values retained where relevant. |
| docs/dqn_final_statistical_decision.md | docs/04_results.md | Main Results, Interpretation | True | Repetitive progress prose omitted; concrete values retained where relevant. |
| docs/dqn_results.md | docs/04_results.md | Main Results | True | Repetitive progress prose omitted; concrete values retained where relevant. |
| docs/experiment_timeline.md | docs/01_project_and_research_design.md | Implementation | True | Repetitive progress prose omitted; concrete values retained where relevant. |
| docs/final_candidate_model_selection.md | docs/04_results.md | Method, Main Results | True | Repetitive progress prose omitted; concrete values retained where relevant. |
| docs/final_dqn_ensemble_test_decision.md | docs/04_results.md | Main Results | True | Repetitive progress prose omitted; concrete values retained where relevant. |
| docs/final_figure_interpretation.md | docs/04_results.md | Related Files | True | Repetitive progress prose omitted; concrete values retained where relevant. |
| docs/final_model_decision_task_interpretation.md | docs/04_results.md | Method | True | Repetitive progress prose omitted; concrete values retained where relevant. |
| docs/final_test_baseline_ladder.md | docs/04_results.md | Main Results | True | Repetitive progress prose omitted; concrete values retained where relevant. |
| docs/final_test_results.md | docs/04_results.md | Main Results | True | Repetitive progress prose omitted; concrete values retained where relevant. |
| docs/limitations.md | docs/05_conclusions_and_limitations.md | Limitations | True | Repetitive progress prose omitted; concrete values retained where relevant. |
| docs/limited_horizon_planning_decision.md | docs/04_results.md | Method, Interpretation | True | Repetitive progress prose omitted; concrete values retained where relevant. |
| docs/planning_and_distillation_results.md | docs/04_results.md | Main Results | True | Repetitive progress prose omitted; concrete values retained where relevant. |
| docs/planning_distillation_decision.md | docs/04_results.md | Interpretation | True | Repetitive progress prose omitted; concrete values retained where relevant. |
| docs/ppo_redesign_reporting_corrections.md | docs/04_results.md | Implementation, Limitations | True | Repetitive progress prose omitted; concrete values retained where relevant. |
| docs/ppo_results.md | docs/04_results.md | Main Results | True | Repetitive progress prose omitted; concrete values retained where relevant. |
| docs/project_progress_and_redesign.md | docs/01_project_and_research_design.md | Implementation | True | Repetitive progress prose omitted; concrete values retained where relevant. |
| docs/repository_archive_audit.md | docs/repository_consolidation_audit.md | Prior archive context | True | Repetitive progress prose omitted; concrete values retained where relevant. |
| docs/reproducibility.md | docs/reproducibility.md | All sections | True | Repetitive progress prose omitted; concrete values retained where relevant. |
| docs/research_design.md | docs/01_project_and_research_design.md | Purpose, Method | True | Repetitive progress prose omitted; concrete values retained where relevant. |
| docs/security_and_file_size_audit.md | docs/consolidation_integrity_check.md | Security and file-size checks | True | Repetitive progress prose omitted; concrete values retained where relevant. |


## Source: docs/reproducibility_checklist.md


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


## Source: archive/audits/archive_integrity_check.md


锘? Archive Integrity Check

Date: 2026-07-20

## Commands Run
- `python --version`
- `python -m pytest tests`
- key artifact existence checks
- non-training secret/path scan
- artifact manifest and SHA-256 checksum generation

## Results
- Python version observed: 3.13.9
- Pytest result: 33 passed in 58.41 seconds
- No PPO, DQN, planning, validation, or held-out test rerun was performed.
- README, LICENSE, CITATION.cff, environment.yml, archive audits, result docs, artifact manifest, checksum manifest, and model artifact documentation exist.
- Raw FreshRetailNet parquet files exceed 50 MB and are excluded from public Git.
- Legacy `archive/prototype_kaggle/` and `docs/machine_specific_path_audit.csv` contain machine-specific paths and are excluded from public Git candidates.
- No confirmed secrets were found in candidate source/documentation/config files.

## Git Limitation
The local project directory currently does not contain `.git`, and `git` was not available in the PowerShell PATH used by Codex. Branch creation, commits, push, and pull request creation remain manual follow-up steps unless Git/GitHub Desktop is installed and the project is initialized or copied into a clone of `willowkatze/ppo-perishable-pricing`.

## Clone Archive Check

After copying into the GitHub clone and excluding raw data/model binaries, `python -m pytest tests` collected 33 tests and skipped 33 tests. This is expected for the public archive because local data and fitted model artifacts are excluded. Full integration tests require restoring the documented local artifacts.


## Source: archive/audits/repository_archive_audit.md


锘? Repository Archive Audit

Project root: `ppo_perishable_pricing_project`.

Git status note: local project directory does not currently contain `.git`, and `git` is not available in the PowerShell PATH used by Codex. Remote branch creation and push therefore require Git/GitHub Desktop on the user machine or a separate clone step.

Total files scanned: 725
Total size: 1953.73 MB
Candidate commit files: 624
Excluded files: 39
Manual-review files: 68

## Largest Files
| Path | Size MB | Decision | Reason/Purpose |
|---|---:|---|---|
| data\operational\raw\freshretail\raw\freshretail_train.parquet | 1696.304 | NO - raw dataset | raw dataset, exclude from public repo |
| data\operational\raw\freshretail\raw\freshretail_eval.parquet | 132.378 | NO - raw dataset | raw dataset, exclude from public repo |
| archive\prototype_kaggle\data\processed\perishable_goods_clean.csv | 24.453 | NO - legacy dataset copy | legacy/prototype archive, mostly exclude except documentation if needed |
| archive\prototype_kaggle\data\raw\kaggle_source\perishable_goods_management.csv | 20.118 | NO - legacy dataset copy | legacy/prototype archive, mostly exclude except documentation if needed |
| outputs\tables\latent_demand_validation_predictions.csv | 10.494 | YES/MAYBE - candidate after secret review | result table |
| outputs\models\freshretail_latent_demand_model.joblib | 5.496 | YES/MAYBE - candidate after secret review | model artifact or metadata |
| archive\prototype_kaggle\outputs\tables\ppo_financial_training_episodes.csv | 5.074 | YES/MAYBE - candidate after secret review | legacy/prototype archive, mostly exclude except documentation if needed |
| archive\prototype_kaggle\outputs\tables\ppo_financial_action_distribution.csv | 4.208 | YES/MAYBE - candidate after secret review | legacy/prototype archive, mostly exclude except documentation if needed |
| archive\prototype_kaggle\outputs\tables\baseline_episode_results.csv | 1.913 | YES/MAYBE - candidate after secret review | legacy/prototype archive, mostly exclude except documentation if needed |
| data\freshretail\processed\freshretail_demand_recovered.parquet | 1.599 | NO - derived dataset | processed data, exclude unless tiny documented sample |
| outputs\models\planning_distillation\best_planning_distilled_model.pkl | 0.98 | YES/MAYBE - candidate after secret review | model artifact or metadata |
| archive\prototype_kaggle\outputs\tables\counterfactual_discount_predictions.csv | 0.778 | YES/MAYBE - candidate after secret review | legacy/prototype archive, mostly exclude except documentation if needed |
| outputs\models\discount_response_recovered.joblib | 0.651 | YES/MAYBE - candidate after secret review | model artifact or metadata |
| outputs\models\discount_response_observed.joblib | 0.651 | YES/MAYBE - candidate after secret review | model artifact or metadata |
| outputs\models\ppo_operational\observed_financial\seed_42\checkpoints\pilot_observed_financial_22288_steps.zip | 0.545 | MAYBE - essential model only, preferably LFS or external artifact manifest | model artifact or metadata |
| outputs\models\ppo_operational\observed_financial\seed_42\checkpoints\pilot_observed_financial_27288_steps.zip | 0.545 | MAYBE - essential model only, preferably LFS or external artifact manifest | model artifact or metadata |
| outputs\models\ppo_training_redesign\balanced\seed_123\final_model.zip | 0.545 | MAYBE - essential model only, preferably LFS or external artifact manifest | model artifact or metadata |
| outputs\models\ppo_operational\recovered_financial\seed_42\final_model.zip | 0.545 | MAYBE - essential model only, preferably LFS or external artifact manifest | model artifact or metadata |
| outputs\models\ppo_training_redesign\balanced\seed_123\checkpoints\redesign_balanced_10000_steps.zip | 0.545 | MAYBE - essential model only, preferably LFS or external artifact manifest | model artifact or metadata |
| outputs\models\ppo_operational\observed_financial\seed_42\best_model.zip | 0.545 | MAYBE - essential model only, preferably LFS or external artifact manifest | model artifact or metadata |
| outputs\models\ppo_operational\observed_financial\seed_42\final_model.zip | 0.545 | MAYBE - essential model only, preferably LFS or external artifact manifest | model artifact or metadata |
| outputs\models\ppo_operational\observed_financial\seed_42\checkpoints\pilot_observed_financial_17288_steps.zip | 0.545 | MAYBE - essential model only, preferably LFS or external artifact manifest | model artifact or metadata |
| outputs\models\ppo_training_redesign\balanced\seed_123\checkpoints\redesign_balanced_15000_steps.zip | 0.545 | MAYBE - essential model only, preferably LFS or external artifact manifest | model artifact or metadata |
| outputs\models\ppo_operational\recovered_financial\seed_42\best_model.zip | 0.545 | MAYBE - essential model only, preferably LFS or external artifact manifest | model artifact or metadata |
| outputs\models\ppo_operational\recovered_financial\seed_42\checkpoints\pilot_recovered_financial_32288_steps.zip | 0.545 | MAYBE - essential model only, preferably LFS or external artifact manifest | model artifact or metadata |

## Archive Policy
- Exclude raw FreshRetailNet parquet files and legacy Kaggle raw/processed copies from public GitHub.
- Exclude caches, temporary files, local IDE metadata, checkpoints not required for scientific claims, and regenerated intermediate datasets.
- Commit source code, tests, documentation, selected configs, selected manifests, scientifically important tables, and report-ready figures.
- Essential binary models should either use Git LFS or be represented by hashes and a model artifact manifest.

## Full Inventory CSV
A machine-readable inventory was generated at `outputs/configs/repository_file_inventory.csv`.


## Source: archive/audits/security_and_file_size_audit.md


锘? Security and File Size Audit

Scope: source code, documentation, configuration-like files, notebooks, scripts, and lightweight CSV/JSON/Markdown files in the local project.

## Secret Scan Summary
- No API keys, GitHub tokens, passwords, SSH private keys, cookies, or email addresses were confirmed in the scanned candidate source/documentation/config files.
- All initial matches were false positives from ordinary variable names or dataset references.
- Raw datasets and large generated data artifacts remain excluded regardless of secret-scan status.

| Path | Finding | Action |
|---|---|---|
| src\train_ppo_recovered_redesign.py | false positive: local variable named token | safe to keep |
| docs\project_progress_and_redesign.md | false positive: Kaggle dataset reference | safe to keep |
| src\freshretail_data_processing.py | false positive: date token parsing terms | safe to keep |
| src\perishability_scenarios.py | false positive: category token parsing terms | safe to keep |

## Files Above 50 MB
| Path | Size MB | Public GitHub decision |
|---|---:|---|
| data\operational\raw\freshretail\raw\freshretail_train.parquet | 1696.3 | Exclude from normal Git |
| data\operational\raw\freshretail\raw\freshretail_eval.parquet | 132.38 | Exclude from normal Git |

## Required Exclusions
- `data/operational/raw/freshretail/raw/freshretail_train.parquet` and `freshretail_eval.parquet` are raw dataset files and should not be committed.
- Legacy Kaggle raw/processed files under `archive/prototype_kaggle/data/` should not be committed to the public repository.
- Caches, notebooks checkpoints, IDE folders, temporary files, logs, and redundant checkpoints should stay ignored.

## Git LFS Recommendation
- Git LFS is optional for this project if final model binaries are excluded and documented by hash.
- If selected DQN/PPO model binaries must be archived, use Git LFS for `*.zip`, `*.pkl`, `*.joblib`, `*.pt`, and `*.pth` only after confirming they are scientifically necessary.

## Machine-Specific Path Exclusions

A follow-up path scan found absolute local Windows paths only in the legacy `archive/prototype_kaggle/` tree and `docs/machine_specific_path_audit.csv`. These files are excluded from public Git candidates through `.gitignore` and should not be staged for the public repository.

