# Final Course Alignment Review

## Course Requirement Coverage

The repository now maps directly to the expected report sections: problem and motivation, prior work, methodology, numerical studies, conclusions, and limitations. The main evidence map is `docs/report_evidence_map.md`.

## Remaining Missing Requirement

Prior work is only summarized at a high level in the repository. The final 8-12 page report should add a polished citation paragraph for FreshRetailNet, stockout-censored demand estimation, dynamic pricing, and RL for pricing/inventory.

## Incorrect Tables Found

One reporting inconsistency was found and corrected: `results/key_tables/final_heldout_baseline_ladder.csv` listed the locked DQN bootstrap CI as approximately [-0.027435, -0.017247], while the locked held-out DQN summary and course audit target give [-0.027486, -0.017319]. The table now uses the locked held-out DQN summary value. Supporting DQN validation rows were clarified as repeated seed-episode observations, not independent episodes.

## Incorrect Figures Found

No retained main figure was corrected. One extra timeline-style figure was moved out of the main figure set because it was less directly useful for the report.

## Obsolete or Duplicated Files Found

The repository had more result files than needed for the final report. Supporting tables and figures remain archived instead of deleted.

## Language Issues Found

Audience-specific and generated-sounding labels such as `teacher-facing` were replaced with plain labels such as `key results`, `main review file`, and `project-results consolidation`.

## Corrections Made

- Main tables reduced to five report tables plus `results/final_summary.csv`.
- Main figures reduced to seven.
- Course requirement audit added.
- Result consistency audit added.
- Table and figure quality audits added.
- Report evidence map added.
- Reproducibility checklist added.

## Files Archived

Two supporting result tables and one supporting figure were moved from `results/` back to `archive/`.

## Final Main Tables

1. `results/key_tables/demand_recovery_summary.csv`
2. `results/key_tables/ppo_original_vs_balanced_comparison.csv`
3. `results/key_tables/final_heldout_baseline_ladder.csv`
4. `results/key_tables/dqn_heldout_test_summary.csv`
5. `results/key_tables/final_profit_waste_pareto_frontier.csv`

## Final Main Figures

1. `results/key_figures/01_observed_vs_recovered_demand.png`
2. `results/key_figures/02_recovered_demand_adjustment_distribution.png`
3. `results/key_figures/03_original_vs_balanced_ppo_error_decomposition.png`
4. `results/key_figures/04_ppo_action_collapse_diagnostic.png`
5. `results/key_figures/05_final_heldout_baseline_ladder.png`
6. `results/key_figures/06_learned_policies_vs_fixed_markdown.png`
7. `results/key_figures/07_profit_waste_pareto_frontier.png`

## PPO/DQN Positioning

PPO remains the main course-aligned RL model. It covers demand-calibration comparison, action-collapse diagnosis, and balanced-training redesign. Balanced recovered PPO is the strongest learned policy on held-out test.

DQN is retained as an additional value-based comparison for the six-action discrete action space. It had a positive validation signal but was weaker than balanced recovered PPO and did not beat always_0pct on held-out test.

## Reproducibility Status

Reproducibility is mostly complete for repository review: source code, configs, seeds, splits, scripts, and key outputs are committed. Full reruns still require excluded raw data and model artifacts. A final GitHub release/tag remains a manual submission step.

## Readiness

The repository is ready to support an 8-12 page final report and a 10-minute presentation, provided the final written report adds a concise prior-work section and uses only the selected main tables and figures.

