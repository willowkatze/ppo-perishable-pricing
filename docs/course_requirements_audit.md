# Course Requirements Audit

| requirement | source | current repository evidence | status | exact issue | required correction | relevant file paths |
|---|---|---|---|---|---|---|
| Problem definition | Lecture 0 report requirements | README.md; START_HERE.md; docs/01_research_design.md | PASS | Problem is stated as perishable markdown decision-making. | Keep concise problem statement. | README.md; START_HERE.md |
| Motivation | Lecture 0 report requirements | START_HERE.md; docs/01_research_design.md | PASS | Motivation is tied to stockout censoring and perishability. | None. | docs/01_research_design.md |
| Prior work | Lecture 0 report requirements | FreshRetailNet paper and general RL/dynamic pricing framing are referenced at a high level. | PARTIAL | Formal literature citations are not fully written in repo docs. | Add polished citations in the final report text. | docs/report_evidence_map.md |
| Methodology | Lecture 0 report requirements | docs/02_data_and_demand_recovery.md; docs/03_environment_and_assumptions.md; docs/04_ppo_analysis.md | PASS | Methods are linked to code and configs. | None. | src/; configs/ |
| Numerical studies | Lecture 0 report requirements | results/final_summary.csv; results/key_tables/ | PASS | Final values trace to source tables. | Use selected tables only in report. | results/ |
| Conclusions | Lecture 0 report requirements | docs/06_final_results.md | PASS | Conclusion states always_0pct strongest overall and balanced PPO strongest learned. | None. | docs/06_final_results.md |
| Limitations | Lecture 0 report requirements | docs/07_limitations.md | PASS | Limitations include semi-synthetic environment and negative held-out baseline result. | None. | docs/07_limitations.md |
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
