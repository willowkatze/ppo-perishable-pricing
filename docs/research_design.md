# Research Design

This project evaluates stockout-aware dynamic markdown decisions for perishable retail under a semi-synthetic environment.

The design follows a locked, staged workflow: data provenance audit, FreshRetailNet-informed demand recovery, perishability scenario construction, environment validation, baseline evaluation, PPO diagnostics, scenario-balanced PPO redesign, limited-horizon planning, planning distillation, DQN training, multi-seed statistical audit, equal-weight ensemble selection, and one held-out test evaluation.

The final decision criterion is not whether a learned policy looks complex, but whether it improves paired normalized accounting profit against a locked no-markdown baseline on held-out episodes without leakage or post-test tuning.
