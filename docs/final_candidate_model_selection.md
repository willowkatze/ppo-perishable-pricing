# Final Candidate Model Selection

## Locked Validation Task

Population: `HIGH_RISK_B`.
Calibration: recovered calibration.
Locked baseline: `always_0pct`.
Test split used: false.

## Expanded Planning Result

Paired validation episodes: 22.
Mean paired planning gain: 0.010792.
Bootstrap 95% CI: [-0.025571, 0.039094].
Win/tie/loss: 0.818 / 0.091 / 0.091.
Waste-rate difference: -0.138921.

## Candidate Model Status

Final validation-only status: `PLANNING_VALUE_EXISTS_BUT_NOT_LEARNED`.

This script locks the expanded validation task and planning result. It does not
silently train long-running PPO/DQN jobs. If the expanded planning result remains
positive, separate explicit trainers can be created for GATED_PPO, GATED_DQN,
and PLANNING_DISTILLED_POLICY using the locked manifest.

## Report-Ready Wording

The project identified a validation-locked high-risk decision population and
evaluated a limited-horizon planning benchmark against the locked always-zero
baseline without using the test split. If learned candidates are not yet trained,
the correct claim is that planning value exists or does not exist on validation,
not that a final learned model beats the baseline.
