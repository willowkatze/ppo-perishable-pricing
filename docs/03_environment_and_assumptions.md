# 03 Environment And Assumptions

## Purpose
Describe perishability environment and accounting assumptions.

## Inputs
configs/environment and archived environment tables.

## Method
Six-action markdown environment with financial, waste, and sell-through metrics.

## Implementation
src/pricing_env_operational.py; src/validate_pricing_env_operational.py.

## Main Results
Final HIGH_RISK_B test population contains 60 paired episodes.

## Interpretation
Environment supports controlled comparison, not deployment.

## Limitations
Modeled perishability/accounting assumptions affect results.

## Related Files
configs/environment; outputs/manifests/high_risk_b_final_test_manifest.csv

