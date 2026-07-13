# Research Plan

## Phase 1: Dataset Exploration

Inspect the raw CSV schema, missing values, time granularity, product identifiers, inventory fields, and candidate demand signals.

## Phase 2: Demand-Model Estimation

Estimate simple demand relationships after the available variables are confirmed.

## Phase 3: Environment Development

Define a Gymnasium environment with documented state, action, transition, and reward assumptions.

## Phase 4: Baseline Policies

Implement fixed pricing, rule-based markdown, and random policy benchmarks.

## Phase 5: PPO Training

Train PPO only after the environment and baselines are validated.

## Phase 6: Evaluation

Compare policy performance using revenue, waste, and markdown behavior metrics.

## Phase 7: Sustainability Extension

Add sustainability-aware reward terms only after the baseline project is reproducible.
