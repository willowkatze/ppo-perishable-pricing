# Project and Research Design

## Problem and Motivation

This project studies dynamic markdown decisions for perishable fresh-retail inventory. Observed sales can understate true demand when products stock out, and perishable inventory creates a tradeoff between selling units and avoiding waste.

## Research Questions

1. Can stockout-aware demand recovery change the demand signal used for markdown learning?
2. Can PPO learn useful markdown behavior after accounting for recovered demand and action-collapse risk?
3. Do learned policies outperform fixed, random, and simple rule-based baselines on a locked held-out test set?

## Project Scope

The project uses FreshRetailNet-informed operational time series and a semi-synthetic pricing environment. The environment and accounting logic are controlled for course experimentation; they are not a deployed retail pricing system.

## Contribution

The project combines demand recovery, PPO diagnosis, balanced PPO training, DQN comparison, and a full held-out baseline ladder. The final comparison keeps both the positive learned-policy result and the stronger no-markdown baseline visible.

## Prior Work

FreshRetailNet motivates the demand-censoring part of the project. The final report should add formal citations for FreshRetailNet, stockout-censored demand estimation, dynamic pricing, and reinforcement learning for pricing or inventory control.

## Experimental Overview

The workflow is: prepare data, recover latent demand, estimate markdown response, build the perishability environment, train PPO, diagnose PPO collapse, train balanced PPO, compare DQN, and evaluate all final policies on the locked held-out test set.

