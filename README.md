# PPO-Based Dynamic Markdown Optimization for Perishable Products

This repository supports a reinforcement learning course project on dynamic markdown optimization for perishable products. The project will use PPO in a later phase, but this initial setup only establishes the repository structure, documentation, and placeholders needed for clean development.

## Project Overview

Perishable products lose value as they approach the end of shelf life. Retailers often use markdowns to increase sell-through, reduce waste, and protect revenue. This project will study how reinforcement learning can learn markdown policies that balance revenue and waste reduction.

## Business Motivation

Better markdown decisions can help retailers reduce spoilage, improve inventory turnover, and preserve margins. A data-driven pricing policy may outperform simple fixed or rule-based strategies when demand and remaining shelf life vary over time.

## Initial Research Objective

The initial objective is to formulate and evaluate a PPO-based markdown policy for perishable inventory using a real dataset after its schema has been inspected. No dataset-specific variable assumptions are implemented yet.

## Planned MDP Formulation

- State: inventory, remaining shelf life, demand information
- Action: markdown level
- Reward baseline: revenue minus waste cost

## Planned Benchmark Strategies

- Fixed pricing
- Rule-based markdown
- Random policy

## Planned Sustainability Extension

A future extension may add sustainability-aware reward terms after the baseline environment and evaluation pipeline are validated. The extension will be based on documented assumptions and will not be added until the project has a reliable baseline.

## Technology Stack

- Python 3.10+
- pandas
- numpy
- matplotlib
- scikit-learn
- Gymnasium
- Stable-Baselines3
- pytest
- Jupyter

## Repository Structure

```text
ppo-perishable-pricing/
|-- data/
|   |-- raw/
|   |   `-- README.md
|   `-- processed/
|       `-- .gitkeep
|-- notebooks/
|   `-- README.md
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- data_processing.py
|   |-- demand_model.py
|   |-- pricing_env.py
|   |-- baselines.py
|   |-- train_ppo.py
|   `-- evaluate.py
|-- tests/
|   |-- __init__.py
|   `-- test_project_structure.py
|-- outputs/
|   |-- figures/
|   |   `-- .gitkeep
|   |-- tables/
|   |   `-- .gitkeep
|   `-- models/
|       `-- .gitkeep
|-- docs/
|   |-- research_plan.md
|   |-- experiment_log.md
|   `-- assumptions.md
|-- .gitignore
|-- requirements.txt
|-- README.md
`-- pyproject.toml
```

## Setup Instructions

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run tests:

```bash
python -m pytest
```

## Dataset Instructions

The Kaggle dataset is not included in this repository. Download the Managing Perishable Inventory Data manually from Kaggle and place the CSV at:

```text
data/raw/perishable_goods_management.csv
```

The real dataset schema will be inspected before any variable assumptions are implemented.

## Current Project Status

- [x] Repository structure initialized
- [x] Dataset location documented
- [x] Python package placeholders created
- [x] Basic project structure tests added
- [ ] Dataset exploration
- [ ] Demand-model estimation
- [ ] Gymnasium environment development
- [ ] Baseline pricing policies
- [ ] PPO training
- [ ] Evaluation and visualization
- [ ] Sustainability-aware reward extensions
