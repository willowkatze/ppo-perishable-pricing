# PPO Perishable Pricing

This repository is a reinforcement-learning course project on dynamic markdown decisions for perishable fresh-retail inventory. It uses FreshRetailNet-informed operational time series, stockout-aware demand recovery, and a semi-synthetic financial perishability environment. The central finding is precise and conservative: balanced recovered PPO is the strongest learned policy, while always_0pct remains the strongest overall held-out policy.

Start with [START_HERE.md](START_HERE.md). Curated tables and figures are under [results/](results/). Full intermediate outputs are retained under [archive/](archive/).

## Main Results

| Policy | Role | Mean normalized profit | Gain vs always_0pct | Interpretation |
|---|---|---:|---:|---|
| always_0pct | strongest overall policy | 0.4479993306199982 | 0.000000 | Best held-out profit. |
| balanced_recovered_ppo | strongest learned policy | 0.4359085234475656 | -0.012090807172432544 | Beats weaker baselines and learned alternatives, not always_0pct. |
| locked_dqn_ensemble | value-based follow-up | 0.42572130058655044 | -0.022278030033447582 | Did not generalize beyond always_0pct. |


The locked DQN ensemble achieved 0.425721 mean normalized profit on held-out test versus 0.447999 for always_0pct. Its paired gain was -0.022278 with bootstrap 95% CI [-0.027486, -0.017319], so it did not beat the primary baseline. The secondary baseline ladder is post-hoc descriptive: balanced recovered PPO beat all evaluated positive fixed-markdown baselines, random pricing, two rule-based strategies, original PPO variants, and DQN, but did not beat always_0pct.

## Dataset

The project is based on FreshRetailNet-informed operational time series with stockout-aware demand recovery. Raw data is intentionally excluded from the repository. The perishability environment and accounting layer are semi-synthetic and documented in [docs/03_environment_and_assumptions.md](docs/03_environment_and_assumptions.md).

## Method Pipeline

1. Prepare FreshRetailNet-derived operational series.
2. Recover latent demand under stockout censoring.
3. Estimate markdown-response behavior.
4. Build a six-action dynamic markdown environment.
5. Train and diagnose PPO policies.
6. Run scenario-balanced PPO redesign.
7. Run DQN as a value-based follow-up comparison.
8. Lock the final held-out test population and evaluate all methods without retraining.

## PPO and DQN Roles

PPO is the primary experimental model. It is used for calibration-sensitivity analysis, action-collapse diagnosis, and scenario-balanced training redesign. Balanced recovered PPO is the strongest learned policy after balancing.

DQN is a value-based follow-up comparison for the discrete action space. It produced a positive validation signal but failed to generalize on the locked held-out test and remained below balanced recovered PPO.

## Repository Structure

- [START_HERE.md](START_HERE.md): five-minute project entry.
- [docs/](docs/): consolidated evidence-based methodology and results.
- [results/](results/): curated tables and figures.
- [archive/](archive/): intermediate tables, diagnostic figures, old configs, and experiment notes.
- [src/](src/): implementation modules retained without import-path rewrites.
- [scripts/](scripts/): numbered wrappers for reproducibility entry points.
- [tests/](tests/): lightweight import and artifact-availability checks.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Full local reproduction requires data and model artifacts that are excluded from GitHub because of file size and data-sharing constraints.

## Reproduction

See [scripts/README.md](scripts/README.md). The scripts are numbered by pipeline order. Training commands are not run automatically.

## Limitations

No learned policy beat always_0pct on the locked held-out test. The environment is semi-synthetic, so the result should be interpreted as a controlled decision-quality experiment rather than a deployed pricing system. The secondary baseline ladder does not replace the primary negative held-out conclusion.
