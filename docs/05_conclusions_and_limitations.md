# Conclusions and Limitations

## Main Findings

The strongest overall held-out policy was always_0pct. The strongest learned held-out policy was balanced_recovered_ppo. Balanced recovered PPO beat the evaluated positive fixed-markdown policies, random_uniform, two simple rule-based policies, original PPO variants, and the locked DQN ensemble, but it did not beat always_0pct.

## What the Project Demonstrates

The project demonstrates a complete workflow for stockout-aware demand recovery, PPO diagnosis, balanced PPO redesign, and policy comparison under a controlled perishability environment.

## What the Project Does Not Demonstrate

The project does not show that learned dynamic markdowns outperform the strongest no-markdown baseline on the locked held-out test set. It also does not validate a deployable retail pricing system.

## Contributions

The main contribution is a transparent negative final result with a clear learned-policy improvement over weaker baselines. The full 13-policy ladder prevents selective reporting.

## Limitations

The environment is semi-synthetic, raw data and model binaries are excluded from GitHub, and full reproduction requires local artifacts. Prior work still needs to be written as a formal citation section in the final report.

## Future Work

Future work could test other reward designs, longer planning horizons, stronger demand models, and external validation on additional retail data.

