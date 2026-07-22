# Conclusions and Limitations

## Main Conclusion

On the locked 60-episode HIGH_RISK_B held-out test set, `always_0pct` was the strongest overall policy with mean normalized profit `0.447999`. `balanced_recovered_ppo` was the strongest learned policy with `0.435909`, and the locked DQN ensemble scored `0.425721`. No learned policy beat the no-markdown baseline.

## What Succeeded

- A stockout-aware demand-recovery pipeline was connected to the pricing experiment.
- The recovery summary documented a 13.317% aggregate increase in estimated demand and a 40.196% adjusted-row share.
- The environment made inventory aging, FEFO issuing, waste, markdown actions, and normalized accounting explicit.
- The balanced PPO experiment reduced unnecessary markdown in validation diagnostics from `0.625` to `0.1948` and increased reported action entropy from `0.3268` to `0.3904`.
- The final ladder reports learned, fixed, random, and rule-based policies on the same held-out episodes.

## What Did Not Succeed

The learned policies did not exceed `always_0pct` on the held-out test. The DQN ensemble's paired gain was negative with a bootstrap interval below zero. Balanced PPO improved behavioral diagnostics and was the strongest learned policy, but its held-out profit remained `0.012091` below `always_0pct`.

## Methodological Contribution

The methodological contribution is the explicit separation of demand recovery, environment assumptions, training diagnostics, paired validation, and locked held-out comparison. The result keeps the strongest baseline visible and reports weaker baselines rather than selectively omitting them. The balanced training stage also tests action imbalance as a possible learning limitation without changing the environment or reward.

## Practical Interpretation

Under the documented semi-synthetic assumptions and HIGH_RISK_B population, the conservative no-markdown strategy is the strongest financial reference. A learned policy may still be useful when operational priorities include waste reduction, service levels, or other objectives, but those objectives require their own locked comparison and should not be inferred from the financial result alone.

## Data and Modeling Limitations

- The raw dataset and trained model binaries are not committed, so full reproduction requires local artifacts.
- The modeling subset is 29,100 rows and 300 complete sequences, not the full public dataset.
- Recovered demand is an ExtraTrees-based proxy and is constrained to be no lower than observed sales; true latent demand remains unobserved.
- Markdown response is observational and model-implied, so the analysis does not establish a causal price effect.
- Shelf life, inventory age, initial inventory, procurement cost, disposal cost, and waste are controlled or simulated assumptions.
- The HIGH_RISK_B held-out population is a focused stress population, not a complete retailer population.
- The environment uses normalized accounting profit and does not estimate direct retailer profit or deployment performance.

## Held-Out Generalization

The held-out result is valid for the locked test manifest and recorded environment protocol. It supports the narrow claim that, under this protocol, the selected learned policies did not beat `always_0pct`. It does not support claims about other stores, products, time periods, retailers, or real-world deployment.

## Future Work

Future work could test alternative reward definitions, longer planning horizons, stronger demand models, stochastic demand validation, broader populations, and external retail datasets. Any such extension should preserve a separate validation protocol and a genuinely untouched test population.

## Related Files

- Final results: `docs/04_results.md`, `results/tables/final_heldout_baseline_ladder.csv`
- Environment assumptions: `docs/03_environment_and_methods.md`, `configs/environment/`
- Reproducibility: `docs/reproducibility.md`, `scripts/README.md`
