# Experiment Timeline

1. Original dataset assessment found the initial Kaggle-style data insufficient for the final research design.
2. The project transitioned to FreshRetailNet-informed operational data.
3. Stockout-aware latent demand recovery was developed.
4. Observed-sales and recovered-demand PPO variants were compared.
5. PPO action collapse and calibration sensitivity were diagnosed.
6. Scenario-balanced PPO redesign tested whether rare positive-markdown opportunities were underrepresented.
7. Strong baseline policies were evaluated.
8. HIGH_RISK_B was locked as the final high-risk task population.
9. Limited-horizon planning evaluated dynamic markdown value on paired validation episodes.
10. Planning distillation created focused training labels for high-risk states.
11. Three-seed DQN training was performed.
12. Pseudoreplication in pooled seed-episode inference was corrected.
13. An equal-weight DQN Q-value ensemble was selected from validation evidence.
14. A locked held-out test was run once.
15. The final conclusion was negative: validation gains did not generalize to held-out test episodes.
