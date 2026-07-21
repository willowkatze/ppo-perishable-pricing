# Final Held-Out Test Results

The locked final candidate was `STANDARD_DQN_EQUAL_WEIGHT_Q_ENSEMBLE`. The locked baseline was `always_0pct`.

Held-out test population: 60 HIGH_RISK_B episodes. Pairing: 60/60 valid.

| Metric | Value |
|---|---:|
| DQN ensemble mean normalized profit | 0.425721 |
| always_0pct mean normalized profit | 0.447999 |
| paired mean gain | -0.022278 |
| paired median gain | -0.017321 |
| bootstrap 95% CI | [-0.027486, -0.017319] |
| win / tie / loss | 0.000 / 0.233 / 0.767 |
| waste-rate difference | -0.000204 |
| sell-through difference | +0.000204 |

Conclusion: the final locked DQN ensemble did not outperform the no-markdown baseline on held-out test episodes. Positive validation signals did not generalize.
