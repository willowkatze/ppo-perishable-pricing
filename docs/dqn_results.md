# DQN Results

DQN was trained in a focused HIGH_RISK_B task using multiple random seeds. Validation suggested a positive but uncertain signal. A statistical-integrity audit found pseudoreplication risk in pooled seed-episode rows, so inference was corrected using episode-clustered summaries.

The final validation candidate was an equal-weight Q-value ensemble of selected DQN checkpoints. It was locked before the held-out test evaluation.
