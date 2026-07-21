# Planning and Distillation Results

Limited-horizon planning on locked validation episodes found modest positive dynamic value in HIGH_RISK_B states. The planning evaluation justified a focused learned-policy attempt but did not itself constitute a deployable learned policy.

Planning distillation generated labels for high-risk states and supported later DQN experiments. Distillation and planning were validation-locked and did not use the held-out test split for tuning.
