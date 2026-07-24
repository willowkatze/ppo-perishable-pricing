"""Artifact-independent checks for locked schemas and evaluation safeguards."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import final_test_baseline_ladder as ladder  # noqa: E402
from pricing_env_operational import OperationalPerishablePricingEnv, observation_bounds, observation_names  # noqa: E402
from train_dqn_high_risk_b import bounded_termination_rollout  # noqa: E402


def test_observation_schema_preserves_legacy_dimension_and_bounds() -> None:
    low, high = observation_bounds()
    expected_high = np.asarray(
        [10.0, 10.0, *([1.0] * 19), *([20.0] * 7), 6.0, 6.0, 1.0, 40.0, *([1.0] * 9)],
        dtype=np.float32,
    )
    assert len(observation_names()) == 41
    assert low.shape == high.shape == (41,)
    assert np.array_equal(high, expected_high)


def test_zero_markdown_prediction_does_not_hide_runtime_errors() -> None:
    env = object.__new__(OperationalPerishablePricingEnv)
    env.current_series = pd.DataFrame({"timestamp": ["2024-01-01"]})
    env.horizon = 1

    def fail(_: float) -> float:
        raise RuntimeError("invalid prediction artifact")

    env._predict_demand = fail
    with pytest.raises(RuntimeError, match="invalid prediction artifact"):
        env._predict_zero_safe()


class _TerminationStub:
    horizon = 3

    def __init__(self, terminal_step: int | None) -> None:
        self.steps = 0
        self.terminal_step = terminal_step

    def step(self, action: int):
        self.steps += 1
        terminated = self.terminal_step is not None and self.steps >= self.terminal_step
        return np.zeros(1), 0.0, terminated, False, {}


def test_bounded_termination_rollout_reaches_terminal_state() -> None:
    result = bounded_termination_rollout(_TerminationStub(terminal_step=3), max_steps=4)
    assert result["flags_are_boolean"] is True
    assert result["terminal_reached"] is True
    assert result["bounded_completion"] is True
    assert result["steps"] == 3


def test_bounded_termination_rollout_detects_unfinished_episode() -> None:
    result = bounded_termination_rollout(_TerminationStub(terminal_step=None), max_steps=4)
    assert result["terminal_reached"] is False
    assert result["bounded_completion"] is False
    assert result["steps"] == 4


def test_final_ladder_rejects_missing_ppo_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        ladder,
        "PPO_CONFIGS",
        {"original_observed_ppo": (tmp_path / "model.zip", tmp_path / "vecnormalize.pkl")},
    )
    with pytest.raises(FileNotFoundError) as exc_info:
        ladder.load_ppo_members("core_001")
    message = str(exc_info.value)
    assert "original_observed_ppo" in message
    assert "model.zip (missing)" in message
    assert "vecnormalize.pkl (missing)" in message


def test_final_ladder_requires_all_policies_and_episodes() -> None:
    rows = [
        {"policy_id": policy_id, "episode_id": f"episode_{episode}"}
        for policy_id in ladder.POLICY_ORDER
        for episode in range(2)
    ]
    complete = pd.DataFrame(rows)
    ladder.validate_complete_ladder(complete, expected_episodes=2)
    with pytest.raises(RuntimeError, match="incomplete"):
        ladder.validate_complete_ladder(complete.loc[complete["policy_id"] != "balanced_recovered_ppo"], expected_episodes=2)
