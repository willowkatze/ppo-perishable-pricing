"""Unit tests for OperationalPerishablePricingEnv."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pricing_env_operational import ACTION_MARKDOWNS, OperationalPerishablePricingEnv  # noqa: E402

REQUIRED_LOCAL_ARTIFACTS = [
    PROJECT_ROOT / "data" / "freshretail" / "processed" / "freshretail_demand_recovered.parquet",
    PROJECT_ROOT / "outputs" / "models" / "discount_response_observed.joblib",
    PROJECT_ROOT / "outputs" / "models" / "discount_response_recovered.joblib",
    PROJECT_ROOT / "outputs" / "models" / "discount_response_config.json",
    PROJECT_ROOT / "outputs" / "tables" / "perishability_core_scenario_grid.csv",
]
pytestmark = pytest.mark.skipif(
    not all(path.exists() for path in REQUIRED_LOCAL_ARTIFACTS),
    reason="local data/model artifacts are excluded from the public archive",
)


def make_env(**kwargs):
    defaults = {"split": "validation", "calibration_mode": "recovered_calibration", "reward_mode": "financial", "random_seed": 123}
    defaults.update(kwargs)
    return OperationalPerishablePricingEnv(**defaults)


def rollout(env, actions):
    obs, info = env.reset(seed=123)
    traj = []
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        traj.append((obs.copy(), reward, terminated, truncated, dict(info)))
        if terminated or truncated:
            break
    return traj


def test_reset_returns_valid_observation_and_info():
    env = make_env()
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    assert np.isfinite(obs).all()
    assert "store_id" in info and "product_id" in info


def test_action_space_and_mapping():
    env = make_env()
    assert env.action_space.n == 6
    for action in range(6):
        assert env.action_space.contains(action)
    assert ACTION_MARKDOWNS[5] == 0.40


def test_step_api_and_finite_values():
    env = make_env()
    obs, info = env.reset(seed=2)
    obs, reward, terminated, truncated, info = env.step(0)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert np.isfinite(obs).all()
    assert np.isfinite(reward)


def test_inventory_sales_waste_invariants():
    env = make_env()
    env.reset(seed=3)
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(2)
        assert info["inventory_after"] >= -1e-8
        assert info["sales"] <= info["inventory_before"] + 1e-8
        assert info["expired_units"] >= -1e-8
        assert info["step_conservation_error"] <= 1e-6
        if terminated:
            assert info["episode_conservation_error"] <= 1e-6
            break


def test_fefo_removes_oldest_inventory_first():
    env = make_env()
    env.reset(seed=4)
    env.inventory[:] = 0
    env.shelf_life = 3
    env.inventory[:3] = np.array([1.0, 2.0, 3.0])
    env._issue_fefo(2.5)
    assert np.allclose(env.inventory[:3], [0.0, 0.5, 3.0])


def test_procurement_cost_charged_once_and_profit_identity():
    env = make_env()
    env.reset(seed=5)
    procurement_seen = []
    terminal = None
    for action in [0, 1, 2, 3, 4, 5, 0, 0]:
        obs, reward, terminated, truncated, info = env.step(action)
        procurement_seen.append(info["cumulative_procurement_cost"])
        if terminated:
            terminal = info
            break
    increments = np.diff([0.0] + procurement_seen)
    assert (increments > 1e-9).sum() == 1
    assert abs(sum(env.raw_financial_steps) - env.accounting_profit) <= 1e-6
    if terminal:
        assert abs(terminal["normalized_accounting_profit"] - env.accounting_profit / env.initial_inventory) <= 1e-6


def test_sustainability_return_identity():
    env = make_env(reward_mode="sustainability", lambda_waste=0.5)
    env.reset(seed=6)
    terminal = None
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(5)
        if terminated:
            terminal = info
            break
    assert terminal is not None
    expected = terminal["normalized_accounting_profit"] - 0.5 * terminal["final_waste_rate"]
    assert abs(env.episode_return - expected) <= 1e-6


def test_same_seed_reproduces_deterministic_trajectory():
    actions = [0, 1, 2, 3, 4, 5]
    env1 = make_env(deterministic_demand=True)
    env2 = make_env(deterministic_demand=True)
    traj1 = rollout(env1, actions)
    traj2 = rollout(env2, actions)
    assert len(traj1) == len(traj2)
    for a, b in zip(traj1, traj2):
        assert np.allclose(a[0], b[0])
        assert a[1] == pytest.approx(b[1])
        assert a[4]["accounting_profit"] == pytest.approx(b[4]["accounting_profit"])


def test_episode_never_crosses_split_boundaries():
    env = make_env(split="validation")
    obs, info = env.reset(seed=7)
    dates = []
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(0)
        dates.append(info["date"])
        if terminated:
            break
    assert min(dates) >= "2024-05-26"
    assert max(dates) <= "2024-06-13"


def test_dynamic_lags_use_simulated_history_after_reset():
    env = make_env()
    env.reset(seed=8)
    before = list(env.sim_history)
    obs, reward, terminated, truncated, info = env.step(5)
    assert env.sim_history[-1] != before[-1] or info["sales"] == before[-1]
    assert len(env.sim_history) <= 14


def test_calibration_modes_use_correct_response_model():
    observed = make_env(calibration_mode="observed_calibration")
    recovered = make_env(calibration_mode="recovered_calibration")
    assert observed._response_bundle().target_name == "observed"
    assert recovered._response_bundle().target_name == "recovered"


def test_invalid_scenario_fails_clearly():
    with pytest.raises(ValueError):
        make_env(scenario_id="not_a_real_scenario")


def test_unsupported_21_day_validation_scenario_rejected():
    env = make_env(scenario_id="core_013", shelf_life_level="high")
    with pytest.raises(ValueError):
        env.reset(seed=9)


def test_terminal_inventory_treatment_matches_config():
    env = make_env()
    env.reset(seed=10)
    terminal = None
    for _ in range(30):
        obs, reward, terminated, truncated, info = env.step(0)
        if terminated:
            terminal = info
            break
    assert terminal is not None
    assert terminal["terminal_salvage_value"] == 0.0

