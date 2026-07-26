from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import train_ppo_operational as train  # noqa: E402
from evaluate_ppo_operational import baseline_action, create_episode_manifest, evaluate_baselines  # noqa: E402
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


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_vectorized_training_environment_can_be_built():
    vec = train.make_vec_env(
        n_envs=2,
        split="train",
        calibration_mode="recovered_calibration",
        reward_mode="financial",
        lambda_waste=0.10,
        scenario_id=train.DEFAULT_SCENARIO_ID,
        deterministic_demand=True,
        seed=42,
        monitor_dir=train.MODELS_DIR / "_pytest_monitor",
    )
    obs = vec.reset()
    assert np.isfinite(obs).all()
    assert vec.num_envs == 2
    vec.close()


def test_ppo_model_initializes_and_tiny_learn_completes(tmp_path):
    vec = train.make_vec_env(
        n_envs=1,
        split="train",
        calibration_mode="recovered_calibration",
        reward_mode="financial",
        lambda_waste=0.10,
        scenario_id=train.DEFAULT_SCENARIO_ID,
        deterministic_demand=True,
        seed=123,
        monitor_dir=tmp_path,
    )
    params = train.ppo_params({"n_steps": 8, "batch_size": 8, "n_epochs": 1})
    model = PPO("MlpPolicy", vec, policy_kwargs=train.policy_kwargs(), seed=123, verbose=0, **params)
    model.learn(total_timesteps=8, progress_bar=False)
    obs = vec.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert int(np.asarray(action).ravel()[0]) in ACTION_MARKDOWNS
    model_path = tmp_path / "model.zip"
    vec_path = tmp_path / "vecnormalize.pkl"
    model.save(model_path)
    vec.save(vec_path)
    assert model_path.exists()
    assert vec_path.exists()
    eval_vec = VecNormalize.load(
        str(vec_path),
        DummyVecEnv(
            [
                lambda: train.make_env(
                    split="validation",
                    calibration_mode="recovered_calibration",
                    reward_mode="financial",
                    lambda_waste=0.10,
                    scenario_id=train.DEFAULT_SCENARIO_ID,
                    deterministic_demand=True,
                    seed=321,
                )
            ]
        ),
    )
    eval_vec.training = False
    eval_vec.norm_reward = False
    loaded = PPO.load(model_path, env=eval_vec)
    obs = eval_vec.reset()
    action, _ = loaded.predict(obs, deterministic=True)
    assert int(np.asarray(action).ravel()[0]) in ACTION_MARKDOWNS
    assert eval_vec.training is False
    eval_vec.close()
    vec.close()


def test_train_one_agent_smoke_metadata_contains_lambda_and_no_test_split():
    record = train.train_one_agent(
        agent=train.AgentSpec("recovered_financial_pytest", "recovered_calibration", "financial"),
        seed=2026,
        total_timesteps=8,
        stage="smoke",
        selected_lambda=None,
        config_overrides={"n_steps": 8, "batch_size": 8, "n_epochs": 1},
        n_envs=1,
    )
    summary = record["summary"]
    assert summary["test_split_used_for_training"] is False
    assert summary["lambda_waste"] == pytest.approx(0.10)
    assert np.isfinite(summary["best_validation_return"])
    model_dir = train.MODELS_DIR / "recovered_financial_pytest" / "seed_2026"
    assert (model_dir / "final_model.zip").exists()
    assert (model_dir / "vecnormalize.pkl").exists()


def test_same_seed_reproduces_deterministic_smoke_result():
    kwargs = dict(
        agent=train.AgentSpec("recovered_financial_repro", "recovered_calibration", "financial"),
        seed=42,
        total_timesteps=8,
        stage="smoke",
        selected_lambda=None,
        config_overrides={"n_steps": 8, "batch_size": 8, "n_epochs": 1},
        n_envs=1,
    )
    first = train.train_one_agent(**kwargs)["summary"]["best_validation_return"]
    second = train.train_one_agent(**kwargs)["summary"]["best_validation_return"]
    assert first == pytest.approx(second, abs=1e-8)


def test_financial_and_sustainability_configs_are_distinct():
    specs = train.make_agent_specs("pilot", selected_lambda=0.25)
    mapping = {spec.agent_id: (spec.calibration_mode, spec.reward_mode) for spec in specs}
    assert mapping["observed_financial"] != mapping["observed_sustainability"]
    assert mapping["recovered_financial"] != mapping["recovered_sustainability"]


def test_observed_and_recovered_artifact_paths_are_distinct():
    paths = train.artifact_paths()
    assert paths["discount_response_observed.joblib"] != paths["discount_response_recovered.joblib"]


def test_action_mapping_exact():
    assert [ACTION_MARKDOWNS[i] for i in sorted(ACTION_MARKDOWNS)] == [0.00, 0.05, 0.10, 0.20, 0.30, 0.40]


def test_cli_default_stage_is_smoke(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_ppo_operational.py"])
    args = train.parse_args()
    assert args.stage == "smoke"


def test_lambda_selection_metadata_written():
    selected, table = train.choose_sustainability_lambda()
    assert (train.TABLES_DIR / "ppo_lambda_selection.csv").exists()
    assert "status" in table.columns
    if selected is None:
        assert table["status"].iloc[0] == "PPO_SUSTAINABILITY_LAMBDA_REQUIRES_REVIEW"


def test_episode_manifest_pairing_is_preserved():
    manifest = create_episode_manifest("validation", episodes=3, scenario_id=train.DEFAULT_SCENARIO_ID, seed=500)
    required = {"episode_id", "store_id", "product_id", "start_date", "scenario_id", "age_profile_seed", "demand_noise_seed"}
    assert required.issubset(manifest.columns)
    assert manifest["episode_id"].is_unique


def test_baseline_evaluation_uses_same_manifest_and_finite_metrics():
    manifest = create_episode_manifest("validation", episodes=2, scenario_id=train.DEFAULT_SCENARIO_ID, seed=600)
    result = evaluate_baselines(
        manifest=manifest,
        calibration_mode="recovered_calibration",
        reward_mode="financial",
        lambda_waste=0.10,
        policies=["always_0pct", "inventory_coverage_rule"],
    )
    assert set(result["episode_id"]) == set(manifest["episode_id"])
    numeric = result.select_dtypes(include=[np.number])
    assert np.isfinite(numeric.to_numpy()).all()


def test_accounting_profit_not_confused_with_rl_return():
    env = OperationalPerishablePricingEnv(split="validation", calibration_mode="recovered_calibration", reward_mode="financial")
    obs, _ = env.reset(seed=99)
    terminated = truncated = False
    episode_return = 0.0
    info = {}
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(0)
        episode_return += reward
    assert "accounting_profit" in info
    assert "normalized_accounting_profit" in info
    assert info["accounting_profit"] != pytest.approx(episode_return)


def test_no_ppo_helper_modifies_source_parquet_or_response_models():
    watched = [
        train.artifact_paths()["freshretail_demand_recovered.parquet"],
        train.artifact_paths()["discount_response_observed.joblib"],
        train.artifact_paths()["discount_response_recovered.joblib"],
    ]
    before = {path: file_hash(path) for path in watched}
    train.audit_environment_compatibility()
    train.write_input_artifact_hashes()
    after = {path: file_hash(path) for path in watched}
    assert before == after


def test_evaluation_baseline_rules_return_valid_actions():
    obs = np.zeros(41, dtype=np.float32)
    obs[1] = 2.0
    obs[23] = 0.75
    rng = np.random.default_rng(1)
    for policy in train.BASELINE_POLICIES:
        assert baseline_action(policy, obs, {}, rng) in ACTION_MARKDOWNS


def test_environment_compatibility_audit_passes():
    audit = train.audit_environment_compatibility()
    assert not audit["compatibility_status"].isin(["fail", "missing"]).any()


def test_experiment_config_contains_artifact_hashes_and_locked_test_split():
    hashes = train.write_input_artifact_hashes()
    config = train.make_experiment_config(None, hashes)
    assert config["environment_parameters"]["locked_test_split"] == "test"
    assert "artifact_hashes" in config
    assert "primary_experiment_matrix" in config


def test_training_evaluation_metrics_are_finite_after_tiny_fit():
    record = train.train_one_agent(
        agent=train.AgentSpec("recovered_financial_metrics", "recovered_calibration", "financial"),
        seed=77,
        total_timesteps=8,
        stage="smoke",
        selected_lambda=None,
        config_overrides={"n_steps": 8, "batch_size": 8, "n_epochs": 1},
        n_envs=1,
    )
    df = record["evaluation"].select_dtypes(include=[np.number])
    assert np.isfinite(df.to_numpy()).all()


def test_smoke_stage_skips_sustainability_when_lambda_unselected():
    specs = train.make_agent_specs("smoke", selected_lambda=None)
    assert len(specs) == 1
    assert specs[0].reward_mode == "financial"


def test_pilot_stage_does_not_include_test_split_in_specs():
    specs = train.make_agent_specs("pilot", selected_lambda=None)
    assert specs
    assert all(spec.reward_mode == "financial" for spec in specs)

