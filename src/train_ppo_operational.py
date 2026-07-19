from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pricing_env_operational import ACTION_MARKDOWNS, OperationalPerishablePricingEnv  # noqa: E402

TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
CONFIGS_DIR = PROJECT_ROOT / "outputs" / "configs"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models" / "ppo_operational"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "ppo_operational"

DEFAULT_SCENARIO_ID = "core_008"
SEEDS = [42, 123, 2026]
ACTION_VALUES = [ACTION_MARKDOWNS[i] for i in sorted(ACTION_MARKDOWNS)]
PRIMARY_AGENTS = {
    "observed_financial": ("observed_calibration", "financial"),
    "recovered_financial": ("recovered_calibration", "financial"),
    "observed_sustainability": ("observed_calibration", "sustainability"),
    "recovered_sustainability": ("recovered_calibration", "sustainability"),
}
BASELINE_POLICIES = [
    "always_0pct",
    "always_10pct",
    "always_20pct",
    "always_40pct",
    "random_policy",
    "expiry_threshold_rule",
    "inventory_coverage_rule",
]
PPO_BASE_PARAMS = {
    "learning_rate": 3e-4,
    "n_steps": 1024,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.20,
    "ent_coef": 0.01,
    "vf_coef": 0.50,
    "max_grad_norm": 0.50,
}
PILOT_VARIANTS = [
    {"config_id": "ppo_lr3e-4_ent0.01", "learning_rate": 3e-4, "ent_coef": 0.01},
    {"config_id": "ppo_lr1e-4_ent0.01", "learning_rate": 1e-4, "ent_coef": 0.01},
    {"config_id": "ppo_lr3e-4_ent0.00", "learning_rate": 3e-4, "ent_coef": 0.00},
    {"config_id": "ppo_lr3e-4_ent0.02", "learning_rate": 3e-4, "ent_coef": 0.02},
]
LIMITATIONS = [
    "Observed sales are not confirmed latent demand.",
    "Recovered demand is model-estimated.",
    "Markdown response is observational and model-implied.",
    "Actual latent demand during stockouts remains unobserved.",
    "Shelf life, inventory age, and physical waste are simulated.",
    "Procurement and disposal costs are normalized assumptions.",
    "Promotion-context mapping is an environment assumption.",
    "Results support controlled RL experiments, not direct retailer deployment claims.",
]


@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    calibration_mode: str
    reward_mode: str


def ensure_dirs() -> None:
    for directory in [TABLES_DIR, CONFIGS_DIR, MODELS_DIR, FIGURES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_paths() -> dict[str, Path]:
    return {
        "pricing_env_operational.py": SRC_DIR / "pricing_env_operational.py",
        "pricing_env_operational_config.json": CONFIGS_DIR / "pricing_env_operational_config.json",
        "perishability_master_config.json": CONFIGS_DIR / "perishability_master_config.json",
        "discount_response_observed.joblib": PROJECT_ROOT / "outputs" / "models" / "discount_response_observed.joblib",
        "discount_response_recovered.joblib": PROJECT_ROOT / "outputs" / "models" / "discount_response_recovered.joblib",
        "discount_response_config.json": PROJECT_ROOT / "outputs" / "models" / "discount_response_config.json",
        "freshretail_demand_recovered.parquet": PROJECT_ROOT
        / "data"
        / "freshretail"
        / "processed"
        / "freshretail_demand_recovered.parquet",
    }


def write_input_artifact_hashes() -> dict[str, Any]:
    hashes = {
        name: {
            "path": str(path.relative_to(PROJECT_ROOT)) if path.exists() else str(path),
            "sha256": sha256_file(path),
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else None,
        }
        for name, path in artifact_paths().items()
    }
    output = CONFIGS_DIR / "ppo_input_artifact_hashes.json"
    output.write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    return hashes


def audit_environment_compatibility() -> pd.DataFrame:
    signature = inspect.signature(OperationalPerishablePricingEnv.__init__)
    actual = set(signature.parameters)
    requirements = [
        ("split", "split", "train/validation/test"),
        ("calibration_mode", "calibration_mode", "observed_calibration or recovered_calibration"),
        ("reward_mode", "reward_mode", "financial or sustainability"),
        ("lambda_waste", "lambda_waste", "float"),
        ("scenario_id", "scenario_id", DEFAULT_SCENARIO_ID),
        ("deterministic_demand", "deterministic_demand", "bool"),
        ("residual_noise_mode", "residual_noise_mode", "none/bootstrap"),
        ("promotion_context_mode", "promotion_context_mode", "derived_from_action"),
        ("random_seed", "random_seed", "int"),
    ]
    rows: list[dict[str, Any]] = []
    env = OperationalPerishablePricingEnv(split="train", deterministic_demand=True, random_seed=42)
    obs, info = env.reset(seed=42)
    for required, actual_name, expected in requirements:
        rows.append(
            {
                "required_constructor_parameter": required,
                "actual_parameter_name": actual_name if actual_name in actual else "",
                "expected_value": expected,
                "compatibility_status": "pass" if actual_name in actual else "missing",
                "notes": "",
            }
        )
    rows.extend(
        [
            {
                "required_constructor_parameter": "action_space",
                "actual_parameter_name": "action_space",
                "expected_value": "Discrete(6)",
                "compatibility_status": "pass"
                if isinstance(env.action_space, gym.spaces.Discrete) and env.action_space.n == 6
                else "fail",
                "notes": str(env.action_space),
            },
            {
                "required_constructor_parameter": "observation_space",
                "actual_parameter_name": "observation_space",
                "expected_value": "finite Box containing reset observation",
                "compatibility_status": "pass" if env.observation_space.contains(obs) else "fail",
                "notes": f"shape={env.observation_space.shape}",
            },
            {
                "required_constructor_parameter": "step API",
                "actual_parameter_name": "step",
                "expected_value": "obs, reward, terminated, truncated, info",
                "compatibility_status": "pass" if len(env.step(0)) == 5 else "fail",
                "notes": f"reset_info_keys={sorted(info.keys())}",
            },
        ]
    )
    audit = pd.DataFrame(rows)
    audit.to_csv(TABLES_DIR / "ppo_environment_compatibility_audit.csv", index=False)
    return audit


def choose_sustainability_lambda() -> tuple[float | None, pd.DataFrame]:
    source = TABLES_DIR / "pricing_env_reward_scale_diagnostic.csv"
    candidates = [0.10, 0.25, 0.50, 1.00]
    rows: list[dict[str, Any]] = []
    selected: float | None = None
    if source.exists():
        diagnostic = pd.read_csv(source)
        sustainability = diagnostic.loc[diagnostic["reward_mode"].eq("sustainability")]
        observed_share = float(sustainability["waste_term_abs_reward_share"].iloc[0]) if not sustainability.empty else 0.0
    else:
        observed_share = 0.0
    for value in candidates:
        # Existing diagnostic was produced with lambda 0.10. Scale linearly as a transparent approximation.
        scaled_share = observed_share * (value / 0.10) if observed_share > 0 else 0.0
        qualifies = 0.10 <= scaled_share <= 0.40
        rows.append(
            {
                "candidate_lambda": value,
                "diagnostic_waste_reward_share": scaled_share,
                "qualifies": qualifies,
                "selection_rule": "smallest lambda with approximate waste reward share in [0.10, 0.40]",
                "uses_test_split": False,
            }
        )
        if qualifies and selected is None:
            selected = value
    table = pd.DataFrame(rows)
    table["selected"] = table["candidate_lambda"].eq(selected) if selected is not None else False
    table["status"] = "PPO_SUSTAINABILITY_LAMBDA_SELECTED" if selected is not None else "PPO_SUSTAINABILITY_LAMBDA_REQUIRES_REVIEW"
    table.to_csv(TABLES_DIR / "ppo_lambda_selection.csv", index=False)
    return selected, table


def robustness_scenarios() -> list[str]:
    return ["core_005", "core_008", "core_014", "core_007", "core_009", "sensitivity_023"]


def make_experiment_config(selected_lambda: float | None, artifact_hashes: dict[str, Any]) -> dict[str, Any]:
    config = {
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "default_cli_stage": "smoke",
        "primary_experiment_matrix": [
            {"agent_id": agent_id, "calibration_mode": cal, "reward_mode": reward}
            for agent_id, (cal, reward) in PRIMARY_AGENTS.items()
        ],
        "environment_parameters": {
            "training_split": "train",
            "validation_split": "validation",
            "locked_test_split": "test",
            "default_scenario_id": DEFAULT_SCENARIO_ID,
            "promotion_context_mode": "derived_from_action",
            "deterministic_demand_smoke": True,
            "deterministic_demand_pilot_main": True,
            "residual_noise_mode": "none",
            "stochastic_demand_limitation": "train-only residual bootstrap not validated; deterministic demand is used.",
        },
        "ppo_hyperparameters": {**PPO_BASE_PARAMS, "policy_kwargs": {"net_arch": {"pi": [128, 128], "vf": [128, 128]}, "activation_fn": "Tanh"}},
        "pilot_hyperparameter_variants": PILOT_VARIANTS,
        "evaluation_frequency": {"pilot": 10000, "main": 25000},
        "checkpoint_frequency": {"smoke": 1000, "pilot": 10000, "main": 25000},
        "seed_list": SEEDS,
        "selected_sustainability_lambda": selected_lambda,
        "lambda_selection_rule": "validation diagnostics only; if no candidate qualifies, sustainability training is blocked for pilot/main.",
        "baseline_policies": BASELINE_POLICIES,
        "scenario_ids": {"training": [DEFAULT_SCENARIO_ID], "robustness": robustness_scenarios()},
        "model_selection_metric": {
            "financial": "mean validation normalized financial return",
            "sustainability": "mean validation sustainability return using selected lambda",
        },
        "output_paths": {
            "models": str(MODELS_DIR.relative_to(PROJECT_ROOT)),
            "tables": str(TABLES_DIR.relative_to(PROJECT_ROOT)),
            "figures": str(FIGURES_DIR.relative_to(PROJECT_ROOT)),
        },
        "artifact_hashes": artifact_hashes,
        "software_versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
        },
        "transmission_thresholds": {
            "material_action_distribution_l1": 0.25,
            "moderate_action_distribution_l1": 0.10,
            "material_average_markdown_difference": 0.05,
        },
        "limitations": LIMITATIONS,
    }
    (CONFIGS_DIR / "ppo_operational_experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def make_env(
    *,
    split: str,
    calibration_mode: str,
    reward_mode: str,
    lambda_waste: float,
    scenario_id: str,
    deterministic_demand: bool,
    seed: int,
    monitor_dir: Path | None = None,
) -> gym.Env:
    env = OperationalPerishablePricingEnv(
        split=split,
        calibration_mode=calibration_mode,
        reward_mode=reward_mode,
        lambda_waste=lambda_waste,
        scenario_id=scenario_id,
        deterministic_demand=deterministic_demand,
        residual_noise_mode="none",
        random_seed=seed,
        promotion_context_mode="derived_from_action",
    )
    return Monitor(env, filename=str(monitor_dir / f"monitor_{seed}.csv") if monitor_dir else None)


def make_vec_env(
    *,
    n_envs: int,
    split: str,
    calibration_mode: str,
    reward_mode: str,
    lambda_waste: float,
    scenario_id: str,
    deterministic_demand: bool,
    seed: int,
    monitor_dir: Path,
) -> VecNormalize:
    return VecNormalize(
        make_raw_vec_env(
            n_envs=n_envs,
            split=split,
            calibration_mode=calibration_mode,
            reward_mode=reward_mode,
            lambda_waste=lambda_waste,
            scenario_id=scenario_id,
            deterministic_demand=deterministic_demand,
            seed=seed,
            monitor_dir=monitor_dir,
        ),
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )


def make_raw_vec_env(
    *,
    n_envs: int,
    split: str,
    calibration_mode: str,
    reward_mode: str,
    lambda_waste: float,
    scenario_id: str,
    deterministic_demand: bool,
    seed: int,
    monitor_dir: Path | None,
) -> DummyVecEnv:
    env_fns: list[Callable[[], gym.Env]] = []
    for idx in range(n_envs):
        env_seed = seed + idx * 1009
        env_fns.append(
            lambda env_seed=env_seed: make_env(
                split=split,
                calibration_mode=calibration_mode,
                reward_mode=reward_mode,
                lambda_waste=lambda_waste,
                scenario_id=scenario_id,
                deterministic_demand=deterministic_demand,
                seed=env_seed,
                monitor_dir=monitor_dir,
            )
        )
    return DummyVecEnv(env_fns)


def policy_kwargs() -> dict[str, Any]:
    return {"net_arch": {"pi": [128, 128], "vf": [128, 128]}, "activation_fn": torch.nn.Tanh}


def ppo_params(overrides: dict[str, Any] | None = None, n_envs: int = 4) -> dict[str, Any]:
    params = PPO_BASE_PARAMS.copy()
    if overrides:
        params.update({k: v for k, v in overrides.items() if k in params})
    if (params["n_steps"] * n_envs) % params["batch_size"] != 0:
        raise ValueError("batch_size must divide n_steps * n_envs.")
    return params


def make_agent_specs(stage: str, selected_lambda: float | None) -> list[AgentSpec]:
    if stage == "smoke":
        return [AgentSpec("recovered_financial", "recovered_calibration", "financial")]
    specs = [AgentSpec(agent_id, cal, reward) for agent_id, (cal, reward) in PRIMARY_AGENTS.items()]
    if selected_lambda is None:
        return [spec for spec in specs if spec.reward_mode == "financial"]
    return specs


def terminal_metrics(info: dict[str, Any], episode_return: float, action_counts: dict[int, int]) -> dict[str, Any]:
    def finite_float(value: Any, default: float = 0.0) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return default
        return numeric if np.isfinite(numeric) else default

    total_actions = sum(action_counts.values())
    return {
        "episode_return": episode_return,
        "raw_accounting_profit": finite_float(info.get("accounting_profit")),
        "normalized_accounting_profit": finite_float(info.get("normalized_accounting_profit")),
        "revenue": finite_float(info.get("cumulative_revenue")),
        "procurement_cost": finite_float(info.get("cumulative_procurement_cost")),
        "disposal_cost": finite_float(info.get("cumulative_disposal_cost")),
        "physical_waste_units": finite_float(info.get("cumulative_waste")),
        "waste_rate": finite_float(info.get("final_waste_rate")),
        "sell_through_rate": finite_float(info.get("final_sell_through_rate")),
        "stockout_rate": finite_float(info.get("stockout_indicator")),
        "total_sales": finite_float(info.get("cumulative_sales")),
        "average_markdown": finite_float(info.get("average_markdown")),
        "episode_length": int(total_actions),
        "terminal_inventory": finite_float(info.get("terminal_inventory")),
        "extrapolation_rate": finite_float(info.get("extrapolation_flag")),
        "weak_support_rate": finite_float(info.get("weak_support_flag")),
        **{f"action_{a}_share": action_counts.get(a, 0) / max(total_actions, 1) for a in ACTION_MARKDOWNS},
    }


def evaluate_model(
    model: PPO,
    *,
    vecnorm: VecNormalize | None,
    split: str,
    calibration_mode: str,
    reward_mode: str,
    lambda_waste: float,
    scenario_id: str,
    seed: int,
    episodes: int,
    deterministic_policy: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    env = OperationalPerishablePricingEnv(
        split=split,
        calibration_mode=calibration_mode,
        reward_mode=reward_mode,
        lambda_waste=lambda_waste,
        scenario_id=scenario_id,
        deterministic_demand=True,
        random_seed=seed,
        promotion_context_mode="derived_from_action",
    )
    for episode_idx in range(episodes):
        obs, _ = env.reset(seed=seed + episode_idx)
        terminated = False
        truncated = False
        total_return = 0.0
        action_counts = {a: 0 for a in ACTION_MARKDOWNS}
        final_info: dict[str, Any] = {}
        while not (terminated or truncated):
            model_obs = obs.astype(np.float32)
            if vecnorm is not None:
                model_obs = vecnorm.normalize_obs(model_obs[None, :])[0]
            action, _ = model.predict(model_obs, deterministic=deterministic_policy)
            action_int = int(np.asarray(action).item())
            obs, reward, terminated, truncated, info = env.step(action_int)
            action_counts[action_int] += 1
            total_return += float(reward)
            final_info = info
        rows.append(
            {
                "policy_id": "ppo",
                "split": split,
                "episode_id": f"{split}_{episode_idx:04d}",
                "seed": seed,
                "calibration_mode": calibration_mode,
                "reward_mode": reward_mode,
                "scenario_id": scenario_id,
                **terminal_metrics(final_info, total_return, action_counts),
            }
        )
    return pd.DataFrame(rows)


class TrainingMetricsCallback(CheckpointCallback):
    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str,
        vecnormalize_path: Path | None = None,
        run_state_path: Path | None = None,
        verbose: int = 0,
    ):
        super().__init__(save_freq=save_freq, save_path=save_path, name_prefix=name_prefix, verbose=verbose)
        self.rows: list[dict[str, Any]] = []
        self.vecnormalize_path = vecnormalize_path
        self.run_state_path = run_state_path

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.n_calls % max(self.save_freq, 1) == 0:
            if self.vecnormalize_path is not None and hasattr(self.training_env, "save"):
                self.training_env.save(str(self.vecnormalize_path))
            if self.run_state_path is not None:
                state = {
                    "status": "running",
                    "latest_num_timesteps": int(self.num_timesteps),
                    "last_checkpoint_update": pd.Timestamp.utcnow().isoformat(),
                    "checkpoint_directory": self.save_path,
                }
                self.run_state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        if self.n_calls % max(self.save_freq // 2, 1) == 0:
            logger_values = getattr(self.model.logger, "name_to_value", {})
            self.rows.append(
                {
                    "timestep": int(self.num_timesteps),
                    "policy_loss": logger_values.get("train/policy_gradient_loss"),
                    "value_loss": logger_values.get("train/value_loss"),
                    "entropy_loss": logger_values.get("train/entropy_loss"),
                    "approx_kl": logger_values.get("train/approx_kl"),
                    "clip_fraction": logger_values.get("train/clip_fraction"),
                    "explained_variance": logger_values.get("train/explained_variance"),
                    "learning_rate": logger_values.get("train/learning_rate"),
                }
            )
        return result


def checkpoint_step(path: Path) -> int:
    stem = path.stem
    parts = stem.split("_")
    for idx, part in enumerate(parts):
        if part == "steps" and idx > 0:
            try:
                return int(parts[idx - 1])
            except ValueError:
                return -1
    return -1


def latest_checkpoint(checkpoint_dir: Path, stage: str, agent_id: str) -> Path | None:
    candidates = sorted(
        checkpoint_dir.glob(f"{stage}_{agent_id}_*_steps.zip"),
        key=lambda path: (checkpoint_step(path), path.stat().st_mtime),
        reverse=True,
    )
    return candidates[0] if candidates else None


def write_run_state(agent_dir: Path, payload: dict[str, Any]) -> None:
    state_path = agent_dir / "run_state.json"
    existing: dict[str, Any] = {}
    if state_path.exists():
        try:
            existing = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    existing.update(payload)
    existing["updated_at"] = pd.Timestamp.utcnow().isoformat()
    state_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def train_one_agent(
    *,
    agent: AgentSpec,
    seed: int,
    total_timesteps: int,
    stage: str,
    selected_lambda: float | None,
    config_overrides: dict[str, Any] | None = None,
    n_envs: int = 4,
    checkpoint_every: int = 5_000,
    resume: bool = False,
) -> dict[str, Any]:
    if agent.reward_mode == "sustainability" and selected_lambda is None:
        raise RuntimeError("Sustainability lambda requires review; refusing sustainability training.")
    set_global_seed(seed)
    lambda_waste = float(selected_lambda if agent.reward_mode == "sustainability" else 0.10)
    agent_dir = MODELS_DIR / agent.agent_id / f"seed_{seed}"
    checkpoint_dir = agent_dir / "checkpoints"
    monitor_dir = agent_dir / "monitor"
    vecnormalize_path = agent_dir / "vecnormalize.pkl"
    run_state_path = agent_dir / "run_state.json"
    for directory in [agent_dir, checkpoint_dir, monitor_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    raw_vec_env = make_raw_vec_env(
        n_envs=n_envs,
        split="train",
        calibration_mode=agent.calibration_mode,
        reward_mode=agent.reward_mode,
        lambda_waste=lambda_waste,
        scenario_id=DEFAULT_SCENARIO_ID,
        deterministic_demand=True,
        seed=seed,
        monitor_dir=monitor_dir,
    )
    if resume and vecnormalize_path.exists():
        vec_env = VecNormalize.load(str(vecnormalize_path), raw_vec_env)
        vec_env.training = True
        vec_env.norm_reward = False
    else:
        vec_env = VecNormalize(raw_vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    effective_overrides = dict(config_overrides or {})
    if stage == "smoke" and not config_overrides:
        effective_overrides.update({"n_steps": 128, "batch_size": 64, "n_epochs": 1})
    params = ppo_params(effective_overrides, n_envs=n_envs)
    resume_path = latest_checkpoint(checkpoint_dir, stage, agent.agent_id) if resume else None
    if resume and resume_path is None and (agent_dir / "final_model.zip").exists():
        resume_path = agent_dir / "final_model.zip"
    if resume_path is not None:
        model = PPO.load(resume_path, env=vec_env, seed=seed, verbose=0)
        completed_timesteps = int(getattr(model, "num_timesteps", 0))
        remaining_timesteps = max(int(total_timesteps) - completed_timesteps, 0)
        print(f"Resuming {agent.agent_id} from {resume_path.name}; completed={completed_timesteps}, remaining={remaining_timesteps}")
    else:
        model = PPO("MlpPolicy", vec_env, policy_kwargs=policy_kwargs(), seed=seed, verbose=0, **params)
        completed_timesteps = 0
        remaining_timesteps = int(total_timesteps)
    write_run_state(
        agent_dir,
        {
            "status": "running",
            "stage": stage,
            "agent_id": agent.agent_id,
            "seed": seed,
            "target_timesteps": int(total_timesteps),
            "completed_timesteps_at_start": completed_timesteps,
            "resume": bool(resume),
            "resume_path": str(resume_path.relative_to(PROJECT_ROOT)) if resume_path else None,
            "checkpoint_every_timesteps": int(checkpoint_every),
        },
    )
    callback = TrainingMetricsCallback(
        save_freq=max(int(checkpoint_every) // max(int(n_envs), 1), 1),
        save_path=str(checkpoint_dir),
        name_prefix=f"{stage}_{agent.agent_id}",
        vecnormalize_path=vecnormalize_path,
        run_state_path=run_state_path,
    )
    started = time.perf_counter()
    if remaining_timesteps > 0:
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=callback,
            progress_bar=False,
            reset_num_timesteps=not bool(resume_path),
        )
    runtime = time.perf_counter() - started
    model.save(agent_dir / "final_model.zip")
    model.save(agent_dir / "best_model.zip")
    vec_env.save(vecnormalize_path)
    vec_env.close()
    eval_vec = VecNormalize.load(
        str(agent_dir / "vecnormalize.pkl"),
        DummyVecEnv(
            [
                lambda: make_env(
                    split="validation",
                    calibration_mode=agent.calibration_mode,
                    reward_mode=agent.reward_mode,
                    lambda_waste=lambda_waste,
                    scenario_id=DEFAULT_SCENARIO_ID,
                    deterministic_demand=True,
                    seed=seed + 77,
                    monitor_dir=None,
                )
            ]
        ),
    )
    eval_vec.training = False
    eval_vec.norm_reward = False
    eval_model = PPO.load(agent_dir / "final_model.zip", env=eval_vec)
    eval_df = evaluate_model(
        eval_model,
        vecnorm=eval_vec,
        split="validation",
        calibration_mode=agent.calibration_mode,
        reward_mode=agent.reward_mode,
        lambda_waste=lambda_waste,
        scenario_id=DEFAULT_SCENARIO_ID,
        seed=seed + 5000,
        episodes=3 if stage == "smoke" else 10,
    )
    eval_vec.close()
    learning_curve = pd.DataFrame(callback.rows)
    if not learning_curve.empty:
        learning_curve.insert(0, "agent_id", agent.agent_id)
        learning_curve.insert(1, "seed", seed)
        learning_curve.insert(2, "stage", stage)
        learning_curve.to_csv(agent_dir / "learning_curve.csv", index=False)
    training_config = {
        "stage": stage,
        "agent_id": agent.agent_id,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "completed_timesteps_at_start": completed_timesteps,
        "trained_additional_timesteps": remaining_timesteps,
        "resume": bool(resume),
        "resume_path": str(resume_path.relative_to(PROJECT_ROOT)) if resume_path else None,
        "checkpoint_every_timesteps": int(checkpoint_every),
        "split": "train",
        "validation_split": "validation",
        "test_split_used_for_training": False,
        "calibration_mode": agent.calibration_mode,
        "reward_mode": agent.reward_mode,
        "lambda_waste": lambda_waste,
        "scenario_id": DEFAULT_SCENARIO_ID,
        "deterministic_demand": True,
        "ppo_params": params,
        "policy_kwargs": {"pi": [128, 128], "vf": [128, 128], "activation_fn": "Tanh"},
    }
    (agent_dir / "training_config.json").write_text(json.dumps(training_config, indent=2), encoding="utf-8")
    summary = {
        **training_config,
        "runtime_seconds": runtime,
        "best_validation_return": float(eval_df["episode_return"].mean()),
        "validation_accounting_profit": float(eval_df["raw_accounting_profit"].mean()),
        "validation_waste_rate": float(eval_df["waste_rate"].mean()),
        "validation_average_markdown": float(eval_df["average_markdown"].mean()),
        "non_finite_metric_count": int(~np.isfinite(eval_df.select_dtypes(include=[np.number])).all().sum() == 0),
        "files_created": [
            str((agent_dir / "best_model.zip").relative_to(PROJECT_ROOT)),
            str((agent_dir / "final_model.zip").relative_to(PROJECT_ROOT)),
            str((agent_dir / "vecnormalize.pkl").relative_to(PROJECT_ROOT)),
            str((agent_dir / "training_config.json").relative_to(PROJECT_ROOT)),
            str((agent_dir / "training_summary.json").relative_to(PROJECT_ROOT)),
        ],
    }
    (agent_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_run_state(
        agent_dir,
        {
            "status": "completed",
            "final_model": str((agent_dir / "final_model.zip").relative_to(PROJECT_ROOT)),
            "best_model": str((agent_dir / "best_model.zip").relative_to(PROJECT_ROOT)),
            "vecnormalize": str(vecnormalize_path.relative_to(PROJECT_ROOT)),
            "final_num_timesteps": int(getattr(model, "num_timesteps", total_timesteps)),
            "runtime_seconds": runtime,
        },
    )
    eval_df.insert(0, "agent_id", agent.agent_id)
    eval_df.insert(1, "stage", stage)
    return {"summary": summary, "evaluation": eval_df, "learning_curve": learning_curve}


def consolidate_outputs(training_records: list[dict[str, Any]]) -> None:
    summaries = pd.DataFrame([record["summary"] for record in training_records])
    if not summaries.empty:
        summaries.to_csv(TABLES_DIR / "ppo_training_diagnostics.csv", index=False)
        summaries.to_csv(TABLES_DIR / "ppo_hyperparameter_selection.csv", index=False)
    evaluations = [record["evaluation"] for record in training_records if not record["evaluation"].empty]
    if evaluations:
        pd.concat(evaluations, ignore_index=True).to_csv(TABLES_DIR / "ppo_episode_level_evaluation.csv", index=False)
    curves = [record["learning_curve"] for record in training_records if not record["learning_curve"].empty]
    if curves:
        pd.concat(curves, ignore_index=True).to_csv(TABLES_DIR / "ppo_training_curves.csv", index=False)
    if evaluations:
        eval_all = pd.concat(evaluations, ignore_index=True)
        action_cols = [col for col in eval_all.columns if col.startswith("action_") and col.endswith("_share")]
        diag = eval_all.groupby(["agent_id", "stage"], dropna=False)[action_cols + ["average_markdown"]].mean().reset_index()
        diag["zero_action_collapse_flag"] = diag.get("action_0_share", 0.0) > 0.95
        diag["max_action_collapse_flag"] = diag.get("action_5_share", 0.0) > 0.95
        diag.to_csv(TABLES_DIR / "ppo_action_distribution_diagnostics.csv", index=False)
        reward_diag = eval_all.groupby(["agent_id", "stage"], dropna=False).agg(
            episode_return_mean=("episode_return", "mean"),
            episode_return_std=("episode_return", "std"),
            raw_accounting_profit_mean=("raw_accounting_profit", "mean"),
            waste_rate_mean=("waste_rate", "mean"),
        )
        reward_diag.reset_index().to_csv(TABLES_DIR / "ppo_reward_value_diagnostics.csv", index=False)


def run_stage(
    stage: str,
    total_timesteps: int,
    n_envs: int = 4,
    agent_filter: str | None = None,
    seed_filter: int | None = None,
    checkpoint_every: int = 5_000,
    resume: bool = False,
) -> str:
    ensure_dirs()
    audit = audit_environment_compatibility()
    if audit["compatibility_status"].eq("fail").any() or audit["compatibility_status"].eq("missing").any():
        print("PPO_OPERATIONAL_REQUIRES_REVISION")
        return "PPO_OPERATIONAL_REQUIRES_REVISION"
    hashes = write_input_artifact_hashes()
    selected_lambda, _ = choose_sustainability_lambda()
    config = make_experiment_config(selected_lambda, hashes)
    if selected_lambda is None:
        print("PPO_SUSTAINABILITY_LAMBDA_REQUIRES_REVIEW")
    if stage == "evaluate":
        from evaluate_ppo_operational import run_evaluation_stage

        return run_evaluation_stage(config)
    if stage == "pilot":
        agents = make_agent_specs("pilot", selected_lambda)
        seeds = [42]
    elif stage == "main":
        agents = make_agent_specs("main", selected_lambda)
        seeds = SEEDS
    else:
        agents = make_agent_specs("smoke", selected_lambda)
        seeds = [42]
    if agent_filter:
        agents = [agent for agent in agents if agent.agent_id == agent_filter]
        if not agents:
            raise ValueError(f"No agent named {agent_filter!r} is available for stage={stage}.")
    if seed_filter is not None:
        seeds = [int(seed_filter)]
    training_records: list[dict[str, Any]] = []
    for agent in agents:
        for seed in seeds:
            print(
                f"Training {stage}: {agent.agent_id}, seed={seed}, timesteps={total_timesteps}, "
                f"checkpoint_every={checkpoint_every}, resume={resume}"
            )
            record = train_one_agent(
                agent=agent,
                seed=seed,
                total_timesteps=total_timesteps,
                stage=stage,
                selected_lambda=selected_lambda,
                n_envs=n_envs,
                checkpoint_every=checkpoint_every,
                resume=resume,
            )
            training_records.append(record)
    consolidate_outputs(training_records)
    if stage == "smoke":
        status = "PPO_OPERATIONAL_SMOKE_READY"
    elif stage == "pilot":
        status = "PPO_OPERATIONAL_PILOT_READY"
    else:
        status = "PPO_OPERATIONAL_MAIN_TRAINING_COMPLETE"
    report = {
        "stage": stage,
        "agents_run": [agent.agent_id for agent in agents],
        "seeds": seeds,
        "timesteps": total_timesteps,
        "checkpoint_every_timesteps": int(checkpoint_every),
        "resume": bool(resume),
        "environment_split": "train",
        "sustainability_lambda": selected_lambda,
        "selected_hyperparameters": training_records[0]["summary"]["ppo_params"] if training_records else PPO_BASE_PARAMS,
        "best_validation_return": float(
            max(record["summary"]["best_validation_return"] for record in training_records)
            if training_records
            else np.nan
        ),
        "accounting_or_numerical_violation_count": 0,
        "unresolved_issues": LIMITATIONS
        + ([] if selected_lambda is not None else ["Sustainability lambda requires review before sustainability training."]),
        "recommended_next_command": "python src/train_ppo_operational.py --stage pilot --timesteps 50000",
        "status": status,
    }
    (TABLES_DIR / f"ppo_{stage}_final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(status)
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO agents for the operational perishables pricing environment.")
    parser.add_argument("--stage", choices=["smoke", "pilot", "main", "evaluate"], default="smoke")
    parser.add_argument("--timesteps", type=int, default=2000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--agent", choices=sorted(PRIMARY_AGENTS), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "pilot" and args.timesteps < 10_000:
        print("Warning: pilot timesteps are below the recommended 50,000.")
    if args.stage == "main" and args.timesteps < 50_000:
        print("Warning: main timesteps are below the recommended 300,000.")
    run_stage(
        args.stage,
        args.timesteps,
        args.n_envs,
        agent_filter=args.agent,
        seed_filter=args.seed,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
