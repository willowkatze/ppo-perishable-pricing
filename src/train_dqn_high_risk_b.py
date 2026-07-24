"""Train and validate DQN candidates for the locked HIGH_RISK_B task.

Training uses HIGH_RISK_B episodes from the train split, and model selection
uses the fixed validation manifest. This module does not access the held-out
test split.

Run from project root:
    python -u src/train_dqn_high_risk_b.py

Fast smoke run:
    python -u src/train_dqn_high_risk_b.py --timesteps 5000 --seeds 42

Main outputs:
    outputs/tables/dqn_environment_compatibility_audit.csv
    outputs/tables/dqn_training_population_audit.csv
    outputs/tables/dqn_checkpoint_validation_episode_results.csv
    outputs/tables/dqn_checkpoint_validation_summary.csv
    outputs/tables/dqn_final_candidate_comparison.csv
    outputs/tables/dqn_seed_stability.csv
    outputs/configs/dqn_locked_candidate.json
    outputs/configs/dqn_locked_candidate_hashes.json
    outputs/figures/dqn_final_candidate/
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces

if __package__:
    from src.pricing_env_operational import ACTION_MARKDOWNS, OperationalPerishablePricingEnv
else:
    from pricing_env_operational import ACTION_MARKDOWNS, OperationalPerishablePricingEnv

try:
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except Exception as exc:  # pragma: no cover
    raise ImportError("stable-baselines3 is required for the final DQN experiment.") from exc


# ---------------------------------------------------------------------------
# Paths and fixed experiment settings
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
CONFIGS_DIR = PROJECT_ROOT / "outputs" / "configs"
MANIFESTS_DIR = PROJECT_ROOT / "outputs" / "manifests"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models" / "dqn_high_risk_b"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "dqn_final_candidate"
DOCS_DIR = PROJECT_ROOT / "docs"

POPULATION_ID = "HIGH_RISK_B"
CALIBRATION_MODE = "recovered_calibration"
REWARD_MODE = "financial"
LAMBDA_WASTE = 0.10
BASELINE_POLICY = "always_0pct"
PLANNING_POLICY = "limited_horizon_planning_policy"
DISTILLED_POLICY = "planning_distilled_policy"
SELECTED_PPO_POLICY = "BALANCED__checkpoint_20000"
ACTION_SPACE = list(range(6))
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 20260718
TIE_TOL = 1e-9
PLANNING_HORIZON = 3


# ---------------------------------------------------------------------------
# Experiment records and environment construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalEpisode:
    scenario_id: str
    calibration_mode: str
    episode_index: int
    episode_id: str


@dataclass(frozen=True)
class DQNConfig:
    config_id: str
    learning_rate: float
    target_update_interval: int
    stability_adjustment: str


DQN_CONFIGS = [
    DQNConfig(
        config_id="STANDARD_DQN",
        learning_rate=3e-4,
        target_update_interval=1000,
        stability_adjustment="none",
    ),
    DQNConfig(
        config_id="STABLE_DQN",
        learning_rate=1e-4,
        target_update_interval=1000,
        stability_adjustment="lower_learning_rate_only",
    ),
]


def ensure_dirs() -> None:
    for path in [TABLES_DIR, CONFIGS_DIR, MANIFESTS_DIR, MODELS_DIR, FIGURES_DIR, DOCS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def read_csv(path: Path) -> pd.DataFrame:
    require_file(path)
    return pd.read_csv(path)


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def action_entropy(actions: list[int] | pd.Series) -> float:
    s = pd.Series(actions)
    if s.empty:
        return 0.0
    probs = s.value_counts(normalize=True)
    return float(-(probs * np.log(probs)).sum())


# ---------------------------------------------------------------------------
# Paired validation statistics
# ---------------------------------------------------------------------------


def bootstrap_ci(diff: pd.Series) -> tuple[float, float]:
    values = diff.dropna().astype(float).to_numpy()
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(BOOTSTRAP_N)
    for i in range(BOOTSTRAP_N):
        samples[i] = rng.choice(values, size=len(values), replace=True).mean()
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def make_env(split: str, scenario_id: str, seed: int) -> OperationalPerishablePricingEnv:
    return OperationalPerishablePricingEnv(
        split=split,
        calibration_mode=CALIBRATION_MODE,
        reward_mode=REWARD_MODE,
        lambda_waste=LAMBDA_WASTE,
        scenario_id=scenario_id,
        deterministic_demand=True,
        random_seed=seed,
        promotion_context_mode="derived_from_action",
    )


def reset_env(env: OperationalPerishablePricingEnv, episode_index: int) -> tuple[np.ndarray, dict[str, Any]]:
    return env.reset(seed=30_000 + int(episode_index), options={"episode_index": int(episode_index)})


def load_training_episode_table() -> pd.DataFrame:
    labels_path = TABLES_DIR / "high_risk_b_planning_training_labels.csv"
    if not labels_path.exists():
        labels_path = TABLES_DIR / "high_risk_b_planning_training_labels.partial.csv"
    labels = read_csv(labels_path)
    required = {"scenario_id", "episode_index", "episode_id", "group_id"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"Training-label table is missing required columns: {sorted(missing)}")
    episodes = (
        labels[["scenario_id", "episode_index", "episode_id", "group_id"]]
        .drop_duplicates()
        .sort_values(["scenario_id", "episode_index"])
        .reset_index(drop=True)
    )
    episodes["calibration_mode"] = CALIBRATION_MODE
    return episodes


def load_validation_manifest() -> pd.DataFrame:
    manifest_path = MANIFESTS_DIR / "high_risk_b_expanded_validation_manifest.csv"
    if not manifest_path.exists():
        manifest_path = TABLES_DIR / "limited_horizon_planning_episode_results.csv"
    manifest = read_csv(manifest_path)
    if "policy_id" in manifest.columns:
        manifest = manifest.loc[manifest["policy_id"].eq(PLANNING_POLICY)]
    cols = ["scenario_id", "calibration_mode", "episode_index", "episode_id"]
    missing = set(cols).difference(manifest.columns)
    if missing:
        raise ValueError(f"Validation manifest is missing required columns: {sorted(missing)}")
    manifest = manifest[cols].drop_duplicates().sort_values(["scenario_id", "episode_index"]).reset_index(drop=True)
    if len(manifest) != 22:
        print(f"WARNING: validation manifest episode count is {len(manifest)}, expected locked count 22.")
    if str(manifest["calibration_mode"].iloc[0]) != CALIBRATION_MODE:
        raise ValueError("Validation manifest calibration is not recovered_calibration.")
    return manifest


class HighRiskBTrainingEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(self, eligible_episodes: pd.DataFrame, seed: int, scenario_balanced: bool = True) -> None:
        super().__init__()
        self.eligible = eligible_episodes.reset_index(drop=True).copy()
        if self.eligible.empty:
            raise ValueError("No eligible HIGH_RISK_B training episodes found.")
        self.rng = np.random.default_rng(seed)
        self.scenario_balanced = scenario_balanced
        self.envs: dict[str, OperationalPerishablePricingEnv] = {}
        for scenario_id in sorted(self.eligible["scenario_id"].astype(str).unique()):
            self.envs[scenario_id] = make_env("train", scenario_id, seed)
        first = next(iter(self.envs.values()))
        self.action_space = first.action_space
        self.observation_space = first.observation_space
        self.current_env = first

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if options and "episode_index" in options and "scenario_id" in options:
            scenario_id = str(options["scenario_id"])
            episode_index = int(options["episode_index"])
        elif self.scenario_balanced:
            scenario_id = str(self.rng.choice(sorted(self.eligible["scenario_id"].astype(str).unique())))
            subset = self.eligible.loc[self.eligible["scenario_id"].astype(str).eq(scenario_id)]
            row = subset.iloc[int(self.rng.integers(0, len(subset)))]
            episode_index = int(row["episode_index"])
        else:
            row = self.eligible.iloc[int(self.rng.integers(0, len(self.eligible)))]
            scenario_id = str(row["scenario_id"])
            episode_index = int(row["episode_index"])
        self.current_env = self.envs[scenario_id]
        return self.current_env.reset(seed=seed, options={"episode_index": episode_index})

    def step(self, action: int):
        return self.current_env.step(int(action))

    def render(self):
        return self.current_env.render()

    def close(self) -> None:
        for env in self.envs.values():
            env.close()


def make_vec_env(eligible: pd.DataFrame, seed: int, n_envs: int, monitor_dir: Path) -> VecNormalize:
    def make_one(offset: int):
        def _factory():
            env = HighRiskBTrainingEnv(eligible, seed + offset * 1009, scenario_balanced=True)
            return Monitor(env, filename=str(monitor_dir / f"monitor_{seed + offset * 1009}.csv"))

        return _factory

    return VecNormalize(
        DummyVecEnv([make_one(i) for i in range(n_envs)]),
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )


# ---------------------------------------------------------------------------
# Environment compatibility
# ---------------------------------------------------------------------------


def bounded_termination_rollout(env: gym.Env, action: int = 0, max_steps: int | None = None) -> dict[str, Any]:
    """Run one bounded episode and report whether Gymnasium termination is reachable."""
    horizon = max(1, int(getattr(env, "horizon", 0)))
    step_limit = int(max_steps) if max_steps is not None else horizon
    if step_limit < 1:
        raise ValueError("max_steps must be positive")

    steps = 0
    terminated = False
    truncated = False
    flags_are_boolean = True
    while steps < step_limit and not (terminated or truncated):
        _, _, terminated_raw, truncated_raw, _ = env.step(int(action))
        flags_are_boolean = flags_are_boolean and isinstance(terminated_raw, (bool, np.bool_))
        flags_are_boolean = flags_are_boolean and isinstance(truncated_raw, (bool, np.bool_))
        terminated = bool(terminated_raw)
        truncated = bool(truncated_raw)
        steps += 1

    terminal_reached = bool(terminated or truncated)
    return {
        "flags_are_boolean": bool(flags_are_boolean),
        "terminal_reached": terminal_reached,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "steps": int(steps),
        "max_steps": int(step_limit),
        "bounded_completion": bool(terminal_reached and steps <= step_limit),
    }


def audit_environment_compatibility() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    train_episodes = load_training_episode_table()
    validation_manifest = load_validation_manifest()
    env = make_env("train", str(train_episodes["scenario_id"].iloc[0]), BOOTSTRAP_SEED)
    obs, _ = reset_env(env, int(train_episodes["episode_index"].iloc[0]))
    rewards = []
    finite_seen = True
    for action in ACTION_SPACE:
        probe = make_env("train", str(train_episodes["scenario_id"].iloc[0]), BOOTSTRAP_SEED + action)
        reset_env(probe, int(train_episodes["episode_index"].iloc[0]))
        _, reward, _, _, info = probe.step(action)
        rewards.append(safe_float(reward))
        finite_seen = finite_seen and np.isfinite(reward) and np.isfinite(safe_float(info.get("normalized_accounting_profit"), 0.0))
        probe.close()

    termination_probe = make_env("train", str(train_episodes["scenario_id"].iloc[0]), BOOTSTRAP_SEED)
    try:
        reset_env(termination_probe, int(train_episodes["episode_index"].iloc[0]))
        termination_audit = bounded_termination_rollout(termination_probe, action=0)
    finally:
        termination_probe.close()
    termination_detail = (
        f"terminated={termination_audit['terminated']}; truncated={termination_audit['truncated']}; "
        f"steps={termination_audit['steps']}; max_steps={termination_audit['max_steps']}"
    )
    checks = [
        ("observation_space_box", isinstance(env.observation_space, spaces.Box), str(env.observation_space)),
        ("observation_dtype_float32", env.observation_space.dtype == np.float32, str(env.observation_space.dtype)),
        ("observation_finite_after_reset", bool(np.isfinite(obs).all()), f"obs_dim={np.asarray(obs).shape[0]}"),
        ("action_space_discrete_6", isinstance(env.action_space, spaces.Discrete) and int(env.action_space.n) == 6, str(env.action_space)),
        ("reward_finite_for_all_actions", finite_seen, f"reward_min={np.nanmin(rewards):.6f}; reward_max={np.nanmax(rewards):.6f}"),
        ("reward_scaling_reasonable", bool(np.nanmax(np.abs(rewards)) < 10.0), f"max_abs_step_reward={np.nanmax(np.abs(rewards)):.6f}"),
        ("termination_flags_boolean", termination_audit["flags_are_boolean"], termination_detail),
        ("episode_terminal_reachable", termination_audit["terminal_reached"], termination_detail),
        ("bounded_rollout_completion", termination_audit["bounded_completion"], termination_detail),
        ("vecnormalize_handling", True, "VecNormalize(norm_obs=True, norm_reward=False) used for DQN; validation uses saved obs stats only"),
        ("train_validation_separated", True, "training split=train; validation split=validation"),
        ("validation_deterministic_paired", validation_manifest["episode_id"].nunique() == len(validation_manifest), f"validation_episodes={len(validation_manifest)}"),
        ("test_split_not_used", True, "script never constructs split='test'"),
    ]
    for name, passed, detail in checks:
        rows.append(
            {
                "check": name,
                "passed": bool(passed),
                "compatibility_status": "pass" if passed else "fail",
                "detail": detail,
            }
        )
    env.close()
    audit = pd.DataFrame(rows)
    audit.to_csv(TABLES_DIR / "dqn_environment_compatibility_audit.csv", index=False)
    return audit


def training_population_audit(episodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append({"metric": "eligible_training_episodes", "value": int(len(episodes)), "notes": "from training HIGH_RISK_B planning labels"})
    rows.append({"metric": "eligible_training_scenarios", "value": int(episodes["scenario_id"].nunique()), "notes": "|".join(sorted(episodes["scenario_id"].astype(str).unique()))})
    for scenario_id, count in episodes["scenario_id"].value_counts().sort_index().items():
        rows.append({"metric": f"scenario_episode_count__{scenario_id}", "value": int(count), "notes": ""})
    sample_rewards = []
    sample_horizons = []
    sample_actions = []
    for _, row in episodes.head(min(len(episodes), 40)).iterrows():
        env = make_env("train", str(row["scenario_id"]), BOOTSTRAP_SEED)
        reset_env(env, int(row["episode_index"]))
        sample_horizons.append(int(getattr(env, "horizon", 0)))
        for action in ACTION_SPACE:
            probe = copy.deepcopy(env)
            _, reward, _, _, _ = probe.step(action)
            sample_rewards.append(float(reward))
            sample_actions.append(action)
        env.close()
    rows.extend(
        [
            {"metric": "available_actions", "value": "|".join(map(str, sorted(set(sample_actions)))), "notes": "existing six markdown actions"},
            {"metric": "sample_reward_mean", "value": float(np.mean(sample_rewards)), "notes": "first-step sample across actions"},
            {"metric": "sample_reward_std", "value": float(np.std(sample_rewards)), "notes": "first-step sample across actions"},
            {"metric": "sample_reward_min", "value": float(np.min(sample_rewards)), "notes": "first-step sample across actions"},
            {"metric": "sample_reward_max", "value": float(np.max(sample_rewards)), "notes": "first-step sample across actions"},
            {"metric": "sample_horizon_mean", "value": float(np.mean(sample_horizons)), "notes": "sampled eligible episodes"},
        ]
    )
    audit = pd.DataFrame(rows)
    audit.to_csv(TABLES_DIR / "dqn_training_population_audit.csv", index=False)
    return audit


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


def dqn_params(config: DQNConfig) -> dict[str, Any]:
    return {
        "learning_rate": config.learning_rate,
        "buffer_size": 50_000,
        "learning_starts": 1_000,
        "batch_size": 64,
        "tau": 1.0,
        "gamma": 0.99,
        "train_freq": (4, "step"),
        "gradient_steps": 1,
        "target_update_interval": config.target_update_interval,
        "exploration_fraction": 0.25,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "policy_kwargs": {"net_arch": [128, 128], "activation_fn": torch.nn.ReLU},
    }


class DQNCheckpointCallback(BaseCallback):
    def __init__(self, checkpoint_every: int, checkpoint_dir: Path, vecnormalize_path: Path, run_state_path: Path) -> None:
        super().__init__(verbose=0)
        self.checkpoint_every = max(1, int(checkpoint_every))
        self.checkpoint_dir = checkpoint_dir
        self.vecnormalize_path = vecnormalize_path
        self.run_state_path = run_state_path
        self.saved_steps: set[int] = set()

    def _on_step(self) -> bool:
        if self.num_timesteps > 0 and self.num_timesteps % self.checkpoint_every == 0:
            self._save_checkpoint(int(self.num_timesteps))
        return True

    def _save_checkpoint(self, step: int) -> None:
        if step in self.saved_steps:
            return
        self.saved_steps.add(step)
        model_path = self.checkpoint_dir / f"dqn_{step}_steps.zip"
        vec_path = self.checkpoint_dir / f"vecnormalize_{step}_steps.pkl"
        self.model.save(model_path)
        if hasattr(self.training_env, "save"):
            self.training_env.save(str(vec_path))
            self.training_env.save(str(self.vecnormalize_path))
        state = {
            "status": "running",
            "latest_num_timesteps": step,
            "latest_checkpoint": str(model_path.relative_to(PROJECT_ROOT)),
            "latest_vecnormalize": str(vec_path.relative_to(PROJECT_ROOT)),
            "updated_utc": pd.Timestamp.utcnow().isoformat(),
        }
        self.run_state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(f"DQN checkpoint saved: {model_path.relative_to(PROJECT_ROOT)}", flush=True)


def train_one(config: DQNConfig, seed: int, timesteps: int, checkpoint_every: int, n_envs: int, resume: bool) -> dict[str, Any]:
    eligible = load_training_episode_table()
    run_dir = MODELS_DIR / config.config_id / f"seed_{seed}"
    checkpoint_dir = run_dir / "checkpoints"
    monitor_dir = run_dir / "monitor"
    for path in [run_dir, checkpoint_dir, monitor_dir]:
        path.mkdir(parents=True, exist_ok=True)
    vec_path = run_dir / "vecnormalize.pkl"
    run_state_path = run_dir / "run_state.json"
    vec_env = make_vec_env(eligible, seed, n_envs, monitor_dir)
    params = dqn_params(config)
    latest = latest_checkpoint(checkpoint_dir)
    if resume and latest is not None:
        model = DQN.load(latest, env=vec_env, seed=seed, verbose=0)
        completed = int(getattr(model, "num_timesteps", checkpoint_step(latest)))
        remaining = max(int(timesteps) - completed, 0)
        print(f"Resuming {config.config_id} seed={seed} from {latest.name}; completed={completed}, remaining={remaining}", flush=True)
    else:
        model = DQN("MlpPolicy", vec_env, seed=seed, verbose=0, **params)
        completed = 0
        remaining = int(timesteps)
    callback = DQNCheckpointCallback(checkpoint_every, checkpoint_dir, vec_path, run_state_path)
    start = time.perf_counter()
    if remaining > 0:
        model.learn(total_timesteps=remaining, callback=callback, progress_bar=False, reset_num_timesteps=(completed == 0))
    runtime = time.perf_counter() - start
    final_model = run_dir / "final_model.zip"
    model.save(final_model)
    vec_env.save(str(vec_path))
    vec_env.close()
    config_out = {
        "config_id": config.config_id,
        "seed": int(seed),
        "timesteps_requested": int(timesteps),
        "timesteps_completed": int(getattr(model, "num_timesteps", timesteps)),
        "checkpoint_every": int(checkpoint_every),
        "n_envs": int(n_envs),
        "training_split": "train",
        "validation_split": "validation",
        "test_split_used": False,
        "population": POPULATION_ID,
        "calibration_mode": CALIBRATION_MODE,
        "dqn_params": {k: str(v) if k == "policy_kwargs" else v for k, v in params.items()},
        "stability_adjustment": config.stability_adjustment,
        "runtime_seconds": runtime,
    }
    (run_dir / "training_config.json").write_text(json.dumps(config_out, indent=2), encoding="utf-8")
    (run_dir / "run_state.json").write_text(json.dumps({**config_out, "status": "complete"}, indent=2), encoding="utf-8")
    return {
        "config_id": config.config_id,
        "seed": seed,
        "final_model_path": str(final_model),
        "vecnormalize_path": str(vec_path),
        "runtime_seconds": runtime,
    }


def checkpoint_step(path: Path) -> int:
    match = re.search(r"(\d+)_steps", path.stem)
    return int(match.group(1)) if match else 0


def latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    candidates = sorted(checkpoint_dir.glob("dqn_*_steps.zip"), key=lambda p: (checkpoint_step(p), p.stat().st_mtime))
    return candidates[-1] if candidates else None


def checkpoint_records(config_ids: list[str], seeds: list[int]) -> pd.DataFrame:
    rows = []
    for config_id in config_ids:
        for seed in seeds:
            run_dir = MODELS_DIR / config_id / f"seed_{seed}"
            checkpoint_dir = run_dir / "checkpoints"
            for model_path in sorted(checkpoint_dir.glob("dqn_*_steps.zip"), key=checkpoint_step):
                step = checkpoint_step(model_path)
                vec_path = checkpoint_dir / f"vecnormalize_{step}_steps.pkl"
                rows.append(
                    {
                        "config_id": config_id,
                        "seed": seed,
                        "checkpoint_timesteps": step,
                        "model_path": str(model_path),
                        "vecnormalize_path": str(vec_path if vec_path.exists() else run_dir / "vecnormalize.pkl"),
                    }
                )
            final_path = run_dir / "final_model.zip"
            if final_path.exists():
                rows.append(
                    {
                        "config_id": config_id,
                        "seed": seed,
                        "checkpoint_timesteps": "final",
                        "model_path": str(final_path),
                        "vecnormalize_path": str(run_dir / "vecnormalize.pkl"),
                    }
                )
    return pd.DataFrame(rows)


def load_vecnormalize(path: Path, validation_scenario: str, seed: int) -> VecNormalize | None:
    if not path.exists():
        return None
    dummy = DummyVecEnv([lambda: Monitor(make_env("validation", validation_scenario, seed))])
    vec = VecNormalize.load(str(path), dummy)
    vec.training = False
    vec.norm_reward = False
    return vec


# ---------------------------------------------------------------------------
# Validation evaluation
# ---------------------------------------------------------------------------


def evaluate_dqn_episode(record: pd.Series, episode: EvalEpisode) -> tuple[dict[str, Any], pd.DataFrame]:
    model_path = Path(str(record["model_path"]))
    vec_path = Path(str(record["vecnormalize_path"]))
    model = DQN.load(model_path, env=None)
    vec = load_vecnormalize(vec_path, episode.scenario_id, int(record["seed"]) + 777)
    env = make_env("validation", episode.scenario_id, int(record["seed"]) + 9001)
    obs, _ = reset_env(env, episode.episode_index)
    terminated = truncated = False
    actions: list[int] = []
    states: list[dict[str, Any]] = []
    final_info: dict[str, Any] = {}
    start = time.perf_counter()
    step = 0
    while not (terminated or truncated):
        model_obs = obs.astype(np.float32)
        if vec is not None:
            model_obs = vec.normalize_obs(model_obs[None, :])[0]
        action, _ = model.predict(model_obs, deterministic=True)
        action_int = int(np.asarray(action).item())
        states.append(
            {
                "episode_id": episode.episode_id,
                "config_id": record["config_id"],
                "seed": record["seed"],
                "checkpoint_timesteps": record["checkpoint_timesteps"],
                "step_index": step,
                "action": action_int,
                "inventory_coverage_proxy": safe_float(obs[1]) if len(obs) > 1 else np.nan,
                "current_inventory_proxy": safe_float(obs[0]) if len(obs) > 0 else np.nan,
                "episode_day_proxy": step,
            }
        )
        obs, _, terminated, truncated, final_info = env.step(action_int)
        actions.append(action_int)
        step += 1
    runtime = time.perf_counter() - start
    if vec is not None:
        vec.close()
    env.close()
    row = {
        "policy_id": f"{record['config_id']}__seed_{record['seed']}__{record['checkpoint_timesteps']}",
        "model_family": "DQN",
        "config_id": record["config_id"],
        "seed": int(record["seed"]),
        "checkpoint_timesteps": record["checkpoint_timesteps"],
        "episode_id": episode.episode_id,
        "scenario_id": episode.scenario_id,
        "calibration_mode": episode.calibration_mode,
        "episode_index": episode.episode_index,
        "normalized_profit": safe_float(final_info.get("normalized_accounting_profit")),
        "raw_accounting_profit": safe_float(final_info.get("raw_accounting_profit")),
        "revenue": safe_float(final_info.get("cumulative_revenue")),
        "waste_rate": safe_float(final_info.get("final_waste_rate")),
        "sell_through": safe_float(final_info.get("final_sell_through_rate")),
        "physical_waste_units": safe_float(final_info.get("cumulative_waste")),
        "average_markdown": safe_float(final_info.get("average_markdown")),
        "action_sequence": "|".join(str(a) for a in actions),
        "action_entropy": action_entropy(actions),
        "runtime_seconds": runtime,
        "episode_conservation_error": safe_float(final_info.get("episode_conservation_error"), 0.0),
        "raw_financial_sum_error": safe_float(final_info.get("raw_financial_sum_error"), 0.0),
    }
    return row, pd.DataFrame(states)


def load_reference_validation_results() -> pd.DataFrame:
    path = TABLES_DIR / "high_risk_b_distilled_validation_episode_results.csv"
    require_file(path)
    refs = pd.read_csv(path)
    refs = refs.loc[refs["policy_id"].isin([BASELINE_POLICY, PLANNING_POLICY, DISTILLED_POLICY])].copy()
    ppo_path = TABLES_DIR / "planning_ppo_baseline_comparison.csv"
    if ppo_path.exists():
        ppo_summary = pd.read_csv(ppo_path)
        ppo_rows = ppo_summary.loc[ppo_summary["policy_id"].astype(str).eq(SELECTED_PPO_POLICY)]
        if not ppo_rows.empty:
            row = ppo_rows.iloc[0]
            refs = pd.concat(
                [
                    refs,
                    pd.DataFrame(
                        [
                            {
                                "policy_id": SELECTED_PPO_POLICY,
                                "episode_id": "__summary_only__",
                                "normalized_profit": safe_float(row.get("mean_normalized_profit")),
                                "waste_rate": safe_float(row.get("waste_rate")),
                                "sell_through": np.nan,
                                "average_markdown": safe_float(row.get("average_markdown")),
                                "action_entropy": np.nan,
                                "runtime_seconds": np.nan,
                                "episode_conservation_error": 0.0,
                                "raw_financial_sum_error": 0.0,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
    return refs


def compare_policy(policy: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, Any]:
    merged = policy.merge(baseline, on="episode_id", suffixes=("_policy", "_baseline"))
    if merged.empty:
        return {
            "paired_episode_count": 0,
            "paired_mean_gain_vs_always_0pct": np.nan,
            "paired_median_gain": np.nan,
            "bootstrap_ci_low": np.nan,
            "bootstrap_ci_high": np.nan,
            "win_share": np.nan,
            "tie_share": np.nan,
            "loss_share": np.nan,
            "waste_rate_difference": np.nan,
            "sell_through_difference": np.nan,
            "worst_decile_paired_gain": np.nan,
            "maximum_episode_loss": np.nan,
        }
    diff = merged["normalized_profit_policy"] - merged["normalized_profit_baseline"]
    ci_low, ci_high = bootstrap_ci(diff)
    worst_decile = float(diff.sort_values().head(max(1, math.ceil(len(diff) * 0.10))).mean())
    return {
        "paired_episode_count": int(len(merged)),
        "paired_mean_gain_vs_always_0pct": float(diff.mean()),
        "paired_median_gain": float(diff.median()),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "win_share": float((diff > TIE_TOL).mean()),
        "tie_share": float((diff.abs() <= TIE_TOL).mean()),
        "loss_share": float((diff < -TIE_TOL).mean()),
        "waste_rate_difference": float((merged["waste_rate_policy"] - merged["waste_rate_baseline"]).mean()),
        "sell_through_difference": float((merged["sell_through_policy"] - merged["sell_through_baseline"]).mean()),
        "worst_decile_paired_gain": worst_decile,
        "maximum_episode_loss": float(diff.min()),
    }


def summarize_results(results: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    baseline = results.loc[results["policy_id"].eq(BASELINE_POLICY)]
    rows = []
    for policy_id, policy in results.groupby("policy_id", dropna=False):
        comp = compare_policy(policy, baseline)
        action_values = []
        for seq in policy.get("action_sequence", pd.Series(dtype=str)).dropna():
            action_values.extend([int(x) for x in str(seq).split("|") if x != ""])
        dominant_share = pd.Series(action_values).value_counts(normalize=True).max() if action_values else 0.0
        zero_share = float((pd.Series(action_values) == 0).mean()) if action_values else np.nan
        pos_share = float((pd.Series(action_values) > 0).mean()) if action_values else np.nan
        state_dependence = "STATE_DEPENDENT" if action_entropy(action_values) > 0.05 and dominant_share < 0.98 else "FIXED_OR_COLLAPSED"
        rows.append(
            {
                "policy_id": policy_id,
                "model_family": policy["model_family"].iloc[0] if "model_family" in policy.columns else "reference",
                "config_id": policy["config_id"].iloc[0] if "config_id" in policy.columns else policy_id,
                "seed": policy["seed"].iloc[0] if "seed" in policy.columns else np.nan,
                "checkpoint_timesteps": policy["checkpoint_timesteps"].iloc[0] if "checkpoint_timesteps" in policy.columns else np.nan,
                "mean_normalized_profit": float(policy["normalized_profit"].mean()),
                "raw_accounting_profit": float(policy["raw_accounting_profit"].mean()) if "raw_accounting_profit" in policy.columns else np.nan,
                "revenue": float(policy["revenue"].mean()) if "revenue" in policy.columns else np.nan,
                "waste_rate": float(policy["waste_rate"].mean()),
                "sell_through": float(policy["sell_through"].mean()) if "sell_through" in policy.columns else np.nan,
                "average_markdown": float(policy["average_markdown"].mean()) if "average_markdown" in policy.columns else np.nan,
                "action_entropy": action_entropy(action_values),
                "zero_action_share": zero_share,
                "positive_action_share": pos_share,
                "dominant_action_share": float(dominant_share),
                "state_dependence": state_dependence,
                "collapse_flag": bool(dominant_share >= 0.98),
                "accounting_valid": bool((policy["episode_conservation_error"].abs().max() <= 1e-6) and (policy["raw_financial_sum_error"].abs().max() <= 1e-6)),
                "runtime_seconds": float(policy["runtime_seconds"].sum()) if "runtime_seconds" in policy.columns else np.nan,
                **comp,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLES_DIR / "dqn_checkpoint_validation_summary.csv", index=False)
    return summary


def evaluate_checkpoints(config_ids: list[str], seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest_df = load_validation_manifest()
    episodes = [
        EvalEpisode(str(row["scenario_id"]), str(row["calibration_mode"]), int(row["episode_index"]), str(row["episode_id"]))
        for _, row in manifest_df.iterrows()
    ]
    records = checkpoint_records(config_ids, seeds)
    if records.empty:
        raise FileNotFoundError("No DQN checkpoints found. Run with --mode train first.")
    rows = []
    state_rows = []
    for ridx, record in records.iterrows():
        print(
            f"Evaluating checkpoint {ridx + 1}/{len(records)}: "
            f"{record['config_id']} seed={record['seed']} step={record['checkpoint_timesteps']}",
            flush=True,
        )
        for eidx, episode in enumerate(episodes, start=1):
            print(f"  episode {eidx}/{len(episodes)}: {episode.episode_id}", flush=True)
            row, states = evaluate_dqn_episode(record, episode)
            rows.append(row)
            state_rows.append(states)
    dqn_results = pd.DataFrame(rows)
    dqn_states = pd.concat(state_rows, ignore_index=True) if state_rows else pd.DataFrame()
    refs = load_reference_validation_results()
    refs["model_family"] = refs["policy_id"].map(
        {
            BASELINE_POLICY: "baseline",
            PLANNING_POLICY: "planning",
            DISTILLED_POLICY: "distillation",
            SELECTED_PPO_POLICY: "ppo_reference",
        }
    ).fillna("reference")
    combined = pd.concat([refs, dqn_results], ignore_index=True, sort=False)
    combined.to_csv(TABLES_DIR / "dqn_checkpoint_validation_episode_results.csv", index=False)
    dqn_states.to_csv(TABLES_DIR / "dqn_checkpoint_state_actions.csv", index=False)
    summary = summarize_results(combined, dqn_states)
    return combined, dqn_states, summary


def seed_stability(summary: pd.DataFrame) -> pd.DataFrame:
    dqn = summary.loc[summary["model_family"].eq("DQN")].copy()
    if dqn.empty:
        stability = pd.DataFrame()
    else:
        best_by_seed = (
            dqn.sort_values(
                ["config_id", "seed", "paired_mean_gain_vs_always_0pct", "win_share", "collapse_flag"],
                ascending=[True, True, False, False, True],
            )
            .groupby(["config_id", "seed"], as_index=False)
            .head(1)
        )
        rows = []
        for config_id, group in best_by_seed.groupby("config_id"):
            rows.append(
                {
                    "config_id": config_id,
                    "seed_count": int(group["seed"].nunique()),
                    "positive_gain_seed_count": int((group["paired_mean_gain_vs_always_0pct"] > 0).sum()),
                    "positive_gain_seed_share": float((group["paired_mean_gain_vs_always_0pct"] > 0).mean()),
                    "mean_gain_across_seed_best": float(group["paired_mean_gain_vs_always_0pct"].mean()),
                    "gain_std_across_seed_best": float(group["paired_mean_gain_vs_always_0pct"].std(ddof=1)) if len(group) > 1 else 0.0,
                    "mean_win_rate_across_seed_best": float(group["win_share"].mean()),
                    "mean_waste_difference_across_seed_best": float(group["waste_rate_difference"].mean()),
                    "selected_checkpoints_by_seed": "|".join(f"{int(r.seed)}:{r.checkpoint_timesteps}" for r in group.itertuples()),
                    "direction_consistency": "PASS" if int((group["paired_mean_gain_vs_always_0pct"] > 0).sum()) >= min(2, len(group)) else "FAIL",
                }
            )
        stability = pd.DataFrame(rows)
    stability.to_csv(TABLES_DIR / "dqn_seed_stability.csv", index=False)
    return stability


def select_candidate(summary: pd.DataFrame, stability: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    dqn = summary.loc[summary["model_family"].eq("DQN")].copy()
    if dqn.empty:
        candidate = pd.DataFrame()
        status = "DQN_EVALUATION_REQUIRES_REVISION"
    else:
        stable_configs = set(stability.loc[stability["direction_consistency"].eq("PASS"), "config_id"]) if not stability.empty else set()
        dqn["seed_stability_pass"] = dqn["config_id"].isin(stable_configs)
        dqn["not_driven_by_few_episodes"] = dqn["worst_decile_paired_gain"] > -0.08
        dqn["eligible_for_final_lock"] = (
            (dqn["paired_mean_gain_vs_always_0pct"] > 0)
            & (dqn["win_share"] >= 0.60)
            & (dqn["waste_rate_difference"] <= 0.02)
            & dqn["accounting_valid"].astype(bool)
            & dqn["state_dependence"].eq("STATE_DEPENDENT")
            & (~dqn["collapse_flag"].astype(bool))
            & dqn["seed_stability_pass"]
            & dqn["not_driven_by_few_episodes"]
        )
        candidate = dqn.sort_values(
            ["eligible_for_final_lock", "paired_mean_gain_vs_always_0pct", "win_share", "waste_rate_difference"],
            ascending=[False, False, False, True],
        ).head(1)
        status = "DQN_BEATS_BASELINE_ON_VALIDATION" if bool(candidate["eligible_for_final_lock"].iloc[0]) else "PLANNING_VALUE_EXISTS_BUT_DQN_FAILED"
    comparison = dqn if not dqn.empty else pd.DataFrame()
    comparison.to_csv(TABLES_DIR / "dqn_final_candidate_comparison.csv", index=False)
    config = {
        "status": status,
        "final_model_locked": bool(status == "DQN_BEATS_BASELINE_ON_VALIDATION"),
        "test_split_used": False,
        "population": POPULATION_ID,
        "calibration_mode": CALIBRATION_MODE,
        "locked_baseline": BASELINE_POLICY,
        "selected_candidate": candidate.iloc[0].to_dict() if not candidate.empty else None,
    }
    (CONFIGS_DIR / "dqn_locked_candidate.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    hash_paths = [
        TABLES_DIR / "dqn_environment_compatibility_audit.csv",
        TABLES_DIR / "dqn_training_population_audit.csv",
        TABLES_DIR / "dqn_checkpoint_validation_episode_results.csv",
        TABLES_DIR / "dqn_checkpoint_validation_summary.csv",
        TABLES_DIR / "dqn_final_candidate_comparison.csv",
        TABLES_DIR / "dqn_seed_stability.csv",
        CONFIGS_DIR / "dqn_locked_candidate.json",
    ]
    hashes = [{"path": str(path.relative_to(PROJECT_ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size if path.exists() else None} for path in hash_paths]
    (CONFIGS_DIR / "dqn_locked_candidate_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    return candidate, status


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------


def make_figures(summary: pd.DataFrame, results: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    dqn = summary.loc[summary["model_family"].eq("DQN")].copy()
    if dqn.empty:
        return
    dqn["step_numeric"] = pd.to_numeric(dqn["checkpoint_timesteps"].replace("final", np.nan), errors="coerce")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for (config_id, seed), group in dqn.groupby(["config_id", "seed"]):
        group = group.sort_values("step_numeric")
        ax.plot(group["step_numeric"], group["mean_normalized_profit"], marker="o", label=f"{config_id} seed {int(seed)}")
    ax.set_title("DQN validation profit by timestep")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean normalized profit")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dqn_validation_profit_by_timestep.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for (config_id, seed), group in dqn.groupby(["config_id", "seed"]):
        group = group.sort_values("step_numeric")
        ax.plot(group["step_numeric"], group["paired_mean_gain_vs_always_0pct"], marker="o", label=f"{config_id} seed {int(seed)}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("DQN paired gain versus always_0pct")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Paired gain")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dqn_paired_gain_vs_baseline.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    dqn.groupby(["config_id", "seed"])["paired_mean_gain_vs_always_0pct"].max().unstack("config_id").plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Seed stability")
    ax.set_ylabel("Best paired gain")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dqn_seed_stability.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ref = summary.loc[summary["policy_id"].isin([BASELINE_POLICY, PLANNING_POLICY, DISTILLED_POLICY, SELECTED_PPO_POLICY])]
    plot_df = pd.concat([ref, dqn.sort_values("paired_mean_gain_vs_always_0pct", ascending=False).head(6)])
    plot_df.set_index("policy_id")["mean_normalized_profit"].plot(kind="bar", ax=ax)
    ax.set_title("DQN versus PPO versus planning")
    ax.set_ylabel("Mean normalized profit")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dqn_vs_ppo_vs_planning.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(summary["waste_rate"], summary["mean_normalized_profit"])
    for _, row in summary.iterrows():
        if row["policy_id"] in [BASELINE_POLICY, PLANNING_POLICY, DISTILLED_POLICY] or row.get("model_family") == "DQN":
            ax.annotate(str(row["policy_id"])[:24], (row["waste_rate"], row["mean_normalized_profit"]), fontsize=6)
    ax.set_xlabel("Waste rate")
    ax.set_ylabel("Mean normalized profit")
    ax.set_title("Profit-waste comparison")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "profit_waste_comparison.png", dpi=160)
    plt.close(fig)

    best_policy = dqn.sort_values("paired_mean_gain_vs_always_0pct", ascending=False)["policy_id"].iloc[0]
    best_results = results.loc[results["policy_id"].eq(best_policy)]
    actions = []
    for seq in best_results["action_sequence"].dropna():
        actions.extend([int(x) for x in str(seq).split("|") if x])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pd.Series(actions).value_counts(normalize=True).sort_index().plot(kind="bar", ax=ax)
    ax.set_title("Action distribution")
    ax.set_xlabel("Action")
    ax.set_ylabel("Share")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "action_distribution.png", dpi=160)
    plt.close(fig)

    baseline = results.loc[results["policy_id"].eq(BASELINE_POLICY)]
    merged = best_results.merge(baseline, on="episode_id", suffixes=("_dqn", "_baseline"))
    if not merged.empty:
        diff = merged["normalized_profit_dqn"] - merged["normalized_profit_baseline"]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        diff.sort_values().plot(kind="bar", ax=ax)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Worst-case episode differences")
        ax.set_ylabel("DQN minus always_0pct")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "worst_case_episode_differences.png", dpi=160)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final validation-locked DQN experiment for HIGH_RISK_B.")
    parser.add_argument("--mode", choices=["audit", "train", "evaluate", "all"], default="all")
    parser.add_argument("--timesteps", type=int, default=30_000)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--configs", nargs="+", choices=[c.config_id for c in DQN_CONFIGS], default=[c.config_id for c in DQN_CONFIGS])
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    ensure_dirs()
    audit = audit_environment_compatibility()
    episodes = load_training_episode_table()
    training_population_audit(episodes)
    if audit["compatibility_status"].eq("fail").any():
        print("DQN_EVALUATION_REQUIRES_REVISION")
        return
    config_map = {c.config_id: c for c in DQN_CONFIGS}
    trained_records: list[dict[str, Any]] = []
    if args.mode in {"train", "all"}:
        for config_id in args.configs:
            for seed in args.seeds:
                print(f"Training {config_id} seed={seed} timesteps={args.timesteps}", flush=True)
                trained_records.append(train_one(config_map[config_id], int(seed), int(args.timesteps), int(args.checkpoint_every), int(args.n_envs), bool(args.resume)))
    if args.mode in {"evaluate", "all"}:
        results, states, summary = evaluate_checkpoints(args.configs, [int(s) for s in args.seeds])
        stability = seed_stability(summary)
        candidate, status = select_candidate(summary, stability)
        make_figures(summary, results)
    else:
        summary = pd.DataFrame()
        stability = pd.DataFrame()
        candidate = pd.DataFrame()
        status = "DQN_AUDIT_OR_TRAINING_COMPLETE"
    runtime = time.perf_counter() - start

    if status not in {"DQN_BEATS_BASELINE_ON_VALIDATION", "PLANNING_VALUE_EXISTS_BUT_DQN_FAILED", "DQN_EVALUATION_REQUIRES_REVISION"}:
        status = "DQN_EVALUATION_REQUIRES_REVISION" if args.mode == "audit" else status
    print(status)
    print(f"configurations_requested={'|'.join(args.configs)}")
    print(f"seeds_requested={'|'.join(str(s) for s in args.seeds)}")
    print(f"eligible_training_episodes={len(episodes)}")
    print(f"validation_episodes={len(load_validation_manifest())}")
    print("test_split_used=False")
    if not candidate.empty:
        row = candidate.iloc[0]
        print(f"best_checkpoint={row['policy_id']}")
        print(f"paired_validation_gain={safe_float(row['paired_mean_gain_vs_always_0pct']):.6f}")
        print(f"ci95=[{safe_float(row['bootstrap_ci_low']):.6f}, {safe_float(row['bootstrap_ci_high']):.6f}]")
        print(f"win_rate={safe_float(row['win_share']):.3f}")
        print(f"waste_difference={safe_float(row['waste_rate_difference']):.6f}")
        print(f"candidate_lock_status={bool(row.get('eligible_for_final_lock', False))}")
    print(f"runtime_seconds={runtime:.1f}")
    print("files_created:")
    for file in [
        "outputs/tables/dqn_environment_compatibility_audit.csv",
        "outputs/tables/dqn_training_population_audit.csv",
        "outputs/tables/dqn_checkpoint_validation_episode_results.csv",
        "outputs/tables/dqn_checkpoint_validation_summary.csv",
        "outputs/tables/dqn_final_candidate_comparison.csv",
        "outputs/tables/dqn_seed_stability.csv",
        "outputs/configs/dqn_locked_candidate.json",
        "outputs/configs/dqn_locked_candidate_hashes.json",
        "outputs/figures/dqn_final_candidate/",
    ]:
        print(f"- {file}")


if __name__ == "__main__":
    main()
