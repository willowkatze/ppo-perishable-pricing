"""Scenario-balanced PPO redesign training for recovered financial calibration.

This module keeps the environment, reward, action space, and validation
manifest fixed while changing only training-scenario sampling and one stable
hyperparameter variant. The goal is to test decision precision, not to force a
policy to beat the no-markdown baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pricing_env_operational import ACTION_MARKDOWNS, OperationalPerishablePricingEnv  # noqa: E402

TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
CONFIGS_DIR = PROJECT_ROOT / "outputs" / "configs"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models" / "ppo_training_redesign"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "ppo_training_redesign"

SEED = 123
DEFAULT_SCENARIO_ID = "core_008"
CHECKPOINT_EVERY = 5_000
DEFAULT_TIMESTEPS = 20_000
N_ENVS = 4
ORIGINAL_REGRET = 0.0257
ACTION_COLS = [f"action_{idx}_share" for idx in range(6)]
ACTION_LABELS = {f"action_{idx}_share": f"{markdown:.0%}" for idx, markdown in ACTION_MARKDOWNS.items()}
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
VARIANTS = {
    "balanced": {
        "variant": "BALANCED",
        "sampling_mode": "scenario_balanced",
        "ppo_overrides": {},
    },
    "balanced_stable": {
        "variant": "BALANCED_STABLE",
        "sampling_mode": "scenario_balanced",
        "ppo_overrides": {"learning_rate": 1e-4, "ent_coef": 0.02},
    },
}
BASELINES = ["always_0pct", "expiry_threshold_rule", "inventory_coverage_rule"]


@dataclass(frozen=True)
class VariantSpec:
    variant: str
    sampling_mode: str
    ppo_overrides: dict[str, Any]


def ensure_dirs() -> None:
    for path in [TABLES_DIR, CONFIGS_DIR, MODELS_DIR, FIGURES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def safe_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric if np.isfinite(numeric) else 0.0


def action_entropy(action_shares: dict[int, float]) -> float:
    entropy = 0.0
    for share in action_shares.values():
        if share > 0:
            entropy -= share * math.log(share)
    return float(entropy)


def load_meaningful_scenarios() -> pd.DataFrame:
    path = TABLES_DIR / "sustainability_scenario_audit.csv"
    audit = pd.read_csv(path)
    return audit.loc[audit["classification"].eq("MEANINGFUL_PROFIT_WASTE_TRADEOFF")].copy()


def scenario_sampling_weights(scenarios: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in scenarios.iterrows():
        weight = 1.0
        if "high_inventory" in str(row.get("inventory_coverage", "")):
            weight *= 2.0
        if "short_shelf_life" in str(row.get("shelf_life_class", "")):
            weight *= 1.5
        if safe_float(row.get("always_zero_waste_rate", 0.0)) > 0.05:
            weight *= 1.5
        rows.append(
            {
                "scenario_id": row["scenario_id"],
                "shelf_life_class": row.get("shelf_life_class"),
                "inventory_coverage": row.get("inventory_coverage"),
                "demand_target": row.get("demand_target"),
                "always_zero_waste_rate": safe_float(row.get("always_zero_waste_rate", 0.0)),
                "raw_sampling_weight": weight,
            }
        )
    weights = pd.DataFrame(rows)
    weights["normalized_sampling_weight"] = weights["raw_sampling_weight"] / weights["raw_sampling_weight"].sum()
    weights.to_csv(TABLES_DIR / "ppo_redesign_sampling_weights.csv", index=False)
    return weights


class ScenarioBalancedEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(self, sampling_weights: pd.DataFrame, seed: int) -> None:
        super().__init__()
        self.weights = sampling_weights.reset_index(drop=True)
        self.rng = np.random.default_rng(seed)
        self.envs: dict[str, OperationalPerishablePricingEnv] = {}
        for scenario_id in self.weights["scenario_id"]:
            self.envs[str(scenario_id)] = OperationalPerishablePricingEnv(
                split="train",
                calibration_mode="recovered_calibration",
                reward_mode="financial",
                lambda_waste=0.10,
                scenario_id=str(scenario_id),
                deterministic_demand=True,
                random_seed=seed,
                promotion_context_mode="derived_from_action",
            )
        first_env = next(iter(self.envs.values()))
        self.action_space = first_env.action_space
        self.observation_space = first_env.observation_space
        self.current_env = first_env
        self.current_scenario_id = first_env.scenario_id

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        probs = self.weights["normalized_sampling_weight"].to_numpy(dtype=float)
        idx = int(self.rng.choice(len(self.weights), p=probs))
        self.current_scenario_id = str(self.weights.iloc[idx]["scenario_id"])
        self.current_env = self.envs[self.current_scenario_id]
        episode_index = int(self.rng.integers(0, 10_000))
        return self.current_env.reset(seed=seed, options={"episode_index": episode_index})

    def step(self, action: int):
        return self.current_env.step(action)

    def render(self):
        return self.current_env.render()


def make_balanced_vec_env(weights: pd.DataFrame, seed: int, monitor_dir: Path) -> VecNormalize:
    def make_one(offset: int):
        def _factory():
            env = ScenarioBalancedEnv(weights, seed + offset * 1009)
            return Monitor(env, filename=str(monitor_dir / f"monitor_{seed + offset * 1009}.csv"))

        return _factory

    return VecNormalize(
        DummyVecEnv([make_one(i) for i in range(N_ENVS)]),
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )


def policy_kwargs() -> dict[str, Any]:
    return {"net_arch": {"pi": [128, 128], "vf": [128, 128]}, "activation_fn": torch.nn.Tanh}


class VecNormalizeCheckpointCallback(CheckpointCallback):
    def __init__(self, *args: Any, vec_path: Path, run_state_path: Path, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.vec_path = vec_path
        self.run_state_path = run_state_path

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.n_calls % max(self.save_freq, 1) == 0:
            if hasattr(self.training_env, "save"):
                self.training_env.save(str(self.vec_path))
            self.run_state_path.write_text(
                json.dumps(
                    {
                        "status": "running",
                        "latest_num_timesteps": int(self.num_timesteps),
                        "last_checkpoint_update": pd.Timestamp.utcnow().isoformat(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        return result


def action_for_baseline(policy_id: str, obs: np.ndarray) -> int:
    if policy_id == "always_0pct":
        return 0
    if policy_id == "expiry_threshold_rule":
        expiring_soon = safe_float(obs[24]) if obs.shape[0] > 24 else 0.0
        if expiring_soon >= 0.50:
            return 5
        if expiring_soon >= 0.25:
            return 3
        if expiring_soon >= 0.10:
            return 2
        return 0
    if policy_id == "inventory_coverage_rule":
        coverage = safe_float(obs[1]) if obs.shape[0] > 1 else 0.0
        if coverage >= 1.50:
            return 4
        if coverage >= 1.10:
            return 3
        if coverage >= 0.80:
            return 2
        return 0
    raise ValueError(policy_id)


def evaluate_policy_on_manifest(
    *,
    model: PPO | None,
    vecnorm: VecNormalize | None,
    policy_id: str,
    manifest: pd.DataFrame,
    scenario_id: str = DEFAULT_SCENARIO_ID,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    env = OperationalPerishablePricingEnv(
        split="validation",
        calibration_mode="recovered_calibration",
        reward_mode="financial",
        lambda_waste=0.10,
        scenario_id=scenario_id,
        deterministic_demand=True,
        random_seed=SEED,
        promotion_context_mode="derived_from_action",
    )
    for _, episode in manifest.iterrows():
        options = {
            "store_id": episode["store_id"],
            "product_id": episode["product_id"],
            "start_date": episode["start_date"],
        }
        obs, _ = env.reset(seed=int(episode.get("age_profile_seed", SEED)), options=options)
        terminated = False
        truncated = False
        total_return = 0.0
        final_info: dict[str, Any] = {}
        counts = {i: 0 for i in range(6)}
        while not (terminated or truncated):
            if model is None:
                action = action_for_baseline(policy_id, obs)
            else:
                model_obs = obs.astype(np.float32)
                if vecnorm is not None:
                    model_obs = vecnorm.normalize_obs(model_obs[None, :])[0]
                predicted, _ = model.predict(model_obs, deterministic=True)
                action = int(np.asarray(predicted).item())
            obs, reward, terminated, truncated, info = env.step(action)
            counts[action] += 1
            total_return += float(reward)
            final_info = info
        total = max(sum(counts.values()), 1)
        shares = {a: counts[a] / total for a in counts}
        rows.append(
            {
                "policy_id": policy_id,
                "episode_id": episode["episode_id"],
                "episode_return": total_return,
                "normalized_accounting_profit": safe_float(final_info.get("normalized_accounting_profit")),
                "raw_accounting_profit": safe_float(final_info.get("accounting_profit")),
                "waste_rate": safe_float(final_info.get("final_waste_rate")),
                "sell_through_rate": safe_float(final_info.get("final_sell_through_rate")),
                "episode_conservation_error": safe_float(final_info.get("episode_conservation_error")),
                "raw_financial_sum_error": safe_float(final_info.get("raw_financial_sum_error")),
                "action_entropy": action_entropy(shares),
                **{f"action_{a}_share": shares[a] for a in range(6)},
            }
        )
    return pd.DataFrame(rows)


def load_manifest() -> pd.DataFrame:
    path = CONFIGS_DIR / "ppo_validation_episode_manifest.csv"
    if path.exists():
        return pd.read_csv(path)
    rows = [{"episode_id": f"validation_{i:04d}", "episode_index": i, "age_profile_seed": 10_000 + i} for i in range(20)]
    return pd.DataFrame(rows)


def state_dependence_classification(zero_share: float, entropy: float) -> str:
    if zero_share >= 0.98 and entropy <= 0.05:
        return "ZERO_ACTION_COLLAPSE"
    if zero_share >= 0.80:
        return "MOSTLY_ZERO_BUT_STATE_DEPENDENT" if entropy > 0.05 else "ZERO_ACTION_COLLAPSE"
    return "STATE_DEPENDENT_POLICY" if entropy > 0.05 else "UNSTABLE_POLICY"


def evaluate_oracle_regret(model: PPO, vecnorm: VecNormalize, policy_id: str) -> pd.DataFrame:
    states = pd.read_csv(TABLES_DIR / "financial_markdown_oracle_diagnostic.csv")
    action_values = pd.read_csv(TABLES_DIR / "financial_markdown_multistep_action_values.csv")
    values = action_values.loc[action_values["continuation_policy"].eq("greedy_financial")].set_index(["state_id", "first_action"])
    rows: list[dict[str, Any]] = []
    for _, state in states.iterrows():
        env = OperationalPerishablePricingEnv(
            split="validation",
            calibration_mode=state["calibration_mode"],
            reward_mode="financial",
            lambda_waste=0.10,
            scenario_id=state["scenario_id"],
            deterministic_demand=True,
            random_seed=SEED,
            promotion_context_mode="derived_from_action",
        )
        obs, _ = env.reset(seed=20_000 + int(state["episode_index"]), options={"episode_index": int(state["episode_index"])})
        prefix = [] if pd.isna(state.get("prefix_actions")) else [int(x) for x in str(state["prefix_actions"]).split("|") if x]
        for action in prefix:
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        model_obs = vecnorm.normalize_obs(obs.astype(np.float32)[None, :])[0]
        predicted, _ = model.predict(model_obs, deterministic=True)
        action = int(np.asarray(predicted).item())
        oracle_action = int(state["rolling_oracle_preferred_action"])
        oracle_value = safe_float(state["rolling_oracle_value"])
        action_value = safe_float(values.loc[(state["state_id"], action), "total_normalized_accounting_profit"]) if (state["state_id"], action) in values.index else safe_float(state["zero_first_action_oracle_value"])
        rows.append(
            {
                "policy_id": policy_id,
                "state_id": state["state_id"],
                "ppo_action": action,
                "oracle_action": oracle_action,
                "oracle_regret": oracle_value - action_value,
                "action_agreement": action == oracle_action,
                "unnecessary_markdown": action != 0 and oracle_action == 0,
                "missed_markdown_opportunity": action == 0 and oracle_action != 0,
                "correct_zero": action == 0 and oracle_action == 0,
                "correct_positive": action != 0 and action == oracle_action,
                "wrong_positive_level": action != 0 and oracle_action != 0 and action != oracle_action,
                "high_risk_state": bool(state["fraction_expiring_within_two_days"] >= 0.5 or state["inventory_coverage_state"] >= 1.5),
            }
        )
    return pd.DataFrame(rows)


def evaluate_checkpoint(model_path: Path, vec_path: Path, variant: str, timestep: int, manifest: pd.DataFrame) -> dict[str, Any]:
    raw_env = DummyVecEnv(
        [
            lambda: OperationalPerishablePricingEnv(
                split="validation",
                calibration_mode="recovered_calibration",
                reward_mode="financial",
                lambda_waste=0.10,
                scenario_id=DEFAULT_SCENARIO_ID,
                deterministic_demand=True,
                random_seed=SEED,
                promotion_context_mode="derived_from_action",
            )
        ]
    )
    vecnorm = VecNormalize.load(str(vec_path), raw_env)
    vecnorm.training = False
    vecnorm.norm_reward = False
    model = PPO.load(model_path)
    policy_id = f"{variant}__checkpoint_{timestep}"
    eval_df = evaluate_policy_on_manifest(model=model, vecnorm=vecnorm, policy_id=policy_id, manifest=manifest)
    baseline = evaluate_policy_on_manifest(model=None, vecnorm=None, policy_id="always_0pct", manifest=manifest)
    merged = eval_df.merge(baseline[["episode_id", "normalized_accounting_profit"]], on="episode_id", suffixes=("", "_always_zero"))
    oracle = evaluate_oracle_regret(model, vecnorm, policy_id)
    zero_share = float(eval_df["action_0_share"].mean())
    entropy = float(eval_df["action_entropy"].mean())
    result = {
        "variant": variant,
        "checkpoint_timestep": timestep,
        "policy_id": policy_id,
        "mean_validation_return": float(eval_df["episode_return"].mean()),
        "mean_normalized_validation_profit": float(eval_df["normalized_accounting_profit"].mean()),
        "paired_profit_difference_vs_always_zero": float((merged["normalized_accounting_profit"] - merged["normalized_accounting_profit_always_zero"]).mean()),
        "oracle_regret": float(oracle["oracle_regret"].mean()),
        "oracle_agreement": float(oracle["action_agreement"].mean()),
        "unnecessary_markdown_rate": float(oracle["unnecessary_markdown"].mean()),
        "missed_markdown_opportunity_rate": float(oracle["missed_markdown_opportunity"].mean()),
        "correct_zero_rate": float(oracle["correct_zero"].mean()),
        "correct_positive_rate": float(oracle["correct_positive"].mean()),
        "wrong_positive_level_rate": float(oracle["wrong_positive_level"].mean()),
        "zero_action_share": zero_share,
        "positive_action_share": 1.0 - zero_share,
        "empirical_action_entropy": entropy,
        "state_dependence_classification": state_dependence_classification(zero_share, entropy),
        "collapse_flag": bool(zero_share >= 0.98 and entropy <= 0.05),
        "mechanical_valid": bool(eval_df["episode_conservation_error"].abs().max() <= 1e-6 and eval_df["raw_financial_sum_error"].abs().max() <= 1e-6),
        **{col: float(eval_df[col].mean()) for col in ACTION_COLS},
    }
    vecnorm.close()
    return result


def training_design_audit(weights: pd.DataFrame) -> None:
    diagnostics = pd.read_csv(TABLES_DIR / "ppo_training_diagnostics.csv") if (TABLES_DIR / "ppo_training_diagnostics.csv").exists() else pd.DataFrame()
    oracle = pd.read_csv(TABLES_DIR / "financial_markdown_oracle_diagnostic.csv")
    rows = [
        {
            "item": "original_training_scenario",
            "value": DEFAULT_SCENARIO_ID,
            "notes": "Existing PPO training used one default scenario in train_ppo_operational.py.",
        },
        {
            "item": "redesign_training_scenarios",
            "value": "|".join(weights["scenario_id"].astype(str)),
            "notes": "Scenario-balanced sampling uses existing meaningful profit-waste scenarios only.",
        },
        {"item": "learning_rate_original", "value": PPO_BASE_PARAMS["learning_rate"], "notes": ""},
        {"item": "entropy_coefficient_original", "value": PPO_BASE_PARAMS["ent_coef"], "notes": ""},
        {"item": "rollout_length", "value": PPO_BASE_PARAMS["n_steps"], "notes": ""},
        {"item": "batch_size", "value": PPO_BASE_PARAMS["batch_size"], "notes": ""},
        {"item": "n_parallel_envs", "value": N_ENVS, "notes": ""},
        {"item": "checkpoint_frequency", "value": CHECKPOINT_EVERY, "notes": ""},
        {
            "item": "oracle_positive_markdown_share",
            "value": float(oracle["positive_markdown_financially_preferred"].mean()),
            "notes": "Diagnostic only; not used as supervised label.",
        },
        {
            "item": "early_stopping_logic",
            "value": "documented",
            "notes": "Stop if collapse and no regret/return improvement persist for two evaluation points.",
        },
    ]
    if not diagnostics.empty:
        recovered = diagnostics.loc[diagnostics["agent_id"].eq("recovered_financial")]
        if not recovered.empty:
            rows.append({"item": "previous_recovered_runtime_seconds", "value": float(recovered["runtime_seconds"].iloc[-1]), "notes": ""})
    pd.DataFrame(rows).to_csv(TABLES_DIR / "ppo_training_design_audit.csv", index=False)


def state_balance_audit(weights: pd.DataFrame) -> None:
    oracle = pd.read_csv(TABLES_DIR / "financial_markdown_oracle_diagnostic.csv")
    high_risk_oracle = oracle["positive_markdown_financially_preferred"].mean()
    weighted_high_waste = float(
        (weights["normalized_sampling_weight"] * (weights["always_zero_waste_rate"] > 0.05).astype(float)).sum()
    )
    original_exposure = 0.0
    classification = "STRONGLY_IMBALANCED" if high_risk_oracle >= 0.10 and original_exposure < high_risk_oracle / 2 else "MODERATELY_IMBALANCED"
    rows = [
        {
            "distribution": "original_training",
            "scenario_count": 1,
            "high_risk_share_estimate": original_exposure,
            "oracle_positive_markdown_share": high_risk_oracle,
            "classification": classification,
            "notes": "Original recovered PPO training used default scenario core_008, outside the six meaningful trade-off scenarios.",
        },
        {
            "distribution": "scenario_balanced_training",
            "scenario_count": int(weights["scenario_id"].nunique()),
            "high_risk_share_estimate": weighted_high_waste,
            "oracle_positive_markdown_share": high_risk_oracle,
            "classification": "BALANCED" if weighted_high_waste >= high_risk_oracle else "MODERATELY_IMBALANCED",
            "notes": "Uses weighted exposure to meaningful scenarios; no oracle labels used during training.",
        },
        {
            "distribution": "oracle_audit",
            "scenario_count": int(oracle["scenario_id"].nunique()),
            "high_risk_share_estimate": high_risk_oracle,
            "oracle_positive_markdown_share": high_risk_oracle,
            "classification": "diagnostic_reference",
            "notes": "Validation-state diagnostic, not training target.",
        },
    ]
    pd.DataFrame(rows).to_csv(TABLES_DIR / "ppo_training_state_balance_audit.csv", index=False)


def train_variant(spec: VariantSpec, timesteps: int) -> dict[str, Any]:
    started = time.perf_counter()
    set_seed(SEED)
    weights = scenario_sampling_weights(load_meaningful_scenarios())
    variant_dir = MODELS_DIR / spec.variant.lower() / f"seed_{SEED}"
    checkpoint_dir = variant_dir / "checkpoints"
    monitor_dir = variant_dir / "monitor"
    for path in [variant_dir, checkpoint_dir, monitor_dir]:
        path.mkdir(parents=True, exist_ok=True)
    vec_env = make_balanced_vec_env(weights, SEED, monitor_dir)
    params = PPO_BASE_PARAMS.copy()
    params.update(spec.ppo_overrides)
    model = PPO("MlpPolicy", vec_env, policy_kwargs=policy_kwargs(), seed=SEED, verbose=0, **params)
    callback = VecNormalizeCheckpointCallback(
        save_freq=max(CHECKPOINT_EVERY // N_ENVS, 1),
        save_path=str(checkpoint_dir),
        name_prefix=f"redesign_{spec.variant.lower()}",
        vec_path=variant_dir / "vecnormalize.pkl",
        run_state_path=variant_dir / "run_state.json",
    )
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=False)
    model.save(variant_dir / "final_model.zip")
    vec_env.save(str(variant_dir / "vecnormalize.pkl"))
    vec_env.close()
    metadata = {
        "variant": spec.variant,
        "seed": SEED,
        "timesteps": timesteps,
        "sampling_mode": spec.sampling_mode,
        "ppo_params": params,
        "sampling_weights_path": str((TABLES_DIR / "ppo_redesign_sampling_weights.csv").relative_to(PROJECT_ROOT)),
        "test_split_used": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    (variant_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def checkpoint_paths(variant: str) -> list[tuple[int, Path]]:
    variant_dir = MODELS_DIR / variant.lower() / f"seed_{SEED}"
    paths: list[tuple[int, Path]] = []
    for path in (variant_dir / "checkpoints").glob("*.zip"):
        tokens = path.stem.split("_")
        step = None
        for token in tokens:
            if token.isdigit():
                step = int(token)
        if step is not None:
            paths.append((step, path))
    final = variant_dir / "final_model.zip"
    if final.exists():
        paths.append((max([step for step, _ in paths], default=0), final))
    return sorted(paths, key=lambda item: item[0])


def evaluate_variants() -> pd.DataFrame:
    manifest = load_manifest()
    rows: list[dict[str, Any]] = []
    for spec in [VariantSpec(**VARIANTS["balanced"]), VariantSpec(**VARIANTS["balanced_stable"])]:
        vec_path = MODELS_DIR / spec.variant.lower() / f"seed_{SEED}" / "vecnormalize.pkl"
        if not vec_path.exists():
            continue
        for step, model_path in checkpoint_paths(spec.variant):
            if model_path.exists():
                rows.append(evaluate_checkpoint(model_path, vec_path, spec.variant, step, manifest))
    results = pd.DataFrame(rows)
    if not results.empty:
        best_profit = float(results["mean_normalized_validation_profit"].max())
        eligible = results.loc[
            results["mechanical_valid"]
            & (results["mean_normalized_validation_profit"] >= best_profit - 0.005)
            & (results["zero_action_share"] < 0.98)
            & (results["empirical_action_entropy"] > 0.05)
        ].copy()
        eligible = eligible.sort_values(["oracle_regret", "unnecessary_markdown_rate", "mean_normalized_validation_profit"], ascending=[True, True, False])
        results["acceptable_by_rule"] = results["policy_id"].isin(eligible["policy_id"].head(1)) if not eligible.empty else False
    results.to_csv(TABLES_DIR / "ppo_training_redesign_results.csv", index=False)
    return results


def compare_and_select(results: pd.DataFrame) -> pd.DataFrame:
    original = {
        "policy_id": "recovered_financial__checkpoint_22288",
        "variant": "ORIGINAL_RECOVERED_22288",
        "checkpoint_timestep": 22288,
        "mean_normalized_validation_profit": 0.436829,
        "oracle_regret": ORIGINAL_REGRET,
        "unnecessary_markdown_rate": 0.6250,
        "missed_markdown_opportunity_rate": 0.041667,
        "zero_action_share": 0.791667,
        "empirical_action_entropy": 0.326752,
        "state_dependence_classification": "MOSTLY_ZERO_BUT_STATE_DEPENDENT",
        "collapse_flag": False,
        "paired_profit_difference_vs_always_zero": -0.0132,
    }
    combined = pd.concat([pd.DataFrame([original]), results], ignore_index=True, sort=False)
    candidates = combined.loc[combined["variant"].ne("ORIGINAL_RECOVERED_22288")].copy()
    if candidates.empty:
        outcome = "REDESIGN_NO_MEANINGFUL_IMPROVEMENT"
        best_policy = original["policy_id"]
    else:
        acceptable = candidates.loc[candidates.get("acceptable_by_rule", False).astype(bool)] if "acceptable_by_rule" in candidates else pd.DataFrame()
        pool = acceptable if not acceptable.empty else candidates
        pool = pool.sort_values(["oracle_regret", "unnecessary_markdown_rate", "mean_normalized_validation_profit"], ascending=[True, True, False])
        best = pool.iloc[0]
        best_policy = best["policy_id"]
        regret_improves = safe_float(best["oracle_regret"]) <= ORIGINAL_REGRET * 0.90
        unnecessary_improves = safe_float(best["unnecessary_markdown_rate"]) < original["unnecessary_markdown_rate"]
        collapse_reduced = not bool(best["collapse_flag"]) and safe_float(best["zero_action_share"]) < 0.98
        profit_competitive = safe_float(best["mean_normalized_validation_profit"]) >= safe_float(candidates["mean_normalized_validation_profit"].max()) - 0.005
        if regret_improves and unnecessary_improves and collapse_reduced and profit_competitive:
            outcome = "REDESIGN_IMPROVES_POLICY_PRECISION"
        elif collapse_reduced:
            outcome = "REDESIGN_REDUCES_COLLAPSE_BUT_NOT_REGRET"
        else:
            outcome = "REDESIGN_NO_MEANINGFUL_IMPROVEMENT"
    combined["selected_best_policy"] = combined["policy_id"].eq(best_policy)
    combined["comparison_outcome"] = outcome
    combined.to_csv(TABLES_DIR / "ppo_multimetric_checkpoint_selection.csv", index=False)
    return combined


def plot_results(results: pd.DataFrame, selection: pd.DataFrame) -> None:
    if results.empty:
        return
    for metric, filename, ylabel in [
        ("mean_normalized_validation_profit", "validation_profit_by_timestep.png", "Validation normalized profit"),
        ("oracle_regret", "oracle_regret_by_timestep.png", "Oracle regret"),
        ("unnecessary_markdown_rate", "unnecessary_markdown_by_timestep.png", "Unnecessary markdown rate"),
        ("zero_action_share", "zero_action_share_by_timestep.png", "Zero-action share"),
        ("empirical_action_entropy", "action_entropy_by_timestep.png", "Empirical action entropy"),
        ("paired_profit_difference_vs_always_zero", "ppo_vs_always_zero_paired_profit.png", "Profit diff vs always-zero"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for variant, group in results.groupby("variant"):
            ax.plot(group["checkpoint_timestep"], group[metric], marker="o", label=variant)
        ax.set_xlabel("Training timesteps")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / filename, dpi=160)
        plt.close(fig)
    selected = selection.loc[selection["selected_best_policy"]]
    action_source = selected if not selected.empty else results.sort_values("oracle_regret").head(2)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    action_data = action_source[["policy_id"] + ACTION_COLS].set_index("policy_id")
    action_data.columns = [ACTION_LABELS[col] for col in action_data.columns]
    action_data.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Action share")
    ax.set_title("Selected action distributions")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "selected_action_distributions.png", dpi=160)
    plt.close(fig)
    ppo_regret = pd.read_csv(TABLES_DIR / "financial_markdown_ppo_regret.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    err = ppo_regret.groupby("ppo_policy_id")[["unnecessary_markdown_action", "missed_positive_markdown_opportunity"]].mean()
    err.plot(kind="bar", ax=ax)
    ax.set_title("Original vs redesigned error decomposition reference")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "original_vs_redesigned_error_decomposition.png", dpi=160)
    plt.close(fig)
    high_risk = results[["variant", "checkpoint_timestep", "zero_action_share", "positive_action_share"]].copy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for variant, group in high_risk.groupby("variant"):
        ax.plot(group["checkpoint_timestep"], group["positive_action_share"], marker="o", label=variant)
    ax.set_ylabel("Positive-action share")
    ax.set_title("High-risk-state action behavior proxy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "high_risk_state_action_behavior.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    summary = selection[["policy_id", "mean_normalized_validation_profit", "oracle_regret", "unnecessary_markdown_rate", "zero_action_share"]].dropna(how="all")
    ax.axis("off")
    ax.table(cellText=summary.round(4).astype(str).values, colLabels=summary.columns, loc="center", fontsize=7)
    ax.set_title("Multi-metric checkpoint-selection summary")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "multimetric_checkpoint_selection_summary.png", dpi=160)
    plt.close(fig)


def final_report(selection: pd.DataFrame, runtime_seconds: float) -> str:
    best = selection.loc[selection["selected_best_policy"]].iloc[0] if not selection.loc[selection["selected_best_policy"]].empty else selection.iloc[0]
    outcome = str(selection["comparison_outcome"].dropna().iloc[0]) if "comparison_outcome" in selection else "REDESIGN_NO_MEANINGFUL_IMPROVEMENT"
    if outcome == "REDESIGN_IMPROVES_POLICY_PRECISION":
        status = "PPO_TRAINING_REDESIGN_SUCCESSFUL"
    elif outcome == "REDESIGN_REDUCES_COLLAPSE_BUT_NOT_REGRET":
        status = "PPO_TRAINING_REDESIGN_PARTIAL"
    elif outcome == "REDESIGN_NO_MEANINGFUL_IMPROVEMENT":
        status = "PPO_TRAINING_REDESIGN_NOT_BENEFICIAL"
    else:
        status = "PPO_TRAINING_REDESIGN_REQUIRES_REVISION"
    balance = pd.read_csv(TABLES_DIR / "ppo_training_state_balance_audit.csv")
    report = {
        "status": status,
        "training_state_imbalance_classification": str(balance.loc[balance["distribution"].eq("original_training"), "classification"].iloc[0]),
        "best_variant": str(best.get("variant", "")),
        "best_checkpoint": str(best.get("policy_id", "")),
        "validation_profit": safe_float(best.get("mean_normalized_validation_profit")),
        "oracle_regret": safe_float(best.get("oracle_regret")),
        "unnecessary_markdown_rate": safe_float(best.get("unnecessary_markdown_rate")),
        "missed_markdown_opportunity_rate": safe_float(best.get("missed_markdown_opportunity_rate")),
        "zero_action_share": safe_float(best.get("zero_action_share")),
        "action_entropy": safe_float(best.get("empirical_action_entropy")),
        "paired_difference_vs_always_zero": safe_float(best.get("paired_profit_difference_vs_always_zero")),
        "runtime_seconds": runtime_seconds,
        "limitations": [
            "Scenario-balanced sampling changes the training distribution.",
            "Validation and test distributions remain unchanged; test split is not used.",
            "Oracle is diagnostic/model-selection evidence only, not a supervised training label.",
            "Only seed 123 is trained for each redesigned variant.",
            "This controlled experiment does not establish real-world pricing effectiveness.",
        ],
        "files_created": [
            "outputs/tables/ppo_training_design_audit.csv",
            "outputs/tables/ppo_training_state_balance_audit.csv",
            "outputs/tables/ppo_redesign_sampling_weights.csv",
            "outputs/tables/ppo_training_redesign_results.csv",
            "outputs/tables/ppo_multimetric_checkpoint_selection.csv",
        ],
        "recommended_final_research_interpretation": "Use this as a small training-stability and decision-precision experiment, not as proof that PPO should replace the zero-markdown baseline.",
    }
    (TABLES_DIR / "ppo_training_redesign_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(status)
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled PPO training-redesign experiment for recovered_financial.")
    parser.add_argument("--mode", choices=["audit", "train", "evaluate", "all"], default="audit")
    parser.add_argument("--variant", choices=["balanced", "balanced_stable", "both"], default="both")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    started = time.perf_counter()
    weights = scenario_sampling_weights(load_meaningful_scenarios())
    training_design_audit(weights)
    state_balance_audit(weights)
    if args.mode in {"train", "all"}:
        variants = ["balanced", "balanced_stable"] if args.variant == "both" else [args.variant]
        for name in variants:
            spec = VariantSpec(**VARIANTS[name])
            print(f"Training {spec.variant} seed={SEED} timesteps={args.timesteps}")
            train_variant(spec, args.timesteps)
    if args.mode in {"evaluate", "all"}:
        results = evaluate_variants()
        selection = compare_and_select(results)
        plot_results(results, selection)
        final_report(selection, time.perf_counter() - started)
    elif args.mode == "audit":
        print("PPO_TRAINING_REDESIGN_AUDIT_READY")


if __name__ == "__main__":
    main()
