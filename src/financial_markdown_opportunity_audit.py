from __future__ import annotations

import copy
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pricing_env_operational as env_module  # noqa: E402
from pricing_env_operational import ACTION_MARKDOWNS, OperationalPerishablePricingEnv  # noqa: E402
from train_ppo_operational import make_env  # noqa: E402

TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
CONFIGS_DIR = PROJECT_ROOT / "outputs" / "configs"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "financial_markdown_opportunity_audit"
SCENARIO_AUDIT_PATH = TABLES_DIR / "sustainability_scenario_audit.csv"
SELECTED_MODELS_PATH = CONFIGS_DIR / "ppo_selected_financial_models.json"
DEFAULT_SCENARIO_ID = "core_008"

ACTIONS = list(ACTION_MARKDOWNS.keys())
EPISODES_PER_SCENARIO = 8
MAX_STATES_PER_SCENARIO_CALIBRATION = 4
CALIBRATIONS = ["recovered_calibration", "observed_calibration"]
EXPLORATION_POLICIES = ["always_0pct", "always_20pct", "always_40pct", "expiry_threshold_rule", "inventory_coverage_rule"]
CONTINUATION_POLICIES = ["always_0pct", "expiry_threshold_rule", "inventory_coverage_rule", "greedy_financial"]
PPO_POLICIES = ["selected_observed_financial_ppo", "selected_recovered_financial_ppo"]
PROFIT_GAIN_TOL = 1e-4
MEANINGFUL_POSITIVE_SHARE = 0.10
ZERO_DOMINANT_SHARE = 0.90
EPS = 1e-9

_ORIGINAL_ELIGIBLE = env_module.eligible_episode_starts
_ELIGIBLE_CACHE: dict[tuple[int, int], pd.DataFrame] = {}


def cached_eligible_episode_starts(data: pd.DataFrame, horizon: int) -> pd.DataFrame:
    key = (id(data), int(horizon))
    if key not in _ELIGIBLE_CACHE:
        _ELIGIBLE_CACHE[key] = _ORIGINAL_ELIGIBLE(data, horizon)
    return _ELIGIBLE_CACHE[key]


env_module.eligible_episode_starts = cached_eligible_episode_starts


@dataclass
class LoadedPPO:
    policy_id: str
    training_calibration_mode: str
    model: PPO
    vecnormalize: VecNormalize
    classification: str


def ensure_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def safe_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric if np.isfinite(numeric) else 0.0


def action_for_policy(policy_id: str, obs: np.ndarray) -> int:
    if policy_id == "always_0pct":
        return 0
    if policy_id == "always_5pct":
        return 1
    if policy_id == "always_10pct":
        return 2
    if policy_id == "always_20pct":
        return 3
    if policy_id == "always_30pct":
        return 4
    if policy_id == "always_40pct":
        return 5
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


def state_features(obs: np.ndarray, day: int, env: OperationalPerishablePricingEnv) -> dict[str, Any]:
    remaining = int(getattr(env, "horizon", 0)) - day
    return {
        "inventory_coverage_state": safe_float(obs[1]) if obs.shape[0] > 1 else 0.0,
        "fraction_expiring_today": safe_float(obs[23]) if obs.shape[0] > 23 else 0.0,
        "fraction_expiring_within_two_days": safe_float(obs[24]) if obs.shape[0] > 24 else 0.0,
        "predicted_baseline_demand": safe_float(obs[26]) if obs.shape[0] > 26 else 0.0,
        "episode_day": day,
        "episode_period": "early" if day <= 1 else "middle" if day < max(getattr(env, "horizon", 1) - 1, 1) else "late",
        "remaining_inventory_life": remaining,
    }


def load_selected_scenarios() -> pd.DataFrame:
    audit = pd.read_csv(SCENARIO_AUDIT_PATH)
    selected = audit.loc[audit["classification"].eq("MEANINGFUL_PROFIT_WASTE_TRADEOFF")].copy()
    return selected


def resolve_path(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_ppos() -> dict[str, LoadedPPO]:
    if not SELECTED_MODELS_PATH.exists():
        return {}
    metadata = json.loads(SELECTED_MODELS_PATH.read_text(encoding="utf-8"))
    loaded: dict[str, LoadedPPO] = {}
    for policy_id, details in metadata.get("selected_models", {}).items():
        if policy_id not in PPO_POLICIES:
            continue
        model_path = resolve_path(details["model_path"])
        vec_path = resolve_path(details["vecnormalize_path"])
        if not model_path.exists() or not vec_path.exists():
            continue
        raw_env = DummyVecEnv(
            [
                lambda: make_env(
                    split="validation",
                    calibration_mode=details["calibration_mode"],
                    reward_mode="financial",
                    lambda_waste=0.10,
                    scenario_id=DEFAULT_SCENARIO_ID,
                    deterministic_demand=True,
                    seed=42,
                    monitor_dir=None,
                )
            ]
        )
        vecnorm = VecNormalize.load(str(vec_path), raw_env)
        vecnorm.training = False
        vecnorm.norm_reward = False
        loaded[policy_id] = LoadedPPO(
            policy_id=policy_id,
            training_calibration_mode=details["calibration_mode"],
            model=PPO.load(model_path),
            vecnormalize=vecnorm,
            classification=details.get("classification", ""),
        )
    return loaded


def collect_candidate_states(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, scenario in selected.iterrows():
        scenario_id = str(scenario["scenario_id"])
        for calibration_mode in CALIBRATIONS:
            print(f"Collecting states {scenario_id} / {calibration_mode}", flush=True)
            for policy_id in EXPLORATION_POLICIES:
                env = OperationalPerishablePricingEnv(
                    split="validation",
                    calibration_mode=calibration_mode,
                    reward_mode="financial",
                    lambda_waste=0.10,
                    scenario_id=scenario_id,
                    deterministic_demand=True,
                    random_seed=42,
                    promotion_context_mode="derived_from_action",
                )
                for episode_index in range(EPISODES_PER_SCENARIO):
                    obs, _ = env.reset(seed=20_000 + episode_index, options={"episode_index": episode_index})
                    prefix: list[int] = []
                    terminated = False
                    truncated = False
                    day = 0
                    while not (terminated or truncated):
                        features = state_features(obs, day, env)
                        rows.append(
                            {
                                "scenario_id": scenario_id,
                                "calibration_mode": calibration_mode,
                                "source_policy": policy_id,
                                "episode_index": episode_index,
                                "day": day,
                                "prefix_actions": "|".join(map(str, prefix)),
                                **features,
                                "shelf_life_class": scenario.get("shelf_life_class"),
                                "scenario_inventory_coverage": scenario.get("inventory_coverage"),
                                "age_profile": scenario.get("age_profile"),
                                "margin_scenario": scenario.get("margin_scenario"),
                                "disposal_scenario": scenario.get("disposal_scenario"),
                                "demand_target": scenario.get("demand_target"),
                                "response_model": scenario.get("response_model"),
                            }
                        )
                        action = action_for_policy(policy_id, obs)
                        obs, _, terminated, truncated, _ = env.step(action)
                        prefix.append(action)
                        day += 1
    return pd.DataFrame(rows)


def select_representative_states(candidates: pd.DataFrame) -> pd.DataFrame:
    selected_rows: list[pd.Series] = []
    feature_targets = [
        ("inventory_coverage_state", [0.10, 0.50, 0.90]),
        ("fraction_expiring_today", [0.10, 0.50, 0.90]),
        ("fraction_expiring_within_two_days", [0.10, 0.50, 0.90]),
        ("predicted_baseline_demand", [0.20, 0.80]),
        ("episode_day", [0.10, 0.50, 0.90]),
        ("remaining_inventory_life", [0.10, 0.50, 0.90]),
    ]
    for keys, group in candidates.groupby(["scenario_id", "calibration_mode"]):
        chosen_idx: set[int] = set()
        for feature, qs in feature_targets:
            values = group[feature].astype(float)
            for q in qs:
                target = float(values.quantile(q))
                idx = int((values - target).abs().idxmin())
                chosen_idx.add(idx)
        high_risk = group.sort_values(
            ["inventory_coverage_state", "fraction_expiring_within_two_days", "predicted_baseline_demand"],
            ascending=[False, False, True],
        ).head(4)
        chosen_idx.update(int(idx) for idx in high_risk.index)
        subset = group.loc[sorted(chosen_idx)].head(MAX_STATES_PER_SCENARIO_CALIBRATION)
        selected_rows.extend(row for _, row in subset.iterrows())
    states = pd.DataFrame(selected_rows).reset_index(drop=True)
    states.insert(0, "state_id", [f"state_{i:04d}" for i in range(len(states))])
    return states


def build_env_at_state(row: pd.Series) -> tuple[OperationalPerishablePricingEnv, np.ndarray]:
    env = OperationalPerishablePricingEnv(
        split="validation",
        calibration_mode=row["calibration_mode"],
        reward_mode="financial",
        lambda_waste=0.10,
        scenario_id=row["scenario_id"],
        deterministic_demand=True,
        random_seed=42,
        promotion_context_mode="derived_from_action",
    )
    obs, _ = env.reset(seed=20_000 + int(row["episode_index"]), options={"episode_index": int(row["episode_index"])})
    prefix = [] if not str(row["prefix_actions"]) or str(row["prefix_actions"]) == "nan" else [int(x) for x in str(row["prefix_actions"]).split("|") if x != ""]
    for action in prefix:
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return env, obs


def one_step_audit(states: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, state in states.iterrows():
        env, _ = build_env_at_state(state)
        base_revenue = safe_float(getattr(env, "cumulative_revenue", 0.0))
        base_disposal = safe_float(getattr(env, "cumulative_disposal_cost", 0.0))
        action_rows: list[dict[str, Any]] = []
        for action in ACTIONS:
            clone = copy.deepcopy(env)
            _, reward, terminated, truncated, info = clone.step(action)
            action_rows.append(
                {
                    **state.to_dict(),
                    "action": action,
                    "markdown": ACTION_MARKDOWNS[action],
                    "predicted_demand_after_action": safe_float(info.get("predicted_demand")),
                    "sales": safe_float(info.get("sales")),
                    "revenue": safe_float(info.get("cumulative_revenue")) - base_revenue,
                    "expired_units": safe_float(info.get("expired_units")),
                    "disposal_cost": safe_float(info.get("cumulative_disposal_cost")) - base_disposal,
                    "immediate_financial_reward": float(reward),
                    "terminated_after_action": bool(terminated or truncated),
                }
            )
        best_action = max(action_rows, key=lambda x: x["immediate_financial_reward"])["action"]
        for item in action_rows:
            item["preferred_one_step_action"] = best_action
            rows.append(item)
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "financial_markdown_state_audit.csv", index=False)
    return out


def greedy_action(env: OperationalPerishablePricingEnv) -> int:
    best_action = 0
    best_reward = -1e18
    for action in ACTIONS:
        clone = copy.deepcopy(env)
        _, reward, _, _, _ = clone.step(action)
        if reward > best_reward + EPS:
            best_reward = float(reward)
            best_action = action
    return best_action


def rollout(env: OperationalPerishablePricingEnv, obs: np.ndarray, continuation_policy: str) -> dict[str, Any]:
    terminated = False
    truncated = False
    final_info: dict[str, Any] = {}
    total_reward = 0.0
    action_counts = {action: 0 for action in ACTIONS}
    while not (terminated or truncated):
        action = greedy_action(env) if continuation_policy == "greedy_financial" else action_for_policy(continuation_policy, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        action_counts[action] += 1
        total_reward += float(reward)
        final_info = info
    return {
        "total_normalized_accounting_profit": safe_float(final_info.get("normalized_accounting_profit")),
        "total_waste_rate": safe_float(final_info.get("final_waste_rate")),
        "sell_through_rate": safe_float(final_info.get("final_sell_through_rate")),
        "terminal_inventory": safe_float(final_info.get("terminal_inventory")),
        "continuation_return": total_reward,
        **{f"continuation_action_{a}_share": action_counts[a] / max(sum(action_counts.values()), 1) for a in ACTIONS},
    }


def multistep_action_values(states: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, state in states.iterrows():
        base_env, _ = build_env_at_state(state)
        for continuation in CONTINUATION_POLICIES:
            action_values: list[dict[str, Any]] = []
            for first_action in ACTIONS:
                env = copy.deepcopy(base_env)
                obs, first_reward, terminated, truncated, info = env.step(first_action)
                if terminated or truncated:
                    result = {
                        "total_normalized_accounting_profit": safe_float(info.get("normalized_accounting_profit")),
                        "total_waste_rate": safe_float(info.get("final_waste_rate")),
                        "sell_through_rate": safe_float(info.get("final_sell_through_rate")),
                        "terminal_inventory": safe_float(info.get("terminal_inventory")),
                        "continuation_return": 0.0,
                    }
                else:
                    result = rollout(env, obs, continuation)
                action_values.append(
                    {
                        **state.to_dict(),
                        "first_action": first_action,
                        "first_markdown": ACTION_MARKDOWNS[first_action],
                        "continuation_policy": continuation,
                        "first_step_reward": float(first_reward),
                        **result,
                    }
                )
            zero_value = next(v["total_normalized_accounting_profit"] for v in action_values if v["first_action"] == 0)
            best_action = max(action_values, key=lambda x: x["total_normalized_accounting_profit"])["first_action"]
            for item in action_values:
                item["action_value_difference_vs_first_action_0pct"] = item["total_normalized_accounting_profit"] - zero_value
                item["preferred_multistep_first_action"] = best_action
                rows.append(item)
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "financial_markdown_multistep_action_values.csv", index=False)
    return out


def evaluate_action_sequence(base_env: OperationalPerishablePricingEnv, sequence: tuple[int, ...]) -> float:
    env = copy.deepcopy(base_env)
    terminated = False
    truncated = False
    info: dict[str, Any] = {}
    start_profit = safe_float(getattr(env, "accounting_profit", 0.0))
    initial_inventory = safe_float(getattr(env, "initial_inventory", 0.0))
    for action in sequence:
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return safe_float(info.get("normalized_accounting_profit"))
    end_profit = safe_float(getattr(env, "accounting_profit", 0.0))
    return (end_profit - start_profit) / max(initial_inventory, EPS)


def oracle_diagnostic_from_multistep(multistep_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    greedy = multistep_df.loc[multistep_df["continuation_policy"].eq("greedy_financial")].copy()
    for state_id, group in greedy.groupby("state_id"):
        best = group.sort_values("total_normalized_accounting_profit", ascending=False).iloc[0]
        zero_value = float(group.loc[group["first_action"].eq(0), "total_normalized_accounting_profit"].iloc[0])
        positive_best = group.loc[group["first_action"].ne(0)].sort_values("total_normalized_accounting_profit", ascending=False).head(1)
        best_value = float(best["total_normalized_accounting_profit"])
        rows.append(
            {
                **{col: best[col] for col in [
                    "state_id",
                    "scenario_id",
                    "calibration_mode",
                    "source_policy",
                    "episode_index",
                    "day",
                    "prefix_actions",
                    "inventory_coverage_state",
                    "fraction_expiring_today",
                    "fraction_expiring_within_two_days",
                    "predicted_baseline_demand",
                    "episode_period",
                    "remaining_inventory_life",
                    "shelf_life_class",
                    "scenario_inventory_coverage",
                    "age_profile",
                    "margin_scenario",
                    "disposal_scenario",
                    "demand_target",
                    "response_model",
                ] if col in best.index},
                "oracle_horizon": "remaining_episode_with_greedy_financial_continuation",
                "rolling_oracle_preferred_action": int(best["first_action"]),
                "rolling_oracle_preferred_markdown": ACTION_MARKDOWNS[int(best["first_action"])],
                "rolling_oracle_value": best_value,
                "zero_first_action_oracle_value": zero_value,
                "oracle_gain_vs_zero": best_value - zero_value,
                "best_positive_action": int(positive_best["first_action"].iloc[0]),
                "best_positive_gain_vs_zero": float(positive_best["total_normalized_accounting_profit"].iloc[0]) - zero_value,
                "positive_markdown_financially_preferred": int(best["first_action"]) != 0 and best_value - zero_value > PROFIT_GAIN_TOL,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "financial_markdown_oracle_diagnostic.csv", index=False)
    return out


def ppo_action(loaded: LoadedPPO, obs: np.ndarray) -> int:
    model_obs = loaded.vecnormalize.normalize_obs(obs.astype(np.float32)[None, :])[0]
    predicted, _ = loaded.model.predict(model_obs, deterministic=True)
    return int(np.asarray(predicted).item())


def ppo_regret(states: pd.DataFrame, oracle_df: pd.DataFrame, ppos: dict[str, LoadedPPO], multistep_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    oracle_map = oracle_df.set_index("state_id")
    greedy_values = multistep_df.loc[multistep_df["continuation_policy"].eq("greedy_financial")].set_index(["state_id", "first_action"])
    for _, state in states.iterrows():
        _, obs = build_env_at_state(state)
        oracle = oracle_map.loc[state["state_id"]]
        for policy_id, loaded in ppos.items():
            action = ppo_action(loaded, obs)
            if (state["state_id"], action) in greedy_values.index:
                action_value = safe_float(greedy_values.loc[(state["state_id"], action), "total_normalized_accounting_profit"])
            else:
                action_value = safe_float(oracle["zero_first_action_oracle_value"])
            rows.append(
                {
                    **state.to_dict(),
                    "ppo_policy_id": policy_id,
                    "ppo_training_calibration_mode": loaded.training_calibration_mode,
                    "ppo_action": action,
                    "ppo_markdown": ACTION_MARKDOWNS[action],
                    "oracle_action": int(oracle["rolling_oracle_preferred_action"]),
                    "oracle_markdown": safe_float(oracle["rolling_oracle_preferred_markdown"]),
                    "action_agrees_with_oracle": action == int(oracle["rolling_oracle_preferred_action"]),
                    "missed_positive_markdown_opportunity": action == 0 and int(oracle["rolling_oracle_preferred_action"]) != 0,
                    "unnecessary_markdown_action": action != 0 and int(oracle["rolling_oracle_preferred_action"]) == 0,
                    "normalized_profit_regret_vs_oracle": safe_float(oracle["rolling_oracle_value"]) - action_value,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "financial_markdown_ppo_regret.csv", index=False)
    return out


def classify_opportunity(oracle_df: pd.DataFrame, ppo_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in oracle_df.groupby(["scenario_id", "calibration_mode"]):
        scenario_id, calibration_mode = keys
        zero_share = float(group["rolling_oracle_preferred_action"].eq(0).mean())
        positive = group.loc[group["positive_markdown_financially_preferred"]]
        positive_share = float(len(positive) / max(len(group), 1))
        mean_positive_gain = float(positive["oracle_gain_vs_zero"].mean()) if not positive.empty else 0.0
        ppo_group = ppo_df.loc[ppo_df["scenario_id"].eq(scenario_id) & ppo_df["calibration_mode"].eq(calibration_mode)]
        missed_share = float(ppo_group["missed_positive_markdown_opportunity"].mean()) if not ppo_group.empty else 0.0
        agreement = float(ppo_group["action_agrees_with_oracle"].mean()) if not ppo_group.empty else 0.0
        if zero_share >= ZERO_DOMINANT_SHARE and mean_positive_gain <= PROFIT_GAIN_TOL:
            classification = "FINANCIALLY_ZERO_DOMINANT"
        elif positive_share >= MEANINGFUL_POSITIVE_SHARE and missed_share >= 0.50:
            classification = "FINANCIAL_OPPORTUNITY_EXISTS_BUT_PPO_MISSES_IT"
        elif positive_share >= MEANINGFUL_POSITIVE_SHARE:
            classification = "FINANCIALLY_STATE_DEPENDENT_MARKDOWN"
        else:
            classification = "INCONCLUSIVE"
        rows.append(
            {
                "scenario_id": scenario_id,
                "calibration_mode": calibration_mode,
                "audited_states": int(len(group)),
                "zero_preferred_share": zero_share,
                "positive_markdown_preferred_share": positive_share,
                "mean_positive_gain_vs_zero": mean_positive_gain,
                "ppo_action_agreement_rate": agreement,
                "ppo_missed_positive_opportunity_share": missed_share,
                "classification": classification,
                "positive_share_threshold": MEANINGFUL_POSITIVE_SHARE,
                "zero_dominant_share_threshold": ZERO_DOMINANT_SHARE,
                "profit_gain_tolerance": PROFIT_GAIN_TOL,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "financial_markdown_opportunity_classification.csv", index=False)
    return out


def save_figures(oracle_df: pd.DataFrame, multistep_df: pd.DataFrame, ppo_df: pd.DataFrame) -> None:
    rec_oracle = oracle_df.loc[oracle_df["calibration_mode"].eq("recovered_calibration")]
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(rec_oracle["inventory_coverage_state"], rec_oracle["fraction_expiring_within_two_days"], c=rec_oracle["rolling_oracle_preferred_action"], cmap="viridis")
    ax.set_xlabel("Inventory coverage")
    ax.set_ylabel("Fraction expiring within two days")
    ax.set_title("Optimal financial action by coverage and expiry pressure")
    plt.colorbar(scatter, ax=ax, label="Action")
    save_fig("optimal_action_by_inventory_and_expiry.png")

    sample = multistep_df.loc[
        multistep_df["calibration_mode"].eq("recovered_calibration")
        & multistep_df["continuation_policy"].eq("greedy_financial")
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    sample.groupby("first_markdown")["total_normalized_accounting_profit"].mean().plot(marker="o", ax=ax)
    ax.set_xlabel("First markdown")
    ax.set_ylabel("Mean total normalized accounting profit")
    ax.set_title("Multi-step action value versus markdown")
    save_fig("multistep_action_value_vs_markdown.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    rec_oracle["rolling_oracle_preferred_action"].value_counts(normalize=True).sort_index().plot(kind="bar", ax=ax)
    ax.set_xlabel("Preferred action")
    ax.set_ylabel("Share of audited states")
    ax.set_title("Share of states preferring each action")
    save_fig("share_states_preferring_each_action.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    if not ppo_df.empty:
        pd.crosstab(ppo_df["ppo_policy_id"], ppo_df["action_agrees_with_oracle"]).plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("PPO versus oracle action comparison")
    save_fig("ppo_vs_oracle_action_comparison.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    rec_oracle["best_positive_gain_vs_zero"].plot(kind="hist", bins=20, ax=ax)
    ax.set_xlabel("Best positive markdown gain over zero")
    ax.set_title("Financial gain of positive markdown over zero")
    save_fig("positive_markdown_gain_over_zero.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    justified = rec_oracle.loc[rec_oracle["positive_markdown_financially_preferred"]]
    ax.scatter(rec_oracle["predicted_baseline_demand"], rec_oracle["fraction_expiring_within_two_days"], c="lightgray", label="zero/no gain")
    if not justified.empty:
        ax.scatter(justified["predicted_baseline_demand"], justified["fraction_expiring_within_two_days"], c="tab:red", label="positive justified")
    ax.set_xlabel("Predicted baseline demand")
    ax.set_ylabel("Fraction expiring within two days")
    ax.legend()
    ax.set_title("State map of financially justified markdown")
    save_fig("state_map_financially_justified_markdown.png")


def save_fig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / name, dpi=150)
    plt.close()


def main() -> None:
    ensure_dirs()
    selected = load_selected_scenarios()
    ppos = load_ppos()
    candidates = collect_candidate_states(selected)
    states = select_representative_states(candidates)
    states.to_csv(TABLES_DIR / "financial_markdown_representative_states.csv", index=False)
    one_step = one_step_audit(states)
    multistep = multistep_action_values(states)
    oracle = oracle_diagnostic_from_multistep(multistep)
    ppo = ppo_regret(states, oracle, ppos, multistep)
    classification = classify_opportunity(oracle, ppo)
    save_figures(oracle, multistep, ppo)

    primary = classification.loc[classification["calibration_mode"].eq("recovered_calibration")]
    if primary["classification"].eq("FINANCIAL_OPPORTUNITY_EXISTS_BUT_PPO_MISSES_IT").any():
        status = "FINANCIAL_PPO_MISSES_EXISTING_OPPORTUNITIES"
    elif primary["classification"].eq("FINANCIALLY_STATE_DEPENDENT_MARKDOWN").any():
        status = "FINANCIAL_MARKDOWN_OPPORTUNITY_CONFIRMED"
    elif primary["classification"].eq("FINANCIALLY_ZERO_DOMINANT").all():
        status = "FINANCIAL_MARKDOWN_ZERO_DOMINANT"
    else:
        status = "FINANCIAL_MARKDOWN_OPPORTUNITY_CONFIRMED" if primary["positive_markdown_preferred_share"].max() >= MEANINGFUL_POSITIVE_SHARE else "FINANCIAL_MARKDOWN_ZERO_DOMINANT"

    report = {
        "status": status,
        "scenarios_evaluated": int(selected["scenario_id"].nunique()),
        "states_audited": int(len(states)),
        "calibrations": CALIBRATIONS,
        "actions": [ACTION_MARKDOWNS[a] for a in ACTIONS],
        "primary_recovered_positive_markdown_preferred_share_mean": float(primary["positive_markdown_preferred_share"].mean()),
        "primary_recovered_zero_preferred_share_mean": float(primary["zero_preferred_share"].mean()),
        "primary_recovered_mean_positive_gain_vs_zero": float(primary["mean_positive_gain_vs_zero"].mean()),
        "ppo_mean_action_agreement_rate": float(ppo["action_agrees_with_oracle"].mean()) if not ppo.empty else None,
        "ppo_mean_missed_positive_opportunity_share": float(ppo["missed_positive_markdown_opportunity"].mean()) if not ppo.empty else None,
        "classification_counts": classification["classification"].value_counts().to_dict(),
        "no_training": True,
        "test_split_used": False,
        "environment_modified": False,
        "files_created": [
            "outputs/tables/financial_markdown_state_audit.csv",
            "outputs/tables/financial_markdown_multistep_action_values.csv",
            "outputs/tables/financial_markdown_oracle_diagnostic.csv",
            "outputs/tables/financial_markdown_ppo_regret.csv",
            "outputs/tables/financial_markdown_opportunity_classification.csv",
        ],
    }
    (TABLES_DIR / "financial_markdown_opportunity_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(status)


if __name__ == "__main__":
    main()
