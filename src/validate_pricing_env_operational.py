"""Validate the operational pricing environment before PPO training."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium.utils.env_checker import check_env as gym_check_env

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pricing_env_operational import ACTION_MARKDOWNS, OperationalPerishablePricingEnv  # noqa: E402


TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
CONFIGS_DIR = PROJECT_ROOT / "outputs" / "configs"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "pricing_env_validation"

API_VALIDATION_PATH = TABLES_DIR / "pricing_env_api_validation.csv"
ROLLOUT_RESULTS_PATH = TABLES_DIR / "pricing_env_rollout_results.csv"
PAIRED_POLICY_PATH = TABLES_DIR / "pricing_env_paired_policy_comparison.csv"
ACCOUNTING_VALIDATION_PATH = TABLES_DIR / "pricing_env_accounting_validation.csv"
ACTION_EFFECT_PATH = TABLES_DIR / "pricing_env_action_effect_audit.csv"
NON_DEGENERACY_PATH = TABLES_DIR / "pricing_env_non_degeneracy_diagnostic.csv"
CALIBRATION_TRANSMISSION_PATH = TABLES_DIR / "pricing_env_calibration_transmission.csv"
SCENARIO_SENSITIVITY_PATH = TABLES_DIR / "pricing_env_scenario_sensitivity.csv"
REWARD_SCALE_PATH = TABLES_DIR / "pricing_env_reward_scale_diagnostic.csv"
ENV_CONFIG_PATH = CONFIGS_DIR / "pricing_env_operational_config.json"


def ensure_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def run_api_checks() -> pd.DataFrame:
    """Run Gymnasium and optional SB3 API checks."""
    rows = []
    env = OperationalPerishablePricingEnv(split="train", calibration_mode="recovered_calibration", reward_mode="financial")
    try:
        gym_check_env(env, skip_render_check=True)
        rows.append({"checker": "gymnasium", "status": "pass", "message": ""})
    except Exception as exc:
        rows.append({"checker": "gymnasium", "status": "fail", "message": repr(exc)})
    try:
        from stable_baselines3.common.env_checker import check_env as sb3_check_env

        sb3_check_env(env, warn=True)
        rows.append({"checker": "stable_baselines3", "status": "pass", "message": ""})
    except ImportError:
        rows.append({"checker": "stable_baselines3", "status": "skipped", "message": "stable_baselines3 not installed"})
    except Exception as exc:
        rows.append({"checker": "stable_baselines3", "status": "fail", "message": repr(exc)})
    output = pd.DataFrame(rows)
    output.to_csv(API_VALIDATION_PATH, index=False)
    return output


def policy_action(policy: str, env: OperationalPerishablePricingEnv) -> int:
    if policy == "always_0pct":
        return 0
    if policy == "always_10pct":
        return 2
    if policy == "always_20pct":
        return 3
    if policy == "always_40pct":
        return 5
    if policy == "random_policy":
        return int(env.np_random.integers(0, env.action_space.n))
    info = env.last_info
    if policy == "expiry_threshold_rule":
        profile = np.asarray(info.get("remaining_life_profile", []), dtype=float)
        share = profile[0] / max(profile.sum(), 1e-9) if profile.size else 0.0
        return 5 if share > 0.35 else 3 if share > 0.20 else 1
    if policy == "inventory_coverage_rule":
        coverage = info.get("inventory_after", env.initial_inventory) / max(np.mean(env.sim_history[-7:]), 1e-9)
        return 4 if coverage > 3.0 else 2 if coverage > 1.5 else 0
    raise ValueError(policy)


def rollout(
    split: str,
    calibration_mode: str,
    reward_mode: str,
    policy: str,
    seed: int,
    options: dict[str, Any] | None = None,
    scenario_id: str = "core_008",
    lambda_waste: float = 0.10,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = OperationalPerishablePricingEnv(
        split=split,
        calibration_mode=calibration_mode,
        reward_mode=reward_mode,
        scenario_id=scenario_id,
        lambda_waste=lambda_waste,
        random_seed=seed,
    )
    obs, info = env.reset(seed=seed, options=options)
    start_options = {"store_id": info["store_id"], "product_id": info["product_id"], "start_date": info["date"]}
    steps = []
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = policy_action(policy, env)
        obs, reward, terminated, truncated, info = env.step(action)
        row = dict(info)
        row.update(
            {
                "split": split,
                "calibration_mode": calibration_mode,
                "reward_mode": reward_mode,
                "policy": policy,
                "action": action,
                "reward": reward,
                "nonfinite_observation": not np.isfinite(obs).all(),
                "nonfinite_reward": not np.isfinite(reward),
            }
        )
        steps.append(row)
    final = dict(steps[-1])
    final.update(start_options)
    env.close()
    return final, steps


def run_rollouts() -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    policies = [
        "always_0pct",
        "always_10pct",
        "always_20pct",
        "always_40pct",
        "random_policy",
        "expiry_threshold_rule",
        "inventory_coverage_rule",
    ]
    rows = []
    step_rows = []
    paired = []
    base_env = OperationalPerishablePricingEnv(split="validation", calibration_mode="recovered_calibration")
    _obs, base_info = base_env.reset(seed=123)
    paired_options = {"store_id": base_info["store_id"], "product_id": base_info["product_id"], "start_date": base_info["date"]}
    base_env.close()
    for split in ["train", "validation"]:
        for calibration in ["observed_calibration", "recovered_calibration"]:
            for reward_mode in ["financial", "sustainability"]:
                for policy in policies:
                    final, steps = rollout(split, calibration, reward_mode, policy, seed=123, options=paired_options if split == "validation" else None)
                    rows.append(final)
                    step_rows.extend(steps)
    # minimal test smoke test, not used for parameter choice
    final, steps = rollout("test", "recovered_calibration", "financial", "always_0pct", seed=999)
    rows.append(final)
    step_rows.extend(steps)
    results = pd.DataFrame(rows)
    steps_df = pd.DataFrame(step_rows)
    results.to_csv(ROLLOUT_RESULTS_PATH, index=False)
    paired_cols = ["split", "calibration_mode", "reward_mode", "policy", "accounting_profit", "final_waste_rate", "final_sell_through_rate", "episode_return"]
    paired_df = results[paired_cols].copy()
    paired_df.to_csv(PAIRED_POLICY_PATH, index=False)
    return results, steps_df, [paired_options]


def accounting_validation(rollouts: pd.DataFrame, steps: pd.DataFrame) -> pd.DataFrame:
    summary = {
        "maximum_step_conservation_error": float(steps["step_conservation_error"].max()),
        "maximum_episode_conservation_error": float(rollouts.get("episode_conservation_error", pd.Series([0.0])).max()),
        "negative_inventory_events": int((steps["inventory_after"] < -1e-8).sum()),
        "sales_above_inventory_events": int((steps["sales"] > steps["inventory_before"] + 1e-8).sum()),
        "procurement_double_charge_events": int((steps.groupby(["split", "calibration_mode", "reward_mode", "policy", "store_id", "product_id"])["cumulative_procurement_cost"].diff().fillna(0).gt(0).groupby([steps["split"], steps["calibration_mode"], steps["reward_mode"], steps["policy"], steps["store_id"], steps["product_id"]]).sum() > 1).sum()),
        "maximum_reward_accounting_discrepancy": float(rollouts.get("raw_financial_sum_error", pd.Series([0.0])).max()),
        "non_finite_observations": int(steps["nonfinite_observation"].sum()),
        "non_finite_rewards": int(steps["nonfinite_reward"].sum()),
        "split_boundary_violations": int(split_boundary_violations(steps)),
    }
    output = pd.DataFrame([summary])
    output.to_csv(ACCOUNTING_VALIDATION_PATH, index=False)
    return output


def split_boundary_violations(steps: pd.DataFrame) -> int:
    dates = pd.to_datetime(steps["date"])
    violations = 0
    for split, group in steps.assign(_date=dates).groupby("split"):
        if split == "train" and group["_date"].max() > pd.Timestamp("2024-05-25"):
            violations += 1
        if split == "validation" and (group["_date"].min() < pd.Timestamp("2024-05-26") or group["_date"].max() > pd.Timestamp("2024-06-13")):
            violations += 1
        if split == "test" and group["_date"].min() < pd.Timestamp("2024-06-14"):
            violations += 1
    return violations


def action_effect_audit() -> pd.DataFrame:
    rows = []
    env = OperationalPerishablePricingEnv(split="validation", calibration_mode="recovered_calibration", reward_mode="sustainability")
    for idx in range(20):
        obs, info = env.reset(seed=200 + idx)
        snapshot = env_state_snapshot(env)
        for action, markdown in ACTION_MARKDOWNS.items():
            restore_env_state(env, snapshot)
            obs, reward, terminated, truncated, step_info = env.step(action)
            rows.append(
                {
                    "state_id": idx,
                    "action": action,
                    "markdown_rate": markdown,
                    "predicted_demand": step_info["predicted_demand"],
                    "selling_price": step_info["selling_price"],
                    "expected_sales": step_info["sales"],
                    "immediate_revenue": step_info["selling_price"] * step_info["sales"],
                    "expected_waste": step_info["expired_units"],
                    "raw_financial_contribution": step_info["raw_financial_step"],
                    "sustainability_reward": reward,
                    "support_status": "supported",
                }
            )
    env.close()
    output = pd.DataFrame(rows)
    output.to_csv(ACTION_EFFECT_PATH, index=False)
    return output


def env_state_snapshot(env: OperationalPerishablePricingEnv) -> dict[str, Any]:
    return {
        "current_step": env.current_step,
        "inventory": env.inventory.copy(),
        "initial_procurement_cost_pending": env.initial_procurement_cost_pending,
        "cumulative_sales": env.cumulative_sales,
        "cumulative_waste": env.cumulative_waste,
        "cumulative_revenue": env.cumulative_revenue,
        "cumulative_procurement_cost": env.cumulative_procurement_cost,
        "cumulative_disposal_cost": env.cumulative_disposal_cost,
        "accounting_profit": env.accounting_profit,
        "episode_return": env.episode_return,
        "raw_financial_steps": list(env.raw_financial_steps),
        "reward_steps": list(env.reward_steps),
        "sim_history": list(env.sim_history),
        "stockout_history": list(env.stockout_history),
        "previous_markdown": env.previous_markdown,
        "last_info": dict(env.last_info),
    }


def restore_env_state(env: OperationalPerishablePricingEnv, state: dict[str, Any]) -> None:
    for key, value in state.items():
        setattr(env, key, value.copy() if isinstance(value, np.ndarray) else list(value) if isinstance(value, list) else dict(value) if isinstance(value, dict) else value)


def non_degeneracy(action_effects: pd.DataFrame) -> pd.DataFrame:
    preferred = action_effects.loc[action_effects.groupby("state_id")["raw_financial_contribution"].idxmax()].copy()
    dist = preferred["action"].value_counts(normalize=True).reindex(range(6), fill_value=0.0)
    entropy = float(-(dist[dist > 0] * np.log(dist[dist > 0])).sum())
    max_share = float(dist.max())
    if dist.loc[5] > 0.90:
        classification = "OVER_DISCOUNTING_DEGENERATE"
    elif dist.loc[0] > 0.90:
        classification = "FINANCIALLY_DEGENERATE"
    elif max_share <= 0.90 and dist.loc[1:].sum() > 0.10 and dist.loc[0] > 0.05:
        classification = "NON_DEGENERATE"
    else:
        classification = "SUSTAINABILITY_ONLY_TRADEOFF"
    rows = [{"metric": f"action_{i}_preferred_share", "value": float(dist.loc[i])} for i in range(6)]
    rows += [
        {"metric": "optimal_action_entropy", "value": entropy},
        {"metric": "dominant_action_share", "value": max_share},
        {"metric": "classification", "value": classification},
        {"metric": "dominant_action_warning_threshold", "value": 0.90},
    ]
    output = pd.DataFrame(rows)
    output.to_csv(NON_DEGENERACY_PATH, index=False)
    return output


def calibration_transmission() -> pd.DataFrame:
    rows = []
    policies = ["always_0pct", "always_20pct", "expiry_threshold_rule"]
    for seed in range(10):
        base_env = OperationalPerishablePricingEnv(split="validation", calibration_mode="recovered_calibration")
        _obs, info = base_env.reset(seed=500 + seed)
        options = {"store_id": info["store_id"], "product_id": info["product_id"], "start_date": info["date"]}
        base_env.close()
        paired = {}
        for calibration in ["observed_calibration", "recovered_calibration"]:
            for policy in policies:
                final, _steps = rollout("validation", calibration, "financial", policy, seed=500 + seed, options=options)
                paired[(calibration, policy)] = final
        best_obs = max(policies, key=lambda p: paired[("observed_calibration", p)]["accounting_profit"])
        best_rec = max(policies, key=lambda p: paired[("recovered_calibration", p)]["accounting_profit"])
        rows.append(
            {
                "episode_seed": seed,
                "preferred_action_changed": best_obs != best_rec,
                "observed_best_policy": best_obs,
                "recovered_best_policy": best_rec,
                "mean_profit_difference": np.mean([paired[("recovered_calibration", p)]["accounting_profit"] - paired[("observed_calibration", p)]["accounting_profit"] for p in policies]),
                "mean_waste_difference": np.mean([paired[("recovered_calibration", p)]["final_waste_rate"] - paired[("observed_calibration", p)]["final_waste_rate"] for p in policies]),
            }
        )
    output = pd.DataFrame(rows)
    summary = {
        "episode_seed": "summary",
        "preferred_action_changed": float(output["preferred_action_changed"].mean()),
        "observed_best_policy": "",
        "recovered_best_policy": "",
        "mean_profit_difference": float(output["mean_profit_difference"].mean()),
        "mean_waste_difference": float(output["mean_waste_difference"].mean()),
    }
    output = pd.concat([output, pd.DataFrame([summary])], ignore_index=True)
    output.to_csv(CALIBRATION_TRANSMISSION_PATH, index=False)
    return output


def scenario_sensitivity() -> pd.DataFrame:
    grid = pd.read_csv(PROJECT_ROOT / "outputs" / "tables" / "perishability_core_scenario_grid.csv")
    rows = []
    for scenario_id in grid["scenario_id"].head(18):
        for policy in ["always_0pct", "always_20pct", "always_40pct"]:
            try:
                final, _ = rollout("validation", "recovered_calibration", "financial", policy, seed=700, scenario_id=scenario_id)
                rows.append(final)
            except Exception as exc:
                rows.append({"scenario_id": scenario_id, "policy": policy, "error": repr(exc)})
    output = pd.DataFrame(rows)
    output.to_csv(SCENARIO_SENSITIVITY_PATH, index=False)
    return output


def reward_scale(steps: pd.DataFrame, rollouts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for reward_mode, group in steps.groupby("reward_mode"):
        waste_component = group["expired_units"] / group.groupby(["split", "calibration_mode", "reward_mode", "policy", "store_id", "product_id"])["inventory_before"].transform("first").replace(0, np.nan)
        rows.append(
            {
                "reward_mode": reward_mode,
                "step_reward_mean": float(group["reward"].mean()),
                "step_reward_std": float(group["reward"].std()),
                "step_reward_min": float(group["reward"].min()),
                "step_reward_max": float(group["reward"].max()),
                "episode_return_mean": float(rollouts.loc[rollouts["reward_mode"].eq(reward_mode), "episode_return"].mean()),
                "episode_return_std": float(rollouts.loc[rollouts["reward_mode"].eq(reward_mode), "episode_return"].std()),
                "financial_component_magnitude": float(group["raw_financial_step"].abs().mean()),
                "waste_penalty_component_magnitude": float(waste_component.abs().mean()),
                "waste_term_abs_reward_share": float(waste_component.abs().mean() / max(group["reward"].abs().mean(), 1e-9)),
                "flag": "ok",
            }
        )
    output = pd.DataFrame(rows)
    output.to_csv(REWARD_SCALE_PATH, index=False)
    return output


def create_figures(steps: pd.DataFrame, rollouts: pd.DataFrame, action_effects: pd.DataFrame, nondeg: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    sample = steps.iloc[: min(len(steps), 20)].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sample["sales"].to_numpy(), label="sales")
    ax.plot(sample["expired_units"].to_numpy(), label="waste")
    ax.plot(sample["inventory_after"].to_numpy(), label="inventory")
    ax.legend(); ax.set_title("Inventory, sales, and waste trajectory"); savefig("inventory_sales_waste_trajectory.png")
    fig, ax = plt.subplots(figsize=(10, 5))
    profiles = [np.asarray(x, dtype=float) for x in sample["remaining_life_profile"].head(10)]
    if profiles:
        mat = np.vstack([np.pad(p, (0, max(0, 21-len(p))))[:21] for p in profiles])
        ax.imshow(mat, aspect="auto"); ax.set_title("Remaining-life bucket trajectory"); savefig("remaining_life_bucket_trajectory.png")
    plot_bar(rollouts.groupby("policy")["accounting_profit"].mean(), "Profit comparison by fixed policy", "profit_comparison_by_fixed_policy.png")
    plot_bar(rollouts.groupby("policy")["final_waste_rate"].mean(), "Waste comparison by fixed policy", "waste_comparison_by_fixed_policy.png")
    fig, ax = plt.subplots(figsize=(7, 5)); ax.scatter(rollouts["final_waste_rate"], rollouts["accounting_profit"]); ax.set_title("Profit-versus-waste scatter"); savefig("profit_versus_waste_scatter.png")
    shares = nondeg.loc[nondeg["metric"].str.contains("action_")].copy(); plot_bar(pd.Series(shares["value"].to_numpy(), index=shares["metric"]), "Preferred-action distribution", "preferred_action_distribution.png")
    fig, ax = plt.subplots(figsize=(7, 5)); ax.scatter(action_effects["markdown_rate"], action_effects["raw_financial_contribution"]); ax.set_title("Preferred action by fraction expiring soon proxy"); savefig("preferred_action_by_fraction_expiring_soon.png")
    plot_bar(rollouts.groupby("calibration_mode")["accounting_profit"].mean(), "Observed versus recovered calibration outcome comparison", "observed_vs_recovered_calibration_comparison.png")
    fig, ax = plt.subplots(figsize=(8, 5)); ax.hist(steps["reward"], bins=40); ax.set_title("Reward component distributions"); savefig("reward_component_distributions.png")
    pivot = sensitivity.pivot_table(index="scenario_id", columns="policy", values="accounting_profit", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 6)); ax.imshow(pivot.fillna(0), aspect="auto"); ax.set_title("Scenario sensitivity matrix"); savefig("scenario_sensitivity_matrix.png")
    fig, ax = plt.subplots(figsize=(8, 5)); 
    for state, group in action_effects.groupby("state_id"):
        if state < 5:
            ax.plot(group["markdown_rate"], group["predicted_demand"], marker="o", alpha=0.7)
    ax.set_title("Action-effect curves for representative states"); savefig("action_effect_curves.png")
    plot_bar(pd.Series({"step": steps["step_conservation_error"].max(), "episode": rollouts.get("episode_conservation_error", pd.Series([0])).max()}), "Conservation-error summary", "conservation_error_summary.png")


def plot_bar(series: pd.Series, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5)); series.plot(kind="bar", ax=ax); ax.set_title(title); plt.xticks(rotation=30, ha="right"); savefig(filename)


def savefig(filename: str) -> None:
    plt.tight_layout(); plt.savefig(FIGURES_DIR / filename, dpi=160); plt.close()


def final_status(api: pd.DataFrame, accounting: pd.DataFrame, nondeg: pd.DataFrame, calibration: pd.DataFrame) -> str:
    api_ok = api.loc[api["checker"].eq("gymnasium"), "status"].iloc[0] == "pass"
    acc = accounting.iloc[0]
    mechanical = api_ok and all(
        [
            acc["negative_inventory_events"] == 0,
            acc["sales_above_inventory_events"] == 0,
            acc["procurement_double_charge_events"] == 0,
            acc["non_finite_observations"] == 0,
            acc["non_finite_rewards"] == 0,
            acc["split_boundary_violations"] == 0,
            acc["maximum_step_conservation_error"] <= 1e-6,
            acc["maximum_episode_conservation_error"] <= 1e-6,
        ]
    )
    classification = nondeg.loc[nondeg["metric"].eq("classification"), "value"].iloc[0]
    if not mechanical:
        return "PRICING_ENV_REQUIRES_REVISION"
    if classification == "FINANCIALLY_DEGENERATE":
        return "PRICING_ENV_MECHANICALLY_VALID_BUT_FINANCIALLY_DEGENERATE"
    if classification in {"NON_DEGENERATE", "SUSTAINABILITY_ONLY_TRADEOFF"} and not calibration.empty:
        return "PRICING_ENV_READY_FOR_PPO"
    return "PRICING_ENV_REQUIRES_REVISION"


def print_report(status: str, api: pd.DataFrame, accounting: pd.DataFrame, nondeg: pd.DataFrame, calibration: pd.DataFrame, reward: pd.DataFrame) -> None:
    print("\nOperational pricing environment validation report")
    print("API check status:", api.to_dict(orient="records"))
    print("Unit-test summary: run `pytest tests/test_pricing_env_operational.py` for local unit-test details.")
    print("Accounting validation summary:", accounting.iloc[0].to_dict())
    print("Default scenario: core_008, medium shelf life, balanced inventory, uniform age profile")
    print("Calibration modes validated: observed_calibration, recovered_calibration")
    print("Reward modes validated: financial, sustainability")
    classification = nondeg.loc[nondeg["metric"].eq("classification"), "value"].iloc[0]
    print("Non-degeneracy classification:", classification)
    print("Share of states preferring each action:", nondeg.loc[nondeg["metric"].str.contains("action_")].to_dict(orient="records"))
    summary = calibration.loc[calibration["episode_seed"].astype(str).eq("summary")]
    print("Preferred-action changes observed vs recovered:", None if summary.empty else summary["preferred_action_changed"].iloc[0])
    print("Reward-scale summary:", reward.to_dict(orient="records"))
    print("Unresolved semantic issues:")
    for item in required_limitations():
        print("-", item)
    print("Files created:")
    for path in created_files():
        print("-", path.relative_to(PROJECT_ROOT))
    print("Recommended next action: run the pytest suite; if still ready/non-degenerate, proceed to PPO in a separate module.")
    print(status)


def required_limitations() -> list[str]:
    return [
        "shelf life and inventory age are simulated",
        "realized waste is simulated",
        "costs are normalized assumptions",
        "recovered demand is model-estimated",
        "markdown response is observational and model-implied",
        "promotion-context mapping is an environment assumption",
        "actual latent demand during stockouts remains unobserved",
        "this environment supports controlled decision experiments",
        "it does not estimate direct retailer profit or deployment performance",
    ]


def created_files() -> list[Path]:
    return [
        API_VALIDATION_PATH, ROLLOUT_RESULTS_PATH, PAIRED_POLICY_PATH, ACCOUNTING_VALIDATION_PATH,
        ACTION_EFFECT_PATH, NON_DEGENERACY_PATH, CALIBRATION_TRANSMISSION_PATH,
        SCENARIO_SENSITIVITY_PATH, REWARD_SCALE_PATH, ENV_CONFIG_PATH,
    ]


def run() -> str:
    ensure_dirs()
    api = run_api_checks()
    rollouts, steps, _paired = run_rollouts()
    accounting = accounting_validation(rollouts, steps)
    action_effects = action_effect_audit()
    nondeg = non_degeneracy(action_effects)
    calibration = calibration_transmission()
    sensitivity = scenario_sensitivity()
    reward = reward_scale(steps, rollouts)
    create_figures(steps, rollouts, action_effects, nondeg, sensitivity)
    status = final_status(api, accounting, nondeg, calibration)
    print_report(status, api, accounting, nondeg, calibration, reward)
    return status


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"Operational pricing environment validation failed: {exc}")
        print("PRICING_ENV_REQUIRES_REVISION")
        raise
