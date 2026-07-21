from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pricing_env_operational as pricing_env_module  # noqa: E402
from pricing_env_operational import ACTION_MARKDOWNS, OperationalPerishablePricingEnv  # noqa: E402

TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
CONFIGS_DIR = PROJECT_ROOT / "outputs" / "configs"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "sustainability_opportunity_audit"

CORE_GRID_PATH = TABLES_DIR / "perishability_core_scenario_grid.csv"
SENSITIVITY_GRID_PATH = TABLES_DIR / "perishability_sensitivity_scenario_grid.csv"
STRESS_GRID_PATH = TABLES_DIR / "perishability_stress_test_grid.csv"
MASTER_CONFIG_PATH = CONFIGS_DIR / "perishability_master_config.json"

POLICIES = [
    "always_0pct",
    "always_5pct",
    "always_10pct",
    "always_20pct",
    "always_40pct",
    "expiry_threshold_rule",
    "inventory_coverage_rule",
]
LAMBDA_VALUES = [0.10, 0.25, 0.50, 1.00]
EPISODES_PER_SCENARIO = 8
EPS = 1e-9
_ORIGINAL_ELIGIBLE_EPISODE_STARTS = pricing_env_module.eligible_episode_starts
_ELIGIBLE_EPISODE_CACHE: dict[tuple[int, int], pd.DataFrame] = {}


def cached_eligible_episode_starts(data: pd.DataFrame, horizon: int) -> pd.DataFrame:
    key = (id(data), int(horizon))
    if key not in _ELIGIBLE_EPISODE_CACHE:
        _ELIGIBLE_EPISODE_CACHE[key] = _ORIGINAL_ELIGIBLE_EPISODE_STARTS(data, horizon)
    return _ELIGIBLE_EPISODE_CACHE[key]


pricing_env_module.eligible_episode_starts = cached_eligible_episode_starts


def ensure_dirs() -> None:
    for directory in [TABLES_DIR, CONFIGS_DIR, FIGURES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def safe_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric if np.isfinite(numeric) else 0.0


def action_entropy(counts: dict[int, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count:
            p = count / total
            entropy -= p * math.log(p)
    return float(entropy)


def action_for_policy(policy: str, obs: np.ndarray, rng: np.random.Generator) -> int:
    if policy == "always_0pct":
        return 0
    if policy == "always_5pct":
        return 1
    if policy == "always_10pct":
        return 2
    if policy == "always_20pct":
        return 3
    if policy == "always_40pct":
        return 5
    if policy == "expiry_threshold_rule":
        fraction_expiring_within_two_days = safe_float(obs[24]) if obs.shape[0] > 24 else 0.0
        if fraction_expiring_within_two_days >= 0.50:
            return 5
        if fraction_expiring_within_two_days >= 0.25:
            return 3
        if fraction_expiring_within_two_days >= 0.10:
            return 2
        return 0
    if policy == "inventory_coverage_rule":
        coverage = safe_float(obs[1]) if obs.shape[0] > 1 else 0.0
        if coverage >= 1.50:
            return 4
        if coverage >= 1.10:
            return 3
        if coverage >= 0.80:
            return 2
        return 0
    raise ValueError(policy)


def read_scenario_grids() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    core = pd.read_csv(CORE_GRID_PATH)
    sensitivity = pd.read_csv(SENSITIVITY_GRID_PATH) if SENSITIVITY_GRID_PATH.exists() else pd.DataFrame()
    stress = pd.read_csv(STRESS_GRID_PATH) if STRESS_GRID_PATH.exists() else pd.DataFrame()
    return core, sensitivity, stress


def evaluate_episode(
    *,
    scenario_id: str,
    policy: str,
    calibration_mode: str,
    episode_index: int,
    reward_mode: str = "financial",
    lambda_waste: float = 0.10,
    env: OperationalPerishablePricingEnv | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if env is None:
        env = OperationalPerishablePricingEnv(
            split="validation",
            calibration_mode=calibration_mode,
            reward_mode=reward_mode,
            lambda_waste=lambda_waste,
            scenario_id=scenario_id,
            deterministic_demand=True,
            random_seed=10_000 + episode_index,
            promotion_context_mode="derived_from_action",
        )
    obs, info = env.reset(seed=20_000 + episode_index, options={"episode_index": episode_index})
    rng = np.random.default_rng(30_000 + episode_index)
    initial_inventory = safe_float(getattr(env, "initial_inventory", 0.0))
    action_counts = {action: 0 for action in ACTION_MARKDOWNS}
    first_markdown_day = -1
    max_markdown = 0.0
    episode_return = 0.0
    step_rows: list[dict[str, Any]] = []
    terminated = False
    truncated = False
    day = 0
    final_info: dict[str, Any] = info
    while not (terminated or truncated):
        action = action_for_policy(policy, obs, rng)
        markdown = ACTION_MARKDOWNS[action]
        if first_markdown_day < 0 and markdown > 0:
            first_markdown_day = day
        max_markdown = max(max_markdown, markdown)
        obs, reward, terminated, truncated, info = env.step(action)
        expired = safe_float(info.get("expired_units"))
        normalized_expired = expired / max(initial_inventory, EPS)
        financial_component = safe_float(info.get("raw_financial_step")) / max(initial_inventory, EPS)
        expected_penalty = -lambda_waste * normalized_expired if reward_mode == "sustainability" else 0.0
        step_rows.append(
            {
                "scenario_id": scenario_id,
                "policy": policy,
                "calibration_mode": calibration_mode,
                "reward_mode": reward_mode,
                "lambda_waste": lambda_waste,
                "episode_index": episode_index,
                "day": day,
                "action": action,
                "markdown": markdown,
                "reward": float(reward),
                "financial_component": financial_component,
                "expired_units": expired,
                "normalized_expired_units": normalized_expired,
                "expected_waste_penalty_component": expected_penalty,
                "expected_sustainability_reward": financial_component + expected_penalty,
                "step_conservation_error": safe_float(info.get("step_conservation_error")),
                "profit_identity_error": safe_float(info.get("profit_identity_error")),
            }
        )
        action_counts[action] += 1
        episode_return += float(reward)
        final_info = info
        day += 1
    total_actions = sum(action_counts.values())
    result = {
        "scenario_id": scenario_id,
        "policy": policy,
        "calibration_mode": calibration_mode,
        "reward_mode": reward_mode,
        "lambda_waste": lambda_waste,
        "episode_index": episode_index,
        "initial_inventory": initial_inventory,
        "normalized_accounting_profit": safe_float(final_info.get("normalized_accounting_profit")),
        "raw_accounting_profit": safe_float(final_info.get("accounting_profit")),
        "episode_return": episode_return,
        "physical_waste_units": safe_float(final_info.get("cumulative_waste")),
        "waste_rate": safe_float(final_info.get("final_waste_rate")),
        "sell_through_rate": safe_float(final_info.get("final_sell_through_rate")),
        "terminal_inventory": safe_float(final_info.get("terminal_inventory")),
        "average_markdown": safe_float(final_info.get("average_markdown")),
        "maximum_markdown": max_markdown,
        "first_markdown_day": first_markdown_day,
        "episode_length": total_actions,
        "stockout_rate": safe_float(final_info.get("stockout_indicator")),
        "episode_conservation_error": safe_float(final_info.get("episode_conservation_error")),
        "raw_financial_sum_error": safe_float(final_info.get("raw_financial_sum_error")),
        "action_entropy": action_entropy(action_counts),
        **{f"action_{action}_share": action_counts[action] / max(total_actions, 1) for action in ACTION_MARKDOWNS},
    }
    return result, step_rows


def evaluate_fixed_policies(core_grid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    episode_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    for _, scenario in core_grid.iterrows():
        scenario_id = str(scenario["scenario_id"])
        print(f"Evaluating scenario {scenario_id}", flush=True)
        env = OperationalPerishablePricingEnv(
            split="validation",
            calibration_mode="recovered_calibration",
            reward_mode="financial",
            lambda_waste=0.10,
            scenario_id=scenario_id,
            deterministic_demand=True,
            random_seed=10_000,
            promotion_context_mode="derived_from_action",
        )
        for policy in POLICIES:
            for episode_index in range(EPISODES_PER_SCENARIO):
                try:
                    result, steps = evaluate_episode(
                        scenario_id=scenario_id,
                        policy=policy,
                        calibration_mode="recovered_calibration",
                        episode_index=episode_index,
                        reward_mode="financial",
                        lambda_waste=0.10,
                        env=env,
                    )
                    result.update(scenario.to_dict())
                    episode_rows.append(result)
                    step_rows.extend(steps)
                except Exception as exc:  # noqa: BLE001
                    failed = {
                        "scenario_id": scenario_id,
                        "policy": policy,
                        "calibration_mode": "recovered_calibration",
                        "episode_index": episode_index,
                        "mechanical_failure": True,
                        "failure_message": str(exc),
                        **scenario.to_dict(),
                    }
                    episode_rows.append(failed)
    return pd.DataFrame(episode_rows), pd.DataFrame(step_rows)


def summarize_fixed_policy(episode_df: pd.DataFrame) -> pd.DataFrame:
    if "mechanical_failure" in episode_df.columns:
        valid_mask = ~episode_df["mechanical_failure"].fillna(False).astype(bool)
    else:
        valid_mask = pd.Series(True, index=episode_df.index)
    valid = episode_df.loc[valid_mask].copy()
    metrics = [
        "normalized_accounting_profit",
        "raw_accounting_profit",
        "physical_waste_units",
        "waste_rate",
        "sell_through_rate",
        "terminal_inventory",
        "average_markdown",
        "first_markdown_day",
        "episode_length",
        "stockout_rate",
        "episode_conservation_error",
        "raw_financial_sum_error",
    ]
    summary = valid.groupby(
        [
            "scenario_id",
            "policy",
            "shelf_life_class",
            "inventory_coverage",
            "age_profile",
            "margin_scenario",
            "disposal_scenario",
        ],
        dropna=False,
    )[metrics].mean().reset_index()
    return summary


def pareto_flags(group: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in group.iterrows():
        dominated = False
        for _, other in group.iterrows():
            if row["policy"] == other["policy"]:
                continue
            better_or_equal = (
                other["normalized_accounting_profit"] >= row["normalized_accounting_profit"] - EPS
                and other["waste_rate"] <= row["waste_rate"] + EPS
            )
            strictly = (
                other["normalized_accounting_profit"] > row["normalized_accounting_profit"] + EPS
                or other["waste_rate"] < row["waste_rate"] - EPS
            )
            if better_or_equal and strictly:
                dominated = True
                break
        rows.append(not dominated)
    out = group.copy()
    out["non_dominated"] = rows
    return out


def classify_scenarios(summary_df: pd.DataFrame, core_grid: pd.DataFrame, sensitivity_grid: pd.DataFrame, stress_grid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tradeoff = summary_df.groupby("scenario_id", group_keys=False).apply(pareto_flags).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for scenario_id, group in summary_df.groupby("scenario_id"):
        zero = group.loc[group["policy"].eq("always_0pct")]
        always_zero_waste = float(zero["waste_rate"].iloc[0]) if not zero.empty else np.nan
        min_waste = float(group["waste_rate"].min())
        max_waste = float(group["waste_rate"].max())
        max_reduction = max(always_zero_waste - min_waste, 0.0) if np.isfinite(always_zero_waste) else 0.0
        min_waste_policy = group.sort_values(["waste_rate", "normalized_accounting_profit"], ascending=[True, False]).iloc[0]
        zero_profit = float(zero["normalized_accounting_profit"].iloc[0]) if not zero.empty else np.nan
        profit_loss = zero_profit - float(min_waste_policy["normalized_accounting_profit"]) if np.isfinite(zero_profit) else np.nan
        corr = group[["normalized_accounting_profit", "waste_rate"]].corr().iloc[0, 1]
        corr = float(corr) if np.isfinite(corr) else 0.0
        non_dominated_count = int(tradeoff.loc[tradeoff["scenario_id"].eq(scenario_id), "non_dominated"].sum())
        policy_rank_profit = list(group.sort_values("normalized_accounting_profit", ascending=False)["policy"])
        policy_rank_waste = list(group.sort_values("waste_rate", ascending=True)["policy"])
        mechanical_invalid = bool(
            (group["episode_conservation_error"].abs().max() > 1e-6)
            or (group["raw_financial_sum_error"].abs().max() > 1e-6)
            or group[["normalized_accounting_profit", "waste_rate"]].isna().any().any()
        )
        if mechanical_invalid:
            classification = "MECHANICALLY_INVALID"
        elif max_waste <= 1e-6:
            classification = "NO_WASTE_OPPORTUNITY"
        elif non_dominated_count <= 1:
            classification = "TRIVIAL_WASTE_PROBLEM"
        elif max_reduction > 1e-4 and abs(profit_loss) > 1e-4 and policy_rank_profit != policy_rank_waste:
            classification = "MEANINGFUL_PROFIT_WASTE_TRADEOFF"
        else:
            classification = "TRIVIAL_WASTE_PROBLEM"
        meta = core_grid.loc[core_grid["scenario_id"].eq(scenario_id)].iloc[0].to_dict()
        rows.append(
            {
                **meta,
                "scenarios_audited_source": "core_grid",
                "always_zero_waste_rate": always_zero_waste,
                "minimum_waste_rate": min_waste,
                "maximum_waste_rate": max_waste,
                "maximum_waste_reduction_vs_always_zero": max_reduction,
                "profit_loss_at_maximum_waste_reduction": profit_loss,
                "non_dominated_policy_count": non_dominated_count,
                "profit_waste_correlation": corr,
                "policy_ranking_changes_between_profit_and_waste": policy_rank_profit != policy_rank_waste,
                "classification": classification,
            }
        )
    # The current environment only supports CORE_GRID_PATH. Record unsupported existing grids explicitly.
    for name, grid in [("sensitivity_grid", sensitivity_grid), ("stress_grid", stress_grid)]:
        if grid.empty:
            rows.append(
                {
                    "scenario_id": f"{name}:missing",
                    "scenarios_audited_source": name,
                    "classification": "NOT_AUDITED_GRID_MISSING",
                    "note": f"{name} file not available.",
                }
            )
        else:
            for _, scenario in grid.iterrows():
                rows.append(
                    {
                        **scenario.to_dict(),
                        "scenarios_audited_source": name,
                        "classification": "NOT_AUDITED_UNSUPPORTED_BY_ENV_INTERFACE",
                        "note": "Operational environment loads perishability_core_scenario_grid.csv only; scenario_id is unsupported without modifying environment.",
                    }
                )
    return pd.DataFrame(rows), tradeoff


def find_waste_episode(scenario_id: str, policy: str) -> int:
    for episode_index in range(EPISODES_PER_SCENARIO):
        result, _ = evaluate_episode(
            scenario_id=scenario_id,
            policy=policy,
            calibration_mode="recovered_calibration",
            episode_index=episode_index,
            reward_mode="financial",
            lambda_waste=0.0,
        )
        if result["waste_rate"] > 1e-9:
            return episode_index
    return 0


def reward_wiring_and_lambda_audit(audit_df: pd.DataFrame, fixed_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    meaningful = audit_df.loc[audit_df["classification"].eq("MEANINGFUL_PROFIT_WASTE_TRADEOFF")]
    wiring_rows: list[dict[str, Any]] = []
    lambda_rows: list[dict[str, Any]] = []
    for _, scenario in meaningful.iterrows():
        scenario_id = str(scenario["scenario_id"])
        scenario_policies = fixed_summary.loc[fixed_summary["scenario_id"].eq(scenario_id)].copy()
        max_waste_policy = str(scenario_policies.sort_values("waste_rate", ascending=False).iloc[0]["policy"])
        audited_policies = list(dict.fromkeys(["always_0pct", max_waste_policy, "expiry_threshold_rule", "always_20pct"]))
        for policy in audited_policies:
            audit_episode_index = find_waste_episode(scenario_id, policy)
            financial_result, financial_steps = evaluate_episode(
                scenario_id=scenario_id,
                policy=policy,
                calibration_mode="recovered_calibration",
                episode_index=audit_episode_index,
                reward_mode="financial",
                lambda_waste=0.0,
            )
            for lambda_value in LAMBDA_VALUES:
                sustain_result, sustain_steps = evaluate_episode(
                    scenario_id=scenario_id,
                    policy=policy,
                    calibration_mode="recovered_calibration",
                    episode_index=audit_episode_index,
                    reward_mode="sustainability",
                    lambda_waste=lambda_value,
                )
                steps = pd.DataFrame(sustain_steps)
                expired_positive = bool((steps["expired_units"] > 1e-9).any()) if not steps.empty else False
                penalty_valid = bool(
                    np.allclose(
                        steps["expected_waste_penalty_component"],
                        -lambda_value * steps["normalized_expired_units"],
                        atol=1e-9,
                    )
                ) if not steps.empty else False
                expected_return = sustain_result["normalized_accounting_profit"] - lambda_value * sustain_result["waste_rate"]
                return_identity_ok = abs(sustain_result["episode_return"] - expected_return) <= 1e-6
                differs_from_financial = abs(sustain_result["episode_return"] - financial_result["episode_return"]) > 1e-9
                if sustain_result["waste_rate"] <= 1e-9:
                    status = "NO_WASTE_SIGNAL"
                elif not penalty_valid or not return_identity_ok:
                    status = "WASTE_GENERATED_BUT_PENALTY_NOT_WIRED"
                elif not differs_from_financial:
                    status = "DIAGNOSTIC_AGGREGATION_ERROR"
                else:
                    status = "SUSTAINABILITY_SIGNAL_VALID"
                wiring_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "policy": policy,
                        "episode_index": audit_episode_index,
                        "lambda_waste": lambda_value,
                        "expired_units_positive": expired_positive,
                        "waste_rate": sustain_result["waste_rate"],
                        "penalty_formula_valid": penalty_valid,
                        "penalty_negative_when_waste_occurs": bool((steps.loc[steps["expired_units"] > 1e-9, "expected_waste_penalty_component"] < 0).all()) if expired_positive else False,
                        "sustainability_differs_from_financial": differs_from_financial,
                        "episode_return_identity_ok": return_identity_ok,
                        "reward_signal_status": status,
                    }
                )
        for lambda_value in LAMBDA_VALUES:
            scenario_policies["sustainability_objective"] = (
                scenario_policies["normalized_accounting_profit"] - lambda_value * scenario_policies["waste_rate"]
            )
            best_policy = str(scenario_policies.sort_values("sustainability_objective", ascending=False).iloc[0]["policy"])
            financial_abs = float(scenario_policies["normalized_accounting_profit"].abs().mean())
            penalty_abs = float((lambda_value * scenario_policies["waste_rate"]).abs().mean())
            total_abs = financial_abs + penalty_abs
            waste_share = penalty_abs / total_abs if total_abs > EPS else 0.0
            ranking = list(scenario_policies.sort_values("sustainability_objective", ascending=False)["policy"])
            dominant_40 = best_policy == "always_40pct"
            if waste_share < 0.10:
                lambda_class = "TOO_WEAK"
            elif waste_share > 0.40 or dominant_40:
                lambda_class = "TOO_DOMINANT"
            else:
                lambda_class = "USABLE"
            lambda_rows.append(
                {
                    "scenario_id": scenario_id,
                    "lambda_waste": lambda_value,
                    "mean_abs_financial_component": financial_abs,
                    "mean_abs_waste_penalty_component": penalty_abs,
                    "waste_share_of_total_abs_reward": waste_share,
                    "best_fixed_policy": best_policy,
                    "fixed_policy_ranking": "|".join(ranking),
                    "maximum_markdown_dominant": dominant_40,
                    "lambda_classification": lambda_class,
                }
            )
    wiring = pd.DataFrame(wiring_rows)
    lambda_audit = pd.DataFrame(lambda_rows)
    return wiring, lambda_audit


def select_sustainability_scenario(audit: pd.DataFrame, wiring: pd.DataFrame, lambda_audit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    meaningful = audit.loc[audit["classification"].eq("MEANINGFUL_PROFIT_WASTE_TRADEOFF")].copy()
    for _, scenario in meaningful.iterrows():
        scenario_id = str(scenario["scenario_id"])
        wiring_valid = (
            not wiring.empty
            and wiring.loc[wiring["scenario_id"].eq(scenario_id), "reward_signal_status"].eq("SUSTAINABILITY_SIGNAL_VALID").any()
        )
        usable = lambda_audit.loc[
            lambda_audit["scenario_id"].eq(scenario_id)
            & lambda_audit["lambda_classification"].eq("USABLE")
        ] if not lambda_audit.empty else pd.DataFrame()
        selected_lambda = float(usable.sort_values("lambda_waste").iloc[0]["lambda_waste"]) if not usable.empty else np.nan
        eligible = wiring_valid and np.isfinite(selected_lambda)
        rows.append(
            {
                "scenario_id": scenario_id,
                "eligible": eligible,
                "scenario_classification": scenario["classification"],
                "reward_wiring_valid": wiring_valid,
                "selected_lambda": selected_lambda,
                "always_zero_waste_rate": scenario["always_zero_waste_rate"],
                "maximum_waste_reduction_vs_always_zero": scenario["maximum_waste_reduction_vs_always_zero"],
                "profit_loss_at_maximum_waste_reduction": scenario["profit_loss_at_maximum_waste_reduction"],
                "selection_score": scenario["maximum_waste_reduction_vs_always_zero"] - abs(scenario["profit_loss_at_maximum_waste_reduction"]) * 0.1,
            }
        )
    selection = pd.DataFrame(rows)
    if not selection.empty:
        selection["selected_primary"] = False
        eligible = selection.loc[selection["eligible"]].sort_values("selection_score", ascending=False)
        if not eligible.empty:
            selection.loc[selection["scenario_id"].eq(eligible.iloc[0]["scenario_id"]), "selected_primary"] = True
    return selection


def save_config(selection: pd.DataFrame, status: str) -> dict[str, Any]:
    config: dict[str, Any] = {
        "status": status,
        "validation_split_only": True,
        "test_split_used": False,
        "ppo_training_requested": False,
        "environment_modified": False,
    }
    if not selection.empty and selection["selected_primary"].any():
        row = selection.loc[selection["selected_primary"]].iloc[0]
        config.update(
            {
                "selected_scenario_id": row["scenario_id"],
                "selected_lambda_waste": None if pd.isna(row["selected_lambda"]) else float(row["selected_lambda"]),
                "recommended_next_action": "Run sustainability PPO pilot only after reviewing this audit.",
            }
        )
    else:
        config.update(
            {
                "selected_scenario_id": None,
                "selected_lambda_waste": None,
                "recommended_next_action": "Do not run sustainability PPO; use empirical profit-waste analysis or audit unsupported scenario interfaces.",
            }
        )
    path = CONFIGS_DIR / "sustainability_extension_config.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def make_figures(fixed: pd.DataFrame, tradeoff: pd.DataFrame, wiring: pd.DataFrame, lambda_audit: pd.DataFrame, selection: pd.DataFrame) -> None:
    candidate = fixed.copy()
    for metric, filename, ylabel in [
        ("waste_rate", "waste_rate_by_scenario_and_policy.png", "Waste rate"),
        ("normalized_accounting_profit", "normalized_profit_by_scenario_and_policy.png", "Normalized profit"),
    ]:
        fig, ax = plt.subplots(figsize=(13, 5))
        pivot = candidate.pivot_table(index="scenario_id", columns="policy", values=metric, aggfunc="mean").fillna(0)
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=80)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / filename, dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(candidate["normalized_accounting_profit"], candidate["waste_rate"], s=20)
    ax.set_xlabel("Normalized profit")
    ax.set_ylabel("Waste rate")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "profit_versus_waste_scatter_candidate_scenarios.png", dpi=160)
    plt.close(fig)
    if not tradeoff.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        scenario_summary = tradeoff.groupby("scenario_id").agg(
            min_waste=("waste_rate", "min"),
            max_profit=("normalized_accounting_profit", "max"),
        ).reset_index()
        ax.bar(scenario_summary["scenario_id"], scenario_summary["min_waste"])
        ax.set_ylabel("Minimum waste rate")
        ax.tick_params(axis="x", rotation=80)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "waste_reduction_versus_profit_sacrifice.png", dpi=160)
        plt.close(fig)
    if not lambda_audit.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        lambda_audit.groupby("lambda_waste")[["mean_abs_financial_component", "mean_abs_waste_penalty_component"]].mean().plot(kind="bar", ax=ax)
        ax.set_ylabel("Mean absolute component")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "reward_component_magnitude_by_lambda.png", dpi=160)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 4))
        counts = lambda_audit.groupby(["lambda_waste", "best_fixed_policy"]).size().unstack(fill_value=0)
        counts.plot(kind="bar", stacked=True, ax=ax)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "fixed_policy_ranking_by_lambda.png", dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    if not selection.empty:
        ax.bar(selection["scenario_id"], selection["maximum_waste_reduction_vs_always_zero"].fillna(0))
    ax.set_ylabel("Max waste reduction vs always-zero")
    ax.tick_params(axis="x", rotation=80)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "selected_sustainability_scenario_overview.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    metadata = {
        "thresholds": {
            "negligible_waste_rate": 1e-6,
            "meaningful_waste_reduction": 1e-4,
            "meaningful_profit_tradeoff": 1e-4,
            "usable_lambda_waste_share_low": 0.10,
            "usable_lambda_waste_share_high": 0.40,
        },
        "episodes_per_scenario_policy": EPISODES_PER_SCENARIO,
        "policies": POLICIES,
        "lambda_values": LAMBDA_VALUES,
        "validation_split_only": True,
        "test_split_used": False,
        "scenario_interface_note": "OperationalPerishablePricingEnv currently loads perishability_core_scenario_grid.csv only.",
    }
    (CONFIGS_DIR / "sustainability_audit_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    core, sensitivity, stress = read_scenario_grids()
    print(f"Evaluating {len(core)} core scenarios x {len(POLICIES)} policies x {EPISODES_PER_SCENARIO} episodes")
    episodes, steps = evaluate_fixed_policies(core)
    fixed_summary = summarize_fixed_policy(episodes)
    scenario_audit, tradeoff = classify_scenarios(fixed_summary, core, sensitivity, stress)
    wiring, lambda_audit = reward_wiring_and_lambda_audit(scenario_audit, fixed_summary)
    selection = select_sustainability_scenario(scenario_audit, wiring, lambda_audit)
    meaningful_exists = scenario_audit["classification"].eq("MEANINGFUL_PROFIT_WASTE_TRADEOFF").any()
    mechanical_invalid = scenario_audit["classification"].eq("MECHANICALLY_INVALID").any()
    selected_ready = not selection.empty and selection["selected_primary"].any()
    if mechanical_invalid:
        status = "SUSTAINABILITY_AUDIT_REQUIRES_REVISION"
    elif selected_ready:
        status = "SUSTAINABILITY_EXTENSION_READY_FOR_PILOT"
    elif meaningful_exists:
        status = "SUSTAINABILITY_ANALYSIS_ONLY"
    else:
        status = "SUSTAINABILITY_NOT_SUPPORTED_BY_CURRENT_SCENARIOS"
    config = save_config(selection, status)
    scenario_audit.to_csv(TABLES_DIR / "sustainability_scenario_audit.csv", index=False)
    fixed_summary.to_csv(TABLES_DIR / "sustainability_fixed_policy_evaluation.csv", index=False)
    tradeoff.to_csv(TABLES_DIR / "sustainability_profit_waste_tradeoff.csv", index=False)
    wiring.to_csv(TABLES_DIR / "sustainability_reward_wiring_audit.csv", index=False)
    lambda_audit.to_csv(TABLES_DIR / "sustainability_lambda_audit.csv", index=False)
    selection.to_csv(TABLES_DIR / "sustainability_scenario_selection.csv", index=False)
    episodes.to_csv(TABLES_DIR / "sustainability_fixed_policy_episode_level.csv", index=False)
    make_figures(fixed_summary, tradeoff, wiring, lambda_audit, selection)
    audited = scenario_audit.loc[scenario_audit["scenarios_audited_source"].eq("core_grid")]
    report = {
        "status": status,
        "scenarios_audited": int(audited.shape[0]),
        "scenario_classifications": audited["classification"].value_counts().to_dict(),
        "selected_sustainability_scenario": config.get("selected_scenario_id"),
        "selected_lambda": config.get("selected_lambda_waste"),
        "always_zero_waste_rate_max": float(audited["always_zero_waste_rate"].max()) if "always_zero_waste_rate" in audited else None,
        "maximum_achievable_waste_reduction": float(audited["maximum_waste_reduction_vs_always_zero"].max()) if "maximum_waste_reduction_vs_always_zero" in audited else None,
        "associated_profit_sacrifice": float(audited.sort_values("maximum_waste_reduction_vs_always_zero", ascending=False).iloc[0]["profit_loss_at_maximum_waste_reduction"]) if not audited.empty else None,
        "reward_wiring_status_counts": wiring["reward_signal_status"].value_counts().to_dict() if not wiring.empty else {},
        "files_created": [
            "outputs/tables/sustainability_scenario_audit.csv",
            "outputs/tables/sustainability_fixed_policy_evaluation.csv",
            "outputs/tables/sustainability_profit_waste_tradeoff.csv",
            "outputs/tables/sustainability_reward_wiring_audit.csv",
            "outputs/tables/sustainability_lambda_audit.csv",
            "outputs/tables/sustainability_scenario_selection.csv",
            "outputs/configs/sustainability_extension_config.json",
            "outputs/configs/sustainability_audit_metadata.json",
        ],
        "recommended_next_action": config["recommended_next_action"],
    }
    (TABLES_DIR / "sustainability_opportunity_audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(status)


if __name__ == "__main__":
    main()
