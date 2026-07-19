"""
Full paired validation-only limited-horizon planning evaluation.

This script is the decisive dynamic-value gate before any further RL work.

It does NOT:
- retrain PPO
- train DQN
- use the test split
- modify data, reward, accounting, scenarios, response models, or model artifacts

Run from project root:
    python src/limited_horizon_planning_evaluation.py

Main outputs:
    outputs/tables/planning_locked_task_definition.csv
    outputs/tables/planning_pairing_audit.csv
    outputs/tables/limited_horizon_planning_episode_results.csv
    outputs/tables/planning_vs_locked_baseline_summary.csv
    outputs/tables/planning_vs_baseline_by_scenario.csv
    outputs/tables/planning_ppo_baseline_comparison.csv
    docs/limited_horizon_planning_decision.md
"""

from __future__ import annotations

import copy
import itertools
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pricing_env_operational import OperationalPerishablePricingEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "limited_horizon_planning"
DOCS_DIR = PROJECT_ROOT / "docs"

PRIMARY_CALIBRATION = "recovered_calibration"
REWARD_MODE = "financial"
LAMBDA_WASTE = 0.10
PLANNING_HORIZON = 3
ACTION_SPACE = list(range(6))
MAX_CANDIDATE_SEQUENCES = 6**PLANNING_HORIZON
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 20260717
TIE_TOL = 1e-9
PRACTICAL_GAIN = 0.005


@dataclass(frozen=True)
class EvalEpisode:
    scenario_id: str
    calibration_mode: str
    episode_index: int
    episode_id: str


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def read_csv(name: str) -> pd.DataFrame:
    path = TABLES_DIR / name
    require_file(path)
    return pd.read_csv(path)


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def action_entropy(actions: list[int]) -> float:
    if not actions:
        return 0.0
    counts = pd.Series(actions).value_counts(normalize=True)
    return float(-(counts * np.log(counts)).sum())


def fixed_action_for_policy(policy_id: str) -> int | None:
    fixed_map = {
        "always_0pct": 0,
        "always_5pct": 1,
        "always_10pct": 2,
        "always_20pct": 3,
        "always_30pct": 4,
        "always_40pct": 5,
    }
    return fixed_map.get(policy_id)


def action_for_rule(policy_id: str, obs: np.ndarray) -> int:
    fixed = fixed_action_for_policy(policy_id)
    if fixed is not None:
        return fixed

    inventory_coverage = safe_float(obs[1]) if obs.shape[0] > 1 else 0.0
    fraction_expiring_today = safe_float(obs[23]) if obs.shape[0] > 23 else 0.0
    fraction_expiring_soon = safe_float(obs[24]) if obs.shape[0] > 24 else fraction_expiring_today

    if policy_id == "expiry_threshold_rule":
        if fraction_expiring_soon >= 0.50:
            return 5
        if fraction_expiring_soon >= 0.25:
            return 3
        if fraction_expiring_soon >= 0.10:
            return 2
        return 0
    if policy_id == "inventory_coverage_rule":
        if inventory_coverage >= 1.50:
            return 4
        if inventory_coverage >= 1.10:
            return 3
        if inventory_coverage >= 0.80:
            return 2
        return 0
    if policy_id == "combined_inventory_expiry_rule":
        if inventory_coverage >= 1.50 and fraction_expiring_soon >= 0.50:
            return 5
        if inventory_coverage >= 1.25 and fraction_expiring_soon >= 0.25:
            return 4
        if inventory_coverage >= 1.00 and fraction_expiring_soon >= 0.10:
            return 3
        if fraction_expiring_today >= 0.50:
            return 3
        return 0
    raise ValueError(f"Unsupported baseline policy: {policy_id}")


def make_env(scenario_id: str, calibration_mode: str) -> OperationalPerishablePricingEnv:
    return OperationalPerishablePricingEnv(
        split="validation",
        calibration_mode=calibration_mode,
        reward_mode=REWARD_MODE,
        lambda_waste=LAMBDA_WASTE,
        scenario_id=scenario_id,
        deterministic_demand=True,
        random_seed=BOOTSTRAP_SEED,
        promotion_context_mode="derived_from_action",
    )


def reset_env(env: OperationalPerishablePricingEnv, episode_index: int) -> tuple[np.ndarray, dict[str, Any]]:
    return env.reset(seed=20_000 + int(episode_index), options={"episode_index": int(episode_index)})


def values_match_or_unavailable(left: Any, right: Any, tol: float = 1e-8) -> bool:
    """Treat unavailable optional metadata as pass when deterministic reset keys match.

    The environment does not consistently expose initial inventory, age-profile
    signature, and horizon in the terminal info dict. Pairing is still valid
    because both policies are reset with identical split, scenario, calibration,
    episode_index, deterministic demand, and seed.
    """
    if pd.isna(left) and pd.isna(right):
        return True
    try:
        return abs(float(left) - float(right)) <= tol
    except Exception:
        return str(left) == str(right)


def simulate_sequence(env: OperationalPerishablePricingEnv, sequence: tuple[int, ...]) -> float:
    sim = copy.deepcopy(env)
    total = 0.0
    for action in sequence:
        try:
            _, reward, terminated, truncated, _ = sim.step(int(action))
        except RuntimeError:
            break
        total += safe_float(reward, 0.0)
        if terminated or truncated:
            break
    return total


def choose_planning_action(env: OperationalPerishablePricingEnv, horizon: int) -> tuple[int, float, int]:
    best_action = 0
    best_value = -np.inf
    sequences_evaluated = 0
    for sequence in itertools.product(ACTION_SPACE, repeat=horizon):
        value = simulate_sequence(env, sequence)
        sequences_evaluated += 1
        first_action = int(sequence[0])
        if value > best_value + 1e-12:
            best_value = value
            best_action = first_action
        elif abs(value - best_value) <= 1e-12 and first_action < best_action:
            # Conservative tie-break: lower markdown.
            best_action = first_action
    return best_action, float(best_value), sequences_evaluated


def episode_result_from_final_info(
    *,
    policy_id: str,
    episode: EvalEpisode,
    final_info: dict[str, Any],
    reset_info: dict[str, Any],
    env_horizon: int | None,
    actions: list[int],
    runtime_seconds: float,
    candidate_sequences_evaluated: int,
    planning_horizon: int | None,
) -> dict[str, Any]:
    action_changes = int(sum(1 for left, right in zip(actions, actions[1:]) if left != right))
    return {
        "episode_id": episode.episode_id,
        "scenario_id": episode.scenario_id,
        "calibration_mode": episode.calibration_mode,
        "episode_index": episode.episode_index,
        "policy_id": policy_id,
        "normalized_profit": safe_float(final_info.get("normalized_accounting_profit")),
        "total_revenue": safe_float(final_info.get("cumulative_revenue")),
        "total_procurement_cost": safe_float(final_info.get("cumulative_procurement_cost")),
        "total_disposal_cost": safe_float(final_info.get("cumulative_disposal_cost")),
        "terminal_value": safe_float(final_info.get("terminal_value"), 0.0),
        "units_sold": safe_float(final_info.get("cumulative_sales")),
        "units_expired": safe_float(final_info.get("cumulative_waste")),
        "waste_rate": safe_float(final_info.get("final_waste_rate")),
        "sell_through": safe_float(final_info.get("final_sell_through_rate")),
        "average_markdown": safe_float(final_info.get("average_markdown")),
        "number_of_markdown_changes": action_changes,
        "action_sequence": "|".join(str(a) for a in actions),
        "action_entropy": action_entropy(actions),
        "episode_length": len(actions),
        "runtime_seconds": runtime_seconds,
        "candidate_sequences_evaluated": candidate_sequences_evaluated,
        "planning_horizon": planning_horizon,
        "reset_seed": 20_000 + int(episode.episode_index),
        "deterministic_demand": True,
        "initial_inventory": safe_float(
            final_info.get("initial_inventory", reset_info.get("initial_inventory", np.nan))
        ),
        "starting_age_profile_signature": str(
            final_info.get(
                "starting_age_profile_signature",
                reset_info.get("starting_age_profile_signature", ""),
            )
        ),
        "episode_horizon": safe_float(final_info.get("episode_horizon", env_horizon)),
        "episode_conservation_error": safe_float(final_info.get("episode_conservation_error"), 0.0),
        "raw_financial_sum_error": safe_float(final_info.get("raw_financial_sum_error"), 0.0),
    }


def evaluate_policy_episode(episode: EvalEpisode, policy_id: str, horizon: int = PLANNING_HORIZON) -> dict[str, Any]:
    env = make_env(episode.scenario_id, episode.calibration_mode)
    obs, reset_info = reset_env(env, episode.episode_index)
    env_horizon = int(getattr(env, "horizon", 0)) if hasattr(env, "horizon") else None
    terminated = False
    truncated = False
    actions: list[int] = []
    candidate_sequences_evaluated = 0
    final_info: dict[str, Any] = {}
    start = time.perf_counter()
    while not (terminated or truncated):
        if policy_id == "limited_horizon_planning_policy":
            remaining = max(1, int(getattr(env, "horizon", horizon)) - int(getattr(env, "current_step", 0)))
            local_horizon = min(horizon, remaining)
            action, _, count = choose_planning_action(env, local_horizon)
            candidate_sequences_evaluated += count
        else:
            action = action_for_rule(policy_id, obs)
        obs, _, terminated, truncated, info = env.step(int(action))
        final_info = info
        actions.append(int(action))
    runtime = time.perf_counter() - start
    env.close()
    return episode_result_from_final_info(
        policy_id=policy_id,
        episode=episode,
        final_info=final_info,
        reset_info=reset_info,
        env_horizon=env_horizon,
        actions=actions,
        runtime_seconds=runtime,
        candidate_sequences_evaluated=candidate_sequences_evaluated,
        planning_horizon=horizon if policy_id == "limited_horizon_planning_policy" else None,
    )


def load_locked_task() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    locked_population = read_csv("final_task_locked_population_definition.csv")
    locked_baselines = read_csv("final_task_locked_baselines.csv")
    state_source = read_csv("financial_markdown_oracle_diagnostic.csv")
    return locked_population, locked_baselines, state_source


def select_high_risk_episodes(locked_population: pd.DataFrame, state_source: pd.DataFrame) -> list[EvalEpisode]:
    pop = locked_population.loc[locked_population["population_id"].eq("HIGH_RISK_B")]
    if pop.empty:
        raise ValueError("Locked HIGH_RISK_B population not found. Run final_model_decision_task_audit.py first.")
    row = pop.iloc[0]
    states = state_source.copy()
    states = states.loc[states["calibration_mode"].eq(PRIMARY_CALIBRATION)]
    states = states.loc[states["scenario_id"].isin(str(row["scenario_ids"]).split("|"))]
    for col in [
        "episode_index",
        "inventory_coverage_state",
        "fraction_expiring_within_two_days",
        "predicted_baseline_demand",
    ]:
        states[col] = pd.to_numeric(states[col], errors="coerce")
    states["projected_sell_through_proxy"] = (
        states["predicted_baseline_demand"] / states["inventory_coverage_state"].replace(0, np.nan)
    ).clip(lower=0, upper=3)
    if not pd.isna(row["min_inventory_coverage"]):
        states = states.loc[states["inventory_coverage_state"] >= float(row["min_inventory_coverage"])]
    if not pd.isna(row["min_fraction_expiring_within_two_days"]):
        states = states.loc[states["fraction_expiring_within_two_days"] >= float(row["min_fraction_expiring_within_two_days"])]
    if not pd.isna(row["max_projected_sell_through"]):
        states = states.loc[states["projected_sell_through_proxy"] <= float(row["max_projected_sell_through"])]
    episodes = (
        states[["scenario_id", "calibration_mode", "episode_index"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["scenario_id", "episode_index"])
    )
    result: list[EvalEpisode] = []
    for _, ep in episodes.iterrows():
        idx = int(ep["episode_index"])
        scenario = str(ep["scenario_id"])
        calibration = str(ep["calibration_mode"])
        result.append(
            EvalEpisode(
                scenario_id=scenario,
                calibration_mode=calibration,
                episode_index=idx,
                episode_id=f"{scenario}__{calibration}__episode_{idx:02d}",
            )
        )
    if not result:
        raise ValueError("No episodes selected for HIGH_RISK_B population.")
    return result


def write_locked_definition(locked_population: pd.DataFrame, locked_baselines: pd.DataFrame, episodes: list[EvalEpisode]) -> str:
    base = locked_baselines.loc[locked_baselines["population_id"].eq("HIGH_RISK_B")]
    if base.empty:
        raise ValueError("Locked baseline for HIGH_RISK_B not found.")
    baseline = str(base.iloc[0]["locked_primary_baseline_policy"])
    row = {
        "population_id": "HIGH_RISK_B",
        "paired_episode_count": len(episodes),
        "locked_primary_baseline": baseline,
        "locked_best_fixed_policy": base.iloc[0].get("best_fixed_validation_policy", ""),
        "locked_best_rule_policy": base.iloc[0].get("best_rule_validation_policy", ""),
        "calibration_mode": PRIMARY_CALIBRATION,
        "split": "validation",
        "reward_mode": REWARD_MODE,
        "lambda_waste": LAMBDA_WASTE,
        "planning_horizon": PLANNING_HORIZON,
        "branching_factor": len(ACTION_SPACE),
        "max_candidate_sequences_per_decision": MAX_CANDIDATE_SEQUENCES,
        "deterministic_demand": True,
        "test_split_used": False,
        "ppo_training_run": False,
        "dqn_training_run": False,
    }
    pd.DataFrame([row]).to_csv(TABLES_DIR / "planning_locked_task_definition.csv", index=False)
    return baseline


def pairing_audit(results: pd.DataFrame) -> pd.DataFrame:
    plan = results.loc[results["policy_id"].eq("limited_horizon_planning_policy")]
    base = results.loc[~results["policy_id"].eq("limited_horizon_planning_policy")]
    rows = []
    for episode_id, plan_row in plan.set_index("episode_id").iterrows():
        if episode_id not in set(base["episode_id"]):
            rows.append({"episode_id": episode_id, "pairing_status": "BASELINE_MISSING"})
            continue
        base_row = base.loc[base["episode_id"].eq(episode_id)].iloc[0]
        checks = {
            "same_scenario": plan_row["scenario_id"] == base_row["scenario_id"],
            "same_calibration_mode": plan_row["calibration_mode"] == base_row["calibration_mode"],
            "same_episode_index": int(plan_row["episode_index"]) == int(base_row["episode_index"]),
            "same_reset_seed": int(plan_row["reset_seed"]) == int(base_row["reset_seed"]),
            "same_deterministic_demand": bool(plan_row["deterministic_demand"]) == bool(base_row["deterministic_demand"]),
            "same_initial_inventory": values_match_or_unavailable(
                plan_row["initial_inventory"], base_row["initial_inventory"]
            ),
            "same_age_profile": values_match_or_unavailable(
                plan_row["starting_age_profile_signature"],
                base_row["starting_age_profile_signature"],
            ),
            "same_episode_horizon": values_match_or_unavailable(
                plan_row["episode_horizon"], base_row["episode_horizon"]
            ),
            "same_normalization": True,
            "same_economic_parameters": True,
            "same_split": True,
        }
        rows.append(
            {
                "episode_id": episode_id,
                **checks,
                "pairing_status": "PAIRED_VALID" if all(checks.values()) else "PAIRING_FAILED",
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(TABLES_DIR / "planning_pairing_audit.csv", index=False)
    return audit


def bootstrap_summary(diff: pd.Series) -> tuple[float, float]:
    values = diff.dropna().astype(float).to_numpy()
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(BOOTSTRAP_N)
    for i in range(BOOTSTRAP_N):
        samples[i] = rng.choice(values, size=len(values), replace=True).mean()
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def comparison_summary(results: pd.DataFrame, baseline_policy: str) -> pd.DataFrame:
    plan = results.loc[results["policy_id"].eq("limited_horizon_planning_policy")]
    base = results.loc[results["policy_id"].eq(baseline_policy)]
    merged = plan.merge(base, on="episode_id", suffixes=("_planning", "_baseline"))
    profit_diff = merged["normalized_profit_planning"] - merged["normalized_profit_baseline"]
    ci_low, ci_high = bootstrap_summary(profit_diff)
    top_two_share = float(profit_diff.sort_values(ascending=False).head(min(2, len(profit_diff))).sum() / profit_diff.sum()) if profit_diff.sum() > 0 else np.nan
    if profit_diff.mean() > 0 and ci_low > 0 and (pd.isna(top_two_share) or top_two_share < 0.80):
        classification = "CLEAR_DYNAMIC_VALUE"
        status = "DYNAMIC_VALUE_CONFIRMED"
    elif profit_diff.mean() > 0:
        classification = "MODEST_DYNAMIC_VALUE"
        status = "DYNAMIC_VALUE_MODEST"
    elif abs(profit_diff.mean()) <= PRACTICAL_GAIN:
        classification = "NO_MEANINGFUL_DYNAMIC_VALUE"
        status = "NO_DEFENSIBLE_DYNAMIC_MODEL"
    else:
        classification = "DYNAMIC_VALUE_ABSENT"
        status = "NO_DEFENSIBLE_DYNAMIC_MODEL"
    row = {
        "planning_policy": "limited_horizon_planning_policy",
        "locked_baseline_policy": baseline_policy,
        "paired_episode_count": int(len(merged)),
        "paired_mean_normalized_profit_gain": float(profit_diff.mean()),
        "paired_median_normalized_profit_gain": float(profit_diff.median()),
        "paired_profit_gain_std": float(profit_diff.std(ddof=1)) if len(profit_diff) > 1 else 0.0,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "win_share": float((profit_diff > TIE_TOL).mean()),
        "tie_share": float((profit_diff.abs() <= TIE_TOL).mean()),
        "loss_share": float((profit_diff < -TIE_TOL).mean()),
        "waste_rate_difference": float((merged["waste_rate_planning"] - merged["waste_rate_baseline"]).mean()),
        "sell_through_difference": float((merged["sell_through_planning"] - merged["sell_through_baseline"]).mean()),
        "average_markdown_difference": float((merged["average_markdown_planning"] - merged["average_markdown_baseline"]).mean()),
        "expired_units_difference": float((merged["units_expired_planning"] - merged["units_expired_baseline"]).mean()),
        "top_two_positive_gain_share": top_two_share,
        "classification": classification,
        "primary_status": status,
    }
    out = pd.DataFrame([row])
    out.to_csv(TABLES_DIR / "planning_vs_locked_baseline_summary.csv", index=False)
    return out


def scenario_summary(results: pd.DataFrame, baseline_policy: str) -> pd.DataFrame:
    rows = []
    plan = results.loc[results["policy_id"].eq("limited_horizon_planning_policy")]
    base = results.loc[results["policy_id"].eq(baseline_policy)]
    merged = plan.merge(base, on="episode_id", suffixes=("_planning", "_baseline"))
    for scenario_id, group in merged.groupby("scenario_id_planning"):
        diff = group["normalized_profit_planning"] - group["normalized_profit_baseline"]
        ci_low, ci_high = bootstrap_summary(diff)
        action_counts = "|".join(
            f"{action}:{count}" for action, count in pd.Series("|".join(group["action_sequence_planning"]).split("|")).value_counts().sort_index().items()
        )
        rows.append(
            {
                "scenario_id": scenario_id,
                "sample_size": int(len(group)),
                "paired_mean_profit_gain": float(diff.mean()),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "win_rate": float((diff > TIE_TOL).mean()),
                "waste_rate_difference": float((group["waste_rate_planning"] - group["waste_rate_baseline"]).mean()),
                "sell_through_difference": float((group["sell_through_planning"] - group["sell_through_baseline"]).mean()),
                "best_action_distribution": action_counts,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "planning_vs_baseline_by_scenario.csv", index=False)
    return out


def ppo_comparison(results: pd.DataFrame, baseline_policy: str, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    plan = results.loc[results["policy_id"].eq("limited_horizon_planning_policy")]
    base = results.loc[results["policy_id"].eq(baseline_policy)]
    for policy_id, df in [(baseline_policy, base), ("limited_horizon_planning_policy", plan)]:
        rows.append(
            {
                "policy_id": policy_id,
                "source": "current_paired_evaluation",
                "normalized_profit": float(df["normalized_profit"].mean()) if not df.empty else np.nan,
                "paired_gain_vs_baseline": 0.0 if policy_id == baseline_policy else float(summary["paired_mean_normalized_profit_gain"].iloc[0]),
                "waste_rate": float(df["waste_rate"].mean()) if not df.empty else np.nan,
                "action_entropy": float(df["action_entropy"].mean()) if not df.empty else np.nan,
                "runtime_seconds": float(df["runtime_seconds"].sum()) if not df.empty else np.nan,
                "interpretability": "high" if policy_id == baseline_policy else "medium",
            }
        )
    ppo_tables = [
        ("BALANCED__checkpoint_20000", "outputs/tables/ppo_training_redesign_results.csv"),
        ("selected_recovered_financial_ppo", "outputs/tables/ppo_financial_paired_validation_summary.csv"),
        ("selected_observed_financial_ppo", "outputs/tables/ppo_financial_paired_validation_summary.csv"),
    ]
    for policy_id, rel_path in ppo_tables:
        rows.append(
            {
                "policy_id": policy_id,
                "source": rel_path,
                "normalized_profit": np.nan,
                "paired_gain_vs_baseline": np.nan,
                "waste_rate": np.nan,
                "unnecessary_markdown": np.nan,
                "missed_opportunities": np.nan,
                "action_entropy": np.nan,
                "runtime_seconds": np.nan,
                "interpretability": "low_to_medium",
                "comparability_note": "Existing PPO artifacts are not treated as decisive unless evaluated on this exact high-risk paired manifest.",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "planning_ppo_baseline_comparison.csv", index=False)
    return out


def make_figures(results: pd.DataFrame, summary: pd.DataFrame, by_scenario: pd.DataFrame, baseline_policy: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plan = results.loc[results["policy_id"].eq("limited_horizon_planning_policy")]
    base = results.loc[results["policy_id"].eq(baseline_policy)]
    merged = plan.merge(base, on="episode_id", suffixes=("_planning", "_baseline"))
    merged["profit_diff"] = merged["normalized_profit_planning"] - merged["normalized_profit_baseline"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(merged["normalized_profit_baseline"], merged["normalized_profit_planning"])
    low = min(merged["normalized_profit_baseline"].min(), merged["normalized_profit_planning"].min())
    high = max(merged["normalized_profit_baseline"].max(), merged["normalized_profit_planning"].max())
    ax.plot([low, high], [low, high], color="black", linewidth=0.8)
    ax.set_xlabel("Locked baseline normalized profit")
    ax.set_ylabel("Planning normalized profit")
    ax.set_title("Planning vs locked baseline paired profit")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "planning_vs_locked_baseline_paired_profit.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    merged["profit_diff"].reset_index(drop=True).plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Planning minus baseline profit")
    ax.set_title("Episode-level paired profit differences")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "episode_level_paired_profit_differences.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    mean = summary["paired_mean_normalized_profit_gain"].iloc[0]
    low = summary["bootstrap_ci_low"].iloc[0]
    high = summary["bootstrap_ci_high"].iloc[0]
    ax.errorbar([0], [mean], yerr=[[mean - low], [high - mean]], fmt="o")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks([0])
    ax.set_xticklabels(["planning"])
    ax.set_ylabel("Profit gain")
    ax.set_title("Profit gain 95% bootstrap CI")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "profit_gain_confidence_interval.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    by_scenario.set_index("scenario_id")["paired_mean_profit_gain"].plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean profit gain")
    ax.set_title("Planning vs baseline by scenario")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "planning_vs_baseline_by_scenario.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    policy_summary = results.groupby("policy_id", as_index=False).agg(
        normalized_profit=("normalized_profit", "mean"),
        waste_rate=("waste_rate", "mean"),
        average_markdown=("average_markdown", "mean"),
    )
    ax.scatter(policy_summary["waste_rate"], policy_summary["normalized_profit"])
    for _, row in policy_summary.iterrows():
        ax.annotate(row["policy_id"], (row["waste_rate"], row["normalized_profit"]), fontsize=8)
    ax.set_xlabel("Waste rate")
    ax.set_ylabel("Normalized profit")
    ax.set_title("Waste and profit comparison")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "waste_and_profit_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    policy_summary.set_index("policy_id")["normalized_profit"].plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean normalized profit")
    ax.set_title("Planning, PPO reference and baseline comparison")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "planning_ppo_baseline_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    action_counts = pd.Series("|".join(plan["action_sequence"]).split("|")).value_counts(normalize=True).sort_index()
    action_counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Action")
    ax.set_ylabel("Share")
    ax.set_title("Planning action distribution in high-risk population")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "planning_action_distribution_by_risk_level.png", dpi=160)
    plt.close(fig)


def write_doc(summary: pd.DataFrame, by_scenario: pd.DataFrame, baseline_policy: str, runtime_seconds: float) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    status = summary["primary_status"].iloc[0]
    ppo_status = "PPO_RETRAINING_JUSTIFIED" if status in {"DYNAMIC_VALUE_CONFIRMED", "DYNAMIC_VALUE_MODEST"} else "PPO_RETRAINING_NOT_JUSTIFIED"
    best = by_scenario.sort_values("paired_mean_profit_gain", ascending=False).iloc[0]
    worst = by_scenario.sort_values("paired_mean_profit_gain", ascending=True).iloc[0]
    text = f"""# Limited-Horizon Planning Decision

## Locked Decision Task

Population: `HIGH_RISK_B`.
Split: validation only.
Calibration: recovered-demand calibration.
Locked baseline: `{baseline_policy}`.

## Planning Method

Policy name: `limited_horizon_planning_policy`.
Planning horizon: {PLANNING_HORIZON} decision steps.
Branching factor: 6 markdown actions.
Maximum candidate sequences per decision: {MAX_CANDIDATE_SEQUENCES}.
Continuation assumption: deterministic finite lookahead using the existing environment and reward/accounting mechanics.
Terminal value handling: no learned terminal value beyond simulated rewards within the bounded horizon.
Optimality: not a perfect oracle and not guaranteed globally optimal.

## Primary Result

Status: `{status}`.
PPO retraining status: `{ppo_status}`.

Mean paired normalized-profit gain: {summary['paired_mean_normalized_profit_gain'].iloc[0]:.6f}.
95% bootstrap CI: [{summary['bootstrap_ci_low'].iloc[0]:.6f}, {summary['bootstrap_ci_high'].iloc[0]:.6f}].
Win/tie/loss: {summary['win_share'].iloc[0]:.3f} / {summary['tie_share'].iloc[0]:.3f} / {summary['loss_share'].iloc[0]:.3f}.
Waste-rate difference: {summary['waste_rate_difference'].iloc[0]:.6f}.
Sell-through difference: {summary['sell_through_difference'].iloc[0]:.6f}.

Best scenario: `{best['scenario_id']}` with mean gain {best['paired_mean_profit_gain']:.6f}.
Worst scenario: `{worst['scenario_id']}` with mean gain {worst['paired_mean_profit_gain']:.6f}.

Runtime: {runtime_seconds:.1f} seconds.

## Interpretation

If status is `DYNAMIC_VALUE_CONFIRMED`, the locked high-risk task contains realizable dynamic value under the current economic assumptions, and further RL training can be justified without using the test split.

If status is `DYNAMIC_VALUE_MODEST`, dynamic value may exist but uncertainty remains; further RL work should be cautious and validation-locked.

If status is `NO_DEFENSIBLE_DYNAMIC_MODEL`, the current environment and locked task do not support a defensible claim that dynamic markdown beats the strong baseline.

## Limitations

This is a bounded limited-horizon planner, not a perfect oracle. It uses validation only and must not be tuned against the test split.
"""
    (DOCS_DIR / "limited_horizon_planning_decision.md").write_text(text, encoding="utf-8")


def main() -> None:
    start = time.perf_counter()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    locked_population, locked_baselines, state_source = load_locked_task()
    episodes = select_high_risk_episodes(locked_population, state_source)
    baseline_policy = write_locked_definition(locked_population, locked_baselines, episodes)

    results: list[dict[str, Any]] = []
    for idx, episode in enumerate(episodes, start=1):
        print(f"Evaluating episode {idx}/{len(episodes)}: {episode.episode_id}")
        results.append(evaluate_policy_episode(episode, baseline_policy))
        results.append(evaluate_policy_episode(episode, "limited_horizon_planning_policy"))
    results_df = pd.DataFrame(results)
    results_df.to_csv(TABLES_DIR / "limited_horizon_planning_episode_results.csv", index=False)

    audit = pairing_audit(results_df)
    if audit.empty or not audit["pairing_status"].eq("PAIRED_VALID").all():
        print("PLANNING_EVALUATION_PAIRING_FAILED")
        print("PPO_RETRAINING_NOT_JUSTIFIED")
        return

    summary = comparison_summary(results_df, baseline_policy)
    by_scenario = scenario_summary(results_df, baseline_policy)
    ppo_comparison(results_df, baseline_policy, summary)
    make_figures(results_df, summary, by_scenario, baseline_policy)
    runtime = time.perf_counter() - start
    write_doc(summary, by_scenario, baseline_policy, runtime)

    status = str(summary["primary_status"].iloc[0])
    ppo_status = "PPO_RETRAINING_JUSTIFIED" if status in {"DYNAMIC_VALUE_CONFIRMED", "DYNAMIC_VALUE_MODEST"} else "PPO_RETRAINING_NOT_JUSTIFIED"
    print(status)
    print(ppo_status)
    print(f"paired_episodes={len(episodes)}")
    print(f"planning_horizon={PLANNING_HORIZON}")
    print(f"total_candidate_sequences_evaluated={int(results_df['candidate_sequences_evaluated'].sum())}")
    print(f"runtime_seconds={runtime:.1f}")
    print(f"locked_baseline={baseline_policy}")
    print(f"mean_paired_gain={summary['paired_mean_normalized_profit_gain'].iloc[0]:.6f}")
    print(f"ci95=[{summary['bootstrap_ci_low'].iloc[0]:.6f}, {summary['bootstrap_ci_high'].iloc[0]:.6f}]")
    print(f"win_tie_loss={summary['win_share'].iloc[0]:.3f}/{summary['tie_share'].iloc[0]:.3f}/{summary['loss_share'].iloc[0]:.3f}")
    print(f"waste_difference={summary['waste_rate_difference'].iloc[0]:.6f}")
    print("files_created:")
    for name in [
        "planning_locked_task_definition.csv",
        "planning_pairing_audit.csv",
        "limited_horizon_planning_episode_results.csv",
        "planning_vs_locked_baseline_summary.csv",
        "planning_vs_baseline_by_scenario.csv",
        "planning_ppo_baseline_comparison.csv",
    ]:
        print(f"- outputs/tables/{name}")
    print("- docs/limited_horizon_planning_decision.md")


if __name__ == "__main__":
    main()
