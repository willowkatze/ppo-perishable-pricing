"""
Final model decision-task audit for the FreshRetailNet perishable markdown project.

This script is intentionally conservative:
- It does not retrain PPO or any model.
- It does not use the test split.
- It does not modify data, environments, rewards, accounting, response models, or model artifacts.
- It locks validation populations and baselines before any held-out test evaluation.

Run from project root:
    python src/final_model_decision_task_audit.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "final_model_decision_task"
DOCS_DIR = PROJECT_ROOT / "docs"

MEANINGFUL_SCENARIOS = ["core_002", "core_003", "core_005", "core_006", "core_009", "core_012"]
CALIBRATIONS = ["recovered_calibration", "observed_calibration"]
PRIMARY_CALIBRATION = "recovered_calibration"

FIXED_POLICIES = [
    "always_0pct",
    "always_5pct",
    "always_10pct",
    "always_20pct",
    "always_30pct",
    "always_40pct",
]
RULE_POLICIES = ["expiry_threshold_rule", "inventory_coverage_rule", "combined_inventory_expiry_rule"]
DYNAMIC_POLICIES = [
    "selected_recovered_financial_ppo",
    "selected_observed_financial_ppo",
    "BALANCED__checkpoint_20000",
]

PRACTICAL_PROFIT_GAIN = 0.005
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20260717
TIE_TOL = 1e-9


@dataclass(frozen=True)
class PopulationDefinition:
    population_id: str
    description: str
    scenario_ids: tuple[str, ...]
    calibration_mode: str
    min_inventory_coverage: float | None = None
    min_fraction_expiring_within_two_days: float | None = None
    max_projected_sell_through: float | None = None
    max_remaining_shelf_life: float | None = None
    selected: bool = False
    selection_reason: str = ""


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


def bootstrap_ci(diff: pd.Series, n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    values = diff.dropna().astype(float).to_numpy()
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    means = np.empty(n, dtype=float)
    for i in range(n):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_summary(dynamic: pd.DataFrame, baseline: pd.DataFrame, metric: str) -> dict[str, float]:
    merged = dynamic.merge(
        baseline[["episode_id", metric]],
        on="episode_id",
        how="inner",
        suffixes=("_dynamic", "_baseline"),
    )
    if merged.empty:
        return {
            "paired_episode_count": 0,
            "paired_mean_difference": np.nan,
            "paired_median_difference": np.nan,
            "bootstrap_ci_low": np.nan,
            "bootstrap_ci_high": np.nan,
            "win_share": np.nan,
            "tie_share": np.nan,
            "loss_share": np.nan,
        }
    diff = merged[f"{metric}_dynamic"].astype(float) - merged[f"{metric}_baseline"].astype(float)
    ci_low, ci_high = bootstrap_ci(diff)
    return {
        "paired_episode_count": int(len(diff)),
        "paired_mean_difference": float(diff.mean()),
        "paired_median_difference": float(diff.median()),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "win_share": float((diff > TIE_TOL).mean()),
        "tie_share": float((diff.abs() <= TIE_TOL).mean()),
        "loss_share": float((diff < -TIE_TOL).mean()),
    }


def load_episode_source() -> pd.DataFrame:
    episode = read_csv("sustainability_tradeoff_episode_level.csv")
    required = {
        "scenario_id",
        "calibration_mode",
        "policy_id",
        "episode_id",
        "normalized_accounting_profit",
        "waste_rate",
        "sell_through_rate",
        "average_markdown",
    }
    missing = required.difference(episode.columns)
    if missing:
        raise ValueError(f"sustainability_tradeoff_episode_level.csv missing columns: {sorted(missing)}")
    episode = episode.loc[
        episode["scenario_id"].isin(MEANINGFUL_SCENARIOS)
        & episode["calibration_mode"].isin(CALIBRATIONS)
    ].copy()
    for col in [
        "episode_index",
        "normalized_accounting_profit",
        "raw_accounting_profit",
        "waste_rate",
        "sell_through_rate",
        "average_markdown",
        "physical_waste_units",
        "stockout_rate",
    ]:
        if col in episode.columns:
            episode[col] = pd.to_numeric(episode[col], errors="coerce")
    return episode


def load_state_source() -> pd.DataFrame:
    states = read_csv("financial_markdown_oracle_diagnostic.csv")
    states = states.loc[states["scenario_id"].isin(MEANINGFUL_SCENARIOS)].copy()
    for col in [
        "episode_index",
        "inventory_coverage_state",
        "fraction_expiring_within_two_days",
        "predicted_baseline_demand",
        "remaining_inventory_life",
        "rolling_oracle_value",
        "zero_first_action_oracle_value",
        "oracle_gain_vs_zero",
    ]:
        if col in states.columns:
            states[col] = pd.to_numeric(states[col], errors="coerce")
    states["projected_sell_through_proxy"] = (
        states["predicted_baseline_demand"] / states["inventory_coverage_state"].replace(0, np.nan)
    ).clip(lower=0, upper=3)
    return states


def build_population_candidates(states: pd.DataFrame, episode: pd.DataFrame) -> pd.DataFrame:
    defs = [
        PopulationDefinition(
            "REPRESENTATIVE_OPERATIONAL_POPULATION",
            "All validation episodes in the six meaningful profit-waste scenarios.",
            tuple(MEANINGFUL_SCENARIOS),
            PRIMARY_CALIBRATION,
            selected=True,
            selection_reason="Preserves the existing validation distribution for the meaningful tradeoff scenario set.",
        ),
        PopulationDefinition(
            "HIGH_RISK_A",
            "Coverage >= 1.5 and fraction expiring within two days >= 0.50.",
            tuple(MEANINGFUL_SCENARIOS),
            PRIMARY_CALIBRATION,
            min_inventory_coverage=1.5,
            min_fraction_expiring_within_two_days=0.50,
        ),
        PopulationDefinition(
            "HIGH_RISK_B",
            "Coverage >= 1.5, fraction expiring within two days >= 0.50, and projected sell-through proxy <= 0.75.",
            tuple(MEANINGFUL_SCENARIOS),
            PRIMARY_CALIBRATION,
            min_inventory_coverage=1.5,
            min_fraction_expiring_within_two_days=0.50,
            max_projected_sell_through=0.75,
            selected=True,
            selection_reason="Operationally interpretable exception-management definition; not selected using PPO wins.",
        ),
        PopulationDefinition(
            "HIGH_RISK_C",
            "Remaining shelf life <= 2 and fraction expiring within two days >= 0.50.",
            tuple(MEANINGFUL_SCENARIOS),
            PRIMARY_CALIBRATION,
            min_fraction_expiring_within_two_days=0.50,
            max_remaining_shelf_life=2,
        ),
    ]
    total_episode_count = episode.loc[episode["calibration_mode"].eq(PRIMARY_CALIBRATION), "episode_id"].nunique()
    rows: list[dict[str, Any]] = []
    for pop in defs:
        subset = states.loc[states["calibration_mode"].eq(pop.calibration_mode)].copy()
        subset = subset.loc[subset["scenario_id"].isin(pop.scenario_ids)]
        if pop.min_inventory_coverage is not None:
            subset = subset.loc[subset["inventory_coverage_state"] >= pop.min_inventory_coverage]
        if pop.min_fraction_expiring_within_two_days is not None:
            subset = subset.loc[subset["fraction_expiring_within_two_days"] >= pop.min_fraction_expiring_within_two_days]
        if pop.max_projected_sell_through is not None:
            subset = subset.loc[subset["projected_sell_through_proxy"] <= pop.max_projected_sell_through]
        if pop.max_remaining_shelf_life is not None:
            subset = subset.loc[subset["remaining_inventory_life"] <= pop.max_remaining_shelf_life]

        episode_indices = sorted(
            pd.to_numeric(subset.get("episode_index", pd.Series(dtype=float)), errors="coerce")
            .dropna()
            .astype(int)
            .unique()
        )
        always_zero = episode.loc[
            episode["calibration_mode"].eq(pop.calibration_mode)
            & episode["policy_id"].eq("always_0pct")
            & episode["scenario_id"].isin(pop.scenario_ids)
        ].copy()
        if pop.population_id != "REPRESENTATIVE_OPERATIONAL_POPULATION" and episode_indices:
            always_zero = always_zero.loc[always_zero["episode_index"].isin(episode_indices)]

        rows.append(
            {
                "population_id": pop.population_id,
                "description": pop.description,
                "calibration_mode": pop.calibration_mode,
                "scenario_ids": "|".join(pop.scenario_ids),
                "min_inventory_coverage": pop.min_inventory_coverage,
                "min_fraction_expiring_within_two_days": pop.min_fraction_expiring_within_two_days,
                "max_projected_sell_through": pop.max_projected_sell_through,
                "max_remaining_shelf_life": pop.max_remaining_shelf_life,
                "state_count": int(len(subset)),
                "episode_count_proxy": int(len(episode_indices)) if pop.population_id != "REPRESENTATIVE_OPERATIONAL_POPULATION" else int(total_episode_count),
                "share_of_total_episodes_proxy": float((len(episode_indices) if pop.population_id != "REPRESENTATIVE_OPERATIONAL_POPULATION" else total_episode_count) / max(total_episode_count, 1)),
                "always_zero_mean_waste_rate": float(always_zero["waste_rate"].mean()) if not always_zero.empty else np.nan,
                "mean_inventory_coverage": float(subset["inventory_coverage_state"].mean()) if not subset.empty else np.nan,
                "mean_fraction_expiring_within_two_days": float(subset["fraction_expiring_within_two_days"].mean()) if not subset.empty else np.nan,
                "mean_projected_sell_through_proxy": float(subset["projected_sell_through_proxy"].mean()) if not subset.empty else np.nan,
                "selected_for_lock": bool(pop.selected),
                "selection_reason": pop.selection_reason,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "final_task_population_candidates.csv", index=False)
    df.loc[df["selected_for_lock"]].to_csv(TABLES_DIR / "final_task_locked_population_definition.csv", index=False)
    return df


def filter_population_episode(episode: pd.DataFrame, population_id: str, candidates: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    pop = candidates.loc[candidates["population_id"].eq(population_id)].iloc[0]
    out = episode.loc[
        episode["calibration_mode"].eq(pop["calibration_mode"])
        & episode["scenario_id"].isin(str(pop["scenario_ids"]).split("|"))
    ].copy()
    if population_id == "REPRESENTATIVE_OPERATIONAL_POPULATION":
        return out

    subset = states.loc[states["calibration_mode"].eq(pop["calibration_mode"])].copy()
    subset = subset.loc[subset["scenario_id"].isin(str(pop["scenario_ids"]).split("|"))]
    if not pd.isna(pop["min_inventory_coverage"]):
        subset = subset.loc[subset["inventory_coverage_state"] >= float(pop["min_inventory_coverage"])]
    if not pd.isna(pop["min_fraction_expiring_within_two_days"]):
        subset = subset.loc[subset["fraction_expiring_within_two_days"] >= float(pop["min_fraction_expiring_within_two_days"])]
    if not pd.isna(pop["max_projected_sell_through"]):
        subset = subset.loc[subset["projected_sell_through_proxy"] <= float(pop["max_projected_sell_through"])]
    if not pd.isna(pop["max_remaining_shelf_life"]):
        subset = subset.loc[subset["remaining_inventory_life"] <= float(pop["max_remaining_shelf_life"])]
    ep_idx = set(pd.to_numeric(subset["episode_index"], errors="coerce").dropna().astype(int).tolist())
    return out.loc[out["episode_index"].isin(ep_idx)]


def summarize_population(episode: pd.DataFrame, candidates: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for population_id in candidates.loc[candidates["selected_for_lock"], "population_id"]:
        pop_ep = filter_population_episode(episode, population_id, candidates, states)
        az = pop_ep.loc[pop_ep["policy_id"].eq("always_0pct")]
        rows.append(
            {
                "population_id": population_id,
                "calibration_mode": pop_ep["calibration_mode"].iloc[0] if not pop_ep.empty else PRIMARY_CALIBRATION,
                "episode_count": int(az["episode_id"].nunique()),
                "policy_count_available": int(pop_ep["policy_id"].nunique()),
                "always_zero_profit": float(az["normalized_accounting_profit"].mean()) if not az.empty else np.nan,
                "always_zero_waste_rate": float(az["waste_rate"].mean()) if not az.empty else np.nan,
                "always_zero_sell_through_rate": float(az["sell_through_rate"].mean()) if not az.empty else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "final_task_population_summary.csv", index=False)
    return df


def lock_baselines(episode: pd.DataFrame, candidates: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for population_id in candidates.loc[candidates["selected_for_lock"], "population_id"]:
        pop_ep = filter_population_episode(episode, population_id, candidates, states)
        means = pop_ep.groupby("policy_id", as_index=False).agg(
            mean_normalized_profit=("normalized_accounting_profit", "mean"),
            mean_waste_rate=("waste_rate", "mean"),
            mean_sell_through_rate=("sell_through_rate", "mean"),
            mean_average_markdown=("average_markdown", "mean"),
            episode_count=("episode_id", "nunique"),
        )
        fixed = means.loc[means["policy_id"].isin(FIXED_POLICIES)].sort_values(
            ["mean_normalized_profit", "mean_waste_rate"], ascending=[False, True]
        )
        rules = means.loc[means["policy_id"].isin(RULE_POLICIES)].sort_values(
            ["mean_normalized_profit", "mean_waste_rate"], ascending=[False, True]
        )
        best_fixed = fixed.iloc[0] if not fixed.empty else None
        best_rule = rules.iloc[0] if not rules.empty else None
        primary_candidates = [x for x in [best_fixed, best_rule] if x is not None]
        primary = sorted(primary_candidates, key=lambda x: (x["mean_normalized_profit"], -x["mean_waste_rate"]), reverse=True)[0] if primary_candidates else None
        rows.append(
            {
                "population_id": population_id,
                "best_fixed_validation_policy": best_fixed["policy_id"] if best_fixed is not None else "NOT_AVAILABLE",
                "best_fixed_validation_profit": float(best_fixed["mean_normalized_profit"]) if best_fixed is not None else np.nan,
                "best_fixed_waste_rate": float(best_fixed["mean_waste_rate"]) if best_fixed is not None else np.nan,
                "best_rule_validation_policy": best_rule["policy_id"] if best_rule is not None else "NOT_AVAILABLE",
                "best_rule_validation_profit": float(best_rule["mean_normalized_profit"]) if best_rule is not None else np.nan,
                "best_rule_waste_rate": float(best_rule["mean_waste_rate"]) if best_rule is not None else np.nan,
                "locked_primary_baseline_policy": primary["policy_id"] if primary is not None else "NOT_AVAILABLE",
                "locked_primary_baseline_profit": float(primary["mean_normalized_profit"]) if primary is not None else np.nan,
                "locked_primary_baseline_waste_rate": float(primary["mean_waste_rate"]) if primary is not None else np.nan,
                "selection_rule": "max(best_fixed_validation_policy, best_rule_validation_policy) by validation normalized profit; ties prefer lower waste.",
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "final_task_locked_baselines.csv", index=False)
    return df


def dynamic_value_audit(episode: pd.DataFrame, candidates: pd.DataFrame, states: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    rows = []
    available_dynamic = [p for p in DYNAMIC_POLICIES + RULE_POLICIES if p in set(episode["policy_id"].unique())]
    for _, base_row in baselines.iterrows():
        population_id = base_row["population_id"]
        baseline_policy = base_row["locked_primary_baseline_policy"]
        pop_ep = filter_population_episode(episode, population_id, candidates, states)
        baseline_ep = pop_ep.loc[pop_ep["policy_id"].eq(baseline_policy)]
        for policy in available_dynamic:
            dyn_ep = pop_ep.loc[pop_ep["policy_id"].eq(policy)]
            if dyn_ep.empty or baseline_ep.empty:
                continue
            profit = paired_summary(dyn_ep, baseline_ep, "normalized_accounting_profit")
            waste = paired_summary(dyn_ep, baseline_ep, "waste_rate")
            sell = paired_summary(dyn_ep, baseline_ep, "sell_through_rate")
            mean_gain = profit["paired_mean_difference"]
            ci_low = profit["bootstrap_ci_low"]
            if pd.notna(mean_gain) and mean_gain >= PRACTICAL_PROFIT_GAIN and pd.notna(ci_low) and ci_low > 0:
                classification = "DYNAMIC_VALUE_PRESENT"
            elif pd.notna(mean_gain) and abs(mean_gain) < PRACTICAL_PROFIT_GAIN:
                classification = "DYNAMIC_VALUE_TOO_SMALL"
            else:
                classification = "DYNAMIC_VALUE_ABSENT"
            rows.append(
                {
                    "population_id": population_id,
                    "dynamic_policy_id": policy,
                    "locked_primary_baseline_policy": baseline_policy,
                    "paired_mean_profit_difference": profit["paired_mean_difference"],
                    "paired_median_profit_difference": profit["paired_median_difference"],
                    "bootstrap_ci_low": profit["bootstrap_ci_low"],
                    "bootstrap_ci_high": profit["bootstrap_ci_high"],
                    "win_share": profit["win_share"],
                    "tie_share": profit["tie_share"],
                    "loss_share": profit["loss_share"],
                    "waste_rate_difference": waste["paired_mean_difference"],
                    "sell_through_difference": sell["paired_mean_difference"],
                    "mean_dynamic_profit": float(dyn_ep["normalized_accounting_profit"].mean()),
                    "mean_baseline_profit": float(baseline_ep["normalized_accounting_profit"].mean()),
                    "dynamic_value_classification": classification,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "final_task_dynamic_value_existing_policy_audit.csv", index=False)
    return df


def planning_diagnostic() -> pd.DataFrame:
    values = read_csv("financial_markdown_multistep_action_values.csv")
    values = values.loc[
        values["scenario_id"].isin(MEANINGFUL_SCENARIOS)
        & values["calibration_mode"].eq(PRIMARY_CALIBRATION)
    ].copy()
    if values.empty:
        out = pd.DataFrame([{"planning_policy_status": "NO_ACTION_VALUE_TABLE_AVAILABLE"}])
        out.to_csv(TABLES_DIR / "final_task_dynamic_value_planning_audit.csv", index=False)
        return out
    values["total_normalized_accounting_profit"] = pd.to_numeric(values["total_normalized_accounting_profit"], errors="coerce")
    best = values.sort_values("total_normalized_accounting_profit", ascending=False).groupby("state_id", as_index=False).first()
    zero = values.loc[pd.to_numeric(values["first_action"], errors="coerce").eq(0), ["state_id", "total_normalized_accounting_profit"]].rename(
        columns={"total_normalized_accounting_profit": "zero_action_value"}
    )
    merged = best.merge(zero, on="state_id", how="left")
    merged["planning_gain_vs_zero"] = merged["total_normalized_accounting_profit"] - merged["zero_action_value"]
    out = pd.DataFrame(
        [
            {
                "planning_policy_id": "limited_horizon_planning_policy",
                "planning_horizon": "finite state-level lookahead from existing financial_markdown_multistep_action_values.csv",
                "candidate_action_sequences": "first markdown action 0..5 with recorded continuation values",
                "continuation_assumptions": "uses precomputed continuation_policy values",
                "computational_approximation": "state-level action-value diagnostic, not full paired episode rollout",
                "guaranteed_optimal": False,
                "state_count_with_action_values": int(merged["state_id"].nunique()),
                "mean_planning_gain_vs_zero": float(merged["planning_gain_vs_zero"].mean()),
                "share_positive_planning_gain_vs_zero": float((merged["planning_gain_vs_zero"] > PRACTICAL_PROFIT_GAIN).mean()),
                "max_planning_gain_vs_zero": float(merged["planning_gain_vs_zero"].max()),
                "dynamic_value_screening_status": "PLANNING_DIAGNOSTIC_ONLY_NOT_FULL_PAIRED_EVALUATION",
            }
        ]
    )
    out.to_csv(TABLES_DIR / "final_task_dynamic_value_planning_audit.csv", index=False)
    return out


def phase_gate(dynamic_existing: pd.DataFrame, planning: pd.DataFrame) -> pd.DataFrame:
    high_risk = dynamic_existing.loc[dynamic_existing["population_id"].eq("HIGH_RISK_B")]
    if high_risk.empty:
        status = "NO_DEFENSIBLE_DYNAMIC_MODEL"
        reason = "No high-risk paired validation comparison was available from existing outputs."
    elif (high_risk["dynamic_value_classification"] == "DYNAMIC_VALUE_PRESENT").any():
        status = "DYNAMIC_VALUE_PRESENT_VALIDATION_ONLY"
        reason = "At least one existing dynamic/rule policy exceeds the locked baseline under validation criteria."
    else:
        plan_gain = safe_float(planning.get("mean_planning_gain_vs_zero", pd.Series([np.nan])).iloc[0]) if not planning.empty else np.nan
        if pd.notna(plan_gain) and plan_gain > PRACTICAL_PROFIT_GAIN:
            status = "DYNAMIC_VALUE_REQUIRES_FULL_PAIRED_PLANNING_EVALUATION"
            reason = "State-level planning diagnostic indicates possible value, but full paired episode-level planning evaluation is required before RL retraining."
        else:
            status = "NO_DEFENSIBLE_DYNAMIC_MODEL"
            reason = "Existing validation comparisons do not show dynamic value over the locked baseline."
    df = pd.DataFrame(
        [
            {
                "phase_gate_status": status,
                "test_split_used": False,
                "ppo_training_run": False,
                "dqn_training_run": False,
                "final_model_locked": False,
                "reason": reason,
                "next_allowed_action": "Do not use test. Run full paired planning validation only if diagnostic suggests possible dynamic value.",
            }
        ]
    )
    df.to_csv(TABLES_DIR / "final_task_phase_gate_decision.csv", index=False)
    return df


def make_figures(candidates: pd.DataFrame, baselines: pd.DataFrame, dynamic_existing: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    candidates.set_index("population_id")["state_count"].plot(kind="bar", ax=ax)
    ax.set_ylabel("State count")
    ax.set_title("Candidate decision populations")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "candidate_population_state_counts.png", dpi=160)
    plt.close(fig)

    if not dynamic_existing.empty:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        pivot = dynamic_existing.pivot_table(
            index="dynamic_policy_id",
            columns="population_id",
            values="paired_mean_profit_difference",
            aggfunc="mean",
        )
        pivot.plot(kind="bar", ax=ax)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(PRACTICAL_PROFIT_GAIN, color="gray", linestyle="--", linewidth=0.8)
        ax.set_ylabel("Paired mean profit difference vs locked baseline")
        ax.set_title("Existing dynamic policies versus locked baseline")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "dynamic_vs_locked_baseline_validation.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    baselines.set_index("population_id")[["best_fixed_validation_profit", "best_rule_validation_profit", "locked_primary_baseline_profit"]].plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean normalized validation profit")
    ax.set_title("Locked validation baselines")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "locked_baseline_selection.png", dpi=160)
    plt.close(fig)


def write_interpretation(candidates: pd.DataFrame, baselines: pd.DataFrame, gate: pd.DataFrame) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    high_risk_row = candidates.loc[candidates["population_id"].eq("HIGH_RISK_B")].iloc[0]
    baseline_row = baselines.loc[baselines["population_id"].eq("HIGH_RISK_B")].iloc[0] if "HIGH_RISK_B" in set(baselines["population_id"]) else baselines.iloc[0]
    status = gate["phase_gate_status"].iloc[0]
    text = f"""# Final Model Decision-Task Audit

## Purpose

This audit implements the instructor's strict success criterion: a final model must beat a strong, pre-specified baseline on a paired held-out task. This script does not use the test split and does not train any model.

## Locked High-Risk Population

Locked high-risk population: `HIGH_RISK_B`.

Definition: inventory coverage >= 1.5, fraction expiring within two days >= 0.50, and projected sell-through proxy <= 0.75, evaluated on the six pre-existing meaningful profit-waste scenarios under recovered calibration.

Selection basis: operational interpretability, inventory pressure, expiry pressure, low sell-through proxy, and existing scenario design. It was not selected based on PPO wins.

Population proxy size: {high_risk_row['episode_count_proxy']} episodes/states proxy, {high_risk_row['share_of_total_episodes_proxy']:.3f} share of the representative meaningful-scenario validation population.

## Locked Baseline

Locked primary baseline for the high-risk population: `{baseline_row['locked_primary_baseline_policy']}`.

Selection rule: choose the better of the best fixed validation policy and the best rule validation policy by mean normalized validation profit; ties prefer lower waste.

## Dynamic Value Result

Phase gate status: `{status}`.

The current audit uses existing validation outputs. It does not claim a final model has been confirmed. A full one-time test evaluation remains forbidden until all task definitions, baselines, algorithms, and model-selection rules are locked.

## Planning Benchmark

The planning output is labelled `limited_horizon_planning_policy` only as a diagnostic unless a full paired episode-level planning rollout is implemented. It is not a perfect oracle and is not guaranteed optimal.

## Report-Ready Language

If dynamic policy wins: \"A pre-specified dynamic markdown policy outperformed the locked strong baseline on paired validation in the high-risk population, justifying final model locking before one-time test evaluation.\"

If only planning wins: \"A limited-horizon planning benchmark indicates dynamic value exists, but learned policies have not yet captured it; additional training can be justified only within the locked high-risk task.\"

If only waste-constrained value exists: \"Dynamic markdown is not financially superior under the unconstrained objective, but may be justified for waste-constrained operations if it improves profit at the same waste constraint.\"

If no dynamic policy wins: \"Within the current economic assumptions and pre-specified high-risk task, strong fixed/rule baselines are sufficient; no defensible dynamic final model is currently supported.\"
"""
    (DOCS_DIR / "final_model_decision_task_interpretation.md").write_text(text, encoding="utf-8")


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    episode = load_episode_source()
    states = load_state_source()
    candidates = build_population_candidates(states, episode)
    summarize_population(episode, candidates, states)
    baselines = lock_baselines(episode, candidates, states)
    dynamic_existing = dynamic_value_audit(episode, candidates, states, baselines)
    planning = planning_diagnostic()
    gate = phase_gate(dynamic_existing, planning)
    make_figures(candidates, baselines, dynamic_existing)
    write_interpretation(candidates, baselines, gate)

    status = gate["phase_gate_status"].iloc[0]
    print(status if status != "NO_DEFENSIBLE_DYNAMIC_MODEL" else "NO_DEFENSIBLE_DYNAMIC_MODEL")
    print("TEST_SPLIT_USED=False")
    print("PPO_TRAINING_RUN=False")
    print("FINAL_MODEL_LOCKED=False")
    print("Created:")
    for name in [
        "final_task_population_candidates.csv",
        "final_task_locked_population_definition.csv",
        "final_task_population_summary.csv",
        "final_task_locked_baselines.csv",
        "final_task_dynamic_value_existing_policy_audit.csv",
        "final_task_dynamic_value_planning_audit.csv",
        "final_task_phase_gate_decision.csv",
    ]:
        print(f"- outputs/tables/{name}")
    print("- docs/final_model_decision_task_interpretation.md")


if __name__ == "__main__":
    main()
