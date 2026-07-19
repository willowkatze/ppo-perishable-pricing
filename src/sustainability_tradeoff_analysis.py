from __future__ import annotations

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
MODELS_DIR = PROJECT_ROOT / "outputs" / "models" / "ppo_operational"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "sustainability_tradeoff_analysis"

SCENARIO_AUDIT_PATH = TABLES_DIR / "sustainability_scenario_audit.csv"
SELECTED_MODELS_PATH = CONFIGS_DIR / "ppo_selected_financial_models.json"
DEFAULT_SCENARIO_ID = "core_008"
EPISODES_PER_SCENARIO = 8
CALIBRATIONS = ["recovered_calibration", "observed_calibration"]
FIXED_POLICIES = [
    "always_0pct",
    "always_5pct",
    "always_10pct",
    "always_20pct",
    "always_40pct",
    "expiry_threshold_rule",
    "inventory_coverage_rule",
]
PPO_POLICIES = ["selected_observed_financial_ppo", "selected_recovered_financial_ppo"]
ALL_POLICIES = FIXED_POLICIES + PPO_POLICIES
ABS_WASTE_CONSTRAINTS = [0.35, 0.25, 0.15, 0.10]
REL_WASTE_REDUCTION_CONSTRAINTS = [0.05, 0.10, 0.20]
PROFIT_TOL = 1e-6
WASTE_TOL = 1e-6
NEAR_DUP_PROFIT_TOL = 1e-4
NEAR_DUP_WASTE_TOL = 1e-4
MATERIAL_PROFIT_DENOM = 1e-4
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
    model_path: Path
    vecnormalize_path: Path
    classification: str


def ensure_dirs() -> None:
    for path in [TABLES_DIR, CONFIGS_DIR, FIGURES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


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
    return float(-sum((c / total) * math.log(c / total) for c in counts.values() if c))


def action_for_fixed_policy(policy_id: str, obs: np.ndarray) -> int:
    if policy_id == "always_0pct":
        return 0
    if policy_id == "always_5pct":
        return 1
    if policy_id == "always_10pct":
        return 2
    if policy_id == "always_20pct":
        return 3
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
    raise ValueError(f"Unsupported fixed policy {policy_id}")


def load_selected_scenarios() -> pd.DataFrame:
    audit = pd.read_csv(SCENARIO_AUDIT_PATH)
    selected = audit.loc[audit["classification"].eq("MEANINGFUL_PROFIT_WASTE_TRADEOFF")].copy()
    selected["calibration_mode_primary"] = "recovered_calibration"
    selected["accounting_validity_confirmed_from_tradeoff_run"] = False
    selected.to_csv(TABLES_DIR / "sustainability_selected_tradeoff_scenarios.csv", index=False)
    return selected


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_ppo_policies() -> dict[str, LoadedPPO]:
    if not SELECTED_MODELS_PATH.exists():
        return {}
    metadata = json.loads(SELECTED_MODELS_PATH.read_text(encoding="utf-8"))
    loaded: dict[str, LoadedPPO] = {}
    for policy_id, details in metadata.get("selected_models", {}).items():
        if policy_id not in PPO_POLICIES:
            continue
        model_path = resolve_project_path(details["model_path"])
        vec_path = resolve_project_path(details["vecnormalize_path"])
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
            model_path=model_path,
            vecnormalize_path=vec_path,
            classification=details.get("classification", ""),
        )
    return loaded


def evaluate_episode(
    env: OperationalPerishablePricingEnv,
    scenario_meta: pd.Series,
    calibration_mode: str,
    policy_id: str,
    episode_index: int,
    ppo_models: dict[str, LoadedPPO],
) -> dict[str, Any]:
    obs, _ = env.reset(seed=20_000 + episode_index, options={"episode_index": episode_index})
    initial_inventory = safe_float(getattr(env, "initial_inventory", 0.0))
    action_counts = {action: 0 for action in ACTION_MARKDOWNS}
    first_markdown_day = -1
    max_markdown = 0.0
    stockout_flags: list[float] = []
    total_return = 0.0
    final_info: dict[str, Any] = {}
    terminated = False
    truncated = False
    day = 0
    while not (terminated or truncated):
        if policy_id in ppo_models:
            loaded = ppo_models[policy_id]
            model_obs = loaded.vecnormalize.normalize_obs(obs.astype(np.float32)[None, :])[0]
            predicted, _ = loaded.model.predict(model_obs, deterministic=True)
            action = int(np.asarray(predicted).item())
        else:
            action = action_for_fixed_policy(policy_id, obs)
        markdown = ACTION_MARKDOWNS[action]
        if first_markdown_day < 0 and markdown > 0:
            first_markdown_day = day
        max_markdown = max(max_markdown, markdown)
        obs, reward, terminated, truncated, info = env.step(action)
        action_counts[action] += 1
        stockout_flags.append(float(bool(info.get("stockout_flag", False))))
        total_return += float(reward)
        final_info = info
        day += 1
    total_actions = sum(action_counts.values())
    ppo_meta = ppo_models.get(policy_id)
    kind = "ppo_cross_scenario_financial_policy" if ppo_meta else "fixed_or_rule_policy"
    return {
        "scenario_id": scenario_meta["scenario_id"],
        "policy_id": policy_id,
        "policy_kind": kind,
        "calibration_mode": calibration_mode,
        "training_calibration_mode": ppo_meta.training_calibration_mode if ppo_meta else calibration_mode,
        "cross_scenario_policy_evaluation": bool(ppo_meta),
        "cross_calibration_policy_evaluation": bool(ppo_meta and ppo_meta.training_calibration_mode != calibration_mode),
        "episode_index": episode_index,
        "episode_id": f"{scenario_meta['scenario_id']}__{calibration_mode}__episode_{episode_index:02d}",
        "shelf_life_class": scenario_meta.get("shelf_life_class"),
        "inventory_coverage": scenario_meta.get("inventory_coverage"),
        "age_profile": scenario_meta.get("age_profile"),
        "margin_scenario": scenario_meta.get("margin_scenario"),
        "disposal_scenario": scenario_meta.get("disposal_scenario"),
        "demand_target": scenario_meta.get("demand_target"),
        "response_model": scenario_meta.get("response_model"),
        "normalized_episode_return": total_return,
        "normalized_accounting_profit": safe_float(final_info.get("normalized_accounting_profit")),
        "raw_accounting_profit": safe_float(final_info.get("accounting_profit")),
        "revenue": safe_float(final_info.get("cumulative_revenue")),
        "procurement_cost": safe_float(final_info.get("cumulative_procurement_cost")),
        "disposal_cost": safe_float(final_info.get("cumulative_disposal_cost")),
        "physical_waste_units": safe_float(final_info.get("cumulative_waste")),
        "waste_rate": safe_float(final_info.get("final_waste_rate")),
        "sell_through_rate": safe_float(final_info.get("final_sell_through_rate")),
        "stockout_rate": float(np.mean(stockout_flags)) if stockout_flags else 0.0,
        "average_markdown": sum(ACTION_MARKDOWNS[a] * c for a, c in action_counts.items()) / max(total_actions, 1),
        "maximum_markdown": max_markdown,
        "first_markdown_day": first_markdown_day,
        "terminal_inventory": safe_float(final_info.get("terminal_inventory")),
        "episode_length": total_actions,
        "initial_inventory": initial_inventory,
        "action_entropy": action_entropy(action_counts),
        "episode_conservation_error": safe_float(final_info.get("episode_conservation_error")),
        "raw_financial_sum_error": safe_float(final_info.get("raw_financial_sum_error")),
        **{f"action_{action}_share": action_counts[action] / max(total_actions, 1) for action in ACTION_MARKDOWNS},
    }


def evaluate_policy_set(selected: pd.DataFrame, ppo_models: dict[str, LoadedPPO]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, scenario in selected.iterrows():
        scenario_id = str(scenario["scenario_id"])
        for calibration_mode in CALIBRATIONS:
            print(f"Evaluating {scenario_id} / {calibration_mode}", flush=True)
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
            for policy_id in ALL_POLICIES:
                if policy_id in PPO_POLICIES and policy_id not in ppo_models:
                    continue
                for episode_index in range(EPISODES_PER_SCENARIO):
                    rows.append(evaluate_episode(env, scenario, calibration_mode, policy_id, episode_index, ppo_models))
    episode_df = pd.DataFrame(rows)
    episode_df.to_csv(TABLES_DIR / "sustainability_tradeoff_episode_level.csv", index=False)
    return episode_df


def summarize(episode_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "normalized_accounting_profit",
        "raw_accounting_profit",
        "revenue",
        "procurement_cost",
        "disposal_cost",
        "physical_waste_units",
        "waste_rate",
        "sell_through_rate",
        "stockout_rate",
        "average_markdown",
        "terminal_inventory",
        "episode_length",
        "action_entropy",
        "episode_conservation_error",
        "raw_financial_sum_error",
    ]
    return (
        episode_df.groupby(["scenario_id", "calibration_mode", "policy_id", "policy_kind"], as_index=False)[metrics]
        .mean()
        .sort_values(["scenario_id", "calibration_mode", "policy_id"])
    )


def vs_always_zero(summary_df: pd.DataFrame) -> pd.DataFrame:
    base = summary_df.loc[summary_df["policy_id"].eq("always_0pct")][
        ["scenario_id", "calibration_mode", "normalized_accounting_profit", "waste_rate", "physical_waste_units", "sell_through_rate", "average_markdown"]
    ].rename(
        columns={
            "normalized_accounting_profit": "normalized_profit_always_zero",
            "waste_rate": "waste_rate_always_zero",
            "physical_waste_units": "physical_waste_units_always_zero",
            "sell_through_rate": "sell_through_rate_always_zero",
            "average_markdown": "average_markdown_always_zero",
        }
    )
    out = summary_df.merge(base, on=["scenario_id", "calibration_mode"], how="left")
    out["normalized_profit_difference"] = out["normalized_accounting_profit"] - out["normalized_profit_always_zero"]
    out["normalized_profit_sacrifice"] = (out["normalized_profit_always_zero"] - out["normalized_accounting_profit"]).clip(lower=0)
    out["physical_waste_reduction"] = out["physical_waste_units_always_zero"] - out["physical_waste_units"]
    out["waste_rate_reduction"] = out["waste_rate_always_zero"] - out["waste_rate"]
    out["sell_through_change"] = out["sell_through_rate"] - out["sell_through_rate_always_zero"]
    out["average_markdown_change"] = out["average_markdown"] - out["average_markdown_always_zero"]
    out.to_csv(TABLES_DIR / "sustainability_tradeoff_vs_always_zero.csv", index=False)
    return out


def nondominated(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in summary_df.groupby(["scenario_id", "calibration_mode"]):
        for _, row in group.iterrows():
            duplicate = False
            dominated = False
            for _, other in group.iterrows():
                if row["policy_id"] == other["policy_id"]:
                    continue
                near_profit = abs(row["normalized_accounting_profit"] - other["normalized_accounting_profit"]) <= NEAR_DUP_PROFIT_TOL
                near_waste = abs(row["waste_rate"] - other["waste_rate"]) <= NEAR_DUP_WASTE_TOL
                if near_profit and near_waste:
                    duplicate = True
                profit_ge = other["normalized_accounting_profit"] >= row["normalized_accounting_profit"] - PROFIT_TOL
                waste_le = other["waste_rate"] <= row["waste_rate"] + WASTE_TOL
                strict = (
                    other["normalized_accounting_profit"] > row["normalized_accounting_profit"] + PROFIT_TOL
                    or other["waste_rate"] < row["waste_rate"] - WASTE_TOL
                )
                if profit_ge and waste_le and strict:
                    dominated = True
            status = "IDENTICAL_OR_NEAR_DUPLICATE" if duplicate else "DOMINATED" if dominated else "NON_DOMINATED"
            rows.append({**row.to_dict(), "nondominated_status": status})
    out = pd.DataFrame(rows)
    out["profit_tolerance"] = PROFIT_TOL
    out["waste_rate_tolerance"] = WASTE_TOL
    out["near_duplicate_profit_tolerance"] = NEAR_DUP_PROFIT_TOL
    out["near_duplicate_waste_tolerance"] = NEAR_DUP_WASTE_TOL
    out["set_description"] = "empirical non-dominated profit-waste set"
    out.to_csv(TABLES_DIR / "sustainability_nondominated_policy_set.csv", index=False)
    return out


def constrained_selection(vs_zero_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in vs_zero_df.groupby(["scenario_id", "calibration_mode"]):
        scenario_id, calibration_mode = keys
        for threshold in ABS_WASTE_CONSTRAINTS:
            feasible = group.loc[group["waste_rate"] <= threshold + WASTE_TOL]
            selected = feasible.sort_values("normalized_accounting_profit", ascending=False).head(1)
            rows.append(selection_row(scenario_id, calibration_mode, f"waste_rate_le_{threshold:.2f}", selected, threshold, "absolute_waste_rate"))
        for threshold in REL_WASTE_REDUCTION_CONSTRAINTS:
            feasible = group.loc[group["waste_rate_reduction"] >= threshold - WASTE_TOL]
            selected = feasible.sort_values("normalized_accounting_profit", ascending=False).head(1)
            rows.append(selection_row(scenario_id, calibration_mode, f"waste_rate_reduction_ge_{threshold:.2f}", selected, threshold, "relative_waste_reduction"))
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "sustainability_waste_constrained_policy_selection.csv", index=False)
    return out


def selection_row(scenario_id: str, calibration_mode: str, constraint_id: str, selected: pd.DataFrame, threshold: float, constraint_type: str) -> dict[str, Any]:
    if selected.empty:
        return {
            "scenario_id": scenario_id,
            "calibration_mode": calibration_mode,
            "constraint_id": constraint_id,
            "constraint_type": constraint_type,
            "threshold": threshold,
            "selection_status": "NO_FEASIBLE_POLICY",
            "selected_policy": "NO_FEASIBLE_POLICY",
        }
    row = selected.iloc[0]
    return {
        "scenario_id": scenario_id,
        "calibration_mode": calibration_mode,
        "constraint_id": constraint_id,
        "constraint_type": constraint_type,
        "threshold": threshold,
        "selection_status": "FEASIBLE_POLICY_SELECTED",
        "selected_policy": row["policy_id"],
        "normalized_accounting_profit": row["normalized_accounting_profit"],
        "waste_rate": row["waste_rate"],
        "normalized_profit_sacrifice": row["normalized_profit_sacrifice"],
        "waste_rate_reduction": row["waste_rate_reduction"],
    }


def efficiency(vs_zero_df: pd.DataFrame) -> pd.DataFrame:
    out = vs_zero_df.loc[vs_zero_df["waste_rate_reduction"] > WASTE_TOL].copy()
    out["efficiency_label"] = "MATERIAL_PROFIT_SACRIFICE"
    no_loss = out["normalized_profit_sacrifice"] <= MATERIAL_PROFIT_DENOM
    out.loc[no_loss, "efficiency_label"] = "WASTE_REDUCTION_WITHOUT_MATERIAL_PROFIT_LOSS"
    out["waste_reduction_efficiency"] = np.where(
        no_loss,
        np.nan,
        out["waste_rate_reduction"] / out["normalized_profit_sacrifice"].clip(lower=MATERIAL_PROFIT_DENOM),
    )
    out["normalized_profit_sacrifice_per_0p10_waste_rate_reduction"] = (
        out["normalized_profit_sacrifice"] / out["waste_rate_reduction"].clip(lower=MATERIAL_PROFIT_DENOM) * 0.10
    )
    out["physical_waste_units_reduced_per_normalized_profit_unit_sacrificed"] = np.where(
        no_loss,
        np.nan,
        out["physical_waste_reduction"] / out["normalized_profit_sacrifice"].clip(lower=MATERIAL_PROFIT_DENOM),
    )
    out.to_csv(TABLES_DIR / "sustainability_waste_reduction_efficiency.csv", index=False)
    return out


def scenario_classification(vs_zero_df: pd.DataFrame, nondom_df: pd.DataFrame, constrained_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in vs_zero_df.groupby(["scenario_id", "calibration_mode"]):
        scenario_id, calibration_mode = keys
        max_reduction = float(group["waste_rate_reduction"].max())
        candidates = group.loc[group["waste_rate_reduction"].eq(max_reduction)]
        profit_sacrifice = float(candidates["normalized_profit_sacrifice"].min()) if not candidates.empty else 0.0
        non_dom_count = int(
            len(nondom_df.loc[
                nondom_df["scenario_id"].eq(scenario_id)
                & nondom_df["calibration_mode"].eq(calibration_mode)
                & nondom_df["nondominated_status"].eq("NON_DOMINATED")
            ])
        )
        strong_feasible = constrained_df.loc[
            constrained_df["scenario_id"].eq(scenario_id)
            & constrained_df["calibration_mode"].eq(calibration_mode)
            & constrained_df["constraint_id"].eq("waste_rate_reduction_ge_0.20")
            & constrained_df["selection_status"].eq("FEASIBLE_POLICY_SELECTED")
        ]
        if max_reduction < 0.05:
            cls = "NO_FEASIBLE_STRONG_WASTE_REDUCTION"
        elif profit_sacrifice <= 0.05:
            cls = "LOW_COST_WASTE_REDUCTION"
        elif profit_sacrifice <= 0.15 and non_dom_count >= 2:
            cls = "MODERATE_PROFIT_WASTE_TRADEOFF"
        elif strong_feasible.empty:
            cls = "NO_FEASIBLE_STRONG_WASTE_REDUCTION"
        else:
            cls = "HIGH_COST_WASTE_REDUCTION"
        rows.append(
            {
                "scenario_id": scenario_id,
                "calibration_mode": calibration_mode,
                "maximum_waste_rate_reduction": max_reduction,
                "normalized_profit_sacrifice_at_max_reduction": profit_sacrifice,
                "non_dominated_policy_count": non_dom_count,
                "strong_waste_reduction_feasible": not strong_feasible.empty,
                "tradeoff_classification": cls,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "sustainability_scenario_tradeoff_classification.csv", index=False)
    return out


def calibration_transmission(summary_df: pd.DataFrame, nondom_df: pd.DataFrame, constrained_df: pd.DataFrame, eff_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario_id, group in summary_df.groupby("scenario_id"):
        obs = group.loc[group["calibration_mode"].eq("observed_calibration")]
        rec = group.loc[group["calibration_mode"].eq("recovered_calibration")]
        merged = rec.merge(obs, on="policy_id", suffixes=("_recovered", "_observed"))
        mean_abs_waste_change = float((merged["waste_rate_recovered"] - merged["waste_rate_observed"]).abs().mean()) if not merged.empty else np.nan
        mean_abs_profit_change = float((merged["normalized_accounting_profit_recovered"] - merged["normalized_accounting_profit_observed"]).abs().mean()) if not merged.empty else np.nan
        selection = constrained_df.loc[constrained_df["scenario_id"].eq(scenario_id)]
        wide_sel = selection.pivot_table(index="constraint_id", columns="calibration_mode", values="selected_policy", aggfunc="first")
        changed_selection_share = float((wide_sel.get("observed_calibration") != wide_sel.get("recovered_calibration")).mean()) if not wide_sel.empty else 0.0
        nd = nondom_df.loc[nondom_df["scenario_id"].eq(scenario_id)]
        nd_obs = set(nd.loc[nd["calibration_mode"].eq("observed_calibration") & nd["nondominated_status"].eq("NON_DOMINATED"), "policy_id"])
        nd_rec = set(nd.loc[nd["calibration_mode"].eq("recovered_calibration") & nd["nondominated_status"].eq("NON_DOMINATED"), "policy_id"])
        nd_change = 1.0 - len(nd_obs & nd_rec) / max(len(nd_obs | nd_rec), 1)
        if mean_abs_waste_change >= 0.05 or changed_selection_share >= 0.30 or nd_change >= 0.30:
            cls = "MATERIAL_SUSTAINABILITY_CALIBRATION_CHANGE"
        elif mean_abs_waste_change >= 0.01 or changed_selection_share >= 0.10 or nd_change >= 0.10:
            cls = "MODERATE_SUSTAINABILITY_CALIBRATION_CHANGE"
        else:
            cls = "LIMITED_SUSTAINABILITY_CALIBRATION_CHANGE"
        rows.append(
            {
                "scenario_id": scenario_id,
                "mean_absolute_waste_rate_change": mean_abs_waste_change,
                "mean_absolute_normalized_profit_change": mean_abs_profit_change,
                "waste_constraint_selection_change_share": changed_selection_share,
                "nondominated_membership_change": nd_change,
                "efficiency_rows_compared": int(len(eff_df.loc[eff_df["scenario_id"].eq(scenario_id)])),
                "calibration_transmission_classification": cls,
            }
        )
    out = pd.DataFrame(rows)
    overall = "LIMITED_SUSTAINABILITY_CALIBRATION_CHANGE"
    if (out["calibration_transmission_classification"] == "MATERIAL_SUSTAINABILITY_CALIBRATION_CHANGE").any():
        overall = "MATERIAL_SUSTAINABILITY_CALIBRATION_CHANGE"
    elif (out["calibration_transmission_classification"] == "MODERATE_SUSTAINABILITY_CALIBRATION_CHANGE").any():
        overall = "MODERATE_SUSTAINABILITY_CALIBRATION_CHANGE"
    out["overall_classification"] = overall
    out.to_csv(TABLES_DIR / "sustainability_calibration_transmission.csv", index=False)
    return out


def managerial_summary(summary_df: pd.DataFrame, vs_zero_df: pd.DataFrame, constrained_df: pd.DataFrame, eff_df: pd.DataFrame, transmission_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in summary_df.groupby(["scenario_id", "calibration_mode"]):
        scenario_id, calibration_mode = keys
        financial = group.sort_values("normalized_accounting_profit", ascending=False).iloc[0]
        waste_min = group.sort_values(["waste_rate", "normalized_accounting_profit"], ascending=[True, False]).iloc[0]
        eff_group = eff_df.loc[eff_df["scenario_id"].eq(scenario_id) & eff_df["calibration_mode"].eq(calibration_mode)]
        if eff_group.empty:
            low_cost_policy = "NO_POSITIVE_WASTE_REDUCTION"
        else:
            no_loss = eff_group.loc[eff_group["efficiency_label"].eq("WASTE_REDUCTION_WITHOUT_MATERIAL_PROFIT_LOSS")]
            low_cost_policy = str((no_loss if not no_loss.empty else eff_group.sort_values("waste_reduction_efficiency", ascending=False)).iloc[0]["policy_id"])
        constraints = constrained_df.loc[constrained_df["scenario_id"].eq(scenario_id) & constrained_df["calibration_mode"].eq(calibration_mode)]
        rows.append(
            {
                "scenario_id": scenario_id,
                "calibration_mode": calibration_mode,
                "financial_performance_max_policy": financial["policy_id"],
                "lowest_waste_policy": waste_min["policy_id"],
                "best_low_cost_waste_reduction_policy": low_cost_policy,
                "constraint_policy_map": "; ".join(f"{r.constraint_id}={r.selected_policy}" for r in constraints.itertuples()),
                "recommendation_changes_by_constraint": constraints["selected_policy"].nunique(dropna=True) > 1,
                "demand_calibration_changes_recommendation": bool(
                    transmission_df.loc[
                        transmission_df["scenario_id"].eq(scenario_id),
                        "calibration_transmission_classification",
                    ].iloc[0]
                    != "LIMITED_SUSTAINABILITY_CALIBRATION_CHANGE"
                ),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "sustainability_managerial_policy_summary.csv", index=False)
    return out


def save_figures(summary_df: pd.DataFrame, vs_zero_df: pd.DataFrame, nondom_df: pd.DataFrame, constrained_df: pd.DataFrame, eff_df: pd.DataFrame, class_df: pd.DataFrame, transmission_df: pd.DataFrame, managerial_df: pd.DataFrame) -> None:
    def save_current(name: str) -> None:
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / name, dpi=150)
        plt.close()

    rec = summary_df.loc[summary_df["calibration_mode"].eq("recovered_calibration")]
    fig, ax = plt.subplots(figsize=(9, 5))
    for scenario_id, group in rec.groupby("scenario_id"):
        ax.scatter(group["waste_rate"], group["normalized_accounting_profit"], label=scenario_id)
    ax.set_xlabel("Waste rate")
    ax.set_ylabel("Normalized accounting profit")
    ax.legend(fontsize=7)
    save_current("profit_vs_waste_scatter_by_scenario.png")

    nd = nondom_df.loc[nondom_df["calibration_mode"].eq("recovered_calibration")]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = nd["nondominated_status"].map({"NON_DOMINATED": "tab:green", "DOMINATED": "tab:red", "IDENTICAL_OR_NEAR_DUPLICATE": "tab:gray"})
    ax.scatter(nd["waste_rate"], nd["normalized_accounting_profit"], c=colors)
    ax.set_xlabel("Waste rate")
    ax.set_ylabel("Normalized accounting profit")
    save_current("empirical_nondominated_policy_set.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    rec_vs = vs_zero_df.loc[vs_zero_df["calibration_mode"].eq("recovered_calibration")]
    ax.scatter(rec_vs["normalized_profit_sacrifice"], rec_vs["waste_rate_reduction"])
    ax.set_xlabel("Normalized profit sacrifice")
    ax.set_ylabel("Waste-rate reduction")
    save_current("waste_reduction_vs_profit_sacrifice.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    rec_sel = constrained_df.loc[constrained_df["calibration_mode"].eq("recovered_calibration")]
    pd.crosstab(rec_sel["constraint_id"], rec_sel["selected_policy"]).plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Scenario count")
    save_current("waste_constrained_policy_recommendations.png")

    for metric, name, ylabel in [
        ("waste_rate", "policy_waste_rate_by_scenario.png", "Waste rate"),
        ("normalized_accounting_profit", "policy_normalized_profit_by_scenario.png", "Normalized profit"),
    ]:
        fig, ax = plt.subplots(figsize=(11, 5))
        pivot = rec.pivot_table(index="scenario_id", columns="policy_id", values=metric)
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        save_current(name)

    fig, ax = plt.subplots(figsize=(9, 5))
    if not eff_df.empty:
        rec_eff = eff_df.loc[eff_df["calibration_mode"].eq("recovered_calibration")]
        ax.scatter(rec_eff["normalized_profit_sacrifice_per_0p10_waste_rate_reduction"], rec_eff["waste_rate_reduction"])
    ax.set_xlabel("Profit sacrifice per 0.10 waste-rate reduction")
    ax.set_ylabel("Waste-rate reduction")
    save_current("waste_reduction_efficiency.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    transmission_df.set_index("scenario_id")["mean_absolute_waste_rate_change"].plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean absolute waste-rate change")
    save_current("observed_vs_recovered_calibration_tradeoff.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    pd.crosstab(class_df["scenario_id"], class_df["tradeoff_classification"]).plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Calibration count")
    save_current("scenario_tradeoff_classification.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    pd.crosstab(managerial_df["scenario_id"], managerial_df["financial_performance_max_policy"]).plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Calibration count")
    save_current("managerial_decision_map.png")


def main() -> None:
    ensure_dirs()
    selected = load_selected_scenarios()
    ppo_models = load_ppo_policies()
    episode_df = evaluate_policy_set(selected, ppo_models)
    summary_df = summarize(episode_df)
    vs_zero_df = vs_always_zero(summary_df)
    nondom_df = nondominated(summary_df)
    constrained_df = constrained_selection(vs_zero_df)
    eff_df = efficiency(vs_zero_df)
    class_df = scenario_classification(vs_zero_df, nondom_df, constrained_df)
    transmission_df = calibration_transmission(summary_df, nondom_df, constrained_df, eff_df)
    managerial_df = managerial_summary(summary_df, vs_zero_df, constrained_df, eff_df, transmission_df)
    save_figures(summary_df, vs_zero_df, nondom_df, constrained_df, eff_df, class_df, transmission_df, managerial_df)

    accounting_ok = bool(
        (episode_df["episode_conservation_error"].abs().max() <= 1e-6)
        and (episode_df["raw_financial_sum_error"].abs().max() <= 1e-6)
    )
    selected["accounting_validity_confirmed_from_tradeoff_run"] = accounting_ok
    selected.to_csv(TABLES_DIR / "sustainability_selected_tradeoff_scenarios.csv", index=False)
    status = "SUSTAINABILITY_TRADEOFF_ANALYSIS_COMPLETE" if len(selected) == 6 and accounting_ok else "SUSTAINABILITY_TRADEOFF_ANALYSIS_REQUIRES_REVISION"

    rec = vs_zero_df.loc[vs_zero_df["calibration_mode"].eq("recovered_calibration")]
    best_financial = summary_df.loc[summary_df["calibration_mode"].eq("recovered_calibration")].sort_values("normalized_accounting_profit", ascending=False).iloc[0]
    lowest_waste = summary_df.loc[summary_df["calibration_mode"].eq("recovered_calibration")].sort_values(["waste_rate", "normalized_accounting_profit"], ascending=[True, False]).iloc[0]
    best_eff = eff_df.loc[eff_df["calibration_mode"].eq("recovered_calibration")]
    if not best_eff.empty:
        no_loss = best_eff.loc[best_eff["efficiency_label"].eq("WASTE_REDUCTION_WITHOUT_MATERIAL_PROFIT_LOSS")]
        most_efficient = str((no_loss if not no_loss.empty else best_eff.sort_values("waste_reduction_efficiency", ascending=False)).iloc[0]["policy_id"])
    else:
        most_efficient = "NO_POSITIVE_WASTE_REDUCTION"
    max_reduction_row = rec.sort_values("waste_rate_reduction", ascending=False).iloc[0]
    report = {
        "status": status,
        "scenarios_evaluated": int(selected["scenario_id"].nunique()),
        "policies_evaluated": int(episode_df["policy_id"].nunique()),
        "best_financial_policy": str(best_financial["policy_id"]),
        "lowest_waste_policy": str(lowest_waste["policy_id"]),
        "maximum_waste_reduction": float(max_reduction_row["waste_rate_reduction"]),
        "associated_normalized_profit_sacrifice": float(max_reduction_row["normalized_profit_sacrifice"]),
        "most_efficient_waste_reduction_policy": most_efficient,
        "calibration_transmission_classification": str(transmission_df["overall_classification"].iloc[0]),
        "no_test_split_used": True,
        "ppo_training_performed": False,
        "interpretation": "Empirical trade-off analysis over a finite evaluated policy set; not a complete mathematical Pareto frontier. Waste and shelf life are simulated, costs are normalized assumptions, PPO policies were trained for financial reward, no sustainability PPO was trained, and results support controlled decision analysis rather than direct retailer deployment.",
        "files_created": [
            "outputs/tables/sustainability_selected_tradeoff_scenarios.csv",
            "outputs/tables/sustainability_tradeoff_episode_level.csv",
            "outputs/tables/sustainability_tradeoff_vs_always_zero.csv",
            "outputs/tables/sustainability_nondominated_policy_set.csv",
            "outputs/tables/sustainability_waste_constrained_policy_selection.csv",
            "outputs/tables/sustainability_waste_reduction_efficiency.csv",
            "outputs/tables/sustainability_scenario_tradeoff_classification.csv",
            "outputs/tables/sustainability_calibration_transmission.csv",
            "outputs/tables/sustainability_managerial_policy_summary.csv",
        ],
    }
    (TABLES_DIR / "sustainability_tradeoff_analysis_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(status)


if __name__ == "__main__":
    main()
