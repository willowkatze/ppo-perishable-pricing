from __future__ import annotations

import json
import math
import re
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

from pricing_env_operational import ACTION_MARKDOWNS, OperationalPerishablePricingEnv  # noqa: E402
from train_ppo_operational import DEFAULT_SCENARIO_ID, MODELS_DIR, TABLES_DIR, CONFIGS_DIR, make_env  # noqa: E402

FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "ppo_checkpoint_validation"
AGENTS = {
    "observed_financial": "observed_calibration",
    "recovered_financial": "recovered_calibration",
}
BASELINES = [
    "always_0pct",
    "always_5pct",
    "always_10pct",
    "expiry_threshold_rule",
    "inventory_coverage_rule",
    "random_policy",
]
METRICS = [
    "episode_return",
    "normalized_accounting_profit",
    "raw_accounting_profit",
    "revenue",
    "waste_rate",
    "physical_waste_units",
    "sell_through_rate",
    "stockout_rate",
    "average_markdown",
]


@dataclass(frozen=True)
class PolicySpec:
    policy_id: str
    agent_id: str
    calibration_mode: str
    kind: str
    model_path: Path | None
    checkpoint_step: int | None


def ensure_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_or_create_validation_manifest(episodes: int = 20) -> pd.DataFrame:
    path = CONFIGS_DIR / "ppo_validation_episode_manifest.csv"
    if path.exists():
        existing = pd.read_csv(path)
        required = {"episode_id", "split", "store_id", "product_id", "start_date", "scenario_id", "age_profile_seed", "demand_noise_seed"}
        if existing.shape[0] >= episodes and required.issubset(existing.columns):
            return existing.head(episodes).copy()
    env = OperationalPerishablePricingEnv(split="validation", scenario_id=DEFAULT_SCENARIO_ID, random_seed=9000)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    attempts = 0
    while len(rows) < episodes and attempts < episodes * 30:
        _, info = env.reset(seed=9000 + attempts)
        key = (str(info["store_id"]), str(info["product_id"]), str(info["date"]))
        attempts += 1
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "episode_id": f"validation_{len(rows):04d}",
                "split": "validation",
                "store_id": key[0],
                "product_id": key[1],
                "start_date": key[2],
                "scenario_id": DEFAULT_SCENARIO_ID,
                "age_profile_seed": 9000 + attempts,
                "demand_noise_seed": 19000 + attempts,
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(path, index=False)
    return manifest


def step_from_name(path: Path) -> int | None:
    match = re.search(r"_(\d+)_steps\.zip$", path.name)
    return int(match.group(1)) if match else None


def discover_policy_specs() -> tuple[list[PolicySpec], pd.DataFrame]:
    specs: list[PolicySpec] = []
    missing: list[dict[str, Any]] = []
    for agent_id, calibration_mode in AGENTS.items():
        agent_dir = MODELS_DIR / agent_id / "seed_42"
        expected_10k = agent_dir / "original_10k_model.zip"
        if expected_10k.exists():
            specs.append(PolicySpec(f"{agent_id}__original_10k", agent_id, calibration_mode, "ppo", expected_10k, 10_000))
        else:
            missing.append(
                {
                    "agent_id": agent_id,
                    "expected_policy": "original_10k",
                    "status": "missing",
                    "note": "The 10k final_model.zip was overwritten by the 30k resume run; no immutable 10k copy was found.",
                }
            )
        checkpoint_paths = sorted(
            (agent_dir / "checkpoints").glob(f"pilot_{agent_id}_*_steps.zip"),
            key=lambda path: step_from_name(path) or -1,
        )
        for path in checkpoint_paths:
            step = step_from_name(path)
            specs.append(PolicySpec(f"{agent_id}__checkpoint_{step}", agent_id, calibration_mode, "ppo", path, step))
        for label in ["best_model", "final_model"]:
            path = agent_dir / f"{label}.zip"
            if path.exists():
                specs.append(PolicySpec(f"{agent_id}__{label}", agent_id, calibration_mode, "ppo", path, None))
            else:
                missing.append({"agent_id": agent_id, "expected_policy": label, "status": "missing", "note": str(path)})
    for baseline in BASELINES:
        specs.append(PolicySpec(baseline, "baseline", "recovered_calibration", "baseline", None, None))
    missing_df = pd.DataFrame(missing)
    if not missing_df.empty:
        missing_df.to_csv(TABLES_DIR / "ppo_checkpoint_missing_artifacts.csv", index=False)
    return specs, missing_df


def baseline_action(policy_id: str, obs: np.ndarray, episode_day: int, rng: np.random.Generator) -> int:
    if policy_id == "always_0pct":
        return 0
    if policy_id == "always_5pct":
        return 1
    if policy_id == "always_10pct":
        return 2
    if policy_id == "random_policy":
        return int(rng.integers(0, 6))
    if policy_id == "expiry_threshold_rule":
        frac_within_two = float(obs[23]) if obs.shape[0] > 23 else 0.0
        if frac_within_two >= 0.50:
            return 5
        if frac_within_two >= 0.25:
            return 3
        if frac_within_two >= 0.10:
            return 2
        return 0
    if policy_id == "inventory_coverage_rule":
        coverage = float(obs[1]) if obs.shape[0] > 1 else 0.0
        if coverage >= 1.50:
            return 4
        if coverage >= 1.10:
            return 3
        if coverage >= 0.80:
            return 2
        return 0
    raise ValueError(policy_id)


def safe_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric if np.isfinite(numeric) else 0.0


def action_entropy(action_counts: dict[int, int]) -> float:
    total = sum(action_counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in action_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log(p)
    return float(entropy)


def load_model_and_vecnorm(spec: PolicySpec) -> tuple[PPO, VecNormalize | None]:
    assert spec.model_path is not None
    agent_dir = MODELS_DIR / spec.agent_id / "seed_42"
    vec_path = agent_dir / "vecnormalize.pkl"
    raw_env = DummyVecEnv(
        [
            lambda: make_env(
                split="validation",
                calibration_mode=spec.calibration_mode,
                reward_mode="financial",
                lambda_waste=0.10,
                scenario_id=DEFAULT_SCENARIO_ID,
                deterministic_demand=True,
                seed=42,
                monitor_dir=None,
            )
        ]
    )
    vecnorm = VecNormalize.load(str(vec_path), raw_env) if vec_path.exists() else None
    if vecnorm is not None:
        vecnorm.training = False
        vecnorm.norm_reward = False
    return PPO.load(spec.model_path), vecnorm


def evaluate_policy(spec: PolicySpec, manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    model: PPO | None = None
    vecnorm: VecNormalize | None = None
    if spec.kind == "ppo":
        model, vecnorm = load_model_and_vecnorm(spec)
    rng = np.random.default_rng(42)
    env = OperationalPerishablePricingEnv(
        split="validation",
        calibration_mode=spec.calibration_mode,
        reward_mode="financial",
        lambda_waste=0.10,
        scenario_id=DEFAULT_SCENARIO_ID,
        deterministic_demand=True,
        random_seed=42,
    )
    for _, episode in manifest.iterrows():
        options = {
            "store_id": episode["store_id"],
            "product_id": episode["product_id"],
            "start_date": episode["start_date"],
        }
        obs, _ = env.reset(seed=int(episode["age_profile_seed"]), options=options)
        terminated = False
        truncated = False
        total_return = 0.0
        first_markdown_day = -1
        action_counts = {action: 0 for action in ACTION_MARKDOWNS}
        final_info: dict[str, Any] = {}
        day = 0
        while not (terminated or truncated):
            state_features = {
                "fraction_expiring_today": safe_float(obs[22]) if obs.shape[0] > 22 else 0.0,
                "fraction_expiring_within_two_days": safe_float(obs[23]) if obs.shape[0] > 23 else 0.0,
                "inventory_coverage": safe_float(obs[1]) if obs.shape[0] > 1 else 0.0,
                "predicted_demand": safe_float(obs[25]) if obs.shape[0] > 25 else 0.0,
                "episode_day": day,
            }
            if model is None:
                action = baseline_action(spec.policy_id, obs, day, rng)
            else:
                model_obs = obs.astype(np.float32)
                if vecnorm is not None:
                    model_obs = vecnorm.normalize_obs(model_obs[None, :])[0]
                predicted, _ = model.predict(model_obs, deterministic=True)
                action = int(np.asarray(predicted).item())
            markdown = ACTION_MARKDOWNS[action]
            if first_markdown_day < 0 and markdown > 0:
                first_markdown_day = day
            state_rows.append(
                {
                    "policy_id": spec.policy_id,
                    "agent_id": spec.agent_id,
                    "episode_id": episode["episode_id"],
                    "action": action,
                    "markdown": markdown,
                    **state_features,
                }
            )
            obs, reward, terminated, truncated, info = env.step(action)
            action_counts[action] += 1
            total_return += float(reward)
            final_info = info
            day += 1
        total_actions = sum(action_counts.values())
        one_action_only = int(sum(1 for count in action_counts.values() if count > 0) == 1)
        rows.append(
            {
                "policy_id": spec.policy_id,
                "agent_id": spec.agent_id,
                "kind": spec.kind,
                "checkpoint_step": spec.checkpoint_step,
                "episode_id": episode["episode_id"],
                "store_id": episode["store_id"],
                "product_id": episode["product_id"],
                "start_date": episode["start_date"],
                "episode_return": total_return,
                "normalized_accounting_profit": safe_float(final_info.get("normalized_accounting_profit")),
                "raw_accounting_profit": safe_float(final_info.get("accounting_profit")),
                "revenue": safe_float(final_info.get("cumulative_revenue")),
                "waste_rate": safe_float(final_info.get("final_waste_rate")),
                "physical_waste_units": safe_float(final_info.get("cumulative_waste")),
                "sell_through_rate": safe_float(final_info.get("final_sell_through_rate")),
                "stockout_rate": safe_float(final_info.get("stockout_indicator")),
                "average_markdown": safe_float(final_info.get("average_markdown")),
                "first_markdown_day": first_markdown_day,
                "markdown_timing_relative_to_expiry": total_actions - first_markdown_day if first_markdown_day >= 0 else -1,
                "action_entropy": action_entropy(action_counts),
                "one_action_episode": one_action_only,
                "weak_support_rate": safe_float(final_info.get("weak_support_flag")),
                "extrapolation_rate": safe_float(final_info.get("extrapolation_flag")),
                **{f"action_{action}_share": action_counts[action] / max(total_actions, 1) for action in ACTION_MARKDOWNS},
            }
        )
    if vecnorm is not None:
        vecnorm.close()
    return pd.DataFrame(rows), pd.DataFrame(state_rows)


def summarize_episode_results(episode_df: pd.DataFrame) -> pd.DataFrame:
    action_cols = [col for col in episode_df.columns if col.startswith("action_") and col.endswith("_share")]
    summary = episode_df.groupby(["policy_id", "agent_id", "kind", "checkpoint_step"], dropna=False)[
        METRICS + ["first_markdown_day", "markdown_timing_relative_to_expiry", "action_entropy", "one_action_episode", "weak_support_rate", "extrapolation_rate"] + action_cols
    ].mean().reset_index()
    return summary


def classify_policy(row: pd.Series) -> str:
    if row.get("action_5_share", 0.0) >= 0.95:
        return "MAX_ACTION_COLLAPSE"
    if row.get("action_0_share", 0.0) >= 0.95:
        return "ZERO_ACTION_COLLAPSE"
    if row.get("action_entropy", 0.0) < 0.20 or row.get("one_action_episode", 0.0) > 0.80:
        return "UNSTABLE_POLICY"
    if row.get("action_0_share", 0.0) >= 0.70 and row.get("action_entropy", 0.0) >= 0.20:
        return "MOSTLY_ZERO_BUT_STATE_DEPENDENT"
    return "STATE_DEPENDENT_POLICY"


def paired_comparisons(episode_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    best_by_agent: dict[str, str] = {}
    selectable = summary_df.loc[summary_df["kind"].eq("ppo")].copy()
    if not selectable.empty:
        selectable["classification"] = selectable.apply(classify_policy, axis=1)
        valid = selectable.loc[~selectable["classification"].isin(["ZERO_ACTION_COLLAPSE", "MAX_ACTION_COLLAPSE", "UNSTABLE_POLICY"])]
        if valid.empty:
            valid = selectable
        for agent_id, group in valid.groupby("agent_id"):
            best_by_agent[agent_id] = str(group.sort_values(["episode_return", "normalized_accounting_profit"], ascending=False).iloc[0]["policy_id"])
    rng = np.random.default_rng(2026)
    for policy_id, policy_data in episode_df.groupby("policy_id"):
        agent_id = str(policy_data["agent_id"].iloc[0])
        references = ["always_0pct"]
        if agent_id in AGENTS:
            original = f"{agent_id}__original_10k"
            if original in set(episode_df["policy_id"]):
                references.append(original)
            if agent_id in best_by_agent:
                references.append(best_by_agent[agent_id])
        for ref_id in dict.fromkeys(references):
            if ref_id == policy_id or ref_id not in set(episode_df["policy_id"]):
                continue
            merged = policy_data.merge(
                episode_df.loc[episode_df["policy_id"].eq(ref_id)],
                on="episode_id",
                suffixes=("_policy", "_reference"),
            )
            if merged.empty:
                continue
            for metric in METRICS:
                diff = merged[f"{metric}_policy"] - merged[f"{metric}_reference"]
                boot = [
                    float(rng.choice(diff.to_numpy(), size=len(diff), replace=True).mean())
                    for _ in range(500)
                ]
                rows.append(
                    {
                        "policy_id": policy_id,
                        "reference_policy_id": ref_id,
                        "metric": metric,
                        "paired_mean_difference": float(diff.mean()),
                        "paired_median_difference": float(diff.median()),
                        "bootstrap_ci_low": float(np.percentile(boot, 2.5)),
                        "bootstrap_ci_high": float(np.percentile(boot, 97.5)),
                        "win_share": float((diff > 0).mean()),
                        "tie_share": float((diff == 0).mean()),
                        "loss_share": float((diff < 0).mean()),
                    }
                )
    return pd.DataFrame(rows)


def state_dependence(state_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy_id, group in state_df.groupby("policy_id"):
        for feature in ["fraction_expiring_today", "fraction_expiring_within_two_days", "inventory_coverage", "predicted_demand", "episode_day"]:
            values = group[feature].to_numpy()
            markdowns = group["markdown"].to_numpy()
            if np.std(values) == 0 or np.std(markdowns) == 0:
                corr = 0.0
            else:
                corr = float(np.corrcoef(values, markdowns)[0, 1])
            rows.append(
                {
                    "policy_id": policy_id,
                    "feature": feature,
                    "markdown_correlation": corr if np.isfinite(corr) else 0.0,
                    "mean_markdown_low_feature": float(group.loc[group[feature] <= group[feature].median(), "markdown"].mean()),
                    "mean_markdown_high_feature": float(group.loc[group[feature] > group[feature].median(), "markdown"].mean()) if (group[feature] > group[feature].median()).any() else 0.0,
                }
            )
    return pd.DataFrame(rows)


def model_selection(summary_df: pd.DataFrame, paired_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    summary = summary_df.copy()
    summary["classification"] = summary.apply(classify_policy, axis=1)
    for agent_id, group in summary.loc[summary["kind"].eq("ppo")].groupby("agent_id"):
        valid = group.loc[~group["classification"].isin(["ZERO_ACTION_COLLAPSE", "MAX_ACTION_COLLAPSE", "UNSTABLE_POLICY"])].copy()
        review_reason = ""
        if valid.empty:
            valid = group.copy()
            review_reason = "No checkpoint passed behavior filters; selecting requires review."
        selected = valid.sort_values(
            ["episode_return", "normalized_accounting_profit", "action_entropy"],
            ascending=False,
        ).iloc[0]
        rows.append(
            {
                "agent_id": agent_id,
                "selected_policy_id": selected["policy_id"],
                "classification": selected["classification"],
                "mean_episode_return": selected["episode_return"],
                "mean_normalized_accounting_profit": selected["normalized_accounting_profit"],
                "mean_raw_accounting_profit": selected["raw_accounting_profit"],
                "action_entropy": selected["action_entropy"],
                "zero_action_share": selected.get("action_0_share", 0.0),
                "selection_note": review_reason or "Selected using validation metrics after finite/accounting and behavior checks.",
            }
        )
    return pd.DataFrame(rows)


def make_figures(summary_df: pd.DataFrame, paired_df: pd.DataFrame, state_df: pd.DataFrame) -> None:
    plot_df = summary_df.sort_values(["agent_id", "checkpoint_step", "policy_id"])
    for metric, filename, ylabel in [
        ("episode_return", "validation_performance_by_checkpoint.png", "Episode return"),
        ("raw_accounting_profit", "accounting_profit_by_checkpoint.png", "Raw accounting profit"),
        ("waste_rate", "waste_rate_by_checkpoint.png", "Waste rate"),
        ("action_entropy", "action_entropy_over_training.png", "Action entropy"),
    ]:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(plot_df["policy_id"], plot_df[metric])
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=80)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / filename, dpi=160)
        plt.close(fig)
    action_cols = [col for col in summary_df.columns if col.startswith("action_") and col.endswith("_share")]
    if action_cols:
        fig, ax = plt.subplots(figsize=(12, 5))
        bottom = np.zeros(len(plot_df))
        x = np.arange(len(plot_df))
        for col in action_cols:
            ax.bar(x, plot_df[col], bottom=bottom, label=col.replace("_share", ""))
            bottom += plot_df[col].to_numpy()
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["policy_id"], rotation=80, ha="right")
        ax.set_ylabel("Action share")
        ax.legend(ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "action_distribution_by_checkpoint.png", dpi=160)
        plt.close(fig)
    profit_diff = paired_df.loc[
        paired_df["reference_policy_id"].eq("always_0pct") & paired_df["metric"].eq("raw_accounting_profit")
    ]
    if not profit_diff.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(profit_diff["policy_id"], profit_diff["paired_mean_difference"])
        ax.axhline(0, color="black", linewidth=1)
        ax.tick_params(axis="x", rotation=80)
        ax.set_ylabel("Paired profit difference vs always_0pct")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "paired_profit_difference_vs_always_zero.png", dpi=160)
        plt.close(fig)
    state_plot = state_df.loc[state_df["feature"].isin(["fraction_expiring_within_two_days", "inventory_coverage"])]
    if not state_plot.empty:
        fig, ax = plt.subplots(figsize=(11, 4))
        labels = state_plot["policy_id"] + " | " + state_plot["feature"]
        ax.bar(labels, state_plot["markdown_correlation"])
        ax.tick_params(axis="x", rotation=80)
        ax.set_ylabel("Corr(feature, markdown)")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "state_dependent_markdown_behavior.png", dpi=160)
        plt.close(fig)
    obs_rec = summary_df.loc[summary_df["agent_id"].isin(AGENTS)]
    if not obs_rec.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        for agent_id, group in obs_rec.groupby("agent_id"):
            ax.scatter(group["checkpoint_step"].fillna(0), group["episode_return"], label=agent_id)
        ax.set_xlabel("Checkpoint step; 0 denotes best/final")
        ax.set_ylabel("Episode return")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "observed_vs_recovered_checkpoint_comparison.png", dpi=160)
        plt.close(fig)


def main() -> None:
    ensure_dirs()
    manifest = load_or_create_validation_manifest()
    specs, missing_df = discover_policy_specs()
    episode_tables: list[pd.DataFrame] = []
    state_tables: list[pd.DataFrame] = []
    for spec in specs:
        if spec.kind == "ppo" and spec.model_path is None:
            continue
        print(f"Evaluating {spec.policy_id}")
        episode_df, state_df = evaluate_policy(spec, manifest)
        episode_tables.append(episode_df)
        state_tables.append(state_df)
    all_episodes = pd.concat(episode_tables, ignore_index=True)
    all_states = pd.concat(state_tables, ignore_index=True)
    summary = summarize_episode_results(all_episodes)
    summary["classification"] = summary.apply(classify_policy, axis=1)
    paired = paired_comparisons(all_episodes, summary)
    action_diag = summary[
        ["policy_id", "agent_id", "kind", "checkpoint_step", "classification", "action_entropy", "one_action_episode"]
        + [col for col in summary.columns if col.startswith("action_") and col.endswith("_share")]
    ].copy()
    state_dep = state_dependence(all_states)
    selection = model_selection(summary, paired)
    all_episodes.to_csv(TABLES_DIR / "ppo_checkpoint_level_evaluation.csv", index=False)
    paired.to_csv(TABLES_DIR / "ppo_checkpoint_paired_comparisons.csv", index=False)
    action_diag.to_csv(TABLES_DIR / "ppo_checkpoint_action_diagnostics.csv", index=False)
    state_dep.to_csv(TABLES_DIR / "ppo_checkpoint_state_dependence.csv", index=False)
    selection.to_csv(TABLES_DIR / "ppo_checkpoint_model_selection.csv", index=False)
    make_figures(summary, paired, state_dep)
    status = "PPO_FINANCIAL_CHECKPOINT_REQUIRES_REVIEW"
    if not selection.empty and set(selection["agent_id"]) >= set(AGENTS):
        recovered = selection.loc[selection["agent_id"].eq("recovered_financial")]
        recovered_zero_equiv = False
        if not recovered.empty:
            selected_policy = str(recovered.iloc[0]["selected_policy_id"])
            zero_cmp = paired.loc[
                paired["policy_id"].eq(selected_policy)
                & paired["reference_policy_id"].eq("always_0pct")
                & paired["metric"].eq("episode_return")
            ]
            if not zero_cmp.empty:
                row = zero_cmp.iloc[0]
                recovered_zero_equiv = abs(float(row["paired_mean_difference"])) < 1e-9 and float(row["tie_share"]) >= 0.95
        if recovered_zero_equiv:
            status = "PPO_FINANCIAL_POLICY_EQUALS_ZERO_BASELINE"
        else:
            status = "PPO_FINANCIAL_CHECKPOINT_SELECTED"
    report = {
        "status": status,
        "policies_evaluated": int(all_episodes["policy_id"].nunique()),
        "episodes_per_policy": int(manifest.shape[0]),
        "missing_artifacts": missing_df.to_dict("records") if not missing_df.empty else [],
        "selected_models": selection.to_dict("records"),
        "outputs": [
            "outputs/tables/ppo_checkpoint_level_evaluation.csv",
            "outputs/tables/ppo_checkpoint_paired_comparisons.csv",
            "outputs/tables/ppo_checkpoint_action_diagnostics.csv",
            "outputs/tables/ppo_checkpoint_state_dependence.csv",
            "outputs/tables/ppo_checkpoint_model_selection.csv",
        ],
    }
    (TABLES_DIR / "ppo_checkpoint_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(status)


if __name__ == "__main__":
    main()
