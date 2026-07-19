"""
Complete the pre-specified STANDARD_DQN multi-seed validation experiment.

This script does not change the DQN configuration, HIGH_RISK_B population,
validation manifest, environment, reward, accounting, action space, or baseline.
It uses only validation-locked evaluation and never uses the test split.

Recommended workflow from project root:
    python -u src/complete_standard_dqn_multiseed.py --mode train_remaining
    python -u src/complete_standard_dqn_multiseed.py --mode evaluate

If seed 123 or 456 is interrupted, rerun the train command; it resumes from the
latest STANDARD_DQN checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_dqn_high_risk_b import (
    BASELINE_POLICY,
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    CALIBRATION_MODE,
    CONFIGS_DIR,
    DISTILLED_POLICY,
    FIGURES_DIR,
    MODELS_DIR,
    PLANNING_POLICY,
    POPULATION_ID,
    PROJECT_ROOT,
    SELECTED_PPO_POLICY,
    TABLES_DIR,
    TIE_TOL,
    DQN_CONFIGS,
    audit_environment_compatibility,
    bootstrap_ci,
    checkpoint_records,
    evaluate_checkpoints,
    load_training_episode_table,
    load_validation_manifest,
    safe_float,
    sha256,
    train_one,
    training_population_audit,
)

STANDARD_CONFIG = next(cfg for cfg in DQN_CONFIGS if cfg.config_id == "STANDARD_DQN")
STANDARD_SEEDS = [42, 123, 456]
REMAINING_SEEDS = [123, 456]
PLANNING_LOCKED_GAIN = 0.010792
WIN_SHARE_THRESHOLD = 0.60
MAX_TIMESTEPS = 30_000
CHECKPOINT_EVERY = 5_000


def ensure_dirs() -> None:
    for path in [TABLES_DIR, CONFIGS_DIR, FIGURES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def action_counts_from_sequences(sequences: pd.Series) -> dict[str, float]:
    actions: list[int] = []
    for seq in sequences.dropna():
        actions.extend([int(x) for x in str(seq).split("|") if x != ""])
    if not actions:
        return {f"action_{i}_share": np.nan for i in range(6)}
    counts = pd.Series(actions).value_counts(normalize=True)
    return {f"action_{i}_share": float(counts.get(i, 0.0)) for i in range(6)}


def checkpoint_sort_value(value: Any) -> int:
    if str(value) == "final":
        return 999_999
    return int(float(value))


def add_action_distribution(summary: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        policy_id = str(row["policy_id"])
        subset = results.loc[results["policy_id"].astype(str).eq(policy_id)]
        rows.append({**row.to_dict(), **action_counts_from_sequences(subset.get("action_sequence", pd.Series(dtype=str)))})
    return pd.DataFrame(rows)


def select_checkpoint_per_seed(summary: pd.DataFrame) -> pd.DataFrame:
    dqn = summary.loc[
        summary["model_family"].astype(str).eq("DQN")
        & summary["config_id"].astype(str).eq("STANDARD_DQN")
        & summary["seed"].notna()
    ].copy()
    if dqn.empty:
        return dqn
    for col in [
        "paired_mean_gain_vs_always_0pct",
        "win_share",
        "worst_decile_paired_gain",
        "maximum_episode_loss",
        "action_entropy",
        "waste_rate_difference",
        "dominant_action_share",
    ]:
        dqn[col] = numeric(dqn[col])
    dqn["accounting_valid_bool"] = dqn["accounting_valid"].astype(str).str.lower().eq("true")
    dqn["collapse_bool"] = dqn["collapse_flag"].astype(str).str.lower().eq("true")
    dqn["state_dependent_bool"] = dqn["state_dependence"].astype(str).eq("STATE_DEPENDENT")
    dqn["positive_gain_bool"] = dqn["paired_mean_gain_vs_always_0pct"] > 0
    dqn["acceptable_worst_decile_bool"] = dqn["worst_decile_paired_gain"] > -0.08
    dqn["checkpoint_sort"] = dqn["checkpoint_timesteps"].map(checkpoint_sort_value)
    dqn["selection_score"] = (
        dqn["accounting_valid_bool"].astype(int) * 1000
        + dqn["positive_gain_bool"].astype(int) * 100
        + dqn["acceptable_worst_decile_bool"].astype(int) * 50
        + dqn["state_dependent_bool"].astype(int) * 25
        + (~dqn["collapse_bool"]).astype(int) * 10
        + dqn["win_share"].fillna(0.0)
    )
    selected = (
        dqn.sort_values(
            [
                "seed",
                "selection_score",
                "paired_mean_gain_vs_always_0pct",
                "win_share",
                "worst_decile_paired_gain",
                "checkpoint_sort",
            ],
            ascending=[True, False, False, False, False, True],
        )
        .groupby("seed", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    selected["selected_for_seed"] = True
    return selected


def build_seed_episode_differences(results: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    baseline = results.loc[results["policy_id"].astype(str).eq(BASELINE_POLICY)]
    planning = results.loc[results["policy_id"].astype(str).eq(PLANNING_POLICY)]
    rows = []
    for _, sel in selected.iterrows():
        policy_id = str(sel["policy_id"])
        policy = results.loc[results["policy_id"].astype(str).eq(policy_id)]
        merged = policy.merge(baseline, on="episode_id", suffixes=("_dqn", "_baseline"))
        merged = merged.merge(
            planning[["episode_id", "normalized_profit", "waste_rate", "sell_through", "action_sequence"]],
            on="episode_id",
            how="left",
        ).rename(
            columns={
                "normalized_profit": "normalized_profit_planning",
                "waste_rate": "waste_rate_planning",
                "sell_through": "sell_through_planning",
                "action_sequence": "action_sequence_planning",
            }
        )
        for _, row in merged.iterrows():
            diff = safe_float(row.get("normalized_profit_dqn")) - safe_float(row.get("normalized_profit_baseline"))
            rows.append(
                {
                    "seed": int(float(sel["seed"])),
                    "selected_policy_id": policy_id,
                    "selected_checkpoint": sel["checkpoint_timesteps"],
                    "episode_id": row["episode_id"],
                    "scenario_id": row.get("scenario_id_dqn", row.get("scenario_id")),
                    "dqn_normalized_profit": safe_float(row.get("normalized_profit_dqn")),
                    "baseline_normalized_profit": safe_float(row.get("normalized_profit_baseline")),
                    "planning_normalized_profit": safe_float(row.get("normalized_profit_planning")),
                    "dqn_minus_baseline": diff,
                    "dqn_waste_rate": safe_float(row.get("waste_rate_dqn")),
                    "baseline_waste_rate": safe_float(row.get("waste_rate_baseline")),
                    "planning_waste_rate": safe_float(row.get("waste_rate_planning")),
                    "dqn_sell_through": safe_float(row.get("sell_through_dqn")),
                    "baseline_sell_through": safe_float(row.get("sell_through_baseline")),
                    "planning_sell_through": safe_float(row.get("sell_through_planning")),
                    "dqn_action_sequence": row.get("action_sequence_dqn"),
                    "planning_action_sequence": row.get("action_sequence_planning"),
                    "baseline_action_sequence": row.get("action_sequence_baseline"),
                }
            )
    diffs = pd.DataFrame(rows)
    diffs.to_csv(TABLES_DIR / "dqn_seed_by_episode_profit_differences.csv", index=False)
    return diffs


def pooled_results(selected: pd.DataFrame, episode_diffs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if episode_diffs.empty:
        pooled = pd.DataFrame()
    else:
        diff = numeric(episode_diffs["dqn_minus_baseline"])
        ci_low, ci_high = bootstrap_ci(diff)
        rows.append(
            {
                "aggregation": "pooled_selected_standard_dqn",
                "seed_count": int(selected["seed"].nunique()) if not selected.empty else 0,
                "episode_comparison_count": int(len(episode_diffs)),
                "pooled_mean_gain": float(diff.mean()),
                "pooled_median_gain": float(diff.median()),
                "pooled_ci_low": ci_low,
                "pooled_ci_high": ci_high,
                "pooled_win_share": float((diff > TIE_TOL).mean()),
                "pooled_tie_share": float((diff.abs() <= TIE_TOL).mean()),
                "pooled_loss_share": float((diff < -TIE_TOL).mean()),
                "pooled_waste_difference": float((episode_diffs["dqn_waste_rate"] - episode_diffs["baseline_waste_rate"]).mean()),
                "pooled_sell_through_difference": float((episode_diffs["dqn_sell_through"] - episode_diffs["baseline_sell_through"]).mean()),
                "planning_gain_capture_ratio": float(diff.mean() / PLANNING_LOCKED_GAIN) if PLANNING_LOCKED_GAIN > 0 else np.nan,
                "worst_decile_pooled_gain": float(diff.sort_values().head(max(1, math.ceil(len(diff) * 0.10))).mean()),
                "maximum_episode_loss": float(diff.min()),
            }
        )
        pooled = pd.DataFrame(rows)
    pooled.to_csv(TABLES_DIR / "dqn_pooled_validation_results.csv", index=False)
    return pooled


def seed_stability(selected: pd.DataFrame, pooled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in selected.iterrows():
        rows.append(
            {
                "seed": int(float(row["seed"])),
                "selected_checkpoint": row["checkpoint_timesteps"],
                "selected_policy_id": row["policy_id"],
                "mean_gain": safe_float(row["paired_mean_gain_vs_always_0pct"]),
                "median_gain": safe_float(row["paired_median_gain"]),
                "win_rate": safe_float(row["win_share"]),
                "waste_difference": safe_float(row["waste_rate_difference"]),
                "worst_decile_gain": safe_float(row["worst_decile_paired_gain"]),
                "action_entropy": safe_float(row["action_entropy"]),
                "state_dependence": row["state_dependence"],
                "collapse_flag": bool(str(row["collapse_flag"]).lower() == "true"),
                "accounting_valid": bool(str(row["accounting_valid"]).lower() == "true"),
                "planning_gain_capture_ratio": safe_float(row["paired_mean_gain_vs_always_0pct"]) / PLANNING_LOCKED_GAIN,
            }
        )
    stability = pd.DataFrame(rows)
    if not stability.empty:
        positive_count = int((stability["mean_gain"] > 0).sum())
        stability["positive_gain_seed_count"] = positive_count
        stability["positive_gain_seed_share"] = positive_count / max(len(stability), 1)
        stability["direction_consistency"] = "PASS" if positive_count >= 2 else "FAIL"
        stability["across_seed_mean_gain"] = float(stability["mean_gain"].mean())
        stability["across_seed_std_gain"] = float(stability["mean_gain"].std(ddof=1)) if len(stability) > 1 else 0.0
        if not pooled.empty:
            stability["pooled_mean_gain"] = safe_float(pooled["pooled_mean_gain"].iloc[0])
            stability["pooled_win_share"] = safe_float(pooled["pooled_win_share"].iloc[0])
    stability.to_csv(TABLES_DIR / "dqn_seed_stability.csv", index=False)
    return stability


def classify_loss(row: pd.Series) -> str:
    dqn_actions = [int(x) for x in str(row.get("dqn_action_sequence", "")).split("|") if x != ""]
    plan_actions = [int(x) for x in str(row.get("planning_action_sequence", "")).split("|") if x != ""]
    if not dqn_actions:
        return "UNKNOWN"
    dqn_positive = sum(a > 0 for a in dqn_actions)
    planning_positive = sum(a > 0 for a in plan_actions)
    if dqn_positive > planning_positive:
        if max(dqn_actions) > (max(plan_actions) if plan_actions else 0):
            return "EXCESSIVE_MARKDOWN_DEPTH"
        return "UNNECESSARY_MARKDOWN"
    if dqn_positive < planning_positive:
        return "MISSED_MARKDOWN_OPPORTUNITY"
    if dqn_actions != plan_actions:
        return "MISTIMED_MARKDOWN"
    return "ACCOUNTING_OR_DEMAND_REALIZATION_DIFFERENCE"


def worst_episode_diagnostics(episode_diffs: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    if episode_diffs.empty:
        out = pd.DataFrame()
    else:
        worst = episode_diffs.sort_values("dqn_minus_baseline").head(top_n).copy()
        worst["baseline_action"] = "always_0pct"
        worst["source_of_profit_loss"] = worst.apply(classify_loss, axis=1)
        worst["initial_inventory_coverage"] = np.nan
        worst["fraction_expiring_within_two_days"] = np.nan
        worst["demand_level"] = np.nan
        out = worst[
            [
                "seed",
                "selected_checkpoint",
                "episode_id",
                "scenario_id",
                "initial_inventory_coverage",
                "fraction_expiring_within_two_days",
                "demand_level",
                "dqn_action_sequence",
                "planning_action_sequence",
                "baseline_action",
                "dqn_minus_baseline",
                "source_of_profit_loss",
            ]
        ]
    out.to_csv(TABLES_DIR / "dqn_worst_episode_diagnostics.csv", index=False)
    return out


def finalize_status(selected: pd.DataFrame, pooled: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    if selected.empty or len(selected["seed"].unique()) < 3 or pooled.empty:
        return "DQN_EVALUATION_REQUIRES_REVISION", pd.DataFrame()
    positive_seed_count = int((selected["paired_mean_gain_vs_always_0pct"].astype(float) > 0).sum())
    pooled_row = pooled.iloc[0]
    all_accounting_valid = selected["accounting_valid"].astype(str).str.lower().eq("true").all()
    no_collapse = ~selected["collapse_flag"].astype(str).str.lower().eq("true").any()
    state_dependent = selected["state_dependence"].astype(str).eq("STATE_DEPENDENT").all()
    eligible = (
        positive_seed_count >= 2
        and safe_float(pooled_row["pooled_mean_gain"]) > 0
        and safe_float(pooled_row["pooled_win_share"]) >= WIN_SHARE_THRESHOLD
        and safe_float(pooled_row["pooled_waste_difference"]) <= 0.02
        and safe_float(pooled_row["worst_decile_pooled_gain"]) > -0.08
        and all_accounting_valid
        and no_collapse
        and state_dependent
    )
    if eligible:
        status = "DQN_BEATS_BASELINE_ON_VALIDATION"
    elif safe_float(pooled_row["pooled_mean_gain"]) > 0:
        status = "DQN_POSITIVE_BUT_NOT_ROBUST"
    else:
        status = "PLANNING_VALUE_EXISTS_BUT_DQN_FAILED"
    canonical = selected.sort_values(
        ["paired_mean_gain_vs_always_0pct", "win_share", "worst_decile_paired_gain"],
        ascending=[False, False, False],
    ).head(1).copy()
    canonical["family_eligible"] = eligible
    canonical["final_status"] = status
    canonical.to_csv(TABLES_DIR / "dqn_canonical_candidate_selection.csv", index=False)
    return status, canonical


def write_config(status: str, selected: pd.DataFrame, pooled: pd.DataFrame, canonical: pd.DataFrame) -> None:
    config = {
        "status": status,
        "final_model_locked": bool(status == "DQN_BEATS_BASELINE_ON_VALIDATION"),
        "test_split_used": False,
        "population": POPULATION_ID,
        "calibration_mode": CALIBRATION_MODE,
        "locked_baseline": BASELINE_POLICY,
        "standard_dqn_config_unchanged": True,
        "selected_checkpoints_by_seed": selected[
            ["seed", "policy_id", "checkpoint_timesteps", "paired_mean_gain_vs_always_0pct", "win_share"]
        ].to_dict("records")
        if not selected.empty
        else [],
        "pooled_validation": pooled.iloc[0].to_dict() if not pooled.empty else None,
        "canonical_candidate": canonical.iloc[0].to_dict() if not canonical.empty else None,
    }
    (CONFIGS_DIR / "dqn_locked_candidate.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    paths = [
        TABLES_DIR / "dqn_checkpoint_evaluation_all_seeds.csv",
        TABLES_DIR / "dqn_seed_stability.csv",
        TABLES_DIR / "dqn_pooled_validation_results.csv",
        TABLES_DIR / "dqn_seed_by_episode_profit_differences.csv",
        TABLES_DIR / "dqn_worst_episode_diagnostics.csv",
        TABLES_DIR / "dqn_canonical_candidate_selection.csv",
        CONFIGS_DIR / "dqn_locked_candidate.json",
    ]
    hashes = [{"path": str(path.relative_to(PROJECT_ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size if path.exists() else None} for path in paths]
    (CONFIGS_DIR / "dqn_locked_candidate_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")


def make_figures(summary: pd.DataFrame, selected: pd.DataFrame, pooled: pd.DataFrame, episode_diffs: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    dqn = summary.loc[summary["model_family"].astype(str).eq("DQN")].copy()
    if dqn.empty:
        return
    dqn["step_numeric"] = dqn["checkpoint_timesteps"].map(checkpoint_sort_value)
    dqn = dqn.loc[dqn["step_numeric"] < 999_999]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for seed, group in dqn.groupby("seed"):
        group = group.sort_values("step_numeric")
        ax.plot(group["step_numeric"], group["mean_normalized_profit"].astype(float), marker="o", label=f"seed {int(float(seed))}")
    ax.set_title("Validation Profit by Timestep")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean normalized profit")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "standard_dqn_validation_profit_by_timestep_all_seeds.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    selected.set_index("seed")["paired_mean_gain_vs_always_0pct"].astype(float).plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Paired Gain by Selected Seed")
    ax.set_ylabel("Gain vs always_0pct")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "standard_dqn_paired_gain_by_selected_seed.png", dpi=160)
    plt.close(fig)

    if not pooled.empty:
        row = pooled.iloc[0]
        mean = safe_float(row["pooled_mean_gain"])
        low = safe_float(row["pooled_ci_low"])
        high = safe_float(row["pooled_ci_high"])
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.errorbar([0], [mean], yerr=[[mean - low], [high - mean]], fmt="o")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks([0])
        ax.set_xticklabels(["pooled DQN"])
        ax.set_title("Pooled Confidence Interval")
        ax.set_ylabel("Gain vs always_0pct")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "standard_dqn_pooled_confidence_interval.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    selected.set_index("seed")["win_share"].astype(float).plot(kind="bar", ax=ax)
    ax.axhline(WIN_SHARE_THRESHOLD, color="black", linewidth=0.8)
    ax.set_title("Seed Stability")
    ax.set_ylabel("Win share")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "standard_dqn_seed_stability.png", dpi=160)
    plt.close(fig)

    refs = summary.loc[summary["policy_id"].isin([BASELINE_POLICY, PLANNING_POLICY, DISTILLED_POLICY, SELECTED_PPO_POLICY])]
    compare = pd.concat([refs, selected], ignore_index=True, sort=False)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    compare.set_index("policy_id")["mean_normalized_profit"].astype(float).plot(kind="bar", ax=ax)
    ax.set_title("DQN versus Planning versus Baseline")
    ax.set_ylabel("Mean normalized profit")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "standard_dqn_vs_planning_baseline.png", dpi=160)
    plt.close(fig)

    action_cols = [f"action_{i}_share" for i in range(6)]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    selected.set_index("seed")[action_cols].astype(float).plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Action Distribution by Seed")
    ax.set_ylabel("Action share")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "standard_dqn_action_distribution_by_seed.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(compare["waste_rate"].astype(float), compare["mean_normalized_profit"].astype(float))
    for _, row in compare.iterrows():
        ax.annotate(str(row["policy_id"])[:24], (safe_float(row["waste_rate"]), safe_float(row["mean_normalized_profit"])), fontsize=7)
    ax.set_title("Profit-Waste Comparison")
    ax.set_xlabel("Waste rate")
    ax.set_ylabel("Mean normalized profit")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "standard_dqn_profit_waste_comparison.png", dpi=160)
    plt.close(fig)

    if not episode_diffs.empty:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        episode_diffs.sort_values("dqn_minus_baseline").head(20).set_index("episode_id")["dqn_minus_baseline"].plot(kind="bar", ax=ax)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Worst-Episode Loss Decomposition")
        ax.set_ylabel("DQN minus always_0pct")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "standard_dqn_worst_episode_loss_decomposition.png", dpi=160)
        plt.close(fig)


def train_remaining(seeds: list[int]) -> None:
    audit = audit_environment_compatibility()
    if audit["compatibility_status"].eq("fail").any():
        print("DQN_EVALUATION_REQUIRES_REVISION")
        return
    training_population_audit(load_training_episode_table())
    for seed in seeds:
        print(f"Training STANDARD_DQN seed={seed} timesteps={MAX_TIMESTEPS} resume=True", flush=True)
        train_one(
            STANDARD_CONFIG,
            int(seed),
            MAX_TIMESTEPS,
            CHECKPOINT_EVERY,
            n_envs=1,
            resume=True,
        )


def evaluate_all(seeds: list[int]) -> str:
    audit = audit_environment_compatibility()
    if audit["compatibility_status"].eq("fail").any():
        print("DQN_EVALUATION_REQUIRES_REVISION")
        return "DQN_EVALUATION_REQUIRES_REVISION"
    records = checkpoint_records(["STANDARD_DQN"], seeds)
    completed = set(pd.to_numeric(records["seed"], errors="coerce").dropna().astype(int).unique()) if not records.empty else set()
    if not set(seeds).issubset(completed):
        missing = sorted(set(seeds).difference(completed))
        print(f"Missing STANDARD_DQN checkpoints for seeds: {missing}")
        print("DQN_EVALUATION_REQUIRES_REVISION")
        return "DQN_EVALUATION_REQUIRES_REVISION"
    results, states, summary = evaluate_checkpoints(["STANDARD_DQN"], seeds)
    summary = add_action_distribution(summary, results)
    summary.to_csv(TABLES_DIR / "dqn_checkpoint_evaluation_all_seeds.csv", index=False)
    selected = select_checkpoint_per_seed(summary)
    episode_diffs = build_seed_episode_differences(results, selected)
    pooled = pooled_results(selected, episode_diffs)
    stability = seed_stability(selected, pooled)
    worst_episode_diagnostics(episode_diffs)
    status, canonical = finalize_status(selected, pooled)
    write_config(status, selected, pooled, canonical)
    make_figures(summary, selected, pooled, episode_diffs)
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Complete STANDARD_DQN multi-seed validation experiment.")
    parser.add_argument("--mode", choices=["train_remaining", "evaluate", "all"], default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=STANDARD_SEEDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    ensure_dirs()
    status = "DQN_EVALUATION_REQUIRES_REVISION"
    if args.mode in {"train_remaining", "all"}:
        seeds_to_train = [seed for seed in args.seeds if seed in REMAINING_SEEDS]
        train_remaining(seeds_to_train)
    if args.mode in {"evaluate", "all"}:
        status = evaluate_all(args.seeds)
    runtime = time.perf_counter() - start
    print(status)
    print(f"seeds_requested={'|'.join(str(s) for s in args.seeds)}")
    print(f"runtime_seconds={runtime:.1f}")
    print("test_split_used=False")
    for path in [
        TABLES_DIR / "dqn_checkpoint_evaluation_all_seeds.csv",
        TABLES_DIR / "dqn_seed_stability.csv",
        TABLES_DIR / "dqn_pooled_validation_results.csv",
        TABLES_DIR / "dqn_seed_by_episode_profit_differences.csv",
        TABLES_DIR / "dqn_worst_episode_diagnostics.csv",
        TABLES_DIR / "dqn_canonical_candidate_selection.csv",
        CONFIGS_DIR / "dqn_locked_candidate.json",
        CONFIGS_DIR / "dqn_locked_candidate_hashes.json",
    ]:
        print(f"- {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
