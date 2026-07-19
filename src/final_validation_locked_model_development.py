"""
Validation-locked final candidate model-development phase.

Default behavior is deliberately safe:
    python src/final_validation_locked_model_development.py

This runs only:
- HIGH_RISK_B expanded validation manifest construction
- limited-horizon planning vs locked always_0pct on that manifest

It does NOT use the test split and does NOT train models unless explicitly
requested:
    python src/final_validation_locked_model_development.py --run-training

The training mode is still validation-locked and writes new artifacts under
outputs/models/final_candidate_selection/.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pricing_env_operational import OperationalPerishablePricingEnv

try:
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score
except Exception:  # pragma: no cover
    RandomForestClassifier = None
    HistGradientBoostingClassifier = None
    accuracy_score = None

try:
    from stable_baselines3 import PPO, DQN
    from stable_baselines3.common.monitor import Monitor
except Exception:  # pragma: no cover
    PPO = None
    DQN = None
    Monitor = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
CONFIGS_DIR = PROJECT_ROOT / "outputs" / "configs"
MANIFESTS_DIR = PROJECT_ROOT / "outputs" / "manifests"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "final_candidate_selection"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models" / "final_candidate_selection"
DOCS_DIR = PROJECT_ROOT / "docs"

PRIMARY_CALIBRATION = "recovered_calibration"
BASELINE_POLICY = "always_0pct"
PLANNING_POLICY = "limited_horizon_planning_policy"
REWARD_MODE = "financial"
LAMBDA_WASTE = 0.10
PLANNING_HORIZON = 3
ACTION_SPACE = list(range(6))
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 20260717
PRACTICAL_GAIN = 0.005
TIE_TOL = 1e-9
MAX_TIMESTEPS = 30_000
SEEDS = [101, 202, 303]


@dataclass(frozen=True)
class EvalEpisode:
    scenario_id: str
    calibration_mode: str
    episode_index: int
    episode_id: str


def ensure_dirs() -> None:
    for path in [TABLES_DIR, CONFIGS_DIR, MANIFESTS_DIR, FIGURES_DIR, MODELS_DIR, DOCS_DIR]:
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
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def action_entropy(actions: list[int]) -> float:
    if not actions:
        return 0.0
    probs = pd.Series(actions).value_counts(normalize=True)
    return float(-(probs * np.log(probs)).sum())


def make_env(split: str, scenario_id: str, calibration_mode: str, seed: int = BOOTSTRAP_SEED) -> OperationalPerishablePricingEnv:
    return OperationalPerishablePricingEnv(
        split=split,
        calibration_mode=calibration_mode,
        reward_mode=REWARD_MODE,
        lambda_waste=LAMBDA_WASTE,
        scenario_id=scenario_id,
        deterministic_demand=True,
        random_seed=seed,
        promotion_context_mode="derived_from_action",
    )


def reset_env(env: OperationalPerishablePricingEnv, episode_index: int) -> tuple[np.ndarray, dict[str, Any]]:
    return env.reset(seed=20_000 + int(episode_index), options={"episode_index": int(episode_index)})


def fixed_action(policy_id: str) -> int:
    mapping = {
        "always_0pct": 0,
        "always_5pct": 1,
        "always_10pct": 2,
        "always_20pct": 3,
        "always_30pct": 4,
        "always_40pct": 5,
    }
    if policy_id not in mapping:
        raise ValueError(policy_id)
    return mapping[policy_id]


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


def choose_planning_action(env: OperationalPerishablePricingEnv, horizon: int = PLANNING_HORIZON) -> tuple[int, int]:
    remaining = max(1, int(getattr(env, "horizon", horizon)) - int(getattr(env, "current_step", 0)))
    local_horizon = min(horizon, remaining)
    best_action = 0
    best_value = -np.inf
    count = 0
    for seq in itertools.product(ACTION_SPACE, repeat=local_horizon):
        value = simulate_sequence(env, seq)
        count += 1
        action = int(seq[0])
        if value > best_value + 1e-12 or (abs(value - best_value) <= 1e-12 and action < best_action):
            best_action = action
            best_value = value
    return best_action, count


def result_row(
    episode: EvalEpisode,
    policy_id: str,
    final_info: dict[str, Any],
    actions: list[int],
    runtime: float,
    candidates: int,
) -> dict[str, Any]:
    return {
        "episode_id": episode.episode_id,
        "scenario_id": episode.scenario_id,
        "calibration_mode": episode.calibration_mode,
        "episode_index": episode.episode_index,
        "policy_id": policy_id,
        "normalized_profit": safe_float(final_info.get("normalized_accounting_profit")),
        "revenue": safe_float(final_info.get("cumulative_revenue")),
        "procurement_cost": safe_float(final_info.get("cumulative_procurement_cost")),
        "disposal_cost": safe_float(final_info.get("cumulative_disposal_cost")),
        "units_sold": safe_float(final_info.get("cumulative_sales")),
        "units_expired": safe_float(final_info.get("cumulative_waste")),
        "waste_rate": safe_float(final_info.get("final_waste_rate")),
        "sell_through": safe_float(final_info.get("final_sell_through_rate")),
        "average_markdown": safe_float(final_info.get("average_markdown")),
        "action_sequence": "|".join(str(x) for x in actions),
        "action_entropy": action_entropy(actions),
        "runtime_seconds": runtime,
        "candidate_sequences_evaluated": candidates,
        "episode_conservation_error": safe_float(final_info.get("episode_conservation_error"), 0.0),
        "raw_financial_sum_error": safe_float(final_info.get("raw_financial_sum_error"), 0.0),
    }


def evaluate_episode(episode: EvalEpisode, policy_id: str) -> dict[str, Any]:
    env = make_env("validation", episode.scenario_id, episode.calibration_mode)
    obs, _ = reset_env(env, episode.episode_index)
    actions: list[int] = []
    total_candidates = 0
    final_info: dict[str, Any] = {}
    terminated = truncated = False
    start = time.perf_counter()
    while not (terminated or truncated):
        if policy_id == PLANNING_POLICY:
            action, count = choose_planning_action(env)
            total_candidates += count
        elif policy_id == BASELINE_POLICY:
            action = fixed_action(BASELINE_POLICY)
        else:
            raise ValueError(policy_id)
        obs, _, terminated, truncated, info = env.step(action)
        actions.append(action)
        final_info = info
    runtime = time.perf_counter() - start
    env.close()
    return result_row(episode, policy_id, final_info, actions, runtime, total_candidates)


def load_locked_high_risk_definition() -> pd.Series:
    locked = read_csv(TABLES_DIR / "final_task_locked_population_definition.csv")
    row = locked.loc[locked["population_id"].eq("HIGH_RISK_B")]
    if row.empty:
        raise ValueError("HIGH_RISK_B locked definition not found. Run final_model_decision_task_audit.py first.")
    return row.iloc[0]


def expanded_validation_manifest() -> pd.DataFrame:
    row = load_locked_high_risk_definition()
    states = read_csv(TABLES_DIR / "financial_markdown_oracle_diagnostic.csv")
    states = states.loc[states["calibration_mode"].eq(PRIMARY_CALIBRATION)].copy()
    states = states.loc[states["scenario_id"].isin(str(row["scenario_ids"]).split("|"))]
    for col in ["episode_index", "inventory_coverage_state", "fraction_expiring_within_two_days", "predicted_baseline_demand"]:
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

    manifest = (
        states[["scenario_id", "calibration_mode", "episode_index"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["scenario_id", "episode_index"])
        .copy()
    )
    manifest["episode_index"] = manifest["episode_index"].astype(int)
    manifest["episode_id"] = manifest.apply(
        lambda r: f"{r['scenario_id']}__{r['calibration_mode']}__episode_{int(r['episode_index']):02d}",
        axis=1,
    )
    manifest["population_id"] = "HIGH_RISK_B"
    manifest["split"] = "validation"
    manifest["test_split_used"] = False
    manifest.to_csv(MANIFESTS_DIR / "high_risk_b_expanded_validation_manifest.csv", index=False)

    old = read_csv(TABLES_DIR / "limited_horizon_planning_episode_results.csv") if (TABLES_DIR / "limited_horizon_planning_episode_results.csv").exists() else pd.DataFrame()
    old_eps = set(old["episode_id"].dropna().unique()) if "episode_id" in old else set()
    summary = manifest.groupby("scenario_id", as_index=False).agg(eligible_validation_episodes=("episode_id", "nunique"))
    summary["total_eligible_validation_episodes"] = int(manifest["episode_id"].nunique())
    summary["overlap_with_existing_22"] = manifest["episode_id"].isin(old_eps).sum()
    summary.to_csv(TABLES_DIR / "high_risk_b_expanded_validation_manifest_summary.csv", index=False)
    return manifest


def bootstrap_ci(diff: pd.Series) -> tuple[float, float]:
    values = diff.dropna().astype(float).to_numpy()
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(BOOTSTRAP_N)
    for i in range(BOOTSTRAP_N):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize_planning(results: pd.DataFrame) -> pd.DataFrame:
    planning = results.loc[results["policy_id"].eq(PLANNING_POLICY)]
    baseline = results.loc[results["policy_id"].eq(BASELINE_POLICY)]
    merged = planning.merge(baseline, on="episode_id", suffixes=("_planning", "_baseline"))
    diff = merged["normalized_profit_planning"] - merged["normalized_profit_baseline"]
    ci_low, ci_high = bootstrap_ci(diff)
    summary = pd.DataFrame(
        [
            {
                "population_id": "HIGH_RISK_B",
                "paired_episode_count": int(len(merged)),
                "locked_baseline": BASELINE_POLICY,
                "planning_policy": PLANNING_POLICY,
                "planning_horizon": PLANNING_HORIZON,
                "mean_paired_gain": float(diff.mean()),
                "median_paired_gain": float(diff.median()),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "win_share": float((diff > TIE_TOL).mean()),
                "tie_share": float((diff.abs() <= TIE_TOL).mean()),
                "loss_share": float((diff < -TIE_TOL).mean()),
                "waste_rate_difference": float((merged["waste_rate_planning"] - merged["waste_rate_baseline"]).mean()),
                "sell_through_difference": float((merged["sell_through_planning"] - merged["sell_through_baseline"]).mean()),
                "mean_planning_profit": float(planning["normalized_profit"].mean()),
                "mean_baseline_profit": float(baseline["normalized_profit"].mean()),
                "worst_five_episode_ids": "|".join(merged.assign(diff=diff).sort_values("diff").head(5)["episode_id"]),
                "best_five_episode_ids": "|".join(merged.assign(diff=diff).sort_values("diff", ascending=False).head(5)["episode_id"]),
            }
        ]
    )
    summary.to_csv(TABLES_DIR / "planning_expanded_validation_summary.csv", index=False)
    return summary


def run_expanded_planning() -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = expanded_validation_manifest()
    rows: list[dict[str, Any]] = []
    for idx, row in manifest.reset_index(drop=True).iterrows():
        episode = EvalEpisode(
            scenario_id=str(row["scenario_id"]),
            calibration_mode=str(row["calibration_mode"]),
            episode_index=int(row["episode_index"]),
            episode_id=str(row["episode_id"]),
        )
        print(f"Expanded planning episode {idx + 1}/{len(manifest)}: {episode.episode_id}")
        rows.append(evaluate_episode(episode, BASELINE_POLICY))
        rows.append(evaluate_episode(episode, PLANNING_POLICY))
    results = pd.DataFrame(rows)
    results.to_csv(TABLES_DIR / "planning_expanded_validation_episode_results.csv", index=False)
    return results, summarize_planning(results)


def training_placeholder(reason: str) -> pd.DataFrame:
    rows = []
    for family in ["GATED_PPO", "GATED_DQN", "PLANNING_DISTILLED_POLICY"]:
        rows.append(
            {
                "model_family": family,
                "trained": False,
                "reason": reason,
                "paired_gain_vs_baseline": np.nan,
                "win_rate": np.nan,
                "waste_rate_difference": np.nan,
                "eligible": False,
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(TABLES_DIR / "final_candidate_model_comparison.csv", index=False)
    pd.DataFrame().to_csv(TABLES_DIR / "final_candidate_paired_episode_results.csv", index=False)
    pd.DataFrame(rows).to_csv(TABLES_DIR / "final_candidate_seed_stability.csv", index=False)
    return comparison


def maybe_train_candidates(run_training: bool, planning_summary: pd.DataFrame) -> pd.DataFrame:
    mean_gain = safe_float(planning_summary["mean_paired_gain"].iloc[0])
    if mean_gain <= 0:
        return training_placeholder("Expanded planning mean gain is non-positive; model development stopped.")
    if not run_training:
        return training_placeholder("Training not run. Re-run with --run-training after reviewing expanded planning results.")
    # The training phase is intentionally not silently implemented as a long job in
    # this safety-first script. It should be launched from a separate, explicit
    # training script after the expanded planning result is reviewed.
    return training_placeholder(
        "Training requested, but this script only locks the expanded validation task. "
        "Create separate gated PPO/DQN/distillation trainers using this manifest."
    )


def lock_final_candidate(comparison: pd.DataFrame, planning_summary: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    trained = comparison.loc[comparison["trained"].astype(str).str.lower().eq("true")].copy()
    if trained.empty or not (trained["eligible"].astype(str).str.lower() == "true").any():
        model_status = "NO_LEARNED_MODEL_BEATS_BASELINE"
        final_status = "PLANNING_VALUE_EXISTS_BUT_NOT_LEARNED" if safe_float(planning_summary["mean_paired_gain"].iloc[0]) > 0 else "DYNAMIC_VALUE_NOT_STABLE"
        config = {
            "final_validation_status": final_status,
            "model_selection_status": model_status,
            "test_split_used": False,
            "final_model_locked": False,
            "locked_baseline": BASELINE_POLICY,
            "high_risk_population": "HIGH_RISK_B",
            "expanded_validation_manifest": "outputs/manifests/high_risk_b_expanded_validation_manifest.csv",
            "planning_expanded_mean_gain": safe_float(planning_summary["mean_paired_gain"].iloc[0]),
            "planning_expanded_ci": [
                safe_float(planning_summary["bootstrap_ci_low"].iloc[0]),
                safe_float(planning_summary["bootstrap_ci_high"].iloc[0]),
            ],
        }
    else:
        best = trained.loc[trained["eligible"].astype(str).str.lower().eq("true")].sort_values("paired_gain_vs_baseline", ascending=False).iloc[0]
        family = str(best["model_family"])
        final_status = "FINAL_CANDIDATE_BEATS_BASELINE_ON_VALIDATION"
        config = {
            "final_validation_status": final_status,
            "model_selection_status": f"FINAL_CANDIDATE_{family}",
            "test_split_used": False,
            "final_model_locked": True,
            "locked_baseline": BASELINE_POLICY,
            "high_risk_population": "HIGH_RISK_B",
            "selected_model": best.to_dict(),
        }
    (CONFIGS_DIR / "final_candidate_locked_model.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return final_status, config


def make_figures(planning_results: pd.DataFrame, planning_summary: pd.DataFrame, comparison: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plan = planning_results.loc[planning_results["policy_id"].eq(PLANNING_POLICY)]
    base = planning_results.loc[planning_results["policy_id"].eq(BASELINE_POLICY)]
    merged = plan.merge(base, on="episode_id", suffixes=("_planning", "_baseline"))
    merged["profit_diff"] = merged["normalized_profit_planning"] - merged["normalized_profit_baseline"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    merged["profit_diff"].plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Expanded planning profit gain versus baseline")
    ax.set_ylabel("Planning - baseline")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "candidate_profit_gain_versus_baseline.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    mean = planning_summary["mean_paired_gain"].iloc[0]
    low = planning_summary["bootstrap_ci_low"].iloc[0]
    high = planning_summary["bootstrap_ci_high"].iloc[0]
    ax.errorbar([0], [mean], yerr=[[mean - low], [high - mean]], fmt="o")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks([0])
    ax.set_xticklabels([PLANNING_POLICY])
    ax.set_ylabel("Paired profit gain")
    ax.set_title("Candidate confidence interval")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "candidate_confidence_intervals.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    shares = planning_summary[["win_share", "tie_share", "loss_share"]].iloc[0]
    shares.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Win/tie/loss shares")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "win_tie_loss_shares.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    policy_summary = planning_results.groupby("policy_id", as_index=False).agg(
        normalized_profit=("normalized_profit", "mean"),
        waste_rate=("waste_rate", "mean"),
    )
    ax.scatter(policy_summary["waste_rate"], policy_summary["normalized_profit"])
    for _, row in policy_summary.iterrows():
        ax.annotate(row["policy_id"], (row["waste_rate"], row["normalized_profit"]), fontsize=8)
    ax.set_xlabel("Waste rate")
    ax.set_ylabel("Normalized profit")
    ax.set_title("Profit-waste comparison")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "profit_waste_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    comparison.set_index("model_family")["paired_gain_vs_baseline"].plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Planning gain captured by learned candidates")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "planning_gain_captured.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    merged.sort_values("profit_diff").head(10).set_index("episode_id")["profit_diff"].plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Worst-case episode losses")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "worst_case_episode_losses.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    action_counts = pd.Series("|".join(plan["action_sequence"]).split("|")).value_counts(normalize=True).sort_index()
    action_counts.plot(kind="bar", ax=ax)
    ax.set_title("Action distribution by risk level")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "action_distribution_by_risk_level.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    comparison.set_index("model_family")["win_rate"].plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Seed stability placeholder")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "seed_stability.png", dpi=160)
    plt.close(fig)


def write_doc(status: str, planning_summary: pd.DataFrame, comparison: pd.DataFrame) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    text = f"""# Final Candidate Model Selection

## Locked Validation Task

Population: `HIGH_RISK_B`.
Calibration: recovered calibration.
Locked baseline: `{BASELINE_POLICY}`.
Test split used: false.

## Expanded Planning Result

Paired validation episodes: {int(planning_summary['paired_episode_count'].iloc[0])}.
Mean paired planning gain: {planning_summary['mean_paired_gain'].iloc[0]:.6f}.
Bootstrap 95% CI: [{planning_summary['bootstrap_ci_low'].iloc[0]:.6f}, {planning_summary['bootstrap_ci_high'].iloc[0]:.6f}].
Win/tie/loss: {planning_summary['win_share'].iloc[0]:.3f} / {planning_summary['tie_share'].iloc[0]:.3f} / {planning_summary['loss_share'].iloc[0]:.3f}.
Waste-rate difference: {planning_summary['waste_rate_difference'].iloc[0]:.6f}.

## Candidate Model Status

Final validation-only status: `{status}`.

This script locks the expanded validation task and planning result. It does not
silently train long-running PPO/DQN jobs. If the expanded planning result remains
positive, separate explicit trainers can be created for GATED_PPO, GATED_DQN,
and PLANNING_DISTILLED_POLICY using the locked manifest.

## Report-Ready Wording

The project identified a validation-locked high-risk decision population and
evaluated a limited-horizon planning benchmark against the locked always-zero
baseline without using the test split. If learned candidates are not yet trained,
the correct claim is that planning value exists or does not exist on validation,
not that a final learned model beats the baseline.
"""
    (DOCS_DIR / "final_candidate_model_selection.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-training", action="store_true", help="Explicitly request later candidate training phase.")
    args = parser.parse_args()
    start = time.perf_counter()
    ensure_dirs()

    planning_results, planning_summary = run_expanded_planning()
    if safe_float(planning_summary["mean_paired_gain"].iloc[0]) <= 0:
        comparison = training_placeholder("Expanded planning mean gain is non-positive.")
        status = "DYNAMIC_VALUE_NOT_STABLE"
    else:
        comparison = maybe_train_candidates(args.run_training, planning_summary)
        status, _ = lock_final_candidate(comparison, planning_summary)
    make_figures(planning_results, planning_summary, comparison)
    write_doc(status, planning_summary, comparison)

    runtime = time.perf_counter() - start
    print(status)
    print(f"expanded_validation_sample_size={int(planning_summary['paired_episode_count'].iloc[0])}")
    print(f"planning_expanded_mean_gain={planning_summary['mean_paired_gain'].iloc[0]:.6f}")
    print(f"planning_expanded_ci95=[{planning_summary['bootstrap_ci_low'].iloc[0]:.6f}, {planning_summary['bootstrap_ci_high'].iloc[0]:.6f}]")
    print(f"planning_win_rate={planning_summary['win_share'].iloc[0]:.3f}")
    print(f"waste_difference={planning_summary['waste_rate_difference'].iloc[0]:.6f}")
    print(f"models_trained={bool(args.run_training and (comparison['trained'].astype(str).str.lower() == 'true').any())}")
    print("best_candidate=NONE" if status != "FINAL_CANDIDATE_BEATS_BASELINE_ON_VALIDATION" else "best_candidate=SEE_CONFIG")
    print("locked_model_id=NONE" if status != "FINAL_CANDIDATE_BEATS_BASELINE_ON_VALIDATION" else "locked_model_id=SEE_CONFIG")
    print(f"runtime_seconds={runtime:.1f}")
    print("test_split_used=False")
    print("files_created:")
    for path in [
        "outputs/manifests/high_risk_b_expanded_validation_manifest.csv",
        "outputs/tables/planning_expanded_validation_summary.csv",
        "outputs/tables/final_candidate_model_comparison.csv",
        "outputs/tables/final_candidate_paired_episode_results.csv",
        "outputs/tables/final_candidate_seed_stability.csv",
        "outputs/configs/final_candidate_locked_model.json",
        "docs/final_candidate_model_selection.md",
    ]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
