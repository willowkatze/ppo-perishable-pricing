"""Evaluate the locked 13-policy baseline ladder on held-out test episodes.

The ladder is a post-hoc descriptive analysis. Its primary reference remains
the paired comparison between the locked DQN ensemble and ``always_0pct``.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if __package__:
    from src.dqn_statistical_audit_and_ensemble import ensemble_action, load_members
    from src.pricing_env_operational import ACTION_MARKDOWNS, observation_names
    from src.train_dqn_high_risk_b import action_entropy, make_env, reset_env, safe_float
else:
    from dqn_statistical_audit_and_ensemble import ensemble_action, load_members
    from pricing_env_operational import ACTION_MARKDOWNS, observation_names
    from train_dqn_high_risk_b import action_entropy, make_env, reset_env, safe_float

# ---------------------------------------------------------------------------
# Paths and locked evaluation settings
# ---------------------------------------------------------------------------

TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "final_test_baseline_ladder"
CONFIGS_DIR = PROJECT_ROOT / "outputs" / "configs"
MANIFEST_PATH = PROJECT_ROOT / "outputs" / "manifests" / "high_risk_b_final_test_manifest.csv"
PRIMARY_TEST_RESULTS = TABLES_DIR / "final_dqn_ensemble_test_episode_results.csv"
PRIMARY_TEST_SUMMARY = TABLES_DIR / "final_dqn_ensemble_test_summary.csv"

BOOTSTRAP_N = 3000
BOOTSTRAP_SEED = 20260721
TIE_TOL = 1e-9
RANDOM_SEED = 8675309

BASELINES = [
    "always_0pct",
    "always_5pct",
    "always_10pct",
    "always_20pct",
    "always_30pct",
    "always_40pct",
    "random_uniform",
    "expiry_rule_10pct",
    "inventory_coverage_rule_10pct",
]

LEARNED_POLICIES = [
    "original_observed_ppo",
    "original_recovered_ppo",
    "balanced_recovered_ppo",
    "locked_dqn_ensemble",
]

POLICY_ORDER = BASELINES + LEARNED_POLICIES

HIERARCHY = {
    "always_0pct": "STRONG_NO_MARKDOWN",
    "always_5pct": "FIXED_MARKDOWN",
    "always_10pct": "FIXED_MARKDOWN",
    "always_20pct": "FIXED_MARKDOWN",
    "always_30pct": "FIXED_MARKDOWN",
    "always_40pct": "FIXED_MARKDOWN",
    "random_uniform": "NAIVE",
    "expiry_rule_10pct": "RULE_BASED",
    "inventory_coverage_rule_10pct": "RULE_BASED",
    "original_observed_ppo": "LEARNED_POLICY",
    "original_recovered_ppo": "LEARNED_POLICY",
    "balanced_recovered_ppo": "LEARNED_POLICY",
    "locked_dqn_ensemble": "LEARNED_POLICY",
}

PPO_CONFIGS = {
    "original_observed_ppo": (
        PROJECT_ROOT / "outputs" / "models" / "ppo_operational" / "observed_financial" / "seed_42" / "checkpoints" / "pilot_observed_financial_22288_steps.zip",
        PROJECT_ROOT / "outputs" / "models" / "ppo_operational" / "observed_financial" / "seed_42" / "vecnormalize.pkl",
    ),
    "original_recovered_ppo": (
        PROJECT_ROOT / "outputs" / "models" / "ppo_operational" / "recovered_financial" / "seed_42" / "checkpoints" / "pilot_recovered_financial_22288_steps.zip",
        PROJECT_ROOT / "outputs" / "models" / "ppo_operational" / "recovered_financial" / "seed_42" / "vecnormalize.pkl",
    ),
    "balanced_recovered_ppo": (
        PROJECT_ROOT / "outputs" / "models" / "ppo_training_redesign" / "balanced" / "seed_123" / "checkpoints" / "redesign_balanced_20000_steps.zip",
        PROJECT_ROOT / "outputs" / "models" / "ppo_training_redesign" / "balanced" / "seed_123" / "vecnormalize.pkl",
    ),
}


@dataclass
class PPOMember:
    policy_id: str
    model: Any
    vecnormalize: VecNormalize

    def action(self, obs: np.ndarray) -> int:
        norm_obs = self.vecnormalize.normalize_obs(obs.astype(np.float32)[None, :])[0]
        action, _ = self.model.predict(norm_obs, deterministic=True)
        return int(action)

    def close(self) -> None:
        self.vecnormalize.close()


# ---------------------------------------------------------------------------
# Locked policy loading
# ---------------------------------------------------------------------------


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required locked artifact not found: {path}")


def action_for_markdown(markdown: float) -> int:
    for action, value in ACTION_MARKDOWNS.items():
        if abs(float(value) - markdown) <= 1e-12:
            return int(action)
    raise ValueError(f"Locked action space does not contain markdown={markdown}")


def load_high_risk_definition() -> dict[str, Any]:
    path = CONFIGS_DIR / "final_locked_dqn_ensemble.json"
    require_file(path)
    return json.loads(path.read_text(encoding="utf-8")).get("high_risk_b_definition", {})


def load_ppo_members(first_scenario: str) -> dict[str, PPOMember]:
    members: dict[str, PPOMember] = {}
    missing: list[str] = []
    for policy_id, (model_path, vec_path) in PPO_CONFIGS.items():
        if not model_path.exists() or not vec_path.exists():
            missing.append(
                f"{policy_id}: model={model_path} ({'present' if model_path.exists() else 'missing'}); "
                f"vecnormalize={vec_path} ({'present' if vec_path.exists() else 'missing'})"
            )
            continue
    if missing:
        details = "\n- ".join(missing)
        raise FileNotFoundError(
            "The final 13-policy ladder requires every locked PPO artifact. Missing requirements:\n- " + details
        )

    for policy_id, (model_path, vec_path) in PPO_CONFIGS.items():
        dummy = DummyVecEnv([lambda scenario=first_scenario: make_env("test", scenario, BOOTSTRAP_SEED)])
        vec = VecNormalize.load(str(vec_path), dummy)
        vec.training = False
        vec.norm_reward = False
        model = PPO.load(str(model_path), env=None)
        members[policy_id] = PPOMember(policy_id=policy_id, model=model, vecnormalize=vec)
    return members


def fixed_or_rule_action(policy_id: str, obs: np.ndarray, step: int, episode_index: int, thresholds: dict[str, float]) -> int:
    obs_names = observation_names()
    idx = {name: i for i, name in enumerate(obs_names)}
    fixed = {
        "always_0pct": 0.00,
        "always_5pct": 0.05,
        "always_10pct": 0.10,
        "always_20pct": 0.20,
        "always_30pct": 0.30,
        "always_40pct": 0.40,
    }
    if policy_id in fixed:
        return action_for_markdown(fixed[policy_id])
    if policy_id == "random_uniform":
        rng = np.random.default_rng(RANDOM_SEED + int(episode_index) * 1009 + int(step))
        return int(rng.integers(0, len(ACTION_MARKDOWNS)))
    if policy_id == "expiry_rule_10pct":
        value = float(obs[idx["fraction_expiring_within_two_days"]])
        return action_for_markdown(0.10) if value >= thresholds["min_fraction_expiring_within_two_days"] else action_for_markdown(0.00)
    if policy_id == "inventory_coverage_rule_10pct":
        value = float(obs[idx["inventory_coverage"]])
        return action_for_markdown(0.10) if value >= thresholds["min_inventory_coverage"] else action_for_markdown(0.00)
    raise ValueError(f"Not a fixed/rule baseline: {policy_id}")


# ---------------------------------------------------------------------------
# Episode evaluation
# ---------------------------------------------------------------------------


def evaluate_episode(
    policy_id: str,
    episode: pd.Series,
    thresholds: dict[str, float],
    dqn_members: list[Any] | None,
    ppo_members: dict[str, PPOMember],
) -> tuple[dict[str, Any], pd.DataFrame]:
    env = make_env("test", str(episode["scenario_id"]), BOOTSTRAP_SEED)
    obs, _ = reset_env(env, int(episode["episode_index"]))
    actions: list[int] = []
    state_rows: list[dict[str, Any]] = []
    terminated = truncated = False
    final_info: dict[str, Any] = {}
    start = time.perf_counter()
    step = 0
    while not (terminated or truncated):
        if policy_id in BASELINES:
            action = fixed_or_rule_action(policy_id, obs, step, int(episode["episode_index"]), thresholds)
        elif policy_id == "locked_dqn_ensemble":
            if dqn_members is None:
                raise ValueError("DQN ensemble members are required")
            action, _ = ensemble_action(dqn_members, obs)
        elif policy_id in ppo_members:
            action = ppo_members[policy_id].action(obs)
        else:
            raise ValueError(f"Policy unavailable or incompatible: {policy_id}")
        markdown = float(ACTION_MARKDOWNS[int(action)])
        obs_names = observation_names()
        idx = {name: i for i, name in enumerate(obs_names)}
        state_rows.append(
            {
                "episode_id": episode["episode_id"],
                "policy_id": policy_id,
                "scenario_id": episode["scenario_id"],
                "episode_index": int(episode["episode_index"]),
                "step_index": step,
                "action": int(action),
                "markdown": markdown,
                "inventory_coverage": float(obs[idx["inventory_coverage"]]),
                "fraction_expiring_within_two_days": float(obs[idx["fraction_expiring_within_two_days"]]),
                "predicted_zero_markdown_demand": float(obs[idx["predicted_zero_markdown_demand"]]),
            }
        )
        obs, _, terminated, truncated, final_info = env.step(int(action))
        actions.append(int(action))
        step += 1
    runtime = time.perf_counter() - start
    env.close()
    markdowns = [float(ACTION_MARKDOWNS[a]) for a in actions]
    row = {
        "episode_id": episode["episode_id"],
        "scenario_id": episode["scenario_id"],
        "episode_index": int(episode["episode_index"]),
        "policy_id": policy_id,
        "baseline_hierarchy": HIERARCHY[policy_id],
        "normalized_profit": safe_float(final_info.get("normalized_accounting_profit")),
        "revenue": safe_float(final_info.get("cumulative_revenue")),
        "waste_rate": safe_float(final_info.get("final_waste_rate")),
        "sell_through": safe_float(final_info.get("final_sell_through_rate")),
        "physical_waste_units": safe_float(final_info.get("cumulative_waste")),
        "average_markdown": float(np.mean(markdowns)) if markdowns else np.nan,
        "action_sequence": "|".join(str(a) for a in actions),
        "action_entropy": action_entropy(actions),
        "runtime_seconds": runtime,
        "episode_conservation_error": safe_float(final_info.get("episode_conservation_error"), 0.0),
        "raw_financial_sum_error": safe_float(final_info.get("raw_financial_sum_error"), 0.0),
    }
    return row, pd.DataFrame(state_rows)


# ---------------------------------------------------------------------------
# Paired statistics
# ---------------------------------------------------------------------------


def bootstrap_ci(values: pd.Series) -> tuple[float, float]:
    arr = values.dropna().astype(float).to_numpy()
    if len(arr) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(BOOTSTRAP_N)
    for i in range(BOOTSTRAP_N):
        samples[i] = rng.choice(arr, size=len(arr), replace=True).mean()
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def paired_diff(results: pd.DataFrame, policy_id: str, ref_id: str) -> pd.Series:
    a = results.loc[results["policy_id"].eq(policy_id), ["episode_id", "normalized_profit"]]
    b = results.loc[results["policy_id"].eq(ref_id), ["episode_id", "normalized_profit"]]
    merged = a.merge(b, on="episode_id", suffixes=("_policy", "_ref"))
    return merged["normalized_profit_policy"].astype(float) - merged["normalized_profit_ref"].astype(float)


def action_distribution(sequences: pd.Series) -> dict[str, float]:
    actions: list[int] = []
    for seq in sequences.dropna():
        actions.extend([int(x) for x in str(seq).split("|") if x != ""])
    s = pd.Series(actions)
    if s.empty:
        return {f"action_{i}_share": np.nan for i in ACTION_MARKDOWNS}
    counts = s.value_counts(normalize=True)
    return {f"action_{i}_share": float(counts.get(i, 0.0)) for i in ACTION_MARKDOWNS}


def state_dependence_label(policy_rows: pd.DataFrame) -> str:
    dist = action_distribution(policy_rows["action_sequence"])
    entropy_values = policy_rows["action_entropy"].astype(float)
    max_share = max(v for v in dist.values() if not pd.isna(v))
    if max_share >= 0.98 or entropy_values.mean() <= 0.05:
        return "NOT_STATE_DEPENDENT"
    return "STATE_DEPENDENT"


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy_id in POLICY_ORDER:
        if policy_id not in set(results["policy_id"]):
            rows.append({"policy_id": policy_id, "evaluation_status": "MISSING_OR_INCOMPATIBLE", "baseline_hierarchy": HIERARCHY[policy_id]})
            continue
        pr = results.loc[results["policy_id"].eq(policy_id)].copy()
        diff_zero = paired_diff(results, policy_id, "always_0pct")
        diff_random = paired_diff(results, policy_id, "random_uniform")
        diff_10 = paired_diff(results, policy_id, "always_10pct")
        ci_low, ci_high = bootstrap_ci(diff_zero)
        dist = action_distribution(pr["action_sequence"])
        rows.append(
            {
                "policy_id": policy_id,
                "evaluation_status": "EVALUATED",
                "baseline_hierarchy": HIERARCHY[policy_id],
                "mean_normalized_profit": float(pr["normalized_profit"].mean()),
                "paired_gain_vs_always_0pct": float(diff_zero.mean()),
                "paired_gain_vs_random_uniform": float(diff_random.mean()),
                "paired_gain_vs_always_10pct": float(diff_10.mean()),
                "median_paired_gain_vs_always_0pct": float(diff_zero.median()),
                "bootstrap_ci_low_vs_always_0pct": ci_low,
                "bootstrap_ci_high_vs_always_0pct": ci_high,
                "win_share_vs_always_0pct": float((diff_zero > TIE_TOL).mean()),
                "tie_share_vs_always_0pct": float((diff_zero.abs() <= TIE_TOL).mean()),
                "loss_share_vs_always_0pct": float((diff_zero < -TIE_TOL).mean()),
                "waste_rate": float(pr["waste_rate"].mean()),
                "sell_through": float(pr["sell_through"].mean()),
                "average_markdown": float(pr["average_markdown"].mean()),
                "worst_decile_gain_vs_always_0pct": float(diff_zero.sort_values().head(max(1, math.ceil(len(diff_zero) * 0.10))).mean()),
                "maximum_episode_loss_vs_always_0pct": float(diff_zero.min()),
                "action_entropy": action_entropy([int(x) for seq in pr["action_sequence"] for x in str(seq).split("|") if x != ""]),
                "state_dependence": state_dependence_label(pr),
                "accounting_validity": bool((pr["episode_conservation_error"].abs() <= 1e-6).all() and (pr["raw_financial_sum_error"].abs() <= 1e-6).all()),
                "episode_count": int(pr["episode_id"].nunique()),
                **dist,
            }
        )
    return pd.DataFrame(rows)


def pairwise(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    evaluated = [p for p in POLICY_ORDER if p in set(results["policy_id"])]
    for a in evaluated:
        for b in evaluated:
            diff = paired_diff(results, a, b)
            low, high = bootstrap_ci(diff)
            rows.append(
                {
                    "policy_a": a,
                    "policy_b": b,
                    "mean_gain_a_minus_b": float(diff.mean()),
                    "median_gain_a_minus_b": float(diff.median()),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "win_share": float((diff > TIE_TOL).mean()),
                    "tie_share": float((diff.abs() <= TIE_TOL).mean()),
                    "loss_share": float((diff < -TIE_TOL).mean()),
                    "paired_episodes": int(len(diff)),
                }
            )
    return pd.DataFrame(rows)


def gain_matrix(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    return pairwise_df.pivot(index="policy_a", columns="policy_b", values="mean_gain_a_minus_b").reset_index()


def validate_complete_ladder(results: pd.DataFrame, expected_episodes: int) -> None:
    """Reject incomplete final tables before any result artifact is written."""
    observed = set(results["policy_id"].astype(str))
    missing = [policy for policy in POLICY_ORDER if policy not in observed]
    unexpected = sorted(observed.difference(POLICY_ORDER))
    counts = results.groupby("policy_id")["episode_id"].nunique().to_dict()
    incomplete = {policy: int(counts.get(policy, 0)) for policy in POLICY_ORDER if counts.get(policy, 0) != expected_episodes}
    if missing or unexpected or incomplete:
        raise RuntimeError(
            "Final baseline ladder is incomplete: "
            f"missing={missing}; unexpected={unexpected}; episode_counts={incomplete}"
        )


def pareto_frontier(summary: pd.DataFrame) -> pd.DataFrame:
    evaluated = summary.loc[summary["evaluation_status"].eq("EVALUATED")].copy()
    frontier = []
    for _, row in evaluated.iterrows():
        dominated = evaluated.loc[
            (evaluated["mean_normalized_profit"] >= row["mean_normalized_profit"] - 1e-12)
            & (evaluated["waste_rate"] <= row["waste_rate"] + 1e-12)
            & (
                (evaluated["mean_normalized_profit"] > row["mean_normalized_profit"] + 1e-12)
                | (evaluated["waste_rate"] < row["waste_rate"] - 1e-12)
            )
        ]
        frontier.append(len(dominated) == 0)
    evaluated["profit_waste_pareto_frontier"] = frontier
    return evaluated[["policy_id", "mean_normalized_profit", "waste_rate", "sell_through", "profit_waste_pareto_frontier"]]


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------


def make_figures(summary: pd.DataFrame, matrix: pd.DataFrame, pairwise_df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    evaluated = summary.loc[summary["evaluation_status"].eq("EVALUATED")].copy()
    evaluated = evaluated.sort_values("mean_normalized_profit", ascending=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(evaluated["policy_id"], evaluated["mean_normalized_profit"])
    ax.axhline(float(evaluated.loc[evaluated["policy_id"].eq("always_0pct"), "mean_normalized_profit"].iloc[0]), color="black", linewidth=1, linestyle="--", label="always_0pct")
    ax.set_ylabel("Mean normalized profit")
    ax.set_title("Final Test Profit Across Baselines and Learned Policies")
    ax.tick_params(axis="x", rotation=60)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "profit_across_all_methods.png", dpi=160)
    plt.close(fig)

    subset = evaluated.loc[evaluated["baseline_hierarchy"].isin(["FIXED_MARKDOWN", "STRONG_NO_MARKDOWN", "LEARNED_POLICY"])]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(subset["policy_id"], subset["mean_normalized_profit"])
    ax.set_ylabel("Mean normalized profit")
    ax.set_title("DQN and PPO versus Fixed-Markdown Baselines")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "learned_vs_fixed_markdown_baselines.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for _, row in evaluated.iterrows():
        ax.scatter(row["waste_rate"], row["mean_normalized_profit"], label=row["policy_id"])
    ax.set_xlabel("Waste rate")
    ax.set_ylabel("Mean normalized profit")
    ax.set_title("Profit-Waste Frontier")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "profit_waste_frontier.png", dpi=160)
    plt.close(fig)

    mat = matrix.set_index("policy_a")
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat.astype(float).to_numpy(), cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=70, ha="right", fontsize=7)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=7)
    ax.set_title("Pairwise Mean Gain Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pairwise_gain_matrix.png", dpi=160)
    plt.close(fig)

    hierarchy = evaluated.groupby("baseline_hierarchy", as_index=False).agg(
        mean_profit=("mean_normalized_profit", "mean"),
        best_profit=("mean_normalized_profit", "max"),
        mean_waste=("waste_rate", "mean"),
        methods=("policy_id", "count"),
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(hierarchy["baseline_hierarchy"], hierarchy["best_profit"])
    ax.set_ylabel("Best mean normalized profit")
    ax.set_title("Baseline Hierarchy Summary")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "baseline_hierarchy_summary.png", dpi=160)
    plt.close(fig)


def final_report(summary: pd.DataFrame, frontier: pd.DataFrame) -> None:
    evaluated = summary.loc[summary["evaluation_status"].eq("EVALUATED")].copy()
    strongest_overall = evaluated.sort_values("mean_normalized_profit", ascending=False).iloc[0]
    learned = evaluated.loc[evaluated["baseline_hierarchy"].eq("LEARNED_POLICY")].sort_values("mean_normalized_profit", ascending=False)
    strongest_learned = learned.iloc[0]

    def beaten_by(policy_id: str) -> list[str]:
        if policy_id not in set(evaluated["policy_id"]):
            return []
        row = evaluated.loc[evaluated["policy_id"].eq(policy_id)].iloc[0]
        return evaluated.loc[evaluated["mean_normalized_profit"] < row["mean_normalized_profit"] - TIE_TOL, "policy_id"].tolist()

    dqn = evaluated.loc[evaluated["policy_id"].eq("locked_dqn_ensemble")].iloc[0]
    always0 = evaluated.loc[evaluated["policy_id"].eq("always_0pct")].iloc[0]
    claim = (
        "The locked DQN ensemble did not outperform always_0pct on held-out test. "
        "This secondary ladder can only support descriptive claims about which weaker baselines were beaten."
    )
    if dqn["mean_normalized_profit"] > always0["mean_normalized_profit"] + TIE_TOL:
        claim = "Unexpectedly, the locked DQN ensemble exceeded always_0pct; verify against the primary final-test table before making any claim."

    print("BASELINE_LADDER_COMPLETE", flush=True)
    print(f"strongest_overall_policy={strongest_overall['policy_id']} mean_profit={strongest_overall['mean_normalized_profit']:.6f}", flush=True)
    print(f"strongest_learned_policy={strongest_learned['policy_id']} mean_profit={strongest_learned['mean_normalized_profit']:.6f}", flush=True)
    print("baselines_beaten_by_original_recovered_ppo=" + "|".join(beaten_by("original_recovered_ppo")), flush=True)
    print("baselines_beaten_by_balanced_recovered_ppo=" + "|".join(beaten_by("balanced_recovered_ppo")), flush=True)
    print("baselines_beaten_by_locked_dqn_ensemble=" + "|".join(beaten_by("locked_dqn_ensemble")), flush=True)
    print("profit_waste_pareto_frontier=" + "|".join(frontier.loc[frontier["profit_waste_pareto_frontier"], "policy_id"].tolist()), flush=True)
    print("defensible_claim=" + claim, flush=True)


def main() -> None:
    start = time.perf_counter()
    require_file(MANIFEST_PATH)
    require_file(PRIMARY_TEST_RESULTS)
    require_file(PRIMARY_TEST_SUMMARY)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST_PATH).sort_values(["scenario_id", "episode_index"]).reset_index(drop=True)
    if len(manifest) != 60:
        raise ValueError(f"Expected locked 60-episode test manifest, found {len(manifest)}")

    high_risk = load_high_risk_definition()
    thresholds = {
        "min_inventory_coverage": float(high_risk.get("min_inventory_coverage", 1.5)),
        "min_fraction_expiring_within_two_days": float(high_risk.get("min_fraction_expiring_within_two_days", 0.5)),
    }
    first_scenario = str(manifest["scenario_id"].iloc[0])
    dqn_members = load_members(first_scenario)
    ppo_members = load_ppo_members(first_scenario)
    policies = POLICY_ORDER.copy()

    rows: list[dict[str, Any]] = []
    state_rows: list[pd.DataFrame] = []
    total = len(manifest) * len(policies)
    done = 0
    for _, episode in manifest.iterrows():
        for policy_id in policies:
            done += 1
            if done == 1 or done % 25 == 0 or done == total:
                print(f"Evaluating {done}/{total}: {policy_id} {episode['episode_id']}", flush=True)
            row, states = evaluate_episode(policy_id, episode, thresholds, dqn_members, ppo_members)
            rows.append(row)
            state_rows.append(states)

    results = pd.DataFrame(rows)
    states = pd.concat(state_rows, ignore_index=True) if state_rows else pd.DataFrame()
    validate_complete_ladder(results, expected_episodes=len(manifest))
    summary = summarize(results)
    pairwise_df = pairwise(results)
    matrix = gain_matrix(pairwise_df)
    frontier = pareto_frontier(summary)
    summary = summary.merge(frontier[["policy_id", "profit_waste_pareto_frontier"]], on="policy_id", how="left")
    summary["primary_conclusion_preserved"] = True
    summary["analysis_type"] = "post_hoc_secondary_descriptive"
    summary["runtime_seconds"] = time.perf_counter() - start

    results.to_csv(TABLES_DIR / "final_test_baseline_ladder_episode_results.csv", index=False)
    states.to_csv(TABLES_DIR / "final_test_baseline_ladder_state_actions.csv", index=False)
    summary.to_csv(TABLES_DIR / "final_test_baseline_ladder.csv", index=False)
    matrix.to_csv(TABLES_DIR / "final_test_model_vs_baseline_matrix.csv", index=False)
    pairwise_df.to_csv(TABLES_DIR / "final_test_baseline_pairwise_comparisons.csv", index=False)
    frontier.to_csv(TABLES_DIR / "final_test_profit_waste_pareto_frontier.csv", index=False)

    make_figures(summary, matrix, pairwise_df)
    final_report(summary, frontier)

    for member in dqn_members:
        close = getattr(member, "close", None)
        if callable(close):
            close()
    for member in ppo_members.values():
        member.close()

if __name__ == "__main__":
    main()


