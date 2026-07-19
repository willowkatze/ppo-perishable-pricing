"""
Final statistical-integrity audit and equal-weight STANDARD_DQN ensemble.

No training or retraining is performed. The script uses the three already
selected STANDARD_DQN checkpoints and the locked HIGH_RISK_B validation
manifest only. It never constructs the test split.

Run from project root:
    python -u src/dqn_statistical_audit_and_ensemble.py
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from train_dqn_high_risk_b import (
    BASELINE_POLICY,
    CALIBRATION_MODE,
    CONFIGS_DIR,
    DISTILLED_POLICY,
    LAMBDA_WASTE,
    PLANNING_POLICY,
    POPULATION_ID,
    PROJECT_ROOT,
    REWARD_MODE,
    SELECTED_PPO_POLICY,
    TABLES_DIR,
    TIE_TOL,
    action_entropy,
    bootstrap_ci,
    load_validation_manifest,
    make_env,
    reset_env,
    safe_float,
)

FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "dqn_final_statistical_audit"
DOCS_DIR = PROJECT_ROOT / "docs"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models" / "dqn_high_risk_b" / "STANDARD_DQN"
BOOTSTRAP_CLUSTER_N = 10_000
BOOTSTRAP_SEED = 20260719
PLANNING_LOCKED_GAIN = 0.010792
WIN_SHARE_THRESHOLD = 0.60
SELECTED = {
    42: 20_000,
    123: 15_000,
    456: 25_000,
}


@dataclass(frozen=True)
class EvalEpisode:
    scenario_id: str
    calibration_mode: str
    episode_index: int
    episode_id: str


@dataclass
class EnsembleMember:
    seed: int
    checkpoint: int
    model: DQN
    vecnormalize: VecNormalize
    model_path: Path
    vecnormalize_path: Path


def ensure_dirs() -> None:
    for path in [TABLES_DIR, CONFIGS_DIR, FIGURES_DIR, DOCS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def load_manifest() -> list[EvalEpisode]:
    manifest = load_validation_manifest()
    return [
        EvalEpisode(str(row["scenario_id"]), str(row["calibration_mode"]), int(row["episode_index"]), str(row["episode_id"]))
        for _, row in manifest.iterrows()
    ]


def audit_pooled_inference() -> pd.DataFrame:
    pooled_path = TABLES_DIR / "dqn_pooled_validation_results.csv"
    diffs_path = TABLES_DIR / "dqn_seed_by_episode_profit_differences.csv"
    require_file(pooled_path)
    require_file(diffs_path)
    pooled = pd.read_csv(pooled_path)
    diffs = pd.read_csv(diffs_path)
    row_count = len(diffs)
    episode_count = diffs["episode_id"].nunique()
    seed_count = diffs["seed"].nunique()
    expected = episode_count * seed_count
    method = "row_bootstrap_over_seed_episode_rows"
    classification = "PSEUDOREPLICATION_RISK" if row_count == expected and episode_count < row_count else "INFERENCE_REQUIRES_REVISION"
    audit = pd.DataFrame(
        [
            {
                "audit_item": "sampling_unit",
                "value": "validation_episode",
                "details": "The 22 locked validation episodes are the independent units.",
            },
            {"audit_item": "unique_episodes", "value": episode_count, "details": ""},
            {"audit_item": "seeds", "value": seed_count, "details": "|".join(map(str, sorted(diffs["seed"].unique())))},
            {
                "audit_item": "dependence_structure",
                "value": "clustered_seed_repeats_within_episode",
                "details": "Each episode appears once for each selected DQN seed, creating repeated dependent rows.",
            },
            {
                "audit_item": "current_bootstrap_method",
                "value": method,
                "details": f"Existing pooled table has {row_count} rows from {episode_count} episodes x {seed_count} seeds.",
            },
            {
                "audit_item": "classification",
                "value": classification,
                "details": "Do not treat the 66 seed-episode rows as independent.",
            },
            {
                "audit_item": "corrected_recommended_method",
                "value": "episode_cluster_bootstrap",
                "details": "Bootstrap episode IDs; either average seeds within episode first or resample episode clusters.",
            },
            {
                "audit_item": "existing_pooled_ci",
                "value": f"[{pooled['pooled_ci_low'].iloc[0]}, {pooled['pooled_ci_high'].iloc[0]}]",
                "details": "Retained for audit trail only.",
            },
        ]
    )
    audit.to_csv(TABLES_DIR / "dqn_pooled_inference_audit.csv", index=False)
    return audit


def clustered_inference() -> tuple[pd.DataFrame, pd.DataFrame]:
    diffs = pd.read_csv(TABLES_DIR / "dqn_seed_by_episode_profit_differences.csv")
    diff_col = "dqn_minus_baseline"
    episode_means = (
        diffs.groupby("episode_id", as_index=False)
        .agg(
            episode_mean_gain=(diff_col, "mean"),
            episode_median_gain=(diff_col, "median"),
            seed_count=("seed", "nunique"),
            scenario_id=("scenario_id", "first"),
        )
        .sort_values("episode_id")
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = episode_means["episode_mean_gain"].to_numpy(dtype=float)
    boot_episode_means = np.empty(BOOTSTRAP_CLUSTER_N)
    for i in range(BOOTSTRAP_CLUSTER_N):
        boot_episode_means[i] = rng.choice(values, size=len(values), replace=True).mean()
    ep_ci_low, ep_ci_high = np.percentile(boot_episode_means, [2.5, 97.5])

    episode_ids = episode_means["episode_id"].to_numpy()
    cluster_boot = np.empty(BOOTSTRAP_CLUSTER_N)
    for i in range(BOOTSTRAP_CLUSTER_N):
        sampled = rng.choice(episode_ids, size=len(episode_ids), replace=True)
        sampled_rows = pd.concat([diffs.loc[diffs["episode_id"].eq(ep)] for ep in sampled], ignore_index=True)
        cluster_boot[i] = sampled_rows[diff_col].astype(float).mean()
    cl_ci_low, cl_ci_high = np.percentile(cluster_boot, [2.5, 97.5])

    rows = [
        {
            "analysis": "EPISODE_MEAN_ACROSS_SEEDS",
            "sampling_unit": "episode_id",
            "unique_episodes": int(len(episode_means)),
            "seed_count": int(diffs["seed"].nunique()),
            "mean_gain": float(values.mean()),
            "median_gain": float(np.median(values)),
            "ci_low": float(ep_ci_low),
            "ci_high": float(ep_ci_high),
            "win_share": float((values > TIE_TOL).mean()),
            "tie_share": float((np.abs(values) <= TIE_TOL).mean()),
            "loss_share": float((values < -TIE_TOL).mean()),
            "worst_decile_gain": float(np.sort(values)[: max(1, math.ceil(len(values) * 0.10))].mean()),
            "maximum_loss": float(values.min()),
            "bootstrap_resamples": BOOTSTRAP_CLUSTER_N,
        },
        {
            "analysis": "CLUSTER_BOOTSTRAP",
            "sampling_unit": "episode_cluster_with_all_seed_rows",
            "unique_episodes": int(len(episode_means)),
            "seed_count": int(diffs["seed"].nunique()),
            "mean_gain": float(diffs[diff_col].astype(float).mean()),
            "median_gain": float(diffs[diff_col].astype(float).median()),
            "ci_low": float(cl_ci_low),
            "ci_high": float(cl_ci_high),
            "win_share": float((episode_means["episode_mean_gain"] > TIE_TOL).mean()),
            "tie_share": float((episode_means["episode_mean_gain"].abs() <= TIE_TOL).mean()),
            "loss_share": float((episode_means["episode_mean_gain"] < -TIE_TOL).mean()),
            "worst_decile_gain": float(np.sort(values)[: max(1, math.ceil(len(values) * 0.10))].mean()),
            "maximum_loss": float(values.min()),
            "bootstrap_resamples": BOOTSTRAP_CLUSTER_N,
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "dqn_clustered_validation_inference.csv", index=False)
    episode_means.to_csv(TABLES_DIR / "dqn_episode_mean_gain_across_seeds.csv", index=False)
    return out, episode_means


def load_vecnormalize(path: Path, scenario_id: str, seed: int) -> VecNormalize:
    require_file(path)
    dummy = DummyVecEnv([lambda: Monitor(make_env("validation", scenario_id, seed))])
    vec = VecNormalize.load(str(path), dummy)
    vec.training = False
    vec.norm_reward = False
    return vec


def load_members(first_scenario: str) -> list[EnsembleMember]:
    members: list[EnsembleMember] = []
    dims = []
    action_dims = []
    for seed, step in SELECTED.items():
        model_path = MODELS_DIR / f"seed_{seed}" / "checkpoints" / f"dqn_{step}_steps.zip"
        vec_path = MODELS_DIR / f"seed_{seed}" / "checkpoints" / f"vecnormalize_{step}_steps.pkl"
        require_file(model_path)
        require_file(vec_path)
        model = DQN.load(model_path, env=None)
        vec = load_vecnormalize(vec_path, first_scenario, seed + 707)
        dims.append(tuple(model.observation_space.shape))
        action_dims.append(int(model.action_space.n))
        members.append(EnsembleMember(seed, step, model, vec, model_path, vec_path))
    comparable = len(set(dims)) == 1 and len(set(action_dims)) == 1 and action_dims[0] == 6
    if not comparable:
        raise RuntimeError(f"DQN_ENSEMBLE_NOT_COMPARABLE: obs_dims={dims}, action_dims={action_dims}")
    return members


def q_values(member: EnsembleMember, obs: np.ndarray) -> np.ndarray:
    norm_obs = member.vecnormalize.normalize_obs(obs.astype(np.float32)[None, :])
    tensor = torch.as_tensor(norm_obs, dtype=torch.float32, device=member.model.device)
    with torch.no_grad():
        q = member.model.policy.q_net(tensor)
    return q.detach().cpu().numpy()[0].astype(float)


def ensemble_action(members: list[EnsembleMember], obs: np.ndarray) -> tuple[int, list[float]]:
    qs = np.vstack([q_values(member, obs) for member in members])
    mean_q = qs.mean(axis=0)
    return int(np.argmax(mean_q)), mean_q.tolist()


def evaluate_ensemble_episode(episode: EvalEpisode, members: list[EnsembleMember]) -> tuple[dict[str, Any], pd.DataFrame]:
    env = make_env("validation", episode.scenario_id, BOOTSTRAP_SEED)
    obs, _ = reset_env(env, episode.episode_index)
    actions: list[int] = []
    state_rows: list[dict[str, Any]] = []
    final_info: dict[str, Any] = {}
    terminated = truncated = False
    start = time.perf_counter()
    step = 0
    while not (terminated or truncated):
        action, mean_q = ensemble_action(members, obs)
        state_rows.append(
            {
                "episode_id": episode.episode_id,
                "scenario_id": episode.scenario_id,
                "step_index": step,
                "action": action,
                **{f"mean_q_action_{i}": mean_q[i] for i in range(6)},
            }
        )
        obs, _, terminated, truncated, final_info = env.step(action)
        actions.append(action)
        step += 1
    runtime = time.perf_counter() - start
    env.close()
    row = {
        "policy_id": "STANDARD_DQN_EQUAL_WEIGHT_Q_ENSEMBLE",
        "episode_id": episode.episode_id,
        "scenario_id": episode.scenario_id,
        "calibration_mode": episode.calibration_mode,
        "episode_index": episode.episode_index,
        "normalized_profit": safe_float(final_info.get("normalized_accounting_profit")),
        "raw_accounting_profit": safe_float(final_info.get("raw_accounting_profit")),
        "revenue": safe_float(final_info.get("cumulative_revenue")),
        "waste_rate": safe_float(final_info.get("final_waste_rate")),
        "sell_through": safe_float(final_info.get("final_sell_through_rate")),
        "average_markdown": safe_float(final_info.get("average_markdown")),
        "physical_waste_units": safe_float(final_info.get("cumulative_waste")),
        "action_sequence": "|".join(str(a) for a in actions),
        "action_entropy": action_entropy(actions),
        "runtime_seconds": runtime,
        "episode_conservation_error": safe_float(final_info.get("episode_conservation_error"), 0.0),
        "raw_financial_sum_error": safe_float(final_info.get("raw_financial_sum_error"), 0.0),
    }
    return row, pd.DataFrame(state_rows)


def action_distribution(sequences: pd.Series) -> dict[str, float]:
    actions: list[int] = []
    for seq in sequences.dropna():
        actions.extend([int(x) for x in str(seq).split("|") if x != ""])
    s = pd.Series(actions)
    if s.empty:
        return {f"action_{i}_share": np.nan for i in range(6)}
    counts = s.value_counts(normalize=True)
    return {f"action_{i}_share": float(counts.get(i, 0.0)) for i in range(6)}


def compare_to_baseline(policy: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, Any]:
    merged = policy.merge(baseline, on="episode_id", suffixes=("_policy", "_baseline"))
    diff = merged["normalized_profit_policy"].astype(float) - merged["normalized_profit_baseline"].astype(float)
    ci_low, ci_high = bootstrap_ci(diff)
    return {
        "paired_episode_count": int(len(merged)),
        "paired_mean_gain_vs_always_0pct": float(diff.mean()),
        "paired_median_gain": float(diff.median()),
        "episode_cluster_ci_low": ci_low,
        "episode_cluster_ci_high": ci_high,
        "win_share": float((diff > TIE_TOL).mean()),
        "tie_share": float((diff.abs() <= TIE_TOL).mean()),
        "loss_share": float((diff < -TIE_TOL).mean()),
        "waste_rate_difference": float((merged["waste_rate_policy"] - merged["waste_rate_baseline"]).mean()),
        "sell_through_difference": float((merged["sell_through_policy"] - merged["sell_through_baseline"]).mean()),
        "worst_decile_gain": float(diff.sort_values().head(max(1, math.ceil(len(diff) * 0.10))).mean()),
        "maximum_episode_loss": float(diff.min()),
    }


def evaluate_ensemble() -> tuple[pd.DataFrame, pd.DataFrame]:
    episodes = load_manifest()
    members = load_members(episodes[0].scenario_id)
    rows = []
    states = []
    for idx, episode in enumerate(episodes, start=1):
        print(f"Evaluating ensemble episode {idx}/{len(episodes)}: {episode.episode_id}", flush=True)
        row, state = evaluate_ensemble_episode(episode, members)
        rows.append(row)
        states.append(state)
    for member in members:
        member.vecnormalize.close()
    ensemble = pd.DataFrame(rows)
    state_df = pd.concat(states, ignore_index=True) if states else pd.DataFrame()
    reference = pd.read_csv(TABLES_DIR / "dqn_checkpoint_validation_episode_results.csv")
    baseline = reference.loc[reference["policy_id"].eq(BASELINE_POLICY)]
    comparison = compare_to_baseline(ensemble, baseline)
    dist = action_distribution(ensemble["action_sequence"])
    actions = []
    for seq in ensemble["action_sequence"]:
        actions.extend([int(x) for x in str(seq).split("|") if x != ""])
    dominant = pd.Series(actions).value_counts(normalize=True).max() if actions else np.nan
    summary = {
        "policy_id": "STANDARD_DQN_EQUAL_WEIGHT_Q_ENSEMBLE",
        "mean_normalized_profit": float(ensemble["normalized_profit"].mean()),
        "raw_accounting_profit": float(ensemble["raw_accounting_profit"].mean()),
        "revenue": float(ensemble["revenue"].mean()),
        "waste_rate": float(ensemble["waste_rate"].mean()),
        "sell_through": float(ensemble["sell_through"].mean()),
        "average_markdown": float(ensemble["average_markdown"].mean()),
        "action_entropy": action_entropy(actions),
        "zero_action_share": float((pd.Series(actions) == 0).mean()) if actions else np.nan,
        "positive_action_share": float((pd.Series(actions) > 0).mean()) if actions else np.nan,
        "dominant_action_share": float(dominant),
        "state_dependence": "STATE_DEPENDENT" if action_entropy(actions) > 0.05 and dominant < 0.98 else "FIXED_OR_COLLAPSED",
        "collapse_flag": bool(dominant >= 0.98),
        "accounting_valid": bool((ensemble["episode_conservation_error"].abs().max() <= 1e-6) and (ensemble["raw_financial_sum_error"].abs().max() <= 1e-6)),
        "runtime_seconds": float(ensemble["runtime_seconds"].sum()),
        **comparison,
        **dist,
    }
    out = pd.DataFrame([summary])
    out.to_csv(TABLES_DIR / "dqn_equal_weight_ensemble_validation.csv", index=False)
    state_df.to_csv(TABLES_DIR / "dqn_equal_weight_ensemble_state_actions.csv", index=False)
    return out, ensemble


def decide_status(clustered: pd.DataFrame, ensemble_summary: pd.DataFrame) -> str:
    episode_mean = clustered.loc[clustered["analysis"].eq("EPISODE_MEAN_ACROSS_SEEDS")].iloc[0]
    ens = ensemble_summary.iloc[0]
    clustered_positive = safe_float(episode_mean["ci_low"]) > 0
    eligible = (
        safe_float(ens["paired_mean_gain_vs_always_0pct"]) > 0
        and safe_float(ens["win_share"]) >= WIN_SHARE_THRESHOLD
        and safe_float(ens["waste_rate_difference"]) <= 0.02
        and str(ens["state_dependence"]) == "STATE_DEPENDENT"
        and not bool(ens["collapse_flag"])
        and bool(ens["accounting_valid"])
        and safe_float(ens["worst_decile_gain"]) > -0.08
        and clustered_positive
    )
    if eligible:
        return "DQN_ENSEMBLE_BEATS_BASELINE_ON_VALIDATION"
    if not clustered_positive:
        return "DQN_POSITIVE_SIGNAL_NOT_STATISTICALLY_CONFIRMED"
    return "DQN_FAMILY_POSITIVE_BUT_NOT_ROBUST"


def make_figures(episode_means: pd.DataFrame, clustered: pd.DataFrame, ensemble_summary: pd.DataFrame, ensemble_episodes: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    episode_means.set_index("episode_id")["episode_mean_gain"].plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Episode-Mean Gain Across Seeds")
    ax.set_ylabel("Mean gain")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "episode_mean_gain_across_seeds.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    row = clustered.loc[clustered["analysis"].eq("EPISODE_MEAN_ACROSS_SEEDS")].iloc[0]
    mean, low, high = safe_float(row["mean_gain"]), safe_float(row["ci_low"]), safe_float(row["ci_high"])
    ax.errorbar([0], [mean], yerr=[[mean - low], [high - mean]], fmt="o")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks([0])
    ax.set_xticklabels(["clustered DQN"])
    ax.set_title("Corrected Clustered Confidence Interval")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "corrected_clustered_confidence_interval.png", dpi=160)
    plt.close(fig)

    seed_stability = pd.read_csv(TABLES_DIR / "dqn_seed_stability.csv")
    comp = pd.concat(
        [
            seed_stability[["selected_policy_id", "mean_gain"]].rename(columns={"selected_policy_id": "policy_id", "mean_gain": "gain"}),
            pd.DataFrame([{"policy_id": "equal_weight_ensemble", "gain": ensemble_summary["paired_mean_gain_vs_always_0pct"].iloc[0]}]),
        ],
        ignore_index=True,
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    comp.set_index("policy_id")["gain"].astype(float).plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Individual Seeds versus Equal-Weight Ensemble")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "individual_seeds_vs_equal_weight_ensemble.png", dpi=160)
    plt.close(fig)

    baseline = pd.read_csv(TABLES_DIR / "dqn_checkpoint_validation_episode_results.csv")
    baseline = baseline.loc[baseline["policy_id"].eq(BASELINE_POLICY)]
    merged = ensemble_episodes.merge(baseline, on="episode_id", suffixes=("_ensemble", "_baseline"))
    diff = merged["normalized_profit_ensemble"].astype(float) - merged["normalized_profit_baseline"].astype(float)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    diff.plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Ensemble Paired Episode Differences")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ensemble_paired_episode_differences.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter([ensemble_summary["waste_rate"].iloc[0]], [ensemble_summary["mean_normalized_profit"].iloc[0]], label="ensemble")
    ax.scatter([baseline["waste_rate"].astype(float).mean()], [baseline["normalized_profit"].astype(float).mean()], label="always_0pct")
    ax.set_xlabel("Waste rate")
    ax.set_ylabel("Mean normalized profit")
    ax.set_title("Profit-Waste Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "profit_waste_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    pd.Series({"ensemble_worst_decile": ensemble_summary["worst_decile_gain"].iloc[0], "family_worst_decile": row["worst_decile_gain"]}).plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Tail-Loss Comparison")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "tail_loss_comparison.png", dpi=160)
    plt.close(fig)


def write_decision_doc(status: str, audit: pd.DataFrame, clustered: pd.DataFrame, ensemble: pd.DataFrame) -> None:
    text = f"""# DQN Final Statistical Decision

Final validation-only status: `{status}`.

The original pooled analysis used seed-episode rows and is classified as
`{audit.loc[audit['audit_item'].eq('classification'), 'value'].iloc[0]}` because
66 rows come from 22 unique validation episodes evaluated across three seeds.

Corrected clustered inference:

```json
{json.dumps(clustered.to_dict('records'), indent=2)}
```

Equal-weight Q ensemble validation:

```json
{json.dumps(ensemble.iloc[0].to_dict(), indent=2)}
```

No test split was used. A final test evaluation is justified only if the
unchanged validation-lock eligibility criteria are satisfied.
"""
    (DOCS_DIR / "dqn_final_statistical_decision.md").write_text(text, encoding="utf-8")


def main() -> None:
    start = time.perf_counter()
    ensure_dirs()
    try:
        audit = audit_pooled_inference()
        clustered, episode_means = clustered_inference()
        ensemble_summary, ensemble_episodes = evaluate_ensemble()
        status = decide_status(clustered, ensemble_summary)
        make_figures(episode_means, clustered, ensemble_summary, ensemble_episodes)
        write_decision_doc(status, audit, clustered, ensemble_summary)
        config = {
            "status": status,
            "final_model_locked": status == "DQN_ENSEMBLE_BEATS_BASELINE_ON_VALIDATION",
            "test_split_used": False,
            "population": POPULATION_ID,
            "calibration_mode": CALIBRATION_MODE,
            "locked_baseline": BASELINE_POLICY,
            "selected_seed_checkpoints": SELECTED,
            "ensemble": "equal_weight_q_average",
            "ensemble_validation": ensemble_summary.iloc[0].to_dict(),
            "clustered_inference": clustered.to_dict("records"),
            "test_evaluation_justified": status == "DQN_ENSEMBLE_BEATS_BASELINE_ON_VALIDATION",
        }
        (CONFIGS_DIR / "dqn_equal_weight_ensemble_candidate.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        status = "DQN_STATISTICAL_AUDIT_REQUIRES_REVISION"
        (CONFIGS_DIR / "dqn_equal_weight_ensemble_candidate.json").write_text(
            json.dumps({"status": status, "error": str(exc), "test_split_used": False}, indent=2),
            encoding="utf-8",
        )
    runtime = time.perf_counter() - start
    print(status)
    print(f"runtime_seconds={runtime:.1f}")
    print("test_split_used=False")
    print("files_created:")
    for path in [
        TABLES_DIR / "dqn_pooled_inference_audit.csv",
        TABLES_DIR / "dqn_clustered_validation_inference.csv",
        TABLES_DIR / "dqn_equal_weight_ensemble_validation.csv",
        CONFIGS_DIR / "dqn_equal_weight_ensemble_candidate.json",
        DOCS_DIR / "dqn_final_statistical_decision.md",
    ]:
        print(f"- {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
