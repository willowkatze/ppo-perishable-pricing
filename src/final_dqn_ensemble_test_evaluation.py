"""
Final locked STANDARD_DQN equal-weight ensemble test evaluation.

This is a one-time held-out test script. It writes and hashes the final
candidate specification and test protocol before constructing or reading the
test split. It does not train, retrain, tune, or select any model.

Run from project root:
    python -u src/final_dqn_ensemble_test_evaluation.py
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

if __package__:
    from src.dqn_statistical_audit_and_ensemble import SELECTED, ensemble_action, load_members
    from src.high_risk_b_planning_distillation import feature_row_from_obs, is_high_risk_b
    from src.train_dqn_high_risk_b import (
        BASELINE_POLICY,
        BOOTSTRAP_SEED,
        CALIBRATION_MODE,
        CONFIGS_DIR,
        LAMBDA_WASTE,
        PLANNING_POLICY,
        POPULATION_ID,
        PROJECT_ROOT,
        REWARD_MODE,
        SELECTED_PPO_POLICY,
        TABLES_DIR,
        TIE_TOL,
        action_entropy,
        load_validation_manifest,
        make_env,
        reset_env,
        safe_float,
        sha256,
    )
else:
    from dqn_statistical_audit_and_ensemble import SELECTED, ensemble_action, load_members
    from high_risk_b_planning_distillation import feature_row_from_obs, is_high_risk_b
    from train_dqn_high_risk_b import (
        BASELINE_POLICY,
        BOOTSTRAP_SEED,
        CALIBRATION_MODE,
        CONFIGS_DIR,
        LAMBDA_WASTE,
        PLANNING_POLICY,
        POPULATION_ID,
        PROJECT_ROOT,
        REWARD_MODE,
        SELECTED_PPO_POLICY,
        TABLES_DIR,
        TIE_TOL,
        action_entropy,
        load_validation_manifest,
        make_env,
        reset_env,
        safe_float,
        sha256,
    )

MANIFESTS_DIR = PROJECT_ROOT / "outputs" / "manifests"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "final_dqn_ensemble_test"
DOCS_DIR = PROJECT_ROOT / "docs"
MODELS_ROOT = PROJECT_ROOT / "outputs" / "models" / "dqn_high_risk_b" / "STANDARD_DQN"
BOOTSTRAP_N = 10_000
TEST_BOOTSTRAP_SEED = 20260719
DEFAULT_MAX_TEST_EPISODES_PER_SCENARIO = 300
DEFAULT_MAX_TEST_EPISODES_TOTAL = 60
PARTIAL_TEST_MANIFEST = MANIFESTS_DIR / "high_risk_b_final_test_manifest.partial.csv"
FINAL_TEST_MANIFEST = MANIFESTS_DIR / "high_risk_b_final_test_manifest.csv"


def ensure_dirs() -> None:
    for path in [TABLES_DIR, CONFIGS_DIR, MANIFESTS_DIR, FIGURES_DIR, DOCS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def read_csv(path: Path) -> pd.DataFrame:
    require_file(path)
    return pd.read_csv(path)


def now_timestamp() -> str:
    return pd.Timestamp.utcnow().isoformat()


def infer_episode_count(env: Any, fallback: int = DEFAULT_MAX_TEST_EPISODES_PER_SCENARIO) -> int:
    for attr in ["n_episodes", "num_episodes", "episode_count"]:
        if hasattr(env, attr):
            value = getattr(env, attr)
            if isinstance(value, (int, np.integer)) and int(value) > 0:
                return int(value)
    for attr in ["episodes", "episode_records", "episode_table", "episode_indices"]:
        if hasattr(env, attr):
            try:
                return int(len(getattr(env, attr)))
            except Exception:
                pass
    return fallback


def selected_component_paths() -> list[dict[str, Any]]:
    rows = []
    for seed, step in SELECTED.items():
        model_path = MODELS_ROOT / f"seed_{seed}" / "checkpoints" / f"dqn_{step}_steps.zip"
        vec_path = MODELS_ROOT / f"seed_{seed}" / "checkpoints" / f"vecnormalize_{step}_steps.pkl"
        require_file(model_path)
        require_file(vec_path)
        rows.append(
            {
                "seed": int(seed),
                "checkpoint_timesteps": int(step),
                "model_id": f"STANDARD_DQN__seed_{seed}__{step}",
                "model_path": str(model_path.relative_to(PROJECT_ROOT)),
                "model_sha256": sha256(model_path),
                "vecnormalize_path": str(vec_path.relative_to(PROJECT_ROOT)),
                "vecnormalize_sha256": sha256(vec_path),
            }
        )
    return rows


def load_high_risk_definition() -> dict[str, Any]:
    locked = read_csv(TABLES_DIR / "final_task_locked_population_definition.csv")
    row = locked.loc[locked["population_id"].astype(str).eq(POPULATION_ID)]
    if row.empty:
        raise ValueError("Locked HIGH_RISK_B definition not found.")
    return row.iloc[0].to_dict()


def write_candidate_lock() -> tuple[Path, Path, dict[str, Any]]:
    lock_path = CONFIGS_DIR / "final_locked_dqn_ensemble.json"
    hashes_path = CONFIGS_DIR / "final_locked_dqn_ensemble_hashes.json"
    if lock_path.exists() and hashes_path.exists():
        candidate = json.loads(lock_path.read_text(encoding="utf-8"))
        print("FINAL_CANDIDATE_LOCKED_BEFORE_TEST", flush=True)
        print(f"Reusing existing candidate lock: {lock_path.relative_to(PROJECT_ROOT)}", flush=True)
        return lock_path, hashes_path, candidate
    locked_def = load_high_risk_definition()
    validation_manifest = MANIFESTS_DIR / "high_risk_b_expanded_validation_manifest.csv"
    if not validation_manifest.exists():
        validation_manifest = TABLES_DIR / "limited_horizon_planning_episode_results.csv"
    component_rows = selected_component_paths()
    candidate = {
        "lock_status": "FINAL_CANDIDATE_LOCKED_BEFORE_TEST",
        "model_name": "STANDARD_DQN_EQUAL_WEIGHT_Q_ENSEMBLE",
        "component_model_ids": [row["model_id"] for row in component_rows],
        "components": component_rows,
        "preprocessing": "Each component uses its own saved VecNormalize observation statistics; norm_reward=False.",
        "ensemble_rule": "Q_ensemble(s,a)=mean(Q_seed42(s,a), Q_seed123(s,a), Q_seed456(s,a)); action=argmax_a Q_ensemble(s,a)",
        "ensemble_weights": {"42": 1 / 3, "123": 1 / 3, "456": 1 / 3},
        "deterministic_inference": True,
        "population_id": POPULATION_ID,
        "high_risk_b_definition": locked_def,
        "calibration_mode": CALIBRATION_MODE,
        "locked_baseline": BASELINE_POLICY,
        "action_mapping": {"0": 0.0, "1": 0.05, "2": 0.10, "3": 0.15, "4": 0.20, "5": 0.30},
        "reward_mode": REWARD_MODE,
        "lambda_waste": LAMBDA_WASTE,
        "accounting_version": "existing pricing_env_operational normalized accounting profit",
        "validation_manifest": str(validation_manifest.relative_to(PROJECT_ROOT)),
        "validation_manifest_sha256": sha256(validation_manifest),
        "test_success_criteria": {
            "primary_endpoint": "paired normalized-profit gain = ensemble - always_0pct",
            "primary_success_condition": "mean paired normalized-profit gain > 0 with valid accounting and pairing",
            "strict_secondary_robustness_diagnostic": "win_share >= 0.60",
        },
        "locked_utc": now_timestamp(),
        "test_split_accessed_before_lock": False,
    }
    lock_path.write_text(json.dumps(candidate, indent=2, default=str), encoding="utf-8")
    hash_rows = [
        {"path": item["model_path"], "sha256": item["model_sha256"], "artifact_type": "component_model"} for item in component_rows
    ] + [
        {"path": item["vecnormalize_path"], "sha256": item["vecnormalize_sha256"], "artifact_type": "vecnormalize"} for item in component_rows
    ] + [
        {"path": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": sha256(lock_path), "artifact_type": "candidate_lock"},
        {"path": str(validation_manifest.relative_to(PROJECT_ROOT)), "sha256": sha256(validation_manifest), "artifact_type": "validation_manifest"},
    ]
    hashes_path.write_text(json.dumps(hash_rows, indent=2), encoding="utf-8")
    print("FINAL_CANDIDATE_LOCKED_BEFORE_TEST", flush=True)
    return lock_path, hashes_path, candidate


def write_protocol() -> Path:
    path = CONFIGS_DIR / "final_test_evaluation_protocol.json"
    if path.exists():
        print(f"Reusing existing locked test protocol: {path.relative_to(PROJECT_ROOT)}", flush=True)
        return path
    protocol = {
        "protocol_status": "LOCKED_BEFORE_TEST",
        "primary_endpoint": "paired normalized-profit gain = ensemble normalized profit - always_0pct normalized profit",
        "primary_success_condition": "mean paired normalized-profit gain > 0",
        "secondary_evidence": [
            "paired median gain",
            "episode-level bootstrap 95% CI",
            "win/tie/loss shares",
            "waste-rate difference",
            "sell-through difference",
            "average markdown difference",
            "worst-decile gain",
            "maximum episode loss",
            "action entropy",
            "zero-action share",
            "positive-action share",
            "accounting validity",
            "numerical validity",
        ],
        "strict_secondary_robustness_diagnostic": "win_share >= 0.60",
        "bootstrap": {"sampling_unit": "test_episode_id", "resamples": BOOTSTRAP_N, "seed": TEST_BOOTSTRAP_SEED},
        "test_split_accessed_before_protocol_lock": False,
        "locked_utc": now_timestamp(),
    }
    path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    return path


def test_episode_metadata(env: Any, episode_index: int, scenario_id: str) -> dict[str, Any]:
    info = getattr(env, "last_info", {}) or {}
    store_id = getattr(env, "store_id", "")
    product_id = getattr(env, "product_id", "")
    start_date = None
    if hasattr(env, "current_series") and not env.current_series.empty and "timestamp" in env.current_series.columns:
        start_date = str(pd.Timestamp(env.current_series.iloc[0]["timestamp"]).date())
    return {
        "scenario_id": scenario_id,
        "calibration_mode": CALIBRATION_MODE,
        "episode_index": int(episode_index),
        "episode_id": f"test__{scenario_id}__{CALIBRATION_MODE}__episode_{episode_index:05d}",
        "store_id": str(store_id),
        "product_id": str(product_id),
        "start_date": start_date,
        "horizon": int(getattr(env, "horizon", 0)),
        "initial_inventory": safe_float(getattr(env, "initial_inventory", np.nan)),
    }


def save_manifest_checkpoint(rows: list[dict[str, Any]]) -> None:
    if rows:
        pd.DataFrame(rows).drop_duplicates(subset=["scenario_id", "episode_index"]).to_csv(PARTIAL_TEST_MANIFEST, index=False)


def build_test_manifest(
    locked_def: dict[str, Any],
    *,
    max_episodes_per_scenario: int,
    max_test_episodes_total: int | None,
    checkpoint_every: int,
    resume_manifest: bool,
) -> pd.DataFrame:
    if resume_manifest and FINAL_TEST_MANIFEST.exists():
        manifest = pd.read_csv(FINAL_TEST_MANIFEST)
        print(f"Reusing existing final test manifest: episodes={len(manifest)}", flush=True)
        return manifest
    rows: list[dict[str, Any]] = []
    completed_keys: set[tuple[str, int]] = set()
    if resume_manifest and PARTIAL_TEST_MANIFEST.exists():
        partial = pd.read_csv(PARTIAL_TEST_MANIFEST)
        rows = partial.to_dict("records")
        completed_keys = {(str(r["scenario_id"]), int(r["episode_index"])) for r in rows}
        print(f"Resuming partial test manifest: selected_episodes={len(rows)}", flush=True)
    scenario_ids = [x for x in str(locked_def["scenario_ids"]).split("|") if x]
    scanned = 0
    for scenario_id in scenario_ids:
        probe = make_env("test", scenario_id, BOOTSTRAP_SEED)
        episode_count = min(infer_episode_count(probe), int(max_episodes_per_scenario))
        probe.close()
        print(
            f"Scanning test scenario {scenario_id}: max episodes={episode_count}, "
            f"selected_so_far={len(rows)}",
            flush=True,
        )
        for episode_index in range(episode_count):
            if (scenario_id, episode_index) in completed_keys:
                continue
            if max_test_episodes_total is not None and len(rows) >= max_test_episodes_total:
                print(f"Reached --max-test-episodes-total={max_test_episodes_total}", flush=True)
                break
            env = make_env("test", scenario_id, BOOTSTRAP_SEED + episode_index)
            try:
                obs, _ = reset_env(env, episode_index)
            except Exception:
                env.close()
                continue
            selected = False
            terminated = truncated = False
            step = 0
            while not (terminated or truncated):
                features = feature_row_from_obs(obs, scenario_id, episode_index, step)
                if is_high_risk_b(features, pd.Series(locked_def)):
                    selected = True
                    break
                obs, _, terminated, truncated, _ = env.step(0)
                step += 1
            if selected:
                meta = test_episode_metadata(env, episode_index, scenario_id)
                meta["first_high_risk_step"] = step
                rows.append(meta)
                completed_keys.add((scenario_id, episode_index))
            env.close()
            scanned += 1
            if scanned == 1 or scanned % max(1, checkpoint_every) == 0:
                save_manifest_checkpoint(rows)
                print(
                    f"Test manifest checkpoint: scanned={scanned}, selected={len(rows)}, "
                    f"partial={PARTIAL_TEST_MANIFEST.relative_to(PROJECT_ROOT)}",
                    flush=True,
                )
        if max_test_episodes_total is not None and len(rows) >= max_test_episodes_total:
            break
    manifest = pd.DataFrame(rows)
    if not manifest.empty:
        manifest = manifest.drop_duplicates(subset=["scenario_id", "episode_index"]).sort_values(["scenario_id", "episode_index"])
    manifest.to_csv(FINAL_TEST_MANIFEST, index=False)
    manifest.to_csv(PARTIAL_TEST_MANIFEST, index=False)
    return manifest


def population_audit(manifest: pd.DataFrame) -> pd.DataFrame:
    validation = load_validation_manifest()
    rows = [
        {"metric": "eligible_test_episodes", "value": int(len(manifest)), "details": ""},
        {"metric": "test_scenarios", "value": int(manifest["scenario_id"].nunique()) if not manifest.empty else 0, "details": "|".join(sorted(manifest["scenario_id"].astype(str).unique())) if not manifest.empty else ""},
        {"metric": "validation_episode_overlap_count", "value": int(set(manifest.get("episode_id", [])).intersection(set(validation["episode_id"])).__len__()), "details": "episode_id namespace includes split"},
        {"metric": "store_product_overlap_with_validation", "value": "not_applicable_without_validation_store_product_manifest", "details": "FreshRetail split design is temporal/split-contained; no threshold changed."},
    ]
    if not manifest.empty:
        for scenario, count in manifest["scenario_id"].value_counts().sort_index().items():
            rows.append({"metric": f"scenario_count__{scenario}", "value": int(count), "details": ""})
        rows.append({"metric": "inventory_risk_distribution", "value": "HIGH_RISK_B_locked", "details": "coverage threshold inherited from lock"})
        rows.append({"metric": "expiry_risk_distribution", "value": "HIGH_RISK_B_locked", "details": "fraction-expiring-within-two-days threshold inherited from lock"})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "final_dqn_ensemble_test_population_audit.csv", index=False)
    return out


def evaluate_policy_episode(policy_id: str, episode: pd.Series, members: list[Any] | None = None) -> tuple[dict[str, Any], pd.DataFrame]:
    env = make_env("test", str(episode["scenario_id"]), BOOTSTRAP_SEED)
    obs, _ = reset_env(env, int(episode["episode_index"]))
    actions: list[int] = []
    state_rows: list[dict[str, Any]] = []
    final_info: dict[str, Any] = {}
    terminated = truncated = False
    start = time.perf_counter()
    step = 0
    while not (terminated or truncated):
        if policy_id == BASELINE_POLICY:
            action = 0
        elif policy_id == "STANDARD_DQN_EQUAL_WEIGHT_Q_ENSEMBLE":
            if members is None:
                raise ValueError("members required for ensemble policy")
            action, mean_q = ensemble_action(members, obs)
        else:
            raise ValueError(policy_id)
        state_rows.append(
            {
                "episode_id": episode["episode_id"],
                "policy_id": policy_id,
                "scenario_id": episode["scenario_id"],
                "episode_index": int(episode["episode_index"]),
                "step_index": step,
                "action": int(action),
            }
        )
        obs, _, terminated, truncated, final_info = env.step(int(action))
        actions.append(int(action))
        step += 1
    runtime = time.perf_counter() - start
    env.close()
    row = {
        "episode_id": episode["episode_id"],
        "scenario_id": episode["scenario_id"],
        "calibration_mode": CALIBRATION_MODE,
        "episode_index": int(episode["episode_index"]),
        "policy_id": policy_id,
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


def bootstrap_summary(diff: pd.Series) -> tuple[float, float]:
    values = diff.dropna().astype(float).to_numpy()
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(TEST_BOOTSTRAP_SEED)
    samples = np.empty(BOOTSTRAP_N)
    for i in range(BOOTSTRAP_N):
        samples[i] = rng.choice(values, size=len(values), replace=True).mean()
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def action_distribution(sequences: pd.Series) -> dict[str, float]:
    actions: list[int] = []
    for seq in sequences.dropna():
        actions.extend([int(x) for x in str(seq).split("|") if x != ""])
    s = pd.Series(actions)
    if s.empty:
        return {f"action_{i}_share": np.nan for i in range(6)}
    counts = s.value_counts(normalize=True)
    return {f"action_{i}_share": float(counts.get(i, 0.0)) for i in range(6)}


def summarize_test(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ens = results.loc[results["policy_id"].eq("STANDARD_DQN_EQUAL_WEIGHT_Q_ENSEMBLE")]
    base = results.loc[results["policy_id"].eq(BASELINE_POLICY)]
    merged = ens.merge(base, on="episode_id", suffixes=("_ensemble", "_baseline"))
    diff = merged["normalized_profit_ensemble"].astype(float) - merged["normalized_profit_baseline"].astype(float)
    ci_low, ci_high = bootstrap_summary(diff)
    ens_dist = action_distribution(ens["action_sequence"])
    actions = []
    for seq in ens["action_sequence"]:
        actions.extend([int(x) for x in str(seq).split("|") if x != ""])
    accounting_valid_rate = float((results["episode_conservation_error"].abs() <= 1e-6).mean() and (results["raw_financial_sum_error"].abs() <= 1e-6).mean())
    summary = pd.DataFrame(
        [
            {
                "ensemble_mean_normalized_profit": float(ens["normalized_profit"].mean()),
                "baseline_mean_normalized_profit": float(base["normalized_profit"].mean()),
                "paired_mean_gain": float(diff.mean()),
                "paired_median_gain": float(diff.median()),
                "paired_gain_std": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "win_share": float((diff > TIE_TOL).mean()),
                "tie_share": float((diff.abs() <= TIE_TOL).mean()),
                "loss_share": float((diff < -TIE_TOL).mean()),
                "waste_rate_difference": float((merged["waste_rate_ensemble"] - merged["waste_rate_baseline"]).mean()),
                "sell_through_difference": float((merged["sell_through_ensemble"] - merged["sell_through_baseline"]).mean()),
                "average_markdown_difference": float((merged["average_markdown_ensemble"] - merged["average_markdown_baseline"]).mean()),
                "worst_decile_gain": float(diff.sort_values().head(max(1, math.ceil(len(diff) * 0.10))).mean()),
                "maximum_episode_loss": float(diff.min()),
                "action_entropy": action_entropy(actions),
                "zero_action_share": float((pd.Series(actions) == 0).mean()) if actions else np.nan,
                "positive_action_share": float((pd.Series(actions) > 0).mean()) if actions else np.nan,
                "accounting_valid_rate": accounting_valid_rate,
                "numerical_valid": bool(np.isfinite(results.select_dtypes(include=[np.number])).all().all()),
                "test_episode_count": int(len(merged)),
                **ens_dist,
            }
        ]
    )
    summary.to_csv(TABLES_DIR / "final_dqn_ensemble_test_summary.csv", index=False)
    pairing_rows = []
    for episode_id, group in results.groupby("episode_id"):
        policies = set(group["policy_id"])
        pairing_rows.append(
            {
                "episode_id": episode_id,
                "has_ensemble": "STANDARD_DQN_EQUAL_WEIGHT_Q_ENSEMBLE" in policies,
                "has_baseline": BASELINE_POLICY in policies,
                "pairing_status": "PAIRED_VALID" if {"STANDARD_DQN_EQUAL_WEIGHT_Q_ENSEMBLE", BASELINE_POLICY}.issubset(policies) else "PAIRING_INVALID",
            }
        )
    pairing = pd.DataFrame(pairing_rows)
    pairing.to_csv(TABLES_DIR / "final_dqn_ensemble_test_pairing_audit.csv", index=False)
    return summary, pairing


def final_status(summary: pd.DataFrame, pairing: pd.DataFrame) -> str:
    row = summary.iloc[0]
    if not pairing["pairing_status"].eq("PAIRED_VALID").all() or safe_float(row["accounting_valid_rate"]) < 1.0 or not bool(row["numerical_valid"]):
        return "FINAL_TEST_EVALUATION_REQUIRES_REVISION"
    if safe_float(row["paired_mean_gain"]) <= 0:
        return "FINAL_DQN_MODEL_NOT_CONFIRMED_ON_TEST"
    if safe_float(row["bootstrap_ci_low"]) <= 0 or safe_float(row["win_share"]) < 0.60:
        return "FINAL_DQN_MODEL_POSITIVE_BUT_UNCERTAIN_ON_TEST"
    return "FINAL_DQN_MODEL_BEATS_BASELINE_ON_TEST"


def make_figures(results: pd.DataFrame, summary: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ens = results.loc[results["policy_id"].eq("STANDARD_DQN_EQUAL_WEIGHT_Q_ENSEMBLE")]
    base = results.loc[results["policy_id"].eq(BASELINE_POLICY)]
    merged = ens.merge(base, on="episode_id", suffixes=("_ensemble", "_baseline"))
    diff = merged["normalized_profit_ensemble"].astype(float) - merged["normalized_profit_baseline"].astype(float)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    pd.Series({"ensemble": ens["normalized_profit"].mean(), "always_0pct": base["normalized_profit"].mean()}).plot(kind="bar", ax=ax)
    ax.set_title("Final Test Normalized Profit")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "final_test_ensemble_vs_baseline_profit.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    diff.plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Paired Test Episode Profit Differences")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "paired_test_episode_profit_differences.png", dpi=160)
    plt.close(fig)

    row = summary.iloc[0]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    mean, low, high = safe_float(row["paired_mean_gain"]), safe_float(row["bootstrap_ci_low"]), safe_float(row["bootstrap_ci_high"])
    ax.errorbar([0], [mean], yerr=[[mean - low], [high - mean]], fmt="o")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks([0])
    ax.set_xticklabels(["test"])
    ax.set_title("Test Bootstrap Confidence Interval")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "test_bootstrap_confidence_interval.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter([ens["waste_rate"].mean()], [ens["normalized_profit"].mean()], label="ensemble")
    ax.scatter([base["waste_rate"].mean()], [base["normalized_profit"].mean()], label="always_0pct")
    ax.set_xlabel("Waste rate")
    ax.set_ylabel("Mean normalized profit")
    ax.set_title("Test Profit-Waste Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "test_profit_waste_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    pd.Series({"win": row["win_share"], "tie": row["tie_share"], "loss": row["loss_share"]}).astype(float).plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Test Win/Tie/Loss Shares")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "test_win_tie_loss_shares.png", dpi=160)
    plt.close(fig)

    validation = pd.read_csv(TABLES_DIR / "dqn_equal_weight_ensemble_validation.csv")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    pd.Series({"validation_gain": validation["paired_mean_gain_vs_always_0pct"].iloc[0], "test_gain": row["paired_mean_gain"]}).astype(float).plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Validation versus Test Comparison")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "validation_vs_test_comparison.png", dpi=160)
    plt.close(fig)

    action_cols = [f"action_{i}_share" for i in range(6)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    summary[action_cols].iloc[0].astype(float).plot(kind="bar", ax=ax)
    ax.set_title("Test Action Distribution")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "test_action_distribution_by_risk_level.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    diff.sort_values().head(12).plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Worst Test Episode Diagnostics")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "worst_test_episode_diagnostics.png", dpi=160)
    plt.close(fig)


def write_final_doc(status: str, candidate: dict[str, Any], manifest: pd.DataFrame, summary: pd.DataFrame) -> None:
    row = summary.iloc[0].to_dict()
    text = f"""# Final DQN Ensemble Test Decision

Final status: `{status}`.

Locked model: `STANDARD_DQN_EQUAL_WEIGHT_Q_ENSEMBLE`.

Locking timestamp: `{candidate['locked_utc']}`.

Locked task: HIGH_RISK_B, recovered calibration, existing financial reward,
six markdown actions, locked baseline `always_0pct`.

Test sample size: {len(manifest)} eligible held-out test episodes.

Test result:

```json
{json.dumps(row, indent=2)}
```

Exact final claim:

The final claim is determined by the locked primary endpoint, paired normalized
profit gain versus always_0pct, with uncertainty and robustness reported
separately. No post-test tuning or model development was performed.

Limitations:

- Test evidence is conditional on the locked HIGH_RISK_B population.
- The ensemble uses normalized accounting assumptions from the existing
  environment, not retailer-deployment accounting.
- The strict win-share threshold is reported as secondary robustness evidence.

Confirmation: no test split was accessed before candidate and protocol locking,
and no post-test tuning was performed by this script.
"""
    (DOCS_DIR / "final_dqn_ensemble_test_decision.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final locked DQN ensemble held-out test evaluation.")
    parser.add_argument("--max-test-episodes-per-scenario", type=int, default=DEFAULT_MAX_TEST_EPISODES_PER_SCENARIO)
    parser.add_argument("--max-test-episodes-total", type=int, default=DEFAULT_MAX_TEST_EPISODES_TOTAL)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--no-resume-manifest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    ensure_dirs()
    try:
        lock_path, hashes_path, candidate = write_candidate_lock()
        protocol_path = write_protocol()
        locked_def = candidate["high_risk_b_definition"]
        manifest = build_test_manifest(
            locked_def,
            max_episodes_per_scenario=args.max_test_episodes_per_scenario,
            max_test_episodes_total=args.max_test_episodes_total,
            checkpoint_every=args.checkpoint_every,
            resume_manifest=not args.no_resume_manifest,
        )
        population_audit(manifest)
        if manifest.empty:
            print("FINAL_TEST_POPULATION_EMPTY")
            return

        members = load_members(str(manifest["scenario_id"].iloc[0]))
        rows: list[dict[str, Any]] = []
        state_rows: list[pd.DataFrame] = []
        for idx, episode in manifest.reset_index(drop=True).iterrows():
            print(f"Evaluating test episode {idx + 1}/{len(manifest)}: {episode['episode_id']}", flush=True)
            base_row, base_states = evaluate_policy_episode(BASELINE_POLICY, episode)
            ens_row, ens_states = evaluate_policy_episode("STANDARD_DQN_EQUAL_WEIGHT_Q_ENSEMBLE", episode, members)
            rows.extend([base_row, ens_row])
            state_rows.extend([base_states, ens_states])
        for member in members:
            member.vecnormalize.close()
        results = pd.DataFrame(rows)
        results.to_csv(TABLES_DIR / "final_dqn_ensemble_test_episode_results.csv", index=False)
        pd.concat(state_rows, ignore_index=True).to_csv(TABLES_DIR / "final_dqn_ensemble_test_state_actions.csv", index=False)
        summary, pairing = summarize_test(results)
        status = final_status(summary, pairing)
        make_figures(results, summary)
        write_final_doc(status, candidate, manifest, summary)
    except Exception as exc:
        status = "FINAL_TEST_EVALUATION_REQUIRES_REVISION"
        (CONFIGS_DIR / "final_test_error.json").write_text(json.dumps({"status": status, "error": str(exc), "test_split_used": "unknown_after_exception"}, indent=2), encoding="utf-8")
    runtime = time.perf_counter() - start
    print(status)
    print(f"runtime_seconds={runtime:.1f}")
    print("files_created:")
    for path in [
        CONFIGS_DIR / "final_locked_dqn_ensemble.json",
        CONFIGS_DIR / "final_locked_dqn_ensemble_hashes.json",
        CONFIGS_DIR / "final_test_evaluation_protocol.json",
        MANIFESTS_DIR / "high_risk_b_final_test_manifest.csv",
        TABLES_DIR / "final_dqn_ensemble_test_episode_results.csv",
        TABLES_DIR / "final_dqn_ensemble_test_summary.csv",
        TABLES_DIR / "final_dqn_ensemble_test_pairing_audit.csv",
        DOCS_DIR / "final_dqn_ensemble_test_decision.md",
    ]:
        print(f"- {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
