from __future__ import annotations

import hashlib
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

from pricing_env_operational import ACTION_MARKDOWNS, OperationalPerishablePricingEnv  # noqa: E402
from train_ppo_operational import DEFAULT_SCENARIO_ID, make_env  # noqa: E402

TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
CONFIGS_DIR = PROJECT_ROOT / "outputs" / "configs"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models" / "ppo_operational"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "ppo_financial_paired_validation"

BOOTSTRAP_SEED = 20260716
VALIDATION_MANIFEST = CONFIGS_DIR / "ppo_validation_episode_manifest.csv"
SELECTED_MODELS = {
    "selected_observed_financial_ppo": {
        "agent_id": "observed_financial",
        "calibration_mode": "observed_calibration",
        "checkpoint_label": "checkpoint_22288",
        "model_path": MODELS_DIR / "observed_financial" / "seed_42" / "checkpoints" / "pilot_observed_financial_22288_steps.zip",
        "vecnormalize_path": MODELS_DIR / "observed_financial" / "seed_42" / "vecnormalize.pkl",
        "training_config_path": MODELS_DIR / "observed_financial" / "seed_42" / "training_config.json",
        "classification": "STATE_DEPENDENT_POLICY",
    },
    "selected_recovered_financial_ppo": {
        "agent_id": "recovered_financial",
        "calibration_mode": "recovered_calibration",
        "checkpoint_label": "checkpoint_22288",
        "model_path": MODELS_DIR / "recovered_financial" / "seed_42" / "checkpoints" / "pilot_recovered_financial_22288_steps.zip",
        "vecnormalize_path": MODELS_DIR / "recovered_financial" / "seed_42" / "vecnormalize.pkl",
        "training_config_path": MODELS_DIR / "recovered_financial" / "seed_42" / "training_config.json",
        "classification": "MOSTLY_ZERO_BUT_STATE_DEPENDENT",
    },
}
BASELINES = [
    "always_0pct",
    "always_5pct",
    "always_10pct",
    "always_20pct",
    "always_40pct",
    "expiry_threshold_rule",
    "inventory_coverage_rule",
    "random_policy",
]
SUMMARY_METRICS = [
    "normalized_accounting_profit",
    "raw_accounting_profit",
    "waste_rate",
    "sell_through_rate",
    "average_markdown",
    "stockout_rate",
]
COMPARISON_METRICS = SUMMARY_METRICS
PRIMARY_COMPARISONS = [
    ("selected_observed_financial_ppo", "always_0pct"),
    ("selected_recovered_financial_ppo", "always_0pct"),
    ("selected_observed_financial_ppo", "expiry_threshold_rule"),
    ("selected_recovered_financial_ppo", "expiry_threshold_rule"),
    ("selected_observed_financial_ppo", "inventory_coverage_rule"),
    ("selected_recovered_financial_ppo", "inventory_coverage_rule"),
    ("selected_recovered_financial_ppo", "selected_observed_financial_ppo"),
]


@dataclass(frozen=True)
class PolicySpec:
    policy_id: str
    kind: str
    calibration_mode: str
    model_path: Path | None = None
    vecnormalize_path: Path | None = None
    classification_prior: str | None = None


def ensure_dirs() -> None:
    for directory in [TABLES_DIR, CONFIGS_DIR, FIGURES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric if np.isfinite(numeric) else 0.0


def action_entropy_from_counts(counts: dict[int, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count:
            p = count / total
            entropy -= p * math.log(p)
    return float(entropy)


def write_selected_model_metadata() -> tuple[dict[str, Any], dict[str, Any]]:
    metadata: dict[str, Any] = {
        "created_for": "formal financial paired validation",
        "validation_split_only": True,
        "test_split_used": False,
        "selected_models": {},
        "limitations": [
            "Only one PPO training seed is available.",
            "Original immutable 10k final models were overwritten by resume training.",
            "Sustainability PPO is not evaluated because lambda wiring remains under review.",
        ],
    }
    hashes: dict[str, Any] = {}
    env_config = CONFIGS_DIR / "pricing_env_operational_config.json"
    selection_table = TABLES_DIR / "ppo_checkpoint_model_selection.csv"
    for policy_id, details in SELECTED_MODELS.items():
        metadata["selected_models"][policy_id] = {
            key: str(value.relative_to(PROJECT_ROOT)) if isinstance(value, Path) and value.exists() else value
            for key, value in details.items()
            if key.endswith("_path") or key in {"agent_id", "calibration_mode", "checkpoint_label", "classification"}
        }
        for key in ["model_path", "vecnormalize_path", "training_config_path"]:
            path = details[key]
            hashes[f"{policy_id}:{key}"] = {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "exists": path.exists(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size if path.exists() else None,
            }
    for label, path in {
        "environment_config": env_config,
        "selected_checkpoint_metadata": selection_table,
        "validation_manifest": VALIDATION_MANIFEST,
    }.items():
        hashes[label] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "exists": path.exists(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size if path.exists() else None,
        }
    (CONFIGS_DIR / "ppo_selected_financial_models.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (CONFIGS_DIR / "ppo_selected_financial_model_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    return metadata, hashes


def load_manifest() -> pd.DataFrame:
    if not VALIDATION_MANIFEST.exists():
        raise FileNotFoundError(f"Missing validation manifest: {VALIDATION_MANIFEST}")
    manifest = pd.read_csv(VALIDATION_MANIFEST)
    required = {
        "episode_id",
        "split",
        "store_id",
        "product_id",
        "start_date",
        "scenario_id",
        "age_profile_seed",
        "demand_noise_seed",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Validation manifest missing columns: {sorted(missing)}")
    manifest = manifest.loc[manifest["split"].eq("validation")].copy()
    if manifest.shape[0] != 20:
        raise ValueError(f"Expected exactly 20 validation episodes, found {manifest.shape[0]}.")
    return manifest


def baseline_action(policy_id: str, obs: np.ndarray, rng: np.random.Generator) -> int:
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
    if policy_id == "random_policy":
        return int(rng.integers(0, 6))
    if policy_id == "expiry_threshold_rule":
        fraction_expiring_within_two_days = safe_float(obs[24]) if obs.shape[0] > 24 else 0.0
        if fraction_expiring_within_two_days >= 0.50:
            return 5
        if fraction_expiring_within_two_days >= 0.25:
            return 3
        if fraction_expiring_within_two_days >= 0.10:
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
    raise ValueError(f"Unknown policy: {policy_id}")


def make_policy_specs() -> list[PolicySpec]:
    specs = [
        PolicySpec(
            policy_id=policy_id,
            kind="ppo",
            calibration_mode=details["calibration_mode"],
            model_path=details["model_path"],
            vecnormalize_path=details["vecnormalize_path"],
            classification_prior=details["classification"],
        )
        for policy_id, details in SELECTED_MODELS.items()
    ]
    specs.extend(PolicySpec(policy_id=baseline, kind="baseline", calibration_mode="recovered_calibration") for baseline in BASELINES)
    return specs


def load_ppo(spec: PolicySpec) -> tuple[PPO, VecNormalize]:
    if spec.model_path is None or spec.vecnormalize_path is None:
        raise ValueError("PPO spec requires model and VecNormalize paths.")
    if not spec.model_path.exists():
        raise FileNotFoundError(spec.model_path)
    if not spec.vecnormalize_path.exists():
        raise FileNotFoundError(spec.vecnormalize_path)
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
    vecnorm = VecNormalize.load(str(spec.vecnormalize_path), raw_env)
    vecnorm.training = False
    vecnorm.norm_reward = False
    return PPO.load(spec.model_path), vecnorm


def eval_policy(spec: PolicySpec, manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model: PPO | None = None
    vecnorm: VecNormalize | None = None
    if spec.kind == "ppo":
        model, vecnorm = load_ppo(spec)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    comparability: list[dict[str, Any]] = []
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
        initial_inventory = safe_float(getattr(env, "initial_inventory", 0.0))
        initial_profile = np.asarray(getattr(env, "inventory", np.array([])), dtype=float).copy()
        horizon = int(getattr(env, "horizon", 0))
        comparability.append(
            {
                "policy_id": spec.policy_id,
                "calibration_mode": spec.calibration_mode,
                "episode_id": episode["episode_id"],
                "store_id": episode["store_id"],
                "product_id": episode["product_id"],
                "start_date": episode["start_date"],
                "scenario_id": episode["scenario_id"],
                "age_profile_seed": episode["age_profile_seed"],
                "demand_noise_seed": episode["demand_noise_seed"],
                "initial_inventory": initial_inventory,
                "procurement_cost_ratio": safe_float(getattr(env, "procurement_cost_ratio", 0.0)),
                "episode_horizon": horizon,
                "starting_age_profile_sum": float(initial_profile.sum()) if initial_profile.size else 0.0,
                "starting_age_profile_signature": "|".join(f"{value:.6f}" for value in initial_profile[:horizon]),
                "inventory_initialization_rule": str(getattr(env, "age_profile", "")),
            }
        )
        terminated = False
        truncated = False
        total_return = 0.0
        action_counts = {action: 0 for action in ACTION_MARKDOWNS}
        first_markdown_day = -1
        max_markdown = 0.0
        final_info: dict[str, Any] = {}
        day = 0
        while not (terminated or truncated):
            state_features = {
                "fraction_expiring_today": safe_float(obs[23]) if obs.shape[0] > 23 else 0.0,
                "fraction_expiring_within_two_days": safe_float(obs[24]) if obs.shape[0] > 24 else 0.0,
                "inventory_coverage": safe_float(obs[1]) if obs.shape[0] > 1 else 0.0,
                "predicted_demand": safe_float(obs[26]) if obs.shape[0] > 26 else 0.0,
                "episode_day": day,
            }
            state_features["stockout_risk_state"] = int(
                state_features["inventory_coverage"] < 0.80 or state_features["predicted_demand"] > 0.80
            )
            if model is None:
                action = baseline_action(spec.policy_id, obs, rng)
            else:
                model_obs = obs.astype(np.float32)
                model_obs = vecnorm.normalize_obs(model_obs[None, :])[0] if vecnorm is not None else model_obs
                predicted, _ = model.predict(model_obs, deterministic=True)
                action = int(np.asarray(predicted).item())
            markdown = ACTION_MARKDOWNS[action]
            if first_markdown_day < 0 and markdown > 0:
                first_markdown_day = day
            max_markdown = max(max_markdown, markdown)
            states.append(
                {
                    "policy_id": spec.policy_id,
                    "kind": spec.kind,
                    "calibration_mode": spec.calibration_mode,
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
        entropy = action_entropy_from_counts(action_counts)
        total_actions = sum(action_counts.values())
        rows.append(
            {
                "policy_id": spec.policy_id,
                "kind": spec.kind,
                "calibration_mode": spec.calibration_mode,
                "episode_id": episode["episode_id"],
                "store_id": episode["store_id"],
                "product_id": episode["product_id"],
                "start_date": episode["start_date"],
                "scenario_id": episode["scenario_id"],
                "normalized_episode_return": total_return,
                "normalized_accounting_profit": safe_float(final_info.get("normalized_accounting_profit")),
                "raw_accounting_profit": safe_float(final_info.get("accounting_profit")),
                "revenue": safe_float(final_info.get("cumulative_revenue")),
                "procurement_cost": safe_float(final_info.get("cumulative_procurement_cost")),
                "disposal_cost": safe_float(final_info.get("cumulative_disposal_cost")),
                "physical_waste_units": safe_float(final_info.get("cumulative_waste")),
                "waste_rate": safe_float(final_info.get("final_waste_rate")),
                "sell_through_rate": safe_float(final_info.get("final_sell_through_rate")),
                "stockout_rate": safe_float(final_info.get("stockout_indicator")),
                "units_sold": safe_float(final_info.get("cumulative_sales")),
                "average_markdown": safe_float(final_info.get("average_markdown")),
                "maximum_markdown": max_markdown,
                "first_markdown_day": first_markdown_day,
                "markdown_timing_relative_to_expiration": total_actions - first_markdown_day if first_markdown_day >= 0 else -1,
                "action_entropy": entropy,
                "episode_length": total_actions,
                "terminal_inventory": safe_float(final_info.get("terminal_inventory")),
                "extrapolation_rate": safe_float(final_info.get("extrapolation_flag")),
                "weak_support_rate": safe_float(final_info.get("weak_support_flag")),
                **{f"action_{action}_share": action_counts[action] / max(total_actions, 1) for action in ACTION_MARKDOWNS},
            }
        )
    if vecnorm is not None:
        vecnorm.close()
    return pd.DataFrame(rows), pd.DataFrame(states), pd.DataFrame(comparability)


def aggregate_summary(episode_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy_id, group in episode_df.groupby("policy_id"):
        for metric in SUMMARY_METRICS:
            values = group[metric].astype(float)
            se = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
            rows.append(
                {
                    "policy_id": policy_id,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "standard_error": se,
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                    "ci95_low": float(values.mean() - 1.96 * se),
                    "ci95_high": float(values.mean() + 1.96 * se),
                    "n_episodes": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def paired_comparisons(episode_df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    for left, right in PRIMARY_COMPARISONS:
        left_df = episode_df.loc[episode_df["policy_id"].eq(left)]
        right_df = episode_df.loc[episode_df["policy_id"].eq(right)]
        merged = left_df.merge(right_df, on="episode_id", suffixes=("_left", "_right"))
        for metric in COMPARISON_METRICS:
            diff = merged[f"{metric}_left"].astype(float) - merged[f"{metric}_right"].astype(float)
            se = float(diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else 0.0
            boot = [float(rng.choice(diff.to_numpy(), size=len(diff), replace=True).mean()) for _ in range(1000)]
            rows.append(
                {
                    "left_policy": left,
                    "right_policy": right,
                    "metric": metric,
                    "paired_mean_difference": float(diff.mean()),
                    "paired_median_difference": float(diff.median()),
                    "standard_error": se,
                    "ci95_low": float(diff.mean() - 1.96 * se),
                    "ci95_high": float(diff.mean() + 1.96 * se),
                    "bootstrap_ci_low": float(np.percentile(boot, 2.5)),
                    "bootstrap_ci_high": float(np.percentile(boot, 97.5)),
                    "win_share": float((diff > 0).mean()),
                    "tie_share": float((diff == 0).mean()),
                    "loss_share": float((diff < 0).mean()),
                    "n_pairs": int(len(diff)),
                }
            )
    return pd.DataFrame(rows)


def profit_comparability(comparability_df: pd.DataFrame) -> pd.DataFrame:
    selected = comparability_df.loc[
        comparability_df["policy_id"].isin(["selected_observed_financial_ppo", "selected_recovered_financial_ppo"])
    ]
    wide = selected.pivot(index="episode_id", columns="policy_id")
    rows: list[dict[str, Any]] = []
    for episode_id in wide.index:
        obs_prefix = ("initial_inventory", "selected_observed_financial_ppo")
        rec_prefix = ("initial_inventory", "selected_recovered_financial_ppo")
        obs_inv = safe_float(wide.loc[episode_id, obs_prefix])
        rec_inv = safe_float(wide.loc[episode_id, rec_prefix])
        obs_sig = str(wide.loc[episode_id, ("starting_age_profile_signature", "selected_observed_financial_ppo")])
        rec_sig = str(wide.loc[episode_id, ("starting_age_profile_signature", "selected_recovered_financial_ppo")])
        obs_horizon = int(wide.loc[episode_id, ("episode_horizon", "selected_observed_financial_ppo")])
        rec_horizon = int(wide.loc[episode_id, ("episode_horizon", "selected_recovered_financial_ppo")])
        rows.append(
            {
                "episode_id": episode_id,
                "observed_initial_inventory": obs_inv,
                "recovered_initial_inventory": rec_inv,
                "initial_inventory_difference": rec_inv - obs_inv,
                "initial_inventory_equal": abs(rec_inv - obs_inv) < 1e-9,
                "episode_horizon_equal": obs_horizon == rec_horizon,
                "starting_age_profile_equal": obs_sig == rec_sig,
                "raw_profit_directly_scale_comparable": abs(rec_inv - obs_inv) < 1e-9 and obs_sig == rec_sig and obs_horizon == rec_horizon,
            }
        )
    audit = pd.DataFrame(rows)
    audit["overall_raw_profit_comparability_status"] = (
        "RAW_PROFIT_DIRECTLY_COMPARABLE" if audit["raw_profit_directly_scale_comparable"].all() else "RAW_PROFIT_NOT_DIRECTLY_SCALE_COMPARABLE"
    )
    return audit


def binned_state_actions(state_df: pd.DataFrame) -> pd.DataFrame:
    selected = state_df.loc[state_df["policy_id"].isin(["selected_observed_financial_ppo", "selected_recovered_financial_ppo"])].copy()
    rows: list[dict[str, Any]] = []
    features = [
        "fraction_expiring_today",
        "fraction_expiring_within_two_days",
        "inventory_coverage",
        "predicted_demand",
        "episode_day",
        "stockout_risk_state",
    ]
    for policy_id, policy_group in selected.groupby("policy_id"):
        for feature in features:
            data = policy_group.copy()
            if feature in {"episode_day", "stockout_risk_state"}:
                data["state_group"] = data[feature].astype(str)
            else:
                q = data[feature].quantile([0.33, 0.66]).to_numpy()
                if q[0] == q[1]:
                    data["state_group"] = np.where(data[feature] <= q[0], "low", "high")
                else:
                    data["state_group"] = pd.cut(data[feature], bins=[-np.inf, q[0], q[1], np.inf], labels=["low", "mid", "high"])
            for group_name, group in data.groupby("state_group", dropna=False):
                counts = group["action"].value_counts().to_dict()
                total = len(group)
                rows.append(
                    {
                        "policy_id": policy_id,
                        "feature": feature,
                        "state_group": str(group_name),
                        "n_states": int(total),
                        "mean_markdown": float(group["markdown"].mean()),
                        "positive_markdown_share": float((group["markdown"] > 0).mean()),
                        "action_entropy": action_entropy_from_counts({a: int(counts.get(a, 0)) for a in ACTION_MARKDOWNS}),
                        **{f"action_{a}_share": float(counts.get(a, 0) / max(total, 1)) for a in ACTION_MARKDOWNS},
                    }
                )
    return pd.DataFrame(rows)


def calibration_transmission(episode_df: pd.DataFrame, state_df: pd.DataFrame) -> pd.DataFrame:
    obs = state_df.loc[state_df["policy_id"].eq("selected_observed_financial_ppo")].copy()
    rec = state_df.loc[state_df["policy_id"].eq("selected_recovered_financial_ppo")].copy()
    obs["state_index"] = obs.groupby("episode_id").cumcount()
    rec["state_index"] = rec.groupby("episode_id").cumcount()
    merged_states = obs.merge(rec, on=["episode_id", "state_index"], suffixes=("_observed", "_recovered"))
    action_diff_share = float((merged_states["action_observed"] != merged_states["action_recovered"]).mean())
    avg_abs_markdown_diff = float((merged_states["markdown_observed"] - merged_states["markdown_recovered"]).abs().mean())
    obs_dist = obs["action"].value_counts(normalize=True)
    rec_dist = rec["action"].value_counts(normalize=True)
    l1 = float(sum(abs(obs_dist.get(a, 0.0) - rec_dist.get(a, 0.0)) for a in ACTION_MARKDOWNS))
    ep_obs = episode_df.loc[episode_df["policy_id"].eq("selected_observed_financial_ppo")]
    ep_rec = episode_df.loc[episode_df["policy_id"].eq("selected_recovered_financial_ppo")]
    ep = ep_rec.merge(ep_obs, on="episode_id", suffixes=("_recovered", "_observed"))
    if action_diff_share >= 0.25 or l1 >= 0.25 or avg_abs_markdown_diff >= 0.05:
        classification = "MATERIAL_POLICY_CHANGE"
    elif action_diff_share >= 0.10 or l1 >= 0.10 or avg_abs_markdown_diff >= 0.02:
        classification = "MODERATE_POLICY_CHANGE"
    else:
        classification = "LIMITED_POLICY_CHANGE"
    state_group_diff = (
        merged_states.assign(markdown_abs_diff=(merged_states["markdown_observed"] - merged_states["markdown_recovered"]).abs())
        .groupby("fraction_expiring_within_two_days_observed", dropna=False)["markdown_abs_diff"]
        .mean()
        .reset_index()
        .sort_values("markdown_abs_diff", ascending=False)
        .head(5)
    )
    return pd.DataFrame(
        [
            {
                "comparison": "selected_recovered_financial_ppo_vs_selected_observed_financial_ppo",
                "state_decision_difference_share": action_diff_share,
                "average_absolute_markdown_difference": avg_abs_markdown_diff,
                "action_distribution_l1_distance": l1,
                "first_markdown_timing_difference": float(
                    (ep["first_markdown_day_recovered"] - ep["first_markdown_day_observed"]).mean()
                ),
                "episode_level_normalized_profit_difference": float(
                    (ep["normalized_accounting_profit_recovered"] - ep["normalized_accounting_profit_observed"]).mean()
                ),
                "episode_level_waste_difference": float((ep["waste_rate_recovered"] - ep["waste_rate_observed"]).mean()),
                "largest_state_group_differences": state_group_diff.to_json(orient="records"),
                "classification": classification,
                "material_threshold_action_diff_share": 0.25,
                "moderate_threshold_action_diff_share": 0.10,
                "material_threshold_action_l1": 0.25,
                "moderate_threshold_action_l1": 0.10,
            }
        ]
    )


def baseline_value_classification(summary: pd.DataFrame, paired: pd.DataFrame, state_actions: pd.DataFrame) -> pd.DataFrame:
    pivot = summary.loc[summary["metric"].eq("normalized_accounting_profit")].set_index("policy_id")
    baseline_ids = BASELINES
    best_baseline = pivot.loc[baseline_ids]["mean"].idxmax()
    rows: list[dict[str, Any]] = []
    for ppo_id in ["selected_observed_financial_ppo", "selected_recovered_financial_ppo"]:
        cmp_row = paired.loc[
            paired["left_policy"].eq(ppo_id)
            & paired["right_policy"].eq(best_baseline)
            & paired["metric"].eq("normalized_accounting_profit")
        ]
        if cmp_row.empty:
            # Required comparisons do not include always_5/10/20/40, so compute direct fallback.
            label = "REQUIRES_REVIEW"
            mean_diff = float(pivot.loc[ppo_id, "mean"] - pivot.loc[best_baseline, "mean"])
            ci_low = np.nan
            win_share = np.nan
        else:
            row = cmp_row.iloc[0]
            mean_diff = float(row["paired_mean_difference"])
            ci_low = float(row["bootstrap_ci_low"])
            win_share = float(row["win_share"])
            if ci_low > 0 and win_share >= 0.60:
                label = "CLEAR_INCREMENTAL_VALUE"
            elif mean_diff > 0 and win_share >= 0.50:
                label = "MODEST_INCREMENTAL_VALUE"
            elif abs(mean_diff) < 0.01:
                label = "EFFECTIVELY_EQUIVALENT_TO_BEST_BASELINE"
            else:
                label = "WORSE_THAN_BEST_BASELINE"
        rows.append(
            {
                "policy_id": ppo_id,
                "best_baseline": best_baseline,
                "classification": label,
                "normalized_profit_mean_difference_vs_best_baseline": mean_diff,
                "bootstrap_ci_low": ci_low,
                "win_share": win_share,
                "state_dependent_behavior_recorded": True,
            }
        )
    return pd.DataFrame(rows)


def profit_waste_tradeoff(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = ["normalized_accounting_profit", "waste_rate", "sell_through_rate", "average_markdown"]
    wide = summary.loc[summary["metric"].isin(metrics)].pivot(index="policy_id", columns="metric", values="mean").reset_index()
    dominated = []
    for _, row in wide.iterrows():
        is_dominated = False
        for _, other in wide.iterrows():
            if row["policy_id"] == other["policy_id"]:
                continue
            better_or_equal = (
                other["normalized_accounting_profit"] >= row["normalized_accounting_profit"]
                and other["sell_through_rate"] >= row["sell_through_rate"]
                and other["waste_rate"] <= row["waste_rate"]
            )
            strictly_better = (
                other["normalized_accounting_profit"] > row["normalized_accounting_profit"]
                or other["sell_through_rate"] > row["sell_through_rate"]
                or other["waste_rate"] < row["waste_rate"]
            )
            if better_or_equal and strictly_better:
                is_dominated = True
                break
        dominated.append(is_dominated)
    wide["dominated_within_evaluated_set"] = dominated
    wide["description"] = "empirical profit-waste trade-off set"
    return wide


def make_figures(summary: pd.DataFrame, paired: pd.DataFrame, state_actions: pd.DataFrame, comparability: pd.DataFrame, transmission: pd.DataFrame) -> None:
    wide = summary.pivot(index="policy_id", columns="metric", values="mean").reset_index()
    for metric, filename, ylabel in [
        ("normalized_accounting_profit", "normalized_accounting_profit_by_policy.png", "Normalized accounting profit"),
        ("waste_rate", "waste_rate_by_policy.png", "Waste rate"),
        ("sell_through_rate", "sell_through_rate_by_policy.png", "Sell-through rate"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(wide["policy_id"], wide[metric])
        ax.tick_params(axis="x", rotation=75)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / filename, dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(wide["normalized_accounting_profit"], wide["waste_rate"])
    for _, row in wide.iterrows():
        ax.annotate(row["policy_id"], (row["normalized_accounting_profit"], row["waste_rate"]), fontsize=7)
    ax.set_xlabel("Normalized accounting profit")
    ax.set_ylabel("Waste rate")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "profit_versus_waste_scatter.png", dpi=160)
    plt.close(fig)
    profit = paired.loc[paired["metric"].eq("normalized_accounting_profit")]
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = profit["left_policy"] + " vs " + profit["right_policy"]
    y = np.arange(len(profit))
    ax.errorbar(
        profit["paired_mean_difference"],
        y,
        xerr=[profit["paired_mean_difference"] - profit["bootstrap_ci_low"], profit["bootstrap_ci_high"] - profit["paired_mean_difference"]],
        fmt="o",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color="black", linewidth=1)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "paired_profit_difference_confidence_intervals.png", dpi=160)
    plt.close(fig)
    action_cols = [col for col in wide.columns if col.startswith("action_")]
    selected_actions = state_actions.loc[
        state_actions["policy_id"].isin(["selected_observed_financial_ppo", "selected_recovered_financial_ppo"])
        & state_actions["feature"].eq("episode_day")
    ]
    action_share_cols = [col for col in selected_actions.columns if col.startswith("action_") and col.endswith("_share")]
    if not selected_actions.empty and action_share_cols:
        dist = selected_actions.groupby("policy_id")[action_share_cols].mean()
        dist.columns = [col.replace("_share", "") for col in dist.columns]
        fig, ax = plt.subplots(figsize=(8, 4))
        dist.plot(kind="bar", stacked=True, ax=ax)
        ax.set_ylabel("Action share")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "observed_vs_recovered_action_distribution.png", dpi=160)
        plt.close(fig)
    soon = state_actions.loc[state_actions["feature"].eq("fraction_expiring_within_two_days")]
    if not soon.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        for policy, group in soon.groupby("policy_id"):
            ax.plot(group["state_group"], group["mean_markdown"], marker="o", label=policy)
        ax.legend(fontsize=7)
        ax.set_ylabel("Mean markdown")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "markdown_by_fraction_expiring_soon.png", dpi=160)
        plt.close(fig)
    wtl = paired.loc[paired["metric"].eq("normalized_accounting_profit")]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(wtl))
    bottom = np.zeros(len(wtl))
    for col in ["win_share", "tie_share", "loss_share"]:
        ax.bar(x, wtl[col], bottom=bottom, label=col)
        bottom += wtl[col].to_numpy()
    ax.set_xticks(x)
    ax.set_xticklabels(wtl["left_policy"] + " vs " + wtl["right_policy"], rotation=75, ha="right", fontsize=7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "policy_win_tie_loss_comparison.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(comparability["initial_inventory_difference"], bins=12)
    ax.set_xlabel("Recovered - observed initial inventory")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "initial_inventory_comparability.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    row = transmission.iloc[0]
    ax.bar(["decision_diff", "abs_markdown_diff", "action_l1"], [row["state_decision_difference_share"], row["average_absolute_markdown_difference"], row["action_distribution_l1_distance"]])
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration_transmission_summary.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    metadata, hashes = write_selected_model_metadata()
    manifest = load_manifest()
    specs = make_policy_specs()
    episode_tables: list[pd.DataFrame] = []
    state_tables: list[pd.DataFrame] = []
    comparability_tables: list[pd.DataFrame] = []
    for spec in specs:
        print(f"Evaluating {spec.policy_id}")
        episode_df, state_df, comp_df = eval_policy(spec, manifest)
        episode_tables.append(episode_df)
        state_tables.append(state_df)
        comparability_tables.append(comp_df)
    episode_level = pd.concat(episode_tables, ignore_index=True)
    state_level = pd.concat(state_tables, ignore_index=True)
    comparability_raw = pd.concat(comparability_tables, ignore_index=True)
    summary = aggregate_summary(episode_level)
    paired = paired_comparisons(episode_level)
    comparability = profit_comparability(comparability_raw)
    state_actions = binned_state_actions(state_level)
    transmission = calibration_transmission(episode_level, state_level)
    value_classification = baseline_value_classification(summary, paired, state_actions)
    tradeoff = profit_waste_tradeoff(summary)
    episode_level.to_csv(TABLES_DIR / "ppo_financial_paired_validation_episode_level.csv", index=False)
    summary.to_csv(TABLES_DIR / "ppo_financial_paired_validation_summary.csv", index=False)
    paired.to_csv(TABLES_DIR / "ppo_financial_paired_policy_comparisons.csv", index=False)
    comparability.to_csv(TABLES_DIR / "ppo_financial_profit_comparability_audit.csv", index=False)
    state_actions.to_csv(TABLES_DIR / "ppo_financial_state_dependent_actions.csv", index=False)
    transmission.to_csv(TABLES_DIR / "ppo_financial_calibration_transmission.csv", index=False)
    value_classification.to_csv(TABLES_DIR / "ppo_financial_baseline_value_classification.csv", index=False)
    tradeoff.to_csv(TABLES_DIR / "ppo_financial_profit_waste_tradeoff.csv", index=False)
    make_figures(summary, paired, state_actions, comparability, transmission)
    status = "PPO_FINANCIAL_BASELINE_VALIDATION_COMPLETE"
    if value_classification["classification"].eq("EFFECTIVELY_EQUIVALENT_TO_BEST_BASELINE").all():
        status = "PPO_FINANCIAL_MODEL_EQUIVALENT_TO_BASELINE"
    if not comparability["episode_horizon_equal"].all() or episode_level.select_dtypes(include=[np.number]).isna().any().any():
        status = "PPO_FINANCIAL_BASELINE_VALIDATION_REQUIRES_REVISION"
    best_baseline = (
        summary.loc[summary["metric"].eq("normalized_accounting_profit")]
        .set_index("policy_id")
        .loc[BASELINES]["mean"]
        .idxmax()
    )
    norm_profit_diffs = paired.loc[paired["metric"].eq("normalized_accounting_profit")][
        ["left_policy", "right_policy", "paired_mean_difference", "bootstrap_ci_low", "bootstrap_ci_high", "win_share", "tie_share", "loss_share"]
    ].to_dict("records")
    report = {
        "status": status,
        "best_baseline": best_baseline,
        "observed_ppo_classification": value_classification.loc[value_classification["policy_id"].eq("selected_observed_financial_ppo"), "classification"].iloc[0],
        "recovered_ppo_classification": value_classification.loc[value_classification["policy_id"].eq("selected_recovered_financial_ppo"), "classification"].iloc[0],
        "paired_normalized_profit_differences": norm_profit_diffs,
        "profit_comparability_status": comparability["overall_raw_profit_comparability_status"].iloc[0],
        "calibration_transmission_classification": transmission["classification"].iloc[0],
        "waste_rate_comparison": summary.loc[summary["metric"].eq("waste_rate")][["policy_id", "mean"]].to_dict("records"),
        "files_created": [
            "outputs/configs/ppo_selected_financial_models.json",
            "outputs/configs/ppo_selected_financial_model_hashes.json",
            "outputs/tables/ppo_financial_paired_validation_episode_level.csv",
            "outputs/tables/ppo_financial_paired_validation_summary.csv",
            "outputs/tables/ppo_financial_paired_policy_comparisons.csv",
            "outputs/tables/ppo_financial_profit_comparability_audit.csv",
            "outputs/tables/ppo_financial_state_dependent_actions.csv",
            "outputs/tables/ppo_financial_calibration_transmission.csv",
            "outputs/tables/ppo_financial_baseline_value_classification.csv",
            "outputs/tables/ppo_financial_profit_waste_tradeoff.csv",
        ],
        "recommended_next_step": "Write the financial PPO validation results; do not run main training until sustainability reward wiring is audited.",
    }
    (TABLES_DIR / "ppo_financial_paired_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(status)


if __name__ == "__main__":
    main()
