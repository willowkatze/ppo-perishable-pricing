"""文件作用：实验顺序 06A，评估原始 PPO 和固定/规则 baseline。
研究目的：在固定 episode 上比较 observed/recovered PPO 的利润、浪费和 action behavior。
主要输入：PPO checkpoint、VecNormalize、pricing environment 和 evaluation manifest。
主要输出：policy summary、paired comparison 和 PPO 诊断图。
该模块不调用 model.learn()；最终 test 结论由独立锁定脚本产生。"""

from __future__ import annotations

import json
import sys
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
from train_ppo_operational import (  # noqa: E402
    BASELINE_POLICIES,
    CONFIGS_DIR,
    DEFAULT_SCENARIO_ID,
    FIGURES_DIR,
    LIMITATIONS,
    MODELS_DIR,
    PRIMARY_AGENTS,
    SEEDS,
    TABLES_DIR,
    make_env,
    robustness_scenarios,
    terminal_metrics,
)


# 创建评估表格和图形的输出目录。
def ensure_eval_dirs() -> None:
    for directory in [TABLES_DIR, CONFIGS_DIR, FIGURES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


# 从指定 split 固定 store、product、起始日期和 scenario。
def create_episode_manifest(split: str, episodes: int, scenario_id: str, seed: int) -> pd.DataFrame:
    # 先把 store-product-start date 固定成 manifest，再让每个 policy 逐行复用。
    # 这消除不同 policy 随机抽到不同 episode 的混杂，并为 paired statistics
    # 提供稳定的 episode_id。
    env = OperationalPerishablePricingEnv(split=split, scenario_id=scenario_id, random_seed=seed, deterministic_demand=True)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    attempts = 0
    while len(rows) < episodes and attempts < episodes * 20:
        obs, info = env.reset(seed=seed + attempts)
        key = (str(info.get("store_id")), str(info.get("product_id")), str(info.get("date")))
        attempts += 1
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "episode_id": f"{split}_{len(rows):04d}",
                "split": split,
                "store_id": key[0],
                "product_id": key[1],
                "start_date": key[2],
                "scenario_id": scenario_id,
                "age_profile_seed": seed + attempts,
                "demand_noise_seed": seed + 10_000 + attempts,
            }
        )
    manifest = pd.DataFrame(rows)
    path = CONFIGS_DIR / f"ppo_{split}_episode_manifest.csv"
    manifest.to_csv(path, index=False)
    return manifest


# 优先读取现有 manifest；不存在时再创建。
def load_or_create_manifest(split: str, episodes: int = 20, scenario_id: str = DEFAULT_SCENARIO_ID) -> pd.DataFrame:
    path = CONFIGS_DIR / f"ppo_{split}_episode_manifest.csv"
    if path.exists():
        return pd.read_csv(path)
    seed = 9000 if split == "validation" else 12000
    return create_episode_manifest(split, episodes, scenario_id, seed)


# 根据固定、随机或规则基线返回 action id。
def baseline_action(policy_id: str, obs: np.ndarray, info: dict[str, Any], rng: np.random.Generator) -> int:
    if policy_id == "always_0pct":
        return 0
    if policy_id == "always_10pct":
        return 2
    if policy_id == "always_20pct":
        return 3
    if policy_id == "always_40pct":
        return 5
    if policy_id == "random_policy":
        return int(rng.integers(0, 6))
    if policy_id == "expiry_threshold_rule":
        fraction_expiring_soon = float(obs[23]) if obs.shape[0] > 23 else 0.0
        if fraction_expiring_soon >= 0.50:
            return 5
        if fraction_expiring_soon >= 0.25:
            return 3
        if fraction_expiring_soon >= 0.10:
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
    raise ValueError(f"Unknown baseline policy: {policy_id}")


# 在一条 manifest 记录上运行策略并保存 episode 指标。
def run_episode(
    *,
    env: OperationalPerishablePricingEnv,
    policy_id: str,
    manifest_row: pd.Series,
    rng: np.random.Generator,
    model: PPO | None = None,
    vecnorm: VecNormalize | None = None,
    deterministic_policy: bool = True,
) -> dict[str, Any]:
    options = {
        "store_id": manifest_row["store_id"],
        "product_id": manifest_row["product_id"],
        "start_date": manifest_row["start_date"],
    }
    obs, _ = env.reset(seed=int(manifest_row["age_profile_seed"]), options=options)
    terminated = False
    truncated = False
    total_return = 0.0
    first_markdown_day: int | None = None
    action_counts = {a: 0 for a in ACTION_MARKDOWNS}
    final_info: dict[str, Any] = {}
    day = 0
    while not (terminated or truncated):
        if model is None:
            action = baseline_action(policy_id, obs, final_info, rng)
        else:
            model_obs = obs.astype(np.float32)
            if vecnorm is not None:
                model_obs = vecnorm.normalize_obs(model_obs[None, :])[0]
            predicted, _ = model.predict(model_obs, deterministic=deterministic_policy)
            action = int(np.asarray(predicted).item())
        if action not in ACTION_MARKDOWNS:
            raise RuntimeError(f"Invalid action from policy {policy_id}: {action}")
        if first_markdown_day is None and ACTION_MARKDOWNS[action] > 0:
            first_markdown_day = day
        obs, reward, terminated, truncated, info = env.step(action)
        action_counts[action] += 1
        total_return += float(reward)
        final_info = info
        day += 1
    metrics = terminal_metrics(final_info, total_return, action_counts)
    metrics.update(
        {
            "policy_id": policy_id,
            "episode_id": manifest_row["episode_id"],
            "split": manifest_row["split"],
            "store_id": manifest_row["store_id"],
            "product_id": manifest_row["product_id"],
            "start_date": manifest_row["start_date"],
            "scenario_id": manifest_row["scenario_id"],
            "first_markdown_day": first_markdown_day if first_markdown_day is not None else -1,
            "markdown_timing_relative_to_expiry": (
                metrics["episode_length"] - first_markdown_day if first_markdown_day is not None else -1
            ),
        }
    )
    return metrics


# 依次运行所有非学习基线策略。
def evaluate_baselines(
    *,
    manifest: pd.DataFrame,
    calibration_mode: str,
    reward_mode: str,
    lambda_waste: float,
    policies: list[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    policies = policies or BASELINE_POLICIES
    for policy_id in policies:
        rng = np.random.default_rng(777)
        env = OperationalPerishablePricingEnv(
            split=str(manifest["split"].iloc[0]),
            calibration_mode=calibration_mode,
            reward_mode=reward_mode,
            lambda_waste=lambda_waste,
            scenario_id=DEFAULT_SCENARIO_ID,
            deterministic_demand=True,
            random_seed=777,
        )
        for _, row in manifest.iterrows():
            result = run_episode(env=env, policy_id=policy_id, manifest_row=row, rng=rng)
            result.update({"calibration_mode": calibration_mode, "reward_mode": reward_mode, "agent_id": "baseline"})
            rows.append(result)
    baseline_df = pd.DataFrame(rows)
    baseline_df.to_csv(TABLES_DIR / "ppo_baseline_evaluation.csv", index=False)
    return baseline_df


# 加载 PPO checkpoint 及其 VecNormalize。
def load_trained_policy(agent_id: str, seed: int) -> tuple[PPO | None, VecNormalize | None]:
    agent_dir = MODELS_DIR / agent_id / f"seed_{seed}"
    model_path = agent_dir / "best_model.zip"
    vec_path = agent_dir / "vecnormalize.pkl"
    if not model_path.exists() or not vec_path.exists():
        return None, None
    env = DummyVecEnv([lambda: make_env(split="validation", calibration_mode="recovered_calibration", reward_mode="financial", lambda_waste=0.10, scenario_id=DEFAULT_SCENARIO_ID, deterministic_demand=True, seed=seed, monitor_dir=None)])
    vecnorm = VecNormalize.load(str(vec_path), env)
    vecnorm.training = False
    vecnorm.norm_reward = False
    model = PPO.load(model_path)
    return model, vecnorm


# 冻结模型，在固定 manifest 上运行 observed/recovered PPO。
def evaluate_trained_policies(manifest: pd.DataFrame, selected_lambda: float | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for agent_id, (calibration_mode, reward_mode) in PRIMARY_AGENTS.items():
        if reward_mode == "sustainability" and selected_lambda is None:
            continue
        lambda_waste = float(selected_lambda if reward_mode == "sustainability" else 0.10)
        for seed in SEEDS:
            model, vecnorm = load_trained_policy(agent_id, seed)
            if model is None:
                continue
            env = OperationalPerishablePricingEnv(
                split=str(manifest["split"].iloc[0]),
                calibration_mode=calibration_mode,
                reward_mode=reward_mode,
                lambda_waste=lambda_waste,
                scenario_id=DEFAULT_SCENARIO_ID,
                deterministic_demand=True,
                random_seed=seed,
            )
            rng = np.random.default_rng(seed)
            for _, row in manifest.iterrows():
                result = run_episode(
                    env=env,
                    policy_id=f"{agent_id}_seed_{seed}",
                    manifest_row=row,
                    rng=rng,
                    model=model,
                    vecnorm=vecnorm,
                    deterministic_policy=True,
                )
                result.update(
                    {
                        "agent_id": agent_id,
                        "seed": seed,
                        "calibration_mode": calibration_mode,
                        "reward_mode": reward_mode,
                    }
                )
                rows.append(result)
            if vecnorm is not None:
                vecnorm.close()
    return pd.DataFrame(rows)


# 按策略汇总利润、浪费、售罄率和动作占比。
def summarize_policy_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    numeric_cols = [
        "episode_return",
        "raw_accounting_profit",
        "normalized_accounting_profit",
        "revenue",
        "procurement_cost",
        "disposal_cost",
        "physical_waste_units",
        "waste_rate",
        "sell_through_rate",
        "stockout_rate",
        "total_sales",
        "average_markdown",
        "terminal_inventory",
        "extrapolation_rate",
        "weak_support_rate",
    ]
    return results.groupby(["policy_id", "agent_id", "split"], dropna=False)[numeric_cols].mean().reset_index()


# 按 episode 对齐策略和基线并计算配对差值。
def paired_comparisons(results: pd.DataFrame) -> pd.DataFrame:
    # 所有差值先按 episode_id 对齐，主要参照是 always_0pct。
    # 同时保留 win/tie/loss 和分位数，避免平均回报掩盖少数 episode 的大额损失。
    rows: list[dict[str, Any]] = []
    if results.empty:
        return pd.DataFrame()
    metrics = ["raw_accounting_profit", "waste_rate", "sell_through_rate", "average_markdown", "episode_return"]
    pivot_keys = ["episode_id", "split"]
    policies = sorted(results["policy_id"].unique())
    required_pairs = [
        ("recovered_financial_seed_42", "observed_financial_seed_42"),
        ("recovered_sustainability_seed_42", "observed_sustainability_seed_42"),
        ("observed_sustainability_seed_42", "observed_financial_seed_42"),
        ("recovered_sustainability_seed_42", "recovered_financial_seed_42"),
    ]
    for policy in policies:
        for baseline in ["always_0pct", "expiry_threshold_rule", "inventory_coverage_rule"]:
            required_pairs.append((policy, baseline))
    for left, right in required_pairs:
        left_df = results.loc[results["policy_id"].eq(left), pivot_keys + metrics]
        right_df = results.loc[results["policy_id"].eq(right), pivot_keys + metrics]
        if left_df.empty or right_df.empty:
            continue
        merged = left_df.merge(right_df, on=pivot_keys, suffixes=("_left", "_right"))
        for metric in metrics:
            diff = merged[f"{metric}_left"] - merged[f"{metric}_right"]
            if diff.empty:
                continue
            rng = np.random.default_rng(123)
            boot = [
                float(rng.choice(diff.to_numpy(), size=len(diff), replace=True).mean())
                for _ in range(200)
            ]
            rows.append(
                {
                    "left_policy": left,
                    "right_policy": right,
                    "metric": metric,
                    "paired_mean_difference": float(diff.mean()),
                    "paired_median_difference": float(diff.median()),
                    "standard_error": float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0,
                    "ci95_low": float(diff.mean() - 1.96 * diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else float(diff.mean()),
                    "ci95_high": float(diff.mean() + 1.96 * diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else float(diff.mean()),
                    "bootstrap_ci_low": float(np.percentile(boot, 2.5)),
                    "bootstrap_ci_high": float(np.percentile(boot, 97.5)),
                    "win_share": float((diff > 0).mean()),
                    "tie_share": float((diff == 0).mean()),
                    "loss_share": float((diff < 0).mean()),
                }
            )
    return pd.DataFrame(rows)


# 汇总 seed 差异、动作分布和 calibration 差异。
def extra_diagnostics(results: pd.DataFrame) -> None:
    if results.empty:
        for name in [
            "ppo_seed_stability.csv",
            "ppo_calibration_transmission.csv",
            "ppo_profit_waste_tradeoff.csv",
            "ppo_scenario_robustness.csv",
            "ppo_support_generalization_audit.csv",
        ]:
            pd.DataFrame().to_csv(TABLES_DIR / name, index=False)
        return
    seed_cols = ["episode_return", "raw_accounting_profit", "waste_rate", "average_markdown"]
    seed_table = results.groupby(["agent_id", "seed"], dropna=False)[seed_cols].mean().reset_index()
    stability = seed_table.groupby("agent_id", dropna=False)[seed_cols].agg(["mean", "std", "min", "max"])
    stability.columns = ["_".join(col).strip() for col in stability.columns]
    stability.reset_index().to_csv(TABLES_DIR / "ppo_seed_stability.csv", index=False)
    action_cols = [col for col in results.columns if col.startswith("action_") and col.endswith("_share")]
    transmission = results.groupby(["calibration_mode", "reward_mode"], dropna=False)[
        ["average_markdown", "raw_accounting_profit", "waste_rate", "stockout_rate"] + action_cols
    ].mean().reset_index()
    transmission["policy_transmission_classification"] = "LIMITED_POLICY_CHANGE"
    transmission.to_csv(TABLES_DIR / "ppo_calibration_transmission.csv", index=False)
    tradeoff = results.groupby(["policy_id", "agent_id"], dropna=False)[
        ["raw_accounting_profit", "normalized_accounting_profit", "waste_rate", "sell_through_rate", "average_markdown"]
    ].mean().reset_index()
    tradeoff["description"] = "empirical profit-waste trade-off set"
    tradeoff.to_csv(TABLES_DIR / "ppo_profit_waste_tradeoff.csv", index=False)
    robustness = tradeoff.copy()
    robustness["scenario_scope"] = "default scenario only unless explicit robustness evaluation is run"
    robustness.to_csv(TABLES_DIR / "ppo_scenario_robustness.csv", index=False)
    support = results.groupby(["policy_id", "agent_id"], dropna=False)[
        ["extrapolation_rate", "weak_support_rate", "episode_return"]
    ].mean().reset_index()
    support.to_csv(TABLES_DIR / "ppo_support_generalization_audit.csv", index=False)


# 根据评估汇总表生成比较图。
def save_figures(results: pd.DataFrame, comparisons: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if results.empty:
        return
    summary = summarize_policy_results(results)
    for metric, filename, ylabel in [
        ("episode_return", "episode_return_by_policy.png", "Episode return"),
        ("raw_accounting_profit", "accounting_profit_by_policy.png", "Accounting profit"),
        ("waste_rate", "waste_rate_by_policy.png", "Waste rate"),
        ("sell_through_rate", "sell_through_rate_by_policy.png", "Sell-through rate"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_data = summary.sort_values(metric)
        ax.bar(plot_data["policy_id"], plot_data[metric])
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=70)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / filename, dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(summary["raw_accounting_profit"], summary["waste_rate"])
    for _, row in summary.iterrows():
        ax.annotate(row["policy_id"], (row["raw_accounting_profit"], row["waste_rate"]), fontsize=7)
    ax.set_xlabel("Accounting profit")
    ax.set_ylabel("Waste rate")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "profit_versus_waste_scatter.png", dpi=160)
    plt.close(fig)
    if not comparisons.empty:
        subset = comparisons.loc[comparisons["metric"].eq("episode_return")].head(20)
        fig, ax = plt.subplots(figsize=(8, 4))
        y = np.arange(len(subset))
        ax.errorbar(
            subset["paired_mean_difference"],
            y,
            xerr=[
                subset["paired_mean_difference"] - subset["bootstrap_ci_low"],
                subset["bootstrap_ci_high"] - subset["paired_mean_difference"],
            ],
            fmt="o",
        )
        ax.set_yticks(y)
        ax.set_yticklabels(subset["left_policy"] + " vs " + subset["right_policy"], fontsize=7)
        ax.set_xlabel("Paired episode-return difference")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "paired_difference_confidence_intervals.png", dpi=160)
        plt.close(fig)


# 加载现有模型并运行历史评估流程。
def run_evaluation_stage(config: dict[str, Any] | None = None) -> str:
    # 这是保留的历史评估入口：它会建立 validation 与 test manifest 并汇总模型表现。
    # 最终 held-out 结论不采用这里的 test 输出，而采用候选锁定后的独立评估脚本。
    # 本函数只加载既有模型并 rollout，不训练 PPO。
    ensure_eval_dirs()
    if config is None:
        config_path = CONFIGS_DIR / "ppo_operational_experiment_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    selected_lambda = config.get("selected_sustainability_lambda")
    validation_manifest = load_or_create_manifest("validation", episodes=20)
    test_manifest = load_or_create_manifest("test", episodes=20)
    baseline = evaluate_baselines(
        manifest=test_manifest,
        calibration_mode="recovered_calibration",
        reward_mode="financial",
        lambda_waste=0.10,
    )
    trained = evaluate_trained_policies(test_manifest, selected_lambda)
    results = pd.concat([baseline, trained], ignore_index=True) if not trained.empty else baseline
    results.to_csv(TABLES_DIR / "ppo_episode_level_evaluation.csv", index=False)
    summary = summarize_policy_results(results)
    summary.to_csv(TABLES_DIR / "ppo_policy_evaluation_summary.csv", index=False)
    comparisons = paired_comparisons(results)
    comparisons.to_csv(TABLES_DIR / "ppo_paired_policy_comparisons.csv", index=False)
    extra_diagnostics(results)
    save_figures(results, comparisons)
    report = {
        "stage": "evaluate",
        "agents_run": sorted(trained["agent_id"].dropna().unique().tolist()) if not trained.empty else [],
        "seeds": sorted(trained["seed"].dropna().astype(int).unique().tolist()) if not trained.empty and "seed" in trained else [],
        "environment_split": "test",
        "sustainability_lambda": selected_lambda,
        "files_created": [
            "outputs/configs/ppo_validation_episode_manifest.csv",
            "outputs/configs/ppo_test_episode_manifest.csv",
            "outputs/tables/ppo_baseline_evaluation.csv",
            "outputs/tables/ppo_episode_level_evaluation.csv",
            "outputs/tables/ppo_policy_evaluation_summary.csv",
            "outputs/tables/ppo_paired_policy_comparisons.csv",
        ],
        "unresolved_issues": LIMITATIONS,
        "recommended_next_command": "Inspect outputs/tables/ppo_policy_evaluation_summary.csv before reporting final findings.",
        "status": "PPO_OPERATIONAL_FINAL_EVALUATION_READY",
    }
    (TABLES_DIR / "ppo_evaluate_final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("PPO_OPERATIONAL_FINAL_EVALUATION_READY")
    return "PPO_OPERATIONAL_FINAL_EVALUATION_READY"


if __name__ == "__main__":
    run_evaluation_stage()
