"""Build final report figures from existing output tables.

This module is reporting-only: it reads selected intermediate outputs and
creates publication figures without retraining models or changing metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
TABLES = OUTPUTS / "tables"
CONFIGS = OUTPUTS / "configs"
MODELS = OUTPUTS / "models" / "ppo_operational"
FIGURES = OUTPUTS / "figures" / "final_report"
APPENDIX = FIGURES / "appendix_training_diagnostics"
DOCS = PROJECT_ROOT / "docs"

ACTION_LABELS = {
    "action_0_share": "0%",
    "action_1_share": "5%",
    "action_2_share": "10%",
    "action_3_share": "20%",
    "action_4_share": "30%",
    "action_5_share": "40%",
}
ACTION_MARKDOWNS = {0: 0.00, 1: 0.05, 2: 0.10, 3: 0.20, 4: 0.30, 5: 0.40}
ACTION_COLS = list(ACTION_LABELS)
ACTION_ORDER = list(ACTION_LABELS.values())
SELECTED_STEP = 22288
COLLAPSE_THRESHOLD = 0.99


def ensure_dirs() -> None:
    for path in [TABLES, FIGURES, APPENDIX, DOCS]:
        path.mkdir(parents=True, exist_ok=True)


def read_csv(name: str) -> pd.DataFrame:
    path = TABLES / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def savefig(name: str) -> str:
    path = FIGURES / name
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return rel(path)


def savefig_appendix(name: str) -> str:
    path = APPENDIX / name
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return rel(path)


def artifact_inventory() -> pd.DataFrame:
    patterns = [
        "outputs/tables/ppo_*.csv",
        "outputs/tables/ppo_*.json",
        "outputs/tables/financial_markdown_*.csv",
        "outputs/tables/financial_markdown_*.json",
        "outputs/tables/sustainability_*.csv",
        "outputs/tables/sustainability_*.json",
        "outputs/configs/ppo_*.json",
        "outputs/configs/ppo_*.csv",
        "outputs/models/ppo_operational/**/*.csv",
        "outputs/models/ppo_operational/**/*.json",
        "outputs/models/ppo_operational/**/*.npz",
        "outputs/models/ppo_operational/**/events.*",
    ]
    rows: list[dict[str, Any]] = []
    for pattern in patterns:
        for path in PROJECT_ROOT.glob(pattern):
            if not path.is_file():
                continue
            text = str(path).lower()
            agent = "observed_financial" if "observed_financial" in text else "recovered_financial" if "recovered_financial" in text else ""
            seed = ""
            for part in path.parts:
                if part.startswith("seed_"):
                    seed = part.replace("seed_", "")
            artifact_type = "table"
            if "monitor" in path.name:
                artifact_type = "monitor_log"
            elif path.suffix == ".npz":
                artifact_type = "evaluation_npz"
            elif "events." in path.name:
                artifact_type = "tensorboard_log"
            elif path.suffix == ".json":
                artifact_type = "json_metadata"
            intended = []
            if "checkpoint" in text:
                intended.append("training curve/action distribution/checkpoint selection")
            if "paired" in text or "baseline" in text:
                intended.append("PPO vs baseline")
            if "financial_markdown" in text:
                intended.append("oracle opportunity/PPO regret")
            if "sustainability" in text:
                intended.append("profit-waste tradeoff")
            if "training_curve" in text or "learning_curve" in text:
                intended.append("training diagnostics")
            rows.append(
                {
                    "artifact_path": rel(path),
                    "artifact_type": artifact_type,
                    "agent": agent,
                    "seed": seed,
                    "timestep_coverage": "available in file" if "checkpoint" in text or "training" in text or "learning" in text else "",
                    "intended_figure": "; ".join(intended) if intended else "supporting artifact",
                    "usable": True,
                    "missing_fields": "",
                    "notes": f"{path.stat().st_size} bytes",
                }
            )
    inventory = pd.DataFrame(rows).drop_duplicates("artifact_path").sort_values("artifact_path")
    inventory.to_csv(TABLES / "final_figure_artifact_inventory.csv", index=False)
    return inventory


def consolidate_training_curve() -> pd.DataFrame:
    ckpt = read_csv("ppo_checkpoint_level_evaluation.csv")
    action_diag = read_csv("ppo_checkpoint_action_diagnostics.csv")
    selected = read_csv("ppo_checkpoint_model_selection.csv")
    trainer = read_csv("ppo_training_curves.csv")
    rows: list[dict[str, Any]] = []
    if not ckpt.empty:
        agg_cols = ["episode_return", "normalized_accounting_profit", "raw_accounting_profit"] + ACTION_COLS
        grouped = ckpt.loc[ckpt["kind"].eq("ppo")].groupby(["agent_id", "checkpoint_step"], as_index=False)[agg_cols].mean()
        diag = action_diag.loc[action_diag["kind"].eq("ppo")][["policy_id", "agent_id", "checkpoint_step", "classification", "action_entropy", "one_action_episode"] + ACTION_COLS]
        grouped = grouped.merge(diag, on=["agent_id", "checkpoint_step"], how="left", suffixes=("", "_diag"))
        selected_steps = dict(zip(selected["agent_id"], selected["selected_policy_id"])) if not selected.empty else {}
        for _, row in grouped.iterrows():
            agent = row["agent_id"]
            step = int(row["checkpoint_step"])
            policy_id = f"{agent}__checkpoint_{step}"
            rows.append(
                {
                    "agent": agent,
                    "seed": 42,
                    "timestep": step,
                    "training_episode_return": np.nan,
                    "smoothed_training_episode_return": np.nan,
                    "validation_return": row["episode_return"],
                    "normalized_accounting_profit": row["normalized_accounting_profit"],
                    "raw_accounting_profit": row["raw_accounting_profit"],
                    "empirical_action_entropy": row.get("action_entropy", np.nan),
                    "policy_entropy": np.nan,
                    "policy_gradient_loss": np.nan,
                    "value_loss": np.nan,
                    "entropy_loss": np.nan,
                    "approx_kl": np.nan,
                    "clip_fraction": np.nan,
                    "explained_variance": np.nan,
                    "collapse_flag": bool(row.get("action_0_share", 0) >= COLLAPSE_THRESHOLD),
                    "selected_checkpoint_flag": selected_steps.get(agent, "") == policy_id,
                    **{ACTION_LABELS[col]: row.get(col, np.nan) for col in ACTION_COLS},
                }
            )
    curve = pd.DataFrame(rows)
    if not trainer.empty and not curve.empty:
        trainer_renamed = trainer.rename(
            columns={
                "agent_id": "agent",
                "policy_loss": "policy_gradient_loss",
            }
        )
        for _, tr in trainer_renamed.iterrows():
            mask = curve["agent"].eq(tr["agent"]) & curve["timestep"].eq(tr["timestep"])
            if mask.any():
                for col in ["policy_gradient_loss", "value_loss", "entropy_loss", "approx_kl", "clip_fraction", "explained_variance"]:
                    curve.loc[mask, col] = tr.get(col, np.nan)
    if not curve.empty:
        curve = curve.sort_values(["agent", "timestep"])
        curve["smoothed_validation_return"] = curve.groupby("agent")["validation_return"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    curve.to_csv(TABLES / "final_ppo_training_curve_data.csv", index=False)
    return curve


def plot_validation_curve(curve: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 5))
    for agent, group in curve.groupby("agent"):
        label = "Observed PPO" if agent == "observed_financial" else "Recovered PPO"
        ax.plot(group["timestep"], group["validation_return"], marker="o", label=label)
        selected = group.loc[group["selected_checkpoint_flag"]]
        ax.scatter(selected["timestep"], selected["validation_return"], s=120, marker="*", zorder=5)
        collapsed = group.loc[group["collapse_flag"]]
        if not collapsed.empty:
            ax.scatter(collapsed["timestep"], collapsed["validation_return"], s=80, marker="x", color="red", label=f"{label} collapse")
        final = group.sort_values("timestep").tail(1)
        ax.scatter(final["timestep"], final["validation_return"], s=70, marker="s")
    ax.axvline(SELECTED_STEP, color="black", linestyle="--", linewidth=1, label="Selected checkpoint 22288")
    ax.set_title("PPO Validation Return Over Training\nModel selection used validation only")
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Mean validation return")
    ax.legend(fontsize=8)
    return savefig("figure_ppo_validation_return_over_training.png")


def plot_action_distribution(curve: pd.DataFrame, agent: str, filename: str) -> str:
    group = curve.loc[curve["agent"].eq(agent)].sort_values("timestep")
    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(group))
    x = np.arange(len(group))
    for label in ACTION_ORDER:
        vals = group[label].fillna(0).to_numpy()
        ax.bar(x, vals, bottom=bottom, label=label)
        bottom += vals
    tick_labels = [str(int(t)) for t in group["timestep"]]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Deterministic validation action share")
    ax.set_xlabel("Checkpoint timestep")
    title_agent = "Observed" if agent == "observed_financial" else "Recovered"
    ax.set_title(f"{title_agent} PPO Action Distribution Over Training")
    if SELECTED_STEP in set(group["timestep"]):
        idx = list(group["timestep"]).index(SELECTED_STEP)
        ax.axvline(idx, color="black", linestyle="--", linewidth=1)
        ax.text(idx, 1.02, "selected", ha="center", fontsize=8)
    collapsed = group.index[group["collapse_flag"]].tolist()
    for idx_value in collapsed:
        idx = list(group.index).index(idx_value)
        ax.text(idx, 0.95, "collapse", rotation=90, color="red", fontsize=8, ha="center")
    ax.legend(title="Markdown", ncol=3, fontsize=8)
    return savefig(filename)


def plot_entropy(curve: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 5))
    for agent, group in curve.groupby("agent"):
        label = "Observed empirical deterministic action entropy" if agent == "observed_financial" else "Recovered empirical deterministic action entropy"
        ax.plot(group["timestep"], group["empirical_action_entropy"], marker="o", label=label)
        selected = group.loc[group["selected_checkpoint_flag"]]
        ax.scatter(selected["timestep"], selected["empirical_action_entropy"], s=120, marker="*")
        collapsed = group.loc[group["collapse_flag"]]
        if not collapsed.empty:
            ax.scatter(collapsed["timestep"], collapsed["empirical_action_entropy"], s=90, marker="x", color="red")
    ax.axvline(SELECTED_STEP, color="black", linestyle="--", linewidth=1)
    ax.set_title("Empirical Action Entropy and Collapse")
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Empirical deterministic action entropy")
    ax.legend(fontsize=8)
    return savefig("figure_policy_entropy_and_collapse.png")


def plot_ppo_vs_baselines() -> str:
    summary = read_csv("ppo_financial_paired_validation_summary.csv")
    comparisons = read_csv("ppo_financial_paired_policy_comparisons.csv")
    policies = [
        "always_0pct",
        "selected_observed_financial_ppo",
        "selected_recovered_financial_ppo",
        "expiry_threshold_rule",
        "inventory_coverage_rule",
    ]
    data = summary.loc[summary["metric"].eq("normalized_accounting_profit") & summary["policy_id"].isin(policies)].copy()
    data["policy_id"] = pd.Categorical(data["policy_id"], policies, ordered=True)
    data = data.sort_values("policy_id")
    yerr = np.vstack([data["mean"] - data["ci95_low"], data["ci95_high"] - data["mean"]])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(data["policy_id"].astype(str), data["mean"], yerr=yerr, capsize=4)
    ax.set_ylabel("Mean normalized accounting profit")
    ax.set_title("PPO vs Baselines: Paired Validation Profit\nRaw profit not used as primary comparison")
    ax.tick_params(axis="x", rotation=30)
    ax.text(1.5, data["mean"].max() * 0.95, "Recovered PPO > observed PPO\nBoth PPO < always_0pct", ha="center", fontsize=9)
    for _, row in comparisons.loc[
        comparisons["metric"].eq("normalized_accounting_profit")
        & comparisons["left_policy"].isin(["selected_observed_financial_ppo", "selected_recovered_financial_ppo"])
        & comparisons["right_policy"].eq("always_0pct")
    ].iterrows():
        ax.text(0.05, 0.05 + 0.06 * (row["left_policy"].startswith("selected_recovered")), f"{row['left_policy'].replace('selected_', '').replace('_financial_ppo','')}: diff {row['paired_mean_difference']:.4f}", transform=ax.transAxes, fontsize=8)
    return savefig("figure_ppo_vs_baselines_normalized_profit.png")


def plot_selected_policy_distribution() -> str:
    selection = read_csv("ppo_checkpoint_model_selection.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    rows = []
    for _, row in selection.iterrows():
        rows.append({"agent": row["agent_id"], "0%": row["zero_action_share"], "nonzero": 1 - row["zero_action_share"]})
    data = pd.DataFrame(rows).set_index("agent")
    data.plot(kind="bar", stacked=True, ax=ax, color=["tab:blue", "tab:orange"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Action share")
    ax.set_title("Selected Observed vs Recovered PPO Policies\nMATERIAL_POLICY_CHANGE; recovered advantage +0.0196")
    ax.text(0, 0.48, "45.0% zero", ha="center")
    ax.text(1, 0.82, "79.2% zero", ha="center")
    return savefig("figure_observed_vs_recovered_selected_policy.png")


def plot_opportunity_map() -> str:
    oracle = read_csv("financial_markdown_oracle_diagnostic.csv")
    rec = oracle.loc[oracle["calibration_mode"].eq("recovered_calibration")]
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        rec["inventory_coverage_state"],
        rec["fraction_expiring_within_two_days"],
        c=rec["rolling_oracle_preferred_markdown"],
        cmap="viridis",
        s=70,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_xlabel("Inventory coverage state")
    ax.set_ylabel("Fraction expiring within two days")
    ax.set_title("Financial Markdown Opportunity Map\n83.33% zero preferred; 16.67% positive markdown")
    plt.colorbar(scatter, ax=ax, label="Oracle-preferred markdown")
    return savefig("figure_financial_markdown_opportunity_map.png")


def plot_ppo_oracle_errors() -> str:
    ppo = read_csv("financial_markdown_ppo_regret.csv")
    rows = []
    for policy, group in ppo.groupby("ppo_policy_id"):
        categories = {
            "correct zero": ((group["ppo_action"] == 0) & (group["oracle_action"] == 0)).mean(),
            "correct positive": ((group["ppo_action"] != 0) & (group["ppo_action"] == group["oracle_action"])).mean(),
            "unnecessary markdown": group["unnecessary_markdown_action"].astype(bool).mean(),
            "missed markdown": group["missed_positive_markdown_opportunity"].astype(bool).mean(),
            "wrong positive level": ((group["ppo_action"] != 0) & (group["oracle_action"] != 0) & (group["ppo_action"] != group["oracle_action"])).mean(),
        }
        rows.append({"policy": policy, **categories})
    data = pd.DataFrame(rows).set_index("policy")
    fig, ax = plt.subplots(figsize=(9, 5))
    data.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of audited policy-state decisions")
    ax.set_title("PPO vs Oracle Error Decomposition\nMain error: unnecessary markdown")
    ax.text(0.5, 0.98, "Agreement: observed 0.1667, recovered 0.1250\nRegret: observed 0.0375, recovered 0.0257", transform=ax.transAxes, ha="center", va="top", fontsize=9)
    return savefig("figure_ppo_vs_oracle_error_decomposition.png")


def plot_oracle_action_values() -> str:
    values = read_csv("financial_markdown_multistep_action_values.csv")
    oracle = read_csv("financial_markdown_oracle_diagnostic.csv")
    greedy = values.loc[values["continuation_policy"].eq("greedy_financial")]
    state_ids = []
    for action in [0, 4, 5]:
        subset = oracle.loc[oracle["rolling_oracle_preferred_action"].eq(action)]
        if not subset.empty:
            state_ids.append(subset.iloc[0]["state_id"])
    if not state_ids:
        state_ids = oracle.head(3)["state_id"].tolist()
    fig, axes = plt.subplots(1, len(state_ids), figsize=(5 * len(state_ids), 4), sharey=True)
    if len(state_ids) == 1:
        axes = [axes]
    for ax, state_id in zip(axes, state_ids):
        group = greedy.loc[greedy["state_id"].eq(state_id)].sort_values("first_action")
        zero = float(group.loc[group["first_action"].eq(0), "total_normalized_accounting_profit"].iloc[0])
        ax.bar([ACTION_MARKDOWNS[a] for a in group["first_action"]], group["total_normalized_accounting_profit"] - zero, width=0.035)
        best = group.sort_values("total_normalized_accounting_profit", ascending=False).iloc[0]
        ax.set_title(f"{state_id}: best {ACTION_MARKDOWNS[int(best['first_action'])]:.0%}")
        ax.set_xlabel("First markdown")
        ax.axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Value gap vs 0%")
    fig.suptitle("Oracle Action Values by Representative State")
    return savefig("figure_oracle_action_values_by_state.png")


def plot_profit_waste_tradeoff() -> str:
    tradeoff = read_csv("sustainability_tradeoff_vs_always_zero.csv")
    rec = tradeoff.loc[tradeoff["calibration_mode"].eq("recovered_calibration")]
    fig, ax = plt.subplots(figsize=(9, 5))
    for policy, group in rec.groupby("policy_id"):
        marker = "*" if policy == "always_0pct" else "D" if policy == "always_5pct" else "o"
        size = 130 if policy in {"always_0pct", "always_5pct"} else 45
        ax.scatter(group["waste_rate"], group["normalized_accounting_profit"], label=policy, marker=marker, s=size)
    ax.set_xlabel("Waste rate")
    ax.set_ylabel("Normalized accounting profit")
    ax.set_title("Empirical Profit-Waste Trade-off\nFinite evaluated policy set, not a complete Pareto frontier")
    ax.text(0.02, 0.05, "Max waste reduction 0.2703\nProfit sacrifice 0.0646", transform=ax.transAxes, fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    return savefig("figure_empirical_profit_waste_tradeoff.png")


def plot_waste_efficiency() -> str:
    eff = read_csv("sustainability_waste_reduction_efficiency.csv")
    rec = eff.loc[eff["calibration_mode"].eq("recovered_calibration")].copy()
    policy = rec.groupby("policy_id", as_index=False).agg(
        waste_rate_reduction=("waste_rate_reduction", "mean"),
        normalized_profit_sacrifice=("normalized_profit_sacrifice", "mean"),
        waste_reduction_efficiency=("waste_reduction_efficiency", "mean"),
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["tab:green" if p == "always_5pct" else "tab:blue" for p in policy["policy_id"]]
    ax.scatter(policy["normalized_profit_sacrifice"], policy["waste_rate_reduction"], s=120, c=colors)
    for _, row in policy.iterrows():
        ax.text(row["normalized_profit_sacrifice"], row["waste_rate_reduction"], row["policy_id"], fontsize=8)
    ax.set_xlabel("Normalized-profit sacrifice vs always-zero")
    ax.set_ylabel("Waste-rate reduction vs always-zero")
    ax.set_title("Waste-Reduction Efficiency\nalways_5pct highlighted as most efficient")
    return savefig("figure_waste_reduction_efficiency.png")


def plot_appendix_diagnostics(curve: pd.DataFrame) -> list[str]:
    files = []
    metrics = {
        "explained_variance": "Explained Variance",
        "value_loss": "Value Loss",
        "policy_gradient_loss": "Policy-Gradient Loss",
        "approx_kl": "Approximate KL",
        "clip_fraction": "Clip Fraction",
        "entropy_loss": "Entropy Loss",
    }
    for metric, title in metrics.items():
        data = curve.loc[curve[metric].notna()]
        if data.groupby("agent")[metric].count().max() < 2 if not data.empty else True:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        for agent, group in data.groupby("agent"):
            ax.plot(group["timestep"], group[metric], marker="o", label=agent)
        ax.set_xlabel("Training timesteps")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        files.append(savefig_appendix(f"appendix_{metric}.png"))
    return files


def write_interpretation(figures: list[str]) -> None:
    entries = [
        ("PPO Validation Return Over Training", "final_ppo_training_curve_data.csv", "timesteps", "mean validation return", "Checkpoint 22288 is marked; later recovered checkpoints show conservatism/collapse.", "Training improved early but longer training did not monotonically improve decision quality.", "This does not prove test-set generalization.", "Validation return over training with validation-only model selection.", "Use this to explain why checkpoint selection matters."),
        ("Recovered Action Distribution Over Training", "final_ppo_training_curve_data.csv", "checkpoint timestep", "action share", "Recovered policy becomes mostly zero markdown later.", "Recovered calibration pushed behavior toward conservatism.", "Do not interpret deterministic action shares as stochastic policy entropy.", "Recovered policy action distribution over checkpoints.", "Show the collapse narrative."),
        ("PPO vs Baselines", "ppo_financial_paired_validation_summary.csv", "policy", "normalized profit", "Always-zero is highest; recovered PPO beats observed PPO.", "Financial PPO is not the best average validation policy.", "Raw profit is not scale-comparable here.", "Paired validation normalized-profit comparison.", "Use this as the main performance result."),
        ("Financial Markdown Opportunity Map", "financial_markdown_oracle_diagnostic.csv", "inventory coverage", "fraction expiring soon", "Most states prefer 0%; sparse high-risk states prefer positive markdown.", "The environment is not degenerate.", "Do not claim markdown is broadly optimal.", "Oracle map of state-dependent markdown opportunities.", "Explain why average zero baseline can win while opportunities exist."),
        ("PPO vs Oracle Error Decomposition", "financial_markdown_ppo_regret.csv", "policy", "decision share", "Main error is unnecessary markdown.", "PPO loses value through over-markdown relative to oracle.", "Do not claim PPO never markdowns.", "Decision errors versus diagnostic oracle.", "Use this to connect RL underperformance to decision quality."),
        ("Empirical Profit-Waste Trade-off", "sustainability_tradeoff_vs_always_zero.csv", "waste rate", "normalized profit", "Waste reduction is possible at measurable profit cost.", "Sustainability is a constrained trade-off, not a weighted PPO success.", "Do not call this a full Pareto frontier.", "Empirical profit-waste trade-off over finite policies.", "Use this for the innovation/sustainability angle."),
    ]
    lines = ["# Final Figure Interpretation\n"]
    for title, source, xaxis, yaxis, pattern, valid, invalid, caption, talk in entries:
        lines.extend(
            [
                f"## {title}",
                f"- Data source: `{source}`",
                f"- x-axis meaning: {xaxis}",
                f"- y-axis meaning: {yaxis}",
                f"- Visible pattern: {pattern}",
                f"- Valid interpretation: {valid}",
                f"- Invalid interpretation: {invalid}",
                f"- Suggested report caption: {caption}",
                f"- Suggested 20-second presentation explanation: {talk}",
                "",
            ]
        )
    lines.extend(
        [
            "## Required Scientific Conclusions",
            "1. PPO training initially improved validation performance.",
            "2. Training longer did not monotonically improve policy quality.",
            "3. Recovered PPO became increasingly conservative and later collapsed to deterministic zero markdown.",
            "4. Checkpoint_22288 retained state dependence and was selected using validation only.",
            "5. Recovered PPO outperformed observed PPO but not always-zero.",
            "6. Zero markdown is optimal in most states, but not all states.",
            "7. Positive markdown is financially valuable in approximately 16.7% of audited recovered-calibration states.",
            "8. PPO mainly loses value through unnecessary markdown.",
            "9. Recovered calibration lowers regret relative to observed calibration.",
            "10. Sustainability results show a measurable profit-waste trade-off.",
            "",
        ]
    )
    (DOCS / "final_figure_interpretation.md").write_text("\n".join(lines), encoding="utf-8")


def figure_selection_table() -> pd.DataFrame:
    rows = [
        ("outputs/figures/final_report/figure_ppo_validation_return_over_training.png", "PPO training", "main_text", 1, "Checkpoint selection and non-monotonic training.", "outputs/tables/final_ppo_training_curve_data.csv"),
        ("outputs/figures/final_report/figure_recovered_action_distribution_over_training.png", "Policy behavior", "main_text", 2, "Recovered policy becomes conservative and collapses later.", "outputs/tables/final_ppo_training_curve_data.csv"),
        ("outputs/figures/final_report/figure_ppo_vs_baselines_normalized_profit.png", "Financial validation", "main_text", 3, "Always-zero beats both PPO policies; recovered beats observed.", "outputs/tables/ppo_financial_paired_validation_summary.csv"),
        ("outputs/figures/final_report/figure_financial_markdown_opportunity_map.png", "Decision opportunity", "main_text", 4, "Sparse positive markdown opportunities exist.", "outputs/tables/financial_markdown_oracle_diagnostic.csv"),
        ("outputs/figures/final_report/figure_ppo_vs_oracle_error_decomposition.png", "Decision quality", "main_text", 5, "PPO mainly loses value through unnecessary markdown.", "outputs/tables/financial_markdown_ppo_regret.csv"),
        ("outputs/figures/final_report/figure_empirical_profit_waste_tradeoff.png", "Sustainability", "main_text", 6, "Empirical profit-waste trade-off over finite policies.", "outputs/tables/sustainability_tradeoff_vs_always_zero.csv"),
        ("outputs/figures/final_report/figure_oracle_action_values_by_state.png", "Decision quality", "appendix", 7, "Representative state action-value gaps.", "outputs/tables/financial_markdown_multistep_action_values.csv"),
        ("outputs/figures/final_report/figure_waste_reduction_efficiency.png", "Sustainability", "appendix", 8, "Efficiency of waste reduction relative to profit sacrifice.", "outputs/tables/sustainability_waste_reduction_efficiency.csv"),
    ]
    df = pd.DataFrame(rows, columns=["figure_file", "report_section", "main_text_or_appendix", "presentation_priority", "key_message", "required_source_table"])
    df["ready_status"] = df["figure_file"].map(lambda p: "READY" if (PROJECT_ROOT / p).exists() else "MISSING")
    df.to_csv(TABLES / "final_report_figure_selection.csv", index=False)
    return df


def main() -> None:
    ensure_dirs()
    inventory = artifact_inventory()
    curve = consolidate_training_curve()
    figures = []
    figures.append(plot_validation_curve(curve))
    figures.append(plot_action_distribution(curve, "observed_financial", "figure_observed_action_distribution_over_training.png"))
    figures.append(plot_action_distribution(curve, "recovered_financial", "figure_recovered_action_distribution_over_training.png"))
    figures.append(plot_entropy(curve))
    figures.append(plot_ppo_vs_baselines())
    figures.append(plot_selected_policy_distribution())
    figures.append(plot_opportunity_map())
    figures.append(plot_ppo_oracle_errors())
    figures.append(plot_oracle_action_values())
    figures.append(plot_profit_waste_tradeoff())
    figures.append(plot_waste_efficiency())
    appendix = plot_appendix_diagnostics(curve)
    figures.extend(appendix)
    write_interpretation(figures)
    selection = figure_selection_table()
    required = [
        "figure_ppo_validation_return_over_training.png",
        "figure_observed_action_distribution_over_training.png",
        "figure_recovered_action_distribution_over_training.png",
        "figure_policy_entropy_and_collapse.png",
        "figure_ppo_vs_baselines_normalized_profit.png",
        "figure_financial_markdown_opportunity_map.png",
        "figure_ppo_vs_oracle_error_decomposition.png",
        "figure_empirical_profit_waste_tradeoff.png",
    ]
    ready = all((FIGURES / name).exists() for name in required)
    status = "FINAL_TRAINING_AND_RESULT_FIGURES_READY" if ready and (selection["ready_status"] == "READY").all() else "FINAL_TRAINING_AND_RESULT_FIGURES_REQUIRE_REVISION"
    missing_diagnostics = []
    for metric in ["explained_variance", "value_loss", "policy_gradient_loss", "approx_kl", "clip_fraction", "entropy_loss"]:
        if not (APPENDIX / f"appendix_{metric}.png").exists():
            missing_diagnostics.append(metric)
    report = {
        "status": status,
        "figures_created": figures,
        "artifact_inventory_rows": int(len(inventory)),
        "missing_diagnostics": missing_diagnostics,
        "selected_report_figures": selection.loc[selection["main_text_or_appendix"].eq("main_text"), "figure_file"].tolist(),
        "selected_presentation_figures": selection.sort_values("presentation_priority").head(5)["figure_file"].tolist(),
        "main_training_interpretation": "Checkpoint_22288 retained useful state dependence; longer recovered training became overly conservative and later collapsed to deterministic zero markdown.",
        "main_decision_quality_interpretation": "Always-zero wins average financial validation because zero markdown is optimal in most audited states, but sparse positive-markdown opportunities exist and PPO often loses value through unnecessary markdown.",
        "limitations": [
            "Validation split only; no test split used.",
            "Empirical deterministic action entropy is available; true stochastic policy entropy is not consistently available.",
            "Some training diagnostics are available only at sparse checkpoints.",
            "Sustainability trade-off is over a finite policy set, not a complete Pareto frontier.",
        ],
    }
    (TABLES / "final_training_and_result_figures_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(status)


if __name__ == "__main__":
    main()
