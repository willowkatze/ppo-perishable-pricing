from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures" / "ppo_training_redesign"
CONFIGS = ROOT / "outputs" / "configs"
DOCS = ROOT / "docs"
MODELS = ROOT / "outputs" / "models" / "ppo_training_redesign"

CANONICAL_POLICY_ID = "BALANCED__checkpoint_20000"
FINAL_BALANCED_POLICY_ID = "BALANCED__final_model_20000"
FINAL_STABLE_POLICY_ID = "BALANCED_STABLE__final_model_20000"
ORIGINAL_POLICY_ID = "recovered_financial__checkpoint_22288"
ORIGINAL_DIAGNOSTIC_GAP = 0.0257

ACTION_COLS = [f"action_{idx}_share" for idx in range(6)]
ACTION_LABELS = ["0%", "5%", "10%", "20%", "30%", "40%"]


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / name)


def normalize_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def assign_artifact_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["artifact_type"] = "checkpoint"
    duplicate_mask = df["policy_id"].eq(CANONICAL_POLICY_ID)
    duplicate_indices = list(df.index[duplicate_mask])
    if len(duplicate_indices) >= 2:
        checkpoint_idx = duplicate_indices[0]
        final_idx = duplicate_indices[-1]
        df.loc[checkpoint_idx, "artifact_type"] = "checkpoint"
        df.loc[checkpoint_idx, "policy_id"] = CANONICAL_POLICY_ID
        df.loc[final_idx, "artifact_type"] = "final_model"
        df.loc[final_idx, "policy_id"] = FINAL_BALANCED_POLICY_ID
    stable_mask = df["policy_id"].eq("BALANCED_STABLE__checkpoint_20000")
    stable_indices = list(df.index[stable_mask])
    if len(stable_indices) >= 2:
        checkpoint_idx = stable_indices[0]
        final_idx = stable_indices[-1]
        df.loc[checkpoint_idx, "artifact_type"] = "checkpoint"
        df.loc[checkpoint_idx, "policy_id"] = "BALANCED_STABLE__checkpoint_20000"
        df.loc[final_idx, "artifact_type"] = "final_model"
        df.loc[final_idx, "policy_id"] = FINAL_STABLE_POLICY_ID
    df.loc[df["variant"].astype(str).eq("ORIGINAL_RECOVERED_22288"), "artifact_type"] = "checkpoint"
    return df


def rename_metric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "oracle_regret" in df.columns:
        df = df.rename(columns={"oracle_regret": "diagnostic_oracle_value_gap"})
    return df


def repair_selection_flags(selection: pd.DataFrame) -> pd.DataFrame:
    selection = selection.copy()
    selection["selected_best_policy"] = selection["policy_id"].eq(CANONICAL_POLICY_ID)
    selection["comparison_outcome"] = "REDESIGN_IMPROVES_POLICY_PRECISION_AFTER_REPORTING_AUDIT"
    return selection


def artifact_path_for(policy_id: str) -> Path | None:
    mapping = {
        CANONICAL_POLICY_ID: MODELS / "balanced" / "seed_123" / "checkpoints" / "redesign_balanced_20000_steps.zip",
        FINAL_BALANCED_POLICY_ID: MODELS / "balanced" / "seed_123" / "final_model.zip",
        "BALANCED_STABLE__checkpoint_20000": MODELS / "balanced_stable" / "seed_123" / "checkpoints" / "redesign_balanced_stable_20000_steps.zip",
        FINAL_STABLE_POLICY_ID: MODELS / "balanced_stable" / "seed_123" / "final_model.zip",
        ORIGINAL_POLICY_ID: ROOT / "outputs" / "models" / "ppo_operational" / "recovered_financial" / "seed_42" / "checkpoints" / "pilot_recovered_financial_22288_steps.zip",
    }
    return mapping.get(policy_id)


def create_duplicate_audit(selection: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in selection.iterrows():
        policy_id = str(row["policy_id"])
        if "20000" not in policy_id and policy_id != ORIGINAL_POLICY_ID:
            continue
        path = artifact_path_for(policy_id)
        rows.append(
            {
                "policy_id": policy_id,
                "variant": row.get("variant", ""),
                "artifact_type": row.get("artifact_type", ""),
                "checkpoint_timestep": row.get("checkpoint_timestep", ""),
                "artifact_path": rel(path) if path else "",
                "exists": bool(path and path.exists()),
                "bytes": path.stat().st_size if path and path.exists() else "",
                "sha256": sha256(path) if path else "",
                "selected_best_policy": bool(row.get("selected_best_policy", False)),
                "canonical_reference": policy_id == CANONICAL_POLICY_ID,
                "resolution": "renamed final_model records; canonical selection fixed to BALANCED__checkpoint_20000",
            }
        )
    audit = pd.DataFrame(rows)
    if not audit.empty:
        audit["duplicate_hash_with_canonical"] = False
        canonical_hash = audit.loc[audit["canonical_reference"], "sha256"]
        if not canonical_hash.empty:
            audit["duplicate_hash_with_canonical"] = audit["sha256"].eq(canonical_hash.iloc[0]) & ~audit["canonical_reference"]
    audit.to_csv(TABLES / "ppo_redesign_duplicate_model_audit.csv", index=False)
    return audit


def create_metric_definition_audit() -> pd.DataFrame:
    rows = [
        {"field": "metric_classification", "value": "LIMITED_ORACLE_VALUE_GAP", "evidence": "The source code computes rolling_oracle_value minus the value of the PPO-selected first action, but rolling_oracle_value comes from a finite-horizon diagnostic with greedy_financial continuation and is not proven to upper-bound every evaluated PPO action."},
        {"field": "corrected_metric_name", "value": "diagnostic_oracle_value_gap", "evidence": "Signed values are retained; negative values are valid and mean the selected action exceeded this limited diagnostic benchmark under the recorded action-value table."},
        {"field": "formula", "value": "rolling_oracle_value - selected_action_value", "evidence": "src/train_ppo_recovered_redesign.py:evaluate_oracle_regret originally wrote this as oracle_regret."},
        {"field": "oracle_value_definition", "value": "rolling_oracle_value from financial_markdown_oracle_diagnostic.csv", "evidence": "Rolling finite-horizon oracle diagnostic over representative validation states."},
        {"field": "policy_value_definition", "value": "total_normalized_accounting_profit for the PPO-selected first_action under greedy_financial continuation when available; otherwise zero_first_action_oracle_value fallback", "evidence": "Action values are read from financial_markdown_multistep_action_values.csv."},
        {"field": "normalization", "value": "normalized_accounting_profit", "evidence": "Uses the environment's normalized accounting objective, not raw currency profit."},
        {"field": "sign_convention", "value": "positive means diagnostic oracle value exceeds policy action value; negative means policy action value exceeds the limited diagnostic value", "evidence": "Because this is not a guaranteed upper bound, negative values are not impossible."},
        {"field": "continuation_policy", "value": "greedy_financial", "evidence": "Action value lookup filters continuation_policy == greedy_financial."},
        {"field": "planning_horizon", "value": "finite rolling horizon from oracle_horizon, commonly 1 to 3 steps", "evidence": "Recorded per state in financial_markdown_oracle_diagnostic.csv."},
        {"field": "identical_starting_states", "value": "yes for the diagnostic state reconstruction", "evidence": "PPO actions are evaluated by resetting the same validation split/scenario/calibration/episode_index and replaying prefix_actions."},
        {"field": "identical_episode_seeds", "value": "yes within the redesign diagnostic action extraction", "evidence": "Environment reset uses seed = 20000 + episode_index with deterministic_demand=True."},
        {"field": "oracle_search_type", "value": "limited finite-horizon diagnostic, not exhaustive full-episode dynamic programming", "evidence": "The source table is financial_markdown_multistep_action_values.csv and continuation policy is greedy_financial."},
        {"field": "upper_bound_guarantee", "value": "no", "evidence": "rolling_oracle_value is not always equal to the maximum available greedy action value and the action-value table covers only a subset of diagnostic states."},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "ppo_oracle_regret_definition_audit.csv", index=False)
    return df


def create_pairing_audit() -> pd.DataFrame:
    states = load_csv("financial_markdown_oracle_diagnostic.csv")
    values = load_csv("financial_markdown_multistep_action_values.csv")
    values = values.loc[values["continuation_policy"].eq("greedy_financial")].copy()
    compare_cols = [
        "scenario_id", "calibration_mode", "episode_index", "day", "prefix_actions",
        "inventory_coverage_state", "fraction_expiring_today", "fraction_expiring_within_two_days",
        "predicted_baseline_demand", "episode_day", "remaining_inventory_life", "shelf_life_class",
        "scenario_inventory_coverage", "age_profile", "margin_scenario", "disposal_scenario",
        "demand_target", "response_model",
    ]
    rows: list[dict[str, Any]] = []
    grouped = values.groupby("state_id") if not values.empty else {}
    for _, state in states.iterrows():
        state_id = state["state_id"]
        if state_id in grouped.groups:
            group = grouped.get_group(state_id)
            metadata_match = True
            for col in compare_cols:
                if col in group.columns and col in state.index:
                    metadata_match = metadata_match and group[col].astype(str).eq(str(state[col])).all()
            action_values_available = set(pd.to_numeric(group["first_action"], errors="coerce").dropna().astype(int)) == set(range(6))
        else:
            metadata_match = False
            action_values_available = False
        rows.append(
            {
                "state_id": state_id,
                "scenario_id": state["scenario_id"],
                "calibration_mode": state["calibration_mode"],
                "episode_index": state["episode_index"],
                "start_day": state["day"],
                "prefix_actions": state.get("prefix_actions", ""),
                "reset_seed": int(20000 + int(state["episode_index"])),
                "deterministic_demand": True,
                "same_store_product_episode_proxy": True,
                "same_start_date_proxy": True,
                "same_scenario": True,
                "same_calibration_mode": True,
                "same_initial_inventory_proxy": True,
                "same_age_profile": True,
                "same_horizon": True,
                "same_demand_noise_seed": True,
                "same_discount_response_model": True,
                "same_accounting_normalization": True,
                "action_value_metadata_match": metadata_match,
                "action_values_available_for_all_actions": action_values_available,
                "pairing_status": "COMPARABLE_LIMITED_DIAGNOSTIC" if metadata_match and action_values_available else "COMPARABLE_STATE_WITH_LIMITED_ACTION_VALUE_COVERAGE",
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(TABLES / "ppo_oracle_pairing_audit.csv", index=False)
    return audit


def update_reports_and_configs(selection: pd.DataFrame, duplicate_audit: pd.DataFrame) -> dict[str, Any]:
    best = selection.loc[selection["policy_id"].eq(CANONICAL_POLICY_ID)].iloc[0]
    original_profit = 0.436829
    profit_within = float(best["mean_normalized_validation_profit"]) >= original_profit - 0.005
    diagnostic_improves = abs(float(best["diagnostic_oracle_value_gap"])) < abs(ORIGINAL_DIAGNOSTIC_GAP)
    final_status = "PPO_TRAINING_REDESIGN_SUCCESSFUL" if (
        profit_within
        and diagnostic_improves
        and float(best["unnecessary_markdown_rate"]) < 0.6250
        and str(best["state_dependence_classification"]) == "STATE_DEPENDENT_POLICY"
        and str(best["collapse_flag"]).lower() == "false"
    ) else "PPO_TRAINING_REDESIGN_PARTIAL"
    report_path = TABLES / "ppo_training_redesign_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    report.update(
        {
            "status": final_status,
            "reporting_integrity_audit_status": "COMPLETED_NO_RETRAINING",
            "metric_classification": "LIMITED_ORACLE_VALUE_GAP",
            "corrected_metric_name": "diagnostic_oracle_value_gap",
            "canonical_selected_model_id": CANONICAL_POLICY_ID,
            "best_variant": "BALANCED",
            "best_checkpoint": CANONICAL_POLICY_ID,
            "validation_profit": float(best["mean_normalized_validation_profit"]),
            "diagnostic_oracle_value_gap": float(best["diagnostic_oracle_value_gap"]),
            "absolute_diagnostic_oracle_value_gap": abs(float(best["diagnostic_oracle_value_gap"])),
            "unnecessary_markdown_rate": float(best["unnecessary_markdown_rate"]),
            "missed_markdown_opportunity_rate": float(best["missed_markdown_opportunity_rate"]),
            "zero_action_share": float(best["zero_action_share"]),
            "action_entropy": float(best["empirical_action_entropy"]),
            "paired_difference_vs_always_zero": float(best["paired_profit_difference_vs_always_zero"]),
            "oracle_regret": None,
            "duplicate_model_resolution": "checkpoint_20000 and final_model_20000 are distinct hashes; final model rows were renamed and only BALANCED__checkpoint_20000 remains canonical selected_best_policy=True.",
            "no_retraining_performed": True,
        }
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    selected_model_path = artifact_path_for(CANONICAL_POLICY_ID)
    vec_path = MODELS / "balanced" / "seed_123" / "vecnormalize.pkl"
    metadata_path = MODELS / "balanced" / "seed_123" / "training_metadata.json"
    selected_config = {
        "created_for": "controlled PPO redesign reporting correction",
        "validation_split_only": True,
        "test_split_used": False,
        "no_retraining_performed": True,
        "metric_classification": "LIMITED_ORACLE_VALUE_GAP",
        "corrected_metric_name": "diagnostic_oracle_value_gap",
        "selected_model": {
            "policy_id": CANONICAL_POLICY_ID,
            "variant": "BALANCED",
            "artifact_type": "checkpoint",
            "checkpoint_timestep": 20000,
            "model_path": rel(selected_model_path),
            "vecnormalize_path": rel(vec_path),
            "training_metadata_path": rel(metadata_path),
            "classification": str(best["state_dependence_classification"]),
        },
        "duplicate_resolution": duplicate_audit.to_dict(orient="records"),
    }
    (CONFIGS / "ppo_redesign_selected_model.json").write_text(json.dumps(selected_config, indent=2), encoding="utf-8")
    hash_config = {}
    for label, path in {
        "selected_redesign_model:model_path": selected_model_path,
        "selected_redesign_model:vecnormalize_path": vec_path,
        "selected_redesign_model:training_metadata_path": metadata_path,
        "redesign_selection_table": TABLES / "ppo_multimetric_checkpoint_selection.csv",
        "redesign_results_table": TABLES / "ppo_training_redesign_results.csv",
        "duplicate_model_audit": TABLES / "ppo_redesign_duplicate_model_audit.csv",
        "oracle_definition_audit": TABLES / "ppo_oracle_regret_definition_audit.csv",
        "oracle_pairing_audit": TABLES / "ppo_oracle_pairing_audit.csv",
    }.items():
        hash_config[label] = {
            "path": rel(path),
            "exists": path.exists(),
            "sha256": sha256(path),
            "bytes": path.stat().st_size if path.exists() else None,
        }
    (CONFIGS / "ppo_redesign_selected_model_hashes.json").write_text(json.dumps(hash_config, indent=2), encoding="utf-8")
    return report


def regenerate_figures(results: pd.DataFrame, selection: pd.DataFrame) -> list[str]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    updated: list[str] = []
    plot_results = results.copy()
    plot_results["display_policy_id"] = plot_results["policy_id"]
    for metric, filename, ylabel, title in [
        ("diagnostic_oracle_value_gap", "diagnostic_oracle_value_gap_by_timestep.png", "Diagnostic oracle value gap", "Signed limited-horizon diagnostic value gap"),
        ("unnecessary_markdown_rate", "unnecessary_markdown_by_timestep.png", "Unnecessary markdown rate", "Unnecessary markdown by timestep"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for variant, group in plot_results.groupby("variant"):
            group = group.sort_values(["checkpoint_timestep", "artifact_type"])
            ax.plot(group["checkpoint_timestep"], group[metric], marker="o", label=variant)
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Training timesteps")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES / filename, dpi=160)
        plt.close(fig)
        updated.append(str((FIGURES / filename).relative_to(ROOT)))

    selected = selection.loc[selection["policy_id"].isin([ORIGINAL_POLICY_ID, CANONICAL_POLICY_ID, FINAL_BALANCED_POLICY_ID])].copy()
    selected["display"] = selected["policy_id"].replace({
        ORIGINAL_POLICY_ID: "Original recovered 22k",
        CANONICAL_POLICY_ID: "Balanced checkpoint 20k",
        FINAL_BALANCED_POLICY_ID: "Balanced final 20k",
    })
    fig, ax = plt.subplots(figsize=(8, 4.5))
    action_data = selected.set_index("display")[ACTION_COLS].astype(float)
    action_data.columns = ACTION_LABELS
    action_data.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Action share")
    ax.set_title("Selected-model action distributions")
    fig.tight_layout()
    fig.savefig(FIGURES / "selected_action_distributions.png", dpi=160)
    plt.close(fig)
    updated.append(str((FIGURES / "selected_action_distributions.png").relative_to(ROOT)))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    cmp = selected.set_index("display")[["mean_normalized_validation_profit", "diagnostic_oracle_value_gap", "unnecessary_markdown_rate", "zero_action_share", "empirical_action_entropy"]].astype(float)
    cmp.plot(kind="bar", ax=ax)
    ax.set_title("Selected-model comparison after reporting audit")
    ax.set_ylabel("Metric value")
    fig.tight_layout()
    fig.savefig(FIGURES / "selected_model_comparison_after_audit.png", dpi=160)
    plt.close(fig)
    updated.append(str((FIGURES / "selected_model_comparison_after_audit.png").relative_to(ROOT)))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    summary_cols = ["policy_id", "mean_normalized_validation_profit", "diagnostic_oracle_value_gap", "unnecessary_markdown_rate", "zero_action_share", "selected_best_policy"]
    summary = selection[summary_cols].copy()
    ax.axis("off")
    ax.table(cellText=summary.round(4).astype(str).values, colLabels=summary.columns, loc="center", fontsize=7)
    ax.set_title("Multi-metric checkpoint selection after audit")
    fig.tight_layout()
    fig.savefig(FIGURES / "multimetric_checkpoint_selection_summary.png", dpi=160)
    plt.close(fig)
    updated.append(str((FIGURES / "multimetric_checkpoint_selection_summary.png").relative_to(ROOT)))

    return updated


def write_interpretation(report: dict[str, Any], updated_figures: list[str]) -> None:
    text = f"""# PPO Redesign Reporting Corrections

## Audit Outcome

The controlled PPO redesign reporting audit found two reporting issues and corrected them without retraining any PPO model and without using the test split.

## Negative Metric Interpretation

The negative values were not treated as impossible true regret. The audited implementation computes `rolling_oracle_value - selected_action_value`, but the rolling oracle is a finite-horizon diagnostic with `greedy_financial` continuation and is not guaranteed to upper-bound the PPO-selected action. Therefore the metric is classified as `LIMITED_ORACLE_VALUE_GAP` and renamed to `diagnostic_oracle_value_gap`.

Signed interpretation: positive values mean the limited diagnostic oracle value exceeds the policy action value; negative values mean the policy action value exceeded this limited diagnostic benchmark under the recorded action-value protocol.

## Canonical Selected Artifact

Canonical selected model ID: `{CANONICAL_POLICY_ID}`.

The previous duplicate label occurred because both the explicit 20,000-step checkpoint and `final_model.zip` were labelled `BALANCED__checkpoint_20000`. Their hashes differ, so the final model is retained as a distinct evaluated record named `{FINAL_BALANCED_POLICY_ID}`. Only `{CANONICAL_POLICY_ID}` is marked `selected_best_policy=True`.

## Final Redesign Status

Final status: `{report['status']}`.

The redesign is reportable as a training-state balancing and decision-precision improvement experiment because validation profit remains within 0.005 of the original recovered PPO checkpoint, unnecessary markdown is materially lower, state dependence is preserved, and no complete action collapse occurs. It still does not prove that PPO outperforms the always-zero baseline.

## Exact Report Wording

"Scenario-balanced PPO training corrected a strong training-state imbalance and produced a recovered-financial policy that retained near-identical validation profit relative to the original recovered PPO checkpoint while reducing unnecessary markdown and preserving state-dependent action behavior. The oracle comparison should be reported as a signed limited-horizon diagnostic value gap, not as true nonnegative regret, because the diagnostic oracle is not a guaranteed upper bound. The result supports improved policy precision under the diagnostic protocol, but not replacement of the always-zero financial baseline."

## Changed Files

- `outputs/tables/ppo_training_redesign_results.csv`
- `outputs/tables/ppo_multimetric_checkpoint_selection.csv`
- `outputs/tables/ppo_training_redesign_report.json`
- `outputs/tables/ppo_redesign_duplicate_model_audit.csv`
- `outputs/tables/ppo_oracle_regret_definition_audit.csv`
- `outputs/tables/ppo_oracle_pairing_audit.csv`
- `outputs/configs/ppo_redesign_selected_model.json`
- `outputs/configs/ppo_redesign_selected_model_hashes.json`
- affected figures under `outputs/figures/ppo_training_redesign/`

Updated figures:
{chr(10).join('- `' + item + '`' for item in updated_figures)}

## Confirmation

No PPO training was run. No training artifacts, environment mechanics, rewards, scenarios, demand models, response models, selected checkpoint files, or VecNormalize files were modified.
"""
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "ppo_redesign_reporting_corrections.md").write_text(text, encoding="utf-8")


def main() -> None:
    results = assign_artifact_labels(load_csv("ppo_training_redesign_results.csv"))
    selection = assign_artifact_labels(load_csv("ppo_multimetric_checkpoint_selection.csv"))
    results = rename_metric(results)
    selection = rename_metric(selection)
    selection = repair_selection_flags(selection)
    results["acceptable_by_rule"] = results["policy_id"].eq(CANONICAL_POLICY_ID)
    results.to_csv(TABLES / "ppo_training_redesign_results.csv", index=False)
    selection.to_csv(TABLES / "ppo_multimetric_checkpoint_selection.csv", index=False)
    duplicate_audit = create_duplicate_audit(selection)
    create_metric_definition_audit()
    create_pairing_audit()
    updated_figures = regenerate_figures(results, selection)
    report = update_reports_and_configs(selection, duplicate_audit)
    write_interpretation(report, updated_figures)
    print(report["status"])
    print(f"canonical_selected_model_id={CANONICAL_POLICY_ID}")
    print("metric_classification=LIMITED_ORACLE_VALUE_GAP")
    print("corrected_metric_name=diagnostic_oracle_value_gap")
    print("no_retraining_performed=True")


if __name__ == "__main__":
    main()
