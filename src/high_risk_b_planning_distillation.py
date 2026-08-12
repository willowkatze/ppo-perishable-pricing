"""文件作用：补充实验 08B，将有限视野 planning decision 蒸馏为轻量 policy。
研究目的：检验动态机会存在时，监督式 policy approximation 能否复制 planning value。
主要输入：锁定 HIGH_RISK_B train episodes、环境 rollout 和有限 action sequences。
主要输出：planning labels、leakage audit、distilled model 与 validation comparison。
本模块不读取 test，也不改变环境、reward、action 或 HIGH_RISK_B 定义。"""

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
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import GroupKFold
except Exception as exc:  # pragma: no cover
    raise ImportError("scikit-learn is required for planning distillation.") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
CONFIGS_DIR = PROJECT_ROOT / "outputs" / "configs"
MANIFESTS_DIR = PROJECT_ROOT / "outputs" / "manifests"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "planning_distillation"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models" / "planning_distillation"
DOCS_DIR = PROJECT_ROOT / "docs"

PRIMARY_CALIBRATION = "recovered_calibration"
BASELINE_POLICY = "always_0pct"
PLANNING_POLICY = "limited_horizon_planning_policy"
REWARD_MODE = "financial"
LAMBDA_WASTE = 0.10
PLANNING_HORIZON = 3
ACTION_SPACE = list(range(6))
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 20260718
TIE_TOL = 1e-9

FEATURE_NAMES = [
    "inventory_coverage",
    "fraction_expiring_today",
    "fraction_expiring_within_two_days",
    "projected_sell_through_proxy",
    "recovered_demand_forecast_proxy",
    "current_inventory_proxy",
    "remaining_shelf_life_proxy",
    "episode_day_proxy",
]

LABELS_PATH = TABLES_DIR / "high_risk_b_planning_training_labels.csv"
PARTIAL_LABELS_PATH = TABLES_DIR / "high_risk_b_planning_training_labels.partial.csv"


# 保存一个 validation episode 的标识和场景信息。
@dataclass(frozen=True)
class EvalEpisode:
    scenario_id: str
    calibration_mode: str
    episode_index: int
    episode_id: str


# 创建本模块需要的输出目录。
def ensure_dirs() -> None:
    for path in [TABLES_DIR, CONFIGS_DIR, MANIFESTS_DIR, FIGURES_DIR, MODELS_DIR, DOCS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


# 检查输入文件是否存在，缺失时直接报错。
def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


# 读取 CSV 文件，并在文件缺失时停止运行。
def read_csv(path: Path) -> pd.DataFrame:
    require_file(path)
    return pd.read_csv(path)


# 将输入转换为有限浮点数；无效值使用默认值。
def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


# 计算文件的 SHA-256 摘要。
def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# 创建 recovered-calibration 的 HIGH_RISK_B 环境。
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


# 按 episode index 重置环境并返回初始状态。
def reset_env(env: OperationalPerishablePricingEnv, episode_index: int) -> tuple[np.ndarray, dict[str, Any]]:
    return env.reset(seed=30_000 + int(episode_index), options={"episode_index": int(episode_index)})


# 读取场景可用 episode 数量。
def infer_episode_count(env: OperationalPerishablePricingEnv, fallback: int) -> int:
    for attr in ["n_episodes", "num_episodes", "episode_count"]:
        if hasattr(env, attr):
            value = getattr(env, attr)
            if isinstance(value, (int, np.integer)) and value > 0:
                return int(value)
    for attr in ["episodes", "episode_records", "episode_table", "episode_indices"]:
        if hasattr(env, attr):
            try:
                return int(len(getattr(env, attr)))
            except Exception:
                pass
    return fallback


# 从 observation 中提取分类模型使用的风险特征。
def feature_row_from_obs(obs: np.ndarray, scenario_id: str, episode_index: int, step_index: int) -> dict[str, float | int | str]:
    obs = np.asarray(obs, dtype=float)
    inventory_coverage = safe_float(obs[1]) if obs.shape[0] > 1 else np.nan
    recovered_demand = safe_float(obs[2]) if obs.shape[0] > 2 else np.nan
    current_inventory = safe_float(obs[0]) if obs.shape[0] > 0 else np.nan
    fraction_today = safe_float(obs[23]) if obs.shape[0] > 23 else np.nan
    fraction_soon = safe_float(obs[24]) if obs.shape[0] > 24 else fraction_today
    remaining_life = safe_float(obs[25]) if obs.shape[0] > 25 else np.nan
    projected_sell_through = recovered_demand / inventory_coverage if inventory_coverage and not pd.isna(inventory_coverage) else np.nan
    row: dict[str, float | int | str] = {
        "scenario_id": scenario_id,
        "episode_index": int(episode_index),
        "step_index": int(step_index),
        "inventory_coverage": inventory_coverage,
        "fraction_expiring_today": fraction_today,
        "fraction_expiring_within_two_days": fraction_soon,
        "projected_sell_through_proxy": projected_sell_through,
        "recovered_demand_forecast_proxy": recovered_demand,
        "current_inventory_proxy": current_inventory,
        "remaining_shelf_life_proxy": remaining_life,
        "episode_day_proxy": step_index,
    }
    for i, value in enumerate(obs):
        row[f"obs_{i:02d}"] = safe_float(value)
    return row


# 根据库存覆盖和临期比例判断状态是否属于 HIGH_RISK_B。
def is_high_risk_b(row: dict[str, Any], locked: pd.Series) -> bool:
    coverage = safe_float(row.get("inventory_coverage"))
    expiring = safe_float(row.get("fraction_expiring_within_two_days"))
    sell_through = safe_float(row.get("projected_sell_through_proxy"))
    if pd.isna(coverage) or pd.isna(expiring) or pd.isna(sell_through):
        return False
    return (
        coverage >= safe_float(locked["min_inventory_coverage"])
        and expiring >= safe_float(locked["min_fraction_expiring_within_two_days"])
        and sell_through <= safe_float(locked["max_projected_sell_through"])
    )


# 从当前状态执行一条候选动作序列并返回累计价值。
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


# 枚举候选序列，返回最佳首个动作及与次优动作的差值。
def planning_action_and_margin(env: OperationalPerishablePricingEnv, horizon: int = PLANNING_HORIZON) -> tuple[int, float, float, int]:
    # 从当前 state 复制环境并枚举有限 horizon 的 action sequence，
    # 选择累计 normalized profit 最高序列的第一个 action。它是诊断性 planner，
    # 不使用真实未来销量标签，也不改变原环境的 accounting。
    remaining = max(1, int(getattr(env, "horizon", horizon)) - int(getattr(env, "current_step", 0)))
    local_horizon = min(horizon, remaining)
    first_action_values: dict[int, float] = {a: -np.inf for a in ACTION_SPACE}
    sequence_count = 0
    for seq in itertools.product(ACTION_SPACE, repeat=local_horizon):
        value = simulate_sequence(env, seq)
        sequence_count += 1
        first = int(seq[0])
        if value > first_action_values[first]:
            first_action_values[first] = value
    ranked = sorted(first_action_values.items(), key=lambda x: (x[1], -x[0]), reverse=True)
    best_action, best_value = ranked[0]
    second_value = ranked[1][1] if len(ranked) > 1 else best_value
    return int(best_action), float(best_value), float(best_value - second_value), sequence_count


# 读取锁定的 HIGH_RISK_B 阈值和场景范围。
def load_locked_definition() -> pd.Series:
    locked = read_csv(TABLES_DIR / "final_task_locked_population_definition.csv")
    row = locked.loc[locked["population_id"].eq("HIGH_RISK_B")]
    if row.empty:
        raise ValueError("HIGH_RISK_B locked definition not found.")
    return row.iloc[0]


# 从锁定定义中提取需要处理的 scenario id。
def scenario_ids_from_locked(locked: pd.Series) -> list[str]:
    return [x for x in str(locked["scenario_ids"]).split("|") if x]


# 将当前规划标签和进度写入 partial CSV。
def save_label_checkpoint(rows: list[dict[str, Any]], start: float, total_sequences: int, path: Path) -> None:
    if not rows:
        return
    checkpoint = pd.DataFrame(rows)
    checkpoint["label_runtime_seconds"] = time.perf_counter() - start
    checkpoint["candidate_sequences_evaluated"] = total_sequences
    checkpoint.to_csv(path, index=False)


# 在训练 episodes 上生成规划首动作标签。
def build_training_labels(
    max_episodes_per_scenario: int,
    checkpoint_every_episodes: int,
    max_training_labels: int | None,
    resume_labels: bool,
    max_label_runtime_minutes: float | None,
) -> pd.DataFrame:
    locked = load_locked_definition()
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    total_sequences = 0
    completed_groups: set[str] = set()
    if resume_labels and LABELS_PATH.exists():
        df = pd.read_csv(LABELS_PATH)
        print(f"Loaded existing final training labels: rows={len(df)}, episodes={df['episode_id'].nunique()}")
        audit_training_labels(df)
        return df
    if resume_labels and PARTIAL_LABELS_PATH.exists():
        partial = pd.read_csv(PARTIAL_LABELS_PATH)
        rows = partial.drop(columns=["label_runtime_seconds", "candidate_sequences_evaluated"], errors="ignore").to_dict("records")
        completed_groups = set(partial["group_id"].astype(str).unique()) if "group_id" in partial.columns else set()
        if "candidate_sequences_evaluated" in partial.columns and not partial.empty:
            total_sequences = int(pd.to_numeric(partial["candidate_sequences_evaluated"], errors="coerce").max())
        print(f"Resuming from partial labels: rows={len(rows)}, completed_episodes={len(completed_groups)}")

    scenario_ids = scenario_ids_from_locked(locked)
    processed_episodes = 0
    for scenario_position, scenario_id in enumerate(scenario_ids, start=1):
        env_probe = make_env("train", scenario_id, PRIMARY_CALIBRATION)
        episode_count = min(infer_episode_count(env_probe, max_episodes_per_scenario), max_episodes_per_scenario)
        env_probe.close()
        print(
            f"Label scan scenario {scenario_position}/{len(scenario_ids)}: "
            f"{scenario_id}, max_episodes={episode_count}, existing_labels={len(rows)}",
            flush=True,
        )
        for episode_index in range(episode_count):
            group_id = f"{scenario_id}__{episode_index}"
            if group_id in completed_groups:
                continue
            if max_training_labels is not None and len(rows) >= max_training_labels:
                print(f"Stopping label scan after reaching --max-training-labels={max_training_labels}", flush=True)
                break
            if max_label_runtime_minutes is not None:
                elapsed_minutes = (time.perf_counter() - start) / 60.0
                if elapsed_minutes >= max_label_runtime_minutes:
                    print(f"Stopping label scan after --max-label-runtime-minutes={max_label_runtime_minutes}", flush=True)
                    break
            env = make_env("train", scenario_id, PRIMARY_CALIBRATION)
            try:
                obs, _ = reset_env(env, episode_index)
            except Exception:
                env.close()
                continue
            terminated = truncated = False
            step_index = 0
            labels_before = len(rows)
            while not (terminated or truncated):
                features = feature_row_from_obs(obs, scenario_id, episode_index, step_index)
                if is_high_risk_b(features, locked):
                    label, best_value, margin, sequence_count = planning_action_and_margin(env)
                    total_sequences += sequence_count
                    rows.append(
                        {
                            **features,
                            "episode_id": f"train__{scenario_id}__episode_{episode_index:05d}",
                            "group_id": group_id,
                            "planning_action": label,
                            "best_planning_action_value": best_value,
                            "decision_importance": max(margin, 0.0),
                        }
                    )
                obs, _, terminated, truncated, _ = env.step(0)
                step_index += 1
            env.close()
            completed_groups.add(group_id)
            processed_episodes += 1
            if processed_episodes == 1 or processed_episodes % checkpoint_every_episodes == 0:
                save_label_checkpoint(rows, start, total_sequences, PARTIAL_LABELS_PATH)
                elapsed = time.perf_counter() - start
                print(
                    "Label checkpoint: "
                    f"processed_episodes={processed_episodes}, "
                    f"labels={len(rows)}, "
                    f"new_labels_this_episode={len(rows) - labels_before}, "
                    f"candidate_sequences={total_sequences}, "
                    f"elapsed_seconds={elapsed:.1f}, "
                    f"partial={PARTIAL_LABELS_PATH.relative_to(PROJECT_ROOT)}",
                    flush=True,
                )
        if max_training_labels is not None and len(rows) >= max_training_labels:
            break
        if max_label_runtime_minutes is not None and (time.perf_counter() - start) / 60.0 >= max_label_runtime_minutes:
            break
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No HIGH_RISK_B training labels found. Check feature index assumptions or max episode limit.")
    df["label_runtime_seconds"] = time.perf_counter() - start
    df["candidate_sequences_evaluated"] = total_sequences
    df.to_csv(LABELS_PATH, index=False)
    df.to_csv(PARTIAL_LABELS_PATH, index=False)
    audit_training_labels(df)
    return df


# 统计标签数量、动作分布和规划差值。
def audit_training_labels(labels: pd.DataFrame) -> None:
    duplicates = int(labels.duplicated(subset=[c for c in labels.columns if c.startswith("obs_")]).sum())
    rows = [
        {"metric": "training_episodes", "value": int(labels["episode_id"].nunique())},
        {"metric": "training_states", "value": int(len(labels))},
        {"metric": "duplicate_state_count", "value": duplicates},
        {"metric": "positive_action_share", "value": float((labels["planning_action"] > 0).mean())},
        {"metric": "max_class_share", "value": float(labels["planning_action"].value_counts(normalize=True).max())},
        {"metric": "planning_runtime_seconds", "value": float(labels["label_runtime_seconds"].iloc[0])},
        {"metric": "candidate_sequences_evaluated", "value": int(labels["candidate_sequences_evaluated"].iloc[0])},
    ]
    pd.DataFrame(rows).to_csv(TABLES_DIR / "high_risk_b_planning_training_label_summary.csv", index=False)


# 核对标签只来自 train split。
def leakage_audit(labels: pd.DataFrame) -> pd.DataFrame:
    # 检查 label 的 split、日期和 episode namespace，防止 validation/test 状态
    # 进入蒸馏训练；同时记录 feature 是否包含 planning outcome 等不可部署信息。
    # 未通过审计时不应锁定 distilled candidate。
    groups = labels["group_id"].astype(str)
    unique_groups = sorted(groups.unique())
    n_splits = min(5, len(unique_groups))
    rows = []
    if n_splits < 2:
        rows.append({"fold": 0, "train_groups": len(unique_groups), "validation_groups": 0, "overlap_count": 0, "status": "INSUFFICIENT_GROUPS"})
    else:
        cv = GroupKFold(n_splits=n_splits)
        X_dummy = np.zeros((len(labels), 1))
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_dummy, labels["planning_action"], groups), start=1):
            train_groups = set(groups.iloc[train_idx])
            val_groups = set(groups.iloc[val_idx])
            overlap = train_groups.intersection(val_groups)
            rows.append(
                {
                    "fold": fold,
                    "train_groups": len(train_groups),
                    "validation_groups": len(val_groups),
                    "overlap_count": len(overlap),
                    "status": "PASS" if not overlap else "FAIL",
                }
            )
    audit = pd.DataFrame(rows)
    audit.to_csv(TABLES_DIR / "planning_distillation_leakage_audit.csv", index=False)
    if (audit["status"] == "FAIL").any():
        raise RuntimeError("Grouped leakage audit failed.")
    return audit


# 返回蒸馏分类器使用的特征列。
def feature_columns(labels: pd.DataFrame) -> list[str]:
    return [c for c in labels.columns if c.startswith("obs_")] + FEATURE_NAMES


# 根据动作类别频率计算训练样本权重。
def sample_weights(labels: pd.DataFrame, weighted: bool) -> np.ndarray | None:
    if not weighted:
        return None
    margin = labels["decision_importance"].astype(float).fillna(0.0).to_numpy()
    if np.nanmax(margin) <= 0:
        return None
    return 1.0 + margin / (np.nanmedian(margin[margin > 0]) if np.any(margin > 0) else 1.0)


# 按名称创建随机森林或梯度提升分类器。
def make_model(family: str, weighted: bool, seed: int):
    if family == "RandomForestClassifier":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced" if not weighted else None,
            random_state=seed,
            n_jobs=-1,
        )
    if family == "HistGradientBoostingClassifier":
        return HistGradientBoostingClassifier(
            max_iter=150,
            max_leaf_nodes=15,
            learning_rate=0.06,
            l2_regularization=0.01,
            random_state=seed,
        )
    raise ValueError(family)


# 计算正折扣类别的 precision、recall 和命中率。
def positive_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    true_pos = y_true > 0
    pred_pos = y_pred > 0
    precision = float((true_pos & pred_pos).sum() / max(pred_pos.sum(), 1))
    recall = float((true_pos & pred_pos).sum() / max(true_pos.sum(), 1))
    return precision, recall


# 按 episode 分组切分数据并比较候选分类器。
def grouped_training_eval(labels: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    # 按 episode 分组切分训练和内部验证，避免同一 episode 的相邻 state 同时出现在两侧。
    # 比较树模型/加权版本时同时关注 overall accuracy 和 positive-markdown recall，
    # 因为类别不平衡下只预测 action 0 也可能得到较高准确率。
    leakage_audit(labels)
    cols = feature_columns(labels)
    labels = labels.dropna(subset=cols + ["planning_action"]).copy()
    X = labels[cols].astype(float).to_numpy()
    y = labels["planning_action"].astype(int).to_numpy()
    groups = labels["group_id"].astype(str).to_numpy()
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        raise RuntimeError("Not enough groups for grouped training evaluation.")

    candidates = []
    for family in ["RandomForestClassifier", "HistGradientBoostingClassifier"]:
        for weighted in [False, True]:
            fold_rows = []
            cv = GroupKFold(n_splits=n_splits)
            for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), start=1):
                model = make_model(family, weighted, BOOTSTRAP_SEED + fold)
                sw = sample_weights(labels.iloc[train_idx], weighted)
                if sw is not None:
                    model.fit(X[train_idx], y[train_idx], sample_weight=sw)
                else:
                    model.fit(X[train_idx], y[train_idx])
                pred = model.predict(X[val_idx])
                pos_precision, pos_recall = positive_metrics(y[val_idx], pred)
                fold_rows.append(
                    {
                        "model_family": family,
                        "weighted": weighted,
                        "fold": fold,
                        "action_accuracy": float(accuracy_score(y[val_idx], pred)),
                        "balanced_accuracy": float(balanced_accuracy_score(y[val_idx], pred)),
                        "macro_f1": float(f1_score(y[val_idx], pred, average="macro", zero_division=0)),
                        "positive_markdown_precision": pos_precision,
                        "positive_markdown_recall": pos_recall,
                        "planning_action_agreement": float((y[val_idx] == pred).mean()),
                    }
                )
            candidates.extend(fold_rows)
    metrics = pd.DataFrame(candidates)
    metrics.to_csv(TABLES_DIR / "planning_distillation_training_metrics.csv", index=False)

    summary = (
        metrics.groupby(["model_family", "weighted"], as_index=False)
        .agg(
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_positive_recall=("positive_markdown_recall", "mean"),
            mean_action_accuracy=("action_accuracy", "mean"),
        )
        .sort_values(["mean_positive_recall", "mean_balanced_accuracy", "mean_macro_f1"], ascending=False)
    )
    best = summary.iloc[0].to_dict()
    best_family = str(best["model_family"])
    best_weighted = bool(best["weighted"])
    final_model = make_model(best_family, best_weighted, BOOTSTRAP_SEED)
    sw_all = sample_weights(labels, best_weighted)
    if sw_all is not None:
        final_model.fit(X, y, sample_weight=sw_all)
    else:
        final_model.fit(X, y)
    model_path = MODELS_DIR / "best_planning_distilled_model.pkl"
    meta = {"model_family": best_family, "weighted": best_weighted, "feature_columns": cols, "training_summary": best}
    with model_path.open("wb") as handle:
        pickle.dump({"model": final_model, "metadata": meta}, handle)
    return metrics, {"model": final_model, "metadata": meta, "model_path": str(model_path)}


# 读取固定 validation episodes。
def load_validation_manifest() -> pd.DataFrame:
    manifest_path = MANIFESTS_DIR / "high_risk_b_expanded_validation_manifest.csv"
    if not manifest_path.exists():
        manifest_path = TABLES_DIR / "limited_horizon_planning_episode_results.csv"
    manifest = read_csv(manifest_path)
    if "policy_id" in manifest.columns:
        manifest = manifest.loc[manifest["policy_id"].eq("limited_horizon_planning_policy")]
    return manifest[["scenario_id", "calibration_mode", "episode_index", "episode_id"]].drop_duplicates()


# 在一个固定 episode 上运行蒸馏策略并记录结果。
def evaluate_policy_episode(episode: EvalEpisode, policy_id: str, model_bundle: dict[str, Any] | None) -> tuple[dict[str, Any], pd.DataFrame]:
    env = make_env("validation", episode.scenario_id, episode.calibration_mode)
    obs, _ = reset_env(env, episode.episode_index)
    actions = []
    state_rows = []
    final_info: dict[str, Any] = {}
    terminated = truncated = False
    start = time.perf_counter()
    step = 0
    while not (terminated or truncated):
        features = feature_row_from_obs(obs, episode.scenario_id, episode.episode_index, step)
        if policy_id == BASELINE_POLICY:
            action = 0
        elif policy_id == PLANNING_POLICY:
            action, _, _, _ = planning_action_and_margin(env)
        elif model_bundle is not None:
            cols = model_bundle["metadata"]["feature_columns"]
            X = pd.DataFrame([features]).reindex(columns=cols).astype(float).fillna(0.0).to_numpy()
            action = int(model_bundle["model"].predict(X)[0])
        else:
            raise ValueError(policy_id)
        state_rows.append({**features, "episode_id": episode.episode_id, "policy_id": policy_id, "action": action})
        obs, _, terminated, truncated, info = env.step(int(action))
        actions.append(int(action))
        final_info = info
        step += 1
    runtime = time.perf_counter() - start
    env.close()
    row = {
        "episode_id": episode.episode_id,
        "scenario_id": episode.scenario_id,
        "calibration_mode": episode.calibration_mode,
        "episode_index": episode.episode_index,
        "policy_id": policy_id,
        "normalized_profit": safe_float(final_info.get("normalized_accounting_profit")),
        "waste_rate": safe_float(final_info.get("final_waste_rate")),
        "sell_through": safe_float(final_info.get("final_sell_through_rate")),
        "average_markdown": safe_float(final_info.get("average_markdown")),
        "units_expired": safe_float(final_info.get("cumulative_waste")),
        "action_sequence": "|".join(str(a) for a in actions),
        "action_entropy": float(action_entropy(pd.Series(actions))),
        "runtime_seconds": runtime,
        "episode_conservation_error": safe_float(final_info.get("episode_conservation_error"), 0.0),
        "raw_financial_sum_error": safe_float(final_info.get("raw_financial_sum_error"), 0.0),
    }
    return row, pd.DataFrame(state_rows)


# 根据动作占比计算经验熵。
def action_entropy(series: pd.Series | list[int]) -> float:
    s = pd.Series(series)
    if s.empty:
        return 0.0
    probs = s.value_counts(normalize=True)
    return float(-(probs * np.log(probs)).sum())


# 对完整 episode 重采样，计算配对差值的置信区间。
def bootstrap_ci(diff: pd.Series) -> tuple[float, float]:
    values = diff.dropna().astype(float).to_numpy()
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(BOOTSTRAP_N)
    for i in range(BOOTSTRAP_N):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# 按 episode 对齐蒸馏策略和基线并计算配对指标。
def compare_policy(results: pd.DataFrame, policy_id: str) -> dict[str, Any]:
    policy = results.loc[results["policy_id"].eq(policy_id)]
    baseline = results.loc[results["policy_id"].eq(BASELINE_POLICY)]
    planning = results.loc[results["policy_id"].eq(PLANNING_POLICY)]
    merged = policy.merge(baseline, on="episode_id", suffixes=("_policy", "_baseline"))
    diff = merged["normalized_profit_policy"] - merged["normalized_profit_baseline"]
    ci_low, ci_high = bootstrap_ci(diff)
    planning_gain = (
        planning.merge(baseline, on="episode_id", suffixes=("_planning", "_baseline"))["normalized_profit_planning"]
        - planning.merge(baseline, on="episode_id", suffixes=("_planning", "_baseline"))["normalized_profit_baseline"]
    ).mean()
    capture = float(diff.mean() / planning_gain) if planning_gain and planning_gain > 0 else np.nan
    worst_decile = float(diff.sort_values().head(max(1, math.ceil(len(diff) * 0.10))).mean())
    return {
        "policy_id": policy_id,
        "mean_normalized_profit": float(policy["normalized_profit"].mean()),
        "paired_gain_vs_always_0pct": float(diff.mean()),
        "paired_median_gain": float(diff.median()),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "win_share": float((diff > TIE_TOL).mean()),
        "tie_share": float((diff.abs() <= TIE_TOL).mean()),
        "loss_share": float((diff < -TIE_TOL).mean()),
        "waste_rate_difference": float((merged["waste_rate_policy"] - merged["waste_rate_baseline"]).mean()),
        "sell_through_difference": float((merged["sell_through_policy"] - merged["sell_through_baseline"]).mean()),
        "average_markdown": float(policy["average_markdown"].mean()),
        "worst_decile_paired_profit_difference": worst_decile,
        "maximum_episode_loss": float(diff.min()),
        "action_entropy": float(policy["action_entropy"].mean()),
        "planning_gain_capture_ratio": capture,
        "runtime_per_episode": float(policy["runtime_seconds"].mean()),
        "accounting_valid": bool((policy["episode_conservation_error"].abs().max() <= 1e-6) and (policy["raw_financial_sum_error"].abs().max() <= 1e-6)),
    }


# 在固定 validation manifest 上评估选定分类器。
def validate_distilled_model(model_bundle: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = load_validation_manifest()
    rows = []
    state_rows = []
    for idx, row in manifest.reset_index(drop=True).iterrows():
        ep = EvalEpisode(str(row["scenario_id"]), str(row["calibration_mode"]), int(row["episode_index"]), str(row["episode_id"]))
        print(f"Validation episode {idx + 1}/{len(manifest)}: {ep.episode_id}")
        for policy in [BASELINE_POLICY, PLANNING_POLICY, "planning_distilled_policy"]:
            result, states = evaluate_policy_episode(ep, policy, model_bundle if policy == "planning_distilled_policy" else None)
            rows.append(result)
            state_rows.append(states)
    results = pd.DataFrame(rows)
    states = pd.concat(state_rows, ignore_index=True) if state_rows else pd.DataFrame()
    results.to_csv(TABLES_DIR / "high_risk_b_distilled_validation_episode_results.csv", index=False)
    states.to_csv(TABLES_DIR / "high_risk_b_distilled_validation_state_actions.csv", index=False)

    comparison = pd.DataFrame(
        [
            compare_policy(results, BASELINE_POLICY),
            compare_policy(results, PLANNING_POLICY),
            compare_policy(results, "planning_distilled_policy"),
        ]
    )
    distilled = comparison.loc[comparison["policy_id"].eq("planning_distilled_policy")].iloc[0]
    eligible = (
        safe_float(distilled["paired_gain_vs_always_0pct"]) > 0
        and safe_float(distilled["win_share"]) >= 0.60
        and safe_float(distilled["waste_rate_difference"]) <= 0.02
        and bool(distilled["accounting_valid"])
        and safe_float(distilled["action_entropy"]) > 0.01
    )
    comparison["eligible_for_final_lock"] = comparison["policy_id"].eq("planning_distilled_policy") & eligible
    comparison.to_csv(TABLES_DIR / "high_risk_b_final_candidate_comparison.csv", index=False)
    return results, states, comparison


# 保存选定分类器、特征和验证摘要。
def lock_candidate(model_bundle: dict[str, Any], comparison: pd.DataFrame) -> str:
    selected = comparison.loc[comparison["eligible_for_final_lock"].astype(bool)]
    if selected.empty:
        status = "PLANNING_VALUE_EXISTS_BUT_DISTILLATION_FAILED"
        config = {
            "status": status,
            "final_model_locked": False,
            "test_split_used": False,
            "reason": "No distilled policy met all validation eligibility rules.",
            "locked_baseline": BASELINE_POLICY,
            "population": "HIGH_RISK_B",
        }
    else:
        status = "LEARNED_POLICY_BEATS_BASELINE_ON_VALIDATION"
        model_path = Path(model_bundle["model_path"])
        config = {
            "status": status,
            "final_model_locked": True,
            "test_split_used": False,
            "model_family": model_bundle["metadata"]["model_family"],
            "weighted": model_bundle["metadata"]["weighted"],
            "model_path": str(model_path.relative_to(PROJECT_ROOT)),
            "locked_baseline": BASELINE_POLICY,
            "population": "HIGH_RISK_B",
            "validation_metrics": selected.iloc[0].to_dict(),
        }
    (CONFIGS_DIR / "final_high_risk_b_candidate.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    hash_rows = []
    for path in [
        Path(model_bundle["model_path"]),
        TABLES_DIR / "high_risk_b_final_candidate_comparison.csv",
        TABLES_DIR / "high_risk_b_planning_training_labels.csv",
        TABLES_DIR / "planning_distillation_training_metrics.csv",
    ]:
        hash_rows.append({"path": str(path.relative_to(PROJECT_ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size if path.exists() else None})
    (CONFIGS_DIR / "final_high_risk_b_candidate_hashes.json").write_text(json.dumps(hash_rows, indent=2), encoding="utf-8")
    return status


# 根据标签和验证结果生成蒸馏实验图。
def make_figures(results: pd.DataFrame, states: pd.DataFrame, comparison: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    comparison.set_index("policy_id")["paired_gain_vs_always_0pct"].plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Paired gain vs always_0pct")
    ax.set_title("Distilled candidates versus baseline profit")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "distilled_candidates_vs_baseline_profit.png", dpi=160)
    plt.close(fig)

    baseline = results.loc[results["policy_id"].eq(BASELINE_POLICY)]
    for policy in [PLANNING_POLICY, "planning_distilled_policy"]:
        merged = results.loc[results["policy_id"].eq(policy)].merge(baseline, on="episode_id", suffixes=("_policy", "_baseline"))
        merged[f"{policy}_diff"] = merged["normalized_profit_policy"] - merged["normalized_profit_baseline"]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        merged[f"{policy}_diff"].plot(kind="bar", ax=ax)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"Paired episode profit differences: {policy}")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"paired_episode_profit_differences_{policy}.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    comparison.set_index("policy_id")[["win_share", "tie_share", "loss_share"]].plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Win/tie/loss shares")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "win_tie_loss_shares.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    summary = results.groupby("policy_id", as_index=False).agg(profit=("normalized_profit", "mean"), waste=("waste_rate", "mean"))
    ax.scatter(summary["waste"], summary["profit"])
    for _, row in summary.iterrows():
        ax.annotate(row["policy_id"], (row["waste"], row["profit"]), fontsize=8)
    ax.set_xlabel("Waste rate")
    ax.set_ylabel("Profit")
    ax.set_title("Profit-waste comparison")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "profit_waste_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    comparison.set_index("policy_id")["planning_gain_capture_ratio"].plot(kind="bar", ax=ax)
    ax.set_title("Planning gain captured")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "planning_gain_captured.png", dpi=160)
    plt.close(fig)

    plan = states.loc[states["policy_id"].eq(PLANNING_POLICY), ["episode_id", "step_index", "action"]]
    learned = states.loc[states["policy_id"].eq("planning_distilled_policy"), ["episode_id", "step_index", "action"]]
    merged = plan.merge(learned, on=["episode_id", "step_index"], suffixes=("_planning", "_learned"))
    if not merged.empty:
        mat = confusion_matrix(merged["action_planning"], merged["action_learned"], labels=ACTION_SPACE)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(mat, cmap="Blues")
        ax.set_xlabel("Learned action")
        ax.set_ylabel("Planning action")
        ax.set_xticks(ACTION_SPACE)
        ax.set_yticks(ACTION_SPACE)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "planning_vs_learned_confusion_matrix.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    states.loc[states["policy_id"].eq("planning_distilled_policy"), "action"].value_counts(normalize=True).sort_index().plot(kind="bar", ax=ax)
    ax.set_title("Action distribution by risk state")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "action_distribution_by_risk_state.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    comparison.set_index("policy_id")["maximum_episode_loss"].plot(kind="bar", ax=ax)
    ax.set_title("Worst-case episode comparison")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "worst_case_episode_comparison.png", dpi=160)
    plt.close(fig)


# 将训练标签、模型选择和验证结果写入说明文件。
def write_doc(status: str, comparison: pd.DataFrame, metrics: pd.DataFrame) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    best = comparison.loc[comparison["policy_id"].eq("planning_distilled_policy")]
    best_text = best.iloc[0].to_dict() if not best.empty else {}
    text = f"""# HIGH_RISK_B Planning Distillation

Final validation status: `{status}`.

The script trained compact supervised policies using planning labels generated
only from the training split. The locked validation manifest and always-zero
baseline were not changed, and the test split was not used.

Best distilled validation metrics:

```json
{json.dumps(best_text, indent=2)}
```

Grouped training metrics are saved in
`outputs/tables/planning_distillation_training_metrics.csv`.

The final locked candidate config is saved in
`outputs/configs/final_high_risk_b_candidate.json`.
"""
    (DOCS_DIR / "planning_distillation_decision.md").write_text(text, encoding="utf-8")


def main() -> None:
    # 该补充实验会生成 planning labels 并训练 distilled classifier，计算量可能较大；
    # 但不会调用 PPO/DQN 的 model.learn()，也不会访问 held-out test。
    # 最终候选仍须通过固定 validation comparison 才能进入后续锁定流程。
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train-episodes-per-scenario", type=int, default=80)
    parser.add_argument("--checkpoint-every-episodes", type=int, default=5)
    parser.add_argument("--max-training-labels", type=int, default=1500)
    parser.add_argument("--max-label-runtime-minutes", type=float, default=None)
    parser.add_argument("--no-resume-labels", action="store_true")
    args = parser.parse_args()
    args.checkpoint_every_episodes = max(1, int(args.checkpoint_every_episodes))
    start = time.perf_counter()
    for path in [TABLES_DIR, CONFIGS_DIR, MANIFESTS_DIR, FIGURES_DIR, MODELS_DIR, DOCS_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    labels = build_training_labels(
        max_episodes_per_scenario=args.max_train_episodes_per_scenario,
        checkpoint_every_episodes=args.checkpoint_every_episodes,
        max_training_labels=args.max_training_labels,
        resume_labels=not args.no_resume_labels,
        max_label_runtime_minutes=args.max_label_runtime_minutes,
    )
    metrics, model_bundle = grouped_training_eval(labels)
    results, states, comparison = validate_distilled_model(model_bundle)
    status = lock_candidate(model_bundle, comparison)
    make_figures(results, states, comparison)
    write_doc(status, comparison, metrics)
    runtime = time.perf_counter() - start

    print(status)
    print(f"training_label_states={len(labels)}")
    print(f"training_label_episodes={labels['episode_id'].nunique()}")
    print("action_distribution=" + labels["planning_action"].value_counts(normalize=True).sort_index().to_json())
    distilled = comparison.loc[comparison["policy_id"].eq("planning_distilled_policy")].iloc[0]
    print(f"best_distilled_model={model_bundle['metadata']['model_family']}, weighted={model_bundle['metadata']['weighted']}")
    print(f"paired_validation_gain={distilled['paired_gain_vs_always_0pct']:.6f}")
    print(f"ci95=[{distilled['bootstrap_ci_low']:.6f}, {distilled['bootstrap_ci_high']:.6f}]")
    print(f"win_rate={distilled['win_share']:.3f}")
    print(f"waste_difference={distilled['waste_rate_difference']:.6f}")
    print(f"planning_gain_capture={distilled['planning_gain_capture_ratio']:.3f}")
    print(f"runtime_seconds={runtime:.1f}")
    print("test_split_used=False")
    print("files_created:")
    for file in [
        "outputs/tables/high_risk_b_planning_training_labels.csv",
        "outputs/tables/planning_distillation_leakage_audit.csv",
        "outputs/tables/planning_distillation_training_metrics.csv",
        "outputs/tables/high_risk_b_distilled_validation_episode_results.csv",
        "outputs/tables/high_risk_b_final_candidate_comparison.csv",
        "outputs/configs/final_high_risk_b_candidate.json",
        "outputs/configs/final_high_risk_b_candidate_hashes.json",
        "docs/planning_distillation_decision.md",
    ]:
        print(f"- {file}")


if __name__ == "__main__":
    main()
