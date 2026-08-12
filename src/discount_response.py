"""文件作用：实验顺序 03，校准 markdown 与 observed/recovered demand 的响应关系。
研究目的：让 pricing environment 能根据 action 估计不同折扣下的需求变化。
主要输入：带 observed sales 和 recovered demand 的 modeling subset。
主要输出：两套 response model、markdown support、混杂审计和响应曲线。
该关系是 observational/model-implied，不应解释为因果价格弹性。"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 42
EPSILON = 1e-6
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_ROOT / "data" / "freshretail" / "processed" / "freshretail_demand_recovered.parquet"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "discount_response"

OBSERVED_MODEL_PATH = MODELS_DIR / "discount_response_observed.joblib"
RECOVERED_MODEL_PATH = MODELS_DIR / "discount_response_recovered.joblib"
CONFIG_PATH = MODELS_DIR / "discount_response_config.json"

MARKDOWN_SUPPORT_PATH = TABLES_DIR / "discount_response_markdown_support.csv"
DESCRIPTIVE_SUMMARY_PATH = TABLES_DIR / "discount_response_descriptive_summary.csv"
CORRELATIONS_PATH = TABLES_DIR / "discount_response_correlations.csv"
BY_PRODUCT_PATH = TABLES_DIR / "discount_response_by_product.csv"
BY_STORE_PATH = TABLES_DIR / "discount_response_by_store.csv"
CONFOUNDING_AUDIT_PATH = TABLES_DIR / "discount_response_confounding_audit.csv"
MODEL_METRICS_PATH = TABLES_DIR / "discount_response_model_metrics.csv"
SEGMENT_METRICS_PATH = TABLES_DIR / "discount_response_segment_metrics.csv"
CURVES_PATH = TABLES_DIR / "discount_response_curves.csv"
SUPPORT_AUDIT_PATH = TABLES_DIR / "discount_response_support_audit.csv"
TARGET_COMPARISON_PATH = TABLES_DIR / "discount_response_target_comparison.csv"
COMPARISON_BY_SEGMENT_PATH = TABLES_DIR / "discount_response_comparison_by_segment.csv"
SENSITIVITY_PATH = TABLES_DIR / "discount_response_implied_sensitivity.csv"
BIAS_TRANSMISSION_PATH = TABLES_DIR / "discount_response_bias_transmission_summary.csv"
ACTION_RANKING_PATH = TABLES_DIR / "discount_response_action_ranking.csv"
ROBUSTNESS_PATH = TABLES_DIR / "discount_response_robustness.csv"

EXPECTED_ROWS = 29_100
EXPECTED_SERIES = 300
TARGETS = {
    "observed": "observed_sales_demand",
    "recovered": "recovered_demand",
}


# 把响应模型、encoder、特征顺序和预测范围保存在同一结构中。
@dataclass
class FittedResponseModel:
    """Fitted response model bundle."""

    target_name: str
    model_type: str
    model: Any
    feature_columns: list[str]
    encoders: dict[str, dict[str, int]]
    prediction_cap: float


# 创建本模块需要的输出目录。
def ensure_dirs() -> None:
    """Create output directories."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# 读取恢复后的需求数据，并检查响应模型需要的字段和 split。
def load_and_validate() -> pd.DataFrame:
    """Load recovered-demand data and validate required integrity."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Recovered-demand dataset not found: {INPUT_PATH}")
    data = pd.read_parquet(INPUT_PATH)
    required = {
        "timestamp",
        "store_id",
        "sku_id",
        "observed_sales_demand",
        "recovered_demand",
        "model_predicted_demand",
        "possible_stockout_flag",
        "recovered_row_flag",
        "recovery_clipped_flag",
        "recovery_extrapolation_flag",
        "stock_hour6_22_cnt",
        "discount",
        "markdown_rate",
        "activity_flag",
        "time_split",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = data.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data["store_id"] = data["store_id"].astype(str)
    data["sku_id"] = data["sku_id"].astype(str)
    if "product_id" not in data.columns:
        data["product_id"] = data["sku_id"]
    data["product_id"] = data["product_id"].astype(str)
    for column in [
        "observed_sales_demand",
        "recovered_demand",
        "model_predicted_demand",
        "stock_hour6_22_cnt",
        "discount",
        "markdown_rate",
        "avg_temperature",
    ]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in ["possible_stockout_flag", "recovered_row_flag", "recovery_clipped_flag", "recovery_extrapolation_flag", "activity_flag"]:
        data[column] = as_bool(data[column])

    duplicates = int(data.duplicated(["store_id", "sku_id", "timestamp"]).sum())
    split_values = set(data["time_split"].dropna().astype(str).unique())
    series_count = int(data[["store_id", "sku_id"]].drop_duplicates().shape[0])
    interval_bad = int(
        (
            data.sort_values(["store_id", "sku_id", "timestamp"])
            .groupby(["store_id", "sku_id"])["timestamp"]
            .diff()
            .dropna()
            .dt.days
            != 1
        ).sum()
    )
    if len(data) != EXPECTED_ROWS:
        warnings.warn(f"Expected {EXPECTED_ROWS} rows, found {len(data)}.")
    if series_count != EXPECTED_SERIES:
        warnings.warn(f"Expected {EXPECTED_SERIES} series, found {series_count}.")
    if duplicates:
        raise ValueError("Duplicate store-product-date records detected.")
    if split_values != {"train", "validation", "test"}:
        raise ValueError(f"Invalid chronological split labels: {split_values}")
    if interval_bad:
        raise ValueError("Time continuity violation detected.")
    if not data["markdown_rate"].between(0, 1, inclusive="both").all():
        raise ValueError("markdown_rate outside [0, 1].")
    if (data["recovered_demand"] + EPSILON < data["observed_sales_demand"]).any():
        raise ValueError("recovered_demand is below observed_sales_demand.")
    return data.sort_values(["store_id", "sku_id", "timestamp"]).reset_index(drop=True)


# 把常见布尔值写法统一转换为布尔序列。
def as_bool(series: pd.Series) -> pd.Series:
    """Convert mixed bool-like values to bool."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    lower = series.astype(str).str.lower().str.strip()
    true_values = {"1", "true", "yes", "y", "t"}
    false_values = {"0", "false", "no", "n", "f", "nan", "none", ""}
    parsed = lower.map(lambda value: True if value in true_values else False if value in false_values else np.nan)
    if parsed.isna().any():
        numeric = pd.to_numeric(series, errors="coerce")
        parsed = parsed.fillna(numeric.fillna(0) > 0)
    return parsed.astype(bool)


# 根据数据中的折扣分布生成分箱边界。
def markdown_bins(data: pd.DataFrame) -> list[float]:
    """Create practical markdown bins while avoiding unsupported upper bins."""
    clean = data["markdown_rate"].dropna().clip(0, 1)
    candidates = [0.0, 0.001, 0.05, 0.10, 0.20, 0.30, 1.0]
    max_markdown = float(clean.max())
    bins = [0.0, 0.001]
    for boundary in candidates[2:-1]:
        if max_markdown >= boundary and ((clean > bins[-1]) & (clean <= boundary)).sum() >= 20:
            bins.append(boundary)
    if max_markdown > bins[-1]:
        bins.append(min(1.0, max_markdown))
    if len(bins) < 3:
        bins = [0.0, 0.001, max(0.01, max_markdown)]
    return sorted(set(bins))


# 按既定边界为每条记录添加折扣分组。
def add_markdown_bin(data: pd.DataFrame, bins: list[float]) -> pd.DataFrame:
    """Attach markdown support bin labels."""
    output = data.copy()
    output["markdown_bin"] = pd.cut(
        output["markdown_rate"].clip(0, 1),
        bins=bins,
        include_lowest=True,
        duplicates="drop",
    ).astype(str)
    output.loc[output["markdown_rate"].eq(0), "markdown_bin"] = "0%"
    return output


# 统计每个折扣档位及业务分组中的样本量。
def audit_markdown_support(data: pd.DataFrame) -> pd.DataFrame:
    # 先确认历史数据在哪些 markdown 区间有足够样本。
    # 环境虽然提供固定 action grid，但弱支持区间的预测属于更强外推，
    # 因此 support flag 必须随模型保存并进入后续 policy 诊断。
    """Save markdown support summary by bin and split."""
    bins = markdown_bins(data)
    data = add_markdown_bin(data, bins)
    rows = []
    for keys, group in data.groupby(["time_split", "markdown_bin"], dropna=False):
        split, markdown_bin = keys
        rows.append(
            {
                "time_split": split,
                "markdown_bin": markdown_bin,
                "row_count": int(len(group)),
                "store_count": int(group["store_id"].nunique()),
                "product_count": int(group["product_id"].nunique()),
                "promotion_share": float(group["activity_flag"].mean()),
                "possible_stockout_share": float(group["possible_stockout_flag"].mean()),
                "observed_demand_mean": float(group["observed_sales_demand"].mean()),
                "observed_demand_median": float(group["observed_sales_demand"].median()),
                "recovered_demand_mean": float(group["recovered_demand"].mean()),
                "recovered_demand_median": float(group["recovered_demand"].median()),
                "inventory_mean": float(group["stock_hour6_22_cnt"].mean()),
                "inventory_median": float(group["stock_hour6_22_cnt"].median()),
            }
        )
    output = pd.DataFrame(rows)
    output.to_csv(MARKDOWN_SUPPORT_PATH, index=False)
    return output


# 按折扣、促销和缺货状态汇总 observed/recovered demand。
def descriptive_analysis(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Save descriptive demand-vs-markdown summaries and correlations."""
    rows = []
    for target_name, target_col in TARGETS.items():
        for markdown_bin, group in data.groupby("markdown_bin", dropna=False):
            values = group[target_col].dropna()
            rows.append(
                {
                    "target": target_name,
                    "markdown_bin": markdown_bin,
                    "n": int(len(values)),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "p10": float(values.quantile(0.10)),
                    "p25": float(values.quantile(0.25)),
                    "p75": float(values.quantile(0.75)),
                    "p90": float(values.quantile(0.90)),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(DESCRIPTIVE_SUMMARY_PATH, index=False)

    corr_rows = []
    for target_name, target_col in TARGETS.items():
        corr_rows += correlation_rows(data, target_name, target_col, "overall", "all")
        for value, group in data.groupby("activity_flag"):
            corr_rows += correlation_rows(group, target_name, target_col, "promotion_status", str(value))
        for value, group in data.groupby("possible_stockout_flag"):
            corr_rows += correlation_rows(group, target_name, target_col, "possible_stockout_status", str(value))
    correlations = pd.DataFrame(corr_rows)
    correlations.to_csv(CORRELATIONS_PATH, index=False)

    by_product = sufficient_group_correlations(data, "product_id")
    by_product.to_csv(BY_PRODUCT_PATH, index=False)
    by_store = sufficient_group_correlations(data, "store_id")
    by_store.to_csv(BY_STORE_PATH, index=False)
    return summary, correlations, by_product, by_store


# 计算一个分组内折扣与目标需求的相关系数。
def correlation_rows(data: pd.DataFrame, target_name: str, target_col: str, scope: str, value: str) -> list[dict[str, Any]]:
    """Calculate Pearson and Spearman correlations for one scope."""
    if len(data) < 20 or data["markdown_rate"].nunique() < 2:
        pearson = np.nan
        spearman = np.nan
    else:
        pearson = data["markdown_rate"].corr(data[target_col], method="pearson")
        spearman = data["markdown_rate"].corr(data[target_col], method="spearman")
    return [
        {
            "target": target_name,
            "scope": scope,
            "scope_value": value,
            "n": int(len(data)),
            "markdown_levels": int(data["markdown_rate"].nunique()),
            "pearson": float(pearson) if pd.notna(pearson) else np.nan,
            "spearman": float(spearman) if pd.notna(spearman) else np.nan,
        }
    ]


# 只对样本量足够的分组计算相关系数。
def sufficient_group_correlations(data: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Calculate correlations for products/stores with sufficient support."""
    rows = []
    for group_value, group in data.groupby(group_col, dropna=False):
        if len(group) < 40 or group["markdown_rate"].nunique() < 3:
            continue
        for target_name, target_col in TARGETS.items():
            rows += correlation_rows(group, target_name, target_col, group_col, str(group_value))
    return pd.DataFrame(rows)


# 比较折扣与商品、门店、时间和库存风险的共同变化。
def confounding_audit(data: pd.DataFrame) -> pd.DataFrame:
    """Audit associations between markdown and potential confounders."""
    # 历史折扣可能与库存、促销、星期和商品类别同时变化。
    # 将这些共同变化写入诊断表，避免把模型响应直接当作因果效应。

    rows = []
    numeric_cols = {
        "activity_flag": data["activity_flag"].astype(int),
        "possible_stockout_flag": data["possible_stockout_flag"].astype(int),
        "stock_hour6_22_cnt": data["stock_hour6_22_cnt"],
        "lag_1_observed_sales": data["lag_1_observed_sales"],
        "lag_1_recovered_demand": data["lag_1_recovered_demand"],
        "day_of_week": data["day_of_week"],
        "trend_index": data["trend_index"],
        "avg_temperature": data["avg_temperature"],
    }
    for name, values in numeric_cols.items():
        valid = pd.concat([data["markdown_rate"], values], axis=1).dropna()
        corr = valid.iloc[:, 0].corr(valid.iloc[:, 1], method="spearman") if len(valid) > 20 and valid.iloc[:, 1].nunique() > 1 else np.nan
        rows.append({"diagnostic": "spearman_with_markdown", "variable": name, "value": corr})

    for group_col in ["store_id", "product_id", "timestamp"]:
        grouped = data.groupby(group_col, dropna=False)["markdown_rate"].mean().sort_values(ascending=False)
        top = grouped.head(10)
        rows.append(
            {
                "diagnostic": f"top_{group_col}_markdown_concentration",
                "variable": group_col,
                "value": "; ".join([f"{idx}:{val:.3f}" for idx, val in top.items()]),
            }
        )
    rows.append(
        {
            "diagnostic": "low_inventory_markdown_difference",
            "variable": "stock_hour6_22_cnt",
            "value": float(
                data.loc[data["stock_hour6_22_cnt"].le(data["stock_hour6_22_cnt"].median()), "markdown_rate"].mean()
                - data.loc[data["stock_hour6_22_cnt"].gt(data["stock_hour6_22_cnt"].median()), "markdown_rate"].mean()
            ),
        }
    )
    output = pd.DataFrame(rows)
    output.to_csv(CONFOUNDING_AUDIT_PATH, index=False)
    return output


# 生成时间、商品、库存、历史需求和折扣交互特征。
def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe features using current and prior information only."""
    output = data.sort_values(["store_id", "sku_id", "timestamp"]).copy()
    output["day_of_week"] = output["timestamp"].dt.dayofweek
    output["week_index"] = ((output["timestamp"] - output["timestamp"].min()).dt.days // 7).astype(int)
    output["month"] = output["timestamp"].dt.month
    output["weekend_flag"] = output["day_of_week"].isin([5, 6]).astype(int)
    output["trend_index"] = (output["timestamp"] - output["timestamp"].min()).dt.days.astype(int)
    output["activity_flag"] = output["activity_flag"].astype(int)
    output["possible_stockout_flag"] = output["possible_stockout_flag"].astype(int)
    output["markdown_x_activity"] = output["markdown_rate"] * output["activity_flag"]
    group = output.groupby(["store_id", "sku_id"], dropna=False)
    output["lag_1_observed_sales"] = group["observed_sales_demand"].shift(1)
    output["lag_7_observed_sales"] = group["observed_sales_demand"].shift(7)
    output["lag_1_recovered_demand"] = group["recovered_demand"].shift(1)
    observed_shift = group["observed_sales_demand"].shift(1)
    recovered_shift = group["recovered_demand"].shift(1)
    output["rolling_mean_7_observed_sales"] = observed_shift.groupby([output["store_id"], output["sku_id"]]).rolling(7, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    output["rolling_mean_7_recovered_demand"] = recovered_shift.groupby([output["store_id"], output["sku_id"]]).rolling(7, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    output["rolling_mean_14_observed_sales"] = observed_shift.groupby([output["store_id"], output["sku_id"]]).rolling(14, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    output["previous_day_possible_stockout_flag"] = group["possible_stockout_flag"].shift(1).fillna(0).astype(int)
    output["previous_day_inventory"] = group["stock_hour6_22_cnt"].shift(1)
    output["demand_volume_group"] = pd.qcut(
        output["recovered_demand"].rank(method="first"),
        q=3,
        labels=["low", "medium", "high"],
    ).astype(str)
    output["stockout_risk_group"] = pd.cut(
        output.groupby(["store_id", "sku_id"])["possible_stockout_flag"].transform("mean"),
        bins=[-0.001, 0.33, 0.66, 1.0],
        labels=["low", "medium", "high"],
        include_lowest=True,
    ).astype(str)
    return output


def feature_columns() -> list[str]:
    """Feature specification shared by observed and recovered models."""
    return [
        "day_of_week",
        "week_index",
        "month",
        "weekend_flag",
        "trend_index",
        "store_id_encoded",
        "product_id_encoded",
        "lag_1_observed_sales",
        "lag_7_observed_sales",
        "rolling_mean_7_observed_sales",
        "rolling_mean_7_recovered_demand",
        "rolling_mean_14_observed_sales",
        "previous_day_possible_stockout_flag",
        "markdown_rate",
        "activity_flag",
        "markdown_x_activity",
        "stock_hour6_22_cnt",
        "previous_day_inventory",
        "possible_stockout_flag",
        "avg_temperature",
    ]


# 用训练集建立 store 和 product 编码。
def fit_encoders(train: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Fit stable ordinal encoders on training data only."""
    encoders = {}
    for col in ["store_id", "product_id"]:
        values = sorted(train[col].astype(str).unique().tolist())
        encoders[col] = {value: index + 1 for index, value in enumerate(values)}
    return encoders


# 把训练集编码应用到其他 split。
def apply_encoders(data: pd.DataFrame, encoders: dict[str, dict[str, int]]) -> pd.DataFrame:
    """Apply train-fitted encoders."""
    output = data.copy()
    output["store_id_encoded"] = output["store_id"].astype(str).map(encoders["store_id"]).fillna(0).astype(int)
    output["product_id_encoded"] = output["product_id"].astype(str).map(encoders["product_id"]).fillna(0).astype(int)
    return output


# 按模型特征顺序生成数值输入矩阵。
def prepare_x(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Prepare numeric model matrix."""
    x = data.reindex(columns=columns).copy()
    for col in x.columns:
        x[col] = pd.to_numeric(x[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return x


# 分别拟合 observed demand 和 recovered demand 的响应模型。
def fit_response_models(data: pd.DataFrame) -> tuple[dict[str, FittedResponseModel], pd.DataFrame, pd.DataFrame]:
    """Fit parallel response models and save metrics."""
    # observed 与 recovered target 分别拟合，其他 feature engineering 和 split 保持一致。
    # 训练只用 train，模型类型依据 validation error 选择；最终保存的 bundle 包含
    # feature 顺序、encoder 和预测上限，确保环境推理与训练预处理一致。
    train = data.loc[data["time_split"].eq("train")].copy()
    encoders = fit_encoders(train)
    data = apply_encoders(data, encoders)
    columns = feature_columns()
    fitted: dict[str, FittedResponseModel] = {}
    metric_rows = []
    segment_rows = []

    for target_name, target_col in TARGETS.items():
        y_train = np.log1p(train[target_col].clip(lower=0))
        train_encoded = apply_encoders(train, encoders)
        x_train = prepare_x(train_encoded, columns)
        model_candidates = {
            "hist_gradient_boosting": HistGradientBoostingRegressor(
                max_iter=180,
                learning_rate=0.06,
                max_leaf_nodes=31,
                l2_regularization=0.05,
                random_state=RANDOM_SEED,
            ),
            "ridge": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=2.0, random_state=RANDOM_SEED)),
                ]
            ),
        }
        candidate_metrics = []
        for model_type, model in model_candidates.items():
            model.fit(x_train, y_train)
            cap = float(train[target_col].quantile(0.995))
            for split in ["validation", "test"]:
                frame = data.loc[data["time_split"].eq(split)].copy()
                pred = predict_original_scale(model, model_type, frame, columns, cap)
                actual = frame[target_col].to_numpy(dtype=float)
                log_actual = np.log1p(np.clip(actual, 0, None))
                log_pred = np.log1p(np.clip(pred, 0, None))
                row = model_metric_row(target_name, model_type, split, actual, pred, log_actual, log_pred)
                metric_rows.append(row)
                if split == "validation":
                    candidate_metrics.append(row)
                segment_rows.extend(segment_metric_rows(target_name, model_type, split, frame, actual, pred, log_actual, log_pred))
        selected_type = select_model_type(candidate_metrics)
        selected_model = model_candidates[selected_type]
        cap = float(train[target_col].quantile(0.995))
        fitted[target_name] = FittedResponseModel(target_name, selected_type, selected_model, columns, encoders, cap)

    metrics = pd.DataFrame(metric_rows)
    segment_metrics = pd.DataFrame(segment_rows)
    metrics.to_csv(MODEL_METRICS_PATH, index=False)
    segment_metrics.to_csv(SEGMENT_METRICS_PATH, index=False)
    return fitted, metrics, segment_metrics


# 根据验证误差选择响应模型类型。
def select_model_type(metric_rows: list[dict[str, Any]]) -> str:
    """Select response model using validation performance only."""
    metrics = pd.DataFrame(metric_rows)
    validation = metrics.loc[metrics["time_split"].eq("validation")].copy()
    validation = validation.sort_values(["mae", "abs_mean_bias", "rmse"])
    return str(validation.iloc[0]["model_type"])


# 将对数预测转换回需求数量并限制到配置范围。
def predict_original_scale(model: Any, model_type: str, frame: pd.DataFrame, columns: list[str], cap: float) -> np.ndarray:
    """Predict original-scale demand from log-scale model."""
    x = prepare_x(frame, columns)
    pred_log = model.predict(x)
    pred = np.expm1(pred_log)
    return np.clip(pred, 0.0, max(cap * 1.5, cap + 1.0))


# 计算一个模型在指定数据上的误差指标。
def model_metric_row(target: str, model_type: str, split: str, actual: np.ndarray, pred: np.ndarray, log_actual: np.ndarray, log_pred: np.ndarray) -> dict[str, Any]:
    """Create one model metric row."""
    error = pred - actual
    return {
        "target": target,
        "model_type": model_type,
        "time_split": split,
        "n": int(len(actual)),
        "mae": float(mean_absolute_error(actual, pred)),
        "rmse": float(math.sqrt(mean_squared_error(actual, pred))),
        "median_absolute_error": float(np.median(np.abs(error))),
        "mean_bias": float(np.mean(error)),
        "abs_mean_bias": float(abs(np.mean(error))),
        "r2_log_target": float(r2_score(log_actual, log_pred)) if len(np.unique(log_actual)) > 1 else np.nan,
    }


# 按业务分组生成模型误差记录。
def segment_metric_rows(target: str, model_type: str, split: str, frame: pd.DataFrame, actual: np.ndarray, pred: np.ndarray, log_actual: np.ndarray, log_pred: np.ndarray) -> list[dict[str, Any]]:
    """Create segment-level metric rows."""
    temp = frame.copy()
    temp["_actual"] = actual
    temp["_pred"] = pred
    temp["_log_actual"] = log_actual
    temp["_log_pred"] = log_pred
    rows = []
    segments = {
        "promotion_status": temp["activity_flag"].map({0: "non_promotion", 1: "promotion"}),
        "possible_stockout_status": temp["possible_stockout_flag"].map({0: "not_possible_stockout", 1: "possible_stockout"}),
        "markdown_bin": temp["markdown_bin"],
        "demand_volume_group": temp["demand_volume_group"],
    }
    for seg_name, labels in segments.items():
        temp["_segment"] = labels.astype(str)
        for seg_value, group in temp.groupby("_segment", dropna=False):
            if len(group) < 20:
                continue
            row = model_metric_row(
                target,
                model_type,
                split,
                group["_actual"].to_numpy(),
                group["_pred"].to_numpy(),
                group["_log_actual"].to_numpy(),
                group["_log_pred"].to_numpy(),
            )
            row["segment"] = seg_name
            row["segment_value"] = seg_value
            rows.append(row)
    return rows


# 从验证集选择覆盖主要分组的代表状态。
def representative_states(data: pd.DataFrame, n: int = 120) -> pd.DataFrame:
    """Select representative real test rows for response curves."""
    test = data.loc[data["time_split"].eq("test")].copy()
    test["state_group"] = (
        test["activity_flag"].astype(str)
        + "_"
        + test["stockout_risk_group"].astype(str)
        + "_"
        + test["demand_volume_group"].astype(str)
    )
    parts = []
    for _, group in test.groupby("state_group", dropna=False):
        take = max(1, int(round(len(group) / len(test) * n)))
        parts.append(group.sample(n=min(take, len(group)), random_state=RANDOM_SEED))
    states = pd.concat(parts, ignore_index=True).drop_duplicates(["store_id", "product_id", "timestamp"])
    if len(states) > n:
        states = states.sample(n=n, random_state=RANDOM_SEED)
    return states.reset_index(drop=True)


# 返回响应曲线需要评估的折扣档位。
def markdown_sweep_levels(data: pd.DataFrame) -> list[float]:
    """Return data-supported markdown sweep levels."""
    max_markdown = float(data["markdown_rate"].max())
    candidates = [0.00, 0.05, 0.10, 0.20, 0.30, 0.40]
    return [level for level in candidates if level <= max_markdown + EPSILON]


# 固定状态特征，逐档改变折扣并生成需求预测。
def response_curves(data: pd.DataFrame, fitted: dict[str, FittedResponseModel]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate model-implied markdown response curves."""
    # 在代表性 state 上只改变 markdown level，其余特征保持不变，得到 model-implied
    # counterfactual sweep。它用于环境的需求响应和 action ranking，
    # 不是对真实世界未观察折扣结果的因果识别。
    states = representative_states(data)
    levels = markdown_sweep_levels(data)
    support = support_for_levels(data, levels)
    curve_rows = []
    for target_name, bundle in fitted.items():
        states_encoded = apply_encoders(states, bundle.encoders)
        for state_idx, state in states_encoded.iterrows():
            base_pred = None
            for level in levels:
                tmp = state.to_frame().T.copy()
                tmp["markdown_rate"] = level
                tmp["discount"] = 1.0 - level
                tmp["markdown_x_activity"] = level * float(tmp["activity_flag"].iloc[0])
                pred = float(predict_original_scale(bundle.model, bundle.model_type, tmp, bundle.feature_columns, bundle.prediction_cap)[0])
                if level == 0:
                    base_pred = pred
                if base_pred is None:
                    base_pred = pred
                curve_rows.append(
                    {
                        "state_id": int(state_idx),
                        "target": target_name,
                        "selected_model_type": bundle.model_type,
                        "store_id": state["store_id"],
                        "product_id": state["product_id"],
                        "activity_flag": int(state["activity_flag"]),
                        "possible_stockout_flag": int(state["possible_stockout_flag"]),
                        "stockout_risk_group": state["stockout_risk_group"],
                        "demand_volume_group": state["demand_volume_group"],
                        "markdown_rate": level,
                        "predicted_demand": pred,
                        "absolute_uplift_vs_zero": pred - base_pred,
                        "percentage_uplift_vs_zero": (pred / max(base_pred, EPSILON) - 1.0) * 100.0,
                        "support_status": support[level]["status"],
                    }
                )
    curves = pd.DataFrame(curve_rows)
    support_audit = pd.DataFrame([{"markdown_rate": k, **v} for k, v in support.items()])
    curves.to_csv(CURVES_PATH, index=False)
    support_audit.to_csv(SUPPORT_AUDIT_PATH, index=False)
    return curves, support_audit


# 为每个状态和折扣组合标记样本支持强度。
def support_for_levels(data: pd.DataFrame, levels: list[float]) -> dict[float, dict[str, Any]]:
    """Audit observed support around each markdown sweep level."""
    support = {}
    for level in levels:
        near = data.loc[(data["markdown_rate"] - level).abs().le(0.025)]
        count = len(near)
        if count >= 500:
            status = "supported"
        elif count >= 50:
            status = "weakly_supported"
        else:
            status = "extrapolated"
        support[level] = {
            "nearby_row_count": int(count),
            "nearby_store_count": int(near["store_id"].nunique()),
            "nearby_product_count": int(near["product_id"].nunique()),
            "status": status,
        }
    return support


# 对齐 observed 和 recovered 模型在相同状态动作下的预测。
def compare_targets(curves: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare observed-target and recovered-target response curves."""
    wide = curves.pivot_table(
        index=["state_id", "store_id", "product_id", "activity_flag", "possible_stockout_flag", "stockout_risk_group", "demand_volume_group", "markdown_rate", "support_status"],
        columns="target",
        values=["predicted_demand", "absolute_uplift_vs_zero", "percentage_uplift_vs_zero"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{target}" for metric, target in wide.columns]
    out = wide.reset_index()
    out["absolute_prediction_difference"] = out["predicted_demand_recovered"] - out["predicted_demand_observed"]
    out["percentage_prediction_difference"] = (
        out["predicted_demand_recovered"] / np.maximum(out["predicted_demand_observed"], EPSILON) - 1.0
    ) * 100.0
    out["difference_in_implied_uplift"] = out["absolute_uplift_vs_zero_recovered"] - out["absolute_uplift_vs_zero_observed"]
    out.to_csv(TARGET_COMPARISON_PATH, index=False)

    segment_rows = []
    for col in ["product_id", "store_id", "activity_flag", "stockout_risk_group", "markdown_rate", "demand_volume_group"]:
        grouped = out.groupby(col, dropna=False).agg(
            rows=("state_id", "size"),
            mean_abs_prediction_difference=("absolute_prediction_difference", "mean"),
            mean_pct_prediction_difference=("percentage_prediction_difference", "mean"),
            mean_difference_in_implied_uplift=("difference_in_implied_uplift", "mean"),
        ).reset_index()
        grouped["segment"] = col
        grouped = grouped.rename(columns={col: "segment_value"})
        segment_rows.append(grouped)
    by_segment = pd.concat(segment_rows, ignore_index=True)
    by_segment.to_csv(COMPARISON_BY_SEGMENT_PATH, index=False)
    return out, by_segment


# 计算相邻折扣档位之间的预测需求变化。
def implied_sensitivity(comparison: pd.DataFrame) -> pd.DataFrame:
    """Calculate finite-difference response per 10 percentage-point markdown."""
    rows = []
    for state_id, group in comparison.sort_values("markdown_rate").groupby("state_id"):
        group = group.sort_values("markdown_rate")
        for target in ["observed", "recovered"]:
            demand_col = f"predicted_demand_{target}"
            for i in range(1, len(group)):
                prev = group.iloc[i - 1]
                curr = group.iloc[i]
                delta_markdown = curr["markdown_rate"] - prev["markdown_rate"]
                if delta_markdown <= 0:
                    continue
                pct_change = (curr[demand_col] / max(prev[demand_col], EPSILON) - 1.0) * 100.0
                rows.append(
                    {
                        "state_id": state_id,
                        "target": target,
                        "from_markdown": prev["markdown_rate"],
                        "to_markdown": curr["markdown_rate"],
                        "pct_change_per_10pp_markdown": pct_change / (delta_markdown / 0.10),
                        "product_id": curr["product_id"],
                        "stockout_risk_group": curr["stockout_risk_group"],
                        "activity_flag": curr["activity_flag"],
                    }
                )
    raw = pd.DataFrame(rows)
    summary_rows = []
    for keys, group in raw.groupby(["target"], dropna=False):
        summary_rows.append(sensitivity_row({"target": keys, "scope": "overall", "scope_value": "all"}, group))
    for scope in ["product_id", "stockout_risk_group", "activity_flag"]:
        for keys, group in raw.groupby(["target", scope], dropna=False):
            summary_rows.append(sensitivity_row({"target": keys[0], "scope": scope, "scope_value": keys[1]}, group))
    output = pd.DataFrame(summary_rows)
    output.to_csv(SENSITIVITY_PATH, index=False)
    return output


def sensitivity_row(prefix: dict[str, Any], group: pd.DataFrame) -> dict[str, Any]:
    """Summarize finite-difference sensitivity."""
    values = group["pct_change_per_10pp_markdown"]
    row = dict(prefix)
    row.update(
        {
            "n": int(len(values)),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "p25": float(values.quantile(0.25)),
            "p75": float(values.quantile(0.75)),
        }
    )
    return row


# 汇总两套需求目标在响应曲线上的差异。
def bias_transmission_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    """Classify whether recovery changes markdown response materially."""
    zero = comparison.loc[comparison["markdown_rate"].eq(0)]
    positive = comparison.loc[comparison["markdown_rate"].gt(0)]
    high_stockout = comparison.loc[comparison["stockout_risk_group"].eq("high")]
    promo = comparison.loc[comparison["activity_flag"].eq(1)]
    mean_abs_diff = float(comparison["absolute_prediction_difference"].abs().mean())
    mean_pct_uplift_diff = float(
        (positive["difference_in_implied_uplift"].abs() / np.maximum(positive["predicted_demand_observed"], EPSILON)).mean() * 100.0
    )
    ranking_changes = action_ranking(comparison)
    changed_share = float(ranking_changes["preferred_action_changed"].mean())
    if mean_abs_diff >= 0.25 or mean_pct_uplift_diff >= 10 or changed_share >= 0.25:
        classification = "MATERIAL_RESPONSE_CHANGE"
    elif mean_abs_diff >= 0.10 or mean_pct_uplift_diff >= 5 or changed_share >= 0.10:
        classification = "MODERATE_RESPONSE_CHANGE"
    else:
        classification = "LIMITED_RESPONSE_CHANGE"
    rows = [
        {"metric": "zero_markdown_baseline_difference", "value": float(zero["absolute_prediction_difference"].mean())},
        {"metric": "positive_markdown_prediction_difference", "value": float(positive["absolute_prediction_difference"].mean())},
        {"metric": "average_implied_uplift_difference", "value": float(positive["difference_in_implied_uplift"].mean())},
        {"metric": "high_stockout_risk_prediction_difference", "value": float(high_stockout["absolute_prediction_difference"].mean())},
        {"metric": "promotion_period_prediction_difference", "value": float(promo["absolute_prediction_difference"].mean())},
        {"metric": "share_recovered_stronger_markdown_response", "value": float((positive["difference_in_implied_uplift"] > 0).mean())},
        {
            "metric": "share_higher_baseline_similar_sensitivity",
            "value": float(
                (
                    (zero["absolute_prediction_difference"] > 0)
                    & (zero["state_id"].isin(
                        positive.groupby("state_id")["difference_in_implied_uplift"].mean().abs().loc[lambda s: s < 0.05].index
                    ))
                ).mean()
            ),
        },
        {"metric": "mean_absolute_demand_prediction_difference", "value": mean_abs_diff},
        {"metric": "mean_absolute_percentage_uplift_difference", "value": mean_pct_uplift_diff},
        {"metric": "preferred_action_changed_share", "value": changed_share},
        {"metric": "stockout_bias_transmission_classification", "value": classification},
    ]
    output = pd.DataFrame(rows)
    output.to_csv(BIAS_TRANSMISSION_PATH, index=False)
    return output


# 按每个状态下的预测需求对折扣动作排序。
def action_ranking(comparison: pd.DataFrame) -> pd.DataFrame:
    """Compare illustrative revenue rankings under normalized price 1.0."""
    rows = []
    for state_id, group in comparison.groupby("state_id"):
        temp = group.copy()
        temp["illustrative_revenue_observed"] = (1 - temp["markdown_rate"]) * temp["predicted_demand_observed"]
        temp["illustrative_revenue_recovered"] = (1 - temp["markdown_rate"]) * temp["predicted_demand_recovered"]
        best_obs = temp.loc[temp["illustrative_revenue_observed"].idxmax()]
        best_rec = temp.loc[temp["illustrative_revenue_recovered"].idxmax()]
        rows.append(
            {
                "state_id": state_id,
                "store_id": best_obs["store_id"],
                "product_id": best_obs["product_id"],
                "observed_revenue_maximizing_markdown": float(best_obs["markdown_rate"]),
                "recovered_revenue_maximizing_markdown": float(best_rec["markdown_rate"]),
                "preferred_action_changed": bool(best_obs["markdown_rate"] != best_rec["markdown_rate"]),
                "observed_best_illustrative_revenue": float(best_obs["illustrative_revenue_observed"]),
                "recovered_best_illustrative_revenue": float(best_rec["illustrative_revenue_recovered"]),
            }
        )
    output = pd.DataFrame(rows)
    output.to_csv(ACTION_RANKING_PATH, index=False)
    return output


# 在不同数据子集上重复拟合并比较响应结果。
def robustness_checks(data: pd.DataFrame, metrics: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    """Run concise robustness diagnostics."""
    rows = []
    checks = {
        "exclude_recovery_clipped_flag": data.loc[~data["recovery_clipped_flag"]],
        "exclude_recovery_extrapolation_flag": data.loc[~data["recovery_extrapolation_flag"]],
        "strict_stockout_definition": data.loc[data["stock_hour6_22_cnt"].gt(0) | data["possible_stockout_flag"]],
        "moderate_stockout_definition": data,
        "multiple_markdown_levels_series": data.groupby(["store_id", "sku_id"]).filter(lambda g: g["markdown_rate"].nunique() >= 3),
        "support_in_promo_and_nonpromo": data.groupby(["store_id", "sku_id"]).filter(lambda g: g["activity_flag"].nunique() >= 2),
    }
    base_diff = float(comparison["absolute_prediction_difference"].mean())
    for name, frame in checks.items():
        rows.append(
            {
                "check": name,
                "rows": int(len(frame)),
                "series": int(frame[["store_id", "sku_id"]].drop_duplicates().shape[0]) if not frame.empty else 0,
                "mean_markdown": float(frame["markdown_rate"].mean()) if not frame.empty else np.nan,
                "observed_mean": float(frame["observed_sales_demand"].mean()) if not frame.empty else np.nan,
                "recovered_mean": float(frame["recovered_demand"].mean()) if not frame.empty else np.nan,
                "mean_recovered_minus_observed": float((frame["recovered_demand"] - frame["observed_sales_demand"]).mean()) if not frame.empty else np.nan,
                "reference_response_difference": base_diff,
            }
        )
    for model_type in metrics["model_type"].unique():
        subset = metrics.loc[metrics["model_type"].eq(model_type) & metrics["time_split"].eq("validation")]
        rows.append(
            {
                "check": f"model_type_{model_type}",
                "rows": int(subset["n"].sum()),
                "series": np.nan,
                "mean_markdown": np.nan,
                "observed_mean": float(subset.loc[subset["target"].eq("observed"), "mae"].mean()),
                "recovered_mean": float(subset.loc[subset["target"].eq("recovered"), "mae"].mean()),
                "mean_recovered_minus_observed": np.nan,
                "reference_response_difference": base_diff,
            }
        )
    output = pd.DataFrame(rows)
    output.to_csv(ROBUSTNESS_PATH, index=False)
    return output


# 保存两套模型、特征定义、encoder 和预测范围。
def save_models_and_config(fitted: dict[str, FittedResponseModel], data: pd.DataFrame, support_audit: pd.DataFrame) -> None:
    """Save selected observed/recovered models and metadata config."""
    joblib.dump(fitted["observed"], OBSERVED_MODEL_PATH)
    joblib.dump(fitted["recovered"], RECOVERED_MODEL_PATH)
    config = {
        "target_definitions": TARGETS,
        "feature_list": feature_columns(),
        "preprocessing_methods": "Stable ordinal encoding for store_id/product_id fitted on train split only; numeric imputation inside Ridge pipeline.",
        "selected_model_types": {target: bundle.model_type for target, bundle in fitted.items()},
        "chronological_split_boundaries": split_boundaries(data),
        "markdown_support_range": {
            "min": float(data["markdown_rate"].min()),
            "max": float(data["markdown_rate"].max()),
        },
        "prediction_clipping_rules": {
            target: f"clip expm1(log prediction) to [0, {bundle.prediction_cap * 1.5:.6f}]"
            for target, bundle in fitted.items()
        },
        "support_audit": support_audit.to_dict(orient="records"),
        "uncertainty_and_limitations": limitations(),
        "no_causal_claim_statement": "Model-implied markdown response curves are observational diagnostics, not causal counterfactuals.",
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")


# 汇总 train、validation 和 test 的日期边界。
def split_boundaries(data: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Return chronological split boundaries."""
    return {
        split: {
            "date_min": str(group["timestamp"].min().date()),
            "date_max": str(group["timestamp"].max().date()),
            "rows": int(len(group)),
        }
        for split, group in data.groupby("time_split", sort=False)
    }


# 返回该模块对应的数据和模型限制说明。
def limitations() -> list[str]:
    """Required limitations."""
    return [
        "Markdown effects are observational and confounded.",
        "Recovered demand is model-estimated.",
        "Actual latent demand during real stockouts remains unobserved.",
        "Discount semantics remain uncertain_multiplier_like.",
        "stock_hour6_22_cnt semantics remain partly uncertain.",
        "Model-implied response curves are not causal counterfactuals.",
        "Results are suitable for simulation calibration and sensitivity analysis, not direct operational pricing claims.",
    ]


# 根据当前参数构建 figures。
def create_figures(
    data: pd.DataFrame,
    descriptive: pd.DataFrame,
    curves: pd.DataFrame,
    comparison: pd.DataFrame,
    metrics: pd.DataFrame,
    robustness: pd.DataFrame,
    action: pd.DataFrame,
) -> None:
    """Create required matplotlib figures."""
    plot_demand_by_markdown(descriptive)
    plot_markdown_support(data)
    plot_response_curves(curves)
    plot_observed_vs_recovered_predictions(comparison)
    plot_uplift_by_markdown(comparison)
    plot_difference_by_segment(comparison, "stockout_risk_group", "response_difference_by_stockout_risk_group.png")
    plot_difference_by_segment(comparison, "activity_flag", "response_difference_by_promotion_status.png")
    plot_action_ranking(action)
    plot_model_performance(metrics)
    plot_robustness(robustness)


# 根据现有表格绘制 demand by markdown 图。
def plot_demand_by_markdown(descriptive: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for target, group in descriptive.groupby("target"):
        ax.plot(group["markdown_bin"], group["mean"], marker="o", label=target)
    ax.set_title("Observed and recovered demand by markdown bin")
    ax.set_xlabel("Markdown bin")
    ax.set_ylabel("Mean demand")
    ax.legend()
    plt.xticks(rotation=25, ha="right")
    savefig("demand_by_markdown_bin.png")


# 根据现有表格绘制 markdown support 图。
def plot_markdown_support(data: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(data["markdown_rate"], bins=40)
    ax.set_title("Markdown distribution and sample support")
    ax.set_xlabel("Markdown rate")
    ax.set_ylabel("Rows")
    savefig("markdown_distribution_support.png")


# 根据现有表格绘制 response curves 图。
def plot_response_curves(curves: pd.DataFrame) -> None:
    summary = curves.groupby(["target", "markdown_rate"])["predicted_demand"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    for target, group in summary.groupby("target"):
        ax.plot(group["markdown_rate"], group["predicted_demand"], marker="o", label=target)
    ax.set_title("Model-implied markdown-response curves")
    ax.set_xlabel("Markdown rate")
    ax.set_ylabel("Predicted demand")
    ax.legend()
    savefig("model_implied_markdown_response_curves.png")


# 根据现有表格绘制 observed vs recovered predictions 图。
def plot_observed_vs_recovered_predictions(comparison: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(comparison["predicted_demand_observed"], comparison["predicted_demand_recovered"], s=10, alpha=0.35)
    max_v = float(max(comparison["predicted_demand_observed"].max(), comparison["predicted_demand_recovered"].max()))
    ax.plot([0, max_v], [0, max_v], color="black", linewidth=1)
    ax.set_title("Observed-target vs recovered-target predicted demand")
    ax.set_xlabel("Observed-target prediction")
    ax.set_ylabel("Recovered-target prediction")
    savefig("observed_vs_recovered_predicted_demand.png")


# 根据现有表格绘制 uplift by markdown 图。
def plot_uplift_by_markdown(comparison: pd.DataFrame) -> None:
    summary = comparison.groupby("markdown_rate").agg(
        observed_uplift=("absolute_uplift_vs_zero_observed", "mean"),
        recovered_uplift=("absolute_uplift_vs_zero_recovered", "mean"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(summary["markdown_rate"], summary["observed_uplift"], marker="o", label="observed")
    ax.plot(summary["markdown_rate"], summary["recovered_uplift"], marker="o", label="recovered")
    ax.set_title("Implied demand uplift by markdown")
    ax.set_xlabel("Markdown rate")
    ax.set_ylabel("Mean uplift vs zero markdown")
    ax.legend()
    savefig("implied_demand_uplift_by_markdown.png")


# 根据现有表格绘制 difference by segment 图。
def plot_difference_by_segment(comparison: pd.DataFrame, segment: str, filename: str) -> None:
    summary = comparison.groupby(segment)["difference_in_implied_uplift"].mean().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    summary.plot(kind="bar", ax=ax)
    ax.set_title(f"Response difference by {segment}")
    ax.set_ylabel("Recovered uplift - observed uplift")
    plt.xticks(rotation=25, ha="right")
    savefig(filename)


# 根据现有表格绘制 action ranking 图。
def plot_action_ranking(action: pd.DataFrame) -> None:
    counts = action.groupby(["observed_revenue_maximizing_markdown", "recovered_revenue_maximizing_markdown"]).size().reset_index(name="count")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(counts["observed_revenue_maximizing_markdown"], counts["recovered_revenue_maximizing_markdown"], s=counts["count"] * 8, alpha=0.6)
    ax.set_title("Action-ranking comparison")
    ax.set_xlabel("Observed-target best markdown")
    ax.set_ylabel("Recovered-target best markdown")
    savefig("action_ranking_comparison.png")


# 根据现有表格绘制 model performance 图。
def plot_model_performance(metrics: pd.DataFrame) -> None:
    validation = metrics.loc[metrics["time_split"].eq("validation")]
    labels = validation["target"] + "_" + validation["model_type"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, validation["mae"])
    ax.set_title("Model performance by target")
    ax.set_ylabel("Validation MAE")
    plt.xticks(rotation=30, ha="right")
    savefig("model_performance_by_target.png")


# 根据现有表格绘制 robustness 图。
def plot_robustness(robustness: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    values = robustness["mean_recovered_minus_observed"].fillna(0)
    ax.bar(robustness["check"], values)
    ax.set_title("Robustness comparison")
    ax.set_ylabel("Mean recovered - observed")
    plt.xticks(rotation=35, ha="right")
    savefig("robustness_comparison.png")


# 调整布局并把当前图保存到主图目录。
def savefig(filename: str) -> None:
    """Save current figure."""
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=160)
    plt.close()


# 根据模型误差、支持度和输入检查生成状态。
def final_status(
    fitted: dict[str, FittedResponseModel],
    metrics: pd.DataFrame,
    curves: pd.DataFrame,
    support_audit: pd.DataFrame,
    comparison: pd.DataFrame,
) -> str:
    """Return final readiness status."""
    required_files = [
        OBSERVED_MODEL_PATH,
        RECOVERED_MODEL_PATH,
        CONFIG_PATH,
        CURVES_PATH,
        SUPPORT_AUDIT_PATH,
        TARGET_COMPARISON_PATH,
        ACTION_RANKING_PATH,
    ]
    has_targets = set(fitted) == {"observed", "recovered"}
    has_validation_test = {"validation", "test"}.issubset(set(metrics["time_split"]))
    has_curves = not curves.empty
    has_support = not support_audit.empty
    has_comparison = not comparison.empty
    saved = all(path.exists() for path in required_files)
    if has_targets and has_validation_test and has_curves and has_support and has_comparison and saved:
        return "DISCOUNT_RESPONSE_CALIBRATION_READY"
    return "DISCOUNT_RESPONSE_CALIBRATION_REQUIRES_REVISION"


# 打印模型选择、支持范围和输出文件。
def print_report(
    status: str,
    fitted: dict[str, FittedResponseModel],
    metrics: pd.DataFrame,
    data: pd.DataFrame,
    comparison: pd.DataFrame,
    bias_summary: pd.DataFrame,
    support_audit: pd.DataFrame,
    action: pd.DataFrame,
) -> None:
    """Print final console report."""
    print("\nDiscount response calibration report")
    print("Selected models:", {k: v.model_type for k, v in fitted.items()})
    print("Validation metrics:")
    print(metrics.loc[metrics["time_split"].eq("validation"), ["target", "model_type", "mae", "rmse", "mean_bias", "r2_log_target"]].to_string(index=False))
    print("Test metrics:")
    print(metrics.loc[metrics["time_split"].eq("test"), ["target", "model_type", "mae", "rmse", "mean_bias", "r2_log_target"]].to_string(index=False))
    print(f"Markdown support range: {data['markdown_rate'].min():.4f} to {data['markdown_rate'].max():.4f}")
    positive = comparison.loc[comparison["markdown_rate"].gt(0)]
    print(f"Mean model-implied observed uplift: {positive['absolute_uplift_vs_zero_observed'].mean():.4f}")
    print(f"Mean model-implied recovered uplift: {positive['absolute_uplift_vs_zero_recovered'].mean():.4f}")
    print(f"Mean recovered-observed prediction difference: {comparison['absolute_prediction_difference'].mean():.4f}")
    classification = bias_summary.loc[bias_summary["metric"].eq("stockout_bias_transmission_classification"), "value"].iloc[0]
    print(f"Stockout-bias transmission classification: {classification}")
    print(f"Percentage of extrapolated response points: {(support_audit['status'].eq('extrapolated').mean() * 100):.2f}%")
    print(f"Action rankings changed in {action['preferred_action_changed'].mean() * 100:.2f}% of representative states.")
    print("Major limitations:")
    for item in limitations():
        print(f"- {item}")
    print("Recommended next module: build a pricing-environment adapter using recovered demand and calibrated response diagnostics.")
    print(status)


def run() -> str:
    """Run full discount-response calibration workflow."""
    # pipeline：输入验证 -> support/confounding audit -> 两套 target 建模
    # -> markdown sweep -> observed/recovered 比较 -> 保存模型和配置。
    # 输出 joblib 随后由 OperationalPerishablePricingEnv 加载。
    ensure_dirs()
    data = load_and_validate()
    data = engineer_features(data)
    bins = markdown_bins(data)
    data = add_markdown_bin(data, bins)
    support = audit_markdown_support(data)
    descriptive, correlations, by_product, by_store = descriptive_analysis(data)
    confounding_audit(data)
    fitted, metrics, segment_metrics = fit_response_models(data)
    curves, support_audit = response_curves(apply_encoders(data, next(iter(fitted.values())).encoders), fitted)
    comparison, comparison_by_segment = compare_targets(curves)
    implied_sensitivity(comparison)
    action = action_ranking(comparison)
    bias_summary = bias_transmission_summary(comparison)
    robustness = robustness_checks(data, metrics, comparison)
    save_models_and_config(fitted, data, support_audit)
    create_figures(data, descriptive, curves, comparison, metrics, robustness, action)
    status = final_status(fitted, metrics, curves, support_audit, comparison)
    print_report(status, fitted, metrics, data, comparison, bias_summary, support_audit, action)
    return status


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"Discount response calibration failed: {exc}")
        print("DISCOUNT_RESPONSE_CALIBRATION_REQUIRES_REVISION")
        raise
