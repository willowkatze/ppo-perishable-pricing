"""Stockout-aware latent-demand recovery for FreshRetailNet modeling subset.

This module is intentionally simple and explainable. It validates the existing
modeling subset, builds leakage-safe time-series features, evaluates recovery
models with artificial censoring, selects a model using validation data only,
and saves a recovered-demand dataset for later demand modeling and RL work.
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


RANDOM_SEED = 42
EPSILON = 1e-6
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_ROOT / "data" / "freshretail" / "processed" / "freshretail_modeling_subset.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "freshretail" / "processed" / "freshretail_demand_recovered.parquet"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "latent_demand_recovery"
MODEL_PATH = MODELS_DIR / "freshretail_latent_demand_model.joblib"
METADATA_PATH = TABLES_DIR / "latent_demand_recovery_metadata.json"

LABEL_COUNTS_PATH = TABLES_DIR / "latent_demand_censoring_labels.csv"
VALIDATION_METRICS_PATH = TABLES_DIR / "latent_demand_validation_metrics.csv"
VALIDATION_PREDICTIONS_PATH = TABLES_DIR / "latent_demand_validation_predictions.csv"
VALIDATION_BY_SEGMENT_PATH = TABLES_DIR / "latent_demand_validation_by_segment.csv"
MODEL_SELECTION_PATH = TABLES_DIR / "latent_demand_model_selection.csv"
RECOVERY_SUMMARY_PATH = TABLES_DIR / "latent_demand_recovery_summary.csv"
RECOVERY_BY_PROMOTION_PATH = TABLES_DIR / "latent_demand_recovery_by_promotion.csv"
RECOVERY_BY_MARKDOWN_PATH = TABLES_DIR / "latent_demand_recovery_by_markdown.csv"
RECOVERY_BY_PRODUCT_PATH = TABLES_DIR / "latent_demand_recovery_by_product.csv"
RECOVERY_BY_STORE_PATH = TABLES_DIR / "latent_demand_recovery_by_store.csv"
PROMOTION_STOCKOUT_PATH = TABLES_DIR / "latent_demand_promotion_stockout_interaction.csv"
ROBUSTNESS_PATH = TABLES_DIR / "latent_demand_recovery_robustness.csv"

EXPECTED_ROWS = 29_100
EXPECTED_SERIES = 300
EXPECTED_DAYS = 97
EXPECTED_DATE_MIN = "2024-03-28"
EXPECTED_DATE_MAX = "2024-07-02"


@dataclass
class FittedModel:
    """Container for a fitted recovery model."""

    name: str
    model: Any
    feature_columns: list[str]
    kind: str


class SeasonalHistoricalBaseline:
    """Transparent hierarchical historical-mean demand baseline."""

    def __init__(self) -> None:
        self.global_mean_: float = 0.0
        self.series_dow_: dict[tuple[str, str, int], float] = {}
        self.series_: dict[tuple[str, str], float] = {}
        self.sku_: dict[str, float] = {}

    def fit(self, frame: pd.DataFrame, target_col: str = "observed_sales_demand") -> "SeasonalHistoricalBaseline":
        clean = frame.dropna(subset=[target_col]).copy()
        self.global_mean_ = float(clean[target_col].mean()) if not clean.empty else 0.0
        self.series_dow_ = (
            clean.groupby(["store_id", "sku_id", "day_of_week"], dropna=False)[target_col].mean().to_dict()
        )
        self.series_ = clean.groupby(["store_id", "sku_id"], dropna=False)[target_col].mean().to_dict()
        self.sku_ = clean.groupby("sku_id", dropna=False)[target_col].mean().to_dict()
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        preds: list[float] = []
        for row in frame[["store_id", "sku_id", "day_of_week"]].itertuples(index=False):
            store_id = str(row.store_id)
            sku_id = str(row.sku_id)
            dow = int(row.day_of_week)
            value = self.series_dow_.get((store_id, sku_id, dow))
            if value is None:
                value = self.series_.get((store_id, sku_id))
            if value is None:
                value = self.sku_.get(sku_id)
            if value is None:
                value = self.global_mean_
            preds.append(float(value))
        return np.asarray(preds, dtype=float)


def ensure_dirs() -> None:
    """Create output directories."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_and_validate_subset() -> pd.DataFrame:
    """Load and validate the modeling subset without modifying it."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Modeling subset not found: {INPUT_PATH}")
    data = pd.read_parquet(INPUT_PATH)
    data = normalize_input_columns(data)
    required = {
        "timestamp",
        "store_id",
        "sku_id",
        "observed_sales_demand",
        "stock_hour6_22_cnt",
        "discount",
        "markdown_rate",
        "possible_stockout_flag",
        "time_split",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data["store_id"] = data["store_id"].astype(str)
    data["sku_id"] = data["sku_id"].astype(str)
    data["observed_sales_demand"] = pd.to_numeric(data["observed_sales_demand"], errors="coerce")
    data["stock_hour6_22_cnt"] = pd.to_numeric(data["stock_hour6_22_cnt"], errors="coerce")
    data["discount"] = pd.to_numeric(data["discount"], errors="coerce")
    data["markdown_rate"] = pd.to_numeric(data["markdown_rate"], errors="coerce")
    data["possible_stockout_flag"] = as_bool(data["possible_stockout_flag"])
    if "promotion" in data.columns:
        data["activity_flag"] = as_bool(data["promotion"])
    elif "activity_flag" in data.columns:
        data["activity_flag"] = as_bool(data["activity_flag"])
    else:
        data["activity_flag"] = data["markdown_rate"].fillna(0.0) > 0
    if "avg_temperature" not in data.columns:
        data["avg_temperature"] = pd.to_numeric(data.get("weather", np.nan), errors="coerce")

    duplicate_count = int(data.duplicated(["store_id", "sku_id", "timestamp"]).sum())
    series_lengths = data.groupby(["store_id", "sku_id"], dropna=False).size()
    interval_bad = int((data.sort_values(["store_id", "sku_id", "timestamp"]).groupby(["store_id", "sku_id"])["timestamp"].diff().dropna().dt.days != 1).sum())
    split_values = set(data["time_split"].dropna().astype(str).unique())
    checks = {
        "rows": len(data),
        "series": len(series_lengths),
        "sequence_length_min": int(series_lengths.min()),
        "sequence_length_max": int(series_lengths.max()),
        "date_min": str(data["timestamp"].min().date()),
        "date_max": str(data["timestamp"].max().date()),
        "duplicate_store_sku_date_rows": duplicate_count,
        "time_continuity_violations": interval_bad,
        "time_split_labels": ",".join(sorted(split_values)),
    }
    if checks["series"] != EXPECTED_SERIES:
        warnings.warn(f"Expected {EXPECTED_SERIES} series, found {checks['series']}.")
    if checks["sequence_length_min"] < EXPECTED_DAYS or checks["sequence_length_max"] != EXPECTED_DAYS:
        warnings.warn(f"Expected complete {EXPECTED_DAYS}-day sequences, found {checks['sequence_length_min']}..{checks['sequence_length_max']}.")
    if checks["date_min"] != EXPECTED_DATE_MIN or checks["date_max"] != EXPECTED_DATE_MAX:
        warnings.warn(f"Unexpected date range: {checks['date_min']} to {checks['date_max']}.")
    if duplicate_count:
        raise ValueError("Duplicate store-product-date rows detected.")
    if split_values != {"train", "validation", "test"}:
        raise ValueError(f"Invalid time_split labels: {split_values}")
    if interval_bad:
        raise ValueError("Temporal continuity violations detected.")
    data.attrs["validation_checks"] = checks
    return data.sort_values(["store_id", "sku_id", "timestamp"]).reset_index(drop=True)


def normalize_input_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize aliases while preserving original modeling-subset columns."""
    output = data.copy()
    if "observed_sales_demand" not in output.columns:
        if "sale_amount" in output.columns:
            output["observed_sales_demand"] = output["sale_amount"]
        elif "sales_qty" in output.columns:
            output["observed_sales_demand"] = output["sales_qty"]
        else:
            raise ValueError("No sale_amount, sales_qty, or observed_sales_demand column found.")
    if "product_id" not in output.columns and "sku_id" in output.columns:
        output["product_id"] = output["sku_id"]
    if "sku_id" not in output.columns and "product_id" in output.columns:
        output["sku_id"] = output["product_id"]
    if "stock_hour6_22_cnt" not in output.columns and "inventory" in output.columns:
        output["stock_hour6_22_cnt"] = output["inventory"]
    if "possible_stockout_flag" not in output.columns:
        if "stockout" in output.columns:
            output["possible_stockout_flag"] = output["stockout"]
        elif "zero_inventory_flag" in output.columns:
            output["possible_stockout_flag"] = output["zero_inventory_flag"]
        else:
            output["possible_stockout_flag"] = output["stock_hour6_22_cnt"].fillna(np.inf) <= 0
    return output


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


def create_censoring_labels(data: pd.DataFrame) -> pd.DataFrame:
    """Create transparent censoring and training-eligibility labels."""
    output = data.copy()
    output["ambiguous_inventory_state"] = output["stock_hour6_22_cnt"].isna()
    output["possible_stockout"] = output["possible_stockout_flag"].astype(bool)
    output["confirmed_or_likely_uncensored"] = (
        ~output["possible_stockout"]
        & ~output["ambiguous_inventory_state"]
        & output["observed_sales_demand"].ge(0)
    )
    output["uncensored_training_flag"] = (
        output["confirmed_or_likely_uncensored"]
        & output["time_split"].isin(["train", "validation", "test"])
    )
    rows = []
    for label in [
        "confirmed_or_likely_uncensored",
        "possible_stockout",
        "ambiguous_inventory_state",
        "uncensored_training_flag",
    ]:
        rows.append({"label": label, "count": int(output[label].sum()), "share": float(output[label].mean())})
    rows.extend(
        [
            {"label": "rule_observed_sales_demand", "count": len(output), "share": 1.0, "rule": "observed_sales_demand equals original sale_amount/sales_qty."},
            {"label": "rule_uncensored_training_flag", "count": len(output), "share": 1.0, "rule": "possible_stockout_flag is false, sales is non-negative, and inventory value is not missing."},
            {"label": "rule_possible_stockout", "count": len(output), "share": 1.0, "rule": "possible_stockout_flag from diagnostic zero-inventory/stockout definition; not latent-demand truth."},
        ]
    )
    pd.DataFrame(rows).to_csv(LABEL_COUNTS_PATH, index=False)
    return output


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe features using only current or previous-period data."""
    output = data.sort_values(["store_id", "sku_id", "timestamp"]).copy()
    output["day_of_week"] = output["timestamp"].dt.dayofweek
    output["day_of_month"] = output["timestamp"].dt.day
    output["week_index"] = ((output["timestamp"] - output["timestamp"].min()).dt.days // 7).astype(int)
    output["month"] = output["timestamp"].dt.month
    output["weekend_flag"] = output["day_of_week"].isin([5, 6]).astype(int)
    output["trend_index"] = (output["timestamp"] - output["timestamp"].min()).dt.days.astype(int)
    output["activity_flag"] = as_bool(output["activity_flag"]).astype(int)
    output["previous_day_possible_stockout_flag"] = (
        output.groupby(["store_id", "sku_id"])["possible_stockout_flag"]
        .shift(1)
        .map(lambda value: bool(value) if pd.notna(value) else False)
        .astype(int)
    )
    output["previous_day_inventory"] = output.groupby(["store_id", "sku_id"])["stock_hour6_22_cnt"].shift(1)
    grouped_sales = output.groupby(["store_id", "sku_id"])["observed_sales_demand"]
    output["lag_1_observed_sales"] = grouped_sales.shift(1)
    output["lag_7_observed_sales"] = grouped_sales.shift(7)
    shifted = grouped_sales.shift(1)
    output["rolling_mean_7_observed_sales"] = shifted.groupby([output["store_id"], output["sku_id"]]).rolling(7, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    output["rolling_median_7_observed_sales"] = shifted.groupby([output["store_id"], output["sku_id"]]).rolling(7, min_periods=1).median().reset_index(level=[0, 1], drop=True)
    output["rolling_std_7_observed_sales"] = shifted.groupby([output["store_id"], output["sku_id"]]).rolling(7, min_periods=2).std().reset_index(level=[0, 1], drop=True)
    output["rolling_mean_14_observed_sales"] = shifted.groupby([output["store_id"], output["sku_id"]]).rolling(14, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    for column in feature_columns():
        if column in output.columns and pd.api.types.is_numeric_dtype(output[column]):
            output[column] = output[column].replace([np.inf, -np.inf], np.nan)
    return output


def feature_columns() -> list[str]:
    """Numeric model feature columns."""
    return [
        "day_of_week",
        "day_of_month",
        "week_index",
        "month",
        "weekend_flag",
        "trend_index",
        "store_id_encoded",
        "product_id_encoded",
        "lag_1_observed_sales",
        "lag_7_observed_sales",
        "rolling_mean_7_observed_sales",
        "rolling_median_7_observed_sales",
        "rolling_std_7_observed_sales",
        "rolling_mean_14_observed_sales",
        "activity_flag",
        "markdown_rate",
        "discount",
        "avg_temperature",
        "stock_hour6_22_cnt",
        "previous_day_inventory",
        "previous_day_possible_stockout_flag",
    ]


def fit_encoders(train: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Fit stable ordinal encoders on training data only."""
    encoders = {}
    for source, _target in [("store_id", "store_id_encoded"), ("sku_id", "product_id_encoded")]:
        values = sorted(train[source].astype(str).unique().tolist())
        encoders[source] = {value: index + 1 for index, value in enumerate(values)}
    return encoders


def apply_encoders(data: pd.DataFrame, encoders: dict[str, dict[str, int]]) -> pd.DataFrame:
    """Apply fitted encoders; unknown categories map to zero."""
    output = data.copy()
    output["store_id_encoded"] = output["store_id"].astype(str).map(encoders["store_id"]).fillna(0).astype(int)
    output["product_id_encoded"] = output["sku_id"].astype(str).map(encoders["sku_id"]).fillna(0).astype(int)
    return output


def prepare_model_matrix(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return numeric feature matrix with simple train-safe missing handling."""
    frame = data.reindex(columns=columns).copy()
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        median = frame[column].median()
        frame[column] = frame[column].fillna(0.0 if pd.isna(median) else median)
    return frame


def train_models(features: pd.DataFrame, train_mask: pd.Series) -> dict[str, FittedModel]:
    """Train simple benchmark recovery models on likely uncensored train rows."""
    train = features.loc[train_mask].copy()
    columns = feature_columns()
    x_train = prepare_model_matrix(train, columns)
    y_train = train["observed_sales_demand"].astype(float)
    models: dict[str, FittedModel] = {}

    baseline = SeasonalHistoricalBaseline().fit(train)
    models["seasonal_baseline"] = FittedModel("seasonal_baseline", baseline, columns, "baseline")

    hgb = HistGradientBoostingRegressor(
        max_iter=180,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=RANDOM_SEED,
    )
    hgb.fit(x_train, y_train)
    models["hist_gradient_boosting"] = FittedModel("hist_gradient_boosting", hgb, columns, "sklearn")

    if len(train) > 12_000:
        train_sample = train.sample(n=12_000, random_state=RANDOM_SEED)
    else:
        train_sample = train
    forest = ExtraTreesRegressor(
        n_estimators=120,
        max_depth=12,
        min_samples_leaf=8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    forest.fit(prepare_model_matrix(train_sample, columns), train_sample["observed_sales_demand"].astype(float))
    models["extra_trees"] = FittedModel("extra_trees", forest, columns, "sklearn")
    return models


def predict_model(fitted: FittedModel, frame: pd.DataFrame) -> np.ndarray:
    """Predict non-negative demand from a fitted model."""
    if fitted.kind == "baseline":
        pred = fitted.model.predict(frame)
    else:
        pred = fitted.model.predict(prepare_model_matrix(frame, fitted.feature_columns))
    return np.maximum(np.asarray(pred, dtype=float), 0.0)


def artificial_censoring_validation(features: pd.DataFrame, models: dict[str, FittedModel]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate models with pseudo-stockout artificial censoring."""
    frames = []
    rng = np.random.default_rng(RANDOM_SEED)
    for split in ["validation", "test"]:
        base = features.loc[
            features["time_split"].eq(split)
            & features["confirmed_or_likely_uncensored"]
            & features["observed_sales_demand"].gt(0)
        ].copy()
        if base.empty:
            continue
        sample_size = min(3_000, len(base))
        base = base.sample(n=sample_size, random_state=RANDOM_SEED)
        caps = [
            ("fixed_cap_q25", float(base["observed_sales_demand"].quantile(0.25))),
            ("fixed_cap_q50", float(base["observed_sales_demand"].quantile(0.50))),
        ]
        scenarios: list[pd.DataFrame] = []
        for severity, cap in caps:
            tmp = base.copy()
            tmp["censoring_mechanism"] = "fixed_cap"
            tmp["censoring_severity"] = severity
            tmp["pseudo_observed_sales"] = np.minimum(tmp["observed_sales_demand"], cap)
            scenarios.append(tmp)
        for fraction in [0.25, 0.50, 0.75]:
            tmp = base.copy()
            tmp["censoring_mechanism"] = "proportional"
            tmp["censoring_severity"] = f"fraction_{fraction:.2f}"
            tmp["pseudo_observed_sales"] = tmp["observed_sales_demand"] * fraction
            scenarios.append(tmp)
        scenario_frame = pd.concat(scenarios, ignore_index=True)
        scenario_frame["pseudo_row_id"] = np.arange(len(scenario_frame)) + rng.integers(0, 1_000_000)
        frames.append(scenario_frame)

    if not frames:
        raise ValueError("No uncensored validation/test rows available for artificial censoring.")
    scenarios_all = pd.concat(frames, ignore_index=True)
    prediction_rows = []
    for model_name, fitted in models.items():
        pred = predict_model(fitted, scenarios_all)
        recovered = np.maximum(scenarios_all["pseudo_observed_sales"].to_numpy(dtype=float), pred)
        tmp = scenarios_all[
            [
                "time_split",
                "store_id",
                "sku_id",
                "timestamp",
                "censoring_mechanism",
                "censoring_severity",
                "pseudo_observed_sales",
                "observed_sales_demand",
                "activity_flag",
                "markdown_rate",
            ]
        ].copy()
        tmp["model"] = model_name
        tmp["true_demand"] = tmp["observed_sales_demand"]
        tmp["predicted_demand"] = pred
        tmp["recovered_pseudo_demand"] = recovered
        tmp["error"] = tmp["recovered_pseudo_demand"] - tmp["true_demand"]
        prediction_rows.append(tmp)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictions["markdown_group"] = markdown_group(predictions["markdown_rate"])
    predictions["sales_volume_group"] = quantile_label(predictions["true_demand"], "low", "medium", "high")
    metrics = calculate_validation_metrics(predictions)
    by_segment = calculate_segment_metrics(predictions)
    metrics.to_csv(VALIDATION_METRICS_PATH, index=False)
    predictions.to_csv(VALIDATION_PREDICTIONS_PATH, index=False)
    by_segment.to_csv(VALIDATION_BY_SEGMENT_PATH, index=False)
    return predictions, metrics, by_segment


def calculate_validation_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate artificial-censoring metrics by model/scenario/split."""
    rows = []
    group_cols = ["time_split", "model", "censoring_mechanism", "censoring_severity"]
    for keys, group in predictions.groupby(group_cols, dropna=False):
        rows.append(metric_row(dict(zip(group_cols, keys)), group))
    return pd.DataFrame(rows)


def calculate_segment_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate validation metrics by important segments."""
    rows = []
    segment_specs = {
        "promotion_status": predictions["activity_flag"].map({0: "non_promotion", 1: "promotion"}),
        "markdown_group": predictions["markdown_group"],
        "sales_volume_group": predictions["sales_volume_group"],
    }
    for segment_name, labels in segment_specs.items():
        tmp = predictions.copy()
        tmp["segment_value"] = labels.astype(str)
        for keys, group in tmp.groupby(["time_split", "model", "segment_value"], dropna=False):
            row = metric_row(
                {"time_split": keys[0], "model": keys[1], "segment": segment_name, "segment_value": keys[2]},
                group,
            )
            rows.append(row)
    series_counts = predictions.groupby(["time_split", "model", "store_id", "sku_id"]).size().reset_index(name="n")
    eligible_series = series_counts.loc[series_counts["n"].ge(20), ["time_split", "model", "store_id", "sku_id"]]
    if not eligible_series.empty:
        merged = predictions.merge(eligible_series, on=["time_split", "model", "store_id", "sku_id"], how="inner")
        for keys, group in merged.groupby(["time_split", "model", "store_id", "sku_id"], dropna=False):
            row = metric_row(
                {
                    "time_split": keys[0],
                    "model": keys[1],
                    "segment": "store_product_series",
                    "segment_value": f"{keys[2]}::{keys[3]}",
                },
                group,
            )
            rows.append(row)
    return pd.DataFrame(rows)


def metric_row(prefix: dict[str, Any], group: pd.DataFrame) -> dict[str, Any]:
    """Create one metric row."""
    true = group["true_demand"].to_numpy(dtype=float)
    pred = group["recovered_pseudo_demand"].to_numpy(dtype=float)
    error = pred - true
    denom = np.maximum(np.abs(true), EPSILON)
    row = dict(prefix)
    row.update(
        {
            "n": int(len(group)),
            "mae": float(mean_absolute_error(true, pred)),
            "rmse": float(math.sqrt(mean_squared_error(true, pred))),
            "median_absolute_error": float(np.median(np.abs(error))),
            "mean_bias": float(np.mean(error)),
            "median_bias": float(np.median(error)),
            "underestimation_rate": float(np.mean(error < -EPSILON)),
            "overestimation_rate": float(np.mean(error > EPSILON)),
            "within_10pct": float(np.mean(np.abs(error) / denom <= 0.10)),
            "within_25pct": float(np.mean(np.abs(error) / denom <= 0.25)),
        }
    )
    return row


def select_recovery_model(metrics: pd.DataFrame) -> str:
    """Select preferred model using validation metrics only."""
    validation = metrics.loc[metrics["time_split"].eq("validation")].copy()
    if validation.empty:
        raise ValueError("No validation metrics available for model selection.")
    summary = (
        validation.groupby("model", dropna=False)
        .agg(
            validation_mae=("mae", "mean"),
            validation_abs_mean_bias=("mean_bias", lambda x: float(np.mean(np.abs(x)))),
            validation_underestimation_rate=("underestimation_rate", "mean"),
            validation_overestimation_rate=("overestimation_rate", "mean"),
            validation_within_25pct=("within_25pct", "mean"),
        )
        .reset_index()
    )
    summary["selection_score"] = (
        summary["validation_mae"]
        + 0.25 * summary["validation_abs_mean_bias"]
        + 0.10 * summary["validation_underestimation_rate"]
    )
    summary = summary.sort_values(["selection_score", "validation_mae", "validation_abs_mean_bias"]).reset_index(drop=True)
    summary["selected"] = False
    summary.loc[0, "selected"] = True
    selected = str(summary.loc[0, "model"])
    test = metrics.loc[metrics["time_split"].eq("test")].groupby("model").agg(test_mae=("mae", "mean"), test_mean_bias=("mean_bias", "mean")).reset_index()
    output = summary.merge(test, on="model", how="left")
    output.to_csv(MODEL_SELECTION_PATH, index=False)
    return selected


def fit_final_model(features: pd.DataFrame, selected_model: str) -> FittedModel:
    """Refit selected model on train+validation likely uncensored rows."""
    trainval_mask = features["time_split"].isin(["train", "validation"]) & features["confirmed_or_likely_uncensored"]
    models = train_models(features, trainval_mask)
    return models[selected_model]


def recover_demand(features: pd.DataFrame, fitted: FittedModel) -> pd.DataFrame:
    """Recover demand for possible-stockout observations."""
    output = features.copy()
    raw_prediction = predict_model(fitted, output)
    train_uncensored = output.loc[
        output["time_split"].isin(["train", "validation"]) & output["confirmed_or_likely_uncensored"],
        "observed_sales_demand",
    ]
    cap_995 = float(train_uncensored.quantile(0.995))
    cap_99 = float(train_uncensored.quantile(0.99))
    output["model_predicted_demand_raw"] = raw_prediction
    output["model_predicted_demand"] = np.clip(raw_prediction, 0.0, cap_995)
    output["recovery_clipped_flag"] = raw_prediction > cap_995
    output["recovery_extrapolation_flag"] = raw_prediction > cap_99
    output["recovered_row_flag"] = output["possible_stockout"].astype(bool)
    recovered = output["observed_sales_demand"].astype(float).copy()
    stockout_mask = output["possible_stockout"].astype(bool)
    recovered.loc[stockout_mask] = np.maximum(
        recovered.loc[stockout_mask],
        output.loc[stockout_mask, "model_predicted_demand"],
    )
    recovered = np.maximum(recovered, 0.0)
    output["recovered_demand"] = recovered
    output["demand_recovery_amount"] = output["recovered_demand"] - output["observed_sales_demand"]
    output["demand_recovery_ratio"] = output["recovered_demand"] / np.maximum(output["observed_sales_demand"], EPSILON)
    output["selected_recovery_model"] = fitted.name
    if (output["recovered_demand"] + EPSILON < output["observed_sales_demand"]).any():
        raise ValueError("Recovered demand fell below observed sales.")
    if output["recovered_demand"].lt(0).any():
        raise ValueError("Negative recovered demand detected.")
    return output


def recovery_diagnostics(recovered: pd.DataFrame) -> dict[str, Any]:
    """Save recovery summary tables."""
    observed_total = float(recovered["observed_sales_demand"].sum())
    recovered_total = float(recovered["recovered_demand"].sum())
    adjusted = recovered["demand_recovery_amount"].gt(EPSILON)
    summary_rows = [
        {"metric": "mean_observed_sales", "value": float(recovered["observed_sales_demand"].mean())},
        {"metric": "mean_recovered_demand", "value": float(recovered["recovered_demand"].mean())},
        {"metric": "median_observed_sales", "value": float(recovered["observed_sales_demand"].median())},
        {"metric": "median_recovered_demand", "value": float(recovered["recovered_demand"].median())},
        {"metric": "total_observed_sales", "value": observed_total},
        {"metric": "total_recovered_demand", "value": recovered_total},
        {"metric": "percentage_increase_aggregate_demand", "value": float((recovered_total / max(observed_total, EPSILON) - 1.0) * 100.0)},
        {"metric": "share_of_rows_adjusted", "value": float(adjusted.mean())},
        {"metric": "mean_adjustment_among_adjusted_rows", "value": float(recovered.loc[adjusted, "demand_recovery_amount"].mean() if adjusted.any() else 0.0)},
    ]
    pd.DataFrame(summary_rows).to_csv(RECOVERY_SUMMARY_PATH, index=False)
    save_group_summary(recovered, ["activity_flag"], RECOVERY_BY_PROMOTION_PATH)
    recovered["markdown_group"] = markdown_group(recovered["markdown_rate"])
    save_group_summary(recovered, ["markdown_group"], RECOVERY_BY_MARKDOWN_PATH)
    save_group_summary(recovered, ["sku_id"], RECOVERY_BY_PRODUCT_PATH)
    save_group_summary(recovered, ["store_id"], RECOVERY_BY_STORE_PATH)
    save_group_summary(recovered, ["activity_flag", "possible_stockout"], PROMOTION_STOCKOUT_PATH)
    return {row["metric"]: row["value"] for row in summary_rows}


def save_group_summary(data: pd.DataFrame, group_cols: list[str], path: Path) -> None:
    """Save recovery diagnostics by group."""
    summary = (
        data.groupby(group_cols, dropna=False)
        .agg(
            rows=("observed_sales_demand", "size"),
            mean_observed_sales=("observed_sales_demand", "mean"),
            mean_recovered_demand=("recovered_demand", "mean"),
            total_observed_sales=("observed_sales_demand", "sum"),
            total_recovered_demand=("recovered_demand", "sum"),
            mean_recovery_amount=("demand_recovery_amount", "mean"),
            share_rows_adjusted=("demand_recovery_amount", lambda x: float((x > EPSILON).mean())),
        )
        .reset_index()
    )
    summary["aggregate_recovery_pct"] = (
        summary["total_recovered_demand"] / summary["total_observed_sales"].replace(0, np.nan) - 1.0
    ) * 100.0
    summary.to_csv(path, index=False)


def robustness_checks(features: pd.DataFrame, recovered: pd.DataFrame, selected_model: str) -> pd.DataFrame:
    """Run concise sensitivity checks for definitions, caps, and models."""
    rows = []
    definitions = {
        "strict": (~features["possible_stockout"]) & features["observed_sales_demand"].gt(0) & features["stock_hour6_22_cnt"].gt(0),
        "moderate": features["confirmed_or_likely_uncensored"],
    }
    for definition_name, mask in definitions.items():
        rows.append(
            {
                "check_type": "uncensored_definition",
                "setting": definition_name,
                "eligible_trainval_rows": int((mask & features["time_split"].isin(["train", "validation"])).sum()),
                "aggregate_recovery_pct_reference": float(
                    recovered["recovered_demand"].sum() / max(recovered["observed_sales_demand"].sum(), EPSILON) - 1.0
                )
                * 100.0,
                "material_change_flag": False,
            }
        )
    for quantile in [0.99, 0.995]:
        cap = float(features.loc[features["confirmed_or_likely_uncensored"], "observed_sales_demand"].quantile(quantile))
        pred = np.minimum(recovered["model_predicted_demand_raw"], cap)
        alt = recovered["observed_sales_demand"].copy()
        mask = recovered["possible_stockout"].astype(bool)
        alt.loc[mask] = np.maximum(alt.loc[mask], pred.loc[mask])
        pct = float(alt.sum() / max(recovered["observed_sales_demand"].sum(), EPSILON) - 1.0) * 100.0
        rows.append(
            {
                "check_type": "recovery_cap",
                "setting": f"p{quantile}",
                "eligible_trainval_rows": int(features["confirmed_or_likely_uncensored"].sum()),
                "aggregate_recovery_pct_reference": pct,
                "material_change_flag": bool(abs(pct - (recovered["recovered_demand"].sum() / max(recovered["observed_sales_demand"].sum(), EPSILON) - 1.0) * 100.0) > 5.0),
            }
        )
    rows.append(
        {
            "check_type": "alternative_model",
            "setting": f"selected_primary={selected_model}; compared_against=seasonal_baseline_in_validation_metrics",
            "eligible_trainval_rows": int((features["confirmed_or_likely_uncensored"] & features["time_split"].isin(["train", "validation"])).sum()),
            "aggregate_recovery_pct_reference": float(
                recovered["recovered_demand"].sum() / max(recovered["observed_sales_demand"].sum(), EPSILON) - 1.0
            )
            * 100.0,
            "material_change_flag": False,
        }
    )
    output = pd.DataFrame(rows)
    output.to_csv(ROBUSTNESS_PATH, index=False)
    return output


def create_figures(recovered: pd.DataFrame, validation_predictions: pd.DataFrame, validation_metrics: pd.DataFrame) -> None:
    """Create matplotlib-only diagnostic figures."""
    plot_distribution_comparison(recovered)
    plot_predicted_vs_true(validation_predictions)
    plot_error_by_model(validation_metrics)
    plot_recovery_amount_by_flag(recovered, "possible_stockout", "recovery_amount_by_stockout_status.png")
    plot_recovery_amount_by_flag(recovered, "activity_flag", "recovery_amount_by_promotion_status.png")
    plot_aggregate_over_time(recovered)
    plot_example_series(recovered)
    plot_recovery_ratio(recovered)
    plot_performance_by_severity(validation_metrics)


def plot_distribution_comparison(data: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    cap = data["recovered_demand"].quantile(0.99)
    ax.hist(data["observed_sales_demand"].clip(upper=cap), bins=40, alpha=0.55, density=True, label="observed")
    ax.hist(data["recovered_demand"].clip(upper=cap), bins=40, alpha=0.55, density=True, label="recovered")
    ax.set_title("Observed versus recovered demand distribution")
    ax.set_xlabel("Demand")
    ax.set_ylabel("Density")
    ax.legend()
    savefig("observed_vs_recovered_distribution.png")


def plot_predicted_vs_true(predictions: pd.DataFrame) -> None:
    sample = predictions.loc[predictions["time_split"].eq("test")].sample(
        n=min(4_000, len(predictions.loc[predictions["time_split"].eq("test")])),
        random_state=RANDOM_SEED,
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(sample["true_demand"], sample["recovered_pseudo_demand"], s=8, alpha=0.35)
    max_value = float(max(sample["true_demand"].max(), sample["recovered_pseudo_demand"].max()))
    ax.plot([0, max_value], [0, max_value], color="black", linewidth=1)
    ax.set_title("Artificial-censoring predicted versus true demand")
    ax.set_xlabel("Pseudo-ground-truth demand")
    ax.set_ylabel("Recovered pseudo demand")
    savefig("artificial_censoring_predicted_vs_true.png")


def plot_error_by_model(metrics: pd.DataFrame) -> None:
    summary = metrics.groupby("model")["mae"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    summary.plot(kind="bar", ax=ax)
    ax.set_title("Recovery error by model")
    ax.set_ylabel("Mean MAE across pseudo-censoring scenarios")
    plt.xticks(rotation=25, ha="right")
    savefig("recovery_error_by_model.png")


def plot_recovery_amount_by_flag(data: pd.DataFrame, column: str, filename: str) -> None:
    groups = [group["demand_recovery_amount"].clip(upper=data["demand_recovery_amount"].quantile(0.99)).to_numpy() for _, group in data.groupby(column)]
    labels = [str(key) for key, _ in data.groupby(column)]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(groups, tick_labels=labels, showfliers=False)
    ax.set_title(f"Recovery amount by {column}")
    ax.set_ylabel("Demand recovery amount")
    savefig(filename)


def plot_aggregate_over_time(data: pd.DataFrame) -> None:
    daily = data.groupby(data["timestamp"].dt.floor("D")).agg(
        observed=("observed_sales_demand", "sum"),
        recovered=("recovered_demand", "sum"),
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(daily.index, daily["observed"], label="observed")
    ax.plot(daily.index, daily["recovered"], label="recovered")
    ax.set_title("Aggregate observed versus recovered demand over time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Aggregate demand")
    ax.legend()
    savefig("aggregate_observed_vs_recovered_over_time.png")


def plot_example_series(data: pd.DataFrame) -> None:
    candidates = data.loc[data["demand_recovery_amount"].gt(EPSILON)].groupby(["store_id", "sku_id"]).size().sort_values(ascending=False)
    if candidates.empty:
        candidates = data.groupby(["store_id", "sku_id"]).size().sort_values(ascending=False)
    store_id, sku_id = candidates.index[0]
    group = data.loc[(data["store_id"] == store_id) & (data["sku_id"] == sku_id)].sort_values("timestamp")
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(group["timestamp"], group["observed_sales_demand"], label="observed sales", color="tab:blue")
    ax1.plot(group["timestamp"], group["recovered_demand"], label="recovered demand", color="tab:green")
    stockout = group.loc[group["possible_stockout"]]
    ax1.scatter(stockout["timestamp"], stockout["observed_sales_demand"], label="possible stockout", color="tab:red", marker="x")
    ax2 = ax1.twinx()
    ax2.plot(group["timestamp"], group["stock_hour6_22_cnt"], label="inventory indicator", color="tab:orange", alpha=0.5)
    ax1.set_title(f"Example recovered series: store={store_id}, product={sku_id}")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Demand")
    ax2.set_ylabel("Inventory-related value")
    ax1.legend(loc="upper left")
    savefig("example_store_product_recovered_series.png")


def plot_recovery_ratio(data: pd.DataFrame) -> None:
    ratio = data["demand_recovery_ratio"].replace([np.inf, -np.inf], np.nan).dropna().clip(upper=20)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(ratio, bins=50)
    ax.set_title("Recovery ratio distribution")
    ax.set_xlabel("Recovered / observed demand")
    ax.set_ylabel("Rows")
    savefig("recovery_ratio_distribution.png")


def plot_performance_by_severity(metrics: pd.DataFrame) -> None:
    summary = metrics.groupby(["censoring_severity", "model"])["mae"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    for model, group in summary.groupby("model"):
        ax.plot(group["censoring_severity"], group["mae"], marker="o", label=model)
    ax.set_title("Model performance by censoring severity")
    ax.set_xlabel("Censoring severity")
    ax.set_ylabel("MAE")
    ax.legend()
    plt.xticks(rotation=25, ha="right")
    savefig("model_performance_by_censoring_severity.png")


def savefig(filename: str) -> None:
    """Save the current matplotlib figure."""
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=160)
    plt.close()


def markdown_group(markdown: pd.Series) -> pd.Series:
    """Create markdown exposure groups."""
    return pd.cut(
        pd.to_numeric(markdown, errors="coerce").fillna(0.0),
        bins=[-0.001, 0.001, 0.10, 1.0],
        labels=["no_markdown", "mild_markdown", "moderate_deep_markdown"],
        include_lowest=True,
    ).astype(str)


def quantile_label(values: pd.Series, low: str, medium: str, high: str) -> pd.Series:
    """Return duplicate-safe tertile labels."""
    clean = pd.to_numeric(values, errors="coerce").rank(method="first")
    return pd.qcut(clean, q=3, labels=[low, medium, high]).astype(str)


def save_model_bundle(fitted: FittedModel, encoders: dict[str, dict[str, int]], cap: float) -> None:
    """Save fitted model and preprocessing artifacts."""
    bundle = {
        "selected_model": fitted.name,
        "model": fitted.model,
        "model_kind": fitted.kind,
        "feature_columns": fitted.feature_columns,
        "encoders": encoders,
        "prediction_cap_995": cap,
        "limitations": limitations(),
    }
    joblib.dump(bundle, MODEL_PATH)


def limitations() -> list[str]:
    """Important methodological limitations."""
    return [
        "Actual latent demand during real stockouts is unobserved.",
        "Artificial censoring only tests recovery under simulated censoring.",
        "possible_stockout_flag is a diagnostic definition, not guaranteed ground truth.",
        "Inventory-field semantics remain partly uncertain.",
        "Discount semantics remain uncertain_multiplier_like.",
        "Recovered demand is a model estimate, not observed truth.",
        "No causal promotion or discount effect has been established.",
    ]


def final_status(validation_metrics: pd.DataFrame, recovered: pd.DataFrame, selected_model: str) -> str:
    """Return final status after validation checks."""
    has_validation = not validation_metrics.empty
    has_test = validation_metrics["time_split"].eq("test").any()
    saved = OUTPUT_PATH.exists()
    bounded = recovered["model_predicted_demand"].ge(0).all() and recovered["recovered_demand"].ge(0).all()
    monotonic = (recovered["recovered_demand"] + EPSILON >= recovered["observed_sales_demand"]).all()
    selected_rows = validation_metrics.loc[validation_metrics["model"].eq(selected_model)]
    severe_bias = bool(selected_rows["mean_bias"].abs().mean() > max(2.0, recovered["observed_sales_demand"].mean() * 2.0))
    baseline_mae = validation_metrics.loc[validation_metrics["model"].eq("seasonal_baseline") & validation_metrics["time_split"].eq("validation"), "mae"].mean()
    selected_mae = validation_metrics.loc[validation_metrics["model"].eq(selected_model) & validation_metrics["time_split"].eq("validation"), "mae"].mean()
    no_better_than_baseline = bool(np.isfinite(baseline_mae) and np.isfinite(selected_mae) and selected_mae > baseline_mae * 1.10)
    if has_validation and has_test and saved and bounded and monotonic and not severe_bias and not no_better_than_baseline:
        return "LATENT_DEMAND_RECOVERY_READY"
    return "LATENT_DEMAND_RECOVERY_REQUIRES_REVISION"


def print_report(status: str, selected_model: str, validation_metrics: pd.DataFrame, recovered: pd.DataFrame, summary: dict[str, Any]) -> None:
    """Print final console report."""
    validation = validation_metrics.loc[
        validation_metrics["time_split"].eq("validation") & validation_metrics["model"].eq(selected_model)
    ]
    test = validation_metrics.loc[
        validation_metrics["time_split"].eq("test") & validation_metrics["model"].eq(selected_model)
    ]
    print("\nLatent demand recovery report")
    print(f"Selected model: {selected_model}")
    print("Validation metrics:", compact_metrics(validation))
    print("Test metrics:", compact_metrics(test))
    print(f"Observed total demand: {summary['total_observed_sales']:.4f}")
    print(f"Recovered total demand: {summary['total_recovered_demand']:.4f}")
    print(f"Aggregate recovery percentage: {summary['percentage_increase_aggregate_demand']:.4f}%")
    print(f"Percentage of rows adjusted: {summary['share_of_rows_adjusted'] * 100.0:.4f}%")
    promo = pd.read_csv(RECOVERY_BY_PROMOTION_PATH)
    print("Recovery by promotion status:")
    for _, row in promo.iterrows():
        print(f"- activity_flag={row['activity_flag']}: aggregate_recovery_pct={row['aggregate_recovery_pct']:.4f}%")
    print("Major assumptions:")
    print("- Likely uncensored rows train the recovery model; possible-stockout rows are not treated as true latent demand.")
    print("- markdown_rate = 1 - discount is used as a multiplier-like diagnostic feature.")
    print("Key limitations:")
    for item in limitations():
        print(f"- {item}")
    print("Recommended next module: stockout-aware discount-response calibration, before Gymnasium or PPO.")
    print(status)


def compact_metrics(metrics: pd.DataFrame) -> dict[str, float]:
    """Summarize metrics for console output."""
    if metrics.empty:
        return {}
    return {
        "mae": float(metrics["mae"].mean()),
        "rmse": float(metrics["rmse"].mean()),
        "mean_bias": float(metrics["mean_bias"].mean()),
        "underestimation_rate": float(metrics["underestimation_rate"].mean()),
        "within_25pct": float(metrics["within_25pct"].mean()),
    }


def save_metadata(status: str, selected_model: str, checks: dict[str, Any], summary: dict[str, Any]) -> None:
    """Save run metadata and limitations."""
    metadata = {
        "status": status,
        "selected_model": selected_model,
        "input_path": str(INPUT_PATH.relative_to(PROJECT_ROOT)),
        "output_path": str(OUTPUT_PATH.relative_to(PROJECT_ROOT)),
        "validation_checks": checks,
        "summary": summary,
        "limitations": limitations(),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")


def run() -> str:
    """Run the full latent-demand recovery workflow."""
    ensure_dirs()
    data = load_and_validate_subset()
    checks = dict(data.attrs.get("validation_checks", {}))
    labeled = create_censoring_labels(data)
    featured = engineer_features(labeled)
    train_for_encoding = featured.loc[featured["time_split"].eq("train") & featured["confirmed_or_likely_uncensored"]]
    encoders = fit_encoders(train_for_encoding)
    featured = apply_encoders(featured, encoders)
    train_mask = featured["time_split"].eq("train") & featured["confirmed_or_likely_uncensored"]
    if train_mask.sum() < 100:
        raise ValueError("Too few uncensored training observations.")
    models = train_models(featured, train_mask)
    validation_predictions, validation_metrics, _ = artificial_censoring_validation(featured, models)
    selected_model = select_recovery_model(validation_metrics)
    final_model = fit_final_model(featured, selected_model)
    recovered = recover_demand(featured, final_model)
    recovered.to_parquet(OUTPUT_PATH, index=False)
    cap = float(featured.loc[featured["confirmed_or_likely_uncensored"], "observed_sales_demand"].quantile(0.995))
    save_model_bundle(final_model, encoders, cap)
    summary = recovery_diagnostics(recovered)
    robustness_checks(featured, recovered, selected_model)
    create_figures(recovered, validation_predictions, validation_metrics)
    status = final_status(validation_metrics, recovered, selected_model)
    save_metadata(status, selected_model, checks, summary)
    print_report(status, selected_model, validation_metrics, recovered, summary)
    return status


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"Latent demand recovery failed: {exc}")
        print("LATENT_DEMAND_RECOVERY_REQUIRES_REVISION")
        raise
