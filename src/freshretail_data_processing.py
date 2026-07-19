"""FreshRetailNet-50K inspection and subset preparation.

This module discovers locally downloaded FreshRetailNet files, inspects schema
without assuming exact file or column names, validates time-series suitability,
and creates a reproducible student-laptop subset for later RL work.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RANDOM_SEED = 42
SUPPORTED_SUFFIXES = {
    ".csv",
    ".gz",
    ".parquet",
    ".json",
    ".jsonl",
    ".ndjson",
}
SAMPLE_ROWS = 500
CSV_CHUNKSIZE = 100_000
MAX_IN_MEMORY_ROWS = 5_000_000
MIN_SEQUENCE_OBSERVATIONS = 14
TARGET_STORES = (20, 30)
TARGET_SKUS = (80, 150)
TARGET_MIN_DAYS = 28
TARGET_MODELING_SERIES = 300
MIN_MODELING_SERIES = 200
MAX_MODELING_SERIES = 500

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "operational" / "raw" / "freshretail" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "freshretail" / "processed"
OLD_STUDENT_SUBSET_PATH = (
    PROJECT_ROOT / "data" / "operational" / "processed" / "freshretail" / "freshretail_student_subset.parquet"
)
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "freshretail_processing"
MODELING_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "freshretail_modeling_subset"

FILE_INVENTORY_PATH = TABLES_DIR / "freshretail_file_inventory.csv"
SCHEMA_SUMMARY_PATH = TABLES_DIR / "freshretail_schema_summary.csv"
COLUMN_MAPPING_PATH = TABLES_DIR / "freshretail_column_mapping.csv"
TIMESERIES_QUALITY_PATH = TABLES_DIR / "freshretail_timeseries_quality.csv"
STOCKOUT_SUMMARY_PATH = TABLES_DIR / "freshretail_stockout_summary.csv"
PROMOTION_SUPPORT_PATH = TABLES_DIR / "freshretail_promotion_support.csv"
SUBSET_SUMMARY_PATH = TABLES_DIR / "freshretail_subset_summary.csv"
SELECTED_SERIES_PATH = TABLES_DIR / "freshretail_selected_series.csv"
PILOT_SUBSET_PATH = PROCESSED_DIR / "freshretail_pilot_subset.parquet"
MODELING_SUBSET_PATH = PROCESSED_DIR / "freshretail_modeling_subset.parquet"
MODELING_SUMMARY_PATH = TABLES_DIR / "freshretail_modeling_subset_summary.csv"
MODELING_SERIES_PATH = TABLES_DIR / "freshretail_modeling_subset_series.csv"
REPRESENTATIVENESS_PATH = TABLES_DIR / "freshretail_subset_representativeness.csv"
TEMPORAL_SPLIT_SUMMARY_PATH = TABLES_DIR / "freshretail_temporal_split_summary.csv"
DISCOUNT_SEMANTICS_AUDIT_PATH = TABLES_DIR / "freshretail_discount_semantics_audit.csv"

CRITICAL_CONCEPTS = ["timestamp", "store_id", "sku_id", "sales_qty", "inventory"]


@dataclass(frozen=True)
class FileInfo:
    """Inventory record for one input file."""

    path: Path
    relative_path: str
    suffix: str
    size_bytes: int


@dataclass(frozen=True)
class ColumnCandidate:
    """Detected candidate mapping for one semantic concept."""

    concept: str
    column: str
    score: int
    status: str
    reason: str


CONCEPT_KEYWORDS: dict[str, list[str]] = {
    "timestamp": ["timestamp", "datetime", "date_time", "time_stamp", "date", "dt", "ds"],
    "date": ["date", "day", "dt"],
    "hour": ["hour", "hh", "time"],
    "city": ["city", "region", "area"],
    "store_id": ["store_id", "store", "shop_id", "shop", "warehouse", "station"],
    "sku_id": ["sku_id", "sku", "product_id", "item_id", "goods_id", "spu", "plu"],
    "category": ["category", "cat", "class", "dept", "department", "group"],
    "sales_qty": [
        "sales_qty",
        "sale_qty",
        "sold_qty",
        "quantity_sold",
        "qty_sold",
        "sales",
        "sale",
        "units_sold",
        "demand",
        "order_qty",
    ],
    "inventory": [
        "inventory",
        "stock",
        "on_hand",
        "available",
        "stock_level",
        "ending_inventory",
        "begin_inventory",
    ],
    "stock_status": ["stock_status", "availability", "available_status"],
    "stockout": ["stockout", "out_of_stock", "oos", "sold_out", "is_oos"],
    "promotion": ["promotion", "promo", "campaign", "is_promoted", "promoted", "activity"],
    "discount_rate": ["discount", "discount_rate", "discount_pct", "off_rate", "markdown"],
    "original_price": ["original_price", "list_price", "base_price", "regular_price", "ori_price"],
    "selling_price": ["selling_price", "sale_price", "price", "actual_price", "transaction_price"],
    "weather": ["weather", "temperature", "temp", "wind", "humidity"],
    "precipitation": ["precipitation", "rain", "snow", "precip"],
    "holiday": ["holiday", "festival", "weekend", "weekday", "dayofweek", "is_holiday"],
}


def ensure_dirs() -> None:
    """Create required output directories."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELING_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def discover_files(raw_dir: Path = RAW_DIR) -> list[FileInfo]:
    """Recursively discover supported local data files."""
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"FreshRetailNet raw directory does not exist: {raw_dir}. "
            "Download files from Hugging Face and place them under "
            "data/operational/raw/freshretail/raw/."
        )
    files: list[FileInfo] = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = normalized_suffix(path)
        if suffix not in SUPPORTED_SUFFIXES:
            continue
        files.append(
            FileInfo(
                path=path,
                relative_path=str(path.relative_to(PROJECT_ROOT)),
                suffix=suffix,
                size_bytes=path.stat().st_size,
            )
        )
    if not files:
        raise FileNotFoundError(f"No supported files found under {raw_dir}.")
    inventory = pd.DataFrame(
        [
            {
                "relative_path": item.relative_path,
                "file_name": item.path.name,
                "extension": item.suffix,
                "size_bytes": item.size_bytes,
                "size_mb": item.size_bytes / (1024**2),
            }
            for item in files
        ]
    )
    inventory["n_files_total"] = len(files)
    inventory["total_size_bytes"] = sum(item.size_bytes for item in files)
    inventory["total_size_mb"] = inventory["total_size_bytes"] / (1024**2)
    inventory.to_csv(FILE_INVENTORY_PATH, index=False)
    print(f"Discovered {len(files)} supported files under {raw_dir}.")
    print(f"Total size: {sum(item.size_bytes for item in files) / (1024**2):.2f} MB")
    return files


def normalized_suffix(path: Path) -> str:
    """Return supported logical suffix, handling csv.gz files."""
    lower = path.name.lower()
    if lower.endswith(".csv.gz"):
        return ".csv"
    if lower.endswith(".jsonl.gz") or lower.endswith(".ndjson.gz"):
        return ".jsonl"
    if lower.endswith(".json.gz"):
        return ".json"
    return path.suffix.lower()


def inspect_schemas(files: list[FileInfo]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Inspect small samples and infer transparent candidate mappings."""
    schema_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    samples: list[pd.DataFrame] = []

    for info in files:
        sample = read_sample(info.path)
        if sample.empty:
            print(f"Warning: sample is empty for {info.relative_path}")
            continue
        samples.append(sample)
        missing = sample.isna().sum()
        for column in sample.columns:
            schema_rows.append(
                {
                    "relative_path": info.relative_path,
                    "file_name": info.path.name,
                    "extension": info.suffix,
                    "column_name": column,
                    "dtype_in_sample": str(sample[column].dtype),
                    "sample_missing_count": int(missing[column]),
                    "sample_missing_rate": float(missing[column] / max(1, len(sample))),
                    "sample_non_null_count": int(sample[column].notna().sum()),
                    "sample_values": compact_sample_values(sample[column]),
                }
            )
        print("\nSchema sample:", info.relative_path)
        print("Columns:", list(sample.columns))
        print("Dtypes:", {column: str(dtype) for column, dtype in sample.dtypes.items()})
        print("Missing counts:", missing.to_dict())
        print(sample.head(5).to_string(index=False))

    if not samples:
        raise ValueError("No readable samples found in supported FreshRetailNet files.")

    combined_sample = pd.concat(samples, ignore_index=True, sort=False).head(max(SAMPLE_ROWS, 1_000))
    candidates = detect_column_candidates(combined_sample)
    confirmed: dict[str, str] = {}
    for concept, concept_candidates in candidates.items():
        for candidate in concept_candidates:
            mapping_rows.append(
                {
                    "concept": candidate.concept,
                    "candidate_column": candidate.column,
                    "score": candidate.score,
                    "status": candidate.status,
                    "reason": candidate.reason,
                }
            )
            if candidate.status == "confirmed":
                confirmed[concept] = candidate.column

    schema = pd.DataFrame(schema_rows)
    mapping = pd.DataFrame(mapping_rows)
    schema.to_csv(SCHEMA_SUMMARY_PATH, index=False)
    mapping.to_csv(COLUMN_MAPPING_PATH, index=False)
    return schema, mapping, confirmed


def read_sample(path: Path, nrows: int = SAMPLE_ROWS) -> pd.DataFrame:
    """Read a small sample from a supported file."""
    suffix = normalized_suffix(path)
    if suffix == ".csv":
        return pd.read_csv(path, nrows=nrows)
    if suffix == ".parquet":
        return read_parquet_sample(path, nrows)
    if suffix in {".json", ".jsonl", ".ndjson"}:
        return read_json_sample(path, nrows)
    raise ValueError(f"Unsupported file type for sample: {path}")


def read_parquet_sample(path: Path, nrows: int) -> pd.DataFrame:
    """Read a small Parquet sample, preferring pyarrow row-group access."""
    try:
        import pyarrow.parquet as pq  # type: ignore

        parquet_file = pq.ParquetFile(path)
        if parquet_file.num_row_groups == 0:
            return pd.DataFrame()
        table = parquet_file.read_row_group(0)
        return table.to_pandas().head(nrows)
    except Exception as exc:
        print(f"pyarrow sample read unavailable for {path.name}: {exc}")
        return pd.read_parquet(path).head(nrows)


def read_json_sample(path: Path, nrows: int) -> pd.DataFrame:
    """Read a JSON/JSONL sample."""
    try:
        return pd.read_json(path, lines=True, nrows=nrows)
    except TypeError:
        return pd.read_json(path).head(nrows)
    except ValueError:
        return pd.read_json(path).head(nrows)


def compact_sample_values(series: pd.Series, limit: int = 5) -> str:
    """Return compact unique sample values for schema audit."""
    values = series.dropna().astype(str).head(limit).tolist()
    return json.dumps(values, ensure_ascii=False)


def detect_column_candidates(sample: pd.DataFrame) -> dict[str, list[ColumnCandidate]]:
    """Detect possible concept-column mappings using transparent matching."""
    candidates: dict[str, list[ColumnCandidate]] = {}
    columns = list(sample.columns)
    for concept, keywords in CONCEPT_KEYWORDS.items():
        scored: list[tuple[str, int, str]] = []
        for column in columns:
            score, reason = score_column(column, sample[column], concept, keywords)
            if score > 0:
                scored.append((column, score, reason))
        scored.sort(key=lambda item: (-item[1], item[0]))
        if not scored:
            candidates[concept] = [
                ColumnCandidate(concept, "", 0, "missing", "No candidate matched configured keywords.")
            ]
            continue
        best_score = scored[0][1]
        top = [item for item in scored if item[1] == best_score]
        status = "confirmed" if len(top) == 1 and best_score >= 4 else "ambiguous"
        concept_candidates: list[ColumnCandidate] = []
        for column, score, reason in scored[:8]:
            item_status = status if score == best_score else "alternative"
            concept_candidates.append(ColumnCandidate(concept, column, score, item_status, reason))
        candidates[concept] = concept_candidates
    return candidates


def score_column(column: str, series: pd.Series, concept: str, keywords: list[str]) -> tuple[int, str]:
    """Score a column as a possible semantic concept."""
    lower = normalize_name(column)
    score = 0
    reasons: list[str] = []
    for keyword in keywords:
        key = normalize_name(keyword)
        if lower == key:
            score += 6
            reasons.append(f"exact:{keyword}")
        elif key in lower:
            score += 3
            reasons.append(f"contains:{keyword}")
    if concept in {"timestamp", "date"} and should_try_datetime_parse(series, lower):
        parsed = pd.to_datetime(series.dropna().head(100), errors="coerce")
        if parsed.notna().mean() >= 0.8:
            score += 3
            reasons.append("datetime_parseable")
    if concept in {"sales_qty", "inventory", "discount_rate", "original_price", "selling_price"}:
        numeric = pd.to_numeric(series.dropna().head(200), errors="coerce")
        if numeric.notna().mean() >= 0.8:
            score += 2
            reasons.append("numeric")
    if concept in {"stockout", "promotion"}:
        unique = series.dropna().astype(str).str.lower().head(500).unique().tolist()
        bool_like = {"0", "1", "true", "false", "yes", "no", "y", "n"}
        if unique and set(unique).issubset(bool_like):
            score += 2
            reasons.append("bool_like")
    return score, ";".join(reasons) if reasons else "no_match"


def should_try_datetime_parse(series: pd.Series, normalized_column: str) -> bool:
    """Avoid treating numeric IDs, flags, prices, and discounts as dates."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return False
    date_tokens = ("date", "dt", "day", "time", "timestamp")
    return any(token in normalized_column for token in date_tokens)


def normalize_name(name: str) -> str:
    """Normalize a column name for matching."""
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name)).strip("_")


def require_critical_mappings(mapping: dict[str, str]) -> list[str]:
    """Return missing critical semantic mappings."""
    return [concept for concept in CRITICAL_CONCEPTS if concept not in mapping or not mapping[concept]]


def load_selected_columns(files: list[FileInfo], mapping: dict[str, str]) -> pd.DataFrame:
    """Load only mapped columns needed for diagnostics and subsetting."""
    selected_columns = sorted(set(mapping.values()))
    chunks: list[pd.DataFrame] = []
    total_rows = 0
    for info in files:
        print(f"Loading selected columns from {info.relative_path}")
        for chunk in iter_selected_chunks(info.path, selected_columns):
            if chunk.empty:
                continue
            chunks.append(chunk)
            total_rows += len(chunk)
            if total_rows > MAX_IN_MEMORY_ROWS:
                raise MemoryError(
                    f"Loaded row count exceeded safety limit ({MAX_IN_MEMORY_ROWS}). "
                    "Increase MAX_IN_MEMORY_ROWS or create a more selective reader."
                )
    if not chunks:
        raise ValueError("No rows loaded from mapped FreshRetailNet columns.")
    data = pd.concat(chunks, ignore_index=True, sort=False)
    return standardize_columns(data, mapping)


def iter_selected_chunks(path: Path, columns: list[str]) -> Iterable[pd.DataFrame]:
    """Yield selected-column chunks from a supported file."""
    suffix = normalized_suffix(path)
    if suffix == ".csv":
        for chunk in pd.read_csv(path, usecols=lambda col: col in columns, chunksize=CSV_CHUNKSIZE):
            yield chunk
    elif suffix == ".parquet":
        yield read_parquet_selected(path, columns)
    elif suffix in {".json", ".jsonl", ".ndjson"}:
        try:
            for chunk in pd.read_json(path, lines=True, chunksize=CSV_CHUNKSIZE):
                yield chunk[[column for column in columns if column in chunk.columns]]
        except ValueError:
            frame = pd.read_json(path)
            yield frame[[column for column in columns if column in frame.columns]]
    else:
        raise ValueError(f"Unsupported file type: {path}")


def read_parquet_selected(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read selected columns from Parquet."""
    try:
        import pyarrow.parquet as pq  # type: ignore

        parquet_file = pq.ParquetFile(path)
        available = parquet_file.schema.names
        selected = [column for column in columns if column in available]
        if not selected:
            return pd.DataFrame()
        tables = [parquet_file.read_row_group(i, columns=selected) for i in range(parquet_file.num_row_groups)]
        if not tables:
            return pd.DataFrame()
        import pyarrow as pa  # type: ignore

        return pa.concat_tables(tables).to_pandas()
    except Exception as exc:
        print(f"Falling back to pandas.read_parquet for {path.name}: {exc}")
        return pd.read_parquet(path, columns=[column for column in columns if column])


def standardize_columns(data: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Create standardized analysis columns from confirmed mappings."""
    output = pd.DataFrame(index=data.index)
    for concept, column in mapping.items():
        if column in data.columns:
            output[concept] = data[column]
    if mapping.get("inventory") in data.columns:
        output["stock_hour6_22_cnt"] = pd.to_numeric(data[mapping["inventory"]], errors="coerce")
    output["timestamp"] = infer_timestamp(output)
    output["store_id"] = output["store_id"].astype(str)
    output["sku_id"] = output["sku_id"].astype(str)
    if "category" in output:
        output["category"] = output["category"].astype(str)
    else:
        output["category"] = "unknown"
    output["sales_qty"] = pd.to_numeric(output["sales_qty"], errors="coerce")
    output["inventory"] = pd.to_numeric(output["inventory"], errors="coerce")
    if "discount_rate" in output:
        output["discount"] = pd.to_numeric(output["discount_rate"], errors="coerce")
        output = output.drop(columns=["discount_rate"])
    elif {"original_price", "selling_price"}.issubset(output.columns):
        original = pd.to_numeric(output["original_price"], errors="coerce")
        selling = pd.to_numeric(output["selling_price"], errors="coerce")
        output["discount"] = (selling / original.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    else:
        output["discount"] = np.nan
    output["markdown_rate"] = np.where(
        output["discount"].between(0, 1, inclusive="both"),
        1.0 - output["discount"],
        np.nan,
    )
    if "promotion" in output:
        output["promotion"] = parse_bool_like(output["promotion"])
    else:
        output["promotion"] = output["markdown_rate"].fillna(0.0) > 0
    if "stockout" in output:
        output["stockout"] = parse_bool_like(output["stockout"])
    else:
        output["stockout"] = output["inventory"].fillna(np.inf) <= 0
    output["zero_inventory_flag"] = output["inventory"].fillna(np.inf) <= 0
    output["positive_sales_flag"] = output["sales_qty"].fillna(0) > 0
    output["possible_stockout_flag"] = output["zero_inventory_flag"]
    output = output.dropna(subset=["timestamp", "store_id", "sku_id", "sales_qty", "inventory"]).copy()
    output = output.sort_values(["store_id", "sku_id", "timestamp"]).reset_index(drop=True)
    return output


def infer_timestamp(output: pd.DataFrame) -> pd.Series:
    """Infer timestamp from timestamp/date/hour columns."""
    if "timestamp" in output:
        timestamp = pd.to_datetime(output["timestamp"], errors="coerce")
    elif "date" in output:
        timestamp = pd.to_datetime(output["date"], errors="coerce")
    else:
        raise ValueError("No confirmed timestamp or date mapping available.")
    if "hour" in output and timestamp.notna().any():
        hour = pd.to_numeric(output["hour"], errors="coerce").fillna(0).astype(int).clip(0, 23)
        timestamp = timestamp.dt.floor("D") + pd.to_timedelta(hour, unit="h")
    return timestamp


def parse_bool_like(series: pd.Series) -> pd.Series:
    """Parse bool-like values robustly."""
    lower = series.astype(str).str.strip().str.lower()
    true_values = {"1", "true", "yes", "y", "t", "stockout", "out_of_stock", "oos", "promoted", "promo"}
    false_values = {"0", "false", "no", "n", "f", "none", "nan", ""}
    result = lower.map(lambda value: True if value in true_values else False if value in false_values else np.nan)
    if result.isna().any():
        numeric = pd.to_numeric(series, errors="coerce")
        result = result.fillna(numeric.fillna(0) > 0)
    return result.astype(bool)


def analyze_timeseries(data: pd.DataFrame) -> pd.DataFrame:
    """Analyze time-series structure and quality."""
    duplicate_count = int(data.duplicated(["store_id", "sku_id", "timestamp"]).sum())
    series = data.groupby(["store_id", "sku_id"], dropna=False)
    intervals = series["timestamp"].diff().dropna()
    interval_hours = intervals.dt.total_seconds() / 3600
    most_common_interval = interval_hours.round(4).mode()
    sequence_lengths = series.size()
    rows = [
        {"metric": "rows", "value": int(len(data))},
        {"metric": "stores", "value": int(data["store_id"].nunique())},
        {"metric": "skus", "value": int(data["sku_id"].nunique())},
        {"metric": "categories", "value": int(data["category"].nunique())},
        {"metric": "date_min", "value": str(data["timestamp"].min())},
        {"metric": "date_max", "value": str(data["timestamp"].max())},
        {"metric": "duplicate_store_sku_timestamp_rows", "value": duplicate_count},
        {"metric": "sequence_count", "value": int(len(sequence_lengths))},
        {"metric": "sequence_length_min", "value": int(sequence_lengths.min())},
        {"metric": "sequence_length_median", "value": float(sequence_lengths.median())},
        {"metric": "sequence_length_max", "value": int(sequence_lengths.max())},
        {"metric": "most_common_interval_hours", "value": float(most_common_interval.iloc[0]) if not most_common_interval.empty else np.nan},
        {"metric": "irregular_interval_rate", "value": irregular_interval_rate(interval_hours)},
        {"metric": "inference_inventory_timing", "value": infer_inventory_timing(data)},
    ]
    output = pd.DataFrame(rows)
    output.to_csv(TIMESERIES_QUALITY_PATH, index=False)
    return output


def irregular_interval_rate(interval_hours: pd.Series) -> float:
    """Estimate irregular interval share."""
    clean = interval_hours.dropna()
    if clean.empty:
        return np.nan
    mode = clean.round(4).mode()
    if mode.empty:
        return np.nan
    return float((clean.round(4) != mode.iloc[0]).mean())


def infer_inventory_timing(data: pd.DataFrame) -> str:
    """Try to infer inventory timing without fabricating certainty."""
    grouped = data.groupby(["store_id", "sku_id"], dropna=False)
    prev_inventory = grouped["inventory"].shift(1)
    current_sales = data["sales_qty"]
    current_inventory = data["inventory"]
    implied_after = (prev_inventory - current_sales - current_inventory).abs()
    rate = float((implied_after.dropna() <= 1e-6).mean()) if implied_after.notna().any() else np.nan
    if np.isfinite(rate) and rate > 0.8:
        return "inventory_may_be_post_interval_or_end_of_interval"
    return "insufficient_metadata_to_determine_inventory_timing"


def analyze_stockouts(data: pd.DataFrame) -> pd.DataFrame:
    """Create stockout diagnosis summary."""
    ordered = data.sort_values(["store_id", "sku_id", "timestamp"]).copy()
    ordered["previous_sales_qty"] = ordered.groupby(["store_id", "sku_id"])["sales_qty"].shift(1)
    ordered["positive_sales_before_stockout"] = ordered["stockout"] & (ordered["previous_sales_qty"] > 0)
    sequence_stockout = ordered.groupby(["store_id", "sku_id"])["stockout"].mean()
    durations = stockout_durations(ordered)
    rows = [
        {"metric": "rows", "value": int(len(ordered))},
        {"metric": "stockout_observation_share", "value": float(ordered["stockout"].mean())},
        {"metric": "zero_stock_share", "value": float((ordered["inventory"] <= 0).mean())},
        {"metric": "zero_sales_share", "value": float((ordered["sales_qty"] <= 0).mean())},
        {"metric": "zero_stock_and_zero_sales_share", "value": float(((ordered["inventory"] <= 0) & (ordered["sales_qty"] <= 0)).mean())},
        {"metric": "positive_sales_immediately_before_stockout_share", "value": float(ordered["positive_sales_before_stockout"].mean())},
        {"metric": "sequence_stockout_frequency_mean", "value": float(sequence_stockout.mean())},
        {"metric": "sequences_with_any_stockout", "value": int((sequence_stockout > 0).sum())},
        {"metric": "stockout_duration_count", "value": int(len(durations))},
        {"metric": "stockout_duration_median_observations", "value": float(np.median(durations)) if durations else 0.0},
        {"metric": "stockout_duration_max_observations", "value": int(max(durations)) if durations else 0},
        {"metric": "sales_appear_censored_during_stockout", "value": infer_censoring(ordered)},
    ]
    output = pd.DataFrame(rows)
    output.to_csv(STOCKOUT_SUMMARY_PATH, index=False)
    return output


def stockout_durations(data: pd.DataFrame) -> list[int]:
    """Calculate consecutive stockout run lengths by store-SKU."""
    durations: list[int] = []
    for _, group in data.groupby(["store_id", "sku_id"], sort=False):
        run = 0
        for flag in group["stockout"].astype(bool).tolist():
            if flag:
                run += 1
            elif run:
                durations.append(run)
                run = 0
        if run:
            durations.append(run)
    return durations


def infer_censoring(data: pd.DataFrame) -> str:
    """Infer whether observed sales appear censored during stockouts."""
    if data["stockout"].sum() == 0:
        return "no_stockout_observations"
    zero_sales_when_stockout = float((data.loc[data["stockout"], "sales_qty"] <= 0).mean())
    if zero_sales_when_stockout > 0.8:
        return "likely_observed_sales_censored_or_zero_during_stockout"
    return "not_clear_from_available_fields"


def analyze_promotions(data: pd.DataFrame) -> pd.DataFrame:
    """Analyze promotion and discount support."""
    discount = data["discount"]
    markdown = data["markdown_rate"].fillna(0.0)
    data = data.copy()
    data["discount_bin"] = pd.cut(
        markdown,
        bins=practical_markdown_bins(markdown),
        include_lowest=True,
        duplicates="drop",
    ).astype(str)
    series = data.groupby(["store_id", "sku_id"], dropna=False)
    rows: list[dict[str, Any]] = [
        {"scope": "overall", "metric": "rows", "value": int(len(data))},
        {"scope": "overall", "metric": "promotion_observation_share", "value": float(data["promotion"].mean())},
        {"scope": "overall", "metric": "discount_min", "value": float(discount.min())},
        {"scope": "overall", "metric": "discount_median", "value": float(discount.median())},
        {"scope": "overall", "metric": "discount_max", "value": float(discount.max())},
        {"scope": "overall", "metric": "markdown_rate_min", "value": float(markdown.min())},
        {"scope": "overall", "metric": "markdown_rate_median", "value": float(markdown.median())},
        {"scope": "overall", "metric": "markdown_rate_max", "value": float(markdown.max())},
        {"scope": "overall", "metric": "series_with_promoted_and_nonpromoted", "value": int(series["promotion"].agg(lambda x: x.nunique() > 1).sum())},
        {"scope": "overall", "metric": "series_with_multiple_discount_levels", "value": int(series["discount"].nunique().gt(1).sum())},
        {"scope": "overall", "metric": "promotion_and_stockout_overlap_share", "value": float((data["promotion"] & data["stockout"]).mean())},
    ]
    for bin_name, count in data["discount_bin"].value_counts(dropna=False).sort_index().items():
        rows.append({"scope": "discount_bin", "metric": str(bin_name), "value": int(count)})
    by_category = (
        data.groupby("category", dropna=False)
        .agg(
            rows=("sales_qty", "size"),
            promotion_rate=("promotion", "mean"),
            stockout_rate=("stockout", "mean"),
            mean_discount=("discount", "mean"),
            mean_markdown_rate=("markdown_rate", "mean"),
        )
        .reset_index()
    )
    for _, row in by_category.iterrows():
        rows.append({"scope": f"category:{row['category']}", "metric": "rows", "value": int(row["rows"])})
        rows.append({"scope": f"category:{row['category']}", "metric": "promotion_rate", "value": float(row["promotion_rate"])})
        rows.append({"scope": f"category:{row['category']}", "metric": "mean_discount", "value": float(row["mean_discount"])})
        rows.append({"scope": f"category:{row['category']}", "metric": "mean_markdown_rate", "value": float(row["mean_markdown_rate"])})
    output = pd.DataFrame(rows)
    output.to_csv(PROMOTION_SUPPORT_PATH, index=False)
    return output


def practical_markdown_bins(markdown: pd.Series) -> list[float]:
    """Create practical markdown-rate bins from observed data."""
    clean = pd.to_numeric(markdown, errors="coerce").fillna(0.0).clip(lower=0, upper=1)
    max_value = float(clean.max())
    base = [0.0, 0.001, 0.05, 0.10, 0.20, 0.30, 0.50, max(1.0, max_value)]
    return sorted(set(base))


def select_modeling_subset(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select complete store-SKU sequences for modeling-oriented analysis."""
    data = data.sort_values(["store_id", "sku_id", "timestamp"]).copy()
    expected_length = int(data.groupby(["store_id", "sku_id"], dropna=False).size().mode().iloc[0])
    series_summary = build_series_summary(data, expected_length)
    eligible = series_summary.loc[
        (series_summary["n_observations"] >= expected_length)
        & (series_summary["interval_ok"])
        & (series_summary["sales_std"].fillna(0) > 0)
    ].copy()
    if eligible.empty:
        eligible = series_summary.loc[series_summary["n_observations"] >= expected_length - 2].copy()
    if eligible.empty:
        raise ValueError("No complete or near-complete store-SKU sequences available for modeling subset.")

    target = min(MAX_MODELING_SERIES, max(TARGET_MODELING_SERIES, MIN_MODELING_SERIES))
    selected_series = stratified_series_sample(eligible, target)
    if len(selected_series) < MIN_MODELING_SERIES:
        raise ValueError(f"Only {len(selected_series)} eligible series selected; need at least {MIN_MODELING_SERIES}.")

    subset = data.merge(selected_series[["store_id", "sku_id"]], on=["store_id", "sku_id"], how="inner")
    subset = subset.sort_values(["store_id", "sku_id", "timestamp"]).reset_index(drop=True)
    subset = add_time_split(subset)
    selected_series = build_series_summary(subset, expected_length)
    selected_series.to_csv(MODELING_SERIES_PATH, index=False)
    save_modeling_subset(subset)
    return subset, selected_series


def build_series_summary(data: pd.DataFrame, expected_length: int) -> pd.DataFrame:
    """Build store-SKU level diagnostics used for stratified sampling."""
    summary = (
        data.groupby(["store_id", "sku_id"], dropna=False)
        .agg(
            category=("category", first_mode),
            n_observations=("sales_qty", "size"),
            sales_mean=("sales_qty", "mean"),
            sales_std=("sales_qty", "std"),
            sales_total=("sales_qty", "sum"),
            inventory_mean=("inventory", "mean"),
            stockout_rate=("stockout", "mean"),
            promotion_rate=("promotion", "mean"),
            markdown_mean=("markdown_rate", "mean"),
            markdown_median=("markdown_rate", "median"),
            markdown_levels=("markdown_rate", "nunique"),
            start_time=("timestamp", "min"),
            end_time=("timestamp", "max"),
        )
        .reset_index()
    )
    intervals = data.groupby(["store_id", "sku_id"], dropna=False)["timestamp"].diff().dt.days
    interval_ok = intervals.dropna().eq(1).groupby([data.loc[intervals.notna(), "store_id"], data.loc[intervals.notna(), "sku_id"]]).all()
    interval_ok = interval_ok.rename("interval_ok").reset_index()
    summary = summary.merge(interval_ok, on=["store_id", "sku_id"], how="left")
    summary["interval_ok"] = summary["interval_ok"].fillna(summary["n_observations"].le(1)).astype(bool)
    summary["complete_sequence"] = summary["n_observations"].ge(expected_length) & summary["interval_ok"]
    summary["stockout_group"] = rate_group(summary["stockout_rate"], "low", "medium", "high")
    summary["promotion_group"] = pd.cut(
        summary["promotion_rate"],
        bins=[-0.001, 0.05, 0.50, 1.0],
        labels=["none_or_very_low", "medium", "high"],
        include_lowest=True,
    ).astype(str)
    summary["sales_group"] = quantile_group(summary["sales_mean"], "low", "medium", "high")
    summary["markdown_group"] = pd.cut(
        summary["markdown_mean"].fillna(0.0),
        bins=[-0.001, 0.001, 0.10, 1.0],
        labels=["no_markdown", "mild_markdown", "moderate_deep_markdown"],
        include_lowest=True,
    ).astype(str)
    return summary


def rate_group(values: pd.Series, low: str, medium: str, high: str) -> pd.Series:
    """Group a rate into low, medium, high tertiles."""
    return quantile_group(values, low, medium, high)


def quantile_group(values: pd.Series, low: str, medium: str, high: str) -> pd.Series:
    """Robust tertile grouping with duplicate-safe fallbacks."""
    clean = pd.to_numeric(values, errors="coerce")
    try:
        grouped = pd.qcut(clean.rank(method="first"), q=3, labels=[low, medium, high])
        return grouped.astype(str)
    except ValueError:
        return pd.cut(clean, bins=3, labels=[low, medium, high], include_lowest=True).astype(str)


def stratified_series_sample(series_summary: pd.DataFrame, target: int) -> pd.DataFrame:
    """Sample series across stockout, promotion, sales, and markdown strata."""
    strata_cols = ["stockout_group", "promotion_group", "sales_group", "markdown_group"]
    selected_parts: list[pd.DataFrame] = []
    grouped = series_summary.groupby(strata_cols, dropna=False)
    for _, group in grouped:
        quota = max(1, int(round(len(group) / len(series_summary) * target)))
        selected_parts.append(group.sample(n=min(quota, len(group)), random_state=RANDOM_SEED))
    selected = pd.concat(selected_parts, ignore_index=True).drop_duplicates(["store_id", "sku_id"])
    if len(selected) > target:
        selected = selected.sample(n=target, random_state=RANDOM_SEED).reset_index(drop=True)
    elif len(selected) < target:
        remaining = series_summary.merge(selected[["store_id", "sku_id"]], on=["store_id", "sku_id"], how="left", indicator=True)
        remaining = remaining.loc[remaining["_merge"].eq("left_only")].drop(columns="_merge")
        if not remaining.empty:
            selected = pd.concat(
                [selected, remaining.sample(n=min(target - len(selected), len(remaining)), random_state=RANDOM_SEED)],
                ignore_index=True,
            )
    return selected.reset_index(drop=True)


def first_mode(series: pd.Series) -> Any:
    """Return the first modal value."""
    mode = series.mode(dropna=True)
    if mode.empty:
        return series.dropna().iloc[0] if series.dropna().size else "unknown"
    return mode.iloc[0]


def trim_to_continuous_window(data: pd.DataFrame, min_days: int) -> pd.DataFrame:
    """Trim to a continuous time window if possible."""
    if data.empty:
        return data
    date_min = data["timestamp"].min()
    date_max = data["timestamp"].max()
    if (date_max - date_min).days < min_days:
        return data.sort_values(["store_id", "sku_id", "timestamp"]).reset_index(drop=True)
    start = date_min
    end = start + pd.Timedelta(days=min_days)
    window = data.loc[(data["timestamp"] >= start) & (data["timestamp"] <= end)].copy()
    if window["store_id"].nunique() >= 1 and window["sku_id"].nunique() >= 1:
        return window.sort_values(["store_id", "sku_id", "timestamp"]).reset_index(drop=True)
    return data.sort_values(["store_id", "sku_id", "timestamp"]).reset_index(drop=True)


def preserve_pilot_subset() -> None:
    """Preserve the previous small subset as inspection-only pilot data."""
    if PILOT_SUBSET_PATH.exists():
        return
    if OLD_STUDENT_SUBSET_PATH.exists():
        pilot = pd.read_parquet(OLD_STUDENT_SUBSET_PATH)
        pilot.to_parquet(PILOT_SUBSET_PATH, index=False)
        print(f"Preserved inspection-only pilot subset: {PILOT_SUBSET_PATH}")


def add_time_split(subset: pd.DataFrame) -> pd.DataFrame:
    """Add global chronological train/validation/test split labels."""
    output = subset.copy()
    dates = pd.Series(sorted(output["timestamp"].dt.floor("D").dropna().unique()))
    if dates.empty:
        output["time_split"] = "unknown"
        return output
    train_end = dates.iloc[max(0, math.ceil(len(dates) * 0.60) - 1)]
    validation_end = dates.iloc[max(0, math.ceil(len(dates) * 0.80) - 1)]
    output["time_split"] = np.select(
        [
            output["timestamp"].dt.floor("D") <= train_end,
            output["timestamp"].dt.floor("D") <= validation_end,
        ],
        ["train", "validation"],
        default="test",
    )
    return output


def save_modeling_subset(subset: pd.DataFrame) -> Path:
    """Save modeling subset as Parquet."""
    subset.to_parquet(MODELING_SUBSET_PATH, index=False)
    print(f"Saved modeling subset parquet: {MODELING_SUBSET_PATH}")
    return MODELING_SUBSET_PATH


def summarize_dataset(full: pd.DataFrame, subset: pd.DataFrame) -> pd.DataFrame:
    """Summarize full loaded metadata and selected subset."""
    rows = summary_rows(full, "loaded_full_selected_columns") + summary_rows(subset, "modeling_subset")
    output = pd.DataFrame(rows)
    output.to_csv(MODELING_SUMMARY_PATH, index=False)
    return output


def summary_rows(data: pd.DataFrame, scope: str) -> list[dict[str, Any]]:
    """Create dataset summary rows."""
    sequence_lengths = data.groupby(["store_id", "sku_id"], dropna=False).size()
    discount = data["discount"]
    markdown = data["markdown_rate"]
    return [
        {"scope": scope, "metric": "rows", "value": int(len(data))},
        {"scope": scope, "metric": "stores", "value": int(data["store_id"].nunique())},
        {"scope": scope, "metric": "skus", "value": int(data["sku_id"].nunique())},
        {"scope": scope, "metric": "categories", "value": int(data["category"].nunique())},
        {"scope": scope, "metric": "time_min", "value": str(data["timestamp"].min())},
        {"scope": scope, "metric": "time_max", "value": str(data["timestamp"].max())},
        {"scope": scope, "metric": "stockout_rate", "value": float(data["stockout"].mean())},
        {"scope": scope, "metric": "promotion_rate", "value": float(data["promotion"].mean())},
        {"scope": scope, "metric": "discount_min", "value": float(discount.min())},
        {"scope": scope, "metric": "discount_median", "value": float(discount.median())},
        {"scope": scope, "metric": "discount_max", "value": float(discount.max())},
        {"scope": scope, "metric": "markdown_rate_min", "value": float(markdown.min())},
        {"scope": scope, "metric": "markdown_rate_median", "value": float(markdown.median())},
        {"scope": scope, "metric": "markdown_rate_max", "value": float(markdown.max())},
        {"scope": scope, "metric": "mean_sales", "value": float(data["sales_qty"].mean())},
        {"scope": scope, "metric": "median_sales", "value": float(data["sales_qty"].median())},
        {"scope": scope, "metric": "inventory_median", "value": float(data["inventory"].median())},
        {"scope": scope, "metric": "zero_inventory_rate", "value": float(data["zero_inventory_flag"].mean())},
        {"scope": scope, "metric": "possible_stockout_rate", "value": float(data["possible_stockout_flag"].mean())},
        {"scope": scope, "metric": "sequence_length_min", "value": int(sequence_lengths.min()) if not sequence_lengths.empty else 0},
        {"scope": scope, "metric": "sequence_length_median", "value": float(sequence_lengths.median()) if not sequence_lengths.empty else 0.0},
        {"scope": scope, "metric": "sequence_length_max", "value": int(sequence_lengths.max()) if not sequence_lengths.empty else 0},
        {"scope": scope, "metric": "memory_mb", "value": float(data.memory_usage(deep=True).sum() / (1024**2))},
    ]


def audit_discount_semantics(data: pd.DataFrame) -> pd.DataFrame:
    """Audit whether discount behaves like a price multiplier."""
    discount = pd.to_numeric(data["discount"], errors="coerce")
    markdown = pd.to_numeric(data["markdown_rate"], errors="coerce")
    metadata_files = list(PROJECT_ROOT.rglob("*README*")) + list(PROJECT_ROOT.rglob("*metadata*")) + list(PROJECT_ROOT.rglob("*dictionary*"))
    local_metadata_hits: list[str] = []
    for path in metadata_files:
        if path.is_file() and path.suffix.lower() in {".md", ".txt", ".csv", ".json"}:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore").lower()
            except OSError:
                continue
            if "discount" in text:
                local_metadata_hits.append(str(path.relative_to(PROJECT_ROOT)))
    within_multiplier = discount.between(0, 1, inclusive="both")
    rows = [
        {"metric": "rows", "value": int(len(data)), "notes": ""},
        {"metric": "discount_non_missing_share", "value": float(discount.notna().mean()), "notes": ""},
        {"metric": "discount_min", "value": float(discount.min()), "notes": ""},
        {"metric": "discount_p01", "value": float(discount.quantile(0.01)), "notes": ""},
        {"metric": "discount_median", "value": float(discount.median()), "notes": ""},
        {"metric": "discount_p99", "value": float(discount.quantile(0.99)), "notes": ""},
        {"metric": "discount_max", "value": float(discount.max()), "notes": ""},
        {"metric": "discount_equal_1_share", "value": float(discount.eq(1.0).mean()), "notes": "Consistent with no-discount multiplier if interpretation is correct."},
        {"metric": "discount_between_0_and_1_share", "value": float(within_multiplier.mean()), "notes": "Rows where markdown_rate = 1 - discount is within [0, 1]."},
        {"metric": "discount_above_1_share", "value": float(discount.gt(1.0).mean()), "notes": "Above-one values prevent certainty about simple multiplier semantics."},
        {"metric": "markdown_rate_non_missing_share", "value": float(markdown.notna().mean()), "notes": "markdown_rate set only where 0 <= discount <= 1."},
        {"metric": "markdown_rate_min", "value": float(markdown.min()), "notes": ""},
        {"metric": "markdown_rate_median", "value": float(markdown.median()), "notes": ""},
        {"metric": "markdown_rate_p90", "value": float(markdown.quantile(0.90)), "notes": ""},
        {"metric": "markdown_rate_max", "value": float(markdown.max()), "notes": ""},
        {"metric": "local_metadata_discount_hits", "value": len(local_metadata_hits), "notes": "; ".join(local_metadata_hits[:10]) or "No local data dictionary found defining discount."},
        {"metric": "discount_semantics_status", "value": "uncertain_multiplier_like", "notes": "Observed values mostly support multiplier behavior, but local metadata does not prove the field definition."},
        {"metric": "stock_hour6_22_cnt_semantics_status", "value": "uncertain_count_or_duration", "notes": "Name and integer range suggest a count over hours 6-22, but no local dictionary was found."},
    ]
    audit = pd.DataFrame(rows)
    audit.to_csv(DISCOUNT_SEMANTICS_AUDIT_PATH, index=False)
    return audit


def compare_representativeness(full: pd.DataFrame, subset: pd.DataFrame) -> pd.DataFrame:
    """Compare full-data and modeling-subset distributions."""
    metrics = [
        ("stockout_rate", full["stockout"].mean(), subset["stockout"].mean()),
        ("promotion_rate", full["promotion"].mean(), subset["promotion"].mean()),
        ("markdown_rate_mean", full["markdown_rate"].mean(), subset["markdown_rate"].mean()),
        ("markdown_rate_median", full["markdown_rate"].median(), subset["markdown_rate"].median()),
        ("sales_mean", full["sales_qty"].mean(), subset["sales_qty"].mean()),
        ("sales_median", full["sales_qty"].median(), subset["sales_qty"].median()),
        ("inventory_mean", full["inventory"].mean(), subset["inventory"].mean()),
        ("inventory_median", full["inventory"].median(), subset["inventory"].median()),
        ("sequence_length_median", full.groupby(["store_id", "sku_id"]).size().median(), subset.groupby(["store_id", "sku_id"]).size().median()),
        ("stores", full["store_id"].nunique(), subset["store_id"].nunique()),
        ("skus", full["sku_id"].nunique(), subset["sku_id"].nunique()),
    ]
    rows = []
    for metric, full_value, subset_value in metrics:
        rows.append(
            {
                "metric": metric,
                "full_value": float(full_value),
                "subset_value": float(subset_value),
                "absolute_difference": float(abs(subset_value - full_value)),
            }
        )
    output = pd.DataFrame(rows)
    output.to_csv(REPRESENTATIVENESS_PATH, index=False)
    return output


def summarize_temporal_split(subset: pd.DataFrame) -> pd.DataFrame:
    """Summarize chronological train/validation/test split."""
    rows = []
    for split, group in subset.groupby("time_split", sort=False):
        rows.append(
            {
                "time_split": split,
                "rows": int(len(group)),
                "series": int(group[["store_id", "sku_id"]].drop_duplicates().shape[0]),
                "date_min": str(group["timestamp"].min()),
                "date_max": str(group["timestamp"].max()),
                "stockout_rate": float(group["stockout"].mean()),
                "promotion_rate": float(group["promotion"].mean()),
                "markdown_rate_median": float(group["markdown_rate"].median()),
            }
        )
    output = pd.DataFrame(rows)
    output.to_csv(TEMPORAL_SPLIT_SUMMARY_PATH, index=False)
    return output


def create_figures(data: pd.DataFrame, subset: pd.DataFrame, selected_series: pd.DataFrame) -> None:
    """Generate FreshRetailNet processing and modeling subset figures."""
    sequence_lengths = data.groupby(["store_id", "sku_id"], dropna=False).size()
    plot_hist(sequence_lengths, "Sequence-length distribution", "Observations per store-SKU", "Count", "sequence_length_distribution.png")
    stockout_by_series = data.groupby(["store_id", "sku_id"], dropna=False)["stockout"].mean()
    plot_hist(stockout_by_series, "Stockout-rate distribution by series", "Stockout rate", "Count", "stockout_rate_distribution_by_series.png")
    plot_hist(data["markdown_rate"].fillna(0.0), "Markdown-rate distribution", "Markdown rate", "Count", "discount_distribution.png")
    promotion_by_category = data.groupby("category", dropna=False)["promotion"].mean().sort_values(ascending=False).head(30)
    plot_bar(promotion_by_category, "Promotion rate by category", "Category", "Promotion rate", "promotion_rate_by_category.png")
    plot_example_series(subset, require_promotion=False, require_stockout=False, filename="example_sales_inventory_timeseries.png")
    plot_example_series(subset, require_promotion=True, require_stockout=True, filename="example_promotion_stockout_timeseries.png")
    category_counts = selected_series["category"].value_counts().head(30)
    plot_bar(category_counts, "Subset category composition", "Category", "Selected series count", "subset_category_composition.png")
    create_modeling_figures(data, subset, selected_series)


def create_modeling_figures(data: pd.DataFrame, subset: pd.DataFrame, selected_series: pd.DataFrame) -> None:
    """Generate required modeling-subset diagnostic figures."""
    plot_rate_comparison(
        "stockout_rate",
        float(data["stockout"].mean()),
        float(subset["stockout"].mean()),
        "Full-data vs subset stockout rate",
        "stockout_rate_comparison.png",
    )
    plot_rate_comparison(
        "promotion_rate",
        float(data["promotion"].mean()),
        float(subset["promotion"].mean()),
        "Full-data vs subset promotion rate",
        "promotion_rate_comparison.png",
    )
    plot_overlay_hist(
        data["markdown_rate"].fillna(0.0),
        subset["markdown_rate"].fillna(0.0),
        "Full-data vs subset markdown distribution",
        "Markdown rate",
        "markdown_distribution_comparison.png",
    )
    plot_overlay_hist(
        data["sales_qty"],
        subset["sales_qty"],
        "Full-data vs subset sales distribution",
        "Sales quantity",
        "sales_distribution_comparison.png",
        upper_quantile=0.99,
    )
    plot_stockout_promotion_scatter(selected_series)
    plot_example_complete_series(subset)
    plot_time_split(subset)


def plot_rate_comparison(metric: str, full_value: float, subset_value: float, title: str, filename: str) -> None:
    """Save full vs subset bar comparison."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["full_data", "modeling_subset"], [full_value, subset_value], color=["tab:blue", "tab:orange"])
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    for i, value in enumerate([full_value, subset_value]):
        ax.text(i, value + 0.02, f"{value:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(MODELING_FIGURES_DIR / filename, dpi=160)
    plt.close()


def plot_overlay_hist(
    full_values: pd.Series,
    subset_values: pd.Series,
    title: str,
    xlabel: str,
    filename: str,
    upper_quantile: float | None = None,
) -> None:
    """Save full vs subset distribution comparison."""
    full_clean = pd.to_numeric(full_values, errors="coerce").dropna()
    subset_clean = pd.to_numeric(subset_values, errors="coerce").dropna()
    if upper_quantile is not None and not full_clean.empty:
        cap = full_clean.quantile(upper_quantile)
        full_clean = full_clean.clip(upper=cap)
        subset_clean = subset_clean.clip(upper=cap)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(full_clean, bins=40, alpha=0.45, label="full_data", density=True)
    ax.hist(subset_clean, bins=40, alpha=0.45, label="modeling_subset", density=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    plt.savefig(MODELING_FIGURES_DIR / filename, dpi=160)
    plt.close()


def plot_stockout_promotion_scatter(selected_series: pd.DataFrame) -> None:
    """Save selected-series stockout vs promotion scatter."""
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        selected_series["promotion_rate"],
        selected_series["stockout_rate"],
        c=selected_series["markdown_mean"].fillna(0.0),
        cmap="viridis",
        s=35,
        alpha=0.8,
    )
    ax.set_xlabel("Promotion rate")
    ax.set_ylabel("Stockout rate")
    ax.set_title("Selected-series stockout vs promotion")
    plt.colorbar(scatter, ax=ax, label="Mean markdown rate")
    plt.tight_layout()
    plt.savefig(MODELING_FIGURES_DIR / "selected_series_stockout_vs_promotion.png", dpi=160)
    plt.close()


def plot_example_complete_series(subset: pd.DataFrame) -> None:
    """Save one complete 97-day example series."""
    longest = subset.groupby(["store_id", "sku_id"], sort=False).size().sort_values(ascending=False)
    if longest.empty:
        return
    store_id, sku_id = longest.index[0]
    group = subset.loc[(subset["store_id"] == store_id) & (subset["sku_id"] == sku_id)].copy()
    group = group.sort_values("timestamp")
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(group["timestamp"], group["sales_qty"], color="tab:blue", label="sales_qty")
    ax1.set_ylabel("Sales quantity")
    ax2 = ax1.twinx()
    ax2.plot(group["timestamp"], group["inventory"], color="tab:orange", label="inventory")
    ax2.set_ylabel("Inventory")
    ax1.set_title(f"Example complete series: store={store_id}, sku={sku_id}, days={len(group)}")
    ax1.set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(MODELING_FIGURES_DIR / "example_complete_97_day_series.png", dpi=160)
    plt.close()


def plot_time_split(subset: pd.DataFrame) -> None:
    """Save train/validation/test date split figure."""
    daily = subset.groupby([subset["timestamp"].dt.floor("D"), "time_split"]).size().reset_index(name="rows")
    fig, ax = plt.subplots(figsize=(11, 4))
    for split, group in daily.groupby("time_split", sort=False):
        ax.bar(group["timestamp"], group["rows"], label=split, width=0.8)
    ax.set_title("Train-validation-test chronological split")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rows")
    ax.legend()
    plt.tight_layout()
    plt.savefig(MODELING_FIGURES_DIR / "train_validation_test_date_split.png", dpi=160)
    plt.close()


def plot_hist(values: pd.Series, title: str, xlabel: str, ylabel: str, filename: str) -> None:
    """Save histogram."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(pd.to_numeric(values, errors="coerce").dropna(), bins=40)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=160)
    plt.close()


def plot_bar(values: pd.Series, title: str, xlabel: str, ylabel: str, filename: str) -> None:
    """Save bar chart."""
    fig, ax = plt.subplots(figsize=(11, 5))
    values.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=160)
    plt.close()


def plot_example_series(
    subset: pd.DataFrame,
    require_promotion: bool,
    require_stockout: bool,
    filename: str,
) -> None:
    """Plot one example store-SKU sales/inventory time series."""
    candidates = []
    for key, group in subset.groupby(["store_id", "sku_id"], sort=False):
        if require_promotion and not group["promotion"].any():
            continue
        if require_stockout and not group["stockout"].any():
            continue
        candidates.append((key, group))
    if not candidates:
        candidates = list(subset.groupby(["store_id", "sku_id"], sort=False))
    if not candidates:
        return
    (store_id, sku_id), group = candidates[0]
    group = group.sort_values("timestamp")
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(group["timestamp"], group["sales_qty"], label="sales_qty", color="tab:blue")
    ax1.set_ylabel("Sales quantity", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(group["timestamp"], group["inventory"], label="inventory", color="tab:orange")
    ax2.set_ylabel("Inventory", color="tab:orange")
    promo = group.loc[group["promotion"]]
    stockout = group.loc[group["stockout"]]
    if not promo.empty:
        ax1.scatter(promo["timestamp"], promo["sales_qty"], marker="o", color="tab:green", label="promotion", s=20)
    if not stockout.empty:
        ax1.scatter(stockout["timestamp"], stockout["sales_qty"], marker="x", color="tab:red", label="stockout", s=35)
    ax1.set_title(f"Example store-SKU time series: store={store_id}, sku={sku_id}")
    ax1.set_xlabel("Timestamp")
    ax1.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=160)
    plt.close()


def final_status(mapping: dict[str, str], subset: pd.DataFrame, representativeness: pd.DataFrame) -> str:
    """Determine final FreshRetailNet modeling-subset status."""
    missing = require_critical_mappings(mapping)
    if missing:
        return "FRESHRETAIL_MODELING_SUBSET_REQUIRES_REVISION"
    series_count = subset[["store_id", "sku_id"]].drop_duplicates().shape[0]
    has_stockout_variation = subset["stockout"].nunique(dropna=True) > 1
    has_promotion_variation = subset["promotion"].nunique(dropna=True) > 1
    has_splits = set(subset["time_split"].dropna().unique()) == {"train", "validation", "test"}
    stockout_diff = metric_difference(representativeness, "stockout_rate")
    promotion_diff = metric_difference(representativeness, "promotion_rate")
    if (
        MODELING_SUBSET_PATH.exists()
        and series_count >= MIN_MODELING_SERIES
        and has_splits
        and has_stockout_variation
        and has_promotion_variation
        and stockout_diff <= 0.05
        and promotion_diff <= 0.05
    ):
        return "FRESHRETAIL_MODELING_SUBSET_READY"
    return "FRESHRETAIL_MODELING_SUBSET_REQUIRES_REVISION"


def metric_difference(representativeness: pd.DataFrame, metric: str) -> float:
    """Fetch absolute representativeness difference for a metric."""
    row = representativeness.loc[representativeness["metric"].eq(metric)]
    if row.empty:
        return float("inf")
    return float(row["absolute_difference"].iloc[0])


def print_final_report(
    status: str,
    files: list[FileInfo],
    mapping: dict[str, str],
    subset: pd.DataFrame,
    representativeness: pd.DataFrame,
    temporal_split: pd.DataFrame,
    discount_audit: pd.DataFrame,
) -> None:
    """Print final processing report."""
    print("\nFreshRetailNet processing report")
    print("Input files used:")
    for item in files:
        print(f"- {item.relative_path}")
    print("Confirmed mappings:")
    for concept, column in sorted(mapping.items()):
        print(f"- {concept}: {column}")
    print(f"Modeling subset path: {MODELING_SUBSET_PATH}")
    print(f"Subset rows: {len(subset)}")
    print(f"Number of series: {subset[['store_id', 'sku_id']].drop_duplicates().shape[0]}")
    print(f"Stores: {subset['store_id'].nunique()}")
    print(f"SKUs: {subset['sku_id'].nunique()}")
    print(f"Date range: {subset['timestamp'].min()} to {subset['timestamp'].max()}")
    print(f"Stockout rate: {subset['stockout'].mean():.4f}")
    print(f"Promotion rate: {subset['promotion'].mean():.4f}")
    quantiles = subset["markdown_rate"].quantile([0, 0.25, 0.5, 0.75, 0.9, 1.0]).to_dict()
    print("Markdown-rate quantiles:", {str(k): round(float(v), 4) for k, v in quantiles.items()})
    print("Train/validation/test rows:", temporal_split.set_index("time_split")["rows"].to_dict())
    print("Representativeness differences:")
    for _, row in representativeness.iterrows():
        if row["metric"] in {"stockout_rate", "promotion_rate", "markdown_rate_median", "sales_mean", "inventory_median"}:
            print(f"- {row['metric']}: {float(row['absolute_difference']):.4f}")
    unresolved = discount_audit.loc[discount_audit["metric"].str.contains("semantics_status", na=False)]
    print("Unresolved field-semantics issues:")
    for _, row in unresolved.iterrows():
        print(f"- {row['metric']}: {row['value']} ({row['notes']})")
    print("Key limitations:")
    print("- Column mappings are inferred from file samples and should be reviewed.")
    print("- Inventory timing is only inferred when the data structure supports it.")
    print("- This module creates a modeling-ready subset only; it does not train demand or RL models.")
    print("Recommended next module: FreshRetailNet demand recovery / discount-response calibration.")
    print(status)


def main() -> None:
    """Run FreshRetailNet processing workflow."""
    ensure_dirs()
    files = discover_files()
    _, mapping_df, mapping = inspect_schemas(files)
    missing = require_critical_mappings(mapping)
    if missing:
        print(f"Critical mappings missing or ambiguous: {missing}")
        print("Review:", COLUMN_MAPPING_PATH)
        print("FRESHRETAIL_MODELING_SUBSET_REQUIRES_REVISION")
        return
    data = load_selected_columns(files, mapping)
    analyze_timeseries(data)
    analyze_stockouts(data)
    analyze_promotions(data)
    preserve_pilot_subset()
    discount_audit = audit_discount_semantics(data)
    subset, selected_series = select_modeling_subset(data)
    summarize_dataset(data, subset)
    representativeness = compare_representativeness(data, subset)
    temporal_split = summarize_temporal_split(subset)
    create_figures(data, subset, selected_series)
    status = final_status(mapping, subset, representativeness)
    if mapping_df["status"].eq("ambiguous").any() and status == "FRESHRETAIL_MODELING_SUBSET_READY":
        print("Warning: some non-critical mappings remain ambiguous; review mapping table.")
    print_final_report(status, files, mapping, subset, representativeness, temporal_split, discount_audit)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FreshRetailNet processing failed: {exc}")
        raise
