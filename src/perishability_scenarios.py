"""文件作用：实验顺序 04，定义易腐库存与经济参数的 semi-synthetic scenarios。
研究目的：补充 FreshRetailNet 未记录的 shelf life、库存年龄、waste 和成本变量。
主要输入：recovered demand、response curves 和数据中的商品/库存特征。
主要输出：scenario grid、变量来源表、FEFO 检查和环境 master config。
该模块不训练 RL；模拟变量必须与真实观测变量明确区分。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib  # noqa: F401  # allowed dependency; models are referenced, not retrained here.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RANDOM_SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "freshretail" / "processed" / "freshretail_demand_recovered.parquet"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
CONFIGS_DIR = PROJECT_ROOT / "outputs" / "configs"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "perishability_scenarios"

CURVES_PATH = TABLES_DIR / "discount_response_curves.csv"
SUPPORT_PATH = TABLES_DIR / "discount_response_support_audit.csv"
OBSERVED_MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "discount_response_observed.joblib"
RECOVERED_MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "discount_response_recovered.joblib"

PRODUCT_AUDIT_PATH = TABLES_DIR / "perishability_product_metadata_audit.csv"
SOURCE_REGISTRY_PATH = TABLES_DIR / "perishability_variable_source_registry.csv"
MARKDOWN_AUDIT_PATH = TABLES_DIR / "perishability_markdown_action_audit.csv"
SHELF_LIFE_SCENARIOS_PATH = TABLES_DIR / "perishability_shelf_life_scenarios.csv"
PRODUCT_ASSIGNMENT_PATH = TABLES_DIR / "perishability_product_scenario_assignment.csv"
INVENTORY_SCENARIOS_PATH = TABLES_DIR / "perishability_inventory_initialization_scenarios.csv"
ECONOMIC_SCENARIOS_PATH = TABLES_DIR / "perishability_economic_scenarios.csv"
SUSTAINABILITY_REWARD_SCENARIOS_PATH = TABLES_DIR / "perishability_sustainability_reward_scenarios.csv"
CORE_GRID_PATH = TABLES_DIR / "perishability_core_scenario_grid.csv"
SENSITIVITY_GRID_PATH = TABLES_DIR / "perishability_sensitivity_scenario_grid.csv"
STRESS_GRID_PATH = TABLES_DIR / "perishability_stress_test_grid.csv"
VALIDATION_PATH = TABLES_DIR / "perishability_pre_environment_validation.csv"
IDENTITY_CHECKS_PATH = TABLES_DIR / "perishability_accounting_identity_checks.csv"
SENSITIVITY_PATH = TABLES_DIR / "perishability_pre_environment_sensitivity.csv"

TRANSITION_SPEC_PATH = CONFIGS_DIR / "perishability_transition_specification.json"
ACCOUNTING_SPEC_PATH = CONFIGS_DIR / "perishability_accounting_specification.json"
FINANCIAL_REWARD_SPEC_PATH = CONFIGS_DIR / "perishability_financial_reward_specification.json"
SUSTAINABILITY_REWARD_SPEC_PATH = CONFIGS_DIR / "perishability_sustainability_reward_specification.json"
MASTER_CONFIG_PATH = CONFIGS_DIR / "perishability_master_config.json"

DEFAULT_ACTION_LEVELS = [0.00, 0.05, 0.10, 0.20, 0.30, 0.40]
DEFAULT_SHELF_LIFE_CLASS = "medium_shelf_life"
DEFAULT_INVENTORY_COVERAGE = "balanced_inventory"
DEFAULT_AGE_PROFILE = "uniform"
DEFAULT_MARGIN = "medium_margin"
DEFAULT_DISPOSAL = "moderate_disposal_cost"


@dataclass(frozen=True)
class Scenario:
    """One fixed-policy simulation scenario."""

    scenario_id: str
    grid_type: str
    shelf_life_class: str
    inventory_coverage: str
    age_profile: str
    margin_scenario: str
    disposal_scenario: str
    demand_target: str
    response_model: str
    lambda_waste: float


# 创建本模块需要的输出目录。
def ensure_dirs() -> None:
    """Create output directories."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# 读取 inputs 数据并返回统一结构。
def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load recovered data and discount-response artifacts."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Recovered-demand dataset not found: {DATA_PATH}")
    data = pd.read_parquet(DATA_PATH).copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data["store_id"] = data["store_id"].astype(str)
    if "product_id" not in data.columns:
        data["product_id"] = data["sku_id"].astype(str)
    data["product_id"] = data["product_id"].astype(str)
    for column in ["observed_sales_demand", "recovered_demand", "stock_hour6_22_cnt", "markdown_rate"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["activity_flag"] = as_bool(data["activity_flag"]).astype(int)
    data["possible_stockout_flag"] = as_bool(data["possible_stockout_flag"]).astype(int)

    curves = pd.read_csv(CURVES_PATH) if CURVES_PATH.exists() else pd.DataFrame()
    support = pd.read_csv(SUPPORT_PATH) if SUPPORT_PATH.exists() else pd.DataFrame()
    if curves.empty:
        raise FileNotFoundError("discount_response_curves.csv is required for model-implied simulation checks.")
    return data, curves, support


# 把常见布尔值写法统一转换为布尔序列。
def as_bool(series: pd.Series) -> pd.Series:
    """Convert mixed bool-like values to bool."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    text = series.astype(str).str.lower().str.strip()
    parsed = text.map(lambda value: value in {"1", "true", "yes", "y", "t"})
    numeric = pd.to_numeric(series, errors="coerce")
    return parsed | numeric.fillna(0).gt(0)


def product_metadata_audit(data: pd.DataFrame) -> pd.DataFrame:
    """Inspect available product identifiers and category fields."""
    category_candidates = [
        col
        for col in data.columns
        if any(token in col.lower() for token in ["category", "department", "class", "type"])
    ]
    rows: list[dict[str, Any]] = [
        {
            "audit_item": "product_identifier",
            "field": "product_id",
            "available": True,
            "value": int(data["product_id"].nunique()),
            "notes": "Numeric/anonymized product identifier; no food category is inferred.",
        },
        {
            "audit_item": "store_identifier",
            "field": "store_id",
            "available": True,
            "value": int(data["store_id"].nunique()),
            "notes": "Store identifier available for grouping.",
        },
    ]
    if category_candidates:
        for col in category_candidates:
            rows.append(
                {
                    "audit_item": "category_field",
                    "field": col,
                    "available": True,
                    "value": int(data[col].nunique(dropna=True)),
                    "notes": "Available as local field; semantic category names are not inferred unless explicit.",
                }
            )
            counts = data[["product_id", col]].drop_duplicates().groupby(col)["product_id"].nunique()
            for category, count in counts.items():
                rows.append(
                    {
                        "audit_item": "products_by_category",
                        "field": col,
                        "available": True,
                        "value": int(count),
                        "notes": str(category),
                    }
                )
    else:
        rows.append(
            {
                "audit_item": "category_field",
                "field": "none",
                "available": False,
                "value": 0,
                "notes": "No reliable local product category metadata found; neutral scenario classes will be used.",
            }
        )
    output = pd.DataFrame(rows)
    output.to_csv(PRODUCT_AUDIT_PATH, index=False)
    return output


def variable_source_registry() -> pd.DataFrame:
    """Separate observed, model-estimated, derived, and assumed variables."""
    # 逐项登记 observed、derived、model-estimated 和 simulated 变量。
    # 这是项目可解释性的关键边界：最终 waste/profit 来自受控情景，
    # 不能表述为 FreshRetailNet 直接提供的真实报废量或零售商利润。
    rows = [
        ("date", "FreshRetailNet processing", "observed", "Calendar date from dt/timestamp.", "low", "Operational time index only."),
        ("store_id", "FreshRetailNet processing", "observed", "Anonymized store identifier.", "low", "Grouping key; no store semantics inferred."),
        ("product_id", "FreshRetailNet processing", "observed", "Anonymized product identifier.", "low", "Grouping key; no food category inferred."),
        ("observed_sales_demand", "latent_demand_recovery", "observed", "Observed sales quantity from sale_amount/sales_qty.", "medium", "Observed sales, not confirmed latent demand."),
        ("stock_hour6_22_cnt", "FreshRetailNet processing", "observed", "Inventory-related count/status field.", "high", "Inventory context only; semantics partly uncertain."),
        ("activity_flag", "FreshRetailNet processing", "observed", "Promotion/activity flag.", "medium", "Promotion context, not causal treatment."),
        ("discount", "FreshRetailNet processing", "observed", "Multiplier-like discount field.", "medium", "Preserved original field; semantics uncertain_multiplier_like."),
        ("avg_temperature", "FreshRetailNet processing", "observed", "Weather covariate.", "low", "Context feature."),
        ("recovered_demand", "latent_demand_recovery", "model_estimated", "Stockout-aware recovered demand estimate.", "high", "Model estimate, not observed truth."),
        ("model-implied markdown response", "discount_response", "model_estimated", "Observed/recovered target response models.", "high", "Observational model-implied curves, not causal counterfactuals."),
        ("markdown_rate", "FreshRetailNet processing", "derived", "1 - discount.", "medium", "Derived markdown rate under multiplier-like interpretation."),
        ("lagged features", "latent_demand_recovery / discount_response", "derived", "Prior-period sales/inventory features.", "medium", "Leakage-safe history variables."),
        ("possible_stockout_flag", "FreshRetailNet processing", "derived", "Diagnostic stockout-risk flag.", "high", "Not ground-truth stockout or latent demand."),
        ("shelf_life_days", "perishability_scenarios", "scenario_assumption", "Short/medium/long shelf-life classes.", "high", "Scenario assumption only."),
        ("age distribution", "perishability_scenarios", "scenario_assumption", "Fresh-heavy/uniform/near-expiry inventory buckets.", "high", "Simulated initial age composition."),
        ("procurement cost ratio", "perishability_scenarios", "scenario_assumption", "Normalized procurement cost / full price.", "high", "Economic scenario parameter."),
        ("disposal cost ratio", "perishability_scenarios", "scenario_assumption", "Disposal handling cost / full price.", "high", "Economic scenario parameter."),
        ("expiration rule", "perishability_scenarios", "scenario_assumption", "Units with one period remaining expire after the period.", "high", "Simulation rule."),
        ("waste penalty", "perishability_scenarios", "scenario_assumption", "Lambda weight on physical waste.", "high", "Reward-scenario parameter."),
        ("terminal salvage value", "perishability_scenarios", "scenario_assumption", "No/partial/full terminal valuation.", "high", "Diagnostic scenario parameter."),
    ]
    output = pd.DataFrame(
        rows,
        columns=["variable", "module", "source_type", "source_description", "uncertainty", "allowed_interpretation"],
    )
    output.to_csv(SOURCE_REGISTRY_PATH, index=False)
    return output


def markdown_action_audit(data: pd.DataFrame) -> pd.DataFrame:
    """Audit observed markdown support and high-markdown concentration."""
    bins = [
        ("markdown_rate == 0", data["markdown_rate"].eq(0)),
        ("0 < markdown_rate <= 0.10", data["markdown_rate"].gt(0) & data["markdown_rate"].le(0.10)),
        ("0.10 < markdown_rate <= 0.20", data["markdown_rate"].gt(0.10) & data["markdown_rate"].le(0.20)),
        ("0.20 < markdown_rate <= 0.30", data["markdown_rate"].gt(0.20) & data["markdown_rate"].le(0.30)),
        ("0.30 < markdown_rate <= 0.40", data["markdown_rate"].gt(0.30) & data["markdown_rate"].le(0.40)),
        ("0.40 < markdown_rate <= 0.50", data["markdown_rate"].gt(0.40) & data["markdown_rate"].le(0.50)),
        ("0.50 < markdown_rate < 1.00", data["markdown_rate"].gt(0.50) & data["markdown_rate"].lt(1.00)),
        ("markdown_rate == 1.00", data["markdown_rate"].eq(1.00)),
    ]
    rows = []
    for label, mask in bins:
        group = data.loc[mask]
        rows.append(
            {
                "scope": "overall",
                "segment": label,
                "rows": int(len(group)),
                "share": float(len(group) / max(len(data), 1)),
                "store_count": int(group["store_id"].nunique()),
                "product_count": int(group["product_id"].nunique()),
                "notes": "100% markdown is not assumed to be a normal future commercial action.",
            }
        )
    high = data.loc[data["markdown_rate"].gt(0.40)].copy()
    for col in ["product_id", "store_id", "timestamp", "activity_flag", "possible_stockout_flag"]:
        top = high.groupby(col).size().sort_values(ascending=False).head(15)
        for key, count in top.items():
            rows.append(
                {
                    "scope": f"high_markdown_gt_0_40_by_{col}",
                    "segment": str(key),
                    "rows": int(count),
                    "share": float(count / max(len(high), 1)),
                    "store_count": np.nan,
                    "product_count": np.nan,
                    "notes": "Concentration audit for high markdown observations.",
                }
            )
    rows.append(
        {
            "scope": "future_action_support",
            "segment": ",".join(f"{x:.2f}" for x in DEFAULT_ACTION_LEVELS),
            "rows": len(data),
            "share": 1.0,
            "store_count": data["store_id"].nunique(),
            "product_count": data["product_id"].nunique(),
            "notes": "Default future action support excludes >40% markdown despite observed extremes.",
        }
    )
    output = pd.DataFrame(rows)
    output.to_csv(MARKDOWN_AUDIT_PATH, index=False)
    return output


def shelf_life_scenarios() -> pd.DataFrame:
    # 数据没有 expiration date，因此使用 short/medium/long 三类离散假设，
    # 并提供 sensitivity level。目的不是猜测单个 SKU 的真实保质期，
    # 而是检验 policy 在不同易腐速度下是否保持一致结论。
    """Create neutral shelf-life assumption classes and designs."""
    rows = []
    specs = {
        "short_shelf_life": {"low": 2, "base": 3, "high": 5},
        "medium_shelf_life": {"low": 5, "base": 7, "high": 10},
        "long_shelf_life": {"low": 10, "base": 14, "high": 21},
    }
    for shelf_class, values in specs.items():
        for level, days in values.items():
            rows.append(
                {
                    "shelf_life_class": shelf_class,
                    "scenario_level": level,
                    "shelf_life_days": days,
                    "source_type": "scenario_assumption",
                    "notes": "Not observed in FreshRetailNet.",
                }
            )
    output = pd.DataFrame(rows)
    output.to_csv(SHELF_LIFE_SCENARIOS_PATH, index=False)
    return output


def product_assignments(data: pd.DataFrame) -> pd.DataFrame:
    """Assign products to neutral shelf-life designs reproducibly."""
    product_stats = (
        data.groupby("product_id", dropna=False)
        .agg(
            rows=("observed_sales_demand", "size"),
            mean_recovered_demand=("recovered_demand", "mean"),
            std_recovered_demand=("recovered_demand", "std"),
        )
        .reset_index()
    )
    product_stats["demand_volatility"] = product_stats["std_recovered_demand"].fillna(0) / product_stats["mean_recovered_demand"].replace(0, np.nan)
    product_stats["demand_volatility"] = product_stats["demand_volatility"].fillna(0)
    classes = ["short_shelf_life", "medium_shelf_life", "long_shelf_life"]
    rows = []
    for _, row in product_stats.sort_values("product_id").iterrows():
        product_id = str(row["product_id"])
        idx = stable_index(product_id, len(classes))
        rows.append(
            {
                "product_id": product_id,
                "design": "homogeneous_medium",
                "shelf_life_class": "medium_shelf_life",
                "assignment_basis": "All products assigned the same neutral class.",
            }
        )
        rows.append(
            {
                "product_id": product_id,
                "design": "balanced_heterogeneous",
                "shelf_life_class": classes[idx],
                "assignment_basis": "Reproducible hash-balanced experimental assignment, not factual classification.",
            }
        )
    demand_rank = product_stats["mean_recovered_demand"].rank(pct=True)
    volatility_rank = product_stats["demand_volatility"].rank(pct=True)
    for i, row in product_stats.iterrows():
        if demand_rank.iloc[i] >= 0.67 or volatility_rank.iloc[i] >= 0.67:
            cls = "short_shelf_life"
        elif demand_rank.iloc[i] <= 0.33 and volatility_rank.iloc[i] <= 0.33:
            cls = "long_shelf_life"
        else:
            cls = "medium_shelf_life"
        rows.append(
            {
                "product_id": str(row["product_id"]),
                "design": "demand_informed_stress_test",
                "shelf_life_class": cls,
                "assignment_basis": "Stress-test assignment using demand/volatility ranks; not factual perishability.",
            }
        )
    output = pd.DataFrame(rows)
    output.to_csv(PRODUCT_ASSIGNMENT_PATH, index=False)
    return output


# 根据稳定哈希把标识映射到固定索引。
def stable_index(value: str, modulo: int) -> int:
    """Stable deterministic pseudo-hash."""
    return sum(ord(ch) for ch in value) % modulo


def inventory_initialization_scenarios(data: pd.DataFrame) -> pd.DataFrame:
    """Create inventory coverage and remaining-life bucket scenarios."""
    demand_q = data["recovered_demand"].quantile([0.05, 0.5, 0.95]).to_dict()
    inv_q = data["stock_hour6_22_cnt"].quantile([0.05, 0.5, 0.95]).to_dict()
    coverage = {
        "lean_inventory": (1.0, 1.5),
        "balanced_inventory": (2.0, 3.0),
        "high_inventory": (4.0, 6.0),
    }
    age_profiles = {
        "fresh_heavy": "Weights increase with remaining life.",
        "uniform": "Equal weight across remaining-life buckets.",
        "near_expiry_heavy": "Weights increase near expiration.",
    }
    rows = []
    for coverage_name, (low, high) in coverage.items():
        for profile, description in age_profiles.items():
            rows.append(
                {
                    "inventory_coverage_scenario": coverage_name,
                    "expected_demand_days_low": low,
                    "expected_demand_days_high": high,
                    "age_profile": profile,
                    "bucket_rule": description,
                    "lower_inventory_bound": float(max(0, inv_q.get(0.05, 0))),
                    "median_recovered_demand": float(demand_q.get(0.5, 0)),
                    "upper_inventory_bound": float(max(inv_q.get(0.95, 1), demand_q.get(0.95, 1) * high)),
                    "source_type": "scenario_assumption",
                    "notes": "Inventory is represented as non-negative remaining-life buckets summing to initial inventory.",
                }
            )
    output = pd.DataFrame(rows)
    output.to_csv(INVENTORY_SCENARIOS_PATH, index=False)
    return output


def economic_scenarios() -> pd.DataFrame:
    """Create normalized price, margin, disposal, and salvage scenarios."""
    margins = {
        "low_margin": 0.70,
        "medium_margin": 0.55,
        "high_margin": 0.40,
    }
    disposal = {
        "zero_disposal_cost": 0.00,
        "moderate_disposal_cost": 0.05,
        "high_disposal_cost": 0.10,
    }
    salvage = {
        "no_salvage_value": 0.00,
        "partial_salvage_value": 0.25,
        "full_normalized_salvage_diagnostic": 1.00,
    }
    rows = []
    for name, ratio in margins.items():
        rows.append({"scenario_type": "margin", "scenario_name": name, "value": ratio, "notes": "procurement_cost_ratio * initial_inventory, charged once."})
    for name, ratio in disposal.items():
        rows.append({"scenario_type": "disposal", "scenario_name": name, "value": ratio, "notes": "disposal_cost_ratio * expired units."})
    for name, ratio in salvage.items():
        rows.append({"scenario_type": "terminal_salvage", "scenario_name": name, "value": ratio, "notes": "Terminal inventory valuation; diagnostic use unless configured."})
    rows.append({"scenario_type": "price", "scenario_name": "reference_full_price", "value": 1.0, "notes": "selling_price = 1 - markdown_rate."})
    output = pd.DataFrame(rows)
    output.to_csv(ECONOMIC_SCENARIOS_PATH, index=False)
    return output


def reward_scenarios() -> pd.DataFrame:
    """Create sustainability reward lambda scenarios."""
    rows = []
    for lam in [0.00, 0.10, 0.25, 0.50, 1.00]:
        rows.append(
            {
                "lambda_waste": lam,
                "reward_family": "financial_plus_physical_waste_penalty",
                "normalization": "financial reward normalized by initial inventory; expired units normalized by initial inventory.",
                "equation": "R_sustainable = normalized_financial_reward - lambda_waste * normalized_expired_units",
                "notes": "Physical waste remains a separate outcome metric.",
            }
        )
    output = pd.DataFrame(rows)
    output.to_csv(SUSTAINABILITY_REWARD_SCENARIOS_PATH, index=False)
    return output


# 将 config specs 写入对应输出文件。
def write_config_specs() -> None:
    """Write transition, accounting, and reward specifications."""
    transition = {
        "not_gymnasium": True,
        "logic": [
            "Action sets markdown_rate.",
            "Demand is generated from selected model-implied response.",
            "Sales are min(demand, available_inventory).",
            "Inventory issued FEFO: first-expire-first-out.",
            "Unsold inventory ages by one period.",
            "Units reaching zero remaining life after aging become physical waste.",
            "Default experiment disables replenishment.",
            "No negative inventory is allowed.",
            "sales + ending_inventory + waste = beginning_inventory per period when no replenishment.",
        ],
        "pseudo_code": [
            "begin_inventory = sum(I_1..I_L)",
            "demand = demand_model(state, markdown_rate)",
            "sales = min(demand, begin_inventory)",
            "issue sales from I_1 then I_2 ... I_L",
            "waste = remaining I_1 after sales",
            "I'_k = remaining I_{k+1} for k=1..L-1",
            "I'_L = 0 when replenishment disabled",
            "assert begin_inventory == sales + waste + sum(I')",
        ],
    }
    accounting = {
        "full_price": 1.0,
        "selling_price": "1.0 - markdown_rate",
        "revenue": "sum(selling_price_t * units_sold_t)",
        "procurement_cost": "procurement_cost_ratio * initial_inventory",
        "procurement_cost_note": "Charged once for initial inventory; not charged again for sold or wasted units.",
        "disposal_cost": "disposal_cost_ratio * total_waste_units",
        "accounting_profit": "revenue - procurement_cost - disposal_cost + terminal_salvage_value",
        "physical_waste_units": "sum(expired_units_t)",
        "waste_rate": "physical_waste_units / initial_inventory",
        "sell_through_rate": "total_units_sold / initial_inventory",
        "terminal_inventory_treatment": "default no_salvage_value; sensitivity may use partial/full diagnostic salvage.",
    }
    financial_reward = {
        "per_period_reward": "sales_revenue_t - disposal_cost_t - initial_procurement_cost_if_first_period + terminal_salvage_if_terminal",
        "initial_procurement_cost": "procurement_cost_ratio * initial_inventory charged at reset or first period.",
        "avoid_double_counting": True,
        "sales_revenue_t": "(1 - markdown_rate_t) * units_sold_t",
        "disposal_cost_t": "disposal_cost_ratio * expired_units_t",
    }
    sustainable = {
        "financial_only": "R_financial = sales_revenue - initial_procurement_cost_when_applicable - disposal_cost",
        "financial_plus_waste": "R_sustainable = normalized_financial_reward - lambda_waste * normalized_expired_units",
        "constrained_later": "maximize accounting_profit subject to waste_rate <= threshold",
        "lambda_waste_values": [0.00, 0.10, 0.25, 0.50, 1.00],
        "normalization": "financial terms and expired units divided by initial_inventory to keep comparable scale.",
    }
    write_json(TRANSITION_SPEC_PATH, transition)
    write_json(ACCOUNTING_SPEC_PATH, accounting)
    write_json(FINANCIAL_REWARD_SPEC_PATH, financial_reward)
    write_json(SUSTAINABILITY_REWARD_SPEC_PATH, sustainable)


def scenario_grids() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create core, sensitivity, and stress-test grids."""
    # core grid 组合 shelf life、inventory coverage、age profile 和 cost assumptions；
    # sensitivity/stress grid 用于检查结论是否依赖单一参数设定。
    # scenario_id 被训练、validation 和最终 paired evaluation 共同引用。
    core_specs = []
    sid = 0
    for shelf in ["short_shelf_life", "medium_shelf_life", "long_shelf_life"]:
        for demand_target, response_model in [
            ("observed_sales_demand", "observed_target_response"),
            ("recovered_demand", "recovered_target_response"),
        ]:
            for inv in ["lean_inventory", "balanced_inventory", "high_inventory"]:
                sid += 1
                core_specs.append(
                    Scenario(
                        f"core_{sid:03d}",
                        "core",
                        shelf,
                        inv,
                        "uniform",
                        "medium_margin",
                        "moderate_disposal_cost",
                        demand_target,
                        response_model,
                        0.10,
                    )
                )
    sensitivity_specs = []
    sid = 0
    for age in ["fresh_heavy", "uniform", "near_expiry_heavy"]:
        for margin in ["low_margin", "medium_margin", "high_margin"]:
            for disposal in ["zero_disposal_cost", "moderate_disposal_cost", "high_disposal_cost"]:
                sid += 1
                sensitivity_specs.append(
                    Scenario(
                        f"sensitivity_{sid:03d}",
                        "sensitivity",
                        "medium_shelf_life",
                        "balanced_inventory",
                        age,
                        margin,
                        disposal,
                        "recovered_demand",
                        "recovered_target_response",
                        0.25,
                    )
                )
    stress_specs = []
    sid = 0
    for shelf, inv, age, lam in [
        ("short_shelf_life", "high_inventory", "near_expiry_heavy", 1.00),
        ("short_shelf_life", "balanced_inventory", "near_expiry_heavy", 0.50),
        ("long_shelf_life", "high_inventory", "fresh_heavy", 0.10),
        ("medium_shelf_life", "lean_inventory", "uniform", 0.00),
    ]:
        for demand_target, response_model in [
            ("observed_sales_demand", "observed_target_response"),
            ("recovered_demand", "recovered_target_response"),
        ]:
            sid += 1
            stress_specs.append(
                Scenario(
                    f"stress_{sid:03d}",
                    "stress",
                    shelf,
                    inv,
                    age,
                    "medium_margin",
                    "high_disposal_cost",
                    demand_target,
                    response_model,
                    lam,
                )
            )
    core = scenario_frame(core_specs)
    sensitivity = scenario_frame(sensitivity_specs)
    stress = scenario_frame(stress_specs)
    core.to_csv(CORE_GRID_PATH, index=False)
    sensitivity.to_csv(SENSITIVITY_GRID_PATH, index=False)
    stress.to_csv(STRESS_GRID_PATH, index=False)
    return core, sensitivity, stress


def scenario_frame(specs: list[Scenario]) -> pd.DataFrame:
    """Convert scenario dataclasses to frame."""
    return pd.DataFrame([s.__dict__ for s in specs])


def shelf_life_days(shelf_class: str, level: str = "base") -> int:
    """Return base shelf life for a class."""
    values = {
        "short_shelf_life": {"low": 2, "base": 3, "high": 5},
        "medium_shelf_life": {"low": 5, "base": 7, "high": 10},
        "long_shelf_life": {"low": 10, "base": 14, "high": 21},
    }
    return int(values[shelf_class][level])


def coverage_midpoint(name: str) -> float:
    """Return expected demand day coverage midpoint."""
    return {
        "lean_inventory": 1.25,
        "balanced_inventory": 2.50,
        "high_inventory": 5.00,
    }[name]


def margin_ratio(name: str) -> float:
    """Return procurement cost ratio."""
    return {"low_margin": 0.70, "medium_margin": 0.55, "high_margin": 0.40}[name]


def disposal_ratio(name: str) -> float:
    """Return disposal cost ratio."""
    return {"zero_disposal_cost": 0.00, "moderate_disposal_cost": 0.05, "high_disposal_cost": 0.10}[name]


def age_weights(profile: str, shelf_life: int) -> np.ndarray:
    """Create non-negative remaining-life bucket weights from I_1..I_L."""
    if profile == "fresh_heavy":
        weights = np.arange(1, shelf_life + 1, dtype=float)
    elif profile == "near_expiry_heavy":
        weights = np.arange(shelf_life, 0, -1, dtype=float)
    else:
        weights = np.ones(shelf_life, dtype=float)
    return weights / weights.sum()


def response_multiplier(curves: pd.DataFrame, demand_target: str, response_model: str, action: float) -> float:
    """Use prior discount-response curves as model-implied demand multiplier."""
    target = "recovered" if "recovered" in demand_target or "recovered" in response_model else "observed"
    target_curves = curves.loc[curves["target"].eq(target)].copy()
    if target_curves.empty:
        return 1.0
    by_level = target_curves.groupby("markdown_rate")["predicted_demand"].mean()
    base = float(by_level.loc[0.0]) if 0.0 in by_level.index else float(by_level.iloc[0])
    nearest = float(by_level.index[np.argmin(np.abs(by_level.index.to_numpy(dtype=float) - action))])
    return float(by_level.loc[nearest] / max(base, 1e-6))


def simulate_policy(
    series: pd.DataFrame,
    scenario: pd.Series,
    policy_name: str,
    curves: pd.DataFrame,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run deterministic FEFO validation simulation for one series/scenario/policy."""
    # 该小型 deterministic simulation 在正式 Gymnasium 环境之前检查 accounting identity
    # 和 FEFO 方向是否合理。库存按 remaining-life bucket 保存，销售优先扣减最临期批次，
    # 日末 bucket 0 计为 expired waste，再将剩余库存年龄前移。
    # 输出只用于 scenario readiness，不用于训练最终 policy。
    shelf = shelf_life_days(str(scenario["shelf_life_class"]))
    initial_demand = float(series["recovered_demand"].mean() if "recovered" in str(scenario["demand_target"]) else series["observed_sales_demand"].mean())
    total_inventory = initial_demand * coverage_midpoint(str(scenario["inventory_coverage"]))
    upper = float(max(series["stock_hour6_22_cnt"].quantile(0.95), initial_demand * coverage_midpoint(str(scenario["inventory_coverage"]))))
    total_inventory = float(np.clip(total_inventory, 0.1, max(0.1, upper)))
    inventory = total_inventory * age_weights(str(scenario["age_profile"]), shelf)
    initial_inventory = float(inventory.sum())
    procurement_cost = margin_ratio(str(scenario["margin_scenario"])) * initial_inventory
    disposal = disposal_ratio(str(scenario["disposal_scenario"]))
    revenue = 0.0
    total_sold = 0.0
    total_waste = 0.0
    max_identity_error = 0.0
    min_inventory_seen = float(inventory.min())
    period_rows: list[dict[str, Any]] = []

    for t, row in enumerate(series.sort_values("timestamp").itertuples(index=False)):
        begin_inventory = float(inventory.sum())
        action = policy_action(policy_name, inventory, initial_demand, shelf, t)
        multiplier = response_multiplier(curves, str(scenario["demand_target"]), str(scenario["response_model"]), action)
        base_demand = float(getattr(row, "recovered_demand") if "recovered" in str(scenario["demand_target"]) else getattr(row, "observed_sales_demand"))
        demand = max(0.0, base_demand * multiplier)
        units_sold = min(demand, begin_inventory)
        remaining_sale = units_sold
        for i in range(shelf):
            issued = min(inventory[i], remaining_sale)
            inventory[i] -= issued
            remaining_sale -= issued
            if remaining_sale <= 1e-9:
                break
        waste = float(inventory[0])
        aged = np.zeros_like(inventory)
        if shelf > 1:
            aged[:-1] = inventory[1:]
        inventory = np.maximum(aged, 0.0)
        ending_inventory = float(inventory.sum())
        identity_error = abs(begin_inventory - units_sold - waste - ending_inventory)
        max_identity_error = max(max_identity_error, identity_error)
        min_inventory_seen = min(min_inventory_seen, float(inventory.min()))
        total_sold += float(units_sold)
        total_waste += waste
        revenue += (1.0 - action) * float(units_sold)
        period_rows.append(
            {
                "timestamp": str(getattr(row, "timestamp")),
                "markdown_rate": action,
                "demand": demand,
                "units_sold": float(units_sold),
                "waste": waste,
                "ending_inventory": ending_inventory,
                "identity_error": identity_error,
            }
        )

    disposal_cost = disposal * total_waste
    accounting_profit = revenue - procurement_cost - disposal_cost
    result = {
        "scenario_id": scenario["scenario_id"],
        "grid_type": scenario["grid_type"],
        "policy": policy_name,
        "shelf_life_class": scenario["shelf_life_class"],
        "inventory_coverage": scenario["inventory_coverage"],
        "age_profile": scenario["age_profile"],
        "margin_scenario": scenario["margin_scenario"],
        "disposal_scenario": scenario["disposal_scenario"],
        "demand_target": scenario["demand_target"],
        "response_model": scenario["response_model"],
        "lambda_waste": float(scenario["lambda_waste"]),
        "initial_inventory": initial_inventory,
        "revenue": revenue,
        "procurement_cost": procurement_cost,
        "disposal_cost": disposal_cost,
        "accounting_profit": accounting_profit,
        "physical_waste_units": total_waste,
        "waste_rate": total_waste / max(initial_inventory, 1e-9),
        "sell_through_rate": total_sold / max(initial_inventory, 1e-9),
        "total_units_sold": total_sold,
        "terminal_inventory": float(inventory.sum()),
        "max_identity_error": max_identity_error,
        "min_inventory_seen": min_inventory_seen,
        "procurement_cost_charged_once": True,
        "terminal_salvage_value": 0.0,
    }
    return result, period_rows


def policy_action(policy_name: str, inventory: np.ndarray, expected_demand: float, shelf: int, t: int) -> float:
    """Fixed policy action function."""
    if policy_name == "always_0pct":
        return 0.00
    if policy_name == "always_10pct":
        return 0.10
    if policy_name == "always_20pct":
        return 0.20
    if policy_name == "always_40pct":
        return 0.40
    if policy_name == "expiry_threshold_rule":
        near_expiry_share = float(inventory[0] / max(inventory.sum(), 1e-9))
        if near_expiry_share > 0.35:
            return 0.40
        if near_expiry_share > 0.20:
            return 0.20
        return 0.05
    if policy_name == "inventory_coverage_rule":
        coverage = float(inventory.sum() / max(expected_demand, 1e-9))
        if coverage > 3.0:
            return 0.30
        if coverage > 1.5:
            return 0.10
        return 0.00
    raise ValueError(f"Unknown policy: {policy_name}")


def run_pre_environment_checks(data: pd.DataFrame, curves: pd.DataFrame, core_grid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run lightweight deterministic fixed-policy FEFO checks."""
    policies = ["always_0pct", "always_10pct", "always_20pct", "always_40pct", "expiry_threshold_rule", "inventory_coverage_rule"]
    sample_series_keys = (
        data.groupby(["store_id", "product_id"])["recovered_demand"].mean().sort_values(ascending=False).head(6).index.tolist()
    )
    rows = []
    identity_rows = []
    detailed = []
    for _, scenario in core_grid.head(18).iterrows():
        for store_id, product_id in sample_series_keys:
            series = data.loc[(data["store_id"].eq(store_id)) & (data["product_id"].eq(product_id))]
            if series.empty:
                continue
            for policy in policies:
                result, period = simulate_policy(series, scenario, policy, curves)
                result["store_id"] = store_id
                result["product_id"] = product_id
                rows.append(result)
                identity_rows.append(
                    {
                        "scenario_id": result["scenario_id"],
                        "policy": policy,
                        "store_id": store_id,
                        "product_id": product_id,
                        "max_identity_error": result["max_identity_error"],
                        "no_negative_inventory": result["min_inventory_seen"] >= -1e-9,
                        "inventory_conservation_pass": result["max_identity_error"] <= 1e-6,
                        "procurement_cost_charged_once": result["procurement_cost_charged_once"],
                        "waste_nonnegative": result["physical_waste_units"] >= -1e-9,
                    }
                )
                if len(detailed) < 2:
                    detailed.extend(period)
    validation = pd.DataFrame(rows)
    identity = pd.DataFrame(identity_rows)
    validation.to_csv(VALIDATION_PATH, index=False)
    identity.to_csv(IDENTITY_CHECKS_PATH, index=False)
    sensitivity = (
        validation.groupby(["shelf_life_class", "inventory_coverage", "age_profile", "margin_scenario", "disposal_scenario", "demand_target", "response_model", "policy"], dropna=False)
        .agg(
            mean_profit=("accounting_profit", "mean"),
            mean_waste_rate=("waste_rate", "mean"),
            mean_sell_through=("sell_through_rate", "mean"),
            scenarios=("scenario_id", "nunique"),
        )
        .reset_index()
    )
    sensitivity.to_csv(SENSITIVITY_PATH, index=False)
    return validation, identity, sensitivity


# 根据当前参数构建 figures。
def create_figures(
    shelf: pd.DataFrame,
    inventory: pd.DataFrame,
    economic: pd.DataFrame,
    validation: pd.DataFrame,
    identity: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> None:
    """Generate required matplotlib figures."""
    plot_shelf_life(shelf)
    plot_age_profiles()
    plot_economic(economic)
    plot_policy_metric(validation, "accounting_profit", "Fixed-policy profit comparison", "fixed_policy_profit_comparison.png")
    plot_policy_metric(validation, "waste_rate", "Fixed-policy waste comparison", "fixed_policy_waste_comparison.png")
    plot_profit_waste(validation)
    plot_observed_recovered_comparison(validation)
    plot_sensitivity(sensitivity, "shelf_life_class", "Sensitivity by shelf life", "sensitivity_by_shelf_life.png")
    plot_sensitivity(sensitivity, "inventory_coverage", "Sensitivity by inventory coverage", "sensitivity_by_inventory_coverage.png")
    plot_identity(identity)


# 根据现有表格绘制 shelf life 图。
def plot_shelf_life(shelf: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot = shelf.pivot(index="shelf_life_class", columns="scenario_level", values="shelf_life_days")
    pivot[["low", "base", "high"]].plot(kind="bar", ax=ax)
    ax.set_title("Shelf-life scenario overview")
    ax.set_ylabel("Days")
    plt.xticks(rotation=25, ha="right")
    savefig("shelf_life_scenario_overview.png")


# 根据现有表格绘制 age profiles 图。
def plot_age_profiles() -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    shelf = 7
    x = np.arange(1, shelf + 1)
    for profile in ["fresh_heavy", "uniform", "near_expiry_heavy"]:
        ax.plot(x, age_weights(profile, shelf), marker="o", label=profile)
    ax.set_title("Inventory age-profile examples")
    ax.set_xlabel("Remaining-life bucket")
    ax.set_ylabel("Initial inventory share")
    ax.legend()
    savefig("inventory_age_profile_examples.png")


# 根据现有表格绘制 economic 图。
def plot_economic(economic: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    rows = economic.loc[economic["scenario_type"].isin(["margin", "disposal"])]
    ax.bar(rows["scenario_name"], rows["value"])
    ax.set_title("Economic scenario comparison")
    ax.set_ylabel("Ratio to normalized full price")
    plt.xticks(rotation=30, ha="right")
    savefig("economic_scenario_comparison.png")


# 根据现有表格绘制 policy metric 图。
def plot_policy_metric(validation: pd.DataFrame, metric: str, title: str, filename: str) -> None:
    summary = validation.groupby("policy")[metric].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    summary.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel(metric)
    plt.xticks(rotation=30, ha="right")
    savefig(filename)


# 根据现有表格绘制 profit waste 图。
def plot_profit_waste(validation: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(validation["waste_rate"], validation["accounting_profit"], alpha=0.4, s=16)
    ax.set_title("Profit-versus-waste scatter")
    ax.set_xlabel("Waste rate")
    ax.set_ylabel("Accounting profit")
    savefig("profit_versus_waste_scatter.png")


# 根据现有表格绘制 observed recovered comparison 图。
def plot_observed_recovered_comparison(validation: pd.DataFrame) -> None:
    summary = validation.groupby(["demand_target", "policy"])["accounting_profit"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    for target, group in summary.groupby("demand_target"):
        ax.plot(group["policy"], group["accounting_profit"], marker="o", label=target)
    ax.set_title("Observed-demand versus recovered-demand scenario comparison")
    ax.set_ylabel("Mean accounting profit")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    savefig("observed_vs_recovered_scenario_comparison.png")


# 根据现有表格绘制 sensitivity 图。
def plot_sensitivity(sensitivity: pd.DataFrame, group_col: str, title: str, filename: str) -> None:
    summary = sensitivity.groupby(group_col)["mean_profit"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    summary.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Mean accounting profit")
    plt.xticks(rotation=25, ha="right")
    savefig(filename)


# 根据现有表格绘制 identity 图。
def plot_identity(identity: pd.DataFrame) -> None:
    summary = identity[["inventory_conservation_pass", "no_negative_inventory", "procurement_cost_charged_once", "waste_nonnegative"]].mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    summary.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Accounting identity validation summary")
    ax.set_ylabel("Pass share")
    plt.xticks(rotation=25, ha="right")
    savefig("accounting_identity_validation_summary.png")


# 调整布局并把当前图保存到主图目录。
def savefig(filename: str) -> None:
    """Save current figure."""
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=160)
    plt.close()


# 将 master config 写入对应输出文件。
def write_master_config(core: pd.DataFrame) -> None:
    """Save master scenario configuration."""
    config = {
        "action_levels": DEFAULT_ACTION_LEVELS,
        "shelf_life_assumptions": {
            "short_shelf_life": {"low": 2, "base": 3, "high": 5},
            "medium_shelf_life": {"low": 5, "base": 7, "high": 10},
            "long_shelf_life": {"low": 10, "base": 14, "high": 21},
            "source_type": "scenario_assumption",
        },
        "inventory_initialization_rules": "initial_inventory = expected_daily_demand * coverage_days, bounded by observed inventory/demand quantiles.",
        "age_profiles": ["fresh_heavy", "uniform", "near_expiry_heavy"],
        "FEFO_rule": "Issue sales from I_1, then I_2, ... I_L before aging.",
        "cost_scenarios": {
            "procurement_cost_ratio": {"low_margin": 0.70, "medium_margin": 0.55, "high_margin": 0.40},
            "disposal_cost_ratio": {"zero": 0.00, "moderate": 0.05, "high": 0.10},
        },
        "waste_definitions": "physical_waste_units are units expiring from I_1 after sales in each period.",
        "financial_reward_equation": "sales_revenue_t - disposal_cost_t - initial_procurement_cost_if_first_period + terminal_salvage_if_terminal",
        "sustainability_reward_equation": "normalized_financial_reward - lambda_waste * normalized_expired_units",
        "terminal_inventory_rule": "default no salvage value; sensitivity may apply terminal value.",
        "demand_model_references": {
            "observed_response_model": str(OBSERVED_MODEL_PATH.relative_to(PROJECT_ROOT)),
            "recovered_response_model": str(RECOVERED_MODEL_PATH.relative_to(PROJECT_ROOT)),
            "response_curves": str(CURVES_PATH.relative_to(PROJECT_ROOT)),
        },
        "random_seed": RANDOM_SEED,
        "source_type_classification": str(SOURCE_REGISTRY_PATH.relative_to(PROJECT_ROOT)),
        "core_scenario_count": int(len(core)),
        "limitations": limitations(),
    }
    write_json(MASTER_CONFIG_PATH, config)


# 将字典以缩进 JSON 格式写入文件。
def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with stable formatting."""
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


# 返回该模块对应的数据和模型限制说明。
def limitations() -> list[str]:
    """Required limitations."""
    return [
        "Shelf life is scenario-assumed, not observed.",
        "Inventory age composition is simulated.",
        "Realized waste is simulated.",
        "Procurement and disposal costs are normalized assumptions.",
        "Recovered demand is model-estimated.",
        "Markdown response is observational and model-implied.",
        "Results are suitable for controlled decision experiments.",
        "Results are not direct estimates of retailer profit or waste.",
    ]


def readiness_status(identity: pd.DataFrame, core: pd.DataFrame) -> str:
    """Determine readiness status."""
    configs_exist = all(
        path.exists()
        for path in [
            TRANSITION_SPEC_PATH,
            ACCOUNTING_SPEC_PATH,
            FINANCIAL_REWARD_SPEC_PATH,
            SUSTAINABILITY_REWARD_SPEC_PATH,
            MASTER_CONFIG_PATH,
        ]
    )
    tables_exist = all(
        path.exists()
        for path in [
            SOURCE_REGISTRY_PATH,
            MARKDOWN_AUDIT_PATH,
            CORE_GRID_PATH,
            VALIDATION_PATH,
            IDENTITY_CHECKS_PATH,
        ]
    )
    identity_pass = bool(
        identity["inventory_conservation_pass"].all()
        and identity["no_negative_inventory"].all()
        and identity["procurement_cost_charged_once"].all()
        and identity["waste_nonnegative"].all()
    )
    if configs_exist and tables_exist and identity_pass and 12 <= len(core) <= 36:
        return "PERISHABILITY_SCENARIOS_READY"
    return "PERISHABILITY_SCENARIOS_REQUIRE_REVISION"


def print_report(status: str, core: pd.DataFrame, validation: pd.DataFrame, identity: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    """Print readiness report."""
    print("\nPerishability scenario calibration report")
    print(f"Selected default shelf-life scenario: {DEFAULT_SHELF_LIFE_CLASS}")
    print(f"Selected default inventory coverage: {DEFAULT_INVENTORY_COVERAGE}")
    print(f"Selected default age profile: {DEFAULT_AGE_PROFILE}")
    print(f"Default margin assumption: {DEFAULT_MARGIN} ({margin_ratio(DEFAULT_MARGIN):.2f})")
    print(f"Default disposal-cost assumption: {DEFAULT_DISPOSAL} ({disposal_ratio(DEFAULT_DISPOSAL):.2f})")
    print(f"Default markdown action set: {DEFAULT_ACTION_LEVELS}")
    print(f"Number of core scenarios: {len(core)}")
    print("Accounting validation results:")
    print(
        identity[["inventory_conservation_pass", "no_negative_inventory", "procurement_cost_charged_once", "waste_nonnegative"]]
        .mean()
        .to_dict()
    )
    top_policy = validation.groupby("policy")["accounting_profit"].mean().sort_values(ascending=False).head(3).to_dict()
    waste_policy = validation.groupby("policy")["waste_rate"].mean().sort_values().head(3).to_dict()
    print(f"Key sensitivity finding - higher mean profit policies in validation simulation: {top_policy}")
    print(f"Key sensitivity finding - lower mean waste policies in validation simulation: {waste_policy}")
    print("Unresolved data-semantic issues:")
    print("- shelf life, realized waste, procurement cost, and disposal cost are not observed in FreshRetailNet.")
    print("- stock_hour6_22_cnt semantics remain partly uncertain.")
    print("- discount semantics remain uncertain_multiplier_like.")
    print("Recommended next module: pricing environment adapter; still no PPO until environment validation passes.")
    print("Files created:")
    for path in created_files():
        print(f"- {path.relative_to(PROJECT_ROOT)}")
    print(status)


# 汇总本次运行生成的文件路径。
def created_files() -> list[Path]:
    """Return expected created files."""
    return [
        PRODUCT_AUDIT_PATH,
        SOURCE_REGISTRY_PATH,
        MARKDOWN_AUDIT_PATH,
        SHELF_LIFE_SCENARIOS_PATH,
        PRODUCT_ASSIGNMENT_PATH,
        INVENTORY_SCENARIOS_PATH,
        TRANSITION_SPEC_PATH,
        ECONOMIC_SCENARIOS_PATH,
        ACCOUNTING_SPEC_PATH,
        FINANCIAL_REWARD_SPEC_PATH,
        SUSTAINABILITY_REWARD_SCENARIOS_PATH,
        SUSTAINABILITY_REWARD_SPEC_PATH,
        CORE_GRID_PATH,
        SENSITIVITY_GRID_PATH,
        STRESS_GRID_PATH,
        VALIDATION_PATH,
        IDENTITY_CHECKS_PATH,
        SENSITIVITY_PATH,
        MASTER_CONFIG_PATH,
    ]


def run() -> str:
    """Run full perishability scenario calibration."""
    # pipeline：变量来源审计 -> shelf/inventory/economic assumptions
    # -> scenario grids -> deterministic FEFO 检查 -> master config。
    # pricing_env_operational.py 读取这里锁定的情景，而不在训练中临时改变参数。
    ensure_dirs()
    data, curves, _support = load_inputs()
    product_audit = product_metadata_audit(data)
    registry = variable_source_registry()
    markdown_audit = markdown_action_audit(data)
    shelf = shelf_life_scenarios()
    assignment = product_assignments(data)
    inventory = inventory_initialization_scenarios(data)
    economic = economic_scenarios()
    rewards = reward_scenarios()
    write_config_specs()
    core, sensitivity_grid, stress = scenario_grids()
    validation, identity, sensitivity = run_pre_environment_checks(data, curves, core)
    create_figures(shelf, inventory, economic, validation, identity, sensitivity)
    write_master_config(core)
    status = readiness_status(identity, core)
    print_report(status, core, validation, identity, sensitivity)
    return status


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"Perishability scenario calibration failed: {exc}")
        print("PERISHABILITY_SCENARIOS_REQUIRE_REVISION")
        raise
