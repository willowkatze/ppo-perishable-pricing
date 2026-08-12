"""文件作用：实验顺序 05，构建易腐品动态 markdown 的 Gymnasium 环境。
研究目的：把需求恢复、折扣响应和半合成保质期连接为可重复的序贯决策问题。
主要输入：恢复需求数据、perishability 情景配置、observed/recovered 响应模型。
主要输出：41 维 state、6 个 action、financial reward、库存与浪费统计。
该环境用于受控实验，不代表零售商真实利润或可直接部署的经营系统。"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import joblib
import numpy as np
import pandas as pd
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Paths and fixed experiment settings
# ---------------------------------------------------------------------------

RANDOM_SEED = 42 #随机种子可复现
EPSILON = 1e-9
MAX_SHELF_LIFE = 21
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "freshretail" / "processed" / "freshretail_demand_recovered.parquet"
DEFAULT_SCENARIO_CONFIG_PATH = PROJECT_ROOT / "outputs" / "configs" / "perishability_master_config.json"
DEFAULT_RESPONSE_OBSERVED_PATH = PROJECT_ROOT / "outputs" / "models" / "discount_response_observed.joblib"
DEFAULT_RESPONSE_RECOVERED_PATH = PROJECT_ROOT / "outputs" / "models" / "discount_response_recovered.joblib"
DEFAULT_RESPONSE_CONFIG_PATH = PROJECT_ROOT / "outputs" / "models" / "discount_response_config.json"
CORE_GRID_PATH = PROJECT_ROOT / "outputs" / "tables" / "perishability_core_scenario_grid.csv"
SUPPORT_AUDIT_PATH = PROJECT_ROOT / "outputs" / "tables" / "discount_response_support_audit.csv"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
CONFIGS_DIR = PROJECT_ROOT / "outputs" / "configs"

ARTIFACT_AUDIT_PATH = TABLES_DIR / "pricing_env_artifact_compatibility_audit.csv"
FEATURE_SOURCE_AUDIT_PATH = TABLES_DIR / "pricing_env_feature_source_audit.csv"
ENV_CONFIG_PATH = CONFIGS_DIR / "pricing_env_operational_config.json"


# 离散动作对应 0%、5%、10%、20%、30%、40% markdown。
ACTION_MARKDOWNS = {
    0: 0.00,
    1: 0.05,
    2: 0.10,
    3: 0.20,
    4: 0.30,
    5: 0.40,
}

# state 同时描述库存年龄结构、需求历史、时间与商品背景。
# 其中 21 个 remaining-life bucket 让策略能区分“库存很多”和“临期库存很多”；
# predicted demand 与历史销量特征提供销售机会信息，其余特征控制时间和商品差异。
# 名称顺序必须与 _get_obs() 生成数组的顺序一致。
_OBSERVATION_NAMES = (
    "normalized_total_inventory",
    "inventory_coverage",
    *[f"remaining_life_bucket_{i}" for i in range(1, MAX_SHELF_LIFE + 1)],
    "fraction_expiring_today",
    "fraction_expiring_within_two_days",
    "weighted_mean_remaining_life",
    "predicted_zero_markdown_demand",
    "lag_1_simulated_target",
    "lag_7_simulated_target",
    "rolling_7_simulated_mean",
    "rolling_14_simulated_mean",
    "simulated_demand_volatility",
    "day_of_week",
    "weekend_indicator",
    "normalized_time_within_episode",
    "normalized_temperature",
    "previous_markdown_action",
    "previous_stockout_indicator",
    "calibration_mode_indicator",
    "store_encoding",
    "product_encoding",
)
_LEGACY_OBSERVATION_DIM = 41


# ---------------------------------------------------------------------------
# Artifact compatibility
# ---------------------------------------------------------------------------


#封装折扣响应模型、特征顺序和元数据；环境通过同一 bundle 读取模型，避免训练与评估使用不同特征定义。
@dataclass
class FittedResponseModel:
    """Compatibility shim for joblib artifacts saved by discount_response.py."""

    target_name: str
    model_type: str
    model: Any
    feature_columns: list[str]
    encoders: dict[str, dict[str, int]]
    prediction_cap: float


# ---------------------------------------------------------------------------
# Environment mechanics
# ---------------------------------------------------------------------------


# 实现 Gymnasium 易腐品定价环境；它把 state、markdown action、库存流转和 financial reward 连接成 agent
# 可交互的序贯决策问题。
class OperationalPerishablePricingEnv(gym.Env):
    """带 FEFO 库存流转的动态定价环境。

    每次 ``step`` 接收一个折扣 action，返回下一 state、reward、终止标记和
    accounting 信息；训练算法只通过这一标准 Gymnasium 接口与业务模拟交互。
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    # 保存环境配置，加载数据和模型，并初始化状态与动作空间。
    def __init__(
        self,
        data_path: str | Path = DEFAULT_DATA_PATH,
        scenario_config_path: str | Path = DEFAULT_SCENARIO_CONFIG_PATH,
        response_model_observed_path: str | Path = DEFAULT_RESPONSE_OBSERVED_PATH,
        response_model_recovered_path: str | Path = DEFAULT_RESPONSE_RECOVERED_PATH,
        response_model_config_path: str | Path = DEFAULT_RESPONSE_CONFIG_PATH,
        split: str = "train",
        calibration_mode: str = "recovered_calibration",
        reward_mode: str = "financial",
        lambda_waste: float = 0.10,
        scenario_id: str = "core_008",
        deterministic_demand: bool = True,
        residual_noise_mode: str = "none",
        random_seed: int = RANDOM_SEED,
        render_mode: str | None = None,
        promotion_context_mode: str = "derived_from_action",
        shelf_life_level: str = "base",
    ) -> None:
        super().__init__()
        self.data_path = Path(data_path)
        self.scenario_config_path = Path(scenario_config_path)
        self.response_model_observed_path = Path(response_model_observed_path)
        self.response_model_recovered_path = Path(response_model_recovered_path)
        self.response_model_config_path = Path(response_model_config_path)
        self.split = split
        self.calibration_mode = calibration_mode
        self.reward_mode = reward_mode
        self.lambda_waste = float(lambda_waste)
        self.scenario_id = scenario_id
        self.deterministic_demand = deterministic_demand
        self.residual_noise_mode = residual_noise_mode
        self.random_seed = int(random_seed)
        self.render_mode = render_mode
        self.promotion_context_mode = promotion_context_mode
        self.shelf_life_level = shelf_life_level

        self._validate_modes()
        self._load_artifacts()
        self.action_space = spaces.Discrete(6)
        low, high = observation_bounds()
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.np_random = np.random.default_rng(self.random_seed)
        self.episode_counter = 0
        self._reset_state()

    # 检查 split、calibration 和 reward mode 是否属于预先允许的组合。提前拒绝未知模式，避免悄悄改变实验定义。
    def _validate_modes(self) -> None:
        if self.split not in {"train", "validation", "test"}:
            raise ValueError(f"Unsupported split: {self.split}")
        valid_calibration = {
            "observed_calibration",
            "recovered_calibration",
            "observed_demand_recovered_response",
            "recovered_demand_observed_response",
        }
        if self.calibration_mode not in valid_calibration:
            raise ValueError(f"Unsupported calibration_mode: {self.calibration_mode}")
        if self.reward_mode not in {"financial", "sustainability"}:
            raise ValueError(f"Unsupported reward_mode: {self.reward_mode}")
        if self.promotion_context_mode not in {"derived_from_action", "historical_exogenous"}:
            raise ValueError(f"Unsupported promotion_context_mode: {self.promotion_context_mode}")
        if self.residual_noise_mode not in {"none", "bootstrap"}:
            raise ValueError(f"Unsupported residual_noise_mode: {self.residual_noise_mode}")

    # 加载场景表、折扣响应模型和模型配置。
    def _load_artifacts(self) -> None:
        for path in [
            self.data_path,
            self.scenario_config_path,
            self.response_model_observed_path,
            self.response_model_recovered_path,
            self.response_model_config_path,
            CORE_GRID_PATH,
        ]:
            if not path.exists():
                raise FileNotFoundError(f"Required artifact not found: {path}")
        self.data = pd.read_parquet(self.data_path).copy()
        self.data["timestamp"] = pd.to_datetime(self.data["timestamp"])
        self.data["store_id"] = self.data["store_id"].astype(str)
        self.data["product_id"] = self.data.get("product_id", self.data["sku_id"]).astype(str)
        self.data["activity_flag"] = self.data["activity_flag"].astype(int)
        self.data["possible_stockout_flag"] = self.data["possible_stockout_flag"].astype(int)
        self.response_config = json.loads(self.response_model_config_path.read_text(encoding="utf-8"))
        self.master_config = json.loads(self.scenario_config_path.read_text(encoding="utf-8"))
        self.scenario_grid = pd.read_csv(CORE_GRID_PATH)
        scenario = self.scenario_grid.loc[self.scenario_grid["scenario_id"].eq(self.scenario_id)]
        if scenario.empty:
            raise ValueError(f"Unsupported scenario_id: {self.scenario_id}")
        self.scenario = scenario.iloc[0].to_dict()
        self.observed_model = load_response_model(self.response_model_observed_path)
        self.recovered_model = load_response_model(self.response_model_recovered_path)
        self._audit_artifacts()
        self._write_feature_source_audit()
        self._write_env_config()

    # 训练前提前核对模型特征、数据列和配置是否兼容；
    def _audit_artifacts(self) -> None:
        rows = []
        for name, bundle, path in [
            ("observed", self.observed_model, self.response_model_observed_path),
            ("recovered", self.recovered_model, self.response_model_recovered_path),
        ]:
            rows.append(
                {
                    "artifact": name,
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "load_status": "loaded_with_FittedResponseModel_compatibility_shim",
                    "model_type": getattr(bundle, "model_type", "unknown"),
                    "feature_count": len(bundle.feature_columns),
                    "has_encoders": bool(bundle.encoders),
                    "prediction_cap": bundle.prediction_cap,
                    "self_contained_pipeline": False,
                    "notes": "Joblib artifact needs compatibility shim because wrapper class was saved from script context.",
                }
            )
        rows.append(
            {
                "artifact": "response_config",
                "path": str(self.response_model_config_path.relative_to(PROJECT_ROOT)),
                "load_status": "loaded",
                "model_type": json.dumps(self.response_config.get("selected_model_types", {})),
                "feature_count": len(self.response_config.get("feature_list", [])),
                "has_encoders": True,
                "prediction_cap": "",
                "self_contained_pipeline": False,
                "notes": "Config supplies feature list, split boundaries, support range, and clipping rules.",
            }
        )
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(ARTIFACT_AUDIT_PATH, index=False)

    # 记录每个 state 特征来自原始数据、恢复模型还是场景参数。
    def _write_feature_source_audit(self) -> None:
        rows = []
        sources = {
            "markdown_rate": "agent_action_current_step",
            "activity_flag": f"promotion_context_mode={self.promotion_context_mode}",
            "lag_1_observed_sales": "simulated_episode_history_after_reset",
            "lag_7_observed_sales": "pre_episode_history_until_available_then_simulated_history",
            "rolling_mean_7_observed_sales": "pre_episode_history_until_available_then_simulated_history",
            "rolling_mean_7_recovered_demand": "pre_episode_history_until_available_then_simulated_history",
            "rolling_mean_14_observed_sales": "pre_episode_history_until_available_then_simulated_history",
            "stock_hour6_22_cnt": "simulated_inventory_state",
            "previous_day_inventory": "simulated_inventory_state",
            "possible_stockout_flag": "simulated_inventory_state",
            "previous_day_possible_stockout_flag": "simulated_episode_history",
            "day_of_week": "current_episode_date",
            "week_index": "current_episode_date",
            "month": "current_episode_date",
            "weekend_flag": "current_episode_date",
            "trend_index": "current_episode_date",
            "avg_temperature": "historical_exogenous_current_date",
            "store_id_encoded": "saved_training_encoder",
            "product_id_encoded": "saved_training_encoder",
        }
        for feature in self.response_config.get("feature_list", []):
            rows.append(
                {
                    "feature": feature,
                    "source": sources.get(feature, "unknown"),
                    "leakage_status": "no_future_target_after_reset" if feature in sources else "requires_review",
                    "notes": "Dynamic lag features are updated from simulated episode outcomes after reset.",
                }
            )
        pd.DataFrame(rows).to_csv(FEATURE_SOURCE_AUDIT_PATH, index=False)

    # 将环境、action 和 reward 配置写入元数据文件。
    def _write_env_config(self) -> None:
        CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "observation_definition": observation_names(),
            "action_mapping": ACTION_MARKDOWNS,
            "calibration_modes": {
                "observed_calibration": "observed target demand history and observed response model",
                "recovered_calibration": "recovered target demand history and recovered response model",
                "cross_pair_modes": ["observed_demand_recovered_response", "recovered_demand_observed_response"],
            },
            "reward_equations": {
                "financial": "raw_financial_step / initial_inventory",
                "sustainability": "normalized_financial_step - lambda_waste * expired_units / initial_inventory",
            },
            "scenario_references": {
                "master_config": str(self.scenario_config_path.relative_to(PROJECT_ROOT)),
                "core_grid": str(CORE_GRID_PATH.relative_to(PROJECT_ROOT)),
            },
            "model_artifact_references": {
                "observed": str(self.response_model_observed_path.relative_to(PROJECT_ROOT)),
                "recovered": str(self.response_model_recovered_path.relative_to(PROJECT_ROOT)),
            },
            "promotion_context_rule": {
                "default": "derived_from_action",
                "derived_from_action": "activity_flag = 1 if markdown_rate > 0 else 0",
                "historical_exogenous": "uses current historical date's activity_flag for sensitivity only",
            },
            "dynamic_lag_update_rules": "pre-episode history initializes lags; post-reset lags use simulated outcomes only",
            "prediction_clipping": self.response_config.get("prediction_clipping_rules", {}),
            "residual_noise_calibration": "bootstrap train residuals not implemented by default; deterministic mode is default",
            "episode_construction": "one store-product series, one split-contained start date, horizon=shelf_life_days",
            "split_boundary_rules": "episodes never cross train/validation/test boundaries",
            "seeding": "Gymnasium reset(seed=...) controls episode sampling and deterministic trajectories",
            "terminal_rules": "terminate when inventory resolves or horizon reached; terminal salvage currently zero",
            "accounting_equations": "profit = revenue - procurement_cost - disposal_cost + terminal_salvage",
            "non_degeneracy_thresholds": {"dominant_action_warning_share": 0.90},
            "limitations": required_limitations(),
        }
        ENV_CONFIG_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    # 清空 episode 内部库存、历史需求和累计核算量；防止上一个 episode 的状态泄漏到下一个 episode。
    def _reset_state(self) -> None:
        self.current_series = pd.DataFrame()
        self.current_step = 0
        self.horizon = 0
        self.inventory = np.zeros(MAX_SHELF_LIFE, dtype=float)
        self.shelf_life = 0
        self.initial_inventory = 0.0
        self.procurement_cost_ratio = 0.55
        self.disposal_cost_ratio = 0.05
        self.initial_procurement_cost_pending = True
        self.cumulative_sales = 0.0
        self.cumulative_waste = 0.0
        self.cumulative_revenue = 0.0
        self.cumulative_procurement_cost = 0.0
        self.cumulative_disposal_cost = 0.0
        self.accounting_profit = 0.0
        self.episode_return = 0.0
        self.raw_financial_steps: list[float] = []
        self.reward_steps: list[float] = []
        self.sim_history: list[float] = []
        self.stockout_history: list[int] = []
        self.previous_markdown = 0.0
        self.last_info: dict[str, Any] = {}
        self.episode_start_date = None
        self.store_id = ""
        self.product_id = ""

    # 按固定 scenario 和起点建立新 episode，并返回首个 observation；固定 manifest 评估依赖这里复现完全相同的初始条件。
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self._reset_state()
        options = options or {}
        # episode 必须完整落在指定 split 内，避免训练状态跨入 validation/test。
        # 评估时可通过 options 固定 store-product-date，使所有 policy 面对同一 episode，
        # 从而形成 paired comparison（配对比较）。
        self.shelf_life = scenario_shelf_life(self.scenario["shelf_life_class"], self.shelf_life_level)
        split_data = self.data.loc[self.data["time_split"].eq(self.split)].copy()
        split_dates = sorted(split_data["timestamp"].dt.floor("D").unique())
        if len(split_dates) < self.shelf_life:
            raise ValueError(
                f"Scenario shelf life {self.shelf_life} days unsupported in split {self.split} with {len(split_dates)} dates."
            )
        eligible = eligible_episode_starts(split_data, self.shelf_life)
        if eligible.empty:
            raise ValueError(f"No eligible episode starts for split={self.split}, scenario={self.scenario_id}.")
        if {"store_id", "product_id", "start_date"}.issubset(options):
            mask = (
                eligible["store_id"].eq(str(options["store_id"]))
                & eligible["product_id"].eq(str(options["product_id"]))
                & eligible["start_date"].eq(pd.Timestamp(options["start_date"]))
            )
            selected = eligible.loc[mask]
            if selected.empty:
                raise ValueError("Requested paired episode start is not eligible for this split/scenario.")
            row = selected.iloc[0]
        elif "episode_index" in options:
            row = eligible.iloc[int(options["episode_index"]) % len(eligible)]
        else:
            row = eligible.iloc[int(self.np_random.integers(0, len(eligible)))]
        self.store_id = str(row["store_id"])
        self.product_id = str(row["product_id"])
        self.episode_start_date = pd.Timestamp(row["start_date"])
        end_date = self.episode_start_date + pd.Timedelta(days=self.shelf_life - 1)
        self.current_series = split_data.loc[
            split_data["store_id"].eq(self.store_id)
            & split_data["product_id"].eq(self.product_id)
            & split_data["timestamp"].between(self.episode_start_date, end_date)
        ].sort_values("timestamp").reset_index(drop=True)
        if len(self.current_series) != self.shelf_life:
            raise ValueError("Episode construction failed to produce full split-contained horizon.")
        self.horizon = self.shelf_life
        self._initialize_inventory(split_data)
        self._initialize_history(split_data)
        obs = self._get_obs()
        info = self._info(markdown_rate=0.0, selling_price=1.0, predicted_demand=0.0, sales=0.0, before=self.initial_inventory, after=self.initial_inventory, expired=0.0, raw_financial=0.0, terminated=False)
        self.last_info = info
        return obs, info

    # 按 shelf-life profile 把初始库存分配到不同年龄层；这是后续 FEFO、临期比例和过期浪费计算的基础。
    def _initialize_inventory(self, split_data: pd.DataFrame) -> None:
        target = self._target_column()
        hist = split_data.loc[split_data["store_id"].eq(self.store_id) & split_data["product_id"].eq(self.product_id), target]
        # FreshRetailNet 没有逐批次保质期库存，因此初始总库存和年龄结构来自透明的
        # semi-synthetic scenario。真实数据提供需求尺度，scenario 提供 coverage 和 age profile。
        expected = float(hist.mean()) if not hist.empty else float(split_data[target].mean())
        coverage = coverage_midpoint(self.scenario["inventory_coverage"])
        total = max(expected * coverage, 0.1)
        upper = max(float(split_data["stock_hour6_22_cnt"].quantile(0.95)), float(split_data[target].quantile(0.95)) * coverage, 0.1)
        total = float(np.clip(total, 0.1, upper))
        weights = age_weights(self.scenario["age_profile"], self.shelf_life)
        self.inventory[: self.shelf_life] = total * weights
        self.initial_inventory = float(self.inventory.sum())
        self.procurement_cost_ratio = margin_ratio(self.scenario["margin_scenario"])
        self.disposal_cost_ratio = disposal_ratio(self.scenario["disposal_scenario"])

    # 从 episode 起点前的数据建立需求历史；agent 因而只看到决策时点之前的信息，避免未来信息泄漏。
    def _initialize_history(self, split_data: pd.DataFrame) -> None:
        target = self._target_column()
        prior = split_data.loc[
            split_data["store_id"].eq(self.store_id)
            & split_data["product_id"].eq(self.product_id)
            & (split_data["timestamp"] < self.episode_start_date)
        ].sort_values("timestamp")
        values = prior[target].tail(14).astype(float).tolist()
        if not values:
            values = [float(split_data[target].mean())]
        while len(values) < 14:
            values.insert(0, values[0])
        self.sim_history = values[-14:]
        self.stockout_history = prior["possible_stockout_flag"].tail(1).astype(int).tolist() or [0]

    # 根据 observed 或 recovered calibration 选择需求目标；两种 PPO 的核心差异由这里进入环境。
    def _target_column(self) -> str:
        if self.calibration_mode in {"observed_calibration", "observed_demand_recovered_response"}:
            return "observed_sales_demand"
        return "recovered_demand"

    # 选择与 calibration mode 对应的折扣响应模型；避免 observed 和 recovered 模型交叉使用。
    def _response_bundle(self) -> FittedResponseModel:
        if self.calibration_mode in {"observed_calibration", "recovered_demand_observed_response"}:
            return self.observed_model
        return self.recovered_model

    # 执行一次 pricing decision：action 映射折扣、预测需求、限制销量、FEFO 出库、库存老化并计算 reward；返回 next state 供
    # agent 继续决策。
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}; expected Discrete(6).")
        # 一次 pricing decision 的数据流为：action -> markdown -> predicted demand
        # -> 实际可售数量 -> FEFO 出库 -> 当日过期 -> accounting reward -> next state。
        # PPO/DQN 不直接读取未来销量或最终利润，只接收当前 observation 和本步 reward。
        if self.current_step >= self.horizon:
            raise RuntimeError("Cannot step terminated episode; call reset().")
        markdown = ACTION_MARKDOWNS[int(action)]
        selling_price = 1.0 - markdown
        before = float(self.inventory.sum())
        predicted = self._predict_demand(markdown)
        demand = predicted
        if not self.deterministic_demand and self.residual_noise_mode == "bootstrap":
            demand = max(0.0, predicted + float(self.np_random.normal(0.0, max(0.01, predicted * 0.05))))
        sales = min(demand, before)
        # FEFO（First-Expire-First-Out）先销售剩余寿命最短的库存，
        # 销售后仍留在 bucket 0 的数量在日末记为 physical waste。
        self._issue_fefo(sales)
        expired = float(self.inventory[0])
        aged = np.zeros_like(self.inventory)
        if self.shelf_life > 1:
            aged[: self.shelf_life - 1] = self.inventory[1 : self.shelf_life]
        self.inventory = np.maximum(aged, 0.0)
        after = float(self.inventory.sum())
        conservation_error = abs(before - sales - expired - after)
        if conservation_error > 1e-6:
            raise RuntimeError(f"Inventory conservation failed: {conservation_error}")
        if np.any(self.inventory < -1e-9):
            raise RuntimeError("Negative inventory detected.")

        # financial reward 使用单位初始库存归一化，减少不同 episode 库存规模造成的梯度差异。
        # procurement cost 只在 episode 首步计入，disposal cost 随当日过期量计入；
        # accounting_profit 保存未归一化的累计财务结果。
        revenue = selling_price * sales
        disposal_cost = self.disposal_cost_ratio * expired
        procurement = self.procurement_cost_ratio * self.initial_inventory if self.initial_procurement_cost_pending else 0.0
        self.initial_procurement_cost_pending = False
        self.cumulative_sales += sales
        self.cumulative_waste += expired
        self.cumulative_revenue += revenue
        self.cumulative_procurement_cost += procurement
        self.cumulative_disposal_cost += disposal_cost
        raw_financial = revenue - disposal_cost - procurement
        self.accounting_profit += raw_financial
        self.raw_financial_steps.append(raw_financial)
        normalized_financial = raw_financial / max(self.initial_inventory, EPSILON)
        reward = normalized_financial
        if self.reward_mode == "sustainability":
            reward -= self.lambda_waste * (expired / max(self.initial_inventory, EPSILON))
        self.episode_return += reward
        self.reward_steps.append(float(reward))
        self.sim_history.append(float(sales if self.calibration_mode.startswith("observed") else demand))
        self.sim_history = self.sim_history[-14:]
        self.stockout_history.append(int(sales + EPSILON < demand))
        self.stockout_history = self.stockout_history[-7:]
        self.previous_markdown = markdown
        self.current_step += 1
        # 库存清空或达到与 shelf life 对齐的 horizon 时自然结束 episode；
        # 这里没有外部时间限制，因此 truncated 始终为 False。
        terminated = bool(after <= 1e-8 or self.current_step >= self.horizon)
        truncated = False
        info = self._info(markdown, selling_price, predicted, sales, before, after, expired, raw_financial, terminated)
        self.last_info = info
        return self._get_obs(), float(reward), terminated, truncated, info

    # 按 FEFO 从最临期批次开始扣减销量；这样过期风险和浪费来自真实的库存年龄结构，而不是只用总库存近似。
    def _issue_fefo(self, sales: float) -> None:
        """从最临期 bucket 开始扣减销量，保持库存年龄流转符合 FEFO。"""

        # bucket 索引越小表示越接近过期，因此正向遍历即可实现 FEFO。
        remaining = float(sales)
        for i in range(self.shelf_life):
            issued = min(float(self.inventory[i]), remaining)
            self.inventory[i] -= issued
            remaining -= issued
            if remaining <= 1e-9:
                break

    # 把当前状态和 markdown 输入响应模型得到本期需求；结果仍受可售库存约束，区分 demand 与 realized sales。
    def _predict_demand(self, markdown: float) -> float:
        bundle = self._response_bundle()
        row = self.current_series.iloc[min(self.current_step, len(self.current_series) - 1)]
        frame = pd.DataFrame([self._feature_row(row, markdown, bundle)])
        x = frame.reindex(columns=bundle.feature_columns)
        for col in x.columns:
            x[col] = pd.to_numeric(x[col], errors="coerce")
        if x.isna().any().any():
            missing = x.columns[x.isna().any()].tolist()
            raise ValueError(f"Missing feature values for response model: {missing}")
        pred_log = bundle.model.predict(x)[0]
        pred = float(np.expm1(pred_log))
        return float(np.clip(pred, 0.0, max(bundle.prediction_cap * 1.5, bundle.prediction_cap + 1.0)))

    # 按训练时保存的特征定义构造单行模型输入；固定列名和顺序可防止推断阶段特征错位。
    def _feature_row(self, row: pd.Series, markdown: float, bundle: FittedResponseModel) -> dict[str, Any]:
        date = pd.Timestamp(row["timestamp"])
        activity = int(markdown > 0) if self.promotion_context_mode == "derived_from_action" else int(row["activity_flag"])
        inventory_total = float(self.inventory.sum())
        lag1 = self.sim_history[-1]
        lag7 = self.sim_history[-7] if len(self.sim_history) >= 7 else self.sim_history[0]
        return {
            "day_of_week": int(date.dayofweek),
            "week_index": int((date - self.data["timestamp"].min()).days // 7),
            "month": int(date.month),
            "weekend_flag": int(date.dayofweek in [5, 6]),
            "trend_index": int((date - self.data["timestamp"].min()).days),
            "store_id_encoded": bundle.encoders["store_id"].get(self.store_id, 0),
            "product_id_encoded": bundle.encoders["product_id"].get(self.product_id, 0),
            "lag_1_observed_sales": lag1,
            "lag_7_observed_sales": lag7,
            "rolling_mean_7_observed_sales": float(np.mean(self.sim_history[-7:])),
            "rolling_mean_7_recovered_demand": float(np.mean(self.sim_history[-7:])),
            "rolling_mean_14_observed_sales": float(np.mean(self.sim_history[-14:])),
            "previous_day_possible_stockout_flag": int(self.stockout_history[-1]),
            "markdown_rate": markdown,
            "activity_flag": activity,
            "markdown_x_activity": markdown * activity,
            "stock_hour6_22_cnt": inventory_total,
            "previous_day_inventory": inventory_total,
            "possible_stockout_flag": int(inventory_total <= 0),
            "avg_temperature": float(row.get("avg_temperature", row.get("weather", 0.0))),
        }

    # 把库存年龄层、临期比例、需求历史、时间与商品信息压缩为固定 41 维 state；固定边界便于 PPO/DQN 共用同一环境。
    def _get_obs(self) -> np.ndarray:
        # 将不同量纲的库存、需求和时间信息整理成固定 41 维 state。
        # 库存 bucket 除以 initial_inventory，便于同一网络跨商品和库存规模学习；
        # 最后的 shape 检查可防止特征顺序改动后模型静默读取错误位置。
        total = float(self.inventory.sum())
        padded = np.zeros(MAX_SHELF_LIFE, dtype=float)
        padded[: self.shelf_life] = self.inventory[: self.shelf_life] / max(self.initial_inventory, EPSILON)
        exp_today = float(self.inventory[0] / max(total, EPSILON)) if total > 0 else 0.0
        exp_two = float(self.inventory[: min(2, self.shelf_life)].sum() / max(total, EPSILON)) if total > 0 else 0.0
        weights = np.arange(1, MAX_SHELF_LIFE + 1)
        mean_life = float((self.inventory * weights).sum() / max(total, EPSILON) / MAX_SHELF_LIFE) if total > 0 else 0.0
        row = self.current_series.iloc[min(self.current_step, max(len(self.current_series) - 1, 0))] if not self.current_series.empty else {}
        obs = np.array(
            [
                total / max(self.initial_inventory, EPSILON),
                total / max(np.mean(self.sim_history[-7:]), EPSILON),
                *padded.tolist(),
                exp_today,
                exp_two,
                mean_life,
                self._predict_zero_safe(),
                self.sim_history[-1],
                self.sim_history[-7] if len(self.sim_history) >= 7 else self.sim_history[0],
                float(np.mean(self.sim_history[-7:])),
                float(np.mean(self.sim_history[-14:])),
                float(np.std(self.sim_history[-7:])),
                float(pd.Timestamp(row.get("timestamp", pd.Timestamp("2024-01-01"))).dayofweek) if isinstance(row, pd.Series) else 0.0,
                float(pd.Timestamp(row.get("timestamp", pd.Timestamp("2024-01-01"))).dayofweek in [5, 6]) if isinstance(row, pd.Series) else 0.0,
                self.current_step / max(self.horizon, 1),
                float(row.get("avg_temperature", row.get("weather", 0.0))) / 40.0 if isinstance(row, pd.Series) else 0.0,
                self.previous_markdown / 0.40,
                float(self.stockout_history[-1]),
                1.0 if "recovered" in self.calibration_mode else 0.0,
                stable_norm(self.store_id),
                stable_norm(self.product_id),
            ],
            dtype=np.float32,
        )
        if obs.shape != (_LEGACY_OBSERVATION_DIM,):
            raise RuntimeError(
                f"Observation schema changed: expected {_LEGACY_OBSERVATION_DIM} values in the locked order, got {obs.shape}."
            )
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(obs, self.observation_space.low, self.observation_space.high).astype(np.float32)

    # 计算 0% markdown 下的参考需求并处理异常值；该参考量用于构造 coverage 等状态特征，而不是作为 oracle action。
    def _predict_zero_safe(self) -> float:
        """Predict zero-markdown demand, using zero only before episode initialization."""
        if self.current_series.empty or self.horizon == 0:
            return 0.0
        return self._predict_demand(0.0)

    # 把 accounting、浪费和动作诊断写入 info；评估脚本使用这些原始量计算利润、sell-through 和 waste 指标。
    def _info(
        self,
        markdown_rate: float,
        selling_price: float,
        predicted_demand: float,
        sales: float,
        before: float,
        after: float,
        expired: float,
        raw_financial: float,
        terminated: bool,
    ) -> dict[str, Any]:
        date = None
        if not self.current_series.empty:
            idx = min(max(self.current_step - 1, 0), len(self.current_series) - 1)
            date = str(pd.Timestamp(self.current_series.iloc[idx]["timestamp"]).date())
        profit_identity = self.cumulative_revenue - self.cumulative_procurement_cost - self.cumulative_disposal_cost
        info = {
            "date": date,
            "store_id": self.store_id,
            "product_id": self.product_id,
            "scenario_id": self.scenario_id,
            "calibration_mode": self.calibration_mode,
            "reward_mode": self.reward_mode,
            "markdown_rate": markdown_rate,
            "selling_price": selling_price,
            "predicted_demand": predicted_demand,
            "sales": sales,
            "inventory_before": before,
            "inventory_after": after,
            "expired_units": expired,
            "cumulative_sales": self.cumulative_sales,
            "cumulative_waste": self.cumulative_waste,
            "cumulative_revenue": self.cumulative_revenue,
            "cumulative_procurement_cost": self.cumulative_procurement_cost,
            "cumulative_disposal_cost": self.cumulative_disposal_cost,
            "accounting_profit": self.accounting_profit,
            "raw_financial_step": raw_financial,
            "waste_rate": self.cumulative_waste / max(self.initial_inventory, EPSILON),
            "sell_through_rate": self.cumulative_sales / max(self.initial_inventory, EPSILON),
            "stockout_flag": bool(sales + EPSILON < predicted_demand),
            "extrapolation_flag": False,
            "weak_support_flag": markdown_rate not in set(ACTION_MARKDOWNS.values()),
            "promotion_context_mode": self.promotion_context_mode,
            "step_conservation_error": abs(before - sales - expired - after),
            "profit_identity_error": abs(self.accounting_profit - profit_identity),
            "remaining_life_profile": self.inventory[: self.shelf_life].copy(),
        }
        if terminated:
            terminal_inventory = float(self.inventory.sum())
            terminal_salvage = 0.0
            info.update(
                {
                    "terminal_inventory": terminal_inventory,
                    "terminal_salvage_value": terminal_salvage,
                    "episode_return": self.episode_return,
                    "normalized_accounting_profit": self.accounting_profit / max(self.initial_inventory, EPSILON),
                    "final_waste_rate": self.cumulative_waste / max(self.initial_inventory, EPSILON),
                    "final_sell_through_rate": self.cumulative_sales / max(self.initial_inventory, EPSILON),
                    "episode_conservation_error": abs(self.initial_inventory - self.cumulative_sales - self.cumulative_waste - terminal_inventory),
                    "raw_financial_sum_error": abs(sum(self.raw_financial_steps) - self.accounting_profit),
                }
            )
        return info

    # 实现 Gymnasium 接口所需的辅助方法；保持环境可被标准 RL 工具安全包装和释放。
    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        info = self.last_info
        text = (
            f"date={info.get('date')} markdown={info.get('markdown_rate'):.2f} "
            f"demand={info.get('predicted_demand'):.3f} sales={info.get('sales'):.3f} "
            f"waste={info.get('expired_units'):.3f} inventory={info.get('inventory_after'):.3f} "
            f"profit={info.get('accounting_profit'):.3f} cumulative_waste={info.get('cumulative_waste'):.3f}"
        )
        return text


# ---------------------------------------------------------------------------
# Artifact and scenario helpers
# ---------------------------------------------------------------------------


# 从 joblib artifact 恢复模型并校验所需字段；集中入口避免不同调用方采用不一致的加载方式。
def load_response_model(path: Path) -> FittedResponseModel:
    """Load a response model artifact using a compatibility shim if needed."""
    import __main__

    setattr(__main__, "FittedResponseModel", FittedResponseModel)
    obj = joblib.load(path)
    required = ["model", "model_type", "feature_columns", "encoders", "prediction_cap"]
    missing = [name for name in required if not hasattr(obj, name)]
    if missing:
        raise ValueError(f"Response model artifact missing required attributes {missing}: {path}")
    return obj


# 筛选具有完整 horizon 的 episode 起点；避免训练或评估在数据末尾提前截断。
def eligible_episode_starts(data: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Return eligible split-contained starts by store-product series."""
    rows = []
    for (store_id, product_id), group in data.groupby(["store_id", "product_id"], sort=False):
        dates = sorted(group["timestamp"].dt.floor("D").unique())
        date_set = set(dates)
        for date in dates:
            end = pd.Timestamp(date) + pd.Timedelta(days=horizon - 1)
            if end in date_set:
                rows.append({"store_id": store_id, "product_id": product_id, "start_date": pd.Timestamp(date)})
    return pd.DataFrame(rows)


# 根据场景表读取保质期、库存覆盖和成本参数。
def scenario_shelf_life(shelf_class: str, level: str = "base") -> int:
    values = {
        "short_shelf_life": {"low": 2, "base": 3, "high": 5},
        "medium_shelf_life": {"low": 5, "base": 7, "high": 10},
        "long_shelf_life": {"low": 10, "base": 14, "high": 21},
    }
    return int(values[str(shelf_class)][level])


def coverage_midpoint(name: str) -> float:
    return {"lean_inventory": 1.25, "balanced_inventory": 2.5, "high_inventory": 5.0}[str(name)]


def margin_ratio(name: str) -> float:
    return {"low_margin": 0.70, "medium_margin": 0.55, "high_margin": 0.40}[str(name)]


def disposal_ratio(name: str) -> float:
    return {"zero_disposal_cost": 0.00, "moderate_disposal_cost": 0.05, "high_disposal_cost": 0.10}[str(name)]


#把库存年龄 profile 转为各年龄层权重；显式规则使 semi-synthetic 库存假设可检查。
def age_weights(profile: str, shelf_life: int) -> np.ndarray:
    if profile == "fresh_heavy":
        weights = np.arange(1, shelf_life + 1, dtype=float)
    elif profile == "near_expiry_heavy":
        weights = np.arange(shelf_life, 0, -1, dtype=float)
    else:
        weights = np.ones(shelf_life, dtype=float)
    return weights / weights.sum()


def stable_norm(value: str) -> float:
    return (sum(ord(ch) for ch in str(value)) % 1000) / 1000.0


# ---------------------------------------------------------------------------
# Observation schema and documented limitations
# ---------------------------------------------------------------------------


# 固定 41 维 observation 的语义顺序；训练、checkpoint 和评估必须共享这一顺序。
def observation_names() -> list[str]:
    """Return the locked observation order used by saved normalization and model artifacts."""
    return list(_OBSERVATION_NAMES)


# 定义每个 state 维度的合法范围；Gymnasium 用它检查 observation，并帮助发现异常状态。
def observation_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Build the legacy bounds by feature name without changing their values."""
    names = observation_names()
    if len(names) != _LEGACY_OBSERVATION_DIM or len(set(names)) != len(names):
        raise RuntimeError("The locked observation schema must contain 41 unique names.")
    index = {name: position for position, name in enumerate(names)}
    low = np.zeros(len(names), dtype=np.float32)
    high = np.ones(len(names), dtype=np.float32)

    bound_groups = {
        10.0: {"normalized_total_inventory", "inventory_coverage"},
        20.0: {
            "remaining_life_bucket_20",
            "remaining_life_bucket_21",
            "fraction_expiring_today",
            "fraction_expiring_within_two_days",
            "weighted_mean_remaining_life",
            "predicted_zero_markdown_demand",
            "lag_1_simulated_target",
        },
        6.0: {"lag_7_simulated_target", "rolling_7_simulated_mean"},
        40.0: {"simulated_demand_volatility"},
    }
    for upper_bound, feature_names in bound_groups.items():
        for feature_name in feature_names:
            high[index[feature_name]] = upper_bound

    # Saved policies rely on this exact 41-value bound vector.
    legacy_high = np.asarray(
        [10.0, 10.0, *([1.0] * 19), *([20.0] * 7), 6.0, 6.0, 1.0, 40.0, *([1.0] * 9)],
        dtype=np.float32,
    )
    if not np.array_equal(high, legacy_high):
        raise RuntimeError("Named observation bounds no longer match the locked legacy vector.")
    return low, high


# 返回研究设计和数据的已知限制；报告负结果时同时说明外推边界。
def required_limitations() -> list[str]:
    return [
        "Shelf life and inventory age are simulated.",
        "Realized waste is simulated.",
        "Costs are normalized assumptions.",
        "Recovered demand is model-estimated.",
        "Markdown response is observational and model-implied.",
        "Promotion-context mapping is an environment assumption.",
        "Actual latent demand during stockouts remains unobserved.",
        "This environment supports controlled decision experiments.",
        "It does not estimate direct retailer profit or deployment performance.",
    ]


__all__ = ["OperationalPerishablePricingEnv", "ACTION_MARKDOWNS", "observation_bounds", "observation_names"]
