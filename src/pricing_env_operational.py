"""Gymnasium-compatible operational dynamic-markdown environment.

The environment combines FreshRetailNet operational states, saved
discount-response models, explicit perishability scenarios, FEFO transitions,
and normalized accounting. It is a controlled semi-synthetic decision
environment; it is not an estimate of direct retailer profit or deployment
performance.
"""

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


RANDOM_SEED = 42
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

ACTION_MARKDOWNS = {
    0: 0.00,
    1: 0.05,
    2: 0.10,
    3: 0.20,
    4: 0.30,
    5: 0.40,
}


@dataclass
class FittedResponseModel:
    """Compatibility shim for joblib artifacts saved by discount_response.py."""

    target_name: str
    model_type: str
    model: Any
    feature_columns: list[str]
    encoders: dict[str, dict[str, int]]
    prediction_cap: float


class OperationalPerishablePricingEnv(gym.Env):
    """Operational dynamic markdown environment with FEFO inventory dynamics."""

    metadata = {"render_modes": ["ansi"], "render_fps": 1}

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
        obs_dim = len(observation_names())
        low = np.zeros(obs_dim, dtype=np.float32)
        high = np.ones(obs_dim, dtype=np.float32)
        high[0:2] = 10.0
        high[21:28] = 20.0
        high[28:30] = 6.0
        high[30] = 1.0
        high[31] = 40.0
        high[32] = 1.0
        high[33] = 1.0
        high[34] = 1.0
        high[35] = 1.0
        high[36] = 1.0
        high[37] = 1.0
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.np_random = np.random.default_rng(self.random_seed)
        self.episode_counter = 0
        self._reset_state()

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

    def _initialize_inventory(self, split_data: pd.DataFrame) -> None:
        target = self._target_column()
        hist = split_data.loc[split_data["store_id"].eq(self.store_id) & split_data["product_id"].eq(self.product_id), target]
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

    def _target_column(self) -> str:
        if self.calibration_mode in {"observed_calibration", "observed_demand_recovered_response"}:
            return "observed_sales_demand"
        return "recovered_demand"

    def _response_bundle(self) -> FittedResponseModel:
        if self.calibration_mode in {"observed_calibration", "recovered_demand_observed_response"}:
            return self.observed_model
        return self.recovered_model

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}; expected Discrete(6).")
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
        terminated = bool(after <= 1e-8 or self.current_step >= self.horizon)
        truncated = False
        info = self._info(markdown, selling_price, predicted, sales, before, after, expired, raw_financial, terminated)
        self.last_info = info
        return self._get_obs(), float(reward), terminated, truncated, info

    def _issue_fefo(self, sales: float) -> None:
        remaining = float(sales)
        for i in range(self.shelf_life):
            issued = min(float(self.inventory[i]), remaining)
            self.inventory[i] -= issued
            remaining -= issued
            if remaining <= 1e-9:
                break

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

    def _get_obs(self) -> np.ndarray:
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
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(obs, self.observation_space.low, self.observation_space.high).astype(np.float32)

    def _predict_zero_safe(self) -> float:
        if self.current_series.empty or self.horizon == 0:
            return 0.0
        try:
            return self._predict_demand(0.0)
        except Exception:
            return float(np.mean(self.sim_history[-7:]))

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


def observation_names() -> list[str]:
    return [
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
    ]


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


__all__ = ["OperationalPerishablePricingEnv", "ACTION_MARKDOWNS"]
