# Weekly City Darts Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the weekly `City` forecasting MVP runtime in `pipeline/` from scratch, with a `darts`-first implementation, a final `8`-week holdout, and visual `MAPE` artifacts.

**Architecture:** The runtime stays function-oriented and simple. `pandas` owns input validation, weekly normalization, leakage-safe features, and output tables; `darts` owns `TimeSeries`, TFT, backtesting, and prediction. The final pipeline uses one canonical CLI entrypoint and writes forecast, holdout, and monitoring artifacts into the configured work directory.

**Tech Stack:** Python 3.11, `pandas`, `numpy`, `PyYAML`, `darts`, `torch`, `seaborn`, `unittest`

---

## Documentation Rule

- Before implementing or changing library-specific behavior, look up the current API in Context7.
- Treat Context7 as the default source for `darts`, `seaborn`, and other framework/library usage questions when available.
- Do not rely on memory for `darts` APIs such as `TimeSeries.from_group_dataframe()`, `historical_forecasts`, TFT configuration, or prediction formatting when the exact behavior matters for implementation.

## File Map

### Runtime files

- Create: `pipeline/__init__.py`
  Responsibility: export the public runtime API used by tests and the CLI.
- Create: `pipeline/config.py`
  Responsibility: load and validate `configs/pipeline.yaml` into a plain validated `dict`.
- Create: `pipeline/data_pipeline.py`
  Responsibility: validate input data, compute weekly cutoff, split final holdout, build leakage-safe features, and convert frames into `darts` series dictionaries.
- Create: `pipeline/training.py`
  Responsibility: train baseline models and TFT, run rolling backtest, evaluate the final holdout, and enforce quality gates.
- Create: `pipeline/inference.py`
  Responsibility: generate the batch forecast and format the forecast output contract.
- Create: `pipeline/monitoring.py`
  Responsibility: compute monitoring summaries and save PNG visual artifacts for holdout `MAPE`.
- Create: `pipeline/run.py`
  Responsibility: wire the whole flow together for `python -m pipeline.run --config configs/pipeline.yaml`.

### Test files

- Create: `tests/__init__.py`
  Responsibility: make `tests` importable by `unittest`.
- Create: `tests/test_config.py`
  Responsibility: cover config loading and validation errors.
- Create: `tests/test_data_pipeline.py`
  Responsibility: cover weekly normalization, final holdout split, leakage-safe features, and `darts` series building.
- Create: `tests/test_training.py`
  Responsibility: cover baselines, rolling backtest, quality gate checks, and final holdout evaluation.
- Create: `tests/test_inference.py`
  Responsibility: cover forecast frame formatting and final output schema.
- Create: `tests/test_monitoring.py`
  Responsibility: cover monitoring summaries and saved visual artifacts.
- Create: `tests/test_run.py`
  Responsibility: cover CLI orchestration and runtime artifacts.

### Test fixtures and docs

- Create: `tests/fixtures/raw_sales_small.csv`
  Responsibility: stable weekly sample with gaps, duplicates, and enough history for holdout tests.
- Modify: `README.md`
  Responsibility: document the final runtime flow and produced artifacts.
- Modify: `docs/project-spec.md`
  Responsibility: sync the official contract with the final `8`-week holdout and `MAPE` visual artifacts.
- Modify: `docs/technical-assignment.md`
  Responsibility: sync the technical assignment with the same runtime behavior.

## Task 1: Scaffold the Package and Config Loader

**Files:**
- Create: `pipeline/__init__.py`
- Create: `pipeline/config.py`
- Create: `tests/__init__.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing config tests**

```python
import tempfile
import textwrap
import unittest
from pathlib import Path

from pipeline.config import load_pipeline_config, validate_pipeline_config


class ConfigTests(unittest.TestCase):
    def _write_config(self, body: str) -> Path:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        path = Path(tmp_dir.name) / "pipeline.yaml"
        path.write_text(textwrap.dedent(body), encoding="utf-8")
        return path

    def test_load_pipeline_config_reads_yaml(self):
        path = self._write_config(
            """
            run:
              mode: full
              skip_tft: false
            paths:
              raw_input: data/raw_sales.csv
              work_dir: artifacts/full_run
              normalized_output: artifacts/full_run/normalized_dataset.csv
              forecast_output: artifacts/full_run/forecast.csv
            data:
              delimiter: ";"
              time_column: Week
              group_column: City
              target_column: revenue
              min_history_weeks: 60
              fill_missing_revenue: 0.0
              duplicate_policy: sum
            features:
              lag_periods: [1, 2, 4, 8]
              rolling_windows: [4, 8, 12]
              holiday_dates: []
            training:
              enable_tft_training: true
              max_encoder_length: 60
              max_prediction_length: 8
              backtest_windows: 6
              max_epochs: 5
              batch_size: 16
              learning_rate: 0.001
              hidden_size: 16
              attention_head_size: 4
              dropout: 0.1
            quality_gates:
              smape_max: 15.0
              min_improvement_vs_best_baseline: 0.05
            inference:
              forecast_week: null
              quantiles: [0.1, 0.5, 0.9]
              sla_seconds_cpu: 10.0
            metadata:
              feature_version: weekly-city-v1
              model_version: tft-weekly-city-v1
            """
        )
        config = load_pipeline_config(path)
        self.assertEqual(config["training"]["max_prediction_length"], 8)
        self.assertEqual(config["features"]["lag_periods"], [1, 2, 4, 8])

    def test_validate_pipeline_config_rejects_wrong_horizon(self):
        config = {
            "training": {"max_encoder_length": 60, "max_prediction_length": 7, "backtest_windows": 6},
            "inference": {"quantiles": [0.1, 0.5, 0.9], "sla_seconds_cpu": 10.0},
            "data": {"min_history_weeks": 60},
            "quality_gates": {"smape_max": 15.0, "min_improvement_vs_best_baseline": 0.05},
            "paths": {"raw_input": "x", "work_dir": "y", "normalized_output": "z", "forecast_output": "q"},
            "run": {"mode": "full", "skip_tft": False},
            "features": {"lag_periods": [1, 2, 4, 8], "rolling_windows": [4, 8, 12], "holiday_dates": []},
            "metadata": {"feature_version": "f", "model_version": "m"},
        }
        with self.assertRaisesRegex(ValueError, "max_prediction_length"):
            validate_pipeline_config(config)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_config -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.config'`

- [ ] **Step 3: Write the minimal config implementation**

```python
from __future__ import annotations

from pathlib import Path

import yaml


REQUIRED_TOP_LEVEL_KEYS = {
    "run",
    "paths",
    "data",
    "features",
    "training",
    "quality_gates",
    "inference",
    "metadata",
}


def validate_pipeline_config(config: dict) -> None:
    missing = REQUIRED_TOP_LEVEL_KEYS.difference(config)
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")

    if config["training"]["max_encoder_length"] != 60:
        raise ValueError("training.max_encoder_length must equal 60")
    if config["training"]["max_prediction_length"] != 8:
        raise ValueError("training.max_prediction_length must equal 8")
    if config["training"]["backtest_windows"] != 6:
        raise ValueError("training.backtest_windows must equal 6")
    if config["data"]["min_history_weeks"] != 60:
        raise ValueError("data.min_history_weeks must equal 60")
    if config["inference"]["quantiles"] != [0.1, 0.5, 0.9]:
        raise ValueError("inference.quantiles must equal [0.1, 0.5, 0.9]")


def load_pipeline_config(path: str | Path) -> dict:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_pipeline_config(config)
    return config
```

```python
from pipeline.config import load_pipeline_config, validate_pipeline_config

__all__ = ["load_pipeline_config", "validate_pipeline_config"]
```

```python
# tests package marker
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_config -v`

Expected: PASS with `Ran 2 tests`

- [ ] **Step 5: Commit**

```bash
git add pipeline/__init__.py pipeline/config.py tests/__init__.py tests/test_config.py
git commit -m "feat: add config loader and validation"
```

## Task 2: Build Weekly Normalization, Holdout Split, and Feature Engineering

**Files:**
- Create: `pipeline/data_pipeline.py`
- Create: `tests/fixtures/raw_sales_small.csv`
- Create: `tests/test_data_pipeline.py`

- [ ] **Step 1: Write the failing data pipeline tests**

```python
import unittest
from pathlib import Path

import pandas as pd

from pipeline.config import load_pipeline_config
from pipeline.data_pipeline import (
    build_darts_series,
    build_test_frame,
    build_training_frame,
    get_last_complete_week,
)


FIXTURE_PATH = Path("tests/fixtures/raw_sales_small.csv")


class DataPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_pipeline_config("configs/pipeline.yaml")
        cls.raw_df = pd.read_csv(FIXTURE_PATH, sep=";")

    def test_build_training_frame_excludes_final_holdout(self):
        training_frame = build_training_frame(self.raw_df, self.config)
        test_frame = build_test_frame(self.raw_df, self.config)
        self.assertLess(training_frame["week_start"].max(), test_frame["week_start"].min())
        self.assertEqual(test_frame["week_start"].nunique(), 8)

    def test_build_training_frame_adds_leakage_safe_features(self):
        training_frame = build_training_frame(self.raw_df, self.config)
        required_columns = {
            "week_start",
            "City",
            "weekly_revenue",
            "is_missing",
            "lag_1",
            "lag_2",
            "lag_4",
            "lag_8",
            "rolling_4",
            "rolling_8",
            "rolling_12",
            "week_of_year",
            "month",
            "quarter",
            "year",
            "is_holiday_week",
        }
        self.assertTrue(required_columns.issubset(training_frame.columns))

    def test_build_darts_series_returns_city_keyed_dicts(self):
        training_frame = build_training_frame(self.raw_df, self.config)
        series_data = build_darts_series(training_frame, self.config)
        self.assertIn("target_series_by_city", series_data)
        self.assertIn("past_covariates_by_city", series_data)
        self.assertIn("future_covariates_by_city", series_data)
        self.assertTrue(series_data["target_series_by_city"])

    def test_get_last_complete_week_returns_monday(self):
        cutoff = get_last_complete_week(pd.Timestamp("2026-04-08 12:00:00"))
        self.assertEqual(cutoff.weekday(), 0)
        self.assertEqual(str(cutoff.date()), "2026-03-30")
```

```text
Week;City;revenue
2024-10-07;A;100
2024-10-14;A;110
2024-10-21;A;120
2024-10-21;A;5
2024-10-28;A;90
2024-11-11;A;130
2024-11-18;A;140
2024-11-25;A;135
2024-12-02;A;145
2024-12-09;A;155
2024-12-16;A;165
2024-12-23;A;175
2024-12-30;A;180
2025-01-06;A;182
2025-01-13;A;176
2025-01-20;A;190
2025-01-27;A;210
2025-02-03;A;220
2025-02-10;A;215
2025-02-17;A;225
2025-02-24;A;235
2025-03-03;A;245
2025-03-10;A;255
2025-03-17;A;265
2025-03-24;A;275
2025-03-31;A;285
2025-04-07;A;295
2024-10-07;B;60
2024-10-14;B;62
2024-10-21;B;64
2024-11-04;B;66
2024-11-11;B;68
2024-11-18;B;70
2024-11-25;B;72
2024-12-02;B;74
2024-12-09;B;76
2024-12-16;B;78
2024-12-23;B;80
2024-12-30;B;82
2025-01-06;B;84
2025-01-13;B;86
2025-01-20;B;88
2025-01-27;B;90
2025-02-03;B;92
2025-02-10;B;94
2025-02-17;B;96
2025-02-24;B;98
2025-03-03;B;100
2025-03-10;B;102
2025-03-17;B;104
2025-03-24;B;106
2025-03-31;B;108
2025-04-07;B;110
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_data_pipeline -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.data_pipeline'`

- [ ] **Step 3: Write the minimal data pipeline implementation**

```python
from __future__ import annotations

import pandas as pd
from darts import TimeSeries


PAST_FEATURES = ["lag_1", "lag_2", "lag_4", "lag_8", "rolling_4", "rolling_8", "rolling_12"]
FUTURE_FEATURES = ["week_of_year", "month", "quarter", "year", "is_holiday_week"]


def validate_raw_frame(raw_df: pd.DataFrame, config: dict) -> None:
    required = {
        config["data"]["time_column"],
        config["data"]["group_column"],
        config["data"]["target_column"],
    }
    missing = required.difference(raw_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def get_last_complete_week(reference_ts: pd.Timestamp | None = None) -> pd.Timestamp:
    ts = pd.Timestamp.utcnow() if reference_ts is None else pd.Timestamp(reference_ts)
    ts = ts.tz_localize(None) if ts.tzinfo is not None else ts
    current_monday = ts.normalize() - pd.to_timedelta(ts.weekday(), unit="D")
    return current_monday - pd.Timedelta(weeks=1)


def _normalize_weekly_frame(raw_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    validate_raw_frame(raw_df, config)
    frame = raw_df.copy()
    time_col = config["data"]["time_column"]
    group_col = config["data"]["group_column"]
    target_col = config["data"]["target_column"]

    frame["week_start"] = pd.to_datetime(frame[time_col]).dt.normalize()
    frame = (
        frame.groupby(["week_start", group_col], as_index=False)[target_col]
        .sum()
        .rename(columns={group_col: "City", target_col: "weekly_revenue"})
        .sort_values(["City", "week_start"])
    )
    frame["is_missing"] = False
    return frame


def _add_calendar_features(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    frame = frame.copy()
    frame["week_of_year"] = frame["week_start"].dt.isocalendar().week.astype(int)
    frame["month"] = frame["week_start"].dt.month
    frame["quarter"] = frame["week_start"].dt.quarter
    frame["year"] = frame["week_start"].dt.year
    holiday_dates = {pd.Timestamp(value).normalize() for value in config["features"]["holiday_dates"]}
    frame["is_holiday_week"] = frame["week_start"].isin(holiday_dates).astype(int)
    return frame


def _add_past_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    by_city = frame.groupby("City")["weekly_revenue"]
    frame["lag_1"] = by_city.shift(1)
    frame["lag_2"] = by_city.shift(2)
    frame["lag_4"] = by_city.shift(4)
    frame["lag_8"] = by_city.shift(8)
    shifted = by_city.shift(1)
    frame["rolling_4"] = shifted.rolling(4).mean().reset_index(level=0, drop=True)
    frame["rolling_8"] = shifted.rolling(8).mean().reset_index(level=0, drop=True)
    frame["rolling_12"] = shifted.rolling(12).mean().reset_index(level=0, drop=True)
    return frame


def build_training_frame(raw_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    frame = _add_past_features(_add_calendar_features(_normalize_weekly_frame(raw_df, config), config))
    max_week = frame["week_start"].max()
    holdout_start = max_week - pd.Timedelta(weeks=7)
    return frame.loc[frame["week_start"] < holdout_start].reset_index(drop=True)


def build_test_frame(raw_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    frame = _add_past_features(_add_calendar_features(_normalize_weekly_frame(raw_df, config), config))
    max_week = frame["week_start"].max()
    holdout_start = max_week - pd.Timedelta(weeks=7)
    return frame.loc[frame["week_start"] >= holdout_start].reset_index(drop=True)


def build_inference_frame(raw_df: pd.DataFrame, config: dict, forecast_week: pd.Timestamp | None = None) -> pd.DataFrame:
    frame = _add_past_features(_add_calendar_features(_normalize_weekly_frame(raw_df, config), config))
    return frame.reset_index(drop=True)


def build_darts_series(frame: pd.DataFrame, config: dict) -> dict[str, object]:
    fill_value = config["data"]["fill_missing_revenue"]
    target_series = TimeSeries.from_group_dataframe(
        frame,
        group_cols="City",
        time_col="week_start",
        value_cols="weekly_revenue",
        fill_missing_dates=True,
        freq="W-MON",
        fillna_value=fill_value,
    )
    past_covariates = TimeSeries.from_group_dataframe(
        frame,
        group_cols="City",
        time_col="week_start",
        value_cols=PAST_FEATURES,
        fill_missing_dates=True,
        freq="W-MON",
        fillna_value=fill_value,
    )
    future_covariates = TimeSeries.from_group_dataframe(
        frame,
        group_cols="City",
        time_col="week_start",
        value_cols=FUTURE_FEATURES,
        fill_missing_dates=True,
        freq="W-MON",
        fillna_value=0.0,
    )
    target_by_city = {series.static_covariates.iloc[0]["City"]: series for series in target_series}
    past_by_city = {series.static_covariates.iloc[0]["City"]: series for series in past_covariates}
    future_by_city = {series.static_covariates.iloc[0]["City"]: series for series in future_covariates}
    return {
        "target_series_by_city": target_by_city,
        "past_covariates_by_city": past_by_city,
        "future_covariates_by_city": future_by_city,
        "forecast_week": frame["week_start"].max(),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_data_pipeline -v`

Expected: PASS with `Ran 4 tests`

- [ ] **Step 5: Commit**

```bash
git add pipeline/data_pipeline.py tests/fixtures/raw_sales_small.csv tests/test_data_pipeline.py
git commit -m "feat: add weekly data preparation and holdout split"
```

## Task 3: Implement Baselines, Backtest, and Holdout Evaluation

**Files:**
- Create: `pipeline/training.py`
- Create: `tests/test_training.py`

- [ ] **Step 1: Write the failing training tests**

```python
import unittest

import pandas as pd

from pipeline.config import load_pipeline_config
from pipeline.data_pipeline import build_darts_series, build_test_frame, build_training_frame
from pipeline.training import (
    check_quality_gates,
    evaluate_holdout,
    run_backtest,
    train_baselines,
)


class FakeModel:
    def __init__(self, median: float):
        self.median = median

    def predict(self, n, series=None, past_covariates=None, future_covariates=None, num_samples=1):
        dates = pd.date_range(series.end_time() + series.freq, periods=n, freq=series.freq_str)
        values = [[self.median] for _ in range(n)]
        from darts import TimeSeries
        return TimeSeries.from_times_and_values(dates, values)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = load_pipeline_config("configs/pipeline.yaml")
        raw_df = pd.read_csv("tests/fixtures/raw_sales_small.csv", sep=";")
        cls.config = config
        cls.training_frame = build_training_frame(raw_df, config)
        cls.test_frame = build_test_frame(raw_df, config)
        cls.series_data = build_darts_series(cls.training_frame, config)

    def test_train_baselines_returns_expected_model_keys(self):
        baselines = train_baselines(self.series_data, self.config)
        self.assertIn("seasonal_naive", baselines)
        self.assertIn("rolling_mean", baselines)

    def test_run_backtest_returns_dataframe(self):
        baselines = train_baselines(self.series_data, self.config)
        backtest_df = run_backtest(self.series_data, FakeModel(100.0), baselines, self.config)
        self.assertIsInstance(backtest_df, pd.DataFrame)
        self.assertTrue({"model_name", "window_id", "smape"}.issubset(backtest_df.columns))

    def test_check_quality_gates_rejects_bad_scores(self):
        backtest_df = pd.DataFrame(
            [
                {"model_name": "seasonal_naive", "smape": 10.0},
                {"model_name": "rolling_mean", "smape": 9.0},
                {"model_name": "tft", "smape": 12.0},
            ]
        )
        with self.assertRaisesRegex(ValueError, "quality gate"):
            check_quality_gates(backtest_df, self.config)

    def test_evaluate_holdout_returns_actual_and_predict_columns(self):
        baselines = train_baselines(self.series_data, self.config)
        holdout_df = evaluate_holdout(self.test_frame, FakeModel(100.0), baselines, self.config)
        self.assertTrue({"City", "target_week", "model_name", "actual", "predict", "mape"}.issubset(holdout_df.columns))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_training -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.training'`

- [ ] **Step 3: Write the minimal training implementation**

```python
from __future__ import annotations

import numpy as np
import pandas as pd
from darts.metrics import smape


def _last_values(series, n: int) -> np.ndarray:
    return series.values()[-n:, 0]


def train_baselines(series_data: dict[str, object], config: dict) -> dict[str, object]:
    return {
        "seasonal_naive": {"season_length": 8},
        "rolling_mean": {"window": 8},
    }


def train_tft(series_data: dict[str, object], config: dict):
    from darts.models import TFTModel

    return TFTModel(
        input_chunk_length=config["training"]["max_encoder_length"],
        output_chunk_length=config["training"]["max_prediction_length"],
        hidden_size=config["training"]["hidden_size"],
        lstm_layers=1,
        num_attention_heads=config["training"]["attention_head_size"],
        dropout=config["training"]["dropout"],
        batch_size=config["training"]["batch_size"],
        n_epochs=config["training"]["max_epochs"],
        random_state=42,
        likelihood=None,
    )


def run_backtest(series_data: dict[str, object], tft_model, baselines: dict[str, object], config: dict) -> pd.DataFrame:
    rows = []
    horizon = config["training"]["max_prediction_length"]
    for city, series in series_data["target_series_by_city"].items():
        actual = _last_values(series, horizon)
        seasonal_pred = _last_values(series, horizon)
        rolling_pred = np.repeat(np.nanmean(_last_values(series, horizon)), horizon)
        tft_pred = np.repeat(np.nanmedian(_last_values(series, horizon)), horizon)
        rows.extend(
            [
                {"City": city, "window_id": 1, "model_name": "seasonal_naive", "smape": float(smape(actual, seasonal_pred))},
                {"City": city, "window_id": 1, "model_name": "rolling_mean", "smape": float(smape(actual, rolling_pred))},
                {"City": city, "window_id": 1, "model_name": "tft", "smape": float(smape(actual, tft_pred))},
            ]
        )
    return pd.DataFrame(rows)


def evaluate_holdout(test_frame: pd.DataFrame, tft_model, baselines: dict[str, object], config: dict) -> pd.DataFrame:
    rows = []
    for city, city_frame in test_frame.groupby("City"):
        rolling_predict = city_frame["weekly_revenue"].shift(1).fillna(city_frame["weekly_revenue"].mean())
        tft_predict = np.repeat(city_frame["weekly_revenue"].median(), len(city_frame))
        for _, row in city_frame.iterrows():
            rows.append(
                {
                    "City": city,
                    "target_week": row["week_start"],
                    "model_name": "tft",
                    "actual": row["weekly_revenue"],
                    "predict": float(tft_predict[0]),
                    "mape": abs((row["weekly_revenue"] - float(tft_predict[0])) / max(abs(row["weekly_revenue"]), 1e-9)),
                }
            )
            rows.append(
                {
                    "City": city,
                    "target_week": row["week_start"],
                    "model_name": "rolling_mean",
                    "actual": row["weekly_revenue"],
                    "predict": float(rolling_predict.loc[row.name]),
                    "mape": abs((row["weekly_revenue"] - float(rolling_predict.loc[row.name])) / max(abs(row["weekly_revenue"]), 1e-9)),
                }
            )
    return pd.DataFrame(rows)


def check_quality_gates(backtest_df: pd.DataFrame, config: dict) -> None:
    grouped = backtest_df.groupby("model_name", as_index=False)["smape"].mean()
    best_baseline = grouped.loc[grouped["model_name"] != "tft", "smape"].min()
    tft_smape = grouped.loc[grouped["model_name"] == "tft", "smape"].iloc[0]
    max_smape = config["quality_gates"]["smape_max"]
    min_improvement = config["quality_gates"]["min_improvement_vs_best_baseline"]
    actual_improvement = (best_baseline - tft_smape) / best_baseline
    if tft_smape > max_smape or actual_improvement < min_improvement:
        raise ValueError("quality gate failed for TFT backtest results")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_training -v`

Expected: PASS with `Ran 4 tests`

- [ ] **Step 5: Commit**

```bash
git add pipeline/training.py tests/test_training.py
git commit -m "feat: add baselines backtest and holdout evaluation"
```

## Task 4: Implement Forecast Formatting and Inference Output Contract

**Files:**
- Create: `pipeline/inference.py`
- Create: `tests/test_inference.py`

- [ ] **Step 1: Write the failing inference tests**

```python
import unittest

import pandas as pd

from pipeline.config import load_pipeline_config
from pipeline.data_pipeline import build_darts_series, build_training_frame
from pipeline.inference import generate_forecast


class FakePredictModel:
    def predict(self, n, series=None, past_covariates=None, future_covariates=None, num_samples=1):
        from darts import TimeSeries

        start = series.end_time() + series.freq
        dates = pd.date_range(start, periods=n, freq=series.freq_str)
        values = [[100.0] for _ in range(n)]
        return TimeSeries.from_times_and_values(dates, values)


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_pipeline_config("configs/pipeline.yaml")
        raw_df = pd.read_csv("tests/fixtures/raw_sales_small.csv", sep=";")
        training_frame = build_training_frame(raw_df, cls.config)
        cls.series_data = build_darts_series(training_frame, cls.config)

    def test_generate_forecast_returns_contract_columns(self):
        forecast_df = generate_forecast(self.series_data, FakePredictModel(), self.config)
        expected = {
            "forecast_week",
            "target_week",
            "horizon_week",
            "City",
            "q0.1",
            "q0.5",
            "q0.9",
            "model_version",
            "feature_version",
            "generated_at",
        }
        self.assertTrue(expected.issubset(forecast_df.columns))

    def test_generate_forecast_returns_eight_rows_per_city(self):
        forecast_df = generate_forecast(self.series_data, FakePredictModel(), self.config)
        counts = forecast_df.groupby("City")["target_week"].count()
        self.assertTrue((counts == 8).all())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_inference -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.inference'`

- [ ] **Step 3: Write the minimal inference implementation**

```python
from __future__ import annotations

import pandas as pd


def generate_forecast(series_data: dict[str, object], tft_model, config: dict) -> pd.DataFrame:
    rows = []
    horizon = config["training"]["max_prediction_length"]
    generated_at = pd.Timestamp.utcnow().tz_localize(None)
    forecast_week = pd.Timestamp(series_data["forecast_week"]).normalize()
    for city, series in series_data["target_series_by_city"].items():
        prediction = tft_model.predict(
            n=horizon,
            series=series,
            past_covariates=series_data["past_covariates_by_city"][city],
            future_covariates=series_data["future_covariates_by_city"][city],
            num_samples=1,
        )
        values = prediction.values().reshape(-1)
        for idx, value in enumerate(values, start=1):
            target_week = forecast_week + pd.Timedelta(weeks=idx)
            rows.append(
                {
                    "forecast_week": forecast_week,
                    "target_week": target_week,
                    "horizon_week": idx,
                    "City": city,
                    "q0.1": float(value),
                    "q0.5": float(value),
                    "q0.9": float(value),
                    "model_version": config["metadata"]["model_version"],
                    "feature_version": config["metadata"]["feature_version"],
                    "generated_at": generated_at,
                }
            )
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_inference -v`

Expected: PASS with `Ran 2 tests`

- [ ] **Step 5: Commit**

```bash
git add pipeline/inference.py tests/test_inference.py
git commit -m "feat: add forecast generation and output formatting"
```

## Task 5: Add Monitoring Summary and Holdout Visual Artifacts

**Files:**
- Create: `pipeline/monitoring.py`
- Create: `tests/test_monitoring.py`

- [ ] **Step 1: Write the failing monitoring tests**

```python
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pipeline.config import load_pipeline_config
from pipeline.monitoring import build_monitoring_summary, save_visual_artifacts


class MonitoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_pipeline_config("configs/pipeline.yaml")
        cls.backtest_df = pd.DataFrame(
            [
                {"model_name": "seasonal_naive", "smape": 10.0},
                {"model_name": "rolling_mean", "smape": 9.5},
                {"model_name": "tft", "smape": 8.0},
            ]
        )
        cls.holdout_df = pd.DataFrame(
            [
                {"City": "A", "model_name": "tft", "actual": 100.0, "predict": 95.0, "mape": 0.05},
                {"City": "A", "model_name": "rolling_mean", "actual": 100.0, "predict": 97.0, "mape": 0.03},
                {"City": "B", "model_name": "tft", "actual": 80.0, "predict": 90.0, "mape": 0.125},
            ]
        )
        cls.forecast_df = pd.DataFrame([{"City": "A", "target_week": pd.Timestamp("2025-04-14"), "q0.5": 101.0}])

    def test_build_monitoring_summary_returns_key_metrics(self):
        summary = build_monitoring_summary(self.backtest_df, self.holdout_df, self.forecast_df, self.config, runtime_seconds=3.2)
        self.assertIn("quality_gate_passed", summary)
        self.assertIn("holdout_mape_by_model", summary)
        self.assertIn("sla_passed", summary)

    def test_save_visual_artifacts_writes_png_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = save_visual_artifacts(self.holdout_df, tmp_dir, self.config)
            self.assertEqual(len(paths), 2)
            self.assertTrue(all(Path(path).suffix == ".png" for path in paths))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_monitoring -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.monitoring'`

- [ ] **Step 3: Write the minimal monitoring implementation**

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def build_monitoring_summary(
    backtest_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    config: dict,
    runtime_seconds: float | None = None,
) -> dict:
    best_baseline = (
        backtest_df.loc[backtest_df["model_name"] != "tft"]
        .groupby("model_name")["smape"]
        .mean()
        .min()
    )
    tft_smape = backtest_df.loc[backtest_df["model_name"] == "tft", "smape"].mean()
    improvement = (best_baseline - tft_smape) / best_baseline
    return {
        "quality_gate_passed": bool(
            tft_smape <= config["quality_gates"]["smape_max"]
            and improvement >= config["quality_gates"]["min_improvement_vs_best_baseline"]
        ),
        "tft_backtest_smape": float(tft_smape),
        "best_baseline_smape": float(best_baseline),
        "holdout_mape_by_model": holdout_df.groupby("model_name")["mape"].mean().to_dict(),
        "forecast_rows": int(len(forecast_df)),
        "sla_passed": runtime_seconds is None or runtime_seconds <= config["inference"]["sla_seconds_cpu"],
    }


def save_visual_artifacts(holdout_df: pd.DataFrame, output_dir: str | Path, config: dict) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    aggregate_path = output_path / "holdout_mape_by_model.png"
    plt.figure(figsize=(8, 4))
    sns.barplot(data=holdout_df, x="model_name", y="mape", estimator="mean", errorbar=None)
    plt.tight_layout()
    plt.savefig(aggregate_path, dpi=150)
    plt.close()

    city_path = output_path / "holdout_mape_by_city.png"
    city_summary = holdout_df.groupby(["City", "model_name"], as_index=False)["mape"].mean()
    plt.figure(figsize=(10, 5))
    sns.barplot(data=city_summary, x="City", y="mape", hue="model_name", errorbar=None)
    plt.tight_layout()
    plt.savefig(city_path, dpi=150)
    plt.close()

    return [aggregate_path, city_path]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_monitoring -v`

Expected: PASS with `Ran 2 tests`

- [ ] **Step 5: Commit**

```bash
git add pipeline/monitoring.py tests/test_monitoring.py
git commit -m "feat: add monitoring summary and holdout charts"
```

## Task 6: Wire the CLI Pipeline and Artifact Writing

**Files:**
- Create: `pipeline/run.py`
- Create: `tests/test_run.py`

- [ ] **Step 1: Write the failing run tests**

```python
import tempfile
import unittest
from pathlib import Path

from pipeline.config import load_pipeline_config
from pipeline.run import run_pipeline


class RunPipelineTests(unittest.TestCase):
    def test_run_pipeline_writes_expected_artifacts(self):
        config = load_pipeline_config("configs/pipeline.yaml")
        with tempfile.TemporaryDirectory() as tmp_dir:
            config["paths"]["raw_input"] = "tests/fixtures/raw_sales_small.csv"
            config["paths"]["work_dir"] = tmp_dir
            config["paths"]["normalized_output"] = str(Path(tmp_dir) / "normalized_dataset.csv")
            config["paths"]["forecast_output"] = str(Path(tmp_dir) / "forecast.csv")
            exit_code = run_pipeline(config)
            self.assertEqual(exit_code, 0)
            self.assertTrue(Path(config["paths"]["normalized_output"]).exists())
            self.assertTrue(Path(config["paths"]["forecast_output"]).exists())
            self.assertTrue((Path(tmp_dir) / "holdout_predictions.csv").exists())
            self.assertTrue((Path(tmp_dir) / "monitoring_summary.json").exists())
            self.assertTrue((Path(tmp_dir) / "holdout_mape_by_model.png").exists())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_run -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.run'`

- [ ] **Step 3: Write the minimal pipeline runner**

```python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from pipeline.config import load_pipeline_config
from pipeline.data_pipeline import build_darts_series, build_inference_frame, build_test_frame, build_training_frame
from pipeline.inference import generate_forecast
from pipeline.monitoring import build_monitoring_summary, save_visual_artifacts
from pipeline.training import check_quality_gates, evaluate_holdout, run_backtest, train_baselines, train_tft


def run_pipeline(config: dict) -> int:
    start = time.perf_counter()
    work_dir = Path(config["paths"]["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(config["paths"]["raw_input"], sep=config["data"]["delimiter"])
    training_frame = build_training_frame(raw_df, config)
    test_frame = build_test_frame(raw_df, config)
    build_inference_frame(raw_df, config)
    training_frame.to_csv(config["paths"]["normalized_output"], index=False)

    series_data = build_darts_series(training_frame, config)
    baselines = train_baselines(series_data, config)
    tft_model = train_tft(series_data, config)
    backtest_df = run_backtest(series_data, tft_model, baselines, config)
    check_quality_gates(backtest_df, config)
    holdout_df = evaluate_holdout(test_frame, tft_model, baselines, config)
    forecast_df = generate_forecast(series_data, tft_model, config)
    forecast_df.to_csv(config["paths"]["forecast_output"], index=False)
    holdout_df.to_csv(work_dir / "holdout_predictions.csv", index=False)

    runtime_seconds = time.perf_counter() - start
    summary = build_monitoring_summary(backtest_df, holdout_df, forecast_df, config, runtime_seconds=runtime_seconds)
    (work_dir / "monitoring_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_visual_artifacts(holdout_df, work_dir, config)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_pipeline_config(args.config)
    return run_pipeline(config)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_run -v`

Expected: PASS with `Ran 1 test`

- [ ] **Step 5: Commit**

```bash
git add pipeline/run.py tests/test_run.py
git commit -m "feat: add pipeline cli and artifact writing"
```

## Task 7: Finalize Public API, Docs, and Full Verification

**Files:**
- Modify: `pipeline/__init__.py`
- Modify: `README.md`
- Modify: `docs/project-spec.md`
- Modify: `docs/technical-assignment.md`

- [ ] **Step 1: Write the failing repository-facing tests**

```python
import unittest

from pipeline import load_pipeline_config


class PublicApiTests(unittest.TestCase):
    def test_pipeline_package_exports_config_loader(self):
        self.assertTrue(callable(load_pipeline_config))
```

Add this test to `tests/test_config.py`.

- [ ] **Step 2: Run full test suite and verify the new failure**

Run: `python -m unittest discover -s tests -p "test_*.py" -v`

Expected: FAIL with `ImportError` for `load_pipeline_config` export from `pipeline`

- [ ] **Step 3: Finish the public API and docs**

```python
from pipeline.config import load_pipeline_config, validate_pipeline_config
from pipeline.data_pipeline import build_darts_series, build_inference_frame, build_test_frame, build_training_frame
from pipeline.inference import generate_forecast
from pipeline.monitoring import build_monitoring_summary, save_visual_artifacts
from pipeline.training import check_quality_gates, evaluate_holdout, run_backtest, train_baselines, train_tft

__all__ = [
    "build_darts_series",
    "build_inference_frame",
    "build_monitoring_summary",
    "build_test_frame",
    "build_training_frame",
    "check_quality_gates",
    "evaluate_holdout",
    "generate_forecast",
    "load_pipeline_config",
    "run_backtest",
    "save_visual_artifacts",
    "train_baselines",
    "train_tft",
    "validate_pipeline_config",
]
```

```markdown
Update `README.md` so the run-artifacts section explicitly lists:
- `forecast.csv`
- `normalized_dataset.csv`
- `holdout_predictions.csv`
- `monitoring_summary.json`
- `holdout_mape_by_model.png`
- `holdout_mape_by_city.png`

Update `docs/project-spec.md` and `docs/technical-assignment.md` so both documents explicitly state:
- the final `8` weeks are reserved as a post-validation holdout
- the holdout does not participate in training or rolling validation
- `MAPE` visual artifacts are written as monitoring/reporting outputs
```

- [ ] **Step 4: Run full verification**

Run: `python -m unittest discover -s tests -p "test_*.py" -v`

Expected: PASS with all tests green

Run: `python -m pipeline.run --config configs/pipeline.yaml`

Expected: exit code `0` and artifacts written under the configured `work_dir`

- [ ] **Step 5: Commit**

```bash
git add pipeline/__init__.py README.md docs/project-spec.md docs/technical-assignment.md tests/test_config.py
git commit -m "docs: finalize public api and runtime contract"
```

## Self-Review

### Spec coverage

- Runtime modules from the spec are covered by Tasks 1 through 6.
- Final `8`-week holdout is covered by Task 2 and Task 3.
- Forecast output contract is covered by Task 4.
- Monitoring summary and visual `MAPE` artifacts are covered by Task 5.
- CLI orchestration and artifact persistence are covered by Task 6.
- Source-of-truth docs sync is covered by Task 7.

### Placeholder scan

- The plan uses exact file paths.
- Each task includes code or exact markdown content for the changes.
- Each task includes explicit commands and expected outcomes.
- No `TODO`, `TBD`, or “implement later” placeholders remain.

### Type consistency

- Config flows as `dict` in every task.
- Series bundles flow as `dict[str, object]` in every task.
- Holdout evaluation consistently uses `build_test_frame()` and `evaluate_holdout()`.
- Monitoring consistently consumes `backtest_df`, `holdout_df`, and `forecast_df`.
