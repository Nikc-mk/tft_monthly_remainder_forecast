"""Вспомогательные функции обучения, бэктеста и оценки holdout для недельного рантайма."""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType

import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
from darts.models.forecasting.tft_model import TFTModel
from darts.utils.likelihood_models import QuantileRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler


TFT_RANDOM_STATE = 42
TFT_PREDICT_SAMPLES = 100


@dataclass
class RuntimeTFTScalers:
    """Скейлеры по городам и глобальные скейлеры для стабилизации обучения и инференса TFT."""

    city_by_code: dict[float, str]
    target_scalers_by_city: dict[str, StandardScaler]
    past_scalers_by_city: dict[str, StandardScaler]
    future_scaler: MinMaxScaler
    static_scaler: MinMaxScaler


def _prediction_index(series: TimeSeries, n: int) -> pd.DatetimeIndex:
    start = series.end_time() + series.freq
    return pd.date_range(start, periods=n, freq=series.freq_str)


def _timeseries_from_prediction(series: TimeSeries, values: np.ndarray) -> TimeSeries:
    return TimeSeries.from_times_and_values(_prediction_index(series, len(values)), values.reshape(-1, 1))


def _series_values(series: TimeSeries) -> np.ndarray:
    return series.values().reshape(-1).astype(float)


def _smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = np.abs(actual) + np.abs(predicted)
    denominator = np.where(denominator == 0.0, 1e-9, denominator)
    return float(np.mean(2.0 * np.abs(predicted - actual) / denominator) * 100.0)


def _mape(actual: float, predicted: float) -> float:
    return abs(actual - predicted) / max(abs(actual), 1e-9)


def _city_code_from_series(series: TimeSeries) -> float:
    static_covariates = series.static_covariates
    if static_covariates is None or "city_code" not in static_covariates.columns:
        raise ValueError("Series is missing the required 'city_code' static covariate")
    return float(static_covariates.iloc[0]["city_code"])


def _transform_static_covariates(
    static_covariates: pd.DataFrame | None,
    static_scaler: MinMaxScaler,
) -> pd.DataFrame | None:
    if static_covariates is None:
        return None
    transformed = static_covariates.copy()
    transformed[["city_code"]] = static_scaler.transform(transformed[["city_code"]].astype(float))
    return transformed


def _transform_series_values(series: TimeSeries, transformer, inverse: bool = False) -> np.ndarray:
    values = series.all_values(copy=False)
    time_steps, components, samples = values.shape
    flattened = values.transpose(0, 2, 1).reshape(-1, components)
    if inverse:
        transformed = transformer.inverse_transform(flattened)
    else:
        transformed = transformer.transform(flattened)
    return transformed.reshape(time_steps, samples, components).transpose(0, 2, 1)


def _series_with_transformed_values(
    series: TimeSeries,
    transformer,
    static_scaler: MinMaxScaler,
    inverse: bool = False,
) -> TimeSeries:
    return TimeSeries.from_times_and_values(
        times=series.time_index,
        values=_transform_series_values(series, transformer, inverse=inverse),
        freq=series.freq_str,
        columns=series.components,
        static_covariates=_transform_static_covariates(series.static_covariates, static_scaler),
        hierarchy=series.hierarchy,
        metadata=series.metadata,
    )


def _fit_runtime_tft_scalers(series_data: dict[str, object]) -> RuntimeTFTScalers:
    target_series_by_city = series_data["target_series_by_city"]
    past_covariates_by_city = series_data["past_covariates_by_city"]
    future_covariates_by_city = series_data["future_covariates_by_city"]

    city_by_code = {
        _city_code_from_series(series): city for city, series in target_series_by_city.items()
    }
    target_scalers_by_city = {
        city: StandardScaler().fit(_series_values(series).reshape(-1, 1))
        for city, series in target_series_by_city.items()
    }
    past_scalers_by_city = {
        city: StandardScaler().fit(series.values(copy=False))
        for city, series in past_covariates_by_city.items()
    }
    future_scaler = MinMaxScaler(feature_range=(0.0, 1.0)).fit(
        np.vstack([series.values(copy=False) for series in future_covariates_by_city.values()])
    )
    static_scaler = MinMaxScaler(feature_range=(0.0, 1.0)).fit(
        pd.DataFrame({"city_code": sorted(city_by_code)})
    )
    return RuntimeTFTScalers(
        city_by_code=city_by_code,
        target_scalers_by_city=target_scalers_by_city,
        past_scalers_by_city=past_scalers_by_city,
        future_scaler=future_scaler,
        static_scaler=static_scaler,
    )


def _scale_series_data_for_tft(series_data: dict[str, object], scalers: RuntimeTFTScalers) -> dict[str, object]:
    return {
        "target_series_by_city": {
            city: _series_with_transformed_values(
                series,
                scalers.target_scalers_by_city[city],
                scalers.static_scaler,
            )
            for city, series in series_data["target_series_by_city"].items()
        },
        "past_covariates_by_city": {
            city: _series_with_transformed_values(
                series,
                scalers.past_scalers_by_city[city],
                scalers.static_scaler,
            )
            for city, series in series_data["past_covariates_by_city"].items()
        },
        "future_covariates_by_city": {
            city: _series_with_transformed_values(
                series,
                scalers.future_scaler,
                scalers.static_scaler,
            )
            for city, series in series_data["future_covariates_by_city"].items()
        },
    }


def _predict_with_runtime_scaling(
    self,
    n: int,
    series: TimeSeries | None = None,
    past_covariates: TimeSeries | None = None,
    future_covariates: TimeSeries | None = None,
    **kwargs: object,
) -> TimeSeries:
    if series is None or past_covariates is None or future_covariates is None:
        raise ValueError("Runtime TFT predict requires series, past_covariates, and future_covariates")
    city_code = _city_code_from_series(series)
    city = self.runtime_tft_scalers.city_by_code[city_code]
    scaled_prediction = self._raw_predict(
        n=n,
        series=_series_with_transformed_values(
            series,
            self.runtime_tft_scalers.target_scalers_by_city[city],
            self.runtime_tft_scalers.static_scaler,
        ),
        past_covariates=_series_with_transformed_values(
            past_covariates,
            self.runtime_tft_scalers.past_scalers_by_city[city],
            self.runtime_tft_scalers.static_scaler,
        ),
        future_covariates=_series_with_transformed_values(
            future_covariates,
            self.runtime_tft_scalers.future_scaler,
            self.runtime_tft_scalers.static_scaler,
        ),
        **kwargs,
    )
    return _series_with_transformed_values(
        scaled_prediction,
        self.runtime_tft_scalers.target_scalers_by_city[city],
        self.runtime_tft_scalers.static_scaler,
        inverse=True,
    )


@dataclass
class SeasonalNaiveModel:
    """Повторять последний сезонный блок как прогноз на следующий горизонт."""

    series_by_city: dict[str, TimeSeries]

    def predict(self, n: int, series: TimeSeries | None = None, **_: object) -> TimeSeries:
        target_series = series if series is not None else next(iter(self.series_by_city.values()))
        values = _series_values(target_series)
        if len(values) == 0:
            forecast = np.zeros(n, dtype=float)
        elif len(values) >= n:
            forecast = values[-n:]
        else:
            forecast = np.resize(values, n)
        return _timeseries_from_prediction(target_series, forecast)


@dataclass
class RollingMeanModel:
    """Строить прогноз по среднему значению последних наблюдений."""

    series_by_city: dict[str, TimeSeries]
    window: int = 4

    def predict(self, n: int, series: TimeSeries | None = None, **_: object) -> TimeSeries:
        target_series = series if series is not None else next(iter(self.series_by_city.values()))
        values = _series_values(target_series)
        if len(values) == 0:
            mean_value = 0.0
        else:
            mean_value = float(np.mean(values[-min(self.window, len(values)) :]))
        forecast = np.repeat(mean_value, n)
        return _timeseries_from_prediction(target_series, forecast)


def _resolve_tft_training_device(config: dict) -> tuple[str, int]:
    requested_accelerator = str(config["training"].get("accelerator", "auto")).lower()
    requested_devices = int(config["training"].get("devices", 1))
    if requested_accelerator == "cpu":
        return "cpu", 1
    if torch.cuda.is_available():
        return "gpu", requested_devices
    return "cpu", 1


def _tft_trainer_kwargs(config: dict) -> dict[str, object]:
    accelerator, devices = _resolve_tft_training_device(config)
    return {
        "accelerator": accelerator,
        "devices": devices,
        "enable_progress_bar": True,
        "logger": False,
        "enable_model_summary": False,
    }


def _get_tft_quantiles(config: dict) -> list[float]:
    return [float(value) for value in config["inference"]["quantiles"]]


def _predict_tft_quantiles(
    tft_model: TFTModel,
    series: TimeSeries,
    past_covariates: TimeSeries,
    future_covariates: TimeSeries,
    horizon: int,
) -> dict[float, np.ndarray]:
    prediction = tft_model.predict(
        n=horizon,
        series=series,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        num_samples=TFT_PREDICT_SAMPLES,
    )
    return {
        0.1: prediction.quantile(0.1).values().reshape(-1).astype(float),
        0.5: prediction.quantile(0.5).values().reshape(-1).astype(float),
        0.9: prediction.quantile(0.9).values().reshape(-1).astype(float),
    }


def train_baselines(series_data: dict[str, object], config: dict) -> dict[str, object]:
    """Создать базовые модели прогноза поверх подготовленных Darts-серий по городам."""
    del config
    series_by_city = series_data["target_series_by_city"]
    return {
        "seasonal_naive": SeasonalNaiveModel(series_by_city),
        "rolling_mean": RollingMeanModel(series_by_city),
    }


def train_tft(series_data: dict[str, object], config: dict) -> TFTModel:
    """Обучить реальную модель Darts TFTModel на подготовленных недельных сериях по городам."""
    runtime_tft_scalers = _fit_runtime_tft_scalers(series_data)
    scaled_series_data = _scale_series_data_for_tft(series_data, runtime_tft_scalers)
    tft_model = TFTModel(
        input_chunk_length=int(config["training"]["max_encoder_length"]),
        output_chunk_length=int(config["training"]["max_prediction_length"]),
        hidden_size=int(config["training"]["hidden_size"]),
        lstm_layers=1,
        num_attention_heads=int(config["training"]["attention_head_size"]),
        dropout=float(config["training"]["dropout"]),
        batch_size=int(config["training"]["batch_size"]),
        n_epochs=int(config["training"]["max_epochs"]),
        add_relative_index=False,
        likelihood=QuantileRegression(_get_tft_quantiles(config)),
        random_state=TFT_RANDOM_STATE,
        force_reset=True,
        save_checkpoints=False,
        optimizer_kwargs={"lr": float(config["training"]["learning_rate"])},
        pl_trainer_kwargs=_tft_trainer_kwargs(config),
    )
    tft_model.fit(
        series=list(scaled_series_data["target_series_by_city"].values()),
        past_covariates=list(scaled_series_data["past_covariates_by_city"].values()),
        future_covariates=list(scaled_series_data["future_covariates_by_city"].values()),
        verbose=True,
    )
    tft_model.runtime_tft_scalers = runtime_tft_scalers
    tft_model._raw_predict = tft_model.predict
    tft_model.predict = MethodType(_predict_with_runtime_scaling, tft_model)
    tft_model.training_series_by_city = series_data["target_series_by_city"]
    tft_model.training_past_covariates_by_city = series_data["past_covariates_by_city"]
    tft_model.training_future_covariates_by_city = series_data["future_covariates_by_city"]
    return tft_model


def run_backtest(
    series_data: dict[str, object],
    tft_model,
    baselines: dict[str, object],
    config: dict,
) -> pd.DataFrame:
    """Запустить rolling weekly backtest и вернуть SMAPE по городам и окнам."""
    rows: list[dict[str, object]] = []
    horizon = int(config["training"]["max_prediction_length"])
    windows = int(config["training"]["backtest_windows"])
    for city, series in series_data["target_series_by_city"].items():
        if len(series) < horizon + windows:
            raise ValueError(f"Series for city '{city}' is too short for the configured backtest")
        values = _series_values(series)
        city_past_covariates = series_data["past_covariates_by_city"][city]
        city_future_covariates = series_data["future_covariates_by_city"][city]
        for window_id in range(1, windows + 1):
            start_idx = len(values) - horizon - windows + window_id - 1
            history = series[:start_idx]
            actual = values[start_idx : start_idx + horizon]

            seasonal_pred = _series_values(baselines["seasonal_naive"].predict(horizon, series=history))
            rolling_pred = _series_values(baselines["rolling_mean"].predict(horizon, series=history))
            tft_pred = _predict_tft_quantiles(
                tft_model,
                series=history,
                past_covariates=city_past_covariates,
                future_covariates=city_future_covariates,
                horizon=horizon,
            )[0.5]

            rows.extend(
                [
                    {
                        "City": city,
                        "window_id": window_id,
                        "model_name": "seasonal_naive",
                        "smape": _smape(actual, seasonal_pred),
                    },
                    {
                        "City": city,
                        "window_id": window_id,
                        "model_name": "rolling_mean",
                        "smape": _smape(actual, rolling_pred),
                    },
                    {
                        "City": city,
                        "window_id": window_id,
                        "model_name": "tft",
                        "smape": _smape(actual, tft_pred),
                    },
                ]
            )
    return pd.DataFrame(rows)


def evaluate_holdout(test_frame: pd.DataFrame, tft_model, baselines: dict[str, object], config: dict) -> pd.DataFrame:
    """Оценить финальный holdout на 8 недель относительно базовых моделей и реального TFT."""
    rows: list[dict[str, object]] = []
    horizon = int(config["training"]["max_prediction_length"])
    for city, city_frame in test_frame.groupby("City", sort=True):
        city_frame = city_frame.sort_values("week_start").reset_index(drop=True)
        actual = city_frame["weekly_revenue"].to_numpy(dtype=float)
        history_series = tft_model.training_series_by_city[city]
        model_predictions = {
            "seasonal_naive": _series_values(baselines["seasonal_naive"].predict(horizon, series=history_series)),
            "rolling_mean": _series_values(baselines["rolling_mean"].predict(horizon, series=history_series)),
            "tft": _predict_tft_quantiles(
                tft_model,
                series=history_series,
                past_covariates=tft_model.training_past_covariates_by_city[city],
                future_covariates=tft_model.training_future_covariates_by_city[city],
                horizon=horizon,
            )[0.5],
        }

        for model_name, predicted in model_predictions.items():
            for idx, row in city_frame.iterrows():
                rows.append(
                    {
                        "City": city,
                        "target_week": row["week_start"],
                        "model_name": model_name,
                        "actual": float(actual[idx]),
                        "predict": float(predicted[idx]),
                        "mape": _mape(float(actual[idx]), float(predicted[idx])),
                    }
                )
    return pd.DataFrame(rows)


def check_quality_gates(backtest_df: pd.DataFrame, config: dict) -> None:
    """Проверить настроенные пороги SMAPE и улучшения на результатах бэктеста."""
    grouped = backtest_df.groupby("model_name", as_index=False)["smape"].mean()
    best_baseline = grouped.loc[grouped["model_name"] != "tft", "smape"].min()
    tft_smape = grouped.loc[grouped["model_name"] == "tft", "smape"].iloc[0]
    max_smape = float(config["quality_gates"]["smape_max"])
    min_improvement = float(config["quality_gates"]["min_improvement_vs_best_baseline"])
    actual_improvement = 0.0 if best_baseline == 0.0 else (best_baseline - tft_smape) / best_baseline
    if tft_smape > max_smape or actual_improvement < min_improvement:
        raise ValueError("quality gate failed for TFT backtest results")
