"""Утилиты подготовки недельных данных для рантайма прогнозирования по городам."""

from __future__ import annotations

import numpy as np
import pandas as pd
from darts import TimeSeries


PAST_FEATURES = ["lag_1", "lag_2", "lag_4", "lag_8", "rolling_4", "rolling_8", "rolling_12"]
FUTURE_FEATURES = ["week_of_year", "month", "quarter", "year", "is_holiday_week"]


def validate_raw_frame(raw_df: pd.DataFrame, config: dict) -> None:
    """Проверить исходный недельный датасет продаж на соответствие входной схеме."""
    required = {
        config["data"]["time_column"],
        config["data"]["group_column"],
        config["data"]["target_column"],
    }
    missing = required.difference(raw_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def get_last_complete_week(reference_ts: pd.Timestamp | None = None) -> pd.Timestamp:
    """Вернуть понедельник последней полностью закрытой ISO-недели для опорного времени."""
    ts = pd.Timestamp.now(tz="UTC") if reference_ts is None else pd.Timestamp(reference_ts)
    ts = ts.tz_localize(None) if ts.tzinfo is not None else ts
    current_monday = ts.normalize() - pd.to_timedelta(ts.weekday(), unit="D")
    return current_monday - pd.Timedelta(weeks=1)


def _to_week_start(series: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(series)
    return timestamps.dt.normalize() - pd.to_timedelta(timestamps.dt.weekday, unit="D")


def _normalize_weekly_frame(raw_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Построить полный грид город-неделя с агрегацией дублей и отметками пропусков."""
    validate_raw_frame(raw_df, config)
    frame = raw_df.copy()
    time_col = config["data"]["time_column"]
    group_col = config["data"]["group_column"]
    target_col = config["data"]["target_column"]
    fill_missing_revenue = float(config["data"]["fill_missing_revenue"])

    frame["week_start"] = _to_week_start(frame[time_col])
    cutoff = get_last_complete_week()
    frame = frame.loc[frame["week_start"] <= cutoff].copy()
    if frame.empty:
        raise ValueError("No raw rows remain after applying the last complete week cutoff")

    aggregated = (
        frame.groupby([group_col, "week_start"], as_index=False)[target_col]
        .sum()
        .rename(columns={group_col: "City", target_col: "weekly_revenue"})
        .sort_values(["City", "week_start"])
        .reset_index(drop=True)
    )

    city_frames: list[pd.DataFrame] = []
    for city, city_frame in aggregated.groupby("City", sort=True):
        full_weeks = pd.date_range(
            city_frame["week_start"].min(),
            city_frame["week_start"].max(),
            freq="W-MON",
        )
        city_grid = pd.DataFrame({"week_start": full_weeks})
        city_grid["City"] = city
        city_grid = city_grid.merge(city_frame, on=["City", "week_start"], how="left")
        city_grid["is_missing"] = city_grid["weekly_revenue"].isna().astype(int)
        city_grid["weekly_revenue"] = city_grid["weekly_revenue"].fillna(fill_missing_revenue)
        city_frames.append(city_grid)

    return pd.concat(city_frames, ignore_index=True).sort_values(["City", "week_start"]).reset_index(drop=True)


def _add_calendar_features(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    frame = frame.copy()
    iso_calendar = frame["week_start"].dt.isocalendar()
    frame["week_of_year"] = iso_calendar.week.astype(int)
    frame["month"] = frame["week_start"].dt.month
    frame["quarter"] = frame["week_start"].dt.quarter
    frame["year"] = frame["week_start"].dt.year
    holiday_dates = sorted({pd.Timestamp(value).normalize() for value in config["features"]["holiday_dates"]})
    if holiday_dates:
        holiday_week_starts = _to_week_start(pd.Series(holiday_dates))
        holiday_counts = holiday_week_starts.value_counts()
        frame["is_holiday_week"] = frame["week_start"].map(holiday_counts).fillna(0).astype(int)
    else:
        frame["is_holiday_week"] = 0
    return frame


def _add_past_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Добавить лаги и скользящие признаки через shift(1), чтобы избежать утечки."""
    frame = frame.copy()
    by_city = frame.groupby("City")["weekly_revenue"]
    shifted = by_city.shift(1)
    frame["lag_1"] = shifted
    frame["lag_2"] = by_city.shift(2)
    frame["lag_4"] = by_city.shift(4)
    frame["lag_8"] = by_city.shift(8)
    frame["rolling_4"] = shifted.groupby(frame["City"]).rolling(4).mean().reset_index(level=0, drop=True)
    frame["rolling_8"] = shifted.groupby(frame["City"]).rolling(8).mean().reset_index(level=0, drop=True)
    frame["rolling_12"] = shifted.groupby(frame["City"]).rolling(12).mean().reset_index(level=0, drop=True)
    return frame


def _build_feature_frame(raw_df: pd.DataFrame, config: dict, forecast_week: pd.Timestamp | None = None) -> pd.DataFrame:
    frame = _normalize_weekly_frame(raw_df, config)
    if forecast_week is not None:
        frame = frame.loc[frame["week_start"] <= pd.Timestamp(forecast_week).normalize()].copy()
    frame = _add_calendar_features(frame, config)
    frame = _add_past_features(frame)
    return frame.reset_index(drop=True)


def build_training_frame(raw_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Вернуть фрейм признаков для обучения и rolling-валидации без финального holdout на 8 недель."""
    frame = _build_feature_frame(raw_df, config)
    max_week = frame["week_start"].max()
    holdout_start = max_week - pd.Timedelta(weeks=7)
    return frame.loc[frame["week_start"] < holdout_start].reset_index(drop=True)


def build_test_frame(raw_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Вернуть последние 8 полностью закрытых недель, выделенных под holdout после валидации."""
    frame = _build_feature_frame(raw_df, config)
    max_week = frame["week_start"].max()
    holdout_start = max_week - pd.Timedelta(weeks=7)
    return frame.loc[frame["week_start"] >= holdout_start].reset_index(drop=True)


def build_inference_frame(
    raw_df: pd.DataFrame,
    config: dict,
    forecast_week: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Вернуть наблюдаемую историю и будущие календарные ковариаты для выбранной недели прогноза."""
    selected_week = get_last_complete_week() if forecast_week is None else pd.Timestamp(forecast_week).normalize()
    observed_frame = _build_feature_frame(raw_df, config, forecast_week=selected_week)

    future_weeks = pd.date_range(
        selected_week + pd.Timedelta(weeks=1),
        periods=int(config["training"]["max_prediction_length"]),
        freq="W-MON",
    )
    future_rows: list[dict[str, object]] = []
    for city in observed_frame["City"].unique():
        for week_start in future_weeks:
            future_rows.append(
                {
                    "week_start": week_start,
                    "City": city,
                    "weekly_revenue": np.nan,
                    "is_missing": 1,
                    "lag_1": np.nan,
                    "lag_2": np.nan,
                    "lag_4": np.nan,
                    "lag_8": np.nan,
                    "rolling_4": np.nan,
                    "rolling_8": np.nan,
                    "rolling_12": np.nan,
                }
            )

    future_frame = pd.DataFrame(future_rows)
    future_frame = _add_calendar_features(future_frame, config)
    combined = pd.concat([observed_frame, future_frame], ignore_index=True, sort=False)
    return combined.sort_values(["City", "week_start"]).reset_index(drop=True)


def _encode_numeric_static_covariates(series_by_city: dict[str, TimeSeries]) -> dict[str, TimeSeries]:
    encoded: dict[str, TimeSeries] = {}
    for city_code, city in enumerate(sorted(series_by_city), start=1):
        series = series_by_city[city]
        static_covariates = pd.DataFrame({"city_code": [float(city_code)]}, index=series.components)
        encoded[city] = series.with_static_covariates(static_covariates)
    return encoded


def build_darts_series(frame: pd.DataFrame, config: dict) -> dict[str, object]:
    """Преобразовать подготовленный недельный фрейм в Darts-серии цели и ковариат по городам."""
    fill_value = float(config["data"]["fill_missing_revenue"])
    observed_frame = frame.loc[frame["weekly_revenue"].notna()].copy()
    past_frame = observed_frame.copy()
    past_frame[PAST_FEATURES] = past_frame[PAST_FEATURES].fillna(fill_value)

    target_series = TimeSeries.from_group_dataframe(
        df=observed_frame,
        group_cols="City",
        time_col="week_start",
        value_cols="weekly_revenue",
        fill_missing_dates=True,
        freq="W-MON",
        fillna_value=fill_value,
    )
    past_covariates = TimeSeries.from_group_dataframe(
        df=past_frame,
        group_cols="City",
        time_col="week_start",
        value_cols=PAST_FEATURES,
        fill_missing_dates=True,
        freq="W-MON",
        fillna_value=fill_value,
    )
    future_covariates = TimeSeries.from_group_dataframe(
        df=frame,
        group_cols="City",
        time_col="week_start",
        value_cols=FUTURE_FEATURES,
        fill_missing_dates=True,
        freq="W-MON",
        fillna_value=0.0,
    )
    target_by_city = _encode_numeric_static_covariates(
        {city: series for city, series in zip(observed_frame["City"].drop_duplicates(), target_series)}
    )
    past_by_city = _encode_numeric_static_covariates(
        {city: series for city, series in zip(observed_frame["City"].drop_duplicates(), past_covariates)}
    )
    future_by_city = _encode_numeric_static_covariates(
        {city: series for city, series in zip(frame["City"].drop_duplicates(), future_covariates)}
    )
    return {
        "target_series_by_city": target_by_city,
        "past_covariates_by_city": past_by_city,
        "future_covariates_by_city": future_by_city,
        "forecast_week": observed_frame["week_start"].max(),
    }
