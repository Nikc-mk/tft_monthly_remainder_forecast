"""Forecast formatting utilities for the weekly city runtime."""

from __future__ import annotations

import pandas as pd


def generate_forecast(series_data: dict[str, object], tft_model, config: dict) -> pd.DataFrame:
    """Generate the final forecast table with one row per city and target week."""
    rows: list[dict[str, object]] = []
    horizon = int(config["training"]["max_prediction_length"])
    generated_at = pd.Timestamp.now(tz="UTC").tz_localize(None)
    forecast_week = pd.Timestamp(series_data["forecast_week"]).normalize()
    for city, series in series_data["target_series_by_city"].items():
        prediction = tft_model.predict(
            n=horizon,
            series=series,
            past_covariates=series_data["past_covariates_by_city"][city],
            future_covariates=series_data["future_covariates_by_city"][city],
            num_samples=100,
        )
        q01 = prediction.quantile(0.1).values().reshape(-1)
        q05 = prediction.quantile(0.5).values().reshape(-1)
        q09 = prediction.quantile(0.9).values().reshape(-1)
        for idx, (value_01, value_05, value_09) in enumerate(zip(q01, q05, q09), start=1):
            target_week = forecast_week + pd.Timedelta(weeks=idx)
            rows.append(
                {
                    "forecast_week": forecast_week,
                    "target_week": target_week,
                    "horizon_week": idx,
                    "City": city,
                    "q0.1": float(value_01),
                    "q0.5": float(value_05),
                    "q0.9": float(value_09),
                    "model_version": config["metadata"]["model_version"],
                    "feature_version": config["metadata"]["feature_version"],
                    "generated_at": generated_at,
                }
            )
    return pd.DataFrame(rows)
