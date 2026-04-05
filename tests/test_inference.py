import unittest

import pandas as pd

from pipeline.config import load_pipeline_config
from pipeline.data_pipeline import build_darts_series, build_inference_frame, build_training_frame
from pipeline.inference import generate_forecast
from pipeline.training import train_tft


def _lightweight_tft_config() -> dict:
    config = load_pipeline_config("configs/pipeline.yaml")
    config["training"]["max_encoder_length"] = 24
    config["training"]["max_epochs"] = 1
    config["training"]["hidden_size"] = 4
    config["training"]["attention_head_size"] = 1
    config["training"]["batch_size"] = 8
    config["training"]["learning_rate"] = 0.01
    return config


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = _lightweight_tft_config()
        raw_df = pd.read_csv("tests/fixtures/raw_sales_small.csv", sep=";")
        cls.training_frame = build_training_frame(raw_df, cls.config)
        inference_frame = build_inference_frame(
            raw_df,
            cls.config,
            forecast_week=cls.training_frame["week_start"].max(),
        )
        cls.series_data = build_darts_series(inference_frame, cls.config)
        cls.tft_model = train_tft(cls.series_data, cls.config)

    def test_generate_forecast_returns_contract_columns(self):
        forecast_df = generate_forecast(self.series_data, self.tft_model, self.config)
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
        forecast_df = generate_forecast(self.series_data, self.tft_model, self.config)
        counts = forecast_df.groupby("City")["target_week"].count()
        self.assertTrue((counts == 8).all())

    def test_generate_forecast_populates_quantile_columns(self):
        forecast_df = generate_forecast(self.series_data, self.tft_model, self.config)
        self.assertFalse(forecast_df[["q0.1", "q0.5", "q0.9"]].isna().any().any())

    def test_generate_forecast_preserves_city_scale(self):
        forecast_df = generate_forecast(self.series_data, self.tft_model, self.config)
        recent_history = (
            self.training_frame.sort_values("week_start")
            .groupby("City")
            .tail(self.config["training"]["max_prediction_length"])
            .groupby("City")["weekly_revenue"]
            .mean()
            .abs()
        )
        forecast_scale = forecast_df.groupby("City")["q0.5"].mean().abs()
        scale_ratio = forecast_scale / recent_history.replace(0.0, 1e-9)

        self.assertTrue(
            (scale_ratio > 0.1).all(),
            msg=f"TFT forecast collapsed in scale: {scale_ratio.to_dict()}",
        )
