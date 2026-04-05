import unittest
from pathlib import Path

import pandas as pd

from pipeline.config import load_pipeline_config
from pipeline.data_pipeline import (
    build_darts_series,
    build_inference_frame,
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

    def test_build_training_frame_fills_missing_weeks_and_sums_duplicates(self):
        training_frame = build_training_frame(self.raw_df, self.config)
        filled_row = training_frame.loc[
            (training_frame["City"] == "A")
            & (training_frame["week_start"] == pd.Timestamp("2024-02-05"))
        ].iloc[0]
        self.assertEqual(filled_row["weekly_revenue"], 0.0)
        self.assertEqual(filled_row["is_missing"], 1)

        duplicate_row = training_frame.loc[
            (training_frame["City"] == "A")
            & (training_frame["week_start"] == pd.Timestamp("2024-03-11"))
        ].iloc[0]
        self.assertEqual(duplicate_row["weekly_revenue"], 135.0)

    def test_build_training_frame_counts_holidays_within_week(self):
        config = load_pipeline_config("configs/pipeline.yaml")
        config["features"]["holiday_dates"] = ["2024-03-11", "2024-03-13", "2024-03-20"]

        training_frame = build_training_frame(self.raw_df, config)
        holiday_row = training_frame.loc[
            (training_frame["City"] == "A")
            & (training_frame["week_start"] == pd.Timestamp("2024-03-11"))
        ].iloc[0]
        following_week_row = training_frame.loc[
            (training_frame["City"] == "A")
            & (training_frame["week_start"] == pd.Timestamp("2024-03-18"))
        ].iloc[0]

        self.assertEqual(holiday_row["is_holiday_week"], 2)
        self.assertEqual(following_week_row["is_holiday_week"], 1)

    def test_build_darts_series_returns_city_keyed_dicts(self):
        training_frame = build_training_frame(self.raw_df, self.config)
        series_data = build_darts_series(training_frame, self.config)
        self.assertIn("target_series_by_city", series_data)
        self.assertIn("past_covariates_by_city", series_data)
        self.assertIn("future_covariates_by_city", series_data)
        self.assertTrue(series_data["target_series_by_city"])

    def test_build_darts_series_keeps_city_as_series_identity_not_future_covariate(self):
        training_frame = build_training_frame(self.raw_df, self.config)
        inference_frame = build_inference_frame(
            self.raw_df,
            self.config,
            forecast_week=training_frame["week_start"].max(),
        )
        series_data = build_darts_series(inference_frame, self.config)

        city = next(iter(series_data["target_series_by_city"]))
        target_series = series_data["target_series_by_city"][city]
        future_covariates = series_data["future_covariates_by_city"][city]

        self.assertNotIn("City", future_covariates.components)
        self.assertEqual(list(future_covariates.components), ["week_of_year", "month", "quarter", "year", "is_holiday_week"])
        self.assertIn("city_code", target_series.static_covariates.columns)
        self.assertIn("city_code", future_covariates.static_covariates.columns)
        self.assertEqual(len(future_covariates), len(target_series) + self.config["training"]["max_prediction_length"])

    def test_get_last_complete_week_returns_monday(self):
        cutoff = get_last_complete_week(pd.Timestamp("2026-04-08 12:00:00"))
        self.assertEqual(cutoff.weekday(), 0)
        self.assertEqual(str(cutoff.date()), "2026-03-30")
