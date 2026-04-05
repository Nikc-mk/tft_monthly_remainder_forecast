import unittest
from pathlib import Path

import pandas as pd
from darts import TimeSeries

from pipeline.config import load_pipeline_config
from pipeline.monitoring import (
    build_monitoring_summary,
    save_holdout_city_html_report,
    save_visual_artifacts,
)


class FakeTFTModel:
    def __init__(self, residual_series: list[TimeSeries]):
        self._residual_series = residual_series
        self.training_series_by_city = {f"City_{idx}": series for idx, series in enumerate(residual_series, start=1)}
        self.training_past_covariates_by_city = self.training_series_by_city
        self.training_future_covariates_by_city = self.training_series_by_city

    def residuals(self, *args, **kwargs):
        return self._residual_series


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
                {
                    "City": "A",
                    "target_week": pd.Timestamp("2025-04-14"),
                    "model_name": "seasonal_naive",
                    "actual": 100.0,
                    "predict": 92.0,
                    "mape": 0.08,
                },
                {
                    "City": "A",
                    "target_week": pd.Timestamp("2025-04-14"),
                    "model_name": "tft",
                    "actual": 100.0,
                    "predict": 95.0,
                    "mape": 0.05,
                },
                {
                    "City": "A",
                    "target_week": pd.Timestamp("2025-04-14"),
                    "model_name": "rolling_mean",
                    "actual": 100.0,
                    "predict": 97.0,
                    "mape": 0.03,
                },
                {
                    "City": "B",
                    "target_week": pd.Timestamp("2025-04-21"),
                    "model_name": "seasonal_naive",
                    "actual": 80.0,
                    "predict": 88.0,
                    "mape": 0.10,
                },
                {
                    "City": "B",
                    "target_week": pd.Timestamp("2025-04-21"),
                    "model_name": "rolling_mean",
                    "actual": 80.0,
                    "predict": 84.0,
                    "mape": 0.05,
                },
                {
                    "City": "B",
                    "target_week": pd.Timestamp("2025-04-21"),
                    "model_name": "tft",
                    "actual": 80.0,
                    "predict": 90.0,
                    "mape": 0.125,
                },
            ]
        )
        cls.forecast_df = pd.DataFrame([{"City": "A", "target_week": pd.Timestamp("2025-04-14"), "q0.5": 101.0}])
        cls.fake_tft_model = FakeTFTModel(
            [
                TimeSeries.from_times_and_values(
                    pd.date_range("2025-01-06", periods=4, freq="W-MON"),
                    [[1.0], [-2.0], [0.5], [1.5]],
                ),
                TimeSeries.from_times_and_values(
                    pd.date_range("2025-01-06", periods=4, freq="W-MON"),
                    [[-0.5], [1.0], [0.25], [-1.0]],
                ),
            ]
        )

    def test_build_monitoring_summary_returns_key_metrics(self):
        summary = build_monitoring_summary(
            self.backtest_df,
            self.holdout_df,
            self.forecast_df,
            self.config,
            runtime_seconds=3.2,
        )
        self.assertIn("quality_gate_passed", summary)
        self.assertIn("holdout_mape_by_model", summary)
        self.assertIn("sla_passed", summary)

    def test_save_visual_artifacts_writes_png_files(self):
        output_dir = Path("artifacts/test_monitoring_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: [path.unlink(missing_ok=True) for path in output_dir.glob("*.png")])
        paths = save_visual_artifacts(self.holdout_df, output_dir, self.config, tft_model=self.fake_tft_model)
        self.assertEqual(len(paths), 4)
        self.assertTrue(all(Path(path).suffix == ".png" for path in paths))
        self.assertIn(output_dir / "holdout_mape_by_week.png", paths)
        self.assertIn(output_dir / "tft_residuals_analysis.png", paths)

    def test_save_holdout_city_html_report_writes_single_city_selector_report(self):
        output_dir = Path("artifacts/test_monitoring_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "holdout_city_report.html"
        self.addCleanup(report_path.unlink, missing_ok=True)

        path = save_holdout_city_html_report(self.holdout_df, output_dir)

        self.assertEqual(path, report_path)
        self.assertTrue(report_path.exists())
        html = report_path.read_text(encoding="utf-8")
        self.assertIn("holdout_city_report", html)
        self.assertIn('"A"', html)
        self.assertIn('"B"', html)
        self.assertIn("seasonal_naive", html)
        self.assertIn("rolling_mean", html)
        self.assertIn("tft", html)
        self.assertIn("actual", html)

    def test_save_holdout_city_html_report_handles_empty_holdout(self):
        output_dir = Path("artifacts/test_monitoring_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "holdout_city_report.html"
        self.addCleanup(report_path.unlink, missing_ok=True)

        empty_holdout = self.holdout_df.iloc[0:0].copy()

        path = save_holdout_city_html_report(empty_holdout, output_dir)

        self.assertEqual(path, report_path)
        self.assertTrue(report_path.exists())
        html = report_path.read_text(encoding="utf-8")
        self.assertIn("No holdout data available", html)
