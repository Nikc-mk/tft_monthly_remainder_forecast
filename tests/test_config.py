import unittest
from pathlib import Path

from pipeline import load_pipeline_config as package_load_pipeline_config
from pipeline.config import load_pipeline_config, validate_pipeline_config


class ConfigTests(unittest.TestCase):
    def test_pipeline_package_exports_config_loader(self):
        self.assertTrue(callable(package_load_pipeline_config))

    def test_load_pipeline_config_reads_yaml(self):
        path = Path("tests/fixtures/pipeline_valid.yaml")
        config = load_pipeline_config(path)
        self.assertEqual(config["training"]["max_prediction_length"], 8)
        self.assertEqual(config["features"]["lag_periods"], [1, 2, 4, 8])

    def test_validate_pipeline_config_rejects_wrong_horizon(self):
        config = {
            "training": {
                "max_encoder_length": 60,
                "max_prediction_length": 7,
                "backtest_windows": 6,
            },
            "inference": {"quantiles": [0.1, 0.5, 0.9], "sla_seconds_cpu": 10.0},
            "data": {"min_history_weeks": 60},
            "quality_gates": {
                "smape_max": 15.0,
                "min_improvement_vs_best_baseline": 0.05,
            },
            "paths": {
                "raw_input": "x",
                "work_dir": "y",
                "normalized_output": "z",
                "forecast_output": "q",
            },
            "run": {"mode": "full", "skip_tft": False},
            "features": {
                "lag_periods": [1, 2, 4, 8],
                "rolling_windows": [4, 8, 12],
                "holiday_dates": [],
            },
            "metadata": {"feature_version": "f", "model_version": "m"},
        }
        with self.assertRaisesRegex(ValueError, "max_prediction_length"):
            validate_pipeline_config(config)
