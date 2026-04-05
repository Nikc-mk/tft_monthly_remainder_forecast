import json
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from pipeline.config import load_pipeline_config
from pipeline.run import PIPELINE_STAGE_LABELS, ConsoleStageProgressReporter, run_pipeline


class RunPipelineTests(unittest.TestCase):
    def _build_test_config(self) -> tuple[dict, Path]:
        config = load_pipeline_config("configs/pipeline.yaml")
        work_dir = Path("artifacts/test_run_output")
        work_dir.mkdir(parents=True, exist_ok=True)
        config["paths"]["raw_input"] = "tests/fixtures/raw_sales_small.csv"
        config["paths"]["work_dir"] = str(work_dir)
        config["paths"]["normalized_output"] = str(work_dir / "normalized_dataset.csv")
        config["paths"]["forecast_output"] = str(work_dir / "forecast.csv")
        config["training"]["max_encoder_length"] = 24
        config["training"]["max_epochs"] = 1
        config["training"]["hidden_size"] = 4
        config["training"]["attention_head_size"] = 1
        config["training"]["batch_size"] = 8
        config["training"]["learning_rate"] = 0.01
        config["quality_gates"]["smape_max"] = 100.0
        config["quality_gates"]["min_improvement_vs_best_baseline"] = -1.0
        return config, work_dir

    def test_run_pipeline_writes_expected_artifacts(self):
        config, work_dir = self._build_test_config()
        exit_code = run_pipeline(config)
        self.assertEqual(exit_code, 0)
        self.assertTrue(Path(config["paths"]["normalized_output"]).exists())
        self.assertTrue(Path(config["paths"]["forecast_output"]).exists())
        self.assertTrue((work_dir / "holdout_predictions.csv").exists())
        self.assertTrue((work_dir / "holdout_city_report.html").exists())
        self.assertTrue((work_dir / "monitoring_summary.json").exists())
        self.assertTrue((work_dir / "holdout_mape_by_model.png").exists())
        self.assertTrue((work_dir / "holdout_mape_by_week.png").exists())
        self.assertTrue((work_dir / "tft_residuals_analysis.png").exists())

    def test_run_pipeline_records_quality_gate_failure_without_crashing(self):
        config, work_dir = self._build_test_config()
        config["quality_gates"]["smape_max"] = 0.0
        exit_code = run_pipeline(config)

        self.assertEqual(exit_code, 0)
        summary = json.loads((work_dir / "monitoring_summary.json").read_text(encoding="utf-8"))
        self.assertFalse(summary["quality_gate_passed"])

    def test_console_stage_progress_reporter_formats_ascii_bar(self):
        stream = StringIO()
        reporter = ConsoleStageProgressReporter(total_stages=4, stream=stream, bar_width=8)

        reporter.report(2, 4, "Train TFT")

        self.assertEqual(stream.getvalue(), "[####----] 2/4 Train TFT\n")

    @patch("pipeline.run.save_holdout_city_html_report")
    @patch("pipeline.run.save_visual_artifacts")
    @patch("pipeline.run.build_monitoring_summary", return_value={"quality_gate_passed": True})
    @patch("pipeline.run.generate_forecast", return_value=pd.DataFrame({"forecast_week": ["2024-01-01"]}))
    @patch("pipeline.run.evaluate_holdout", return_value=pd.DataFrame({"City": ["A"]}))
    @patch("pipeline.run.check_quality_gates")
    @patch("pipeline.run.run_backtest", return_value=pd.DataFrame({"model_name": ["tft"], "smape": [1.0]}))
    @patch("pipeline.run.train_tft", return_value=object())
    @patch("pipeline.run.train_baselines", return_value={"seasonal_naive": object(), "rolling_mean": object()})
    @patch("pipeline.run.build_darts_series", return_value={"target_series_by_city": {}})
    @patch(
        "pipeline.run.build_inference_frame",
        side_effect=[
            pd.DataFrame({"week_start": pd.to_datetime(["2024-01-01"])}),
            pd.DataFrame({"week_start": pd.to_datetime(["2024-01-08"])}),
        ],
    )
    @patch("pipeline.run.build_test_frame", return_value=pd.DataFrame({"week_start": pd.to_datetime(["2024-01-15"])}))
    @patch("pipeline.run.build_training_frame", return_value=pd.DataFrame({"week_start": pd.to_datetime(["2024-01-01"])}))
    @patch(
        "pipeline.run.pd.read_csv",
        return_value=pd.DataFrame({"Week": ["2024-01-01"], "City": ["A"], "revenue": [10.0]}),
    )
    def test_run_pipeline_reports_all_named_stages(
        self,
        _mock_read_csv,
        _mock_build_training_frame,
        _mock_build_test_frame,
        _mock_build_inference_frame,
        _mock_build_darts_series,
        _mock_train_baselines,
        _mock_train_tft,
        _mock_run_backtest,
        _mock_check_quality_gates,
        _mock_evaluate_holdout,
        _mock_generate_forecast,
        _mock_build_monitoring_summary,
        _mock_save_holdout_city_html_report,
        _mock_save_visual_artifacts,
    ):
        config, _work_dir = self._build_test_config()
        reported_stages: list[tuple[int, int, str]] = []

        exit_code = run_pipeline(config, progress_reporter=lambda index, total, label: reported_stages.append((index, total, label)))

        self.assertEqual(exit_code, 0)
        _mock_save_holdout_city_html_report.assert_called_once()
        self.assertEqual([label for _, _, label in reported_stages], list(PIPELINE_STAGE_LABELS))
        self.assertTrue(all(total == len(PIPELINE_STAGE_LABELS) for _, total, _ in reported_stages))
