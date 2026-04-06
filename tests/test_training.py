import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
from darts.models.forecasting.tft_model import TFTModel

from pipeline.config import load_pipeline_config
from pipeline.data_pipeline import build_darts_series, build_inference_frame, build_test_frame, build_training_frame
from pipeline.training import (
    _tft_trainer_kwargs,
    check_quality_gates,
    evaluate_holdout,
    run_backtest,
    train_baselines,
    train_tft,
)


def _lightweight_tft_config() -> dict:
    config = load_pipeline_config("configs/pipeline.yaml")
    config["training"]["max_encoder_length"] = 24
    config["training"]["max_epochs"] = 1
    config["training"]["hidden_size"] = 4
    config["training"]["attention_head_size"] = 1
    config["training"]["batch_size"] = 8
    config["training"]["learning_rate"] = 0.01
    config["training"]["accelerator"] = "cpu"
    config["training"]["devices"] = 1
    config["quality_gates"]["smape_max"] = 100.0
    config["quality_gates"]["min_improvement_vs_best_baseline"] = -1.0
    return config


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = _lightweight_tft_config()
        raw_df = pd.read_csv("tests/fixtures/raw_sales_small.csv", sep=";")
        cls.training_frame = build_training_frame(raw_df, cls.config)
        cls.test_frame = build_test_frame(raw_df, cls.config)
        cls.inference_frame = build_inference_frame(
            raw_df,
            cls.config,
            forecast_week=cls.training_frame["week_start"].max(),
        )
        cls.series_data = build_darts_series(cls.inference_frame, cls.config)

    def test_train_baselines_returns_expected_models(self):
        baselines = train_baselines(self.series_data, self.config)
        self.assertEqual(set(baselines), {"seasonal_naive", "rolling_mean"})

    def test_run_backtest_returns_scores_for_all_models_and_windows(self):
        baselines = train_baselines(self.series_data, self.config)
        tft_model = train_tft(self.series_data, self.config)
        backtest_df = run_backtest(self.series_data, tft_model, baselines, self.config)
        self.assertTrue({"City", "window_id", "model_name", "smape"}.issubset(backtest_df.columns))
        self.assertEqual(set(backtest_df["model_name"]), {"seasonal_naive", "rolling_mean", "tft"})
        self.assertEqual(backtest_df["window_id"].nunique(), self.config["training"]["backtest_windows"])

    def test_evaluate_holdout_returns_actual_vs_predict_rows(self):
        baselines = train_baselines(self.series_data, self.config)
        tft_model = train_tft(self.series_data, self.config)
        holdout_df = evaluate_holdout(self.test_frame, tft_model, baselines, self.config)
        self.assertTrue(
            {"City", "target_week", "model_name", "actual", "predict", "mape"}.issubset(holdout_df.columns)
        )
        counts = holdout_df.groupby(["City", "model_name"])["target_week"].count()
        self.assertTrue((counts == 8).all())

    def test_check_quality_gates_rejects_bad_backtest(self):
        strict_config = _lightweight_tft_config()
        strict_config["quality_gates"]["smape_max"] = 15.0
        strict_config["quality_gates"]["min_improvement_vs_best_baseline"] = 0.05
        backtest_df = pd.DataFrame(
            [
                {"model_name": "seasonal_naive", "smape": 8.0},
                {"model_name": "rolling_mean", "smape": 9.0},
                {"model_name": "tft", "smape": 8.5},
            ]
        )
        with self.assertRaisesRegex(ValueError, "quality gate failed"):
            check_quality_gates(backtest_df, strict_config)

    def test_check_quality_gates_accepts_runtime_backtest(self):
        permissive_config = _lightweight_tft_config()
        permissive_config["quality_gates"]["smape_max"] = 1000.0
        permissive_config["quality_gates"]["min_improvement_vs_best_baseline"] = -100.0
        baselines = train_baselines(self.series_data, self.config)
        tft_model = train_tft(self.series_data, self.config)
        backtest_df = run_backtest(self.series_data, tft_model, baselines, self.config)
        check_quality_gates(backtest_df, permissive_config)

    def test_train_tft_returns_real_tft_model(self):
        tft_model = train_tft(self.series_data, self.config)
        self.assertIsInstance(tft_model, TFTModel)

    def test_tft_trainer_kwargs_enable_progress_bar(self):
        trainer_kwargs = _tft_trainer_kwargs(self.config)
        self.assertTrue(trainer_kwargs["enable_progress_bar"])

    @patch("pipeline.training.torch.cuda.is_available", return_value=False)
    def test_tft_trainer_kwargs_fall_back_to_cpu_when_gpu_unavailable(self, _mock_cuda):
        config = _lightweight_tft_config()
        config["training"]["accelerator"] = "gpu"
        config["training"]["devices"] = 1

        trainer_kwargs = _tft_trainer_kwargs(config)

        self.assertEqual(trainer_kwargs["accelerator"], "cpu")
        self.assertEqual(trainer_kwargs["devices"], 1)

    @patch("pipeline.training.torch.cuda.is_available", return_value=True)
    def test_tft_trainer_kwargs_use_gpu_when_available(self, _mock_cuda):
        config = _lightweight_tft_config()
        config["training"]["accelerator"] = "auto"
        config["training"]["devices"] = 1

        trainer_kwargs = _tft_trainer_kwargs(config)

        self.assertEqual(trainer_kwargs["accelerator"], "gpu")
        self.assertEqual(trainer_kwargs["devices"], 1)

    @patch("pipeline.training.TFTModel")
    def test_train_tft_uses_verbose_fit(self, mock_tft_model):
        model_instance = MagicMock()
        mock_tft_model.return_value = model_instance

        train_tft(self.series_data, self.config)

        self.assertTrue(mock_tft_model.call_args.kwargs["pl_trainer_kwargs"]["enable_progress_bar"])
        self.assertTrue(model_instance.fit.call_args.kwargs["verbose"])

    @patch("pipeline.training.TFTModel")
    def test_train_tft_passes_resolved_trainer_kwargs(self, mock_tft_model):
        model_instance = MagicMock()
        mock_tft_model.return_value = model_instance

        train_tft(self.series_data, self.config)

        self.assertEqual(mock_tft_model.call_args.kwargs["pl_trainer_kwargs"]["accelerator"], "cpu")
        self.assertEqual(mock_tft_model.call_args.kwargs["pl_trainer_kwargs"]["devices"], 1)

    def test_train_tft_predict_returns_probabilistic_forecast(self):
        tft_model = train_tft(self.series_data, self.config)
        city = next(iter(self.series_data["target_series_by_city"]))
        prediction = tft_model.predict(
            n=self.config["training"]["max_prediction_length"],
            series=self.series_data["target_series_by_city"][city],
            future_covariates=self.series_data["future_covariates_by_city"][city],
            num_samples=20,
        )

        self.assertEqual(len(prediction), self.config["training"]["max_prediction_length"])
        self.assertGreater(prediction.n_samples, 1)

    @patch("pipeline.training.TFTModel")
    def test_train_tft_does_not_pass_past_covariates_to_fit(self, mock_tft_model):
        model_instance = MagicMock()
        mock_tft_model.return_value = model_instance

        train_tft(self.series_data, self.config)

        self.assertNotIn("past_covariates", model_instance.fit.call_args.kwargs)
