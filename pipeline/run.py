"""CLI-точка входа для недельного рантайма прогнозирования по городам."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TextIO

import pandas as pd

from pipeline.config import load_pipeline_config
from pipeline.data_pipeline import build_darts_series, build_inference_frame, build_test_frame, build_training_frame
from pipeline.inference import generate_forecast
from pipeline.monitoring import (
    build_monitoring_summary,
    save_holdout_city_html_report,
    save_visual_artifacts,
)
from pipeline.training import check_quality_gates, evaluate_holdout, run_backtest, train_baselines, train_tft

PIPELINE_STAGE_LABELS = (
    "Load raw data",
    "Build weekly frames",
    "Build Darts series",
    "Train baselines",
    "Train TFT",
    "Run backtest",
    "Evaluate holdout",
    "Generate forecast",
    "Build monitoring artifacts",
)


@dataclass
class ConsoleStageProgressReporter:
    total_stages: int
    stream: TextIO | None = None
    bar_width: int = 12

    def report(self, stage_index: int, total_stages: int, label: str) -> None:
        completed = max(0, min(stage_index, total_stages))
        filled = int(round(self.bar_width * completed / max(total_stages, 1)))
        bar = "#" * filled + "-" * (self.bar_width - filled)
        output = self.stream if self.stream is not None else sys.stdout
        print(f"[{bar}] {completed}/{total_stages} {label}", file=output, flush=True)


def _default_progress_reporter() -> Callable[[int, int, str], None]:
    return ConsoleStageProgressReporter(total_stages=len(PIPELINE_STAGE_LABELS)).report


def _report_stage(progress_reporter: Callable[[int, int, str], None], stage_index: int) -> None:
    progress_reporter(stage_index, len(PIPELINE_STAGE_LABELS), PIPELINE_STAGE_LABELS[stage_index - 1])


def run_pipeline(config: dict, progress_reporter: Callable[[int, int, str], None] | None = None) -> int:
    """Запустить полный недельный рантайм и сохранить прогноз, holdout и артефакты мониторинга."""
    start = time.perf_counter()
    work_dir = Path(config["paths"]["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    reporter = progress_reporter or _default_progress_reporter()

    _report_stage(reporter, 1)
    raw_df = pd.read_csv(config["paths"]["raw_input"], sep=config["data"]["delimiter"])

    _report_stage(reporter, 2)
    training_frame = build_training_frame(raw_df, config)
    test_frame = build_test_frame(raw_df, config)
    inference_frame = build_inference_frame(raw_df, config, forecast_week=training_frame["week_start"].max())
    forecast_inference_frame = build_inference_frame(raw_df, config)
    training_frame.to_csv(config["paths"]["normalized_output"], index=False)

    _report_stage(reporter, 3)
    training_series_data = build_darts_series(inference_frame, config)
    forecast_series_data = build_darts_series(forecast_inference_frame, config)

    _report_stage(reporter, 4)
    baselines = train_baselines(training_series_data, config)

    _report_stage(reporter, 5)
    tft_model = train_tft(training_series_data, config)

    _report_stage(reporter, 6)
    backtest_df = run_backtest(training_series_data, tft_model, baselines, config)
    try:
        check_quality_gates(backtest_df, config)
    except ValueError as exc:
        if "quality gate failed" not in str(exc):
            raise

    _report_stage(reporter, 7)
    holdout_df = evaluate_holdout(test_frame, tft_model, baselines, config)

    _report_stage(reporter, 8)
    forecast_df = generate_forecast(forecast_series_data, tft_model, config)
    forecast_df.to_csv(config["paths"]["forecast_output"], index=False)
    holdout_df.to_csv(work_dir / "holdout_predictions.csv", index=False)

    _report_stage(reporter, 9)
    runtime_seconds = time.perf_counter() - start
    summary = build_monitoring_summary(backtest_df, holdout_df, forecast_df, config, runtime_seconds=runtime_seconds)
    (work_dir / "monitoring_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_holdout_city_html_report(holdout_df, work_dir)
    save_visual_artifacts(holdout_df, work_dir, config, tft_model=tft_model)
    return 0


def main() -> int:
    """Разобрать аргументы CLI, загрузить конфиг и выполнить недельный рантайм."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_pipeline_config(args.config)
    return run_pipeline(config)


if __name__ == "__main__":
    raise SystemExit(main())
