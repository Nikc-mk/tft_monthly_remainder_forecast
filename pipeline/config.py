"""Configuration loading and validation for the weekly city runtime."""

from __future__ import annotations

from pathlib import Path

import yaml


REQUIRED_TOP_LEVEL_KEYS = {
    "run",
    "paths",
    "data",
    "features",
    "training",
    "quality_gates",
    "inference",
    "metadata",
}


def validate_pipeline_config(config: dict) -> None:
    """Validate the subset of config invariants required by the MVP runtime."""
    missing = REQUIRED_TOP_LEVEL_KEYS.difference(config)
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")

    if config["training"]["max_encoder_length"] != 60:
        raise ValueError("training.max_encoder_length must equal 60")
    if config["training"]["max_prediction_length"] != 8:
        raise ValueError("training.max_prediction_length must equal 8")
    if config["training"]["backtest_windows"] != 6:
        raise ValueError("training.backtest_windows must equal 6")
    if config["data"]["min_history_weeks"] != 60:
        raise ValueError("data.min_history_weeks must equal 60")
    if config["inference"]["quantiles"] != [0.1, 0.5, 0.9]:
        raise ValueError("inference.quantiles must equal [0.1, 0.5, 0.9]")


def load_pipeline_config(path: str | Path) -> dict:
    """Load pipeline YAML config from disk and validate required invariants."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_pipeline_config(config)
    return config
