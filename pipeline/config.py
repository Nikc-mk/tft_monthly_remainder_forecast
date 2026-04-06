"""Загрузка и проверка конфигурации недельного рантайма по городам."""

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


def _normalize_training_device_config(config: dict) -> None:
    training_config = config.setdefault("training", {})
    training_config["accelerator"] = str(training_config.get("accelerator", "auto")).lower()
    try:
        training_config["devices"] = int(training_config.get("devices", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError("training.devices must be a positive integer") from exc


def validate_pipeline_config(config: dict) -> None:
    """Проверить набор инвариантов конфигурации, обязательных для MVP-рантайма."""
    missing = REQUIRED_TOP_LEVEL_KEYS.difference(config)
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")

    requested_accelerator = str(config["training"].get("accelerator", "auto")).lower()
    requested_devices = int(config["training"].get("devices", 1))
    if requested_accelerator not in {"auto", "cpu", "gpu"}:
        raise ValueError("training.accelerator must be one of: auto, cpu, gpu")
    if requested_devices < 1:
        raise ValueError("training.devices must be a positive integer")
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
    """Загрузить YAML-конфигурацию пайплайна с диска и проверить обязательные инварианты."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    _normalize_training_device_config(config)
    validate_pipeline_config(config)
    return config
