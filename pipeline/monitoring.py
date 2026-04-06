"""Сводки мониторинга и визуальные артефакты для недельного рантайма прогнозирования."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd
from matplotlib import pyplot as plt
from darts import TimeSeries
from darts.utils.statistics import plot_residuals_analysis

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - environment dependent
    sns = None


REPORT_MODEL_ORDER = ("seasonal_naive", "rolling_mean", "tft")


def build_monitoring_summary(
    backtest_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    config: dict,
    runtime_seconds: float | None = None,
) -> dict:
    """Собрать компактную сводку мониторинга по quality gates, holdout MAPE и статусу SLA."""
    best_baseline = (
        backtest_df.loc[backtest_df["model_name"] != "tft"]
        .groupby("model_name")["smape"]
        .mean()
        .min()
    )
    tft_smape = backtest_df.loc[backtest_df["model_name"] == "tft", "smape"].mean()
    improvement = 0.0 if best_baseline == 0.0 else (best_baseline - tft_smape) / best_baseline
    return {
        "quality_gate_passed": bool(
            tft_smape <= config["quality_gates"]["smape_max"]
            and improvement >= config["quality_gates"]["min_improvement_vs_best_baseline"]
        ),
        "tft_backtest_smape": float(tft_smape),
        "best_baseline_smape": float(best_baseline),
        "holdout_mape_by_model": holdout_df.groupby("model_name")["mape"].mean().to_dict(),
        "forecast_rows": int(len(forecast_df)),
        "sla_passed": runtime_seconds is None or runtime_seconds <= config["inference"]["sla_seconds_cpu"],
    }


def _plot_bar(data: pd.DataFrame, x: str, y: str, output_path: Path, hue: str | None = None) -> None:
    plt.figure(figsize=(10, 5))
    if sns is not None:
        sns.barplot(data=data, x=x, y=y, hue=hue, estimator="mean", errorbar=None)
    else:
        if hue is None:
            summary = data.groupby(x, as_index=False)[y].mean()
            plt.bar(summary[x], summary[y])
        else:
            pivot = data.pivot(index=x, columns=hue, values=y).fillna(0.0)
            positions = range(len(pivot.index))
            width = 0.8 / max(len(pivot.columns), 1)
            for idx, column in enumerate(pivot.columns):
                offsets = [pos + (idx * width) - (width * (len(pivot.columns) - 1) / 2) for pos in positions]
                plt.bar(offsets, pivot[column], width=width, label=str(column))
            plt.xticks(list(positions), pivot.index)
            plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_line(data: pd.DataFrame, x: str, y: str, hue: str, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    ordered = data.sort_values([x, hue]).copy()
    ordered[x] = pd.to_datetime(ordered[x])
    if sns is not None:
        sns.lineplot(data=ordered, x=x, y=y, hue=hue, marker="o")
    else:
        for label, group in ordered.groupby(hue):
            plt.plot(group[x], group[y], marker="o", label=str(label))
        plt.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _build_aggregate_tft_residual_series(tft_model: object) -> TimeSeries | None:
    training_series_by_city = getattr(tft_model, "training_series_by_city", None)
    training_past_covariates_by_city = getattr(tft_model, "training_past_covariates_by_city", None)
    training_future_covariates_by_city = getattr(tft_model, "training_future_covariates_by_city", None)
    if not training_series_by_city or not training_future_covariates_by_city:
        return None

    residual_kwargs = {
        "series": list(training_series_by_city.values()),
        "future_covariates": list(training_future_covariates_by_city.values()),
        "forecast_horizon": 1,
        "retrain": False,
        "last_points_only": True,
        "verbose": False,
        "show_warnings": False,
    }
    if training_past_covariates_by_city:
        residual_kwargs["past_covariates"] = list(training_past_covariates_by_city.values())

    residual_series = tft_model.residuals(**residual_kwargs)
    if not residual_series:
        return None

    residual_frames = []
    for series in residual_series:
        residual_frames.append(
            pd.DataFrame(
                {
                    "week_start": pd.to_datetime(series.time_index),
                    "residual": series.values(copy=False).reshape(-1).astype(float),
                }
            )
        )

    aggregate = (
        pd.concat(residual_frames, ignore_index=True)
        .groupby("week_start", as_index=False)["residual"]
        .sum()
        .sort_values("week_start")
        .reset_index(drop=True)
    )
    if len(aggregate) < 2:
        return None

    return TimeSeries.from_times_and_values(
        times=pd.DatetimeIndex(aggregate["week_start"]),
        values=aggregate["residual"].to_numpy(),
        freq="W-MON",
    )


def _save_tft_residual_analysis_artifact(tft_model: object, output_path: Path) -> Path | None:
    residual_series = _build_aggregate_tft_residual_series(tft_model)
    if residual_series is None:
        return None

    acf_max_lag = max(1, min(24, len(residual_series) - 1))
    plt.close("all")
    plot_residuals_analysis(residual_series, acf_max_lag=acf_max_lag)
    figure = plt.gcf()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def _build_holdout_city_report_payload(holdout_df: pd.DataFrame) -> dict[str, dict[str, object]]:
    if holdout_df.empty:
        return {}

    ordered = holdout_df.copy()
    ordered["target_week"] = pd.to_datetime(ordered["target_week"])
    ordered = ordered.sort_values(["City", "target_week", "model_name"]).reset_index(drop=True)

    payload: dict[str, dict[str, object]] = {}
    for city, city_frame in ordered.groupby("City", sort=True):
        week_index = (
            city_frame.loc[:, ["target_week"]]
            .drop_duplicates()
            .sort_values("target_week")
            .reset_index(drop=True)
        )
        weeks = [value.strftime("%Y-%m-%d") for value in week_index["target_week"]]
        actual_by_week = (
            city_frame.loc[:, ["target_week", "actual"]]
            .drop_duplicates(subset=["target_week"])
            .set_index("target_week")["actual"]
        )
        predictions: dict[str, list[float | None]] = {}
        mapes: dict[str, list[float | None]] = {}
        for model_name in REPORT_MODEL_ORDER:
            model_frame = city_frame.loc[city_frame["model_name"] == model_name, ["target_week", "predict", "mape"]]
            prediction_by_week = model_frame.set_index("target_week")["predict"] if not model_frame.empty else pd.Series(dtype=float)
            mape_by_week = model_frame.set_index("target_week")["mape"] if not model_frame.empty else pd.Series(dtype=float)
            predictions[model_name] = [
                None if week not in prediction_by_week.index else float(prediction_by_week.loc[week])
                for week in week_index["target_week"]
            ]
            mapes[model_name] = [
                None if week not in mape_by_week.index else float(mape_by_week.loc[week]) * 100.0
                for week in week_index["target_week"]
            ]

        payload[city] = {
            "weeks": weeks,
            "actual": [float(actual_by_week.loc[week]) for week in week_index["target_week"]],
            "predictions": predictions,
            "mape": mapes,
        }
    return payload


def _render_holdout_city_html_report(city_payload: dict[str, dict[str, object]]) -> str:
    payload_json = json.dumps(city_payload, ensure_ascii=True)
    model_order_json = json.dumps(list(REPORT_MODEL_ORDER))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>holdout_city_report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f3eb;
      --panel: #fffdf8;
      --line: #d6cfbf;
      --text: #2f2a22;
      --muted: #6b6458;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background:
        radial-gradient(circle at top left, rgba(213, 184, 127, 0.18), transparent 28%),
        linear-gradient(180deg, #fbf8f2 0%, var(--bg) 100%);
      color: var(--text);
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 20px 40px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 32px;
      font-weight: 700;
    }}
    p {{
      margin: 0;
      color: var(--muted);
      font-size: 16px;
      line-height: 1.5;
    }}
    .controls {{
      margin-top: 24px;
      padding: 16px 18px;
      background: rgba(255, 253, 248, 0.92);
      border: 1px solid var(--line);
      border-radius: 16px;
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
    }}
    label {{
      font-size: 14px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    select {{
      min-width: 220px;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: white;
      font: inherit;
      color: var(--text);
    }}
    .chart-panel {{
      margin-top: 18px;
      padding: 18px;
      background: rgba(255, 253, 248, 0.94);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: 0 10px 30px rgba(55, 45, 27, 0.06);
    }}
    .chart-panel h2 {{
      margin: 0 0 12px;
      font-size: 20px;
    }}
    .chart {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 14px;
      background:
        linear-gradient(180deg, rgba(214, 207, 191, 0.16), rgba(255, 255, 255, 0.9));
    }}
    .empty {{
      margin-top: 18px;
      padding: 28px;
      border-radius: 20px;
      border: 1px dashed var(--line);
      background: rgba(255, 253, 248, 0.94);
      color: var(--muted);
      text-align: center;
    }}
    .legend {{
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      margin-top: 12px;
      color: var(--muted);
      font-size: 14px;
    }}
    .legend span {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}
    .swatch {{
      width: 12px;
      height: 12px;
      border-radius: 999px;
      display: inline-block;
    }}
    @media (max-width: 640px) {{
      main {{
        padding: 24px 14px 28px;
      }}
      h1 {{
        font-size: 28px;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>Holdout city report</h1>
    <p>Weekly holdout comparison for actual revenue, model predictions, and per-week MAPE.</p>

    <div class="controls">
      <label for="city-select">City</label>
      <select id="city-select"></select>
    </div>

    <div id="empty-state" class="empty" hidden>No holdout data available</div>

    <section id="prediction-panel" class="chart-panel" hidden>
      <h2>Actual vs prediction by week</h2>
      <svg id="prediction-chart" class="chart" viewBox="0 0 960 360" aria-label="Holdout prediction chart"></svg>
      <div id="prediction-legend" class="legend"></div>
    </section>

    <section id="mape-panel" class="chart-panel" hidden>
      <h2>MAPE by week</h2>
      <svg id="mape-chart" class="chart" viewBox="0 0 960 360" aria-label="Holdout MAPE chart"></svg>
      <div id="mape-legend" class="legend"></div>
    </section>
  </main>

  <script>
    const CITY_PAYLOAD = {payload_json};
    const MODEL_ORDER = {model_order_json};
    const COLORS = {{
      actual: "#111827",
      seasonal_naive: "#2563eb",
      rolling_mean: "#d97706",
      tft: "#059669",
    }};

    function setLegend(containerId, seriesNames) {{
      const legend = document.getElementById(containerId);
      legend.innerHTML = "";
      seriesNames.forEach((name) => {{
        const item = document.createElement("span");
        const swatch = document.createElement("i");
        swatch.className = "swatch";
        swatch.style.background = COLORS[name];
        item.appendChild(swatch);
        item.appendChild(document.createTextNode(name));
        legend.appendChild(item);
      }});
    }}

    function valueExtent(seriesMap) {{
      const values = [];
      Object.values(seriesMap).forEach((series) => {{
        series.forEach((value) => {{
          if (value !== null && value !== undefined && Number.isFinite(value)) {{
            values.push(value);
          }}
        }});
      }});
      if (values.length === 0) {{
        return [0, 1];
      }}
      const min = Math.min(...values);
      const max = Math.max(...values);
      if (min === max) {{
        const padding = Math.abs(min) > 0 ? Math.abs(min) * 0.1 : 1;
        return [min - padding, max + padding];
      }}
      const padding = (max - min) * 0.12;
      return [min - padding, max + padding];
    }}

    function createSvgNode(name, attrs) {{
      const node = document.createElementNS("http://www.w3.org/2000/svg", name);
      Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
      return node;
    }}

    function drawLineChart(svgId, weeks, seriesMap, percentMode) {{
      const svg = document.getElementById(svgId);
      svg.innerHTML = "";
      const width = 960;
      const height = 360;
      const margin = {{ top: 26, right: 28, bottom: 70, left: 68 }};
      const plotWidth = width - margin.left - margin.right;
      const plotHeight = height - margin.top - margin.bottom;
      const [minValue, maxValue] = valueExtent(seriesMap);

      const xForIndex = (index) => {{
        if (weeks.length <= 1) {{
          return margin.left + plotWidth / 2;
        }}
        return margin.left + (plotWidth * index) / (weeks.length - 1);
      }};

      const yForValue = (value) => {{
        if (maxValue === minValue) {{
          return margin.top + plotHeight / 2;
        }}
        return margin.top + plotHeight - ((value - minValue) / (maxValue - minValue)) * plotHeight;
      }};

      svg.appendChild(createSvgNode("rect", {{
        x: margin.left,
        y: margin.top,
        width: plotWidth,
        height: plotHeight,
        fill: "rgba(255,255,255,0.65)",
        rx: 14,
      }}));

      for (let tick = 0; tick < 5; tick += 1) {{
        const ratio = tick / 4;
        const y = margin.top + plotHeight - ratio * plotHeight;
        const value = minValue + ratio * (maxValue - minValue);
        svg.appendChild(createSvgNode("line", {{
          x1: margin.left,
          y1: y,
          x2: margin.left + plotWidth,
          y2: y,
          stroke: "#d6cfbf",
          "stroke-width": 1,
        }}));
        const label = createSvgNode("text", {{
          x: margin.left - 10,
          y: y + 4,
          "text-anchor": "end",
          fill: "#6b6458",
          "font-size": 12,
        }});
        label.textContent = percentMode ? `${{value.toFixed(1)}}%` : value.toFixed(1);
        svg.appendChild(label);
      }}

      weeks.forEach((week, index) => {{
        const x = xForIndex(index);
        svg.appendChild(createSvgNode("line", {{
          x1: x,
          y1: margin.top,
          x2: x,
          y2: margin.top + plotHeight,
          stroke: "rgba(214, 207, 191, 0.45)",
          "stroke-width": 1,
        }}));
        const label = createSvgNode("text", {{
          x,
          y: height - 24,
          "text-anchor": "middle",
          fill: "#6b6458",
          "font-size": 12,
        }});
        label.textContent = week;
        svg.appendChild(label);
      }});

      Object.entries(seriesMap).forEach(([name, values]) => {{
        const points = values
          .map((value, index) => value === null || value === undefined ? null : `${{xForIndex(index)}},${{yForValue(value)}}`)
          .filter((value) => value !== null);
        if (points.length === 0) {{
          return;
        }}
        svg.appendChild(createSvgNode("polyline", {{
          points: points.join(" "),
          fill: "none",
          stroke: COLORS[name],
          "stroke-width": 3,
          "stroke-linecap": "round",
          "stroke-linejoin": "round",
        }}));
        values.forEach((value, index) => {{
          if (value === null || value === undefined) {{
            return;
          }}
          svg.appendChild(createSvgNode("circle", {{
            cx: xForIndex(index),
            cy: yForValue(value),
            r: 4,
            fill: COLORS[name],
          }}));
        }});
      }});
    }}

    function drawCity(city) {{
      const cityData = CITY_PAYLOAD[city];
      const predictionSeries = {{
        actual: cityData.actual,
        seasonal_naive: cityData.predictions.seasonal_naive || [],
        rolling_mean: cityData.predictions.rolling_mean || [],
        tft: cityData.predictions.tft || [],
      }};
      drawLineChart("prediction-chart", cityData.weeks, predictionSeries, false);
      drawLineChart("mape-chart", cityData.weeks, cityData.mape, true);
      setLegend("prediction-legend", ["actual", ...MODEL_ORDER]);
      setLegend("mape-legend", MODEL_ORDER);
    }}

    function render() {{
      const cities = Object.keys(CITY_PAYLOAD);
      const select = document.getElementById("city-select");
      const empty = document.getElementById("empty-state");
      const predictionPanel = document.getElementById("prediction-panel");
      const mapePanel = document.getElementById("mape-panel");

      if (cities.length === 0) {{
        empty.hidden = false;
        predictionPanel.hidden = true;
        mapePanel.hidden = true;
        return;
      }}

      cities.forEach((city) => select.add(new Option(city, city)));
      select.value = cities[0];
      select.addEventListener("change", () => drawCity(select.value));
      empty.hidden = true;
      predictionPanel.hidden = false;
      mapePanel.hidden = false;
      drawCity(select.value);
    }}

    render();
  </script>
</body>
</html>
"""


def save_holdout_city_html_report(holdout_df: pd.DataFrame, output_dir: str | Path) -> Path:
    """Сохранить автономный HTML-отчет для просмотра недельного holdout по городам."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "holdout_city_report.html"
    city_payload = _build_holdout_city_report_payload(holdout_df)
    report_path.write_text(_render_holdout_city_html_report(city_payload), encoding="utf-8")
    return report_path


def save_visual_artifacts(
    holdout_df: pd.DataFrame,
    output_dir: str | Path,
    config: dict,
    tft_model: object | None = None,
) -> list[Path]:
    """Сохранить PNG-графики holdout MAPE для мониторинга на уровне моделей и городов."""
    del config
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifact_paths: list[Path] = []

    aggregate_path = output_path / "holdout_mape_by_model.png"
    _plot_bar(holdout_df, x="model_name", y="mape", output_path=aggregate_path)
    artifact_paths.append(aggregate_path)

    city_path = output_path / "holdout_mape_by_city.png"
    city_summary = holdout_df.groupby(["City", "model_name"], as_index=False)["mape"].mean()
    _plot_bar(city_summary, x="City", y="mape", hue="model_name", output_path=city_path)
    artifact_paths.append(city_path)

    week_path = output_path / "holdout_mape_by_week.png"
    week_summary = holdout_df.groupby(["target_week", "model_name"], as_index=False)["mape"].mean()
    _plot_line(week_summary, x="target_week", y="mape", hue="model_name", output_path=week_path)
    artifact_paths.append(week_path)

    if tft_model is not None:
        residual_path = _save_tft_residual_analysis_artifact(
            tft_model,
            output_path / "tft_residuals_analysis.png",
        )
        if residual_path is not None:
            artifact_paths.append(residual_path)

    return artifact_paths
