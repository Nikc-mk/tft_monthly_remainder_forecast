# Holdout City HTML Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a self-contained `holdout_city_report.html` artifact that lets users choose a `City` and inspect weekly holdout actuals, predictions, and MAPE for `seasonal_naive`, `rolling_mean`, and `tft`.

**Architecture:** Keep the new behavior inside the monitoring layer so `pipeline/run.py` only orchestrates artifact writing. Reuse `holdout_df` as the single source of truth, serialize grouped city payloads into one HTML document, and leave the existing PNG artifacts unchanged.

**Tech Stack:** Python, pandas, pathlib, json, inline HTML/CSS/JavaScript, unittest

---

## File Map

- Modify: `pipeline/monitoring.py`
  Responsibility: prepare grouped city payloads, render the self-contained HTML string, and save `holdout_city_report.html`.
- Modify: `pipeline/run.py`
  Responsibility: invoke the new HTML artifact writer during the existing monitoring-artifacts stage.
- Modify: `tests/test_monitoring.py`
  Responsibility: lock in HTML artifact behavior, including city payloads and empty-state generation.
- Modify: `tests/test_run.py`
  Responsibility: ensure the pipeline writes the new HTML artifact alongside existing CSV and PNG outputs.

## Task 1: Lock the HTML artifact contract in monitoring tests

**Files:**
- Modify: `tests/test_monitoring.py`
- Test: `tests/test_monitoring.py`

- [ ] **Step 1: Write the failing HTML artifact test**

Add a new test method to `MonitoringTests` that writes the HTML report and asserts that the artifact exists and contains both cities plus all required series labels.

```python
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
```

- [ ] **Step 2: Add the empty-data regression test**

Add a second test method to verify the generator still writes a valid HTML file when `holdout_df` is empty.

```python
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
```

- [ ] **Step 3: Update imports so the new tests can run**

Extend the monitoring test import list.

```python
from pipeline.monitoring import (
    build_monitoring_summary,
    save_holdout_city_html_report,
    save_visual_artifacts,
)
```

- [ ] **Step 4: Run the monitoring tests to verify they fail**

Run: `python -m unittest tests.test_monitoring -v`

Expected: FAIL with `ImportError` or `AttributeError` because `save_holdout_city_html_report` does not exist yet.

- [ ] **Step 5: Commit the red test state**

```bash
git add tests/test_monitoring.py
git commit -m "test: add holdout html artifact coverage"
```

## Task 2: Implement grouped city payloads and the self-contained HTML report

**Files:**
- Modify: `pipeline/monitoring.py`
- Test: `tests/test_monitoring.py`

- [ ] **Step 1: Add a helper to reshape holdout rows into city-keyed chart payloads**

Add a focused helper near the HTML code path so the report has one consistent data structure.

```python
def _build_holdout_city_report_payload(holdout_df: pd.DataFrame) -> dict[str, dict[str, object]]:
    if holdout_df.empty:
        return {}

    ordered = holdout_df.copy()
    ordered["target_week"] = pd.to_datetime(ordered["target_week"])
    ordered = ordered.sort_values(["City", "target_week", "model_name"]).reset_index(drop=True)

    payload: dict[str, dict[str, object]] = {}
    for city, city_frame in ordered.groupby("City", sort=True):
        weeks = [value.strftime("%Y-%m-%d") for value in city_frame["target_week"].drop_duplicates()]
        actual_series = (
            city_frame.loc[:, ["target_week", "actual"]]
            .drop_duplicates()
            .sort_values("target_week")
        )
        predictions = {}
        mapes = {}
        for model_name, model_frame in city_frame.groupby("model_name", sort=True):
            model_frame = model_frame.sort_values("target_week")
            predictions[model_name] = [float(value) for value in model_frame["predict"]]
            mapes[model_name] = [float(value) * 100.0 for value in model_frame["mape"]]
        payload[city] = {
            "weeks": weeks,
            "actual": [float(value) for value in actual_series["actual"]],
            "predictions": predictions,
            "mape": mapes,
        }
    return payload
```

- [ ] **Step 2: Add the HTML renderer and file writer**

Add a public helper that writes the self-contained HTML artifact and returns its path.

```python
def save_holdout_city_html_report(holdout_df: pd.DataFrame, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "holdout_city_report.html"
    city_payload = _build_holdout_city_report_payload(holdout_df)
    html = _render_holdout_city_html_report(city_payload)
    report_path.write_text(html, encoding="utf-8")
    return report_path
```

- [ ] **Step 3: Render the report as inline HTML, CSS, and JavaScript**

Implement `_render_holdout_city_html_report(...)` in `pipeline/monitoring.py` so it:
- injects `json.dumps(city_payload, ensure_ascii=True)`
- renders a `<select>` with all cities
- shows an empty-state message when there are no cities
- draws two inline SVG charts: one for `actual` plus model predictions, one for per-model `MAPE`

Use a compact structure like this:

```python
def _render_holdout_city_html_report(city_payload: dict[str, dict[str, object]]) -> str:
    payload_json = json.dumps(city_payload, ensure_ascii=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>holdout_city_report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }}
    .chart {{ width: 100%; height: 320px; border: 1px solid #d1d5db; margin-top: 16px; }}
    .controls {{ margin-bottom: 16px; }}
    .empty {{ padding: 24px; border: 1px dashed #9ca3af; }}
  </style>
</head>
<body>
  <h1>Holdout city report</h1>
  <div class="controls">
    <label for="city-select">City</label>
    <select id="city-select"></select>
  </div>
  <div id="empty-state" class="empty" hidden>No holdout data available</div>
  <svg id="prediction-chart" class="chart" viewBox="0 0 960 320"></svg>
  <svg id="mape-chart" class="chart" viewBox="0 0 960 320"></svg>
  <script>
    const CITY_PAYLOAD = {payload_json};
    const MODEL_ORDER = ["seasonal_naive", "rolling_mean", "tft"];
    const COLORS = {{
      actual: "#111827",
      seasonal_naive: "#2563eb",
      rolling_mean: "#d97706",
      tft: "#059669",
    }};

    function render() {{
      const cities = Object.keys(CITY_PAYLOAD);
      const select = document.getElementById("city-select");
      const empty = document.getElementById("empty-state");
      const predictionChart = document.getElementById("prediction-chart");
      const mapeChart = document.getElementById("mape-chart");
      if (cities.length === 0) {{
        empty.hidden = false;
        predictionChart.innerHTML = "";
        mapeChart.innerHTML = "";
        return;
      }}
      if (select.options.length === 0) {{
        cities.forEach((city) => select.add(new Option(city, city)));
        select.value = cities[0];
        select.addEventListener("change", () => drawCity(select.value));
      }}
      empty.hidden = true;
      drawCity(select.value);
    }}
  </script>
</body>
</html>
"""
```

- [ ] **Step 4: Keep the charts deterministic and offline-friendly**

Within the inline JavaScript:
- hardcode the model order as `["seasonal_naive", "rolling_mean", "tft"]`
- draw `actual` only on the top chart
- format MAPE values as percentages
- avoid any external library usage

Use simple polyline generation so the artifact remains readable and maintainable.

```javascript
const MODEL_ORDER = ["seasonal_naive", "rolling_mean", "tft"];
const COLORS = {
  actual: "#111827",
  seasonal_naive: "#2563eb",
  rolling_mean: "#d97706",
  tft: "#059669",
};

function drawCity(city) {
  const cityData = CITY_PAYLOAD[city];
  drawLineChart("prediction-chart", cityData.weeks, {
    actual: cityData.actual,
    seasonal_naive: cityData.predictions.seasonal_naive || [],
    rolling_mean: cityData.predictions.rolling_mean || [],
    tft: cityData.predictions.tft || [],
  }, false);
  drawLineChart("mape-chart", cityData.weeks, cityData.mape, true);
}
```

- [ ] **Step 5: Run the monitoring tests to verify they pass**

Run: `python -m unittest tests.test_monitoring -v`

Expected: PASS, including the two new HTML artifact tests and the existing PNG artifact test.

- [ ] **Step 6: Commit the monitoring implementation**

```bash
git add pipeline/monitoring.py tests/test_monitoring.py
git commit -m "feat: add holdout city html artifact"
```

## Task 3: Wire the HTML artifact into the pipeline runtime

**Files:**
- Modify: `pipeline/run.py`
- Modify: `tests/test_run.py`
- Test: `tests/test_run.py`

- [ ] **Step 1: Extend the runtime artifact test**

In `RunPipelineTests.test_run_pipeline_writes_expected_artifacts`, add one more assertion for the HTML report.

```python
        self.assertTrue((work_dir / "holdout_city_report.html").exists())
```

- [ ] **Step 2: Extend the orchestration mock test**

Patch the new helper in `tests/test_run.py` and assert it is called once with the holdout frame and the work directory.

```python
    @patch("pipeline.run.save_holdout_city_html_report")
    @patch("pipeline.run.save_visual_artifacts")
    @patch("pipeline.run.build_monitoring_summary", return_value={"quality_gate_passed": True})
```

Then add the new mock parameter and assertion:

```python
        _mock_save_holdout_city_html_report,
        _mock_save_visual_artifacts,
    ):
        ...
        _mock_save_holdout_city_html_report.assert_called_once()
```

- [ ] **Step 3: Run the runtime tests to verify they fail**

Run: `python -m unittest tests.test_run -v`

Expected: FAIL because `pipeline.run` does not import or call `save_holdout_city_html_report` yet.

- [ ] **Step 4: Import and invoke the new artifact writer in `pipeline/run.py`**

Update the monitoring import and call the helper during stage 9 after `monitoring_summary.json` is written.

```python
from pipeline.monitoring import (
    build_monitoring_summary,
    save_holdout_city_html_report,
    save_visual_artifacts,
)
```

```python
    summary = build_monitoring_summary(backtest_df, holdout_df, forecast_df, config, runtime_seconds=runtime_seconds)
    (work_dir / "monitoring_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_holdout_city_html_report(holdout_df, work_dir)
    save_visual_artifacts(holdout_df, work_dir, config, tft_model=tft_model)
```

- [ ] **Step 5: Run the runtime tests to verify they pass**

Run: `python -m unittest tests.test_run -v`

Expected: PASS, with both artifact existence and orchestration assertions succeeding.

- [ ] **Step 6: Commit the runtime integration**

```bash
git add pipeline/run.py tests/test_run.py
git commit -m "feat: wire holdout html report into runtime"
```

## Task 4: Final verification

**Files:**
- Test: `tests/test_monitoring.py`
- Test: `tests/test_run.py`

- [ ] **Step 1: Run the focused verification suite**

Run: `python -m unittest tests.test_monitoring tests.test_run -v`

Expected: PASS

- [ ] **Step 2: Smoke-check the generated artifact names**

Run: `python -m pipeline.run --config configs/pipeline.yaml`

Expected: exit code `0` and a work directory that includes `holdout_city_report.html` alongside `holdout_predictions.csv`, `monitoring_summary.json`, and the existing PNG files.

- [ ] **Step 3: Review the produced HTML artifact locally from disk**

Open `artifacts/full_run/holdout_city_report.html` from the filesystem and confirm:
- the city selector is populated
- the first chart shows `actual`, `seasonal_naive`, `rolling_mean`, and `tft`
- the second chart shows weekly MAPE for the three models
- switching the city updates both charts without errors

- [ ] **Step 4: Commit the verification-safe final state**

```bash
git status --short
```

Expected: only the intended files for the HTML artifact work remain modified or committed.
