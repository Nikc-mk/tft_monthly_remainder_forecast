# 2026-04-05 Holdout City HTML Report Design

## 1. Objective

Add a new monitoring artifact that saves an interactive HTML holdout report in the runtime artifacts directory.

The report must let a user choose a `City` and inspect weekly holdout behavior without opening separate files for each city.

## 2. Problem Statement

The current runtime already saves:
- `holdout_predictions.csv`
- `holdout_mape_by_model.png`
- `holdout_mape_by_city.png`
- `holdout_mape_by_week.png`

Those outputs are useful for aggregate monitoring, but they do not provide a convenient city-level inspection flow inside a single artifact.

The user needs a local file in `artifacts` that:
- opens without a server
- keeps existing PNG artifacts unchanged
- allows city selection in one page
- shows weekly holdout comparisons for all three models

## 3. Design Decision

Add one self-contained HTML artifact named `holdout_city_report.html` to the runtime work directory.

The report will:
- be generated from `holdout_df` during the normal pipeline run
- embed all required city-level holdout data directly into the HTML as JSON
- require no external JavaScript, no CDN, and no network access
- support choosing a `City` from a dropdown
- update charts in-place when the selected city changes

This report is additive. Existing PNG artifacts remain part of the monitoring output.

## 4. Report Content

For the selected `City`, the HTML report must render two weekly charts.

### 4.1 Weekly Actual vs Prediction Chart

This chart shows four lines across holdout weeks:
- `actual`
- `seasonal_naive`
- `rolling_mean`
- `tft`

The x-axis is `target_week`.

The y-axis is weekly revenue.

### 4.2 Weekly MAPE Chart

This chart shows three lines across holdout weeks:
- `seasonal_naive`
- `rolling_mean`
- `tft`

The x-axis is `target_week`.

The y-axis is weekly `MAPE`, displayed as percent values.

## 5. Data Contract for the HTML Artifact

The HTML report uses the same holdout-level source data already produced by the runtime.

Expected fields from `holdout_df`:
- `City`
- `target_week`
- `model_name`
- `actual`
- `predict`
- `mape`

Before embedding data into the HTML:
- rows must be sorted by `City`, `target_week`, `model_name`
- `target_week` must be serialized in a browser-safe string format
- data should be grouped by `City`

For each city, the embedded payload must contain enough information to draw:
- a single `actual` weekly series
- one predicted weekly series per model
- one weekly `MAPE` series per model

## 6. Runtime Integration

Implementation should stay inside the monitoring/reporting layer.

Recommended integration:
- add a dedicated HTML artifact builder in `pipeline/monitoring.py`
- call it from `pipeline/run.py` during the "Build monitoring artifacts" stage
- save the file into the configured `work_dir`

The runtime output set will therefore include:
- existing PNG monitoring artifacts
- existing CSV artifacts
- new `holdout_city_report.html`

## 7. UI and Interaction Requirements

The page should remain intentionally simple and local-file friendly.

Required UI elements:
- report title
- `City` dropdown selector
- two charts stacked vertically

Interaction behavior:
- the first available city is selected by default
- changing the city updates both charts immediately
- no browser popups, file uploads, or extra navigation are required

## 8. Empty or Minimal Data Behavior

If `holdout_df` is empty, the runtime should still create the HTML file.

In that case, the page should show a clear empty-state message instead of failing during generation or rendering.

If data exists for only one city or one week, the page should still render valid output without special user action.

## 9. Testing Strategy

Tests should verify the artifact generator rather than browser behavior.

Minimum coverage:
- HTML file is created in the output directory
- HTML contains the expected report filename and city payload
- HTML includes all three model names
- HTML generation does not fail on empty holdout data

Integration coverage can additionally verify that the pipeline stage writes the new HTML artifact alongside the existing monitoring files.

## 10. Boundaries and Non-Goals

This design does not:
- replace existing PNG artifacts
- introduce per-city HTML files
- add weekly tables to the report
- change holdout metric definitions
- change training, inference, or quality-gate logic
- require a browser preview step during runtime execution

## 11. Risks and Mitigations

Risk: the HTML may depend on internet-hosted assets and fail offline.
Mitigation: keep the report fully self-contained with inline CSS, JavaScript, and data.

Risk: the file may become harder to maintain if chart logic is mixed into `run.py`.
Mitigation: keep HTML generation isolated in `pipeline/monitoring.py`.

Risk: city-level weekly points may render in the wrong order.
Mitigation: sort by `target_week` before building chart payloads.

## 12. Acceptance Criteria

This design is satisfied when:
- the runtime saves `holdout_city_report.html` in the artifacts work directory
- the report contains a `City` selector in a single page
- the report shows one weekly actual-vs-prediction chart with `actual`, `seasonal_naive`, `rolling_mean`, and `tft`
- the report shows one weekly `MAPE` chart with `seasonal_naive`, `rolling_mean`, and `tft`
- selecting a city updates both charts without reloading external data
- existing PNG monitoring artifacts are still generated
