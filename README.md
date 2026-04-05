# Weekly City Forecasting

Source of truth:
- [docs/project-spec.md](/d:/VS_project/tft_monthly_remainder_forecast/docs/project-spec.md)
- [docs/technical-assignment.md](/d:/VS_project/tft_monthly_remainder_forecast/docs/technical-assignment.md)

## Current Contract

This repository implements a weekly `City` forecasting contract with a fixed `8`-week horizon.

Forecast output fields:
- `forecast_week`
- `target_week`
- `horizon_week`
- `City`
- `q0.1`
- `q0.5`
- `q0.9`
- `model_version`
- `feature_version`
- `generated_at`

## Input Contract

The runtime starts from weekly input rows with these columns:
- `Week`
- `City`
- `revenue`

`Week` is the ISO week start date, Monday. The input `revenue` is already the weekly sales total for that week. During normalization, the prepared feature layer uses `weekly_revenue` as the target column. Negative revenue values are valid.

## MVP Repository Layout

The canonical runtime package is `pipeline/`.

```text
project/
|- configs/
|  `- pipeline.yaml
|- pipeline/
|  |- run.py
|  |- data_pipeline.py
|  |- training.py
|  |- inference.py
|  `- monitoring.py
|- notebooks/
|  |- README.md
|  |- darts_tft.ipynb
|  |- darts_tft_clean.ipynb
|  `- tft_darts_pipeline.ipynb
|- docs/
|- tests/
`- README.md
```

`notebooks/` is research only. It must not contain runtime logic, and it is not part of the production entrypoint.

## Canonical Runtime Entry Point

The canonical config is `configs/pipeline.yaml`.

Run the MVP pipeline with:

```powershell
python -m pipeline.run --config configs/pipeline.yaml
```

## Run Artifacts

The runtime writes these artifacts into the configured work directory:

- `forecast.csv`
- `normalized_dataset.csv`
- `holdout_predictions.csv`
- `monitoring_summary.json`
- `holdout_mape_by_model.png`
- `holdout_mape_by_city.png`
- `holdout_mape_by_week.png`
- `tft_residuals_analysis.png`

The final `8` fully closed weeks are reserved as a post-validation holdout. They do not participate in training or rolling validation, and the holdout evaluation writes `MAPE` visual artifacts alongside the forecast outputs.

## Verification

Run the repository language test with:

```powershell
python -m unittest tests.test_repository_language -v
```
