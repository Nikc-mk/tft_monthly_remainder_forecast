# 2026-04-06 Weekly City Article Notebook Design

## 1. Objective

Create a new research notebook in `notebooks/` that follows the structure and coding style of the provided Darts TFT article as closely as possible, while making only the minimum necessary changes to run on the project's weekly `City`-level dataset in `data/sales.csv`.

The notebook must remain a research artifact only. It must not become a runtime entrypoint and must not replace the canonical pipeline in `pipeline/`.

## 2. Problem Statement

The article describes a richer forecasting setup than the current repository data provides:

- the article uses daily data
- the article uses multivariate targets
- the article relies on future covariates derived from an existing forecasting model
- the article discusses store-level identifiers and multiple static features

The repository contract is different:

- input data is weekly
- the target is single-variable weekly `revenue`
- the grouping level is `City`
- the forecast horizon is fixed at `8` weeks
- the history window is fixed at `60` weeks
- the available raw data in `data/sales.csv` contains only `Week`, `City`, `revenue`

The notebook therefore needs a contract-safe adaptation that stays faithful to the article's flow without inventing unsupported data sources.

## 3. Design Decision

Use an "article-first / contract-safe" adaptation.

This means:

- preserve the article's notebook flow and code style wherever possible
- keep the main Darts abstractions from the article:
  - `TimeSeries.from_group_dataframe(...)`
  - `Scaler(...)`
  - `StaticCovariatesTransformer(...)`
  - `TFTModel(...)`
  - `TFTExplainer(...)`
- adapt only the parts that are incompatible with the repository contract or data

The notebook will not simulate external forecast covariates from another model. Instead, it will use the project's supported weekly covariates:

- future covariates:
  - `week_of_year`
  - `month`
  - `quarter`
  - `year`
  - `is_holiday_week`
- past covariates:
  - `lag_1`
  - `lag_2`
  - `lag_4`
  - `lag_8`
  - `rolling_4`
  - `rolling_8`
  - `rolling_12`

## 4. Notebook Scope

The notebook will live in `notebooks/` as a new file rather than replacing the existing notebook by default.

It will include the following sections, aligned to the article:

1. data loading and weekly preparation
2. creation of target, past covariates, and future covariates data frames
3. creation of Darts `TimeSeries`
4. normalization and static covariate handling
5. train/validation/test split
6. TFT model configuration and training
7. holdout forecasting
8. inverse transforms and visualization
9. holdout metrics
10. TFT explainability

It may include short markdown notes where the article's original assumptions do not apply to this dataset.

## 5. Data Contracts Inside the Notebook

### 5.1 Input

The notebook reads:

- `data/sales.csv`

Expected schema:

- `Week`
- `City`
- `revenue`

The file is parsed with delimiter `;`.

### 5.2 Prepared Weekly Frame

The notebook builds a complete weekly grid per `City` on `W-MON`, fills missing weeks, and works on the normalized weekly columns:

- `date`
- `City`
- `revenue`
- `is_missing`

This keeps the notebook naming close to the article while still reflecting the project's weekly contract.

### 5.3 Derived Frames

`df_target` will contain:

- `date`
- `City`
- `revenue`

plus static covariate support for `City`.

`df_past_covs` will contain:

- `date`
- `City`
- `lag_1`
- `lag_2`
- `lag_4`
- `lag_8`
- `rolling_4`
- `rolling_8`
- `rolling_12`

`df_future_covs` will contain:

- `date`
- `City`
- `week_of_year`
- `month`
- `quarter`
- `year`
- `is_holiday_week`

## 6. Feature Rules

The notebook must respect the repository forecasting contract:

- weekly frequency only
- grouping by `City`
- fixed history window of `60` weeks
- fixed forecast horizon of `8` weeks
- no future-aware features
- every lag or rolling feature based on `revenue` must be built only after `shift(1)`

The feature rules are:

- `lag_1`, `lag_2`, `lag_4`, `lag_8` are created per `City`
- `rolling_4`, `rolling_8`, `rolling_12` are created from `revenue.shift(1).rolling(...)`
- future covariates are calendar-based and may be extended into the future horizon
- past covariates are historical only and must not leak future target information

## 7. TimeSeries Construction

The notebook will stay close to the article by building grouped Darts series with `TimeSeries.from_group_dataframe(...)`.

The series layout is:

- target series:
  - grouped by `City`
  - time column `date`
  - value column `revenue`
- past covariates series:
  - grouped by `City`
  - time column `date`
  - value columns `lag_*` and `rolling_*`
- future covariates series:
  - grouped by `City`
  - time column `date`
  - value columns `week_of_year`, `month`, `quarter`, `year`, `is_holiday_week`

`City` will be used as a static categorical covariate so the notebook can preserve the article's narrative around static features with minimal invention.

## 8. Scaling Strategy

The notebook will keep the article's scaling pattern as closely as practical:

- target series:
  - per-series `StandardScaler`
- past covariates:
  - per-series `StandardScaler`
- future covariates:
  - global `MinMaxScaler(feature_range=(0, 1))`
- static covariates:
  - `StaticCovariatesTransformer(...)`

This is intentionally close to the article, with only the minimum necessary adaptation to the project's actual feature set.

## 9. Split Strategy

The article-style notebook will use a readable train/validation/test split rather than reproducing the full runtime backtest loop.

The split rules are:

- `input_chunk_length = 60`
- `output_chunk_length = 8`
- the final `8` fully observed weeks form the holdout test period
- the validation slice immediately precedes the test period and must be long enough to support windows of length `60 + 8`
- all earlier usable history forms the training slice

The notebook will likely retain a helper similar to the article's `series_splitter(...)`, adapted to weekly date boundaries and to all three series collections:

- targets
- past covariates
- future covariates

The notebook must also explain that the production-like project contract still requires rolling weekly backtesting over `6` windows, even if the notebook itself uses a simpler article-style split for readability.

## 10. TFT Configuration

The TFT model should keep the article's overall configuration style while changing only contract-sensitive parameters.

Parameters preserved where practical:

- `hidden_size = 64`
- `lstm_layers = 2`
- `num_attention_heads = 4`
- `full_attention = True`
- `hidden_continuous_size = 16`

Parameters adapted to the project:

- `input_chunk_length = 60`
- `output_chunk_length = 8`
- quantiles:
  - `0.1`
  - `0.5`
  - `0.9`

Operational settings such as `batch_size` and `n_epochs` may be moderated to keep the notebook runnable on a local developer machine without changing the conceptual structure.

## 11. Forecasting and Evaluation

The notebook will:

- run probabilistic holdout prediction with Darts TFT
- inverse-transform predictions back to original scale
- plot a representative city:
  - recent history
  - actual holdout values
  - median forecast
  - interval from quantiles
- compute holdout metrics in notebook-friendly form

Metrics to include:

- `MAPE`
- `WAPE`

Optional:

- `SMAPE`, if it improves alignment with the repository contract without materially complicating the notebook

## 12. Explainability

The notebook will keep the article's explainability section using `TFTExplainer` when it works reliably on the adapted setup.

Planned outputs:

- variable selection visualization
- attention visualization

If explainability is unstable or too heavy for the notebook's default path, the section may be kept as optional cells with a short note explaining the limitation.

## 13. Boundaries and Non-Goals

This design does not:

- change any runtime behavior in `pipeline/`
- replace the canonical config in `configs/pipeline.yaml`
- replace the canonical runtime command
- introduce synthetic "external forecast" covariates
- reintroduce daily or monthly-remainder logic
- make the notebook the source of truth for the forecasting contract

The notebook is a research adaptation of an article, not a new production interface.

## 14. Risks and Mitigations

Risk: the article's section about covariates from an existing forecast model has no direct equivalent in the project data.
Mitigation: replace that block with the project's supported future and past covariates, and explicitly note the adaptation in markdown.

Risk: the article assumes richer static features and possibly multivariate targets.
Mitigation: keep only `City` as the static categorical feature and use a univariate target `revenue`.

Risk: a simplified notebook split may be confused with the runtime validation contract.
Mitigation: explicitly note that the notebook uses article-style readability, while the runtime contract still requires `6` rolling weekly backtest windows.

Risk: `TFTExplainer` may be slow or brittle in the local notebook setting.
Mitigation: keep the explainability section, but allow it to be optional or guarded with explanatory markdown if necessary.

## 15. Acceptance Criteria

This design is satisfied when:

- a new notebook is created in `notebooks/`
- the notebook reads `data/sales.csv` with the repository delimiter and schema
- the notebook follows the article's structure and coding style as closely as possible
- the notebook uses weekly `City`-level forecasting with target `revenue`
- the notebook includes both past covariates and future covariates
- lag and rolling features are created only after `shift(1)`
- the notebook uses `input_chunk_length = 60` and `output_chunk_length = 8`
- the notebook uses Darts `TFTModel`
- the notebook includes prediction, inverse transform, plotting, and metric cells
- the notebook keeps runtime ownership in `pipeline/` and remains research-only
