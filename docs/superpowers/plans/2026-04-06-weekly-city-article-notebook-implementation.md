# Weekly City Article Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a new research notebook that adapts the Darts TFT article to `data/sales.csv` with minimal structural changes while staying aligned to the repository's weekly `City` forecasting contract.

**Architecture:** The implementation stays notebook-first and research-only. The new notebook keeps the article's narrative order and core Darts APIs, but swaps the incompatible daily, multivariate, and external-forecast assumptions for the project's weekly `City`, `revenue`, `past_covariates`, and calendar-based `future_covariates`. Validation focuses on notebook structure, required code fragments, and JSON smoke checks instead of turning the notebook into runtime code.

**Tech Stack:** Python 3.12, `pandas`, `numpy`, `matplotlib`, `darts`, `torch`, `scikit-learn`, `nbformat`, `unittest`

---

## Documentation Rule

- Before writing or changing any Darts-specific notebook cell, check the current API in Context7.
- Treat Context7 as the default source for `darts` APIs such as `TimeSeries.from_group_dataframe()`, `Scaler`, `StaticCovariatesTransformer`, `TFTModel`, `QuantileRegression`, and `TFTExplainer`.
- Do not rely on memory for TFT explainability, probabilistic prediction, or grouped `TimeSeries` behavior when exact call signatures matter.

## File Map

### Notebook files

- Create: `notebooks/darts_tft_weekly_city_article.ipynb`
  Responsibility: article-style weekly Darts TFT walkthrough on `data/sales.csv`.
- Modify: `notebooks/README.md`
  Responsibility: mention the new notebook as a research notebook and clarify its scope.

### Validation files

- Create: `tests/test_article_notebook.py`
  Responsibility: verify the notebook exists, loads as valid JSON, carries required headings, and includes contract-critical code fragments.

## Task 1: Scaffold the Notebook Artifact and Structure Checks

**Files:**
- Create: `notebooks/darts_tft_weekly_city_article.ipynb`
- Create: `tests/test_article_notebook.py`

- [ ] **Step 1: Write the failing notebook structure test**

```python
import json
import unittest
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/darts_tft_weekly_city_article.ipynb")


class ArticleNotebookStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.sources = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
        )

    def test_notebook_has_python_metadata(self):
        kernelspec = self.notebook.get("metadata", {}).get("kernelspec", {})
        self.assertEqual(kernelspec.get("name"), "python3")

    def test_notebook_has_required_article_sections(self):
        required_headings = [
            "# TFT - Weekly Revenue Forecast (8-week horizon)",
            "## 1. Загрузка и подготовка данных",
            "## 2. Формирование датафреймов target и covariates",
            "## 3. Формирование TimeSeries",
            "## 4. Нормализация",
            "## 5. Разделение на train / val / test",
            "## 6. Создание и обучение TFT",
            "## 7. Выполнение прогнозов",
            "## 8. Оценка качества и визуализация",
            "## 9. Интерпретация результатов",
        ]
        for heading in required_headings:
            self.assertIn(heading, self.sources)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_article_notebook -v`

Expected: ERROR with `FileNotFoundError` for `notebooks/darts_tft_weekly_city_article.ipynb`

- [ ] **Step 3: Create the notebook skeleton with article-style sections**

```python
from pathlib import Path

import nbformat as nbf


notebook = nbf.v4.new_notebook(
    metadata={
        "kernelspec": {
            "display_name": ".venv (3.12.10)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12.10",
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
        },
    }
)

notebook.cells = [
    nbf.v4.new_markdown_cell("# TFT - Weekly Revenue Forecast (8-week horizon)"),
    nbf.v4.new_markdown_cell(
        "Ноутбук повторяет структуру статьи и минимально адаптирует её под weekly `City`-уровень проекта."
    ),
    nbf.v4.new_markdown_cell("## 1. Загрузка и подготовка данных"),
    nbf.v4.new_code_cell("# заполним на следующем шаге"),
    nbf.v4.new_markdown_cell("## 2. Формирование датафреймов target и covariates"),
    nbf.v4.new_code_cell("# заполним на следующем шаге"),
    nbf.v4.new_markdown_cell("## 3. Формирование TimeSeries"),
    nbf.v4.new_code_cell("# заполним на следующем шаге"),
    nbf.v4.new_markdown_cell("## 4. Нормализация"),
    nbf.v4.new_code_cell("# заполним на следующем шаге"),
    nbf.v4.new_markdown_cell("## 5. Разделение на train / val / test"),
    nbf.v4.new_code_cell("# заполним на следующем шаге"),
    nbf.v4.new_markdown_cell("## 6. Создание и обучение TFT"),
    nbf.v4.new_code_cell("# заполним на следующем шаге"),
    nbf.v4.new_markdown_cell("## 7. Выполнение прогнозов"),
    nbf.v4.new_code_cell("# заполним на следующем шаге"),
    nbf.v4.new_markdown_cell("## 8. Оценка качества и визуализация"),
    nbf.v4.new_code_cell("# заполним на следующем шаге"),
    nbf.v4.new_markdown_cell("## 9. Интерпретация результатов"),
    nbf.v4.new_code_cell("# заполним на следующем шаге"),
]

Path("notebooks/darts_tft_weekly_city_article.ipynb").write_text(
    nbf.writes(notebook),
    encoding="utf-8",
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_article_notebook -v`

Expected: PASS with `Ran 2 tests`

- [ ] **Step 5: Commit**

```bash
git add notebooks/darts_tft_weekly_city_article.ipynb tests/test_article_notebook.py
git commit -m "feat: scaffold weekly city article notebook"
```

## Task 2: Add Weekly Data Preparation and Covariate Construction Cells

**Files:**
- Modify: `notebooks/darts_tft_weekly_city_article.ipynb`
- Modify: `tests/test_article_notebook.py`

- [ ] **Step 1: Extend the failing tests for weekly data and feature rules**

```python
    def test_notebook_reads_project_sales_file(self):
        self.assertIn('pd.read_csv("data/sales.csv", sep=";")', self.sources)

    def test_notebook_builds_weekly_grid_and_leakage_safe_features(self):
        required_snippets = [
            'freq="W-MON"',
            'df["date"] = pd.to_datetime(df["Week"])',
            'df.groupby(["City", "date"], as_index=False)["revenue"].sum()',
            'shifted = df.groupby("City")["revenue"].shift(1)',
            'df["lag_1"] = shifted',
            'df["rolling_4"] = shifted.groupby(df["City"]).rolling(4).mean()',
            'df["rolling_12"] = shifted.groupby(df["City"]).rolling(12).mean()',
        ]
        for snippet in required_snippets:
            self.assertIn(snippet, self.sources)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_article_notebook -v`

Expected: FAIL because the placeholder notebook does not yet contain the required data-prep code

- [ ] **Step 3: Replace the placeholder cells with the actual article-style data preparation**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List

from darts import TimeSeries
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler, StaticCovariatesTransformer
from darts.utils.likelihood_models import QuantileRegression
from darts.explainability import TFTExplainer

import torch
import torchmetrics

from sklearn.preprocessing import MinMaxScaler, StandardScaler, OrdinalEncoder
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
```

```python
df = pd.read_csv("data/sales.csv", sep=";")
df["date"] = pd.to_datetime(df["Week"])
df = (
    df.groupby(["City", "date"], as_index=False)["revenue"]
    .sum()
    .sort_values(["City", "date"])
    .reset_index(drop=True)
)

city_frames = []
for city, city_frame in df.groupby("City", sort=True):
    full_weeks = pd.date_range(city_frame["date"].min(), city_frame["date"].max(), freq="W-MON")
    city_grid = pd.DataFrame({"date": full_weeks})
    city_grid["City"] = city
    city_grid = city_grid.merge(city_frame, on=["City", "date"], how="left")
    city_grid["is_missing"] = city_grid["revenue"].isna().astype(int)
    city_grid["revenue"] = city_grid["revenue"].fillna(0.0)
    city_frames.append(city_grid)

df = pd.concat(city_frames, ignore_index=True).sort_values(["City", "date"]).reset_index(drop=True)
```

```python
iso_calendar = df["date"].dt.isocalendar()
df["week_of_year"] = iso_calendar.week.astype(int)
df["month"] = df["date"].dt.month
df["quarter"] = df["date"].dt.quarter
df["year"] = df["date"].dt.year
df["is_holiday_week"] = 0

shifted = df.groupby("City")["revenue"].shift(1)
df["lag_1"] = shifted
df["lag_2"] = df.groupby("City")["revenue"].shift(2)
df["lag_4"] = df.groupby("City")["revenue"].shift(4)
df["lag_8"] = df.groupby("City")["revenue"].shift(8)
df["rolling_4"] = shifted.groupby(df["City"]).rolling(4).mean().reset_index(level=0, drop=True)
df["rolling_8"] = shifted.groupby(df["City"]).rolling(8).mean().reset_index(level=0, drop=True)
df["rolling_12"] = shifted.groupby(df["City"]).rolling(12).mean().reset_index(level=0, drop=True)

df_target = df[["date", "City", "revenue"]].copy()
df_past_covs = df[
    ["date", "City", "lag_1", "lag_2", "lag_4", "lag_8", "rolling_4", "rolling_8", "rolling_12"]
].copy()
df_future_covs = df[["date", "City", "week_of_year", "month", "quarter", "year", "is_holiday_week"]].copy()
```

```python
display(df_target.head())
display(df_past_covs.head())
display(df_future_covs.head())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_article_notebook -v`

Expected: PASS with `Ran 4 tests`

- [ ] **Step 5: Commit**

```bash
git add notebooks/darts_tft_weekly_city_article.ipynb tests/test_article_notebook.py
git commit -m "feat: add weekly data preparation to article notebook"
```

## Task 3: Add Darts TimeSeries, Scaling, and Split Cells

**Files:**
- Modify: `notebooks/darts_tft_weekly_city_article.ipynb`
- Modify: `tests/test_article_notebook.py`

- [ ] **Step 1: Extend the failing tests for Darts series construction and split logic**

```python
    def test_notebook_uses_darts_grouped_series_and_scalers(self):
        required_snippets = [
            "TimeSeries.from_group_dataframe(",
            'group_cols="City"',
            'time_col="date"',
            'value_cols="revenue"',
            "Scaler(",
            "StaticCovariatesTransformer(",
            "StandardScaler()",
            "MinMaxScaler(feature_range=(0, 1))",
        ]
        for snippet in required_snippets:
            self.assertIn(snippet, self.sources)

    def test_notebook_uses_weekly_encoder_decoder_lengths(self):
        required_snippets = [
            "input_len = 60",
            "output_len = 8",
            "window_len = input_len + output_len",
            "def series_splitter(series_list: List[TimeSeries]):",
        ]
        for snippet in required_snippets:
            self.assertIn(snippet, self.sources)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_article_notebook -v`

Expected: FAIL because the notebook does not yet define grouped Darts series, scaling, or split logic

- [ ] **Step 3: Add the grouped series, scaling, and split cells**

```python
ts_target_list = TimeSeries.from_group_dataframe(
    df=df_target.assign(City_static=df_target["City"]),
    group_cols="City",
    time_col="date",
    value_cols="revenue",
    static_cols=["City_static"],
)

ts_past_covs_list = TimeSeries.from_group_dataframe(
    df=df_past_covs.fillna(0.0),
    group_cols="City",
    time_col="date",
    value_cols=["lag_1", "lag_2", "lag_4", "lag_8", "rolling_4", "rolling_8", "rolling_12"],
)

ts_future_covs_list = TimeSeries.from_group_dataframe(
    df=df_future_covs,
    group_cols="City",
    time_col="date",
    value_cols=["week_of_year", "month", "quarter", "year", "is_holiday_week"],
)
```

```python
target_series_scaler = Scaler(scaler=StandardScaler(), global_fit=False)
target_static_scaler = StaticCovariatesTransformer(
    transformer_cat=OrdinalEncoder(),
    cols_cat=["City_static"],
)
past_covs_scaler = Scaler(scaler=StandardScaler(), global_fit=False)
future_covs_scaler = Scaler(scaler=MinMaxScaler(feature_range=(0, 1)), global_fit=True)

ts_target_list = target_series_scaler.fit_transform(ts_target_list)
ts_target_list = target_static_scaler.fit_transform(ts_target_list)
ts_past_covs_list = past_covs_scaler.fit_transform(ts_past_covs_list)
ts_future_covs_list = future_covs_scaler.fit_transform(ts_future_covs_list)
```

```python
input_len = 60
output_len = 8
window_len = input_len + output_len

max_dt = df["date"].max()
test_min_dt = max_dt - pd.Timedelta(weeks=output_len - 1)
test_context_min_dt = test_min_dt - pd.Timedelta(weeks=input_len)
val_max_dt = test_min_dt - pd.Timedelta(weeks=1)
val_min_dt = val_max_dt - pd.Timedelta(weeks=window_len - 1)
train_min_dt = df["date"].min()
train_max_dt = val_min_dt - pd.Timedelta(weeks=1)
```

```python
def series_splitter(series_list: List[TimeSeries]):
    train: List[TimeSeries] = []
    val: List[TimeSeries] = []
    test: List[TimeSeries] = []

    for series in tqdm(series_list):
        train_series = series[train_min_dt:train_max_dt]
        val_series = series[val_min_dt:val_max_dt]
        test_series = series[test_context_min_dt:max_dt]

        if len(train_series) >= window_len:
            train.append(train_series)
        if len(val_series) >= window_len:
            val.append(val_series)
        if len(test_series) >= window_len:
            test.append(test_series)

    return train, val, test
```

```python
ts_target_train, ts_target_val, ts_target_test = series_splitter(ts_target_list)
ts_past_covs_train, ts_past_covs_val, ts_past_covs_test = series_splitter(ts_past_covs_list)
ts_future_covs_train, ts_future_covs_val, ts_future_covs_test = series_splitter(ts_future_covs_list)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_article_notebook -v`

Expected: PASS with `Ran 6 tests`

- [ ] **Step 5: Commit**

```bash
git add notebooks/darts_tft_weekly_city_article.ipynb tests/test_article_notebook.py
git commit -m "feat: add darts series scaling and split cells"
```

## Task 4: Add TFT Training, Forecasting, Metrics, and Explainability Cells

**Files:**
- Modify: `notebooks/darts_tft_weekly_city_article.ipynb`
- Modify: `tests/test_article_notebook.py`

- [ ] **Step 1: Extend the failing tests for model, prediction, and explainability**

```python
    def test_notebook_configures_tft_for_project_horizon(self):
        required_snippets = [
            "tft_model = TFTModel(",
            "input_chunk_length=input_len",
            "output_chunk_length=output_len",
            "hidden_size=64",
            "lstm_layers=2",
            "num_attention_heads=4",
            "full_attention=True",
            "hidden_continuous_size=16",
            "QuantileRegression(",
            "quantiles=[0.1, 0.5, 0.9]",
        ]
        for snippet in required_snippets:
            self.assertIn(snippet, self.sources)

    def test_notebook_contains_prediction_metrics_and_explainer(self):
        required_snippets = [
            "tft_model.fit(",
            "tft_model.predict(",
            "target_static_scaler.inverse_transform(",
            "target_series_scaler.inverse_transform(",
            "def wape(actual, predicted):",
            "TFTExplainer(",
            "explainer.plot_variable_selection(",
            "explainer.plot_attention(",
        ]
        for snippet in required_snippets:
            self.assertIn(snippet, self.sources)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_article_notebook -v`

Expected: FAIL because the notebook does not yet contain the TFT, prediction, metrics, and explainability cells

- [ ] **Step 3: Add the TFT training and forecasting cells**

```python
static_cat_embedding_sizes = {"City_static": len(df_target["City"].unique())}

encoders = {
    "cyclic": {"future": ["week", "month", "quarter"]},
}

tft_model = TFTModel(
    model_name="weekly_city_article_tft",
    input_chunk_length=input_len,
    output_chunk_length=output_len,
    batch_size=64,
    n_epochs=5,
    use_static_covariates=True,
    categorical_embedding_sizes=static_cat_embedding_sizes,
    add_encoders=encoders,
    hidden_size=64,
    lstm_layers=2,
    num_attention_heads=4,
    full_attention=True,
    hidden_continuous_size=16,
    work_dir="./logs",
    save_checkpoints=True,
    log_tensorboard=True,
    show_warnings=True,
    loss_fn=None,
    likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
    torch_metrics=torchmetrics.WeightedMeanAbsolutePercentageError(),
    optimizer_cls=torch.optim.Adam,
    optimizer_kwargs={"lr": 1e-3},
    lr_scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
    lr_scheduler_kwargs={"mode": "min", "factor": 0.5, "patience": 2, "min_lr": 1e-6},
    pl_trainer_kwargs={
        "log_every_n_steps": 10,
        "callbacks": [
            EarlyStopping(monitor="val_loss", patience=3, mode="min"),
            LearningRateMonitor(logging_interval="epoch"),
        ],
    },
)
```

```python
tft_model.fit(
    series=ts_target_train,
    past_covariates=ts_past_covs_train,
    future_covariates=ts_future_covs_train,
    val_series=ts_target_val,
    val_past_covariates=ts_past_covs_val,
    val_future_covariates=ts_future_covs_val,
    verbose=True,
)
```

```python
tft_preds_scaled = tft_model.predict(
    n=output_len,
    series=ts_target_test,
    past_covariates=ts_past_covs_test,
    future_covariates=ts_future_covs_test,
    verbose=True,
    num_samples=200,
)
tft_preds = target_static_scaler.inverse_transform(tft_preds_scaled)
tft_preds = target_series_scaler.inverse_transform(tft_preds)
```

```python
def mape(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denominator = np.where(np.abs(actual) == 0.0, 1e-9, np.abs(actual))
    return np.mean(np.abs(actual - predicted) / denominator)


def wape(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return np.sum(np.abs(actual - predicted)) / max(np.sum(np.abs(actual)), 1e-9)
```

```python
series_idx = 0
actual_test = target_series_scaler.inverse_transform(
    target_static_scaler.inverse_transform(ts_target_test[series_idx])
)
pred_test = tft_preds[series_idx]

plt.figure(figsize=(14, 6))
actual_test.plot(label="actual")
pred_test.plot(label="forecast", low_quantile=0.1, high_quantile=0.9, central_quantile=0.5)
plt.legend()
plt.title("Weekly revenue forecast with quantile interval")
plt.show()
```

```python
actual_values = actual_test[-output_len:].values().reshape(-1)
pred_values = pred_test.quantile_timeseries(0.5).values().reshape(-1)

print("MAPE:", round(float(mape(actual_values, pred_values)), 4))
print("WAPE:", round(float(wape(actual_values, pred_values)), 4))
```

```python
explainer = TFTExplainer(
    tft_model,
    background_series=ts_target_test[0],
    background_past_covariates=ts_past_covs_test[0],
    background_future_covariates=ts_future_covs_test[0],
)
explainability_result = explainer.explain()
explainer.plot_variable_selection(explainability_result)
explainer.plot_attention(explainability_result, plot_type="all")
explainer.plot_attention(explainability_result, plot_type="time")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_article_notebook -v`

Expected: PASS with `Ran 8 tests`

- [ ] **Step 5: Commit**

```bash
git add notebooks/darts_tft_weekly_city_article.ipynb tests/test_article_notebook.py
git commit -m "feat: add weekly tft training and explainability cells"
```

## Task 5: Document the Notebook and Run Final Verification

**Files:**
- Modify: `notebooks/README.md`
- Modify: `tests/test_article_notebook.py`

- [ ] **Step 1: Extend the failing tests for notebook documentation**

```python
    def test_notebook_readme_mentions_new_article_adaptation(self):
        readme_text = Path("notebooks/README.md").read_text(encoding="utf-8")
        self.assertIn("darts_tft_weekly_city_article.ipynb", readme_text)
        self.assertIn("research only", readme_text.lower())
```

Add `from pathlib import Path` at the top of `tests/test_article_notebook.py` if it is not already imported.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_article_notebook -v`

Expected: FAIL because `notebooks/README.md` does not yet mention the new notebook

- [ ] **Step 3: Update the notebook README**

```markdown
# Notebooks Workspace

`notebooks/` is research only.

Use this directory for:
- exploratory data analysis
- model experiments
- one-off validation checks
- documenting research findings

Key notebooks:
- `tft_darts_pipeline.ipynb` - legacy reference notebook with older experiment flow
- `darts_tft_weekly_city_article.ipynb` - article-style weekly `City` TFT adaptation for `data/sales.csv`

Repository rules:
- notebooks must not contain runtime logic
- notebooks must not become the canonical batch entrypoint
- shared runtime behavior belongs in `pipeline/`
- the canonical config remains `configs/pipeline.yaml`
- the canonical command remains `python -m pipeline.run --config configs/pipeline.yaml`

If an experiment becomes part of the maintained pipeline, move that logic into `pipeline/` and keep the notebook as supporting research only.
```

- [ ] **Step 4: Run full verification**

Run: `python -m unittest tests.test_article_notebook -v`

Expected: PASS with all notebook structure tests green

Run: `python -c "import json, pathlib; nb=json.loads(pathlib.Path('notebooks/darts_tft_weekly_city_article.ipynb').read_text(encoding='utf-8')); print(len(nb['cells']))"`

Expected: a positive integer cell count printed with exit code `0`

Run: `python -c "import json, pathlib; nb=json.loads(pathlib.Path('notebooks/darts_tft_weekly_city_article.ipynb').read_text(encoding='utf-8')); src='\\n'.join(''.join(c.get('source', [])) for c in nb['cells']); assert 'pd.read_csv(\"data/sales.csv\", sep=\";\")' in src; assert 'input_chunk_length=input_len' in src; assert 'TFTExplainer(' in src; print('notebook smoke check passed')"`

Expected: `notebook smoke check passed`

- [ ] **Step 5: Commit**

```bash
git add notebooks/README.md tests/test_article_notebook.py
git commit -m "docs: document weekly city article notebook"
```

## Self-Review

### Spec coverage

- New notebook creation in `notebooks/` is covered by Tasks 1 through 4.
- Article-style section order is covered by Task 1.
- Weekly `City` input and `data/sales.csv` parsing are covered by Task 2.
- Both past covariates and future covariates are covered by Task 2 and Task 3.
- `shift(1)` leakage-safe lag and rolling features are covered by Task 2.
- `input_chunk_length = 60` and `output_chunk_length = 8` are covered by Task 3 and Task 4.
- Darts `TFTModel`, prediction, inverse transform, plotting, metrics, and `TFTExplainer` are covered by Task 4.
- Research-only notebook ownership and README clarification are covered by Task 5.

### Placeholder scan

- All tasks use exact file paths.
- Each implementation step contains concrete code or exact markdown.
- Every verification step includes an exact command and expected outcome.
- No `TODO`, `TBD`, or deferred placeholders remain.

### Type consistency

- The new notebook path is consistently `notebooks/darts_tft_weekly_city_article.ipynb`.
- The dataframe names remain consistent as `df_target`, `df_past_covs`, and `df_future_covs`.
- The Darts series names remain consistent as `ts_target_list`, `ts_past_covs_list`, and `ts_future_covs_list`.
- The encoder and decoder lengths remain consistent as `60` and `8`.
