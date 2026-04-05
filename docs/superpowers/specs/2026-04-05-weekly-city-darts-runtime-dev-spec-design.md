# 2026-04-05 Weekly City Darts Runtime Dev Spec

## 1. Назначение

Этот документ задает спецификацию на разработку runtime-кода проекта с нуля в пустом `pipeline/`.

Документ опирается на:
- `docs/project-spec.md`
- `docs/technical-assignment.md`

Если реализация, старые заметки или черновики расходятся с этими документами, источником истины остаются `docs/project-spec.md`, `docs/technical-assignment.md` и этот dev-spec.

## 2. Цель разработки

Нужно реализовать простой, читаемый и воспроизводимый MVP runtime для недельного прогнозирования продаж по `City`.

Runtime должен:
- принимать уже агрегированные недельные продажи в формате `Week | City | revenue`
- использовать только последнюю полностью закрытую ISO-неделю как точку отсечения
- строить прямой вероятностный прогноз на `8` недель вперед
- выделять последние полные `8` недель в отдельный финальный `test holdout`
- возвращать результат в табличном контракте проекта
- использовать `darts` как основной runtime-слой для time series, TFT, backtesting и prediction
- оставаться простым набором модулей, а не превращаться в собственный forecasting framework

## 3. Ключевое архитектурное решение

Принять подход `Darts-first orchestration`.

Это означает:
- `pipeline/` строится вокруг небольших и понятных модулей
- `pandas` отвечает за табличные контракты, валидацию данных и feature engineering
- `darts` отвечает за `TimeSeries`, преобразование табличных данных в ряды, TFT, backtesting, historical forecasts и inference
- если задача уже хорошо решается средствами `darts`, не нужно дублировать ее самописной логикой
- исключения допустимы только там, где код действительно становится проще или где библиотека не покрывает нужное поведение

Предпочтительный стек:
- `darts`
- `pandas`
- `numpy`
- `pytorch`
- `seaborn`

## 4. Принципы реализации

Runtime-код в `pipeline/` должен быть:
- коротким и читаемым
- функциональным по стилю, без лишних классов и прослоек
- понятным по данным: на входе и выходе модулей в основном используются `dict`, `pandas.DataFrame`, `darts.TimeSeries` и объект модели
- документированным: все публичные функции и модули обязаны иметь подробные `docstring`

Нельзя:
- вводить много промежуточных `dataclass` и `Artifact`-объектов без явной пользы
- строить абстрактный orchestration/framework поверх `darts`
- прятать ключевую feature logic в сложные helper chains

Допустимый уровень строгости для MVP:
- конфиг можно держать как валидированный `dict`
- данные между модулями можно передавать как простые словари с понятными ключами
- если один маленький helper упрощает код, он допустим

## 5. Целевой layout runtime

Итоговая структура `pipeline/`:

```text
pipeline/
|- __init__.py
|- config.py
|- data_pipeline.py
|- training.py
|- inference.py
|- monitoring.py
`- run.py
```

Роль модулей:
- `config.py` - загрузка и валидация YAML-конфига
- `data_pipeline.py` - подготовка weekly данных и covariates
- `training.py` - baselines, TFT и rolling backtest
- `inference.py` - batch forecast и приведение к выходному контракту
- `monitoring.py` - легковесная сводка по качеству и SLA
- `run.py` - единый CLI entrypoint

## 6. Граница между pandas и darts

### 6.1 Что делает pandas

`pandas` отвечает за:
- проверку входного контракта `Week | City | revenue`
- приведение `Week` к началу ISO-недели
- агрегацию дублей по правилу `sum`
- отсечение данных по последней полностью закрытой неделе
- построение leakage-safe признаков
- сборку финального `forecast DataFrame`

### 6.2 Что делает darts

`darts` отвечает за:
- создание `TimeSeries` из табличного слоя
- работу с `past_covariates` и `future_covariates`
- TFT training
- `historical_forecasts`
- метрики и backtesting там, где это удобно закрывается библиотекой
- probabilistic prediction на квантили `0.1`, `0.5`, `0.9`

### 6.3 Использование from_group_dataframe

При подготовке рядов по `City` основным путем должен быть:
- `TimeSeries.from_group_dataframe()`

Документация `darts` подтверждает, что этот метод умеет:
- собирать список рядов по `group_cols`
- автоматически добавлять `group_cols` как static covariates
- восстанавливать пропущенные даты при `fill_missing_dates=True`
- заполнять такие пропуски значением из `fillna_value`

Поэтому в MVP допустимо использовать:

```python
TimeSeries.from_group_dataframe(
    df=frame,
    group_cols="City",
    time_col="week_start",
    value_cols="weekly_revenue",
    fill_missing_dates=True,
    freq="W-MON",
    fillna_value=config["data"]["fill_missing_revenue"],
)
```

Это означает, что полный weekly grid не обязательно достраивать вручную в `pandas`, если `darts` делает это штатно и код остается проще.

## 7. Поток данных

Целевой поток данных:

```text
raw weekly sales
-> validated pandas frame
-> normalized weekly frame
-> feature-enriched frame
-> final 8-week holdout split
-> darts target/covariate series
-> baselines + TFT
-> rolling backtest
-> final holdout evaluation
-> batch forecast
-> forecast DataFrame
-> monitoring summary
```

Правило: бизнес-нормализация и защита от leakage должны быть явными в `data_pipeline.py`, а не неявно размазаны по training и inference.

## 8. Правило финального holdout

Помимо rolling backtest, в runtime нужно выделять отдельный финальный test-набор:
- это последние полные `8` недель относительно доступной истории
- эти `8` недель не участвуют ни в обучении, ни в rolling validation
- rolling backtest выполняется только на истории, которая заканчивается перед этим holdout
- final holdout используется как отдельная финальная оценка качества после выбора и обучения моделей

Назначение этого holdout:
- проверить качество на самом свежем отложенном отрезке
- получить понятное сравнение `actual test` против `predict`
- сформировать итоговые визуальные артефакты для ревью

## 9. Leakage-safe правила

Следующие правила обязательны:
- cutoff равен последней полностью закрытой ISO-неделе
- текущая незакрытая неделя не используется ни в обучении, ни в признаках, ни в инференсе как факт
- target равен `weekly_revenue`
- любой lag считается только из прошлого
- любой rolling считается только после `shift(1)`
- future covariates могут использовать только календарную информацию, известную на момент прогноза
- validation выполняется только rolling weekly backtest

Прямо зафиксировать в коде и `docstring`:

```python
rolling = weekly_revenue.shift(1).rolling(window)
```

Нельзя:
- строить rolling от текущего значения без `shift(1)`
- использовать будущий `revenue`
- подмешивать незакрытую неделю в history window

## 10. Публичный API модулей

### 10.1 `pipeline/config.py`

Публичные функции:

```python
def load_pipeline_config(path: str | Path) -> dict: ...
def validate_pipeline_config(config: dict) -> None: ...
```

Требования:
- конфиг читается из YAML
- на выходе используется обычный валидированный `dict`
- структура конфига соответствует `configs/pipeline.yaml`
- ошибки конфигурации должны быть короткими и понятными

### 10.2 `pipeline/data_pipeline.py`

Публичные функции:

```python
def validate_raw_frame(raw_df: pd.DataFrame, config: dict) -> None: ...

def get_last_complete_week(reference_ts: pd.Timestamp | None = None) -> pd.Timestamp: ...

def build_training_frame(raw_df: pd.DataFrame, config: dict) -> pd.DataFrame: ...

def build_test_frame(raw_df: pd.DataFrame, config: dict) -> pd.DataFrame: ...

def build_inference_frame(
    raw_df: pd.DataFrame,
    config: dict,
    forecast_week: pd.Timestamp | None = None,
) -> pd.DataFrame: ...

def build_darts_series(frame: pd.DataFrame, config: dict) -> dict[str, object]: ...
```

Назначение функций:
- `validate_raw_frame()` проверяет схему входа и базовые инварианты
- `get_last_complete_week()` определяет безопасную точку отсечения
- `build_training_frame()` строит leakage-safe слой для обучения и rolling backtest без последних `8` недель holdout
- `build_test_frame()` возвращает последние полные `8` недель как отдельный финальный test holdout
- `build_inference_frame()` строит слой для batch inference от выбранной `forecast_week`
- `build_darts_series()` превращает `DataFrame` в словарь с рядами для `darts`

Ожидаемый формат результата `build_darts_series()`:

```python
{
    "target_series_by_city": dict[str, TimeSeries],
    "past_covariates_by_city": dict[str, TimeSeries],
    "future_covariates_by_city": dict[str, TimeSeries],
    "forecast_week": pd.Timestamp,
}
```

### 10.3 `pipeline/training.py`

Публичные функции:

```python
def train_baselines(series_data: dict[str, object], config: dict) -> dict[str, object]: ...

def train_tft(series_data: dict[str, object], config: dict): ...

def run_backtest(
    series_data: dict[str, object],
    tft_model,
    baselines: dict[str, object],
    config: dict,
) -> pd.DataFrame: ...

def evaluate_holdout(
    test_frame: pd.DataFrame,
    tft_model,
    baselines: dict[str, object],
    config: dict,
) -> pd.DataFrame: ...

def check_quality_gates(backtest_df: pd.DataFrame, config: dict) -> None: ...
```

Требования:
- baseline-модели: `seasonal_naive` и `rolling_mean`
- основная модель: `TFT`
- rolling backtest: `6` окон
- горизонт каждого тестового окна: `8` недель
- final holdout: последние полные `8` недель
- `run_backtest()` возвращает обычный `DataFrame`, а не отдельный report-класс
- `evaluate_holdout()` возвращает обычный `DataFrame` с фактом, предсказанием и метриками по финальному holdout
- в `DataFrame` должны быть поля, достаточные для сравнения baseline и TFT

### 10.4 `pipeline/inference.py`

Публичные функции:

```python
def generate_forecast(
    series_data: dict[str, object],
    tft_model,
    config: dict,
) -> pd.DataFrame: ...
```

Требования:
- inference работает batch-wise по всем `City`
- функция не читает CSV и не строит признаки заново
- на выходе возвращается готовый `forecast DataFrame` в контракте проекта

### 10.5 `pipeline/monitoring.py`

Публичные функции:

```python
def build_monitoring_summary(
    backtest_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    config: dict,
    runtime_seconds: float | None = None,
) -> dict: ...

def save_visual_artifacts(
    holdout_df: pd.DataFrame,
    output_dir: str | Path,
    config: dict,
) -> list[Path]: ...
```

Требования:
- monitoring summary остается простым словарем
- отдельный monitoring framework не нужен
- summary должен включать quality gate status и SLA flag
- `save_visual_artifacts()` сохраняет простые PNG-артефакты для сравнения `test` и `predict` на финальном holdout

### 10.6 `pipeline/run.py`

Публичные функции:

```python
def run_pipeline(config: dict) -> int: ...
def main() -> int: ...
```

Канонический запуск:

```bash
python -m pipeline.run --config configs/pipeline.yaml
```

## 11. Правила подготовки данных и признаков

Входной контракт:

```text
Week | City | revenue
```

Нормализованный рабочий слой:

```text
week_start | City | weekly_revenue | is_missing
```

Обязательные признаки:

Static:
- `City`

Future covariates:
- `week_of_year`
- `month`
- `quarter`
- `year`
- `is_holiday_week`

Past covariates:
- `lag_1`
- `lag_2`
- `lag_4`
- `lag_8`
- `rolling_4`
- `rolling_8`
- `rolling_12`

Требования:
- `revenue` трактуется как недельная сумма для `Week`
- отрицательные значения допустимы
- пропуски недель допустимы
- дубликаты по `Week + City` агрегируются через `sum`
- lag/rolling-фичи вычисляются в `pandas` до передачи в `darts`
- логика feature engineering должна быть читаемой и находиться в одном модуле

## 12. Формат результата инференса

Итоговый прогноз должен быть в формате:
- одна строка на один `City` и одну `target_week`
- для каждого `City` на один запуск получается `8` строк, по одной на каждую неделю горизонта

Это значит:
- не одна строка на город с набором полей `h1...h8`
- а восемь отдельных строк на город, где `horizon_week` хранится в данных

Обязательные поля:
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

## 13. Требования к backtest, holdout и quality gates

Нужно сравнивать:
- `seasonal_naive`
- `rolling_mean`
- `TFT`

Требования:
- все модели проверяются на одной и той же weekly-постановке
- используется `6` weekly test windows
- горизонт каждого окна равен `8`
- последние полные `8` недель выделяются в отдельный финальный holdout
- финальный holdout не участвует в обучении и rolling validation
- обязательная контрактная метрика: `SMAPE`
- допустимо дополнительно считать `MAPE` как вспомогательную диагностику

Quality gates:
- `SMAPE <= 15%`
- улучшение TFT не менее `5%` относительно лучшего baseline

Правило интерпретации:
- quality gates проверяются на rolling backtest
- final holdout не заменяет rolling validation
- final holdout нужен для отдельной финальной проверки и визуального сравнения `test` против `predict`

Если quality gate не пройден:
- pipeline должен явно сигнализировать об этом
- мониторинг должен сохранить соответствующий статус

Обязательные артефакты holdout-оценки:
- табличный файл с фактом и прогнозом по каждой паре `City + target_week`
- `PNG`-график со сравнением aggregate `MAPE` по моделям на final holdout
- `PNG`-график со сравнением `MAPE` по `City` на final holdout

Допустимая реализация визуализаций:
- `seaborn.barplot` для aggregate `MAPE` по моделям
- `seaborn.heatmap` или `seaborn.barplot` для `MAPE` по городам и моделям

## 14. Порядок выполнения pipeline

Целевой порядок в `run_pipeline()`:

1. Загрузить и провалидировать конфиг.
2. Прочитать входной CSV в `pandas`.
3. Подготовить training frame без последних `8` недель holdout.
4. Подготовить final test holdout из последних полных `8` недель.
5. Подготовить inference frame.
6. Построить `darts`-ряды.
7. Обучить baseline-модели.
8. Обучить TFT.
9. Выполнить rolling backtest.
10. Проверить quality gates.
11. Выполнить финальную оценку на holdout.
12. Построить batch forecast.
13. Сохранить forecast output.
14. Построить monitoring summary.
15. Сохранить monitoring artifacts.
16. Сохранить visual artifacts по final holdout.
17. Вернуть `0` при успехе или non-zero code при ошибке.

## 15. Требования к docstring

Все публичные модули и функции в `pipeline/` обязаны иметь подробные `docstring`.

Минимум для каждой публичной функции:
- что делает функция
- какие аргументы принимает
- что возвращает
- какие ошибки может выбросить

Для функций из `data_pipeline.py` дополнительно обязательно:
- кратко описывать leakage-safe поведение
- явно указывать, что lag/rolling не используют будущее

Для `generate_forecast()` и `run_pipeline()` обязательно:
- описывать выходной табличный контракт

Стиль `docstring` должен быть единым по проекту.

Также для `build_test_frame()` и `evaluate_holdout()` обязательно:
- явно описывать, что final holdout не участвует в обучении и rolling validation
- описывать формат сравнения `actual` и `predict`

## 16. Тестовая стратегия

Тесты должны оставаться прикладными и проверять контракт, а не внутреннюю архитектуру.

Минимальный набор:

### 16.1 Конфиг

- YAML читается корректно
- обязательные ключи присутствуют
- некорректные значения вызывают понятную ошибку

### 16.2 Data pipeline

- схема входа валидируется
- `Week` приводится к `week_start`
- дубликаты агрегируются через `sum`
- незакрытая неделя не попадает в обучение
- последние полные `8` недель выделяются в отдельный `test holdout`
- `test holdout` не пересекается с training и backtest history
- `lag_*` и `rolling_*` строятся только через прошлое
- `build_darts_series()` корректно собирает ряды по `City`
- восстановление пропущенных недель через `from_group_dataframe(... fill_missing_dates=True ...)` работает корректно

### 16.3 Training

- baselines и TFT запускаются на одном наборе рядов
- rolling backtest использует `6` окон и горизонт `8`
- backtest возвращает `DataFrame` с результатами по моделям и окнам
- финальный holdout оценивается отдельно после обучения
- `evaluate_holdout()` возвращает сравнение `actual` и `predict` по последним полным `8` неделям
- quality gate корректно срабатывает при нарушении порогов

### 16.4 Inference

- результат имеет одну строку на один `City` и одну `target_week`
- на каждый `City` получается `8` строк горизонта
- все обязательные поля присутствуют
- квантильные прогнозы разложены в `q0.1`, `q0.5`, `q0.9`

### 16.5 Run

- pipeline запускается через `python -m pipeline.run --config configs/pipeline.yaml`
- при успешном прогоне сохраняются forecast и monitoring artifacts
- при успешном прогоне сохраняются visual artifacts по финальному holdout
- при ошибке или нарушении контракта возвращается non-zero code

## 17. Границы и нецели

Этот dev-spec:
- не меняет weekly-контракт проекта
- не меняет горизонт `8` недель
- не меняет history window `60` недель
- не переносит runtime-логику в ноутбуки
- не требует сложной объектной архитектуры

Этот dev-spec не предполагает:
- большого набора `dataclass`
- отдельной artifact-модели
- разделения на сервисы, менеджеры и фасады без явной необходимости

## 18. Критерии приемки спецификации

Спецификация считается корректной, если она:
- согласована с `docs/project-spec.md`
- согласована с `docs/technical-assignment.md`
- фиксирует `Darts-first orchestration` как основной подход
- сохраняет код простым и читаемым
- задает разработку с нуля для пустого `pipeline/`
- не вводит лишнюю объектную архитектуру
- фиксирует понятные публичные функции модулей
- требует подробные `docstring`
- оставляет `darts` основным инструментом для time series runtime-задач
- фиксирует отдельный финальный `8`-недельный holdout вне обучения и validation
- требует визуальные `MAPE`-артефакты для сравнения `test` и `predict`
