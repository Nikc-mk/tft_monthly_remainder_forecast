Спецификация проекта для Codex (MVP runtime)

## Назначение

Этот репозиторий описывает MVP-пайплайн прогнозирования недельных продаж по `City`.

Проект предназначен для:

- подготовки недельных данных без leakage
- сравнения baseline и TFT на одной постановке
- запуска единого batch runtime через один пакет `pipeline/`
- хранения исследовательских ноутбуков отдельно от runtime-кода

## Основная задача

Система должна:

- принимать фактические продажи до последней полностью закрытой ISO-недели
- принимать уже агрегированные недельные продажи в колонке `Week`
- прогнозировать недельные продажи на горизонт `W+1 ... W+8`
- работать по оси агрегации `City`

Формализация:

`target(city, W, h) = weekly_revenue(city, W+h), h in {1..8}`

где:

- `W` - последняя полностью закрытая неделя
- неделя определяется по ISO calendar (`пн-вс`)

## Непереговорные правила

- прогноз всегда недельный
- горизонт всегда `8` недель
- используется фиксированное окно истории `60` недель
- любые lag и rolling-признаки строятся только после `shift(1)`
- запрещены future-aware признаки и leakage
- валидация выполняется rolling weekly backtest
- inference остается batch-oriented и строит прямой multi-horizon forecast
- отрицательные значения допустимы
- `pipeline/` должен оставаться простым и читаемым MVP runtime, а не самописным forecasting framework
- приоритетный стек проекта: `darts`, `pandas`, `numpy`, `pytorch`, `seaborn`
- для forecasting, backtesting, time-series transforms и связанной runtime orchestration по умолчанию используется `darts`, если задача поддерживается библиотекой
- самописная реализация вместо возможностей приоритетных библиотек допустима только как исключение с явным кратким обоснованием в документации
- публичные runtime-модули, классы и функции в `pipeline/` обязаны иметь содержательные `docstring`

## Публичные контракты

### Входные данные

Базовый вход:

`Week | City | revenue`

Нормализованный недельный слой:

`week_start | City | weekly_revenue | is_missing`

Характеристики:

- исходная частота данных: Weekly
- модельная частота: weekly
- `revenue` во входе трактуется как недельная сумма продаж для `Week`
- в нормализованном слое целевая колонка хранится как `weekly_revenue`
- возможны отрицательные значения
- возможны пропуски дат и недель

### Выход

Одна строка на `City + target_week`.

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

## MVP Layout

Канонический runtime-пакет: `pipeline/`.

Канонический конфиг: `configs/pipeline.yaml`.

Канонический запуск:

`python -m pipeline.run --config configs/pipeline.yaml`

Структура MVP:

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

`notebooks/` is research only. It must not contain runtime logic. Ноутбуки используются для EDA, экспериментов и проверки идей, а production-like выполнение должно идти через `pipeline/`.

[notebooks/tft_darts_pipeline.ipynb](../notebooks/tft_darts_pipeline.ipynb) может использоваться только как reference implementation style для простого и читаемого MVP на `darts`, но не как источник runtime-логики.

## Подготовка данных

### Целевая переменная

`target = weekly_revenue`

### Feature Engineering

Static:

- `City`

Future covariates:

- `week_of_year`
- `month`
- `quarter`
- `year`
- `is_holiday_week`
  count of configured holiday dates within the ISO week (`Mon-Sun`)

Past covariates:

- `lag_1`
- `lag_2`
- `lag_4`
- `lag_8`
- `rolling_4`
- `rolling_8`
- `rolling_12`

Обязательное правило:

`rolling = weekly_revenue.shift(1).rolling(...)`

## Моделирование и валидация

- baseline: `seasonal_naive` и `rolling_mean`
- основная модель: `TFT`
- rolling backtest: `6` weekly windows
- тестовый горизонт каждого окна: `8` недель
- последние `8` полных недель резервируются как отдельный финальный holdout после rolling validation
- финальный holdout не участвует ни в обучении, ни в rolling validation
- quality gates: `SMAPE <= 15%` и улучшение TFT не менее `5%` относительно лучшего baseline

## Инференс и SLA

- прямой multi-horizon прогноз на `8` недель
- batch inference по всем `City`
- CPU runtime не более `10` секунд для дефолтного batch

## Артефакты runtime

В рабочую директорию сохраняются:

- `normalized_dataset.csv`
- `forecast.csv`
- `holdout_predictions.csv`
- `monitoring_summary.json`
- `holdout_mape_by_model.png`
- `holdout_mape_by_city.png`
- `holdout_mape_by_week.png`
- `tft_residuals_analysis.png`

PNG-артефакты используются как monitoring/reporting outputs для визуального сравнения `MAPE` на финальном holdout.

## Definition of Done

Изменение считается завершенным, если:

- не нарушает weekly multi-horizon на `8` недель
- не содержит leakage
- использует `configs/pipeline.yaml` как основной конфиг
- запускается через `python -m pipeline.run --config configs/pipeline.yaml`
- не переносит runtime-логику в `notebooks/`
- использует возможности `darts` по умолчанию там, где библиотека покрывает runtime-задачу forecasting или backtesting
- не добавляет самописную forecasting-инфраструктуру без явного краткого обоснования в документации
- сохраняет простоту и читаемость runtime-кода в `pipeline/`
- добавляет содержательные `docstring` для публичных runtime-модулей, классов и функций в `pipeline/`
- обновляет документацию при изменении публичного контракта
