Техническое задание (MVP runtime)
Исследовательский пайплайн прогнозирования недельных продаж по `City`

## 1. Цель проекта

Проект должен дать простой и воспроизводимый MVP runtime для недельного прогноза по `City`.

Система должна:

- принимать продажи до последней полностью закрытой ISO-недели
- принимать уже агрегированные недельные продажи, где `Week` - дата начала ISO-недели
- прогнозировать недельные продажи на горизонт `W+1 ... W+8`
- использовать строго `60` прошедших недель как историю модели
- запускаться через единый пакет `pipeline/`

Здесь:

- `W` - последняя полностью закрытая ISO-неделя
- `City` - идентификатор города

## 2. Постановка задачи

Формализация:

`target(city, W, h) = weekly_revenue(city, W+h), h in {1..8}`

где:

- `weekly_revenue(city, week) = sum(revenue)` внутри ISO-недели
- `h` - номер недели горизонта

### Требования к прогнозу

- multi-horizon forecast на `8` недель
- вероятностный прогноз с `q0.1`, `q0.5`, `q0.9`
- одна строка результата на `City + target_week`

## 3. Данные

### Формат входных данных

Исходные данные приходят на недельном уровне:

`Week | City | revenue`

После нормализации и заполнения пропусков рабочий ряд имеет вид:

`week_start | City | weekly_revenue | is_missing`

### Гарантии

- базовая частота исходных данных: Weekly
- `revenue` во входе уже является недельной суммой продаж для `Week`
- модельная целевая колонка после нормализации: `weekly_revenue`
- неделя определяется по ISO calendar (`пн-вс`)
- возможны отрицательные значения
- возможны пропуски дат и недель
- leakage не допускается

## 4. Feature Engineering

### 4.1 Static

- `City`

### 4.2 Time-varying known

- `week_of_year`
- `month`
- `quarter`
- `year`
- `is_holiday_week`
  count of configured holiday dates within the ISO week (`Mon-Sun`)

### 4.3 Time-varying unknown

Лаги:

- `lag_1`
- `lag_2`
- `lag_4`
- `lag_8`

Rolling с обязательным `shift(1)`:

- `rolling_4`
- `rolling_8`
- `rolling_12`

Запрещено:

- rolling без `shift(1)`
- использование текущей незакрытой недели
- использование будущего `revenue`

## 5. Моделирование и валидация

- baseline: `seasonal_naive`
- baseline: `rolling_mean`
- основная модель: `TFT`
- rolling backtest по неделям
- `6` weekly test windows
- тестовый горизонт каждого окна: `8` недель
- последние `8` полных недель выделяются в отдельный финальный holdout
- финальный holdout не участвует ни в обучении, ни в rolling validation
- quality gates: `SMAPE <= 15%` и улучшение TFT минимум на `5%` относительно лучшего baseline

## 6. Инференс

Результат должен быть в формате `pandas.DataFrame`.

### Обязательный выходной контракт

Одна строка на `City + target_week`.

Поля:

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

### SLA

- batch inference по всем `City`
- время одного batch-прогона на CPU: не более `10` секунд

## 7. Архитектура MVP

Каноническая структура репозитория:

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

`pipeline/` является единственным каноническим runtime package.

`notebooks/` is research only. It must not contain runtime logic. Ноутбуки хранят эксперименты, EDA и промежуточные проверки, но не заменяют batch entrypoint.

### 7.1 Принципы реализации MVP runtime

`pipeline/` должен оставаться простым, читаемым и исследовательским MVP runtime, а не самописным framework для прогнозирования.

Приоритетный стек проекта:

- `darts`
- `pandas`
- `numpy`
- `pytorch`
- `seaborn`

Обязательные правила реализации:

- для forecasting, backtesting, time-series transforms и связанной runtime orchestration по умолчанию нужно использовать `darts`, если библиотека покрывает задачу
- `pandas`, `numpy`, `pytorch` и `seaborn` являются приоритетными библиотеками поддержки вокруг runtime и исследований
- самописная реализация поверх низкоуровневых API допустима только как исключение, если нужная возможность отсутствует или недостаточно покрыта приоритетными библиотеками
- любое такое исключение должно быть кратко и явно обосновано в документации проекта
- [notebooks/tft_darts_pipeline.ipynb](../notebooks/tft_darts_pipeline.ipynb) используется только как reference implementation style для ожидаемой простоты и читаемости, но не как источник runtime-логики

## 8. Конфигурация и запуск

Используется единый YAML-конфиг:

- `configs/pipeline.yaml`

Канонический запуск MVP:

`python -m pipeline.run --config configs/pipeline.yaml`

Любые старые команды из `scripts/` или старые конфиги не считаются публичным контрактом этого MVP.

## 9. Артефакты

Сохраняются:

- `normalized_dataset.csv`
- `forecast.csv`
- `holdout_predictions.csv`
- `monitoring_summary.json`
- `holdout_mape_by_model.png`
- `holdout_mape_by_city.png`
- `holdout_mape_by_week.png`
- `tft_residuals_analysis.png`

PNG-файлы являются monitoring/reporting outputs для визуального сравнения `MAPE` на финальном holdout.

## 10. Требования к качеству

Проект должен:

- быть читаемым
- иметь простую структуру
- предпочитать зрелые библиотечные abstractions вместо самописной инфраструктуры
- использовать `darts` как библиотеку по умолчанию для runtime-задач forecasting, если задача поддерживается библиотекой
- запускаться через единый entrypoint
- хранить runtime-код в `pipeline/`
- хранить исследовательские ноутбуки только в `notebooks/`
- иметь содержательные `docstring` у публичных runtime-модулей, классов и функций в `pipeline/`
