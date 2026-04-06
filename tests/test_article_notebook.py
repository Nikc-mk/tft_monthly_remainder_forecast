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

    def test_notebook_reads_project_sales_file(self):
        self.assertIn('pd.read_csv("data/sales.csv", sep=";")', self.sources)

    def test_notebook_builds_weekly_grid_and_leakage_safe_features(self):
        required_snippets = [
            'freq="W-MON"',
            'df["date"] = pd.to_datetime(df["Week"])',
            'df.groupby(["City", "date"], as_index=False)["revenue"]',
            'shifted = df.groupby("City")["revenue"].shift(1)',
            'df["lag_1"] = shifted',
            'df["rolling_4"] = shifted.groupby(df["City"]).rolling(4).mean()',
            'df["rolling_12"] = shifted.groupby(df["City"]).rolling(12).mean()',
        ]
        for snippet in required_snippets:
            self.assertIn(snippet, self.sources)
        self.assertNotIn('df["is_holiday_week"] = 0', self.sources)

    def test_notebook_uses_ru_holidays_for_holiday_counts(self):
        required_snippets = [
            "import holidays",
            'ru_holidays = holidays.country_holidays("RU")',
            'week_holidays = pd.date_range(row["date"], periods=7, freq="D")',
            'holiday_count = sum(day in ru_holidays for day in week_holidays)',
            'df["is_holiday_week"] = df.apply(count_ru_holidays_in_week, axis=1)',
        ]
        for snippet in required_snippets:
            self.assertIn(snippet, self.sources)

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

    def test_notebook_contains_russian_explanatory_comments(self):
        required_snippets = [
            "# Читаем weekly-датасет проекта и приводим даты к ISO-неделям",
            "# Восстанавливаем полный недельный грид по каждому городу",
            "# Календарные признаки известны на всём горизонте прогноза",
            "# Считаем число праздничных дней РФ внутри каждой ISO-недели",
            "# Лаги и rolling считаем только после shift(1), чтобы не было утечки",
            "# Добавляем будущие недели только для future covariates",
            "# Масштабируем ряды по аналогии со статьёй, но под weekly-контракт проекта",
            "# Делаем article-style split с параметрами проекта: 60 недель истории и 8 недель горизонта",
            "# Обучаем TFT в конфигурации, близкой к статье, но адаптированной под weekly revenue",
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

    def test_notebook_readme_mentions_new_article_adaptation(self):
        readme_text = Path("notebooks/README.md").read_text(encoding="utf-8")
        self.assertIn("darts_tft_weekly_city_article.ipynb", readme_text)
        self.assertIn("research only", readme_text.lower())
