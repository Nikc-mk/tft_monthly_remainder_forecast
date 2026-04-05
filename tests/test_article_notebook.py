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
