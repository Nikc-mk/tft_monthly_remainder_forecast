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
