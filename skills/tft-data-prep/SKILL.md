---
name: tft-data-prep
description: Prepare leakage-safe weekly datasets for TFT sales forecasting by City. Use when Codex needs to aggregate daily sales into ISO weeks, normalize the weekly grid, define lag and rolling features, review data contracts, or check preprocessing for temporal leakage.
---

# TFT Data Prep

## Overview

Use this skill for all work around weekly dataset normalization, target generation, and TFT-compatible features. Treat temporal correctness as the main responsibility.

## Required Sources

- Read `docs/technical-assignment.md`.
- Read `docs/project-spec.md`.
- Read `references/leakage-checklist.md`.

## Workflow

1. Start from the raw schema: `date`, `City`, `revenue`.
2. Aggregate daily sales into ISO weeks (`Mon-Sun`) with `weekly_revenue = sum(revenue)`.
3. Expand every city to a complete weekly calendar.
4. Fill synthetic rows with `weekly_revenue = 0` and `is_missing = 1`.
5. Build the target as direct weekly revenue for the future horizon.
6. Add calendar features known for the full horizon.
7. Add lag and rolling features only after `shift(1)` by `City`.
8. Verify that no feature depends on the future, the open week, or the current target week.

## Required Feature Rules

- Keep `City` as the static categorical feature.
- Keep `week_of_year`, `month`, `quarter`, `year`, and `is_holiday_week` as known features.
- Keep `lag_1`, `lag_2`, `lag_4`, `lag_8`, `rolling_4`, `rolling_8`, and `rolling_12` as unknown real features.
- Use a fixed history window of `60` weeks in downstream dataset design.
- Preserve negative revenue values.

## Reject Immediately

- rolling windows applied before `shift(1)`
- any use of future values
- any use of the current open week
- sparse per-city weekly calendars left unfilled
- missing `is_missing` tracking for synthetic rows

## Output Expectations

Return:

- input assumptions
- aggregation and normalization rules
- target logic
- feature definitions
- leakage checks
- downstream contract for training and inference
