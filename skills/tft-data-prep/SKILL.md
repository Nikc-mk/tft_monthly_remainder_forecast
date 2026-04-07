---
name: tft-data-prep
description: Prepare leakage-safe weekly datasets for TFT sales forecasting by City. Use when Codex needs to normalize the weekly grid, review the weekly data contract, define allowed covariates, or check preprocessing for temporal leakage.
---

# TFT Data Prep

## Overview

Use this skill for all work around weekly dataset normalization, target generation, and TFT-compatible features. Treat temporal correctness as the main responsibility.

## Required Sources

- Read `docs/technical-assignment.md`.
- Read `docs/project-spec.md`.
- Read `references/leakage-checklist.md`.

## Workflow

1. Start from the weekly input schema: `Week`, `City`, `revenue`.
2. Treat `revenue` as the already aggregated weekly sum for `Week`, where `Week` is the start of the ISO week.
3. Normalize the input into `week_start | week_idx | City | weekly_revenue | is_missing`.
4. Expand every city to a complete weekly calendar.
5. Fill synthetic rows with `weekly_revenue = 0` and `is_missing = 1`.
6. Build the target as direct weekly revenue for the future horizon.
7. Add only the calendar features known for the full horizon.
8. Verify that no feature depends on the future, the open week, or the current target week.

## Required Feature Rules

- Keep `City` as the static categorical feature.
- Keep `week_idx`, `week_of_year`, `month`, `quarter`, `year`, and `is_holiday_week` as known features.
- Keep manual lag and rolling features out of the current weekly TFT runtime unless the stored spec is updated first.
- Use a fixed history window of `60` weeks in downstream dataset design.
- Preserve negative revenue values.

## Reject Immediately

- rolling windows applied before `shift(1)`
- any use of future values
- any use of the current open week
- any requirement to aggregate from daily input when the stored contract is already weekly
- any mandatory manual lag or rolling feature set in the current runtime
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
