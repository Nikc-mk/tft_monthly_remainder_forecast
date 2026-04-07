# Leakage Checklist

## Must hold

- weekly input schema is `Week | City | revenue`
- full weekly grid per `City`
- `weekly_revenue = 0` for added rows
- `is_missing = 1` for added rows
- `week_idx` is present in the normalized layer
- target excludes the current open week
- manual lag or rolling features are absent in the current runtime; if introduced later, every such feature starts from `shift(1)`

## Known features

- `week_idx`
- `week_of_year`
- `month`
- `quarter`
- `year`
- `is_holiday_week`

## Red flags

- rolling over raw `weekly_revenue`
- requiring daily aggregation for the canonical weekly pipeline
- introducing manual lag or rolling features into the current runtime contract
- using data from an open week
- target generation that leaks future data
- leaving calendar gaps by `City`
