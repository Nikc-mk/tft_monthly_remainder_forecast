# Leakage Checklist

## Must hold

- full weekly grid per `City`
- `weekly_revenue = 0` for added rows
- `is_missing = 1` for added rows
- target excludes the current open week
- every lag or rolling feature starts from `shift(1)`

## Known features

- `week_of_year`
- `month`
- `quarter`
- `year`
- `is_holiday_week`

## Unknown features

- `lag_1`
- `lag_2`
- `lag_4`
- `lag_8`
- `rolling_4`
- `rolling_8`
- `rolling_12`

## Red flags

- rolling over raw `weekly_revenue`
- using data from an open week
- target generation that leaks future data
- leaving calendar gaps by `City`
