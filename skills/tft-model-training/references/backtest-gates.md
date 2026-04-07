# Backtest and Quality Gates

## Required baselines

1. seasonal naive
2. rolling mean over recent weekly history

## Required TFT settings

- encoder length exactly 60
- prediction length 8
- quantile loss for `0.1`, `0.5`, `0.9`
- use library defaults unless a documented project need requires an override

## Backtest shape

- 6 rolling weekly windows
- each window forecasts the next 8 weeks
- the final holdout is the last 8 fully closed weeks after rolling validation
- the final holdout is excluded from training and rolling validation
- no single split substitute

## Report always

- SMAPE
- MAE
- quantile coverage
- comparison versus the best baseline

## Fail if

- SMAPE is above 15 percent
- improvement over best baseline is below 5 percent
- baselines are missing
- results come from a single split
