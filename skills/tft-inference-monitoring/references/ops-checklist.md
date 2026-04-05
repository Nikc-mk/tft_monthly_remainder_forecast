# Inference and Monitoring Checklist

## Runtime artifacts

- model checkpoint
- scaler config
- feature pipeline config
- inference entrypoint

## Inference rules

- batch over all `City`
- direct weekly prediction
- no autoregression
- no per-city dataset rebuild loop
- CPU SLA below 10 seconds

## Edge cases

- negative revenue allowed
- missing weeks handled upstream with `is_missing`
- cold-start fallback documented

## Monitoring

- weekly revenue drift
- `City` distribution drift
- SMAPE degradation
- quantile coverage
